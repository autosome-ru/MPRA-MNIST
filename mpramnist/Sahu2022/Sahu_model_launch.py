"""
Sahu_model_launch.py

Runs a benchmark: for each task in `--task`, trains a fresh single-output
model on the Sahu2022 dataset, repeating the whole process `--runs` times.

Most tasks (RandomEnhancer, GenomicPromoter, CapturePromoter,
GenomicEnhancer, AtacSeq) are plain regression/classification tasks trained
with `LitModel_Sahu` and a standard model head.

The "Binary" task is a *paired* task: the model's head is stripped off (it
only produces an embedding) and a task-specific `LitModel_Sahu_binary_*`
Lightning module handles pairing two sequences and comparing their
embeddings. Each architecture therefore needs (a) a small subclass that
overrides `forward()` to drop the head, and (b) a matching
`LitModel_Sahu_binary_*` class.

NOTE: Lyra and ReporterNet do not yet have a "Binary" variant — see the
NotImplementedError raised in `build_binary_model` for details.

Results (one row per run, one column per task) are appended to a TSV file
given by `--result_dir`.
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import mpramnist.transforms as t
from mpramnist.Sahu2022 import (
    SahuDataset,
    LitModel_Sahu,
    LitModel_Sahu_binary_legnet,
    LitModel_Sahu_binary_mprann,
    LitModel_Sahu_binary_malinois,
    LitModel_Sahu_binary_parm,
    LitModel_Sahu_binary_dream_rnn,
)
from mpramnist.models import (
    HumanLegNet,
    initialize_weights,
    BassetBranched,
    MPRAnn,
    PARM,
    DREAM_RNN,
    Lyra,
    ReporterNet,
    initialize_weights_reporternet,
)
from mpramnist.models.DREAM_RNN import (
    AutosomeFinalLayersBlock,
    BHICoreBlock,
    BHIFirstLayersBlock,
    PrixFixeNet,
)


# =============================================================================
# CLI arguments
# =============================================================================

MODEL_CHOICES = [
    "MPRALegNet",
    "MPRAnn",
    "Malinois",
    "PARM",
    "DREAM-RNN",
    "Lyra",
    "ReporterNet",
]

# Models with a working "Binary" (paired) variant. Lyra and ReporterNet are
# intentionally excluded until a LitModel_Sahu_binary_* class exists for
# them — see build_binary_model().
BINARY_CAPABLE_MODELS = {"MPRALegNet", "MPRAnn", "Malinois", "PARM", "DREAM-RNN", "DREAM_RNN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a single-output model on the Sahu2022 dataset, per task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general = parser.add_argument_group("general", "General run settings")
    general.add_argument(
        "--result_dir",
        type=str,
        default="./sahu.tsv",
        help="Path to the TSV file where per-run test AUPR values are appended.",
    )
    general.add_argument("--device", type=int, default=0, help="GPU device index to use.")
    general.add_argument(
        "--num_workers", type=int, default=8, help="Number of DataLoader worker processes."
    )
    general.add_argument("--batch_size", type=int, default=1024, help="Batch size.")
    general.add_argument(
        "--runs", type=int, default=5, help="Number of independent training runs (repeats)."
    )
    general.add_argument(
        "--model",
        type=str,
        default="MPRALegNet",
        choices=MODEL_CHOICES,
        help="Model architecture to train.",
    )

    dataset_args = parser.add_argument_group("dataset", "Dataset settings")
    dataset_args.add_argument("--root", type=str, default="../data/", help="Path to the data root.")
    dataset_args.add_argument(
        "--task",
        nargs="+",
        default=[
            "RandomEnhancer",
            "GenomicPromoter",
            "CapturePromoter",
            "GenomicEnhancer",
            "AtacSeq",
            "Binary",
        ],
        help="List of tasks to train and evaluate on.",
    )

    trainer_args = parser.add_argument_group("trainer", "Training hyperparameters")
    trainer_args.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    trainer_args.add_argument("--wd", type=float, default=1e-1, help="Weight decay (AdamW).")
    trainer_args.add_argument("--epoch_num", type=int, default=10, help="Max number of epochs.")

    args = parser.parse_args()

    if isinstance(args.task, str):
        args.task = [args.task]

    return args


# =============================================================================
# Transforms
# =============================================================================

def build_transforms(sequence_first: bool = False) -> dict:
    """Sahu sequences are fixed-length and need no cropping/padding here."""
    return {
        "train": t.Compose([t.Seq2Tensor(sequence_first)]),
        "test": t.Compose([t.Seq2Tensor(sequence_first)]),
    }


# =============================================================================
# Task model factory (every task except "Binary")
# =============================================================================

def build_model(model_name: str, seq_len: int):
    """Instantiate the requested model for a standard (non-"Binary") task.

    All Sahu inputs are 4-channel one-hot sequences, so `in_channels` is
    fixed at 4 here (matching the original script).
    """
    loss = nn.BCEWithLogitsLoss()

    if model_name == "MPRALegNet":
        model = HumanLegNet(
            in_ch=4,
            output_dim=1,
            stem_ch=64,
            stem_ks=11,
            ef_ks=9,
            ef_block_sizes=[32, 64, 128, 128, 256, 512, 256],
            pool_sizes=[1, 2, 1, 2, 1, 2, 1],
            resize_factor=4,
        )
        model.apply(initialize_weights)

    elif model_name == "MPRAnn":
        model = MPRAnn(output_dim=1)

    elif model_name == "Malinois":
        model = BassetBranched(input_len=seq_len, n_outputs=1)

    elif model_name == "PARM":
        model = PARM(n_block=5, type_loss="mse", output_dim=1)

    elif model_name in ("DREAM-RNN", "DREAM_RNN"):
        model = DREAM_RNN(4, seq_len, 1)

    elif model_name.lower() == "lyra":
        model = Lyra(d_input=4, d_output=1, d_model=512, dropout=0.1)

    elif model_name == "ReporterNet":
        model = ReporterNet(dropout_rate=0.2, output_dim=1)
        model.apply(initialize_weights_reporternet)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, loss


# =============================================================================
# Binary-task ("paired") model + Lightning module factory
# =============================================================================

def build_binary_model(model_name: str, seq_len: int):
    """Instantiate the head-stripped model + matching LitModel_Sahu_binary_*
    class for the "Binary" (paired) task.

    Returns (model, lit_model_class, loss).
    """
    loss = nn.BCEWithLogitsLoss()

    if model_name == "MPRALegNet":

        class HumanLegNetBinary(HumanLegNet):
            def __init__(self):
                super().__init__(
                    in_ch=4,
                    output_dim=1,
                    stem_ch=64,
                    stem_ks=11,
                    ef_ks=9,
                    ef_block_sizes=[80, 96, 112, 128],
                    pool_sizes=[2, 2, 2, 2],
                    resize_factor=4,
                    activation=nn.SiLU,
                )

            def forward(self, x):
                x = self.stem(x)
                x = self.main(x)
                x = self.mapper(x)
                x = F.adaptive_avg_pool1d(x, 1)
                x = x.squeeze(-1)  # without head; head is applied in the trainer
                return x

        model = HumanLegNetBinary()
        model.apply(initialize_weights)
        return model, LitModel_Sahu_binary_legnet, loss

    elif model_name == "MPRAnn":

        class MPRAnnBinary(MPRAnn):
            def __init__(self):
                super().__init__(output_dim=1)

            def forward(self, x):
                seq = self.conv1(x)
                seq = F.relu(seq)
                seq = self.bn1(seq)
                seq = self.conv2(seq)
                seq = F.softmax(seq, dim=1)
                seq = self.bn2(seq)
                seq = self.pool1(seq)
                seq = self.dropout1(seq)
                seq = self.conv3(seq)
                seq = F.softmax(seq, dim=1)
                seq = self.bn3(seq)
                seq = self.conv4(seq)
                seq = F.softmax(seq, dim=1)
                seq = self.bn4(seq)
                seq = self.global_pool(seq)
                seq = seq.squeeze(-1)
                seq = self.dropout2(seq)
                seq = seq.reshape((seq.shape[0], -1))
                seq = self.fc1(seq)
                seq = F.sigmoid(seq)
                seq = self.dropout3(seq)
                return seq

        model = MPRAnnBinary()
        return model, LitModel_Sahu_binary_mprann, loss

    elif model_name == "Malinois":

        class MalinoisBinary(BassetBranched):
            def __init__(self, input_len, n_outputs):
                super().__init__(
                    input_len=input_len,
                    conv1_channels=300,
                    conv1_kernel_size=19,
                    conv2_channels=200,
                    conv2_kernel_size=11,
                    conv3_channels=200,
                    conv3_kernel_size=7,
                    n_linear_layers=1,
                    linear_channels=1000,
                    linear_activation="ReLU",
                    linear_dropout_p=0.11625456877954289,
                    n_branched_layers=3,
                    branched_channels=140,
                    branched_activation="ReLU",
                    branched_dropout_p=0.5757068086404574,
                    n_outputs=n_outputs,
                    use_batch_norm=True,
                    use_weight_norm=False,
                    loss_criterion="L1KLmixed",
                    loss_args={},
                )

            def forward(self, x):
                encoded = self.encode(x)
                decoded = self.decode(encoded)
                return decoded

        model = MalinoisBinary(input_len=seq_len, n_outputs=1)
        return model, LitModel_Sahu_binary_malinois, loss

    elif model_name == "PARM":

        class PARMBinary(PARM):
            def __init__(self, n_block, type_loss, output_dim):
                super().__init__(
                    n_block=n_block,
                    filter_size=125,
                    output_dim=output_dim,
                    weight_file=None,
                    cell_line=False,
                    type_loss=type_loss,
                    validation=False,
                    index_interested_output=False,
                    maxglobalpool=True,
                    vocab=4,
                    use_AttentionPool=True,
                )

            def forward(self, x):
                out = self.stem(x)
                out = self.conv_tower(out)
                if self.maxglobalpool:
                    out = torch.max(out, dim=-1).values
                out = out.view(out.size(0), -1)
                return out

        model = PARMBinary(n_block=5, type_loss="mse", output_dim=1)
        return model, LitModel_Sahu_binary_parm, loss

    elif model_name in ("DREAM-RNN", "DREAM_RNN"):

        class AutosomeFinalLayersBlockBinary(AutosomeFinalLayersBlock):
            def forward(self, x):
                x = self.mapper(x)
                x = F.adaptive_avg_pool1d(x, 1)
                x = x.squeeze(2)
                return x

        def dream_rnn_binary(in_channels, seqsize):
            first = BHIFirstLayersBlock(
                in_channels=in_channels,
                out_channels=320,
                seqsize=seqsize,
                kernel_sizes=[9, 15],
                pool_size=1,
                dropout=0.2,
            )
            core = BHICoreBlock(
                in_channels=first.out_channels,
                out_channels=320,
                seqsize=first.infer_outseqsize(),
                lstm_hidden_channels=320,
                kernel_sizes=[9, 15],
                pool_size=1,
                dropout1=0.2,
                dropout2=0.5,
            )
            # out_channels is unused here (the linear layer is dropped), but
            # AutosomeFinalLayersBlock's constructor still requires it.
            final = AutosomeFinalLayersBlockBinary(
                in_channels=core.out_channels, seqsize=core.infer_outseqsize(), out_channels=1
            )
            return PrixFixeNet(first=first, core=core, final=final, generator=torch.Generator())

        model = dream_rnn_binary(in_channels=4, seqsize=seq_len)
        return model, LitModel_Sahu_binary_dream_rnn, loss

    elif model_name.lower() == "lyra":
        raise NotImplementedError(
            "No binary ('paired') variant of Lyra exists yet: this needs (a) a head-stripped "
            "subclass mirroring HumanLegNetBinary/MPRAnnBinary above, and (b) a matching "
            "LitModel_Sahu_binary_lyra Lightning module analogous to LitModel_Sahu_binary_legnet. "
            "Neither is implemented — add them (or run Lyra on the non-Binary tasks only)."
        )

    elif model_name == "ReporterNet":
        raise NotImplementedError(
            "No binary ('paired') variant of ReporterNet exists yet: this needs (a) a "
            "head-stripped subclass mirroring HumanLegNetBinary/MPRAnnBinary above, and (b) a "
            "matching LitModel_Sahu_binary_reporternet Lightning module. Neither is implemented "
            "— add them (or run ReporterNet on the non-Binary tasks only)."
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Single train + eval run for one task
# =============================================================================

def run_single_task(task: str, args: argparse.Namespace, transforms: dict) -> float:
    """Train a fresh model on one task and return its test AUPR."""
    is_binary = task.lower() == "binary"

    # ---- Data ---------------------------------------------------------
    train_dataset = SahuDataset(split="train", task=task, transform=transforms["train"], root=args.root)
    val_dataset = SahuDataset(split="val", task=task, transform=transforms["test"], root=args.root)
    test_dataset = SahuDataset(split="test", task=task, transform=transforms["test"], root=args.root)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # ---- Model ----------------------------------------------------------
    if not is_binary:
        # Binary task's items are dicts ({"seq": ..., ...}); every other
        # task's items are plain (seq, target) tuples.
        seq_len = len(train_dataset[0][0][0])
        model, loss = build_model(model_name=args.model, seq_len=seq_len)
        lit_model_class = LitModel_Sahu
        lit_model = lit_model_class(model=model, loss=loss, weight_decay=args.wd, lr=args.lr, print_each=1)
    else:
        if args.model not in BINARY_CAPABLE_MODELS:
            raise ValueError(
                f"Model '{args.model}' has no 'Binary' task support yet "
                f"(supported: {sorted(BINARY_CAPABLE_MODELS)})."
            )
        seq_len = len(train_dataset[0][0]["seq"][0])
        model, lit_model_class, loss = build_binary_model(model_name=args.model, seq_len=seq_len)
        lit_model = lit_model_class(model=model, loss=loss, weight_decay=args.wd, lr=args.lr, print_each=1)

    # ---- Trainer setup ----------------------------------------------------
    checkpoint_callback = ModelCheckpoint(monitor="val_aupr", mode="max", save_top_k=1, save_last=False)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=[args.device],
        max_epochs=args.epoch_num,
        gradient_clip_val=1,
        precision="16-mixed",
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
    )

    # ---- Train --------------------------------------------------------
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Reload the best checkpoint (by val_aupr) before evaluating.
    best_model_path = checkpoint_callback.best_model_path
    lit_model = lit_model_class.load_from_checkpoint(
        best_model_path, model=model, loss=nn.BCEWithLogitsLoss(), weight_decay=args.wd, lr=args.lr, print_each=1
    )

    # ---- Test -----------------------------------------------------------
    result = trainer.test(lit_model, dataloaders=test_loader)
    return result[0]["test_aupr"]


# =============================================================================
# Main
# =============================================================================

def load_or_create_results(result_path: str, task: list[str]) -> pd.DataFrame:
    if os.path.exists(result_path):
        return pd.read_csv(result_path, sep="\t")
    return pd.DataFrame(columns=task)


def main() -> None:
    args = parse_args()
    results = load_or_create_results(args.result_dir, args.task)

    sequence_first = args.model.lower() == "lyra"
    transforms = build_transforms(sequence_first)

    for run_idx in range(args.runs):
        print(f"\n### Run {run_idx + 1}/{args.runs} — model: {args.model} — tasks: {args.task} ###")

        res = [run_single_task(task, args, transforms) for task in args.task]

        results.loc[len(results)] = res
        results.to_csv(args.result_dir, sep="\t", index=False)


if __name__ == "__main__":
    main()