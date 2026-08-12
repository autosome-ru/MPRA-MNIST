"""
Vaishnav_model_launch.py

Runs a benchmark: for each dataset environment type (e.g. "defined",
"complex"), trains a fresh single-output model on the Vaishnav2024 dataset,
repeating the whole process `--runs` times. Test-time evaluation averages
forward-strand and reverse-complement predictions and reports Pearson r for
each of the "native", "drift", and "paired" test subsets. For "paired" the
evaluated quantity is the ref/alt effect size (alt - ref) rather than the
raw prediction.

Results (one row per run, one column per data_env_type x test_type
combination) are appended to a TSV file given by `--result_dir`.
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import lightning.pytorch as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics import PearsonCorrCoef

import mpramnist.transforms as t
from mpramnist.Vaishnav2024 import VaishnavDataset, LitModel_Vaishnav
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

SEQ_LEN = 110  # fixed insert length used by the padding/cropping transforms below
TEST_TYPES = ["native", "drift", "paired"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a single-output model on the Vaishnav2024 dataset, "
        "per dataset environment type.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general = parser.add_argument_group("general", "General run settings")
    general.add_argument(
        "--result_dir",
        type=str,
        default="./vaishnav.tsv",
        help="Path to the TSV file where per-run Pearson correlations are appended.",
    )
    general.add_argument("--device", type=int, default=0, help="GPU device index to use.")
    general.add_argument(
        "--num_workers", type=int, default=16, help="Number of DataLoader worker processes."
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
        "--data_env_type",
        nargs="+",
        default=["defined", "complex"],
        help="List of dataset environment types; a separate model is trained per environment type.",
    )

    trainer_args = parser.add_argument_group("trainer", "Training hyperparameters")
    trainer_args.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    trainer_args.add_argument("--wd", type=float, default=1e-2, help="Weight decay (AdamW).")
    trainer_args.add_argument("--epoch_num", type=int, default=5, help="Max number of epochs.")

    args = parser.parse_args()

    if isinstance(args.data_env_type, str):
        args.data_env_type = [args.data_env_type]

    return args


# =============================================================================
# Transforms
# =============================================================================

def build_transforms(dataset_cls: type[VaishnavDataset], sequence_first: bool = False) -> dict:
    """Build all the sequence transforms used for training/testing/inference.

    The insert is embedded in its native plasmid context: the flanks are
    taken directly from `VaishnavDataset.PLASMID` around the 80-N insert
    site, then cropped/left-aligned to `SEQ_LEN`.

    `sequence_first` controls the axis order produced by `Seq2Tensor`. Most
    models expect channels-first tensor (`sequence_first=False`), but some
    (e.g. Lyra) expect the sequence-length axis first, so they need
    `sequence_first=True`.

    Returns a dict with keys: "train", "test", "forward", "reverse".
    """
    plasmid = dataset_cls.PLASMID.upper()
    insert_start = plasmid.find("N" * 80)
    right_flank = dataset_cls.RIGHT_FLANK
    left_flank = plasmid[insert_start - SEQ_LEN : insert_start]

    train_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.LeftCrop(SEQ_LEN, SEQ_LEN),
            t.ReverseComplement(0.5),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )
    test_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.LeftCrop(SEQ_LEN, SEQ_LEN),
            t.ReverseComplement(0),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )

    # Used at test time to average forward-strand and reverse-complement predictions.
    forward_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.LeftCrop(SEQ_LEN, SEQ_LEN),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )
    reverse_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.LeftCrop(SEQ_LEN, SEQ_LEN),
            t.ReverseComplement(1),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )

    return {
        "train": train_transform,
        "test": test_transform,
        "forward": forward_transform,
        "reverse": reverse_transform,
    }


# =============================================================================
# Model factory
# =============================================================================

def build_model(model_name: str, in_channels: int, seq_len: int):
    """Instantiate the requested model + loss function.

    `in_channels` is the number of one-hot encoded channels (nucleotides).
    `seq_len` is the length of the (cropped) input sequence.
    The model always has a single output for this dataset.
    """
    loss = nn.MSELoss()
    learning_rate_reduce = True
    lr_sheduler_to_use = "one_cycle"

    if model_name == "MPRALegNet":
        model = HumanLegNet(
            in_ch=in_channels,
            output_dim=1,
            stem_ch=64,
            stem_ks=11,
            ef_ks=9,
            ef_block_sizes=[80, 96, 112, 128],
            pool_sizes=[2, 2, 2, 2],
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
        model = DREAM_RNN(in_channels=in_channels, seqsize=seq_len, out_channels=1)

    elif model_name.lower() == "lyra":
        model = Lyra(d_input=in_channels, d_output=1, d_model=512, dropout=0.1)
        lr_sheduler_to_use = "reducelronplateau"

    elif model_name == "ReporterNet":
        model = ReporterNet(dropout_rate=0.2, output_dim=1)
        model.apply(initialize_weights_reporternet)
        learning_rate_reduce = None

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, loss, lr_sheduler_to_use, learning_rate_reduce


# =============================================================================
# Evaluation helpers
# =============================================================================

def evaluate_with_reverse_complement(
    forward_loader: DataLoader,
    reverse_loader: DataLoader,
    trainer: L.Trainer,
    lit_model: LitModel_Vaishnav,
    name: str,
    is_paired: bool,
) -> torch.Tensor:
    """Predict on forward-strand and reverse-complement inputs separately,
    average ref (and, for the "paired" test type, alt) predictions across
    strands, and compute the Pearson correlation against the targets.

    For the "paired" test type the evaluated quantity is the ref/alt effect
    size (mean_alt - mean_ref); otherwise it's the raw (averaged) prediction.
    """
    forward_preds = trainer.predict(lit_model, dataloaders=forward_loader)
    targets = torch.cat([batch["target"] for batch in forward_preds])
    forward_ref = torch.cat([batch["ref_predicted"] for batch in forward_preds])

    reverse_preds = trainer.predict(lit_model, dataloaders=reverse_loader)
    reverse_ref = torch.cat([batch["ref_predicted"] for batch in reverse_preds])

    mean_ref = torch.mean(torch.stack([forward_ref, reverse_ref]), dim=0)

    pearson_metric = PearsonCorrCoef()

    if is_paired:
        forward_alt = torch.cat([batch["alt_predicted"] for batch in forward_preds])
        reverse_alt = torch.cat([batch["alt_predicted"] for batch in reverse_preds])
        mean_alt = torch.mean(torch.stack([forward_alt, reverse_alt]), dim=0)
        pred = mean_alt - mean_ref
    else:
        pred = mean_ref

    pearson = pearson_metric(pred, targets)

    print(f"{name} Pearson correlation")
    print(pearson.item())

    return pearson


# =============================================================================
# Single train + eval run for one data_env_type
# =============================================================================

def run_single_env(env: str, args: argparse.Namespace, transforms: dict) -> list[float]:
    """Train a fresh single-output model on one dataset environment type and
    return the Pearson r for each of the "native", "drift", "paired" test
    subsets.
    """
    # ---- Data ---------------------------------------------------------
    train_dataset = VaishnavDataset(
        split="train", dataset_env_type=env, transform=transforms["train"], root=args.root
    )
    val_dataset = VaishnavDataset(
        split="val", dataset_env_type=env, transform=transforms["test"], root=args.root
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    sample_seq = train_dataset[0][0]
    if args.model.lower() == "lyra":
        # sequence_first=True -> tensor shape is (L, C)
        seq_len, in_channels = sample_seq.shape[0], sample_seq.shape[1]
    else:
        # channels-first -> tensor shape is (C, L)
        in_channels, seq_len = sample_seq.shape[0], sample_seq.shape[1]

    # ---- Model ----------------------------------------------------------
    model, loss, lr_sheduler_to_use, learning_rate_reduce = build_model(model_name=args.model, in_channels=in_channels, seq_len=seq_len)

    lit_model = LitModel_Vaishnav(model=model, loss=loss, weight_decay=args.wd, lr=args.lr, print_each=1, lr_sheduler_to_use=lr_sheduler_to_use, 
                                  learning_rate_reduce=learning_rate_reduce)

    # ---- Trainer setup --------------------------------------------------
    logger = pl_loggers.TensorBoardLogger(f"./{args.model}_logs", name=env)

    early_stopping_callback = EarlyStopping(
        monitor="val_pearson", mode="max", patience=8, min_delta=0.0
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_pearson", mode="max", save_top_k=1, save_last=False
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=[args.device],
        max_epochs=args.epoch_num,
        gradient_clip_val=1,
        precision="16-mixed",
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger = logger
    )

    # ---- Train --------------------------------------------------------
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Reload the best checkpoint (by val_pearson) before evaluating.
    best_model_path = checkpoint_callback.best_model_path
    lit_model = LitModel_Vaishnav.load_from_checkpoint(
        best_model_path, model=model, loss=nn.MSELoss(), weight_decay=args.wd, lr=args.lr, print_each=1
    )

    # ---- Test: per test_type, average forward-strand and reverse-complement
    #      predictions, then compute Pearson r --------------------------------
    r_values = []
    for test_type in TEST_TYPES:
        is_paired = test_type == "paired"

        test_forward_dataset = VaishnavDataset(
            split="test",
            dataset_env_type=env,
            test_dataset_type=test_type,
            transform=transforms["forward"],
            root=args.root,
        )
        test_reverse_dataset = VaishnavDataset(
            split="test",
            dataset_env_type=env,
            test_dataset_type=test_type,
            transform=transforms["reverse"],
            root=args.root,
        )

        forward_loader = DataLoader(
            test_forward_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        reverse_loader = DataLoader(
            test_reverse_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        pearson = evaluate_with_reverse_complement(
            forward_loader, reverse_loader, trainer, lit_model, test_type, is_paired
        )

        r_values.append(float(pearson.numpy()))

    return r_values


# =============================================================================
# Main
# =============================================================================

def build_result_columns(data_env_type: list[str]) -> list[str]:
    return [env + "_" + test_type for env in data_env_type for test_type in TEST_TYPES]


def load_or_create_results(result_path: str, data_env_type: list[str]) -> pd.DataFrame:
    if os.path.exists(result_path):
        return pd.read_csv(result_path, sep="\t")
    return pd.DataFrame(columns=build_result_columns(data_env_type))


def main() -> None:
    args = parse_args()
    results = load_or_create_results(args.result_dir, args.data_env_type)

    # Lyra expects sequence-length-first tensors; every other model here
    # expects channels-first tensors.
    sequence_first = args.model.lower() == "lyra"
    transforms = build_transforms(VaishnavDataset, sequence_first=sequence_first)

    for run_idx in range(args.runs):
        print(f"\n### Run {run_idx + 1}/{args.runs} — model: {args.model} — envs: {args.data_env_type} ###")

        r_values = []
        for env in args.data_env_type:
            r_values.extend(run_single_env(env, args, transforms))

        results.loc[len(results)] = r_values
        results.to_csv(args.result_dir, sep="\t", index=False)


if __name__ == "__main__":
    main()