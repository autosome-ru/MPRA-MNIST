"""
Kircher_model_launch.py

Runs a benchmark: trains a single-output model on the AgarwalSingle
dataset, repeating the whole process `--runs` times, then evaluates it
(without further fine-tuning) on the Kircher2019 dataset. Kircher sequences
are embedded in AgarwalSingle's promoter/enhancer flank context so the
trained model can score them directly. Test-time evaluation averages
forward-strand and reverse-complement predictions, then reports the Pearson
correlation of the ref/alt effect size (alt - ref) against the targets.

Results (one row per run, one column, named after `--cell_types`) are
appended to a TSV file given by `--result_dir`.
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
from mpramnist.Agarwal2025.dataset import AgarwalSingleDataset
from mpramnist.Kircher2019.dataset import KircherDataset
from mpramnist.Kircher2019.trainer import LitModel_Kircher
from mpramnist.models import (
    HumanLegNet,
    initialize_weights,
    BassetBranched,
    L1KLmixed,
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

KIRCHER_ELEMENT_LEN = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a single-output model on AgarwalSingle and evaluate it on Kircher2019.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general = parser.add_argument_group("general", "General run settings")
    general.add_argument(
        "--result_dir",
        type=str,
        default="./kircher.tsv",
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
        "--cell_types", type=str, default="HepG2", help="AgarwalSingle cell type to train on."
    )
    dataset_args.add_argument(
        "--elements",
        nargs="+",
        default=["LDLR.2", "LDLR"],
        help="List of Kircher promoter/enhancer elements to evaluate on.",
    )

    trainer_args = parser.add_argument_group("trainer", "Training hyperparameters")
    trainer_args.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    trainer_args.add_argument("--wd", type=float, default=0.1, help="Weight decay (AdamW).")
    trainer_args.add_argument("--epoch_num", type=int, default=50, help="Max number of epochs.")

    return parser.parse_args()


# =============================================================================
# Transforms
# =============================================================================

def build_transforms(dataset_cls: type[AgarwalSingleDataset], sequence_first: bool = False) -> dict:
    """Build all the sequence transforms used for training/validation/inference.

    Training/validation use `AgarwalSingleDataset`'s own flanks. The final
    test-time transforms ("forward"/"reverse") also use AgarwalSingle's
    flanks — they're applied to *Kircher* sequences, embedding them in the
    same promoter/enhancer context the model was trained on.

    `sequence_first` controls the axis order produced by `Seq2Tensor`. Most
        models expect channels-first tensor (`sequence_first=False`), but some
        (e.g. Lyra) expect the sequence-length axis first, so they need
        `sequence_first=True`.

    Returns a dict with keys: "train", "val", "forward", "reverse".
    """
    constant_left_flank = dataset_cls.CONSTANT_LEFT_FLANK
    constant_right_flank = dataset_cls.CONSTANT_RIGHT_FLANK
    right_flank = dataset_cls.RIGHT_FLANK

    train_transform = t.Compose(
        [
            t.AddFlanks(constant_left_flank, constant_right_flank),
            t.AddFlanks("", right_flank),
            t.RightCrop(230, 260),  # shift augmentation
            t.LeftCrop(230, 230),
            t.ReverseComplement(0.5),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )
    val_transform = t.Compose(
        [
            t.AddFlanks(constant_left_flank, constant_right_flank),
            t.ReverseComplement(0),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )

    # Used at test time (on KircherDataset) to average forward-strand and
    # reverse-complement predictions.
    forward_transform = t.Compose(
        [
            t.AddFlanks(constant_left_flank, constant_right_flank),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )
    reverse_transform = t.Compose(
        [
            t.AddFlanks(constant_left_flank, constant_right_flank),
            t.ReverseComplement(1),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )

    return {
        "train": train_transform,
        "val": val_transform,
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

    Returns (model, loss, lr_scheduler_to_use, learning_rate_reduce).
    """
    loss = nn.MSELoss()
    learning_rate_reduce = True
    lr_scheduler_to_use = "one_cycle"

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
        model = BassetBranched(input_len=seq_len, n_outputs=1, loss_criterion=L1KLmixed)
        loss = L1KLmixed()

    elif model_name == "PARM":
        model = PARM(n_block=5, type_loss="mse", output_dim=1)

    elif model_name in ("DREAM-RNN", "DREAM_RNN"):
        model = DREAM_RNN(in_channels, seq_len, 1)

    elif model_name.lower() == "lyra":
        model = Lyra(d_input=in_channels, d_output=1, d_model=512, dropout=0.1)
        lr_scheduler_to_use = "reducelronplateau"

    elif model_name == "ReporterNet":
        model = ReporterNet(dropout_rate=0.2, output_dim=1)
        model.apply(initialize_weights_reporternet)
        learning_rate_reduce = None

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, loss, lr_scheduler_to_use, learning_rate_reduce


# =============================================================================
# Evaluation helpers
# =============================================================================

def evaluate_ref_alt_effect(
    forward_loader: DataLoader,
    reverse_loader: DataLoader,
    trainer: L.Trainer,
    lit_model: LitModel_Kircher,
    name,
) -> torch.Tensor:
    """Predict on forward-strand and reverse-complement inputs separately,
    average ref and alt predictions across strands, and compute the Pearson
    correlation between the ref/alt effect size (mean_alt - mean_ref) and
    the targets.
    """
    forward_preds = trainer.predict(lit_model, dataloaders=forward_loader)
    targets = torch.cat([batch["target"] for batch in forward_preds])
    forward_ref = torch.cat([batch["ref_predicted"] for batch in forward_preds])
    forward_alt = torch.cat([batch["alt_predicted"] for batch in forward_preds])

    reverse_preds = trainer.predict(lit_model, dataloaders=reverse_loader)
    reverse_ref = torch.cat([batch["ref_predicted"] for batch in reverse_preds])
    reverse_alt = torch.cat([batch["alt_predicted"] for batch in reverse_preds])

    mean_ref = torch.mean(torch.stack([forward_ref, reverse_ref]), dim=0)
    mean_alt = torch.mean(torch.stack([forward_alt, reverse_alt]), dim=0)
    pred = (mean_alt - mean_ref).squeeze(-1)

    pearson_metric = PearsonCorrCoef()
    pearson = pearson_metric(pred, targets)

    print(name, " Pearson correlation")
    print(pearson.item())

    return pearson


# =============================================================================
# Single train + eval run
# =============================================================================

def run_single_training_run(run_idx: int, args: argparse.Namespace, transforms: dict) -> torch.Tensor:
    """Train a fresh single-output model on AgarwalSingle and return the
    Pearson correlation of its ref/alt effect predictions on Kircher.
    """
    print(f"\n### Run {run_idx + 1}/{args.runs} — model: {args.model} — cell type: {args.cell_types} ###")

    # ---- Data (AgarwalSingle, for training) ----------------------------
    train_dataset = AgarwalSingleDataset(
        cell_type=args.cell_types, split="train", transform=transforms["train"], root=args.root
    )
    val_dataset = AgarwalSingleDataset(
        cell_type=args.cell_types, split="val", transform=transforms["val"], root=args.root
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
    model, loss, lr_scheduler_to_use, learning_rate_reduce = build_model(
        model_name=args.model, in_channels=in_channels, seq_len=seq_len
    )

    lit_model = LitModel_Kircher(
        model=model,
        loss=loss,
        weight_decay=args.wd,
        lr=args.lr,
        print_each=1,
        lr_sheduler_to_use=lr_scheduler_to_use,
        learning_rate_reduce=learning_rate_reduce,
    )

    # ---- Trainer setup ----------------------------------------------------
    logger = pl_loggers.TensorBoardLogger(f"./{args.model}_logs", name="_".join(args.cell_types))

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
    lit_model = LitModel_Kircher.load_from_checkpoint(
        best_model_path, model=model, loss=nn.MSELoss(), weight_decay=args.wd, lr=args.lr, print_each=1
    )

    # ---- Test: evaluate on Kircher, averaging forward-strand and
    #      reverse-complement predictions -----------------------------------
    test_forward_dataset = KircherDataset(
        length=KIRCHER_ELEMENT_LEN, elements=args.elements, transform=transforms["forward"], root=args.root
    )
    test_reverse_dataset = KircherDataset(
        length=KIRCHER_ELEMENT_LEN, elements=args.elements, transform=transforms["reverse"], root=args.root
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

    pearson = evaluate_ref_alt_effect(forward_loader, reverse_loader, trainer, lit_model, args.elements)

    return pearson


# =============================================================================
# Main
# =============================================================================

def load_or_create_results(result_path: str, cell_types: str) -> pd.DataFrame:
    if os.path.exists(result_path):
        return pd.read_csv(result_path, sep="\t")
    return pd.DataFrame(columns=[cell_types])


def main() -> None:
    args = parse_args()
    results = load_or_create_results(args.result_dir, args.cell_types)

    # Lyra expects sequence-length-first tensors; every other model here
    # expects channels-first tensors.
    sequence_first = args.model.lower() == "lyra"
    transforms = build_transforms(AgarwalSingleDataset, sequence_first=sequence_first)

    for run_idx in range(args.runs):
        pearson = run_single_training_run(run_idx, args, transforms)
        results.loc[len(results)] = pearson.numpy()
        results.to_csv(args.result_dir, sep="\t", index=False)


if __name__ == "__main__":
    main()