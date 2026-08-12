"""
AgarwalSingle_model_launch.py

Runs a benchmark: for each requested model architecture, trains and evaluates it
on one or more cell types from the Agarwal MPRA dataset, repeating the whole
process `--runs` times to get a distribution of Pearson correlations.

Results (one row per run, one column per cell type) are appended to a TSV file
given by `--result_dir`.
"""

import argparse
import os
from pathlib import Path

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
from mpramnist.Agarwal2025.trainer import LitModel_AgarwalSingle
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a model on the Agarwal (single-cell-type) MPRA dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general = parser.add_argument_group("general", "General run settings")
    general.add_argument(
        "--result_dir",
        type=str,
        default="./agarwalsingle.tsv",
        help="Path to the TSV file where per-run Pearson correlations are appended.",
    )
    general.add_argument("--device", type=int, default=0, help="GPU device index to use.")
    general.add_argument(
        "--num_workers", type=int, default=16, help="Number of DataLoader worker processes."
    )
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
        "--cell_types",
        nargs="+",
        default=["HepG2", "K562", "WTC11"],
        help="Cell type(s) to train/evaluate on, one model run per cell type.",
    )

    trainer_args = parser.add_argument_group("trainer", "Training hyperparameters")
    trainer_args.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    trainer_args.add_argument("--wd", type=float, default=0.1, help="Weight decay (AdamW).")
    trainer_args.add_argument("--batch_size", type=int, default=1024, help="Batch size.")
    trainer_args.add_argument("--epoch_num", type=int, default=50, help="Max number of epochs.")

    args = parser.parse_args()

    if isinstance(args.cell_types, str):
        args.cell_types = [args.cell_types]

    return args


# =============================================================================
# Transforms
# =============================================================================

def build_transforms(dataset_cls: type[AgarwalSingleDataset], sequence_first: bool = False) -> dict:
    """Build all the sequence transforms used for training/testing/inference.

    `sequence_first` controls the axis order produced by `Seq2Tensor`. Most
    models expect channels-first tensor (`sequence_first=False`), but some
    (e.g. Lyra) expect the sequence-length axis first, so they need
    `sequence_first=True`.

    Returns a dict with keys: "train", "test", "forward", "reverse".
    """
    constant_left_flank = dataset_cls.CONSTANT_LEFT_FLANK  # required for every sequence
    constant_right_flank = dataset_cls.CONSTANT_RIGHT_FLANK
    right_flank = dataset_cls.RIGHT_FLANK  # original flank from human_legnet

    train_transform = t.Compose(
        [
            t.AddFlanks(constant_left_flank, constant_right_flank),
            t.AddFlanks("", right_flank),  # matches original human_legnet preprocessing
            t.RightCrop(230, 260),  # shift augmentation
            t.LeftCrop(230, 230),
            t.ReverseComplement(0.5),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )

    test_transform = t.Compose(
        [
            t.AddFlanks(constant_left_flank, constant_right_flank),
            t.ReverseComplement(0),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )

    # Used at test time to average forward-strand and reverse-complement predictions.
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
        "test": test_transform,
        "forward": forward_transform,
        "reverse": reverse_transform,
    }


# =============================================================================
# Model factory
# =============================================================================

def build_model(model_name: str, in_channels: int, seq_len: int):
    """Instantiate the requested model + loss function + whether to use a
    ReduceLROnPlateau scheduler.

    `in_channels` is the number of one-hot encoded channels (nucleotides).
    `seq_len` is the length of the (cropped) input sequence.
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
        model = DREAM_RNN(in_channels, 230, 1)

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
    lit_model: LitModel_AgarwalSingle,
    cell_type: str,
    num_outputs: int,
) -> torch.Tensor:
    """Predict on forward-strand and reverse-complement inputs separately,
    average the two predictions, and compute the Pearson correlation against
    the (shared) targets.
    """
    forward_preds = trainer.predict(lit_model, dataloaders=forward_loader)
    targets = torch.cat([batch["target"] for batch in forward_preds])
    forward_pred_values = torch.cat([batch["predicted"] for batch in forward_preds])

    reverse_preds = trainer.predict(lit_model, dataloaders=reverse_loader)
    reverse_pred_values = torch.cat([batch["predicted"] for batch in reverse_preds])

    mean_pred = torch.mean(torch.stack([forward_pred_values, reverse_pred_values]), dim=0)

    pearson_metric = PearsonCorrCoef(num_outputs=num_outputs)
    pearson = pearson_metric(mean_pred, targets)

    print("===========")
    print(f"{cell_type} Pearson correlation: {pearson}")
    print("===========")

    return pearson


# =============================================================================
# Single train + eval run for one cell type
# =============================================================================

def run_single_cell_type(cell_type: str, args: argparse.Namespace, transforms: dict) -> torch.Tensor:
    """Train a fresh model on one cell type and return its test Pearson correlation."""

    # ---- Data -------------------------------------------------------------
    train_dataset = AgarwalSingleDataset(
        cell_type=cell_type, split="train", transform=transforms["train"], root=args.root
    )
    val_dataset = AgarwalSingleDataset(
        cell_type=cell_type, split="val", transform=transforms["test"], root=args.root
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

    # ---- Model --------------------------------------------------------
    model, loss, lr_sheduler_to_use, learning_rate_reduce = build_model(
        model_name=args.model,
        in_channels=in_channels,
        seq_len=seq_len,
    )

    lit_model = LitModel_AgarwalSingle(
        model=model,
        loss=loss,
        weight_decay=args.wd,
        lr=args.lr,
        print_each=1,
        lr_sheduler_to_use=lr_sheduler_to_use,
        learning_rate_reduce=learning_rate_reduce,
    )

    # ---- Trainer setup --------------------------------------------------
    logger = pl_loggers.TensorBoardLogger(f"./{args.model}_logs", name=cell_type)

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

    # ---- Train ------------------------------------------------------------
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Reload the best checkpoint (by val_pearson) before evaluating.
    best_model_path = checkpoint_callback.best_model_path
    lit_model = LitModel_AgarwalSingle.load_from_checkpoint(
        best_model_path, model=model, loss=loss, weight_decay=args.wd, lr=args.lr, print_each=1
    )

    # ---- Test: average forward-strand and reverse-complement predictions --
    test_forward_dataset = AgarwalSingleDataset(
        cell_type=cell_type, split="test", transform=transforms["forward"], root=args.root
    )
    test_reverse_dataset = AgarwalSingleDataset(
        cell_type=cell_type, split="test", transform=transforms["reverse"], root=args.root
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
        forward_loader, reverse_loader, trainer, lit_model, cell_type, num_outputs=1
    )

    return pearson


# =============================================================================
# Main
# =============================================================================

def load_or_create_results(result_path: str, cell_types: list[str]) -> pd.DataFrame:
    if os.path.exists(result_path):
        return pd.read_csv(result_path, sep="\t")
    return pd.DataFrame(columns=cell_types)


def main() -> None:
    args = parse_args()
    results = load_or_create_results(args.result_dir, args.cell_types)

    # Lyra expects sequence-length-first tensors; every other model here
    # expects channels-first tensors.
    sequence_first = args.model.lower() == "lyra"
    transforms = build_transforms(AgarwalSingleDataset, sequence_first=sequence_first)

    for run in range(args.runs):
        print(f"\n### Run {run + 1}/{args.runs} — model: {args.model} ###")

        pearsons_per_cell_type = []
        for cell_type in args.cell_types:
            pearson = run_single_cell_type(cell_type, args, transforms)
            pearsons_per_cell_type.append(pearson.numpy())

        results.loc[len(results)] = pearsons_per_cell_type
        results.to_csv(args.result_dir, sep="\t", index=False)


if __name__ == "__main__":
    main()