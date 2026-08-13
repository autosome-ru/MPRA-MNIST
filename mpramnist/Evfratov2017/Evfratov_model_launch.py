"""
Evfratov_model_launch.py

Runs a benchmark: for each requested sequence length, trains a classification
model on the Evfratov2017 dataset and records its test F1 score, repeating
the whole process `--runs` times.

Results (one row per run, one column per sequence length: "<length>_f1") are
appended to a TSV file given by `--result_dir`.
"""

import argparse
import os

import pandas as pd
import torch.nn as nn
import lightning.pytorch as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

import mpramnist.transforms as t
from mpramnist.Evfratov2017 import EvfratovDataset, LitModel_Evfratov
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

# Whether the two lowest-expression classes are merged into one (see
# EvfratovDataset). Kept as a constant since the original script always set
# it to True.
MERGE_LAST_CLASSES = True

MALINOIS_PAD_LEN = 200  # Malinois needs a fixed input length, unlike the others


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a classification model on the Evfratov2017 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general = parser.add_argument_group("general", "General run settings")
    general.add_argument(
        "--result_dir",
        type=str,
        default="./evfratov.tsv",
        help="Path to the TSV file where per-run test F1 scores are appended.",
    )
    general.add_argument("--device", type=int, default=0, help="GPU device index to use.")
    general.add_argument(
        "--num_workers", type=int, default=16, help="Number of DataLoader worker processes."
    )
    general.add_argument("--batch_size", type=int, default=32, help="Batch size.")
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
        "--length",
        nargs="+",
        default=["23", "33"],
        help="Sequence length(s) to train/evaluate on; one model run per length.",
    )

    trainer_args = parser.add_argument_group("trainer", "Training hyperparameters")
    trainer_args.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    trainer_args.add_argument("--wd", type=float, default=0.1, help="Weight decay (AdamW).")
    trainer_args.add_argument("--epoch_num", type=int, default=50, help="Max number of epochs.")

    args = parser.parse_args()

    if isinstance(args.length, str):
        args.length = [args.length]

    return args


# =============================================================================
# Transforms
# =============================================================================

def build_transforms(model_name: str) -> dict:
    """Build the sequence transforms used for training/testing.

    Malinois needs a fixed-length, padded input; the other models work
    directly on the (variable-length) raw sequences.

    Returns a dict with keys: "train", "test".
    """
    if model_name == "Malinois":
        train_transform = t.Compose(
            [t.Padding(MALINOIS_PAD_LEN), t.ReverseComplement(0.5), t.Seq2Tensor()]
        )
        test_transform = t.Compose(
            [t.Padding(MALINOIS_PAD_LEN), t.Seq2Tensor(), t.ReverseComplement(0)]
        )
    else:
        train_transform = t.Compose([t.ReverseComplement(0.5), t.Seq2Tensor()])
        test_transform = t.Compose([t.Seq2Tensor(), t.ReverseComplement(0)])

    return {"train": train_transform, "test": test_transform}


# =============================================================================
# Model factory
# =============================================================================

def build_model(model_name: str, in_channels: int, seq_len: int, n_classes: int):
    """Instantiate the requested model.

    `in_channels` is the number of one-hot encoded channels (nucleotides).
    `seq_len` is the length of the input sequence.
    `n_classes` is the number of expression bins to classify into.
    """
    is_reporter_net = False

    if model_name == "MPRALegNet":
        model = HumanLegNet(
            in_ch=in_channels,
            output_dim=n_classes,
            stem_ch=64,
            stem_ks=11,
            ef_ks=9,
            ef_block_sizes=[80, 96, 112, 128],
            pool_sizes=[2, 2, 2, 2],
            resize_factor=4,
        )
        model.apply(initialize_weights)

    elif model_name == "MPRAnn":
        model = MPRAnn(output_dim=n_classes)

    elif model_name == "Malinois":
        model = BassetBranched(input_len=seq_len, n_outputs=n_classes)

    elif model_name == "PARM":
        model = PARM(n_block=5, type_loss="mse", output_dim=n_classes)

    elif model_name in ("DREAM-RNN", "DREAM_RNN"):
        model = DREAM_RNN(in_channels, seq_len, n_classes)

    elif model_name.lower() == "lyra":
        model = Lyra(d_input=in_channels, d_output=n_classes, d_model=512, dropout=0.1)

    elif model_name == "ReporterNet":
        model = ReporterNet(dropout_rate=0.2, output_dim=n_classes)
        model.apply(initialize_weights_reporternet)
        is_reporter_net = True

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, is_reporter_net


# =============================================================================
# Single train + eval run for one sequence length
# =============================================================================

def run_single_length(length: str, args: argparse.Namespace, transforms: dict) -> float:
    """Train a fresh model for one sequence length and return its test F1 score."""

    # ---- Data -------------------------------------------------------------
    train_dataset = EvfratovDataset(
        split="train",
        merge_last_classes=MERGE_LAST_CLASSES,
        length_of_seq=length,
        transform=transforms["train"],
        root=args.root,
    )
    val_dataset = EvfratovDataset(
        split="val",
        merge_last_classes=MERGE_LAST_CLASSES,
        length_of_seq=length,
        transform=transforms["test"],
        root=args.root,
    )
    test_dataset = EvfratovDataset(
        split="test",
        merge_last_classes=MERGE_LAST_CLASSES,
        length_of_seq=length,
        transform=transforms["test"],
        root=args.root,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    n_classes = train_dataset.n_classes

    sample_seq = train_dataset[0][0]
    if args.model.lower() == "lyra":
        # sequence_first=True -> tensor shape is (L, C)
        seq_len, in_channels = sample_seq.shape[0], sample_seq.shape[1]
    else:
        # channels-first -> tensor shape is (C, L)
        in_channels, seq_len = sample_seq.shape[0], sample_seq.shape[1]

    
    # ---- Model ------------------------------------------------------------
    model, is_reporter_net = build_model(args.model, in_channels=in_channels, seq_len=seq_len, n_classes=n_classes)

    lit_model = LitModel_Evfratov(
        model=model,
        loss=nn.CrossEntropyLoss(),
        n_classes=n_classes,
        weight_decay=args.wd,
        lr=args.lr,
        print_each=1,
        is_reporter_net=is_reporter_net
    )

    # ---- Trainer setup ------------------------------------------------------
    logger = pl_loggers.TensorBoardLogger(f"./{args.model}_logs", name=f"length_{length}")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_auroc", mode="max", save_top_k=1, save_last=False
    )

    early_stopping_callback = EarlyStopping(
            monitor="val_pearson", mode="max", patience=8, min_delta=0.0
        )
    
    trainer = L.Trainer(
        accelerator="gpu",
        devices=[args.device],
        max_epochs=args.epoch_num,
        gradient_clip_val=1,
        precision="16-mixed",
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    # ---- Train --------------------------------------------------------------
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Reload the best checkpoint (by val_auroc) before evaluating.
    best_model_path = checkpoint_callback.best_model_path
    lit_model = LitModel_Evfratov.load_from_checkpoint(
        best_model_path,
        model=model,
        loss=nn.CrossEntropyLoss(),
        n_classes=n_classes,
        weight_decay=args.wd,
        lr=args.lr,
        print_each=1,
    )

    # ---- Test ---------------------------------------------------------------
    result = trainer.test(lit_model, dataloaders=test_loader)
    return result[0]["val_f1"]


# =============================================================================
# Main
# =============================================================================

def load_or_create_results(result_path: str, lengths: list[str]) -> pd.DataFrame:
    if os.path.exists(result_path):
        return pd.read_csv(result_path, sep="\t")
    columns = [f"{length}_f1" for length in lengths]
    return pd.DataFrame(columns=columns)


def main() -> None:
    args = parse_args()
    results = load_or_create_results(args.result_dir, args.length)
    transforms = build_transforms(args.model)

    for run in range(args.runs):
        print(f"\n### Run {run + 1}/{args.runs} — model: {args.model} — lengths: {args.length} ###")

        f1_per_length = [run_single_length(length, args, transforms) for length in args.length]

        results.loc[len(results)] = f1_per_length
        results.to_csv(args.result_dir, sep="\t", index=False)


if __name__ == "__main__":
    main()