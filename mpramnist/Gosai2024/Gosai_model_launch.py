"""
Gosai_model_launch.py

Runs a benchmark: trains a single multi-output model (one output per cell
type) on the Gosai MPRA dataset, repeating the whole process `--runs` times
to get a distribution of Pearson correlations per cell type.

Results (one row per run, one column per cell type) are appended to a TSV file
given by `--result_dir`.
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
from mpramnist.Gosai2024.dataset import GosaiDataset
from mpramnist.Gosai2024.trainer import LitModel_Gosai
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

# GosaiDataset-specific "own" filtration settings, reused for every split.
FILTRATION = "own"
STDERR_THRESHOLD = 1.0
STD_MULTIPLE_CUT = 6.0
UP_CUTOFF_MOVE = 3.0
DUPLICATION_CUTOFF = 0.5  # only used for the train split
SEQ_LEN = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a multi-output model (one output per "
        "cell type) on the Gosai MPRA dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general = parser.add_argument_group("general", "General run settings")
    general.add_argument(
        "--result_dir",
        type=str,
        default="./gosai.tsv",
        help="Path to the TSV file where per-run Pearson correlations are appended.",
    )
    general.add_argument("--device", type=int, default=0, help="GPU device index to use.")
    general.add_argument(
        "--num_workers", type=int, default=8, help="Number of DataLoader worker processes."
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
        default=["HepG2", "K562", "SKNSH"],
        help="Cell types to train on jointly; the model gets one output per cell type.",
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
# Dataset helper
# =============================================================================

def build_dataset(split: str, transform, cell_types: list[str], root: str) -> GosaiDataset:
    """Instantiate a GosaiDataset split with the shared "own" filtration
    settings. The duplication cutoff is only meaningful for the train split;
    GosaiDataset ignores it otherwise, so it's always passed for simplicity.
    """
    stderr_columns = [cell + "_lfcSE" for cell in cell_types]

    return GosaiDataset(
        split=split,
        transform=transform,
        filtration=FILTRATION,
        cell_types=cell_types.copy(),
        stderr_columns=stderr_columns,
        stderr_threshold=STDERR_THRESHOLD,
        std_multiple_cut=STD_MULTIPLE_CUT,
        up_cutoff_move=UP_CUTOFF_MOVE,
        duplication_cutoff=DUPLICATION_CUTOFF,
        root=root,
    )


# =============================================================================
# Transforms
# =============================================================================

def build_transforms(dataset_cls: type[GosaiDataset], sequence_first: bool = False) -> dict:
    """Build all the sequence transforms used for training/testing/inference.

    `sequence_first` controls the axis order produced by `Seq2Tensor`. Most
        models expect channels-first tensor (`sequence_first=False`), but some
        (e.g. Lyra) expect the sequence-length axis first, so they need
        `sequence_first=True`.

    Returns a dict with keys: "train", "test", "forward", "reverse".
    """
    left_flank = dataset_cls.LEFT_FLANK
    right_flank = dataset_cls.RIGHT_FLANK

    train_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.CenterCrop(SEQ_LEN),
            t.ReverseComplement(0.5),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )

    test_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.CenterCrop(SEQ_LEN),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )

    # Used at test time to average forward-strand and reverse-complement predictions.
    forward_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.CenterCrop(SEQ_LEN),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )
    reverse_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.CenterCrop(SEQ_LEN),
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

def build_model(model_name: str, in_channels: int, seq_len: int, num_outputs: int):
    """Instantiate the requested model + loss function.

    `in_channels` is the number of one-hot encoded channels (nucleotides).
    `seq_len` is the length of the (cropped) input sequence.
    `num_outputs` is the number of cell types (one model output per cell type).
    """
    loss = nn.MSELoss()
    learning_rate_reduce = True
    lr_sheduler_to_use = "one_cycle"

    if model_name == "MPRALegNet":
        model = HumanLegNet(
            in_ch=in_channels,
            output_dim=num_outputs,
            stem_ch=64,
            stem_ks=11,
            ef_ks=9,
            ef_block_sizes=[80, 96, 112, 128],
            pool_sizes=[2, 2, 2, 2],
            resize_factor=4,
        )
        model.apply(initialize_weights)

    elif model_name == "MPRAnn":
        model = MPRAnn(output_dim=num_outputs)

    elif model_name == "Malinois":
        model = BassetBranched(input_len=seq_len, n_outputs=num_outputs)
        loss = L1KLmixed()
        lr_sheduler_to_use = "cosineannealing"

    elif model_name == "PARM":
        model = PARM(n_block=5, type_loss="mse", output_dim=num_outputs)

    elif model_name in ("DREAM-RNN", "DREAM_RNN"):
        model = DREAM_RNN(in_channels=in_channels, seqsize=seq_len, out_channels=num_outputs)

    elif model_name.lower() == "lyra":
        model = Lyra(d_input=in_channels, d_output=num_outputs, d_model=512, dropout=0.1)
        lr_sheduler_to_use = "reducelronplateau"

    elif model_name == "ReporterNet":
        model = ReporterNet(dropout_rate=0.2, output_dim=num_outputs)
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
    lit_model: LitModel_Gosai,
    cell_types: list[str],
    num_outputs: int,
) -> torch.Tensor:
    """Predict on forward-strand and reverse-complement inputs separately,
    average the two predictions, and compute the Pearson correlation (per
    cell type) against the (shared) targets.
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
    print(cell_types, " Pearson correlation")
    print(pearson)
    print("===========")

    return pearson


# =============================================================================
# Single train + eval run (all cell types trained jointly)
# =============================================================================

def run_single_training_run(run_idx: int, args: argparse.Namespace, transforms: dict) -> torch.Tensor:
    """Train a fresh multi-output model on all requested cell types and
    return the per-cell-type test Pearson correlations.
    """
    print(f"\n### Run {run_idx + 1}/{args.runs} — model: {args.model} — cell types: {args.cell_types} ###")

    # ---- Data ---------------------------------------------------------
    train_dataset = build_dataset(
        split="train", transform=transforms["train"], cell_types=args.cell_types, root=args.root
    )
    val_dataset = build_dataset(
        split="val", transform=transforms["test"], cell_types=args.cell_types, root=args.root
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=4,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=4,
        num_workers=args.num_workers,
    )

    sample_seq = train_dataset[0][0]
    if args.model.lower() == "lyra":
        # sequence_first=True -> tensor shape is (L, C)
        seq_len, in_channels = sample_seq.shape[0], sample_seq.shape[1]
    else:
        # channels-first -> tensor shape is (C, L)
        in_channels, seq_len = sample_seq.shape[0], sample_seq.shape[1]

    # ---- Model ----------------------------------------------------------
    model, loss, lr_sheduler_to_use, learning_rate_reduce = build_model(
        model_name=args.model,
        in_channels=in_channels,
        seq_len=seq_len,
        num_outputs=len(args.cell_types),
    )

    lit_model = LitModel_Gosai(
        model=model,
        loss=loss,
        weight_decay=args.wd,
        lr=args.lr,
        cell_types=args.cell_types,
        print_each=1,
        lr_sheduler_to_use=lr_sheduler_to_use,
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
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    # ---- Train --------------------------------------------------------
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Reload the best checkpoint (by val_pearson) before evaluating.
    best_model_path = checkpoint_callback.best_model_path
    lit_model = LitModel_Gosai.load_from_checkpoint(
        best_model_path,
        model=model,
        loss=loss,
        weight_decay=args.wd,
        lr=args.lr,
        cell_types=args.cell_types,
        print_each=1,
    )

    # ---- Test: average forward-strand and reverse-complement predictions --
    test_forward_dataset = build_dataset(
        split="test", transform=transforms["forward"], cell_types=args.cell_types, root=args.root
    )
    test_reverse_dataset = build_dataset(
        split="test", transform=transforms["reverse"], cell_types=args.cell_types, root=args.root
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
        forward_loader,
        reverse_loader,
        trainer,
        lit_model,
        args.cell_types,
        num_outputs=len(args.cell_types),
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
    transforms = build_transforms(GosaiDataset, sequence_first=sequence_first)

    for run_idx in range(args.runs):
        pearson = run_single_training_run(run_idx, args, transforms)
        results.loc[len(results)] = pearson.numpy()
        results.to_csv(args.result_dir, sep="\t", index=False)


if __name__ == "__main__":
    main()