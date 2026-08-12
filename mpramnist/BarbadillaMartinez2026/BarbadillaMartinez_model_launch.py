"""
BarbadillaMartinez_model_launch.py

Runs a benchmark: trains a multi-output model (one output per cell type) on
the BarbadillaMartinez2026 dataset, repeating the whole process `--runs`
times. Test-time evaluation aggregates predictions by the `FEAT` column
(multiple sequence variants belonging to the same feature are averaged)
before computing the Pearson correlation per cell type.

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
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

import mpramnist.transforms as t
from mpramnist.BarbadillaMartinez2026 import BarbadillaMartinezDataset, LitModel_BarbadillaMartinez
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

SEQ_LEN = 600  # fixed sequence length used by the padding/cropping transforms below


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a multi-output model (one output per "
        "cell type) on the BarbadillaMartinez2026 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general = parser.add_argument_group("general", "General run settings")
    general.add_argument(
        "--result_dir",
        type=str,
        default="./barbadillamartinez.tsv",
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
        "--genomes",
        type=str,
        default="all",
        help="Which genomes to use (applicable for genome-wide libraries only).",
    )
    dataset_args.add_argument(
        "--cell_types",
        nargs="+",
        default=["HepG2", "K562", "MCF7", "U2OS", "HCT116", "HEK293", "LNCaP"],
        help="Cell types to train on jointly; the model gets one output per cell type.",
    )

    trainer_args = parser.add_argument_group("trainer", "Training hyperparameters")
    trainer_args.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    trainer_args.add_argument("--wd", type=float, default=2e-1, help="Weight decay (AdamW).")
    trainer_args.add_argument("--epoch_num", type=int, default=50, help="Max number of epochs.")

    args = parser.parse_args()

    if isinstance(args.cell_types, str):
        args.cell_types = [args.cell_types]

    return args


# =============================================================================
# Transforms
# =============================================================================

def build_transforms(dataset_cls: type[BarbadillaMartinezDataset], sequence_first: bool = False) -> dict:
    """Build the sequence transforms used for training/testing.

    `sequence_first` controls the axis order produced by `Seq2Tensor`. Most
    models expect channels-first tensor (`sequence_first=False`), but some
    (e.g. Lyra) expect the sequence-length axis first, so they need
    `sequence_first=True`.

    Returns a dict with keys: "train", "test".
    """
    train_transform = t.Compose(
        [
            t.AddFlanks(dataset_cls.LEFT_FLANK, dataset_cls.RIGHT_FLANK),
            t.RandomPadding(SEQ_LEN),
            t.RightCrop(SEQ_LEN, SEQ_LEN),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )

    test_transform = t.Compose(
        [
            t.AddFlanks(dataset_cls.LEFT_FLANK, dataset_cls.RIGHT_FLANK),
            t.Padding(SEQ_LEN),
            t.RightCrop(SEQ_LEN, SEQ_LEN),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )

    return {"train": train_transform, "test": test_transform}


# =============================================================================
# Model factory
# =============================================================================

def build_model(model_name: str, in_channels: int, seq_len: int, num_outputs: int):
    """Instantiate the requested model + loss function.

    `in_channels` is the number of one-hot encoded channels (nucleotides).
    `seq_len` is the length of the (padded/cropped) input sequence.
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

def evaluate_by_feature_aggregation(
    trainer: L.Trainer,
    lit_model: LitModel_BarbadillaMartinez,
    test_dataset: BarbadillaMartinezDataset,
    args: argparse.Namespace,
) -> list[float]:
    """Predict on the test set, average predictions belonging to the same
    `FEAT` group (multiple sequence variants of the same feature), and
    compute the Pearson correlation per cell type against the averaged
    ground truth.
    """
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    predictions = trainer.predict(lit_model, dataloaders=test_loader)
    predicted = torch.cat([batch["predicted"] for batch in predictions]).numpy()

    target_columns = test_dataset.target_columns
    real = test_dataset._data[target_columns + ["FEAT"]].copy()

    pred_columns = []
    for i, target_col in enumerate(target_columns):
        pred_col = f"{target_col}_pred"
        real[pred_col] = predicted[:, i]
        pred_columns.append(pred_col)

    aggregated = real.groupby("FEAT")[pred_columns + target_columns].mean()

    pearsons = []
    for target_col, pred_col in zip(target_columns, pred_columns):
        corr, _ = pearsonr(aggregated[target_col], aggregated[pred_col])
        pearsons.append(float(corr))

    return pearsons


# =============================================================================
# Single train + eval run (all cell types trained jointly)
# =============================================================================

def run_single_training_run(run_idx: int, args: argparse.Namespace, transforms: dict) -> list[float]:
    """Train a fresh multi-output model on all requested cell types and
    return the per-cell-type test Pearson correlations.
    """
    print(f"\n### Run {run_idx + 1}/{args.runs} — model: {args.model} — cell types: {args.cell_types} ###")

    # ---- Data ---------------------------------------------------------
    train_dataset = BarbadillaMartinezDataset(
        split=[0, 1, 2, 3],
        transform=transforms["train"],
        genomes=args.genomes,
        cell_type=args.cell_types,
        root=args.root,
    )
    val_dataset = BarbadillaMartinezDataset(
        split=[4],
        transform=transforms["test"],
        genomes=args.genomes,
        cell_type=args.cell_types,
        root=args.root,
    )
    test_dataset = BarbadillaMartinezDataset(
        split="test",
        transform=transforms["test"],
        genomes=args.genomes,
        cell_type=args.cell_types,
        root=args.root,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
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

    lit_model = LitModel_BarbadillaMartinez(
        model=model,
        cell_types=args.cell_types,
        loss=loss,
        weight_decay=args.wd,
        lr=args.lr,
        print_each=1,
        lr_sheduler_to_use=lr_sheduler_to_use,
        learning_rate_reduce=learning_rate_reduce
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
    lit_model = LitModel_BarbadillaMartinez.load_from_checkpoint(
        best_model_path,
        model=model,
        cell_types=args.cell_types,
        loss=loss,
        weight_decay=args.wd,
        lr=args.lr,
        print_each=1,
    )

    # ---- Test: aggregate predictions by FEAT, then compute Pearson --------
    pearsons = evaluate_by_feature_aggregation(trainer, lit_model, test_dataset, args)

    print("===========")
    print(args.cell_types, " Pearson correlation")
    print(pearsons)
    print("===========")

    return pearsons


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
    transforms = build_transforms(BarbadillaMartinezDataset, sequence_first=sequence_first)

    for run_idx in range(args.runs):
        pearsons = run_single_training_run(run_idx, args, transforms)
        results.loc[len(results)] = pearsons
        results.to_csv(args.result_dir, sep="\t", index=False)


if __name__ == "__main__":
    main()