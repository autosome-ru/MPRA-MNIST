"""
Arensbergen_model_launch.py

Runs a benchmark: for each genome id, trains a multi-output model (one
output per cell type) on the Arensbergen2019 dataset, repeating the whole
process `--runs` times. Sequences have variable length, so batches are
padded on the fly via `pad_collate`; models that require a fixed input
length (Malinois, DREAM-RNN) instead get their sequences padded/cropped to
a fixed length by the transforms.

Results (one row per run, one column per genome_id x cell_type combination)
are appended to a TSV file given by `--result_dir`.
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import lightning.pytorch as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics import PearsonCorrCoef

import mpramnist.transforms as t
from mpramnist.Arensbergen2019.dataset import ArensbergenDataset
from mpramnist.Arensbergen2019.trainer import LitModel_Arensbergen_Reg
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

# Models that require a fixed input length (get padded/cropped to FIXED_SEQ_LEN
# by the transforms). Every other model here consumes variable-length
# sequences, padded per-batch via `pad_collate`.
FIXED_LENGTH_MODELS = {"Malinois", "DREAM-RNN", "DREAM_RNN"}
FIXED_SEQ_LEN = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a multi-output model (one output per "
        "cell type) on the Arensbergen2019 dataset, per genome id.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general = parser.add_argument_group("general", "General run settings")
    general.add_argument(
        "--result_dir",
        type=str,
        default="./arensbergen.tsv",
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
        "--genome_ids",
        nargs="+",
        default=["SuRE42_HG02601", "SuRE43_GM18983", "SuRE44_HG01241", "SuRE45_HG03464"],
        help="List of genome ids; a separate model is trained per genome id.",
    )
    dataset_args.add_argument(
        "--cell_types",
        nargs="+",
        default=["K562", "HepG2"],
        help="Cell types to train on jointly; the model gets one output per cell type.",
    )

    trainer_args = parser.add_argument_group("trainer", "Training hyperparameters")
    trainer_args.add_argument("--lr", type=float, default=5e-3, help="Learning rate.")
    trainer_args.add_argument("--wd", type=float, default=1e-2, help="Weight decay (AdamW).")
    trainer_args.add_argument("--epoch_num", type=int, default=50, help="Max number of epochs.")

    args = parser.parse_args()

    if isinstance(args.genome_ids, str):
        args.genome_ids = [args.genome_ids]
    if isinstance(args.cell_types, str):
        args.cell_types = [args.cell_types]

    return args


# =============================================================================
# Collate function (variable-length sequences)
# =============================================================================

def make_pad_collate(channels_last: bool):
    """Build a collate function that pads variable-length sequences.

    `Seq2Tensor(sequence_first=True)` is used for every model in this file
    (not just Lyra), because `pad_sequence` pads along the first dimension
    of each tensor and that must be the length axis `L`, giving individual
    sequences shape `(L, C)`. After padding, the batch is `(B, L, C)`
    (channels-last).

    Only Lyra actually wants channels-last input. Every other model here
    (conv1d-based: MPRALegNet, MPRAnn, Malinois, PARM, DREAM-RNN,
    ReporterNet) expects channels-first `(B, C, L)`, so this collate
    transposes the padded batch back unless `channels_last` is True.
    """

    def collate(batch):
        seq, targets = zip(*batch)
        seq = pad_sequence(seq, batch_first=True, padding_value=0.25)  # (B, L, C)
        if not channels_last:
            seq = seq.transpose(1, 2).contiguous()  # (B, C, L)
        return seq, torch.vstack(targets)

    return collate

# =============================================================================
# Transforms
# =============================================================================

def build_transforms(dataset_cls: type[ArensbergenDataset], needs_fixed_length: bool) -> dict:
    """Build all the sequence transforms used for training/testing/inference.

    Sequences are always produced sequence-length-first (`sequence_first=True`)
    since `pad_sequence` in `pad_collate` pads along the first dimension of
    each tensor. Models that need a fixed input length (Malinois, DREAM-RNN)
    additionally get padded/cropped to `FIXED_SEQ_LEN`; all other models
    consume the raw variable-length sequences.

    Returns a dict with keys: "train", "test", "forward", "reverse".
    """
    if needs_fixed_length:
        train_transform = t.Compose(
            [
                t.Padding(FIXED_SEQ_LEN),
                t.LeftCrop(FIXED_SEQ_LEN, FIXED_SEQ_LEN),
                t.ReverseComplement(0.5),
                t.Seq2Tensor(sequence_first=True),
            ]
        )
        test_transform = t.Compose(
            [
                t.Padding(FIXED_SEQ_LEN),
                t.LeftCrop(FIXED_SEQ_LEN, FIXED_SEQ_LEN),
                t.Seq2Tensor(sequence_first=True),
            ]
        )
        forward_transform = t.Compose(
            [
                t.Padding(FIXED_SEQ_LEN),
                t.LeftCrop(FIXED_SEQ_LEN, FIXED_SEQ_LEN),
                t.Seq2Tensor(sequence_first=True),
            ]
        )
        reverse_transform = t.Compose(
            [
                t.Padding(FIXED_SEQ_LEN),
                t.LeftCrop(FIXED_SEQ_LEN, FIXED_SEQ_LEN),
                t.ReverseComplement(1),
                t.Seq2Tensor(sequence_first=True),
            ]
        )
    else:
        train_transform = t.Compose([t.ReverseComplement(0.5), t.Seq2Tensor(sequence_first=True)])
        test_transform = t.Compose([t.Seq2Tensor(sequence_first=True)])
        forward_transform = t.Compose([t.Seq2Tensor(sequence_first=True)])
        reverse_transform = t.Compose([t.ReverseComplement(1), t.Seq2Tensor(sequence_first=True)])

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
    `seq_len` is the fixed input length (only meaningful for the
    fixed-length models; ignored otherwise).
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

def evaluate_with_reverse_complement(
    forward_loader: DataLoader,
    reverse_loader: DataLoader,
    trainer: L.Trainer,
    lit_model: LitModel_Arensbergen_Reg,
    name: str,
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
    print(name, " Pearson correlation")
    print(pearson)
    print("===========")

    return pearson


# =============================================================================
# Single train + eval run for one genome id (all cell types trained jointly)
# =============================================================================

def run_single_genome(genome_id: str, args: argparse.Namespace, transforms: dict) -> list[float]:
    """Train a fresh multi-output model on one genome id (all requested cell
    types jointly) and return the per-cell-type test Pearson correlations.
    """

    # regression or classification
    task = "regression"

    # ---- Data ---------------------------------------------------------
    train_dataset = ArensbergenDataset(
        task=task,
        cell_type=args.cell_types,
        genome_id=genome_id,
        split="train",
        transform=transforms["train"],
        root=args.root,
    )
    val_dataset = ArensbergenDataset(
        task=task,
        cell_type=args.cell_types,
        genome_id=genome_id,
        split="val",
        transform=transforms["test"],
        root=args.root,
    )

    channels_last = args.model.lower() == "lyra"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=make_pad_collate(channels_last),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=make_pad_collate(channels_last),
    )

    # sequence is (L, C) since sequence_first=True is always used here.
    sample_seq = train_dataset[0][0]
    in_channels = len(sample_seq[0])

    # ---- Model ----------------------------------------------------------
    model, loss, lr_sheduler_to_use, learning_rate_reduce = build_model(
        model_name=args.model,
        in_channels=in_channels,
        seq_len=FIXED_SEQ_LEN,
        num_outputs=len(args.cell_types),
    )

    lit_model = LitModel_Arensbergen_Reg(
        model=model,
        cell_types=args.cell_types,
        loss=loss,
        weight_decay=args.wd,
        lr=args.lr,
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
    lit_model = LitModel_Arensbergen_Reg.load_from_checkpoint(
        best_model_path,
        model=model,
        cell_types=args.cell_types,
        loss=nn.MSELoss(),
        weight_decay=args.wd,
        lr=args.lr,
        print_each=1,
    )

    # ---- Test: average forward-strand and reverse-complement predictions --
    test_forward_dataset = ArensbergenDataset(
        task=task,
        cell_type=args.cell_types,
        genome_id=genome_id,
        split="test",
        transform=transforms["forward"],
        root=args.root,
    )
    test_reverse_dataset = ArensbergenDataset(
        task=task,
        cell_type=args.cell_types,
        genome_id=genome_id,
        split="test",
        transform=transforms["reverse"],
        root=args.root,
    )

    forward_loader = DataLoader(
        test_forward_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=make_pad_collate(channels_last),
    )
    reverse_loader = DataLoader(
        test_reverse_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=make_pad_collate(channels_last),
    )

    pearson = evaluate_with_reverse_complement(
        forward_loader,
        reverse_loader,
        trainer,
        lit_model,
        genome_id + str(args.cell_types),
        num_outputs=len(args.cell_types),
    ).numpy()

    if pearson.ndim == 0:
        pearson = [pearson.item()]
    else:
        pearson = list(pearson)

    return pearson


# =============================================================================
# Main
# =============================================================================

def build_result_columns(genome_ids: list[str], cell_types: list[str]) -> list[str]:
    return [genome_id + "_" + cell for genome_id in genome_ids for cell in cell_types]


def load_or_create_results(result_path: str, genome_ids: list[str], cell_types: list[str]) -> pd.DataFrame:
    if os.path.exists(result_path):
        return pd.read_csv(result_path, sep="\t")
    return pd.DataFrame(columns=build_result_columns(genome_ids, cell_types))


def main() -> None:
    args = parse_args()
    results = load_or_create_results(args.result_dir, args.genome_ids, args.cell_types)

    needs_fixed_length = args.model in FIXED_LENGTH_MODELS
    transforms = build_transforms(ArensbergenDataset, needs_fixed_length=needs_fixed_length)

    for run_idx in range(args.runs):
        print(f"\n### Run {run_idx + 1}/{args.runs} — model: {args.model} — cell types: {args.cell_types} ###")

        pears = []
        for genome_id in args.genome_ids:
            pears.extend(run_single_genome(genome_id, args, transforms))

        results.loc[len(results)] = pears
        results.to_csv(args.result_dir, sep="\t", index=False)


if __name__ == "__main__":
    main()