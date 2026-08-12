"""
Fromel_model_launch.py

Runs a benchmark: trains a multi-output model (one output per activity
target) on the Fromel2025 dataset, repeating the whole process `--runs`
times. Test-time evaluation averages forward-strand and reverse-complement
predictions and computes a masked Pearson correlation per target (some
targets are missing/masked for a given sequence).

For `--cell_types K562` there's a single test split ("test") and a single
output file. For `--cell_types HSPC` (or any other cell type), results are
tracked and written separately per entry in `--test_type` (each to its own
TSV file, matching the original script's file-naming convention).
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

import mpramnist.transforms as t
from mpramnist.Fromel2025 import FromelDataset, LitModel_Fromel, MaskedMSE, MaskedPearsonCorrCoef
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

TRAIN_CROP_LEN = 245  # fixed insert length used by the padding/cropping transforms below


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a multi-output model (one output per "
        "activity target) on the Fromel2025 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general = parser.add_argument_group("general", "General run settings")
    general.add_argument(
        "--result_dir",
        type=str,
        default="./fromel.tsv",
        help="Base path for the TSV file(s) where per-run Pearson correlations are appended "
        "(a per-test_type suffix is added — see module docstring).",
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
        "--cell_types", type=str, default="HSPC", help="Cell type used (e.g. 'HSPC' or 'K562')."
    )
    dataset_args.add_argument(
        "--targets",
        nargs="+",
        default=["State_1M", "State_2D", "State_3E", "State_4M", "State_5M", "State_6N", "State_7M"],
        help="Activity targets to train on jointly; the model gets one output per target.",
    )
    dataset_args.add_argument(
        "--test_type",
        nargs="+",
        default=["synthetic", "test", "genome", "generated"],
        help="List of test subdatasets to evaluate on (ignored for --cell_types K562).",
    )

    trainer_args = parser.add_argument_group("trainer", "Training hyperparameters")
    trainer_args.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    trainer_args.add_argument("--wd", type=float, default=2e-1, help="Weight decay (AdamW).")
    trainer_args.add_argument("--epoch_num", type=int, default=50, help="Max number of epochs.")

    args = parser.parse_args()

    # K562 only has one target/test split; this mirrors the original script.
    if args.cell_types == "K562":
        args.targets = ["State_9K"]
        args.test_type = ["test"]

    return args


# =============================================================================
# Transforms
# =============================================================================

def build_transforms(dataset_cls: type[FromelDataset], sequence_first: bool = False) -> dict:
    """Build all the sequence transforms used for training/validation/inference.

    `sequence_first` controls the axis order produced by `Seq2Tensor`. Most
        models expect channels-first tensor (`sequence_first=False`), but some
        (e.g. Lyra) expect the sequence-length axis first, so they need
        `sequence_first=True`. `AddFeatureChannels` runs before `Seq2Tensor`
        (it operates on the pre-tensor representation), so it's unaffected by
        this axis choice.

    Returns a dict with keys: "train", "val", "forward", "reverse".
    """
    train_transform = t.Compose(
        [
            t.AddFlanks(dataset_cls.CONSTANT_LEFT_FLANK, dataset_cls.CONSTANT_RIGHT_FLANK),
            t.AddFlanks("", dataset_cls.RIGHT_FLANK),  # matches original human_legnet preprocessing
            t.RightCrop(TRAIN_CROP_LEN, 270),  # shift augmentation
            t.LeftCrop(TRAIN_CROP_LEN, TRAIN_CROP_LEN),
            t.ReverseComplement(0.5),
            t.AddFeatureChannels(["batch"]),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )

    val_transform = t.Compose(
        [
            t.AddFlanks(dataset_cls.CONSTANT_LEFT_FLANK, dataset_cls.CONSTANT_RIGHT_FLANK),
            t.LeftCrop(TRAIN_CROP_LEN, TRAIN_CROP_LEN),
            t.ReverseComplement(0),
            t.AddFeatureChannels(["batch"]),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )

    # Used at test time to average forward-strand and reverse-complement predictions.
    forward_transform = t.Compose(
        [
            t.AddFlanks(dataset_cls.CONSTANT_LEFT_FLANK, dataset_cls.CONSTANT_RIGHT_FLANK),
            t.LeftCrop(TRAIN_CROP_LEN, TRAIN_CROP_LEN),
            t.ReverseComplement(0),
            t.AddFeatureChannels(["batch"]),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )

    reverse_transform = t.Compose(
        [
            t.AddFlanks(dataset_cls.CONSTANT_LEFT_FLANK, dataset_cls.CONSTANT_RIGHT_FLANK),
            t.LeftCrop(TRAIN_CROP_LEN, TRAIN_CROP_LEN),
            t.ReverseComplement(1),
            t.AddFeatureChannels(["batch"]),
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

def build_model(model_name: str, in_channels: int, seq_len: int, num_outputs: int):
    """Instantiate the requested model + loss function.

    `in_channels` is dynamic here (not a fixed 4), because
    `AddFeatureChannels(['batch'])` appends extra channels beyond the
    one-hot nucleotide encoding.
    `seq_len` is the length of the (cropped) input sequence.
    `num_outputs` is the number of activity targets (one model output per
    target).

    Returns (model, loss, lr_scheduler_to_use, learning_rate_reduce).
    """
    loss = MaskedMSE()
    learning_rate_reduce = True
    lr_scheduler_to_use = "one_cycle"

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
        model = MPRAnn(in_channels=in_channels, output_dim=num_outputs)

    elif model_name == "Malinois":
        model = BassetBranched(input_len=seq_len, n_channels=in_channels, n_outputs=num_outputs)

    elif model_name == "PARM":
        model = PARM(n_block=5, type_loss="mse", output_dim=num_outputs, vocab=in_channels)

    elif model_name in ("DREAM-RNN", "DREAM_RNN"):
        model = DREAM_RNN(in_channels=in_channels, seqsize=seq_len, out_channels=num_outputs)
        loss = nn.MSELoss()  # matches the original script — DREAM-RNN uses plain MSE, not MaskedMSE

    elif model_name.lower() == "lyra":
        model = Lyra(d_input=in_channels, d_output=num_outputs, d_model=512, dropout=0.1)
        lr_scheduler_to_use = "reducelronplateau"

    elif model_name == "ReporterNet":
        model = ReporterNet(dropout_rate=0.2, output_dim=num_outputs)
        model.apply(initialize_weights_reporternet)
        learning_rate_reduce = None

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, loss, lr_scheduler_to_use, learning_rate_reduce


# =============================================================================
# Evaluation helpers
# =============================================================================

def evaluate_with_reverse_complement(
    forward_loader: DataLoader,
    reverse_loader: DataLoader,
    trainer: L.Trainer,
    lit_model: LitModel_Fromel,
    name: str,
    num_outputs: int,
) -> torch.Tensor:
    """Predict on forward-strand and reverse-complement inputs separately,
    average the two predictions, and compute the masked Pearson correlation
    (per target) against the (shared) targets.
    """
    forward_preds = trainer.predict(lit_model, dataloaders=forward_loader)
    targets = torch.cat([batch["target"] for batch in forward_preds])
    forward_pred_values = torch.cat([batch["predicted"] for batch in forward_preds])

    reverse_preds = trainer.predict(lit_model, dataloaders=reverse_loader)
    reverse_pred_values = torch.cat([batch["predicted"] for batch in reverse_preds])

    mean_pred = torch.mean(torch.stack([forward_pred_values, reverse_pred_values]), dim=0)

    pearson_metric = MaskedPearsonCorrCoef(num_outputs=num_outputs)
    pearson = pearson_metric(mean_pred, targets)

    print("===========")
    print(name, " Pearson correlation")
    print(pearson)
    print("===========")

    return pearson


# =============================================================================
# Single train + eval run (all targets trained jointly)
# =============================================================================

def run_single_training_run(run_idx: int, args: argparse.Namespace, transforms: dict) -> dict[str, torch.Tensor]:
    """Train a fresh multi-output model on all requested targets and return
    the per-target test Pearson correlations, keyed by test_type.
    """
    print(f"\n### Run {run_idx + 1}/{args.runs} — model: {args.model} — cell type: {args.cell_types} "
          f"— targets: {args.targets} ###")

    # ---- Data ---------------------------------------------------------
    train_dataset = FromelDataset(
        cell_type=args.cell_types,
        targets=args.targets,
        split="train",
        transform=transforms["train"],
        root=args.root,
    )
    val_dataset = FromelDataset(
        cell_type=args.cell_types,
        targets=args.targets,
        split="val",
        transform=transforms["val"],
        root=args.root,
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
        model_name=args.model,
        in_channels=in_channels,
        seq_len=seq_len,
        num_outputs=len(args.targets),
    )

    lit_model = LitModel_Fromel(
        model=model,
        loss=loss,
        weight_decay=args.wd,
        lr=args.lr,
        activity_columns=args.targets,
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
    lit_model = LitModel_Fromel.load_from_checkpoint(
        best_model_path,
        model=model,
        loss=loss,
        weight_decay=args.wd,
        lr=args.lr,
        activity_columns=args.targets,
        print_each=1,
    )

    # ---- Test: per test_type, average forward-strand and reverse-complement
    #      predictions, then compute masked Pearson correlation ------------
    pearsons_by_test_type = {}
    for test_type in args.test_type:
        test_forward_dataset = FromelDataset(
            cell_type=args.cell_types,
            targets=args.targets,
            split=test_type,
            transform=transforms["forward"],
            root=args.root,
        )
        test_reverse_dataset = FromelDataset(
            cell_type=args.cell_types,
            targets=args.targets,
            split=test_type,
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

        pearsons_by_test_type[test_type] = evaluate_with_reverse_complement(
            forward_loader,
            reverse_loader,
            trainer,
            lit_model,
            args.cell_types,
            num_outputs=len(args.targets),
        )

    return pearsons_by_test_type


# =============================================================================
# Main
# =============================================================================

def result_path_for(result_dir: str, cell_types: str, test_type: str) -> str:
    base = result_dir.split(".tsv")[0]
    if cell_types == "K562":
        return f"{base}_K562_synthetic.tsv"
    return f"{base}_HSPC_{test_type}.tsv"


def load_or_create_result_dfs(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    result_dfs = {}
    for test_type in args.test_type:
        path = result_path_for(args.result_dir, args.cell_types, test_type)
        result_dfs[test_type] = pd.read_csv(path, sep="\t") if os.path.exists(path) else pd.DataFrame(
            columns=args.targets
        )
    return result_dfs


def main() -> None:
    args = parse_args()
    result_dfs = load_or_create_result_dfs(args)

    # Lyra expects sequence-length-first tensors; every other model here
    # expects channels-first tensors.
    sequence_first = args.model.lower() == "lyra"
    transforms = build_transforms(FromelDataset, sequence_first=sequence_first)

    for run_idx in range(args.runs):
        pearsons_by_test_type = run_single_training_run(run_idx, args, transforms)

        for test_type, pearson in pearsons_by_test_type.items():
            result_dfs[test_type].loc[len(result_dfs[test_type])] = pearson.numpy()
            path = result_path_for(args.result_dir, args.cell_types, test_type)
            result_dfs[test_type].to_csv(path, sep="\t", index=False)


if __name__ == "__main__":
    main()