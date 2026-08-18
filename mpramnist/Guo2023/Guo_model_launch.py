"""
Guo_model_launch.py

Runs a benchmark: trains a multi-output model (one output per cell type) on
the Gosai2024 MPRA dataset, repeating the whole process `--runs` times. After
each run, the model is used (without further fine-tuning) to score variant
effects in the Guo2023 dataset: forward-strand and reverse-complement
predictions are averaged, the ref/alt effect size is computed, and its
Pearson correlation with the targets is reported per cell type (both over all
tested variants and restricted to significant daSNPs, fdr < 0.05).

Results for each run are written to a separate TSV file under
`--result_dir`.
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
from mpramnist.Guo2023.dataset import GuoMultiDataset
from mpramnist.Guo2023.trainer import LitModel_Guo
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
GOSAI_CROP_LEN = 600
GUO_VARIANT_LEN = 145

DEFAULT_CELL_TYPES_VARIANT = [
    "AST", "ES", "N-D2", "N-D4", "N-D10", "A-NPC", "D283", "D341",
    "IMR.diff", "IMR.prog", "SHSY5Y.diff", "SHSY5Y.prog", "HEK293T",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-output model on Gosai2024 and evaluate variant effects on Guo2023.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general = parser.add_argument_group("general", "General run settings")
    general.add_argument(
        "--result_dir",
        type=str,
        default="./guo.tsv",
        help="Directory where per-run variant-effect TSVs are written.",
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
        "--cell_types_train",
        nargs="+",
        default=["HepG2", "K562", "SKNSH"],
        help="List of cell types for training from the Gosai dataset.",
    )
    dataset_args.add_argument(
        "--cell_types_variant",
        nargs="+",
        default=DEFAULT_CELL_TYPES_VARIANT,
        help="List of cell types for the Guo2023 variant evaluation.",
    )

    trainer_args = parser.add_argument_group("trainer", "Training hyperparameters")
    trainer_args.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    trainer_args.add_argument("--wd", type=float, default=0.1, help="Weight decay (AdamW).")
    trainer_args.add_argument("--epoch_num", type=int, default=50, help="Max number of epochs.")

    args = parser.parse_args()

    if isinstance(args.cell_types_train, str):
        args.cell_types_train = [args.cell_types_train]

    if isinstance(args.cell_types_variant, str):
        args.cell_types_variant = [args.cell_types_variant]

    return args


# =============================================================================
# Transforms
# =============================================================================

def build_transforms(dataset_cls: type[GosaiDataset], sequence_first: bool = False) -> dict:
    """Build all the sequence transforms used for training/validation/inference.

    Training/validation use `GosaiDataset`'s own flanks. The final test-time
    transforms ("forward"/"reverse") also use Gosai's flanks — they're
    applied to *Guo* variant sequences, embedding them in the same context
    the model was trained on.

    Returns a dict with keys: "train", "val", "forward", "reverse".
    """
    left_flank = dataset_cls.LEFT_FLANK
    right_flank = dataset_cls.RIGHT_FLANK

    train_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.CenterCrop(GOSAI_CROP_LEN),
            t.ReverseComplement(0.5),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )
    val_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.CenterCrop(GOSAI_CROP_LEN),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )

    # Used at test time (on GuoMultiDataset) to average forward-strand and
    # reverse-complement predictions.
    forward_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.CenterCrop(GOSAI_CROP_LEN),
            t.Seq2Tensor(sequence_first=sequence_first),
        ]
    )
    reverse_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.CenterCrop(GOSAI_CROP_LEN),
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

def build_model(model_name: str, in_channels: int, seq_len: int, num_outputs: int):
    """Instantiate the requested model + loss function.

    `in_channels` is the number of one-hot encoded channels (nucleotides).
    `seq_len` is the length of the (cropped) input sequence.
    `num_outputs` is the number of cell types (one model output per cell type).

    Returns (model, loss, lr_scheduler_to_use, learning_rate_reduce).
    """
    loss = nn.MSELoss()
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
        model = MPRAnn(output_dim=num_outputs)

    elif model_name == "Malinois":
        model = BassetBranched(input_len=seq_len, n_outputs=num_outputs)
        loss = L1KLmixed()

    elif model_name == "PARM":
        model = PARM(n_block=5, type_loss="mse", output_dim=num_outputs)

    elif model_name in ("DREAM-RNN", "DREAM_RNN"):
        model = DREAM_RNN(in_channels=in_channels, seqsize=GOSAI_CROP_LEN, out_channels=num_outputs)

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

def evaluate_variant_effects(
    forward_preds: list[dict],
    reverse_preds: list[dict],
    cell_types: list[str],
) -> pd.DataFrame:
    """Combine forward-strand and reverse-complement ref/alt predictions and
    compute the Pearson correlation (per cell type) against the targets —
    both over all tested variants (excluding cell types with a NaN target)
    and restricted to significant daSNPs (fdr < 0.05).
    """
    targets = torch.cat([batch["target"] for batch in forward_preds])
    fdrs = torch.cat([batch["fdr"] for batch in forward_preds])

    forward_ref = torch.cat([batch["ref_predicted"] for batch in forward_preds])
    forward_alt = torch.cat([batch["alt_predicted"] for batch in forward_preds])

    reverse_ref = torch.cat([batch["ref_predicted"] for batch in reverse_preds])
    reverse_alt = torch.cat([batch["alt_predicted"] for batch in reverse_preds])

    mean_ref = torch.mean(torch.stack([forward_ref, reverse_ref]), dim=0)
    mean_alt = torch.mean(torch.stack([forward_alt, reverse_alt]), dim=0)

    variant_prediction = mean_alt - mean_ref

    pearson_metric = PearsonCorrCoef()
    results = []

    for i in range(len(cell_types)):
        all_mask = ~torch.isnan(targets[:, i].squeeze())  # keep only variants tested in this cell type
        pearsonr_all = pearson_metric(
            variant_prediction.squeeze()[all_mask], targets[:, i].squeeze()[all_mask]
        )

        significant_mask = (~torch.isnan(targets[:, i].squeeze())) & (fdrs[:, i].squeeze() < 0.05)  # keep only significant daSNPs
        pearsonr_daSNP = pearson_metric(
            variant_prediction.squeeze()[significant_mask], targets[:, i].squeeze()[significant_mask]
        )

        results.append(
            {
                "cell_type": cell_types[i],
                "n_all": all_mask.sum().item(),
                "pearsonr_all": pearsonr_all.item(),
                "n_daSNP": significant_mask.sum().item(),
                "pearsonr_daSNP": pearsonr_daSNP.item(),
            }
        )

    return pd.DataFrame(results)


# =============================================================================
# Single train + eval run
# =============================================================================

def run_single_training_run(run_idx: int, args: argparse.Namespace, transforms: dict) -> pd.DataFrame:
    """Train a fresh multi-output model on Gosai2024 and return the
    per-cell-type variant-effect Pearson correlations on Guo2023.
    """
    print(f"\n### Run {run_idx + 1}/{args.runs} — model: {args.model} — cell types: {args.cell_types_train} ###")

    std_err = [cell + "_lfcSE" for cell in args.cell_types_train]

    # ---- Data (Gosai2024, for training) --------------------------------
    train_dataset = GosaiDataset(
        split="train",
        transform=transforms["train"],
        filtration="own",
        cell_types=args.cell_types_train,
        stderr_columns=std_err,
        stderr_threshold=1.0,
        std_multiple_cut=6.0,
        up_cutoff_move=3.0,
        duplication_cutoff=0.5,
        root=args.root,
    )
    # Use the same filtration parameters for validation.
    val_dataset = GosaiDataset(
        split="val",
        transform=transforms["val"],
        filtration="own",
        cell_types=args.cell_types_train,
        stderr_columns=std_err,
        stderr_threshold=1.0,
        std_multiple_cut=6.0,
        up_cutoff_move=3.0,
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
        num_outputs=len(args.cell_types_train),
    )

    lit_model = LitModel_Guo(
        model=model,
        loss=loss,
        weight_decay=args.wd,
        lr=args.lr,
        print_each=1,
        lr_sheduler_to_use=lr_scheduler_to_use,
        learning_rate_reduce=learning_rate_reduce,
    )

    # ---- Trainer setup ----------------------------------------------------
    logger = pl_loggers.TensorBoardLogger(f"./{args.model}_logs", name="_".join(args.cell_types_train))

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

    # ---- Train ----------------------------------------------------------
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Reload the best checkpoint (by val_pearson) before evaluating.
    best_model_path = checkpoint_callback.best_model_path
    lit_model = LitModel_Guo.load_from_checkpoint(
        best_model_path,
        model=model,
        loss=nn.MSELoss(),
        weight_decay=args.wd,
        lr=args.lr,
        print_each=1,
    )

    # ---- Test: evaluate variant effects on Guo2023, averaging
    #      forward-strand and reverse-complement predictions ---------------
    predict_forward_dataset = GuoMultiDataset(
        split="test",
        length=GUO_VARIANT_LEN,
        cell_types=args.cell_types_variant,
        transform=transforms["forward"],
        root=args.root,
    )
    predict_reverse_dataset = GuoMultiDataset(
        split="test",
        length=GUO_VARIANT_LEN,
        cell_types=args.cell_types_variant,
        transform=transforms["reverse"],
        root=args.root,
    )

    forward_loader = DataLoader(
        predict_forward_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    reverse_loader = DataLoader(
        predict_forward_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    forward_preds = trainer.predict(lit_model, dataloaders=forward_loader)
    reverse_preds = trainer.predict(lit_model, dataloaders=reverse_loader)

    results = evaluate_variant_effects(forward_preds, reverse_preds, args.cell_types_variant)

    return results


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    # Lyra expects sequence-length-first tensors; every other model here
    # expects channels-first tensors.
    sequence_first = args.model.lower() == "lyra"
    transforms = build_transforms(GosaiDataset, sequence_first=sequence_first)

    for run_idx in range(args.runs):
        results = run_single_training_run(run_idx, args, transforms)
        output_file = f"{args.result_dir}/{args.model}_run{run_idx}.tsv"
        results.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    main()