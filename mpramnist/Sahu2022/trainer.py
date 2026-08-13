"""
LitModel_Sahu.py

Lightning modules for the Sahu2022 benchmark.

- `LitModel_Sahu` is the base binary-classification module used by every
  non-"Binary" Sahu task: a single sequence goes in, a single logit comes
  out, and the module tracks the standard binary-classification metrics
  (accuracy, AUROC, AUPR, precision, recall, F1) for val/test.

- `LitModel_Sahu_binary_*` classes handle the "Binary" (paired) task: two
  sequences ("seq_enh" and "seq") are each embedded by the same backbone
  `self.model` (which has its head stripped off upstream), concatenated,
  and passed through a small architecture-specific `head` to produce a
  single logit. Each subclass only needs to define that head and how to
  reach it — everything else (training/validation/test loops, metric
  bookkeeping, logging) is inherited unchanged from `LitModel_Sahu_binary_base`,
  which is why `_process_batch` is the sole extension point.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping
from torchmetrics import Accuracy, AUROC, AveragePrecision, Precision, Recall, F1Score


# =============================================================================
# Metric bookkeeping helpers
# =============================================================================

# The six metrics tracked for both "val" and "test". Order here also
# controls the order they're printed in.
_METRIC_NAMES = ("acc", "auroc", "aupr", "precision", "recall", "f1")


def _make_binary_metrics() -> dict:
    """Build one fresh instance of each tracked metric (binary-classification)."""
    return {
        "acc": Accuracy(task="binary"),
        "auroc": AUROC(task="binary"),
        "aupr": AveragePrecision(task="binary"),
        "precision": Precision(task="binary"),
        "recall": Recall(task="binary"),
        "f1": F1Score(task="binary"),
    }


# =============================================================================
# Base module: single-sequence binary classification
# =============================================================================

class LitModel_Sahu(L.LightningModule):
    """Binary-classification Lightning module for the standard (non-"Binary")
    Sahu2022 tasks: one sequence in, one logit out.
    """

    def __init__(
        self,
        model,
        loss=nn.BCEWithLogitsLoss(),
        print_each: int = 1,
        weight_decay: float = 1e-2,
        lr: float = 3e-4,
        # Early stopping
        early_stopping_patience: int = 10,
        early_stopping_metric: str = "val_loss",
        early_stopping_mode: str = "min",
    ):
        super().__init__()

        self.model = model
        self.loss = loss
        self.print_each = print_each
        self.weight_decay = weight_decay
        self.lr = lr

        # Early stopping config — consumed by configure_callbacks()
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode

        for name, metric in _make_binary_metrics().items():
            setattr(self, f"val_{name}", metric)
        for name, metric in _make_binary_metrics().items():
            setattr(self, f"test_{name}", metric)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def configure_callbacks(self):
        return [
            EarlyStopping(
                monitor=self.early_stopping_metric,
                patience=self.early_stopping_patience,
                mode=self.early_stopping_mode,
                verbose=True,
            )
        ]

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Note: unlike the Agarwal trainers (which offer a ReduceLROnPlateau
        # toggle), the Sahu classification tasks are trained with a fixed
        # one-cycle schedule tied to the known step count.
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            three_phase=False,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            cycle_momentum=False,
        )
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "cycle_lr",
        }

        return [optimizer], [lr_scheduler_config]

    # ------------------------------------------------------------------
    # Metric helpers (operate on self.val_* / self.test_* attributes by name)
    # ------------------------------------------------------------------

    def _update_metrics(self, prefix: str, y_hat, y):
        for name in _METRIC_NAMES:
            metric = getattr(self, f"{prefix}_{name}")
            # AveragePrecision expects integer targets; every other metric
            # here is happy with the float targets used elsewhere.
            metric(y_hat, y.long() if name == "aupr" else y)

    def _compute_metrics(self, prefix: str) -> dict:
        return {name: getattr(self, f"{prefix}_{name}").compute() for name in _METRIC_NAMES}

    def _reset_metrics(self, prefix: str):
        for name in _METRIC_NAMES:
            getattr(self, f"{prefix}_{name}").reset()

    def _log_and_print_epoch_metrics(self, prefix: str, label: str, include_epoch: bool, always_print: bool):
        metrics = self._compute_metrics(prefix)

        self.log(f"{prefix}_aupr", metrics["aupr"], on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_auroc", metrics["auroc"], on_epoch=True, prog_bar=True)

        should_print = always_print or (self.current_epoch + 1) % self.print_each == 0
        if should_print:
            epoch_part = f"| Epoch: {self.current_epoch} " if include_epoch else ""
            line1 = (
                f"{epoch_part}"
                f"| {label} Acc: {metrics['acc']} "
                f"| {label} AUROC: {metrics['auroc']} "
                f"| {label} AUPR: {metrics['aupr']} |"
            )
            line2 = (
                f"| {label} Precision: {metrics['precision']} "
                f"| {label} Recall: {metrics['recall']} "
                f"| {label} F1: {metrics['f1']} "
            )
            border = "-" * max(len(line1), len(line2))
            print("\n".join(["", border, line1, line2, border, ""]))

        self._reset_metrics(prefix)

    # ------------------------------------------------------------------
    # Forward / batch handling
    # ------------------------------------------------------------------

    def forward(self, x):
        return self.model(x)

    def _process_batch(self, batch):
        """Extension point: turn a raw batch into (predicted_logits, targets).

        Overridden by the "Binary" (paired) subclasses below, which embed
        two sequences and combine them through an architecture-specific
        head instead of calling `self.model` on a single input.
        """
        x, y = batch
        return self.model(x), y

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_nb):
        y_hat, y = self._process_batch(batch)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self._process_batch(batch)
        loss = self.loss(y_hat, y)
        self._update_metrics("val", y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self._log_and_print_epoch_metrics("val", "Val", include_epoch=True, always_print=False)

    def test_step(self, batch, batch_idx):
        y_hat, y = self._process_batch(batch)
        loss = self.loss(y_hat, y)
        self._update_metrics("test", y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        self._log_and_print_epoch_metrics("test", "Test", include_epoch=False, always_print=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat, y = self._process_batch(batch)
        return {
            "predicted": y_hat.squeeze().cpu().detach(),
            "target": y.cpu().detach().float(),
        }


# =============================================================================
# "Binary" (paired) task modules
# =============================================================================
#
# Shared shape: batches look like ({"seq_enh": ..., "seq": ...}, labels).
# Both sequences are embedded with the same backbone (`self.model`, whose
# head has already been stripped off by the caller), concatenated, and
# passed through a small architecture-specific `head` to produce one logit.
#
# `LitModel_Sahu_binary_base` owns all of that shared machinery
# (`_process_batch`); every concrete architecture below only implements
# `_build_head()`, which returns the one nn.Module specific to it.

class LitModel_Sahu_binary_base(LitModel_Sahu):
    """Shared paired-task machinery for the "Binary" Sahu task.

    Subclasses must implement `_build_head()`, returning the nn.Module that
    maps a concatenated (enhancer_embedding, promoter_embedding) pair to a
    single logit.
    """

    def __init__(
        self,
        model,
        loss=nn.BCEWithLogitsLoss(),
        print_each: int = 1,
        weight_decay: float = 1e-2,
        lr: float = 3e-4,
        early_stopping_patience: int = 10,
        early_stopping_metric: str = "val_loss",
        early_stopping_mode: str = "min",
    ):
        super().__init__(
            model=model,
            loss=loss,
            print_each=print_each,
            weight_decay=weight_decay,
            lr=lr,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            early_stopping_mode=early_stopping_mode,
        )
        self.head = self._build_head()

    def _build_head(self) -> nn.Module:
        raise NotImplementedError

    def _process_batch(self, batch):
        seqs, labels = batch
        enhancer = self.model(seqs["seq_enh"])
        promoter = self.model(seqs["seq"])
        embedding = torch.cat([enhancer, promoter], dim=1)
        y_hat = self.head(embedding).squeeze(-1)
        return y_hat, labels


class LitModel_Sahu_binary_legnet(LitModel_Sahu_binary_base):
    """Paired-task head for the MPRALegNet backbone."""

    def _build_head(self) -> nn.Module:
        embedding_dim = 256  # HumanLegNetBinary's embedding size
        return nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, 1),
        )


class LitModel_Sahu_binary_mprann(LitModel_Sahu_binary_base):
    """Paired-task head for the MPRAnn backbone."""

    def _build_head(self) -> nn.Module:
        embedding_dim = 300  # MPRAnnBinary's embedding size
        return nn.Sequential(
            nn.Linear(embedding_dim * 2, 1),
            nn.Sigmoid(),
        )


class GroupedLinear(nn.Module):
    """`groups` independent Linear(in_group_size -> out_group_size) layers,
    applied in parallel via batched matmul instead of a Python loop.
    """

    def __init__(self, in_group_size: int, out_group_size: int, groups: int):
        super().__init__()

        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.groups = groups

        self.weight = nn.Parameter(torch.zeros(groups, in_group_size, out_group_size))
        self.bias = nn.Parameter(torch.zeros(groups, 1, out_group_size))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(3))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # (N, groups * in_group_size) -> (groups, N, in_group_size)
        reorg = x.permute(1, 0).reshape(self.groups, self.in_group_size, -1).permute(0, 2, 1)
        out = torch.bmm(reorg, self.weight) + self.bias
        # (groups, N, out_group_size) -> (N, groups * out_group_size)
        return out.permute(0, 2, 1).reshape(self.out_group_size * self.groups, -1).permute(1, 0)


class LitModel_Sahu_binary_malinois(LitModel_Sahu_binary_base):
    """Paired-task head for the Malinois (BassetBranched) backbone."""

    def _build_head(self) -> nn.Module:
        branched_channels = 140
        n_outputs = 1
        return GroupedLinear(branched_channels * 2, 1, n_outputs)


class LitModel_Sahu_binary_parm(LitModel_Sahu_binary_base):
    """Paired-task head for the PARM backbone."""

    def _build_head(self) -> nn.Module:
        filter_size = 125
        return nn.Linear(filter_size * 2, 1)


class LitModel_Sahu_binary_dream_rnn(LitModel_Sahu_binary_base):
    """Paired-task head for the DREAM-RNN backbone."""

    def _build_head(self) -> nn.Module:
        embedding_dim = 256  # AutosomeFinalLayersBlockBinary's embedding size
        return nn.Linear(embedding_dim * 2, 1)