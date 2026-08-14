import torch
import torch.nn as nn
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping
from torchmetrics import PearsonCorrCoef

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import lightning.pytorch as L

from torchmetrics.classification import (
    Accuracy,
    AUROC,
    AveragePrecision,
    F1Score,
    Precision,
    Recall,
)

from sklearn.metrics import auc, roc_curve
    
class LitModel_Reddy_Reg(L.LightningModule):
    def __init__(
        self,
        model,
        loss=nn.MSELoss(),
        print_each: int = 1,
        weight_decay: float = 1e-2,
        lr: float = 3e-4,
        cell_types: list[str] = ["JURKAT", "K562", "THP1"],
        lr_sheduler_to_use = "one_cycle", # or reducelronplateau
        # Early stopping
        early_stopping_patience: int = 10,
        early_stopping_metric: str = "val_loss",
        early_stopping_mode: str = "min",
        # LR scheduling
        learning_rate_reduce: bool = True,
        learning_rate_reduce_patience: int = 5,
        learning_rate_reduce_metric: str = "val_loss",
        learning_rate_reduce_mode: str = "min",
    ):
        super().__init__()

        self.model = model
        self.loss = loss
        self.print_each = print_each
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_sheduler_to_use = lr_sheduler_to_use

        # Early stopping config — consumed by configure_callbacks()
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode

        # ReduceLROnPlateau config — consumed by configure_optimizers()
        self.learning_rate_reduce = learning_rate_reduce
        self.learning_rate_reduce_patience = learning_rate_reduce_patience
        self.learning_rate_reduce_metric = learning_rate_reduce_metric
        self.learning_rate_reduce_mode = learning_rate_reduce_mode

        if isinstance(cell_types, str):
            cell_types = [cell_types]

        self.cell_types = cell_types
        self.num_outputs = len(cell_types)

        self.train_pearson = PearsonCorrCoef(num_outputs=self.num_outputs)
        self.val_pearson = PearsonCorrCoef(num_outputs=self.num_outputs)
        self.test_pearson = PearsonCorrCoef(num_outputs=self.num_outputs)

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

    def forward(self, x):
        return self.model(x)

    def labels_and_predicted_unsqueeze(self, pred, targets):
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)       # [N] -> [N, 1]
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1) # [N] -> [N, 1]
        return pred, targets

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
    
            if not self.learning_rate_reduce:
                return optimizer
    
            if self.lr_sheduler_to_use == "one_cycle":
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
            elif self.lr_sheduler_to_use == "reducelronplateau":
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=self.learning_rate_reduce_mode,
                    patience=self.learning_rate_reduce_patience,
                )
                lr_scheduler_config = {
                    "scheduler": lr_scheduler,
                    "monitor": self.learning_rate_reduce_metric,
                    "interval": "epoch",
                    "frequency": 1,
                    "name": "reduce_lr_on_plateau",
                }
    
            return [optimizer], [lr_scheduler_config]

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_nb):
        X, y = batch
        y_hat = self.forward(X)
        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y)
        loss = self.loss(y_hat, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.train_pearson.update(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y)
        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.val_pearson.update(y_hat, y)

    def on_validation_epoch_end(self):
        val_str = ""
        train_str = ""

        train_pearson = self.train_pearson.compute()
        val_pearson = self.val_pearson.compute()

        for i in range(self.num_outputs):
            tr_pearson = train_pearson[i] if self.num_outputs > 1 else train_pearson
            v_pearson = val_pearson[i] if self.num_outputs > 1 else val_pearson

            self.log(
                f"train_{self.cell_types[i]}_pearson",
                tr_pearson,
                prog_bar=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"val_{self.cell_types[i]}_pearson",
                v_pearson,
                prog_bar=True,
                on_epoch=True,
                logger=True,
            )

            val_str += f"| Val Pearson {self.cell_types[i]}: {v_pearson:.5f} "
            train_str += f"| Train Pearson {self.cell_types[i]}: {tr_pearson:.5f} "

        mean_val_pearson = val_pearson.mean()
        mean_train_pearson = train_pearson.mean()

        self.log("val_pearson", mean_val_pearson, prog_bar=True, on_epoch=True, logger=True)

        self.train_pearson.reset()
        self.val_pearson.reset()

        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Loss: {self.trainer.callback_metrics['val_loss']:.5f} "

            if self.num_outputs > 1:
                val_str += f"| Mean Val Pearson: {mean_val_pearson:.5f} "
                train_str += f"| Mean Train Pearson: {mean_train_pearson:.5f} "

            border = "-" * max(len(res_str), len(val_str), len(train_str))
            print(
                "\n".join(
                    ["", border, res_str, val_str + "|", train_str + "|", border, ""]
                )
            )

    def test_step(self, batch, _):
        x, y = batch
        y_hat = self.forward(x)
        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y)
        loss = self.loss(y_hat, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.test_pearson.update(y_hat, y)

    def on_test_epoch_end(self):
        test_pearson = self.test_pearson.compute()

        for i in range(self.num_outputs):
            te_pearson = test_pearson[i] if self.num_outputs > 1 else test_pearson
            self.log(f"test_{self.cell_types[i]}_pearson", te_pearson, prog_bar=True)

        self.test_pearson.reset()

    def predict_step(self, batch, _):
        x, y = batch
        pred = self.forward(x)
        pred, y = self.labels_and_predicted_unsqueeze(pred, y)
        return {
            "predicted": pred.cpu().detach().float(),
            "target": y.cpu().detach().float(),
        }


class LitModel_Reddy_Clas(L.LightningModule):
    def __init__(
        self,
        model,
        loss=nn.BCEWithLogitsLoss(),
        print_each: int = 1,
        weight_decay: float = 1e-2,
        lr: float = 3e-4,
        n_labels: int = 3,
        show_figure: bool = True,
    ):
        super().__init__()

        self.model = model
        self.loss = loss

        self.print_each = print_each
        self.weight_decay = weight_decay
        self.lr = lr

        self.n_labels = n_labels
        self.show_figure = show_figure

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------

        self.val_acc = Accuracy(
            task="multilabel",
            num_labels=n_labels,
        )
        self.val_auroc = AUROC(
            task="multilabel",
            num_labels=n_labels,
        )
        self.val_aupr = AveragePrecision(
            task="multilabel",
            num_labels=n_labels,
        )
        self.val_precision = Precision(
            task="multilabel",
            num_labels=n_labels,
            average="macro",
        )
        self.val_recall = Recall(
            task="multilabel",
            num_labels=n_labels,
            average="macro",
        )
        self.val_f1 = F1Score(
            task="multilabel",
            num_labels=n_labels,
            average="macro",
        )

        self.test_acc = Accuracy(
            task="multilabel",
            num_labels=n_labels,
        )
        self.test_auroc = AUROC(
            task="multilabel",
            num_labels=n_labels,
        )
        self.test_aupr = AveragePrecision(
            task="multilabel",
            num_labels=n_labels,
        )
        self.test_precision = Precision(
            task="multilabel",
            num_labels=n_labels,
            average="macro",
        )
        self.test_recall = Recall(
            task="multilabel",
            num_labels=n_labels,
            average="macro",
        )
        self.test_f1 = F1Score(
            task="multilabel",
            num_labels=n_labels,
            average="macro",
        )

        # ------------------------------------------------------------------
        # Predictions for test-time plots
        # ------------------------------------------------------------------

        self.test_scores = []
        self.test_targets = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_metrics(self, y_hat, y, metrics_prefix: str):
        """
        Update all classification metrics for validation or test.
        """
        y_metrics = y.long()

        getattr(self, f"{metrics_prefix}_acc").update(y_hat, y_metrics)
        getattr(self, f"{metrics_prefix}_auroc").update(y_hat, y_metrics)
        getattr(self, f"{metrics_prefix}_aupr").update(y_hat, y_metrics)
        getattr(self, f"{metrics_prefix}_precision").update(y_hat, y_metrics)
        getattr(self, f"{metrics_prefix}_recall").update(y_hat, y_metrics)
        getattr(self, f"{metrics_prefix}_f1").update(y_hat, y_metrics)

    def _log_metrics(self, metrics_prefix: str, prog_bar: bool = False):
        """
        Compute and log all classification metrics.
        """
        metric_names = [
            "acc",
            "auroc",
            "aupr",
            "precision",
            "recall",
            "f1",
        ]

        values = {}

        for metric_name in metric_names:
            metric = getattr(self, f"{metrics_prefix}_{metric_name}")
            value = metric.compute()

            self.log(
                f"{metrics_prefix}_{metric_name}",
                value,
                prog_bar=prog_bar,
                on_epoch=True,
                logger=True,
            )

            values[metric_name] = value

        return values

    def _reset_metrics(self, metrics_prefix: str):
        """
        Reset all classification metrics.
        """
        metric_names = [
            "acc",
            "auroc",
            "aupr",
            "precision",
            "recall",
            "f1",
        ]

        for metric_name in metric_names:
            getattr(self, f"{metrics_prefix}_{metric_name}").reset()

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

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
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)
        y = y.float()

        loss = self.loss(y_hat, y)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)
        y = y.float()

        loss = self.loss(y_hat, y)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        self._update_metrics(
            y_hat,
            y,
            metrics_prefix="val",
        )

    def on_validation_epoch_end(self):
        metrics = self._log_metrics(
            metrics_prefix="val",
            prog_bar=True,
        )

        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Acc: {metrics['acc']:.5f} "
            res_str += f"| Val AUROC: {metrics['auroc']:.5f} "
            res_str += f"| Val AUPR: {metrics['aupr']:.5f} "
            res_str += f"| Val Precision: {metrics['precision']:.5f} "
            res_str += f"| Val Recall: {metrics['recall']:.5f} "
            res_str += f"| Val F1: {metrics['f1']:.5f} |"

            border = "-" * len(res_str)

            print(
                "\n".join(
                    [
                        "",
                        border,
                        res_str,
                        border,
                        "",
                    ]
                )
            )

        self._reset_metrics("val")

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)
        y = y.float()

        loss = self.loss(y_hat, y)

        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        self._update_metrics(
            y_hat,
            y,
            metrics_prefix="test",
        )

        # Save predictions for plots
        self.test_scores.append(y_hat.detach())
        self.test_targets.append(y.detach())

    def on_test_epoch_end(self):
        metrics = self._log_metrics(
            metrics_prefix="test",
            prog_bar=True,
        )

        res_str = f"| Test Acc: {metrics['acc']:.5f} "
        res_str += f"| Test AUROC: {metrics['auroc']:.5f} "
        res_str += f"| Test AUPR: {metrics['aupr']:.5f} "
        res_str += f"| Test Precision: {metrics['precision']:.5f} "
        res_str += f"| Test Recall: {metrics['recall']:.5f} "
        res_str += f"| Test F1: {metrics['f1']:.5f} |"

        border = "-" * len(res_str)

        print(
            "\n".join(
                [
                    "",
                    border,
                    res_str,
                    border,
                    "",
                ]
            )
        )

        # --------------------------------------------------------------
        # Test plots
        # --------------------------------------------------------------

        if self.test_scores:
            y_score = torch.cat(self.test_scores, dim=0)
            y_true = torch.cat(self.test_targets, dim=0)

            if self.show_figure:
                fig, (ax1, ax2) = plt.subplots(
                    1,
                    2,
                    figsize=(12, 4),
                )
            else:
                ax1 = None
                ax2 = None

            self.calculate_auroc(
                y_score,
                y_true,
                ax=ax1,
            )

            self.plot_hist(
                y_score,
                y_true,
                ax=ax2,
            )

            if self.show_figure:
                plt.tight_layout()
                plt.show()

        self._reset_metrics("test")

        self.test_scores.clear()
        self.test_targets.clear()

    # ------------------------------------------------------------------
    # Test plots
    # ------------------------------------------------------------------

    def calculate_auroc(
        self,
        y_score,
        y_true,
        ax=None,
    ):
        """
        Calculate and optionally plot ROC curves for each label.
        """
        y_score = torch.sigmoid(y_score.float()).cpu().numpy()
        y_true = y_true.cpu().numpy()

        fpr = {}
        tpr = {}
        roc_auc = {}

        for label_idx in range(self.n_labels):
            fpr[label_idx], tpr[label_idx], _ = roc_curve(
                y_true[:, label_idx],
                y_score[:, label_idx],
            )

            roc_auc[label_idx] = auc(
                fpr[label_idx],
                tpr[label_idx],
            )

        if ax is None:
            return

        for label_idx in range(self.n_labels):
            ax.plot(
                fpr[label_idx],
                tpr[label_idx],
                lw=1,
                label=f"Label {label_idx} (AUC = {roc_auc[label_idx]:.2f})",
            )

        ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            lw=1,
            label="Random (AUC = 0.5)",
        )

        ax.set_xlim([-0.05, 1.0])
        ax.set_ylim([0.0, 1.05])

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves for Each Label")
        ax.legend(loc="lower right")

    def plot_hist(
        self,
        y_score,
        y_true,
        ax=None,
    ):
        """
        Plot the number of positive predictions for each label.
        """
        if ax is None:
            return

        y_score = torch.sigmoid(y_score.float()).cpu().numpy()

        y_pred = (y_score > 0.5).astype(int)

        positive_counts = np.sum(
            y_pred,
            axis=0,
        )

        ax.bar(
            np.arange(self.n_labels),
            positive_counts,
            edgecolor="black",
        )

        for label_idx, count in enumerate(positive_counts):
            ax.text(
                label_idx,
                count,
                str(count),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xlabel("Label")
        ax.set_ylabel("Positive Predictions Count")
        ax.set_title("Positive Predictions per Label")
        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.7,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)

        return {
            "y": y.float().cpu().detach(),
            "pred": y_hat.cpu().detach(),
        }