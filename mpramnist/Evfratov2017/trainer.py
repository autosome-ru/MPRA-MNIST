import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping

from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize


class LitModel_Evfratov(L.LightningModule):
    def __init__(
        self,
        model,
        loss=nn.CrossEntropyLoss(),
        print_each: int = 1,
        weight_decay: float = 1e-2,
        lr: float = 3e-4,
        n_classes: int = 10,
        is_reporter_net = False,
        # Early stopping
        early_stopping_patience: int = 10,
        early_stopping_metric: str = "val_loss",
        early_stopping_mode: str = "min",
        # LR scheduling (OneCycleLR)
        lr_pct_start: float = 0.3,
    ):
        super().__init__()

        self.model = model
        self.loss = loss
        self.print_each = print_each
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_pct_start = lr_pct_start
        self.is_reporter_net = is_reporter_net

        # Early stopping config — consumed by configure_callbacks()
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode

        self.n_classes = n_classes

        self.y_score = torch.tensor([])
        self.y_true = torch.tensor([])

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

    def setup(self, stage=None):
        self.y_score = self.y_score.to(self.device)
        self.y_true = self.y_true.to(self.device)

    # ------------------------------------------------------------------

    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.is_reporter_net:
            return optimizer

        # Note: unlike the Agarwal trainers (which offer a ReduceLROnPlateau
        # toggle), the Evfratov classification tasks are trained with a fixed
        # one-cycle schedule tied to the known step count.
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            three_phase=False,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.lr_pct_start,
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
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        y = y.long()

        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y = y.long()

        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.y_score = torch.cat([self.y_score, y_hat])
        self.y_true = torch.cat([self.y_true, y])

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.print_each == 0:
            print("\n| {}: {:.5f} |\n".format("Current_epoch", self.current_epoch))
            self.shared_test_val_epoch_end()
        self.y_score = torch.tensor([], device=self.device)
        self.y_true = torch.tensor([], device=self.device)

    def test_step(self, batch, _):
        x, y = batch
        y_hat = self.forward(x)
        y = y.long()

        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        self.y_score = torch.cat([self.y_score, y_hat])
        self.y_true = torch.cat([self.y_true, y])

    def on_test_epoch_end(self):
        self.shared_test_val_epoch_end(show_figure=True)
        self.y_score = torch.tensor([], device=self.device)
        self.y_true = torch.tensor([], device=self.device)

    def predict_step(self, batch, _):
        x, y = batch
        y_hat = self.forward(x)
        return {
            "predicted": y_hat.cpu().detach().float(),
            "target": y.cpu().detach().float(),
        }

    # ------------------------------------------------------------------
    # Metrics / reporting
    # ------------------------------------------------------------------

    def shared_test_val_epoch_end(self, show_figure: bool = False):
        border = "-" * 100

        fig, ax1, ax2 = (None, None, None)
        if show_figure:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        auroc = self.calculate_auroc(
            y_score=self.y_score,
            y_true=self.y_true,
            n_classes=self.n_classes,
            show_figure=show_figure,
            ax=ax1,
        )
        precision, recall, accuracy, f1, aupr = self.calculate_aupr(
            y_score=self.y_score, y_true=self.y_true, n_classes=self.n_classes
        )

        self.log("val_auroc", auroc, prog_bar=True, on_epoch=True, logger=True)
        self.log("val_aupr", aupr, prog_bar=False, on_epoch=True, logger=True)
        self.log("val_f1", f1, prog_bar=False, on_epoch=True, logger=True)

        res_str = "| "
        res_str += f"Precision: {precision:.5f} |"
        res_str += f" Recall: {recall:.5f} |"
        res_str += f" Accuracy: {accuracy:.5f} |"
        res_str += f" F1: {f1:.5f} |"
        res_str += f" AUROC: {auroc:.5f} |"
        res_str += f" AUPR: {aupr:.5f} |"

        print("\n".join(["", border, res_str, border, ""]))

        if show_figure:
            self.plot_hist(self.y_score, self.y_true, self.n_classes, ax2)
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    def calculate_auroc(self, y_score, y_true, n_classes, show_figure=False, ax=None):
        y_score = F.softmax(y_score.float(), dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        if show_figure and ax is not None:
            colors = cycle(
                ["orange", "green", "red", "purple", "blue", "yellow", "cyan", "brown"]
            )
            for i, color in zip(range(n_classes), colors):
                ax.plot(
                    fpr[i], tpr[i], color=color, lw=1,
                    label=f"Class {i} (AUC = {roc_auc[i]:0.2f})",
                )
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("multi-class ROC Curves")
            ax.legend(loc="lower right")

        return roc_auc_score(y_true_bin, y_score, multi_class="ovr", average="macro")

    def calculate_aupr(self, y_score, y_true, n_classes):
        y_score = F.softmax(y_score.float(), dim=1).cpu().numpy()
        y_pred = np.argmax(y_score, axis=1)
        y_true = y_true.cpu().numpy()

        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        pr_auc = average_precision_score(y_true_bin, y_score, average="macro")
        return precision, recall, accuracy, f1, pr_auc

    def plot_hist(self, y_score, y_true, n_classes, ax=None):
        if ax is None:
            return

        y_score = F.softmax(y_score.float(), dim=1).cpu().numpy()
        y_pred = np.argmax(y_score, axis=1)

        counts = np.bincount(y_pred, minlength=n_classes)
        ax.bar(np.arange(n_classes), counts, color="skyblue", edgecolor="black")

        for i, count in enumerate(counts):
            ax.text(i, count, str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xlabel("Class Label")
        ax.set_ylabel("Count")
        ax.set_title("Predicted Class Distribution")
        ax.grid(axis="y", linestyle="--", alpha=0.7)