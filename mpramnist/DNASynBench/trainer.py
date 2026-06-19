import torch
import torch.nn as nn
import lightning.pytorch as L
import numpy as np
from torchmetrics import PearsonCorrCoef, AUROC, Precision, Recall, F1Score
from sklearn.metrics import (
    auc,
    roc_curve,
)
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle

class LitModel_DNASyn_REG(L.LightningModule):
    def __init__(self, model, loss, print_each, weight_decay=1e-2, lr=3e-4):
        super().__init__()

        self.model = model

        self.loss = loss
        self.print_each = print_each
        self.weight_decay = weight_decay

        self.lr = lr
        self.train_pearson = PearsonCorrCoef()
        self.val_pearson = PearsonCorrCoef()
        self.test_pearson = PearsonCorrCoef()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        X, y = batch
        y_hat = self.forward(X)
        
        loss = self.loss(y_hat, y)

        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )

        self.train_pearson.update(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_pearson.update(y_hat, y)

    def on_validation_epoch_end(self):
        train_pearson = self.train_pearson.compute()
        val_pearson = self.val_pearson.compute()

        self.log("val_pearson", val_pearson, prog_bar=True)
        self.log("train_pearson", train_pearson)

        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Loss: {self.trainer.callback_metrics['val_loss']:.5f} "
            res_str += f"| Val Pearson: {val_pearson:.5f} "

            res_str += f"| Train Pearson: {train_pearson:.5f} "
            border = "-" * len(res_str)
            print("\n".join(["", border, res_str, border, ""]))

        self.train_pearson.reset()
        self.val_pearson.reset()

    def test_step(self, batch, _):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.test_pearson.update(y_hat, y)

    def on_test_epoch_end(self):
        test_pearson = self.test_pearson.compute()
        self.log("test_pearson", test_pearson, prog_bar=True)
        self.test_pearson.reset()

    def predict_step(self, batch, _):
        x, y = batch
        pred = self.forward(x)

        return {
            "predicted": pred.cpu().detach().float(),
            "target": y.cpu().detach().float(),
        }

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
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

        return [self.optimizer], [lr_scheduler_config]

class LitModel_DNASyn_CLS(L.LightningModule):
    def __init__(
        self,
        weight_decay,
        lr,
        n_classes,
        show_figure=True,
        model=None,
        loss=nn.CrossEntropyLoss(),
        print_each=1,
    ):
        super().__init__()

        self.model = model

        self.loss = loss
        self.print_each = print_each
        self.weight_decay = weight_decay

        self.lr = lr

        self.val_auroc = AUROC(task="multiclass", num_classes=n_classes)
        self.val_precision = Precision(task="multiclass", num_classes=n_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=n_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=n_classes, average="macro")

        self.test_auroc = AUROC(task="multiclass", num_classes=n_classes)
        self.test_precision = Precision(task="multiclass", num_classes=n_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=n_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=n_classes, average="macro")

        # for plotting
        self.n_classes = n_classes
        self.show_figure = show_figure
        self.y_score = torch.tensor([])
        self.y_true = torch.tensor([])

    def setup(self, stage=None):
        self.y_score = self.y_score.to(self.device)
        self.y_true = self.y_true.to(self.device)

    def training_step(self, batch, batch_nb):
        X, y = batch
        y_hat = self.model(X)
        y = y.long()

        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.long()

        loss = self.loss(y_hat, y)

        self.val_auroc(y_hat, y)
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_f1(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        val_auroc = self.val_auroc.compute()
        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val AUROC: {val_auroc} "
            res_str += f"\n| Val Precision: {self.val_precision.compute()} "
            res_str += f"| Val Recall: {self.val_recall.compute()} "
            res_str += f"| Val F1: {self.val_f1.compute()} "
            border = "-" * 100
            print("\n".join(["", border, res_str, border, ""]))
        self.log("val_auroc", val_auroc, on_epoch=True, prog_bar=True)
        
        self.val_auroc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.long()

        loss = self.loss(y_hat, y)

        self.test_auroc(y_hat, y)
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_f1(y_hat, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

        # for plotting
        self.y_score = torch.cat([self.y_score, y_hat])
        self.y_true = torch.cat([self.y_true, y])

    def on_test_epoch_end(self):
        res_str = f"| Epoch: {self.current_epoch} "
        res_str += f"| Test AUROC: {self.test_auroc.compute()} "
        res_str += f"\n| Test Precision: {self.test_precision.compute()} "
        res_str += f"| Test Recall: {self.test_recall.compute()} "
        f1 = self.test_f1.compute()
        self.log("test_f1", f1, on_epoch=True, prog_bar=True)
        res_str += f"| Test F1: {f1} "
        border = "-" * 100
        print("\n".join(["", border, res_str, border, ""]))

        if self.show_figure:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        self.calculate_auroc(
            self.y_score, self.y_true, self.n_classes, ax1 if self.show_figure else None
        )
        self.plot_hist(
            self.y_score, self.y_true, self.n_classes, ax2 if self.show_figure else None
        )

        if self.show_figure:
            plt.tight_layout()
            plt.show()

        self.test_auroc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.y_score = torch.tensor([], device=self.device)
        self.y_true = torch.tensor([], device=self.device)

    def calculate_auroc(self, y_score, y_true, n_classes, ax=None):
        y_score = F.softmax(y_score.float(), dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes+1))

        fpr, tpr, roc_auc = dict(), dict(), dict()

        # Compute ROC curve and AUC for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        if ax is not None:
            colors = cycle(
                ["orange", "green", "red", "purple", "blue", "yellow", "cyan", "brown"]
            )

            # Plot ROC curves for each class
            for i, color in zip(range(n_classes), colors):
                ax.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    lw=1,
                    label=f"Class {i} (AUC = {roc_auc[i]:0.2f})",
                )

            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend(loc="lower right")

    def plot_hist(self, y_score, y_true, n_classes, ax=None):
        y_score = F.softmax(y_score.float(), dim=1).cpu().numpy()
        y_pred = np.argmax(y_score, axis=1)
        y_true = y_true.cpu().numpy()

        # Plot histogram if axis is provided
        if ax is not None:
            counts = np.bincount(y_pred, minlength=n_classes)
            ax.bar(np.arange(n_classes), counts, color="skyblue", edgecolor="black")

            for i, count in enumerate(counts):
                ax.text(
                    i,
                    count,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set_xticks(np.arange(n_classes))
            ax.set_xlabel("Class Label")
            ax.set_ylabel("Count")
            ax.set_title("Predicted Class Distribution")
            ax.grid(axis="y", linestyle="--", alpha=0.7)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return {
            "y": y.squeeze().long().cpu().detach().float(),
            "pred": y_hat.cpu().detach().float(),
        }

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
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

        return [self.optimizer], [lr_scheduler_config]
