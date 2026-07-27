import torch
import torch.nn as nn
import lightning.pytorch as L
import numpy as np

from torchmetrics import PearsonCorrCoef
from torchmetrics import (
    Accuracy,
    AUROC,
    AveragePrecision,
    Precision,
    Recall,
    F1Score,
)
from sklearn.metrics import (
    auc,
    roc_curve,
)

import matplotlib.pyplot as plt
from itertools import cycle

class LitModel_Reddy_Reg(L.LightningModule):

    def __init__(self, model,cell_types, loss = nn.MSELoss(), print_each = 1, weight_decay=1e-2, lr=3e-4, ):

        super().__init__()

        self.model = model

        self.loss = loss
        self.print_each = print_each
        self.weight_decay = weight_decay
        self.lr = lr

        num_outputs = len(cell_types)

        self.num_outputs = num_outputs

        if isinstance(cell_types, str):
            cell_types = [cell_types]

        self.cell_types = cell_types

        self.train_pearson = PearsonCorrCoef(num_outputs=num_outputs)
        self.val_pearson = PearsonCorrCoef(num_outputs=num_outputs)
        self.test_pearson = PearsonCorrCoef(num_outputs=num_outputs)

    def forward(self, x):

        return self.model(x)
    
    def labels_and_predicted_unsqueeze(self, pred, targets):
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)  # [1076] -> [1076, 1]
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)  # [1076] -> [1076, 1]
        return pred, targets

    def training_step(self, batch, batch_nb):
        
        x, y = batch
        if x.ndim == 3 and x.shape[2] < x.shape[1]:
            x = x.permute(0, 2, 1)
        y_hat = self.forward(x)
                  
        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y) # [1076] -> [1076, 1]
        
        loss = self.loss(y_hat, y)

        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )

        self.train_pearson.update(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        if x.ndim == 3 and x.shape[2] < x.shape[1]:
            x = x.permute(0, 2, 1)
        y_hat = self.forward(x)

        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y) # [1076] -> [1076, 1]

        loss = self.loss(y_hat, y)
        self.log(
            "val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )

        self.val_pearson.update(y_hat, y)

    def on_validation_epoch_end(self):

        val_str = ""
        train_str = ""

        train_pearson = self.train_pearson.compute()
        val_pearson = self.val_pearson.compute()

        for i in range(self.num_outputs):
            name_train_metric = "train_" + self.cell_types[i] + "_pearson"
            name_val_metric = "val_" + self.cell_types[i] + "_pearson"

            tr_pearson = train_pearson[i] if self.num_outputs > 1 else train_pearson
            v_pearson = val_pearson[i] if self.num_outputs > 1 else val_pearson
            self.log(
                name_train_metric,
                tr_pearson,
                prog_bar=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                name_val_metric,
                v_pearson,
                prog_bar=True,
                on_epoch=True,
                logger=True,
            )

            val_str += (
                f"| Val Pearson {self.cell_types[i]}: {v_pearson:.5f} "
            )
            train_str += f"| Train Pearson {self.cell_types[i]}: {tr_pearson:.5f} "

        mean_val_pearson = val_pearson.mean()
        mean_train_pearson = train_pearson.mean()

        self.log(
            "val_pearson", mean_val_pearson, prog_bar=True, on_epoch=True, logger=True
        )

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
        if x.ndim == 3 and x.shape[2] < x.shape[1]:
            x = x.permute(0, 2, 1)
        y_hat = self.forward(x)

        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y) # [1076] -> [1076, 1]

        loss = self.loss(y_hat, y)
        self.log(
            "test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )

        self.test_pearson.update(y_hat, y)

    def on_test_epoch_end(self):
        test_pearson = self.test_pearson.compute()

        for i in range(self.num_outputs):
            te_pearson = test_pearson[i] if self.num_outputs > 1 else test_pearson
            name_of_metric = "test_" + self.cell_types[i] + "_pearson"
            self.log(name_of_metric, te_pearson, prog_bar=True)

        self.test_pearson.reset()

    def predict_step(self, batch, _):
        x, y = batch
        if x.ndim == 3 and x.shape[2] < x.shape[1]:
            x = x.permute(0, 2, 1)
        pred = self.forward(x)

        pred, y = self.labels_and_predicted_unsqueeze(pred, y) # [1076] -> [1076, 1]

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


class LitModel_Reddy_Clas(L.LightningModule):
    def __init__(
        self,
        weight_decay,
        lr,
        n_labels=3,
        show_figure=True,
        model=None,
        loss=nn.BCEWithLogitsLoss(),
        print_each=1,
    ):
        super().__init__()

        self.model = model

        self.loss = loss
        self.print_each = print_each
        self.weight_decay = weight_decay
        self.lr = lr

        self.val_acc = Accuracy(task="multilabel", num_labels=n_labels)
        self.val_auroc = AUROC(task="multilabel", num_labels=n_labels)
        self.val_aupr = AveragePrecision(task="multilabel", num_labels=n_labels)
        self.val_precision = Precision(
            task="multilabel", num_labels=n_labels, average="macro"
        )
        self.val_recall = Recall(
            task="multilabel", num_labels=n_labels, average="macro"
        )
        self.val_f1 = F1Score(task="multilabel", num_labels=n_labels, average="macro")

        self.test_acc = Accuracy(task="multilabel", num_labels=n_labels)
        self.test_auroc = AUROC(task="multilabel", num_labels=n_labels)
        self.test_aupr = AveragePrecision(task="multilabel", num_labels=n_labels)
        self.test_precision = Precision(
            task="multilabel", num_labels=n_labels, average="macro"
        )
        self.test_recall = Recall(
            task="multilabel", num_labels=n_labels, average="macro"
        )
        self.test_f1 = F1Score(task="multilabel", num_labels=n_labels, average="macro")

        # for plotting
        self.n_labels = n_labels
        self.show_figure = show_figure
        self.y_score = torch.tensor([])
        self.y_true = torch.tensor([])

    def setup(self, stage=None):
        self.y_score = self.y_score.to(self.device)
        self.y_true = self.y_true.to(self.device)

    def training_step(self, batch, batch_nb):
        X, y = batch
        y_hat = self.model(X)
        y = y.float()

        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.float()

        loss = self.loss(y_hat, y)

        y_metrics = y.long()

        self.val_acc(y_hat, y_metrics)
        self.val_auroc(y_hat, y_metrics)
        self.val_aupr(y_hat, y_metrics)
        self.val_precision(y_hat, y_metrics)
        self.val_recall(y_hat, y_metrics)
        self.val_f1(y_hat, y_metrics)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Acc: {self.val_acc.compute()} "
            res_str += f"| Val AUROC: {self.val_auroc.compute()} "
            res_str += f"| Val AUPR: {self.val_aupr.compute()} |"
            res_str += f"\n| Val Precision: {self.val_precision.compute()} "
            res_str += f"| Val Recall: {self.val_recall.compute()} "
            res_str += f"| Val F1: {self.val_f1.compute()} "
            border = "-" * 100
            print("\n".join(["", border, res_str, border, ""]))

        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_aupr.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.float()

        loss = self.loss(y_hat, y)

        y_metrics = y.long()

        self.test_acc(y_hat, y_metrics)
        self.test_auroc(y_hat, y_metrics)
        self.test_aupr(y_hat, y_metrics)
        self.test_precision(y_hat, y_metrics)
        self.test_recall(y_hat, y_metrics)
        self.test_f1(y_hat, y_metrics)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

        # for plotting
        self.y_score = torch.cat([self.y_score, y_hat])
        self.y_true = torch.cat([self.y_true, y])

    def on_test_epoch_end(self):
        res_str = f"| Test Acc: {self.test_acc.compute()} "
        res_str += f"| Test AUROC: {self.test_auroc.compute()} "
        res_str += f"| Test AUPR: {self.test_aupr.compute()} |"
        res_str += f"\n| Test Precision: {self.test_precision.compute()} "
        res_str += f"| Test Recall: {self.test_recall.compute()} "
        res_str += f"| Test F1: {self.test_f1.compute()} "
        border = "-" * 100
        print("\n".join(["", border, res_str, border, ""]))

        if self.show_figure:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        self.calculate_auroc(
            self.y_score, self.y_true, self.n_labels, ax1 if self.show_figure else None
        )
        self.plot_hist(
            self.y_score, self.y_true, self.n_labels, ax2 if self.show_figure else None
        )

        if self.show_figure:
            plt.tight_layout()
            plt.show()

        self.test_acc.reset()
        self.test_auroc.reset()
        self.test_aupr.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.y_score = torch.tensor([], device=self.device)
        self.y_true = torch.tensor([], device=self.device)

    def calculate_auroc(self, y_score, y_true, n_labels, ax=None):
        y_score = torch.sigmoid(y_score.float()).cpu().numpy()
        y_true = y_true.cpu().numpy()

        fpr, tpr, roc_auc = dict(), dict(), dict()

        # Compute ROC curve and AUC for each label
        for i in range(n_labels):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        if ax is not None:
            colors = cycle(
                ["orange", "green", "red", "purple", "blue", "yellow", "cyan", "brown"]
            )

            # Plot ROC curves for each label
            for i, color in zip(range(n_labels), colors):
                ax.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    lw=1,
                    label=f"Label {i} (AUC = {roc_auc[i]:0.2f})",
                )

            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves for each label")
            ax.legend(loc="lower right")

    def plot_hist(self, y_score, y_true, n_labels, ax=None):
        y_score = torch.sigmoid(y_score.float()).cpu().numpy()
        y_pred = (y_score > 0.5).astype(int)
        y_true = y_true.cpu().numpy()

        # Plot histogram if axis is provided
        if ax is not None:
            pos_counts = np.sum(y_pred, axis=0)

            ax.bar(np.arange(n_labels), pos_counts, color="skyblue", edgecolor="black")

            for i, count in enumerate(pos_counts):
                ax.text(
                    i,
                    count,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set_xlabel("Label")
            ax.set_ylabel("Positive predictions count")
            ax.set_title("Positive predictions per label")
            ax.grid(axis="y", linestyle="--", alpha=0.7)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return {"y": y.squeeze().float().cpu().detach(), "pred": y_hat.cpu().detach()}
    
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