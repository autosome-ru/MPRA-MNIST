import torch
import torch.nn as nn
import lightning.pytorch as L
import numpy as np

from torchmetrics import PearsonCorrCoef

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
    roc_curve,
)
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
    
class LitModel_Arensbergen_Reg(L.LightningModule):
    def __init__( self, model, loss = nn.MSELoss(), print_each = 1, weight_decay=1e-2, lr=3e-4, cell_types=["HepG2", "K562"],):

        super().__init__()

        self.model = model

        self.loss = loss
        self.print_each = print_each
        self.weight_decay = weight_decay
        self.lr = lr

        if isinstance(cell_types, str):
            cell_types = [cell_types]

        self.cell_types = cell_types
        self.num_outputs = len(cell_types)

        self.train_pearson = PearsonCorrCoef(num_outputs=self.num_outputs)
        self.val_pearson = PearsonCorrCoef(num_outputs=self.num_outputs)
        self.test_pearson = PearsonCorrCoef(num_outputs=self.num_outputs)

    def forward(self, x):
        return self.model(x)
    
    def labels_and_predicted_unsqueeze(self, pred, targets):
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)  # [1076] -> [1076, 1]
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)  # [1076] -> [1076, 1]
        return pred, targets

    def training_step(self, batch, batch_nb):
        X, y = batch
        if X.ndim == 3 and X.shape[2] < X.shape[1]:
            X = X.permute(0, 2, 1)
        y_hat = self.model(X)

        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y) # [1076] -> [1076, 1]

        loss = self.loss(y_hat, y)

        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )
        self.train_pearson.update(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if x.ndim == 3 and x.shape[2] < x.shape[1]:
            x = x.permute(0, 2, 1)
        y_hat = self.model(x)

        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y) # [1076] -> [1076, 1]

        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
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
        y_hat = self.model(x)

        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y) # [1076] -> [1076, 1]

        loss = self.loss(y_hat, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

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
        y_hat = self.model(x)

        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y) # [1076] -> [1076, 1]

        return {
            "predicted": y_hat.cpu().detach().float(),
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
    
class LitModel_Arensbergen_Clas(L.LightningModule):
    def __init__(
        self,
        weight_decay,
        lr,
        cell_types=["K562", "HepG2"],
        n_classes=10,
        model=None,
        loss=nn.CrossEntropyLoss(),
        print_each=1,
    ):
        super().__init__()

        self.weight_decay = weight_decay
        self.lr = lr
        self.model = model
        self.loss = loss

        self.cell_types = cell_types
        self.loss = torch.nn.CrossEntropyLoss()
        self.print_each = print_each
        self.n_classes = n_classes

        self.y_score = torch.tensor([])
        self.y_true = torch.tensor([])

    def setup(self, stage=None):
        self.y_score = self.y_score.to(self.device)
        self.y_true = self.y_true.to(self.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        X, y = batch
        if x.ndim == 3 and x.shape[2] < x.shape[1]:
            x = x.permute(0, 2, 1)
        y_hat = self.model(x)
        y = y.long()

        loss = self.loss(y_hat[:, 0:5], y[:, 0])
        loss += self.loss(y_hat[:, 5:10], y[:, 1])

        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if x.ndim == 3 and x.shape[2] < x.shape[1]:
            x = x.permute(0, 2, 1)
        y_hat = self.model(x)
        y = y.long()

        loss = self.loss(y_hat[:, 0:5], y[:, 0])
        loss += self.loss(y_hat[:, 5:10], y[:, 1])

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.y_score = torch.cat([self.y_score, y_hat])
        self.y_true = torch.cat([self.y_true, y])

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.print_each == 0:
            print("\n| {}: {:.5f} |\n".format("Current_epoch", self.current_epoch))
            self.shared_test_val_epoch_end()
        self.y_score = torch.tensor([], device=self.device)
        self.y_true = torch.tensor([], device=self.device)

        return None

    def test_step(self, batch, _):
        x, y = batch
        if x.ndim == 3 and x.shape[2] < x.shape[1]:
            x = x.permute(0, 2, 1)
        y_hat = self.model(x)
        y = y.long()

        loss = self.loss(y_hat[:, 0:5], y[:, 0])
        loss += self.loss(y_hat[:, 5:10], y[:, 1])
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        self.y_score = torch.cat([self.y_score, y_hat])
        self.y_true = torch.cat([self.y_true, y])

    def on_test_epoch_end(self):
        self.shared_test_val_epoch_end(show_figure=True)

        self.y_score = torch.tensor([], device=self.device)
        self.y_true = torch.tensor([], device=self.device)

    def shared_test_val_epoch_end(self, show_figure=False):
        border = "-" * 100
        plt_index = 0

        if show_figure:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle("ROC Curves Comparison", fontsize=14)
        else:
            fig, ax1, ax2 = None, None, None

        for i in range(len(self.cell_types)):
            auroc = self.calculate_auroc(
                y_score=self.y_score[:, i * 5 : i * 5 + 5],
                y_true=self.y_true[:, i],
                n_classes=self.n_classes // 2,
                show_figure=show_figure,
                name=self.cell_types[i],
                ax=ax1 if plt_index == 0 else ax2 if show_figure else None,
            )
            precision, recall, accuracy, f1, aupr = self.calculate_aupr(
                y_score=self.y_score[:, i * 5 : i * 5 + 5],
                y_true=self.y_true[:, i],
                n_classes=self.n_classes // 2,
            )

            class_str = f"| {self.cell_types[i]}: |"
            class_str += f"| Precision: {precision:.5f} |"
            class_str += f" Recall: {recall:.5f} |"
            class_str += f" Accuracy: {accuracy:.5f} |"
            class_str += f" F1: {f1:.5f} |"
            class_str += f" Val_AUCROC: {auroc:.5f} |"
            class_str += f" Val_AUPR: {aupr:.5f} |"

            print("\n".join(["", border, class_str, border, ""]))

            if show_figure and plt_index == 1:
                plt.tight_layout()
                plt.show()
                plt.close(fig)
                plt_index = 0
            elif show_figure:
                plt_index += 1

    def calculate_auroc(
        self, y_score, y_true, n_classes, name, show_figure=False, ax=None
    ):
        y_score = F.softmax(y_score.float(), dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()

        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

        fpr, tpr, roc_auc = dict(), dict(), dict()

        # Compute ROC curve and AUC for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        if show_figure and ax is not None:
            colors = cycle(["orange", "green", "red", "purple", "blue"])

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
            ax.set_title(f"{name} multi-class ROC Curves")
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


