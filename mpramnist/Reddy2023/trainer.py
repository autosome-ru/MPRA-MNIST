import torch
import torch.nn as nn
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping
from torchmetrics import PearsonCorrCoef

import numpy as np
import torch.nn.functional as F
from itertools import cycle

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
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
        loss=nn.CrossEntropyLoss(),
        print_each: int = 1,
        weight_decay: float = 1e-2,
        lr: float = 3e-4,
        cell_types: list[str] = ["JURKAT", "K562", "THP1"],
        n_classes: int = 10,
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

        # Early stopping config — consumed by configure_callbacks()
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode

        if isinstance(cell_types, str):
            cell_types = [cell_types]

        self.cell_types = cell_types
        self.n_classes = n_classes
        # assumes classes are split evenly across cell types, e.g. 10 classes / 2 cell
        # types -> 5 classes per cell type (logits sliced accordingly in compute_loss)
        self.classes_per_cell_type = n_classes // len(cell_types)

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

    def compute_loss(self, y_hat, y):
        """Sums the per-cell-type CrossEntropy loss over sliced logit blocks."""
        n = self.classes_per_cell_type
        loss = 0.0
        for i in range(len(self.cell_types)):
            loss = loss + self.loss(y_hat[:, i * n : (i + 1) * n], y[:, i])
        return loss

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

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

        loss = self.compute_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y = y.long()

        loss = self.compute_loss(y_hat, y)
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

        loss = self.compute_loss(y_hat, y)
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
        plt_index = 0
        n = self.classes_per_cell_type

        fig, ax1, ax2 = (None, None, None)
        if show_figure:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle("ROC Curves Comparison", fontsize=14)

        for i, cell_type in enumerate(self.cell_types):
            y_score_i = self.y_score[:, i * n : (i + 1) * n]
            y_true_i = self.y_true[:, i]

            auroc = self.calculate_auroc(
                y_score=y_score_i,
                y_true=y_true_i,
                n_classes=n,
                show_figure=show_figure,
                name=cell_type,
                ax=ax1 if plt_index == 0 else ax2 if show_figure else None,
            )
            precision, recall, accuracy, f1, aupr = self.calculate_aupr(
                y_score=y_score_i, y_true=y_true_i, n_classes=n
            )

            self.log(f"val_{cell_type}_auroc", auroc, prog_bar=False, on_epoch=True, logger=True)
            self.log(f"val_{cell_type}_aupr", aupr, prog_bar=False, on_epoch=True, logger=True)
            self.log(f"val_{cell_type}_f1", f1, prog_bar=False, on_epoch=True, logger=True)

            class_str = f"| {cell_type}: |"
            class_str += f"| Precision: {precision:.5f} |"
            class_str += f" Recall: {recall:.5f} |"
            class_str += f" Accuracy: {accuracy:.5f} |"
            class_str += f" F1: {f1:.5f} |"
            class_str += f" Val_AUCROC: {auroc:.5f} |"
            class_str += f" Val_AUPR: {aupr:.5f} |"

            print("\n".join(["", border, class_str, border, ""]))

            if show_figure:
                if plt_index == 1:
                    plt.tight_layout()
                    plt.show()
                    plt.close(fig)
                    plt_index = 0
                else:
                    plt_index += 1

    def calculate_auroc(self, y_score, y_true, n_classes, name, show_figure=False, ax=None):
        y_score = F.softmax(y_score.float(), dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        if show_figure and ax is not None:
            colors = cycle(["orange", "green", "red", "purple", "blue"])
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