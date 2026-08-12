import torch
import torch.nn as nn
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping
from torchmetrics import PearsonCorrCoef

import torch
import torch.nn as nn
from torchmetrics import Metric


class MaskedMSE(nn.Module):
    def forward(self, y_pred, y_true):
        mask = ~torch.isnan(y_true)
        diff = torch.where(mask, y_pred - y_true, torch.zeros_like(y_true))
        loss = diff.pow(2)
        return loss.sum() / mask.sum().clamp(min=1)

class MaskedPearsonCorrCoef(Metric):
    full_state_update = False

    def __init__(self, num_outputs=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_outputs = num_outputs

        self.add_state("sum_x", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_y", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_x2", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_y2", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_xy", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(num_outputs), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.dim() == 1:
            preds = preds.unsqueeze(-1)
        if target.dim() == 1:
            target = target.unsqueeze(-1)

        mask = ~torch.isnan(target)

        preds = torch.where(mask, preds, torch.zeros_like(preds))
        target = torch.where(mask, target, torch.zeros_like(target))

        self.sum_x += preds.sum(dim=0)
        self.sum_y += target.sum(dim=0)
        self.sum_x2 += (preds ** 2).sum(dim=0)
        self.sum_y2 += (target ** 2).sum(dim=0)
        self.sum_xy += (preds * target).sum(dim=0)
        self.count += mask.sum(dim=0)

    def compute(self):
        n = self.count.clamp(min=1)

        numerator = self.sum_xy - (self.sum_x * self.sum_y / n)
        denominator = torch.sqrt(
            (self.sum_x2 - (self.sum_x ** 2) / n) *
            (self.sum_y2 - (self.sum_y ** 2) / n)
        )

        corr = numerator / denominator.clamp(min=1e-12)

        invalid = (self.count < 2) | (denominator <= 1e-12)
        corr = torch.where(invalid, torch.full_like(corr, float("nan")), corr)

        if self.num_outputs == 1:
            return corr.squeeze(0)
        return corr

class LitModel_Fromel(L.LightningModule):
    def __init__(
        self,
        model,
        loss=None,
        print_each: int = 1,
        weight_decay: float = 1e-2,
        lr: float = 3e-4,
        activity_columns: list[str] = [
            "State_1M",
            "State_2D",
            "State_3E",
            "State_4M",
            "State_5M",
            "State_6N",
            "State_7M",
        ],
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

        if loss is None:
            loss = MaskedMSE()

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

        if isinstance(activity_columns, str):
            activity_columns = [activity_columns]

        self.activity_columns = activity_columns
        self.num_outputs = len(activity_columns)

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
                f"train_{self.activity_columns[i]}_pearson",
                tr_pearson,
                prog_bar=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"val_{self.activity_columns[i]}_pearson",
                v_pearson,
                prog_bar=True,
                on_epoch=True,
                logger=True,
            )

            val_str += f"| Val Pearson {self.activity_columns[i]}: {v_pearson:.5f} "
            train_str += f"| Train Pearson {self.activity_columns[i]}: {tr_pearson:.5f} "

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
            self.log(f"test_{self.activity_columns[i]}_pearson", te_pearson, prog_bar=True)

        self.test_pearson.reset()

    def predict_step(self, batch, _):
        x, y = batch
        pred = self.forward(x)
        pred, y = self.labels_and_predicted_unsqueeze(pred, y)
        return {
            "predicted": pred.cpu().detach().float(),
            "target": y.cpu().detach().float(),
        }