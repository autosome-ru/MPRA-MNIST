from torch.utils.data import DataLoader
from mpramnist.trainers import LitModel
import lightning.pytorch as L

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


class LitModel_Fromel(LitModel):
    def __init__(
        self,
        weight_decay,
        lr,
        activity_columns=[
            "State_1M",
            "State_2D",
            "State_3E",
            "State_4M",
            "State_5M",
            "State_6N",
            "State_7M",
        ],
        model=None,
        loss=None,
        print_each=1,
    ):
        if loss is None:
            loss = MaskedMSE()

        super().__init__(model, loss, print_each, weight_decay, lr)

        self.activity_columns = activity_columns
        self.num_outputs = len(self.activity_columns)

        self.train_pearson = MaskedPearsonCorrCoef(num_outputs=self.num_outputs)
        self.val_pearson = MaskedPearsonCorrCoef(num_outputs=self.num_outputs)
        self.test_pearson = MaskedPearsonCorrCoef(num_outputs=self.num_outputs)

    def forward(self, x):
        return self.model(x)

    def _ensure_2d(self, y_hat, y):
        if y_hat.dim() == 1:
            y_hat = y_hat.unsqueeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        x, y = batch
        if x.ndim == 3 and x.shape[2] < x.shape[1]:
            x = x.permute(0, 2, 1)
        y_hat = self.forward(x)
        y_hat, y = self._ensure_2d(y_hat, y)

        loss = self.loss(y_hat, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        self.train_pearson.update(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if x.ndim == 3 and x.shape[2] < x.shape[1]:
            x = x.permute(0, 2, 1)
        y_hat = self.forward(x)
        y_hat, y = self._ensure_2d(y_hat, y)

        loss = self.loss(y_hat, y)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        self.val_pearson.update(y_hat, y)

    def on_validation_epoch_end(self):
        train_pearson = self.train_pearson.compute()
        val_pearson = self.val_pearson.compute()

        val_str = ""
        train_str = ""

        if self.num_outputs == 1:
            train_value = train_pearson
            val_value = val_pearson

            self.log(
                f"train_{self.activity_columns[0]}_pearson",
                train_value,
                prog_bar=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"val_{self.activity_columns[0]}_pearson",
                val_value,
                prog_bar=True,
                on_epoch=True,
                logger=True,
            )

            mean_train_pearson = train_value
            mean_val_pearson = val_value

            train_str = f"| Train Pearson {self.activity_columns[0]}: {train_value:.5f} "
            val_str = f"| Val Pearson {self.activity_columns[0]}: {val_value:.5f} "
        else:
            for i, col in enumerate(self.activity_columns):
                self.log(
                    f"train_{col}_pearson",
                    train_pearson[i],
                    prog_bar=False,
                    on_epoch=True,
                    logger=True,
                )
                self.log(
                    f"val_{col}_pearson",
                    val_pearson[i],
                    prog_bar=True,
                    on_epoch=True,
                    logger=True,
                )

                train_str += f"| Train Pearson {col}: {train_pearson[i]:.5f} "
                val_str += f"| Val Pearson {col}: {val_pearson[i]:.5f} "

            mean_train_pearson = torch.nanmean(train_pearson)
            mean_val_pearson = torch.nanmean(val_pearson)

            train_str += f"| Mean Train Pearson: {mean_train_pearson:.5f} "
            val_str += f"| Mean Val Pearson: {mean_val_pearson:.5f} "

        self.log(
            "val_pearson",
            mean_val_pearson,
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )

        self.train_pearson.reset()
        self.val_pearson.reset()

        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Loss: {self.trainer.callback_metrics['val_loss']:.5f} "

            border = "-" * max(len(res_str), len(val_str), len(train_str))
            print(
                "\n".join(
                    ["", border, res_str, val_str + "|", train_str + "|", border, ""]
                )
            )

    def test_step(self, batch, batch_idx):
        x, y = batch
        if x.ndim == 3 and x.shape[2] < x.shape[1]:
            x = x.permute(0, 2, 1)
        y_hat = self.forward(x)
        y_hat, y = self._ensure_2d(y_hat, y)

        loss = self.loss(y_hat, y)
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        self.test_pearson.update(y_hat, y)

    def on_test_epoch_end(self):
        test_pearson = self.test_pearson.compute()

        if self.num_outputs == 1:
            self.log(
                f"test_{self.activity_columns[0]}_pearson",
                test_pearson,
                prog_bar=True,
            )
        else:
            for i, col in enumerate(self.activity_columns):
                self.log(
                    f"test_{col}_pearson",
                    test_pearson[i],
                    prog_bar=True,
                )
            self.log(
                "test_pearson",
                torch.nanmean(test_pearson),
                prog_bar=True,
            )

        self.test_pearson.reset()

    def predict_step(self, batch, batch_idx):
        x, y = batch
        if x.ndim == 3 and x.shape[2] < x.shape[1]:
            x = x.permute(0, 2, 1)
        pred = self.forward(x)
        pred, y = self._ensure_2d(pred, y)

        return {
            "predicted": pred.detach().cpu().float(),
            "target": y.detach().cpu().float(),
        }