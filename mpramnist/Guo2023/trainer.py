import torch
import torch.nn as nn
import lightning.pytorch as L
import numpy as np

from lightning.pytorch.callbacks import EarlyStopping
from torchmetrics import PearsonCorrCoef


class LitModel_Guo(L.LightningModule):

    def __init__(
        self,
        model,
        loss=nn.MSELoss(),
        print_each: int = 1,
        weight_decay: float = 1e-2,
        lr: float = 3e-4,
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

        self.train_pearson = PearsonCorrCoef()
        self.val_pearson = PearsonCorrCoef()
        self.test_pearson = PearsonCorrCoef()

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
        self.optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        if self.lr_sheduler_to_use == "one_cycle":

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
        
        elif self.lr_sheduler_to_use == "reducelronplateau":
            
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
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
        
        else:
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                betas=(0.8661062881299633, 0.879223105336538),
                eps=1e-08,
                weight_decay=self.weight_decay,
                lr=self.lr,
                amsgrad=True,
            )

            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=self.optimizer, T_0=4096, T_mult=1, eta_min=0.0, last_epoch=-1
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "learning_rate",
            }

        return [self.optimizer], [lr_scheduler_config]

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def labels_and_predicted_unsqueeze(self, pred, targets):
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)  # [1076] -> [1076, 1]
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)  # [1076] -> [1076, 1]
        return pred, targets

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        X, y = batch
        y_hat = self.forward(X)

        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y)  # [1076] -> [1076, 1]

        loss = self.loss(y_hat, y)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        self.train_pearson.update(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y)  # [1076] -> [1076, 1]

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

        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y)  # [1076] -> [1076, 1]

        loss = self.loss(y_hat, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.test_pearson.update(y_hat, y)

    def on_test_epoch_end(self):
        test_pearson = self.test_pearson.compute()
        self.log("test_pearson", test_pearson, prog_bar=True)
        self.test_pearson.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        seqs, labels, fdrs = batch

        if isinstance(seqs, dict):
            seq_x = seqs.get("seq")
            seq_alt_x = seqs.get("seq_alt")

            ref_pred = self.model(seq_x)
            alt_pred = self.model(seq_alt_x)

        else:
            ref_pred = self.model(seqs)
            alt_pred = None

        result = {
            "ref_predicted": ref_pred.cpu().detach().float(),
            "alt_predicted": alt_pred.cpu().detach().float(),
            "target": labels.cpu().detach().float(),
            "fdr": fdrs.cpu().detach().float(),
        }

        return result