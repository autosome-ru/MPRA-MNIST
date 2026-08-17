import torch
import torch.nn as nn
import lightning.pytorch as L
import numpy as np

from torchmetrics import PearsonCorrCoef

class LitModel_Lee(L.LightningModule):
    def __init__(self, model, loss, print_each, weight_decay=1e-2, lr=3e-4, use_one_cycle=False):
        super().__init__()

        self.model = model

        self.loss = loss
        self.print_each = print_each
        self.weight_decay = weight_decay

        self.lr = lr
        self.train_pearson = PearsonCorrCoef()
        self.val_pearson = PearsonCorrCoef()
        self.test_pearson = PearsonCorrCoef()

        self.use_one_cycle = use_one_cycle

        VARIANT_TYPE_MAPPING = {1: 'emVar', 2: 'MPRA-Allelic', 3: 'MPRA-nonallelic', 4: 'Uncertain'}

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

        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y) # [1076] -> [1076, 1]

        loss = self.loss(y_hat, y)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
    
        self.train_pearson.update(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y) # [1076] -> [1076, 1]

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

        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y) # [1076] -> [1076, 1]

        loss = self.loss(y_hat, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.test_pearson.update(y_hat, y)

    def on_test_epoch_end(self):
        test_pearson = self.test_pearson.compute()
        self.log("test_pearson", test_pearson, prog_bar=True)
        self.test_pearson.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        seqs, labels, var_type, rev_pred = batch

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
            "variant_type": var_type.cpu().detach().int(),
            "reverse_prediction": rev_pred.cpu().detach().float(),
                }

        return result

    def configure_optimizers(self):
        if self.use_one_cycle:
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