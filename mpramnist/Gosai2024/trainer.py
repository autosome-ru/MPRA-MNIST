import torch
import torch.nn as nn
import lightning.pytorch as L
import numpy as np

from torchmetrics import PearsonCorrCoef

class LitModel_Gosai(L.LightningModule):
    def __init__(self, model, loss = nn.MSELoss(), print_each = 1, weight_decay=1e-2, lr=3e-4, cell_types=["K562", "HepG2", "SKNSH"], use_one_cycle=False):

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

        self.use_one_cycle = use_one_cycle

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

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        self.train_pearson.update(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if x.ndim == 3 and x.shape[2] < x.shape[1]:
            x = x.permute(0, 2, 1)
        y_hat = self.forward(x)

        y_hat, y = self.labels_and_predicted_unsqueeze(y_hat, y) # [1076] -> [1076, 1]

        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
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
    
class LitModel_Gosai_Pairwise(LitModel_Gosai):

    def __init__(self, model, loss = nn.MSELoss(), print_each = 1, weight_decay=1e-2, lr=3e-4, cell_types=["K562", "HepG2", "SKNSH"], use_one_cycle=False):

        super().__init__(model = model, loss = loss, print_each=print_each,weight_decay=weight_decay,lr=lr,cell_types=cell_types,use_one_cycle=use_one_cycle)

    def process_batch(self, batch):
        seqs, labels = batch

        seq_x = seqs.get("seq")
        seq_alt_x = seqs.get("seq_alt")

        ref_pred = self.model(seq_x)

        alt_pred = self.model(seq_alt_x)

        diff = ref_pred - alt_pred

        loss = self.loss(diff, labels)
        
        return loss, diff, labels

    def training_step(self, batch, batch_nb):

        loss, y_hat, y = self.process_batch(batch)

        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )

        self.train_pearson.update(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):

        loss, y_hat, y = self.process_batch(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_pearson.update(y_hat, y)

    def test_step(self, batch, _):

        loss, y_hat, y = self.process_batch(batch)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        self.test_pearson.update(y_hat, y)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        seqs, labels = batch

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
            "target": labels.cpu().detach().float(),
        }

        if alt_pred is not None:
            result["alt_predicted"] = alt_pred.cpu().detach().float()

        return result

    

    