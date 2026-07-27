import torch
import torch.nn as nn
import lightning.pytorch as L
import math

from torchmetrics import (
    Accuracy,
    AUROC,
    AveragePrecision,
    Precision,
    Recall,
    F1Score,
)

class LitModel_Sahu(L.LightningModule):
    def __init__(
        self, weight_decay, lr, model=None, loss=nn.BCEWithLogitsLoss(), print_each=1
    ):
        super().__init__()

        self.model = model

        self.loss = loss
        self.print_each = print_each
        self.weight_decay = weight_decay

        self.lr = lr

        # Validation metrics for binary classification
        self.val_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_aupr = AveragePrecision(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_f1 = F1Score(task="binary")

        # Test metrics for binary classification
        self.test_acc = Accuracy(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.test_aupr = AveragePrecision(task="binary")
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_f1 = F1Score(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        X, y = batch
        y_hat = self.model(X)

        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)

        loss = self.loss(y_hat, y)

        self.val_acc(y_hat, y)
        self.val_auroc(y_hat, y)
        self.val_aupr(y_hat, y.long())
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_f1(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        val_auroc = self.val_auroc.compute()
        val_aupr = self.val_aupr.compute()
        val_prec = self.val_precision.compute()
        val_rec = self.val_recall.compute()
        val_f1 = self.val_f1.compute()

        self.log("val_aupr", val_aupr, on_epoch=True, prog_bar=True)
        self.log("val_auroc", val_auroc, on_epoch=True, prog_bar=True)

        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Acc: {val_acc} "
            res_str += f"| Val AUROC: {val_auroc} "
            res_str += f"| Val AUPR: {val_aupr} |"
            res_str += f"\n| Val Precision: {val_prec} "
            res_str += f"| Val Recall: {val_rec} "
            res_str += f"| Val F1: {val_f1} "
            border = "-" * 100
            print("\n".join(["", border, res_str, border, ""]))

        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_aupr.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)

        loss = self.loss(y_hat, y)

        self.test_acc(y_hat, y)
        self.test_auroc(y_hat, y)
        self.test_aupr(y_hat, y.long())
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_f1(y_hat, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        test_aupr = self.test_aupr.compute()
        test_auroc = self.test_auroc.compute()

        self.log("test_aupr", test_aupr, on_epoch=True, prog_bar=True)
        self.log("test_auroc", test_auroc, on_epoch=True, prog_bar=True)

        res_str = f"| Test Acc: {self.test_acc.compute()} "
        res_str += f"| Test AUROC: {test_auroc} "
        res_str += f"| Test AUPR: {test_aupr} |"
        res_str += f"\n| Test Precision: {self.test_precision.compute()} "
        res_str += f"| Test Recall: {self.test_recall.compute()} "
        res_str += f"| Test F1: {self.test_f1.compute()} "
        border = "-" * 100
        print("\n".join(["", border, res_str, border, ""]))

        self.test_acc.reset()
        self.test_auroc.reset()
        self.test_aupr.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)

        return {
            "predicted": y_hat.squeeze().cpu().detach(),
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
    
class LitModel_Sahu_binary_legnet(LitModel_Sahu):
    def __init__(
        self, weight_decay, lr, model=None, loss=nn.BCEWithLogitsLoss(), print_each=1
    ):
        super().__init__(
            model=model,
            loss=loss,
            print_each=print_each,
            weight_decay=weight_decay,
            lr=lr,
        )
        self.output_dim = 1

        out_ch = 256

        activation = nn.SiLU

        self.head = nn.Sequential(
            nn.Linear(out_ch * 2, out_ch),
            nn.BatchNorm1d(out_ch),
            activation(),
            nn.Linear(out_ch, self.output_dim),
        )

    def _process_batch(self, batch):
        seqs, labels = batch
        
        enhancer = self.model(seqs["seq_enh"])
        promoter = self.model(seqs["seq"])

        concat = torch.cat([enhancer, promoter], dim=1)

        out = self.head(concat).squeeze(-1)
        return out, labels

    def training_step(self, batch, batch_nb):
        y_hat, y = self._process_batch(batch)

        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self._process_batch(batch)
        loss = self.loss(y_hat, y)

        self.val_acc(y_hat, y)
        self.val_auroc(y_hat, y)
        self.val_aupr(y_hat, y.long())
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_f1(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        y_hat, y = self._process_batch(batch)

        loss = self.loss(y_hat, y)

        self.test_acc(y_hat, y)
        self.test_auroc(y_hat, y)
        self.test_aupr(y_hat, y.long())
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_f1(y_hat, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat, y = self._process_batch(batch)

        return {
            "predicted": y_hat.squeeze().cpu().detach(),
            "target": y.cpu().detach().float(),
        }


class LitModel_Sahu_binary_mprann(LitModel_Sahu_binary_legnet):
        def __init__(
            self, weight_decay, lr, model=None, loss=nn.BCEWithLogitsLoss(), print_each=1
        ):
            super().__init__(
                model=model,
                loss=loss,
                print_each=print_each,
                weight_decay=weight_decay,
                lr=lr,
            )
            self.output_dim = 1

            self.fc2 = nn.Linear(300 *2, self.output_dim)

            self.activation = nn.Sigmoid()

            self.head = nn.Sequential(
                nn.Linear(300 *2, self.output_dim),
                nn.Sigmoid()
            )

        """def head(self, seq):
            seq = self.fc2(seq)
            seq = self.activation(seq)
            return seq"""
        
class GroupedLinear(nn.Module):
    def __init__(self, in_group_size, out_group_size, groups):
        super().__init__()

        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.groups = groups

        # initialize weights
        self.weight = torch.nn.Parameter(
            torch.zeros(groups, in_group_size, out_group_size)
        )
        self.bias = torch.nn.Parameter(torch.zeros(groups, 1, out_group_size))

        # change weights to kaiming
        self.reset_parameters(self.weight, self.bias)

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        reorg = (
            x.permute(1, 0)
            .reshape(self.groups, self.in_group_size, -1)
            .permute(0, 2, 1)
        )
        hook = torch.bmm(reorg, self.weight) + self.bias
        reorg = (
            hook.permute(0, 2, 1)
            .reshape(self.out_group_size * self.groups, -1)
            .permute(1, 0)
        )

        return reorg
    
class LitModel_Sahu_binary_malinois(LitModel_Sahu_binary_legnet):
        def __init__(
            self, weight_decay, lr, model=None, loss=nn.BCEWithLogitsLoss(), print_each=1
        ):
            super().__init__(
                model=model,
                loss=loss,
                print_each=print_each,
                weight_decay=weight_decay,
                lr=lr,
            )

            branched_channels = 140
            n_outputs = 1

            self.output = GroupedLinear(branched_channels*2, 1, n_outputs)
            
        def head(self, seq):
            seq = self.output(seq)
            return seq
        def _process_batch(self, batch):
            seqs, labels = batch
            enhancer = self.model(seqs["seq_enh"])
            promoter = self.model(seqs["seq"])

            concat = torch.cat([enhancer, promoter], dim=1)

            out = self.head(concat).squeeze(-1)
            return out, labels
        
class LitModel_Sahu_binary_parm(LitModel_Sahu_binary_legnet):
        def __init__(
            self, weight_decay, lr, model=None, loss=nn.BCEWithLogitsLoss(), print_each=1
        ):
            super().__init__(
                model=model,
                loss=loss,
                print_each=print_each,
                weight_decay=weight_decay,
                lr=lr,
            )
            filter_size = 125
            output_dim = 1

            self.linear1 = nn.Linear(filter_size*2, output_dim)
            
        def head(self, seq):
            seq = self.linear1(seq)
            return seq
        
        def _process_batch(self, batch):
            seqs, labels = batch
            enhancer = self.model(seqs["seq_enh"])
            promoter = self.model(seqs["seq"])

            concat = torch.cat([enhancer, promoter], dim=1)

            out = self.head(concat).squeeze(1)
            return out, labels

class LitModel_Sahu_binary_dream_rnn(LitModel_Sahu_binary_legnet):
    def __init__(
        self, weight_decay, lr, model=None, loss=nn.BCEWithLogitsLoss(), print_each=1
    ):
        super().__init__(
            model=model,
            loss=loss,
            print_each=print_each,
            weight_decay=weight_decay,
            lr=lr,
        )
        out_ch = 256  # размер эмбеддинга из AutosomeFinalLayersBlockBinary
        output_dim = 1

        self.linear1 = nn.Linear(out_ch * 2, output_dim)

    def head(self, seq):
        seq = self.linear1(seq)
        return seq

    def _process_batch(self, batch):
        seqs, labels = batch
        enhancer = self.model(seqs["seq_enh"])
        promoter = self.model(seqs["seq"])

        concat = torch.cat([enhancer, promoter], dim=1)

        out = self.head(concat).squeeze(-1)
        return out, labels
