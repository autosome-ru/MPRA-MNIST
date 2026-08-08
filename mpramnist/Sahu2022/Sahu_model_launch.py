import torch
import torch.nn as nn
import os
import pandas as pd

from mpramnist.Sahu2022 import SahuDataset
from mpramnist.Sahu2022 import LitModel_Sahu, LitModel_Sahu_binary_legnet, LitModel_Sahu_binary_mprann,LitModel_Sahu_binary_malinois, LitModel_Sahu_binary_parm, LitModel_Sahu_binary_dream_rnn

from mpramnist.models import HumanLegNet
from mpramnist.models import initialize_weights

from mpramnist.models import BassetBranched
from mpramnist.models import L1KLmixed

from mpramnist.models import MPRAnn

from mpramnist.models import PARM

from mpramnist.models import DREAM_RNN
from mpramnist.models.DREAM_RNN import AutosomeFinalLayersBlock, BHICoreBlock, BHIFirstLayersBlock, PrixFixeNet

import mpramnist.transforms as t

from torch.utils.data import DataLoader

import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.nn import functional as F

import argparse 
parser = argparse.ArgumentParser()

general = parser.add_argument_group('general args', 
                                    'general_argumens')

general.add_argument("--result_dir",
                     type=str,
                     default = "./sahu.tsv")
general.add_argument("--device", 
                     type=int,
                     default=0)
general.add_argument("--num_workers",
                     type=int, 
                     default=8)
general.add_argument("--batch_size",
                     type=int, 
                     default=1024)
general.add_argument("--runs",
                     type=int, 
                     default=5)
general.add_argument("--model",
                     type=str, 
                     default="MPRALegNet") # or Malinois/MPRAnn/PARM

dataset_args =  parser.add_argument_group('dataset args', 
                                'dataset arguments')

dataset_args.add_argument("--root", 
                     type=str, 
                     default="../data/")
dataset_args.add_argument("--task", 
                     nargs='+',            # accepts one or more values
                     default=["RandomEnhancer", "GenomicPromoter", "CapturePromoter", "GenomicEnhancer", "AtacSeq", "Binary",],
                     help="List of available tasks")

trainer_args = parser.add_argument_group('trainer args', 
                                'trainer arguments')

trainer_args.add_argument("--lr",
                     type=float,
                     default=1e-3)
trainer_args.add_argument("--wd",
                     type=float,
                     default=1e-1)
trainer_args.add_argument("--epoch_num",
                            type=int,
                            default=10)

args = parser.parse_args()

if isinstance(args.task, str):
    args.task = [args.task]

if os.path.exists(args.result_dir):
    results = pd.read_csv(args.result_dir, sep = "\t")
else:
    results = pd.DataFrame(columns = args.task)

for run in list(range(args.runs)):
    res = []
    for task in args.task:
        train_transform = t.Compose([t.Seq2Tensor()])
        val_test_transform = t.Compose([t.Seq2Tensor()])

        train_dataset = SahuDataset(split="train",task=task,transform=train_transform,root=args.root)
        val_dataset = SahuDataset(split="val",task=task,transform=val_test_transform,root=args.root)
        test_dataset = SahuDataset(split="test",task=task,transform=val_test_transform,root=args.root)

        # encapsulate data into dataloader form
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if args.model == "MPRALegNet" and task.lower() != "binary":
            model = HumanLegNet(
                in_ch=4,
                output_dim=1,
                stem_ch=64,
                stem_ks=11,
                ef_ks=9,
                ef_block_sizes=[32, 64, 128, 128, 256, 512, 256],
                pool_sizes=[1, 2, 1, 2, 1, 2, 1],
                resize_factor=4)
            model.apply(initialize_weights)
            loss =torch.nn.BCEWithLogitsLoss()
            seq_model = LitModel_Sahu(model=model, loss=loss, weight_decay=args.wd, lr=args.lr, print_each=1)
        elif args.model == "MPRAnn" and task.lower() != "binary":
            model = MPRAnn(output_dim=1)
            loss = torch.nn.BCEWithLogitsLoss()
            seq_model = LitModel_Sahu(model=model, loss=loss, weight_decay=args.wd, lr=args.lr, print_each=1)
        elif args.model == "Malinois" and task.lower() != "binary":
            length = len(train_dataset[0][0][0])
            model = BassetBranched(input_len=length, n_outputs=1)
            loss = torch.nn.BCEWithLogitsLoss()
            seq_model = LitModel_Sahu(model=model, loss=loss, weight_decay=args.wd, lr=args.lr, print_each=1)
        elif args.model == "PARM" and task.lower() != "binary":
            model = PARM(n_block=5, type_loss="mse", output_dim=1)
            loss = torch.nn.BCEWithLogitsLoss()
            seq_model = LitModel_Sahu(model=model, loss=loss, weight_decay=args.wd, lr=args.lr, print_each=1)
        elif args.model == "DREAM-RNN" or args.model == "DREAM_RNN" and task.lower() != "binary":
            length = len(train_dataset[0][0][0])
            model = DREAM_RNN(4, length, 1)
            loss = torch.nn.BCEWithLogitsLoss()
            seq_model = LitModel_Sahu(model=model, loss=loss, weight_decay=args.wd, lr=args.lr, print_each=1)
        elif args.model == "MPRALegNet" and task.lower() == "binary":
            class HumanLegNetBinary(HumanLegNet):
                def __init__(
                    self,
                ):
                    super().__init__(
                        in_ch=4,
                        output_dim=1,
                        stem_ch=64,
                        stem_ks=11,
                        ef_ks=9,
                        ef_block_sizes=[80, 96, 112, 128],
                        pool_sizes=[2, 2, 2, 2],
                        resize_factor=4,
                        activation=nn.SiLU,
                    )

                def forward(self, x):
                    x = self.stem(x)
                    x = self.main(x)
                    x = self.mapper(x)
                    x = F.adaptive_avg_pool1d(x, 1)
                    x = x.squeeze(-1)  # without head
                    # x = self.head(x)
                    # x = x.squeeze(-1)
                    # head will be used in trainer because paired task has paired sequences
                    return x
            model = HumanLegNetBinary()
            model.apply(initialize_weights)
            loss =torch.nn.BCEWithLogitsLoss()
            seq_model = LitModel_Sahu_binary_legnet(model=model, loss=loss, weight_decay=args.wd, lr=args.lr, print_each=1)
        elif args.model == "MPRAnn" and task.lower() == "binary":
            class MPRAnnBinary(MPRAnn):
                def __init__(
                    self,
                ):
                    super().__init__(
                        output_dim=1
                    )
                def forward(self, x):
                    seq = self.conv1(x)
                    seq = F.relu(seq)
                    seq = self.bn1(seq)
                    seq = self.conv2(seq)
                    seq = F.softmax(seq, dim=1)
                    seq = self.bn2(seq)
                    seq = self.pool1(seq)
                    seq = self.dropout1(seq)
                    seq = self.conv3(seq)
                    seq = F.softmax(seq, dim=1)
                    seq = self.bn3(seq)
                    seq = self.conv4(seq)
                    seq = F.softmax(seq, dim=1)
                    seq = self.bn4(seq)
                    seq = self.global_pool(seq) 
                    seq = seq.squeeze(-1)  
                    seq = self.dropout2(seq)
                    seq = seq.reshape((seq.shape[0], -1))
                    seq = self.fc1(seq)
                    seq = F.sigmoid(seq)
                    seq = self.dropout3(seq)
                    #seq = self.fc2(seq)
                    #seq = F.sigmoid(seq)
                    #seq = seq.squeeze(-1)
                    return seq
            model = MPRAnnBinary()
            loss = torch.nn.BCEWithLogitsLoss()
            seq_model = LitModel_Sahu_binary_mprann(model=model, loss=loss, weight_decay=args.wd, lr=args.lr, print_each=1)
        elif args.model == "Malinois" and task.lower() == "binary":
            length = len(train_dataset[0][0]["seq"][0])
            class MalinoisBinary(BassetBranched):
                def __init__(
                    self,
                    input_len,
                    n_outputs
                ):
                    super().__init__(
                        input_len=input_len,
                        conv1_channels=300,
                        conv1_kernel_size=19,
                        conv2_channels=200,
                        conv2_kernel_size=11,
                        conv3_channels=200,
                        conv3_kernel_size=7,
                        n_linear_layers=1,
                        linear_channels=1000,
                        linear_activation="ReLU",
                        linear_dropout_p=0.11625456877954289,
                        n_branched_layers=3,
                        branched_channels=140,
                        branched_activation="ReLU",
                        branched_dropout_p=0.5757068086404574,
                        n_outputs=n_outputs,
                        use_batch_norm=True,
                        use_weight_norm=False,
                        loss_criterion="L1KLmixed",
                        loss_args={},
                    )

                def forward(self, x):
                    encoded = self.encode(x)
                    decoded = self.decode(encoded)
                    #output = self.classify(decoded)
                    #output = output.squeeze(-1)
                    return decoded
                
            model = MalinoisBinary(input_len=length, n_outputs=1)
            loss = torch.nn.BCEWithLogitsLoss()
            seq_model = LitModel_Sahu_binary_malinois(model=model, loss=loss, weight_decay=args.wd, lr=args.lr, print_each=1)
        elif args.model == "PARM" and task.lower() == "binary":
            class PARMBinary(PARM):

                def __init__(self, n_block, type_loss, output_dim):

                    super().__init__(n_block=n_block, filter_size=125, output_dim=output_dim, weight_file=None, 
                            cell_line=False,
                            type_loss=type_loss, validation=False, index_interested_output=False, maxglobalpool=True,
                            vocab=4, use_AttentionPool=True)

                def forward(self, x):
                    out = self.stem(x)
                    out = self.conv_tower(out)
                    if self.maxglobalpool:
                        #max in length
                        out = torch.max(out, dim=-1).values
                    out = out.view(out.size(0), -1)
                    #out = self.linear1(out)
                    return out
                
            model = PARMBinary(n_block=5, type_loss="mse", output_dim=1)
            loss = torch.nn.BCEWithLogitsLoss()
            seq_model = LitModel_Sahu_binary_parm(model=model, loss=loss, weight_decay=args.wd, lr=args.lr, print_each=1)
        elif args.model == "DREAM-RNN" or args.model == "DREAM_RNN" and task.lower() == "binary":
            class AutosomeFinalLayersBlockBinary(AutosomeFinalLayersBlock):
                def forward(self, x):
                    x = self.mapper(x)
                    x = F.adaptive_avg_pool1d(x, 1)
                    x = x.squeeze(2)
                    return x
            def DREAM_RNN_Binary(in_channels, seqsize):
                first = BHIFirstLayersBlock(
                    in_channels=in_channels,
                    out_channels=320,
                    seqsize=seqsize,
                    kernel_sizes=[9, 15],
                    pool_size=1,
                    dropout=0.2
                )

                core = BHICoreBlock(
                    in_channels=first.out_channels,
                    out_channels=320,
                    seqsize=first.infer_outseqsize(),
                    lstm_hidden_channels=320,
                    kernel_sizes=[9, 15],
                    pool_size=1,
                    dropout1=0.2,
                    dropout2=0.5
                )

                # out_channels здесь не используется (linear отброшен),
                # но параметр нужен конструктору AutosomeFinalLayersBlock
                final = AutosomeFinalLayersBlockBinary(
                    in_channels=core.out_channels,
                    seqsize=core.infer_outseqsize(),
                    out_channels=1
                )

                generator = torch.Generator()

                model = PrixFixeNet(
                    first=first,
                    core=core,
                    final=final,
                    generator=generator
                )
                return model
                
            length = len(train_dataset[0][0]["seq"][0])
            model = DREAM_RNN_Binary(in_channels=4, seqsize=length)
            loss = torch.nn.BCEWithLogitsLoss()
            seq_model = LitModel_Sahu_binary_dream_rnn(model=model, loss=loss, weight_decay=args.wd, lr=args.lr, print_each=1)

        checkpoint_callback = ModelCheckpoint(monitor="val_aupr", mode="max", save_top_k=1, save_last=False)

        # Initialize a trainer
        trainer = L.Trainer(
            accelerator="gpu",
            devices=[args.device],
            max_epochs=args.epoch_num,
            gradient_clip_val=1,
            precision="16-mixed",
            enable_progress_bar=True,
            num_sanity_val_steps=0,
            callbacks=[checkpoint_callback],
        )

        # Train the model
        trainer.fit(seq_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_model_path = checkpoint_callback.best_model_path
        if task.lower() != "binary":
            seq_model = LitModel_Sahu.load_from_checkpoint(best_model_path,model=model, loss=torch.nn.BCEWithLogitsLoss(), weight_decay=args.wd, lr=args.lr, print_each=1)
        else:
            if args.model == "MPRALegNet":
                seq_model = LitModel_Sahu_binary_legnet.load_from_checkpoint(best_model_path, model=model, loss=torch.nn.BCEWithLogitsLoss(), weight_decay=args.wd, lr=args.lr, print_each=1)
            elif args.model == "MPRAnn":
                seq_model = LitModel_Sahu_binary_mprann.load_from_checkpoint(best_model_path, model=model, loss=torch.nn.BCEWithLogitsLoss(), weight_decay=args.wd, lr=args.lr, print_each=1)
            elif args.model == "Malinois":
                seq_model = LitModel_Sahu_binary_malinois.load_from_checkpoint(best_model_path, model=model, loss=torch.nn.BCEWithLogitsLoss(), weight_decay=args.wd, lr=args.lr, print_each=1)
            elif args.model == "PARM":
                seq_model = LitModel_Sahu_binary_parm.load_from_checkpoint(best_model_path, model=model, loss=torch.nn.BCEWithLogitsLoss(), weight_decay=args.wd, lr=args.lr, print_each=1)
            elif args.model == "DREAM-RNN" or args.model == "DREAM_RNN":
                seq_model = LitModel_Sahu_binary_dream_rnn.load_from_checkpoint(best_model_path, model=model, loss=torch.nn.BCEWithLogitsLoss(), weight_decay=args.wd, lr=args.lr, print_each=1)

        result = trainer.test(seq_model, dataloaders=test_loader)
        res.append(result[0]["test_aupr"])
        
    results.loc[len(results)] = res

    results.to_csv(args.result_dir, sep = "\t", index = False)