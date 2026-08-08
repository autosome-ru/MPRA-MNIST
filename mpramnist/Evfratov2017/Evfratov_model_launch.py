import torch
import torch.nn as nn
import os
import pandas as pd

from mpramnist.Evfratov2017 import EvfratovDataset
from mpramnist.Evfratov2017 import LitModel_Evfratov

from mpramnist.models import HumanLegNet
from mpramnist.models import initialize_weights

from mpramnist.models import BassetBranched
from mpramnist.models import L1KLmixed

from mpramnist.models import MPRAnn

from mpramnist.models import PARM

from mpramnist.models import DREAM_RNN
import mpramnist.transforms as t

from torch.utils.data import DataLoader

import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint

import argparse 
parser = argparse.ArgumentParser()

general = parser.add_argument_group('general args', 
                                    'general_argumens')

general.add_argument("--result_dir",
                     type=str,
                     default = "./evfratov.tsv")
general.add_argument("--device", 
                     type=int,
                     default=0)
general.add_argument("--num_workers",
                     type=int, 
                     default=16)
general.add_argument("--batch_size",
                     type=int, 
                     default=32)
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
dataset_args.add_argument("--length",
                     nargs='+',            # accepts one or more values
                     default=["23", "33"],
                     help="Length of sequence")

trainer_args =  parser.add_argument_group('trainer args', 
                                'trainer arguments')

trainer_args.add_argument("--lr",
                     type=float,
                     default=0.01)
trainer_args.add_argument("--wd",
                     type=float,
                     default=0.1)
trainer_args.add_argument("--epoch_num",
                            type=int,
                            default=50)


args = parser.parse_args()

if isinstance(args.length, str):
    args.length = [args.length]

if os.path.exists(args.result_dir):
    results = pd.read_csv(args.result_dir, sep = "\t")
else:
    length = [l+"_f1" for l in args.length]
    results = pd.DataFrame(columns = length)

# preprocessing
train_transform = t.Compose([t.ReverseComplement(0.5),t.Seq2Tensor(),])
test_transform = t.Compose([t.Seq2Tensor(),t.ReverseComplement(0),])
if args.model == "Malinois":
    train_transform = t.Compose([t.Padding(200),t.ReverseComplement(0.5),t.Seq2Tensor(),])
    test_transform = t.Compose([t.Padding(200),t.Seq2Tensor(),t.ReverseComplement(0),])

for run in list(range(args.runs)):
    res = []
    for length in args.length:

        merge_last_classes = True
        # Read the MPRAdata, preprocess them and encapsulate them into dataloader form.

        train_dataset = EvfratovDataset(split="train",merge_last_classes=merge_last_classes,length_of_seq=length,transform=train_transform,root=args.root)
        val_dataset = EvfratovDataset(split="val",merge_last_classes=merge_last_classes,length_of_seq=length,transform=test_transform,root=args.root)
        test_dataset = EvfratovDataset(split="test",merge_last_classes=merge_last_classes,length_of_seq=length,transform=test_transform,root=args.root)

        # encapsulate data into dataloader form
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        N_CLASSES = train_dataset.n_classes
        if args.model == "MPRALegNet":
            model = HumanLegNet(
                in_ch=len(train_dataset[0][0]),
                output_dim=N_CLASSES,
                stem_ch=64,
                stem_ks=11,
                ef_ks=9,
                ef_block_sizes=[80, 96, 112, 128],
                pool_sizes=[2, 2, 2, 2],
                resize_factor=4)
            model.apply(initialize_weights)
        elif args.model == "MPRAnn":
            model = MPRAnn(output_dim=N_CLASSES)
        elif args.model == "Malinois":
            length = len(train_dataset[0][0][0])
            model = BassetBranched(input_len=length, n_outputs=N_CLASSES)
        elif args.model == "PARM":
            model = PARM(n_block=5, type_loss="mse", output_dim=N_CLASSES)
        elif args.model =="DREAM-RNN" or args.model == "DREAM_RNN":
            model = DREAM_RNN(len(train_dataset[0][0]), len(train_dataset[0][0][0]), N_CLASSES)

        seq_model = LitModel_Evfratov(model=model,loss=nn.CrossEntropyLoss(),n_classes=N_CLASSES,weight_decay=args.wd, lr=args.lr, print_each=1)

        checkpoint_callback = ModelCheckpoint(monitor="val_auroc", mode="max", save_top_k=1, save_last=False)

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
        seq_model = LitModel_Evfratov.load_from_checkpoint(best_model_path,model=model,loss=nn.CrossEntropyLoss(),n_classes=N_CLASSES,weight_decay=args.wd, lr=args.lr, print_each=1)
        result = trainer.test(seq_model, dataloaders=test_loader)
        res.append(result[0]["test_f1"])

    results.loc[len(results)] = res

    results.to_csv(args.result_dir, sep = "\t", index = False)