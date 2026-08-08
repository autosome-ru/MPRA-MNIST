import torch
import torch.nn as nn
import os
import pandas as pd

from mpramnist.Vaishnav2024 import VaishnavDataset
from mpramnist.Vaishnav2024 import LitModel_Vaishnav

from mpramnist.models import HumanLegNet
from mpramnist.models import initialize_weights

from mpramnist.models import BassetBranched
from mpramnist.models import L1KLmixed

from mpramnist.models import MPRAnn

from mpramnist.models import PARM

from mpramnist.models import DREAM_RNN
import mpramnist.transforms as t

from torch.utils.data import DataLoader
from torchmetrics import PearsonCorrCoef

import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint

import argparse 
parser = argparse.ArgumentParser()

general = parser.add_argument_group('general args', 
                                    'general_argumens')

general.add_argument("--result_dir",
                     type=str,
                     default = "./vaishnav.tsv")
general.add_argument("--device", 
                     type=int,
                     default=0)
general.add_argument("--num_workers",
                     type=int, 
                     default=16)
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
dataset_args.add_argument("--data_env_type", 
                     nargs='+',            # accepts one or more values
                     default=["defined", "complex"],
                     help="List of dataset environment types")

trainer_args =  parser.add_argument_group('trainer args', 
                                'trainer arguments')

trainer_args.add_argument("--lr",
                     type=float,
                     default=1e-2)
trainer_args.add_argument("--wd",
                     type=float,
                     default=1e-2)
trainer_args.add_argument("--epoch_num",
                            type=int,
                            default=5)

args = parser.parse_args()

if isinstance(args.data_env_type, str):
    args.data_env_type = [args.data_env_type]

if os.path.exists(args.result_dir):
    results = pd.read_csv(args.result_dir, sep = "\t")
else:
    columns = []
    for i in args.data_env_type:
        for type in ["native","drift","paired"]:
            columns.append(i+"_"+type)
    results = pd.DataFrame(columns = columns)

length = 110
plasmid = VaishnavDataset.PLASMID.upper()
insert_start = plasmid.find("N" * 80)
right_flank = VaishnavDataset.RIGHT_FLANK
left_flank = plasmid[insert_start - length : insert_start]

# preprocessing
train_transform = t.Compose([t.AddFlanks(left_flank, right_flank),t.LeftCrop(length, length),t.ReverseComplement(0.5),t.Seq2Tensor(),])
val_test_transform = t.Compose([t.AddFlanks(left_flank, right_flank),t.LeftCrop(length, length),t.ReverseComplement(0),t.Seq2Tensor(),])

forw_transform = t.Compose([t.AddFlanks(left_flank, right_flank), t.LeftCrop(length, length), t.Seq2Tensor()])
rev_transform = t.Compose([t.AddFlanks(left_flank, right_flank),t.LeftCrop(length, length),t.ReverseComplement(1),t.Seq2Tensor(),])

def meaned_prediction(forw, rev, trainer, seq_model, name, is_paired=False):
    predictions_forw = trainer.predict(seq_model, dataloaders=forw)
    targets = torch.cat([pred["target"] for pred in predictions_forw])
    y_preds_forw = torch.cat([pred["ref_predicted"] for pred in predictions_forw])

    predictions_rev = trainer.predict(seq_model, dataloaders=rev)
    y_preds_rev = torch.cat([pred["ref_predicted"] for pred in predictions_rev])

    mean_forw = torch.mean(torch.stack([y_preds_forw, y_preds_rev]), dim=0)

    pears = PearsonCorrCoef()
    print(name + " Pearson correlation")

    if is_paired:
        y_preds_forw_alt = torch.cat(
            [pred["alt_predicted"] for pred in predictions_forw]
        )
        y_preds_rev_alt = torch.cat([pred["alt_predicted"] for pred in predictions_rev])
        mean_alt = torch.mean(torch.stack([y_preds_forw_alt, y_preds_rev_alt]), dim=0)
        pred = mean_alt - mean_forw
        return pears(pred, targets)

    return pears(mean_forw, targets)

for run in list(range(args.runs)):
    r_array = []
    for env in args.data_env_type:
        train_dataset = VaishnavDataset(split="train",dataset_env_type=env,transform=train_transform,root=args.root)
        val_dataset = VaishnavDataset(split="val",dataset_env_type=env,transform=val_test_transform,root=args.root)

        # encapsulate data into dataloader form
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        if args.model == "MPRALegNet":
            model = HumanLegNet(
                in_ch=len(train_dataset[0][0]),
                output_dim=1,
                stem_ch=64,
                stem_ks=11,
                ef_ks=9,
                ef_block_sizes=[80, 96, 112, 128],
                pool_sizes=[2, 2, 2, 2],
                resize_factor=4)
            model.apply(initialize_weights)
            loss =nn.MSELoss()
        elif args.model == "MPRAnn":
            model = MPRAnn(output_dim=1)
            loss = nn.MSELoss()
        elif args.model == "Malinois":
            length = len(train_dataset[0][0][0])
            model = BassetBranched(input_len=length, n_outputs=1)
            loss = nn.MSELoss()
        elif args.model == "PARM":
            model = PARM(n_block=5, type_loss="mse", output_dim=1)
            loss = nn.MSELoss()
        elif args.model =="DREAM-RNN" or args.model == "DREAM_RNN":
            length = len(train_dataset[0][0][0])
            model = DREAM_RNN(in_channels=len(train_dataset[0][0]), seqsize=length, out_channels=1)
            loss = nn.MSELoss()

        seq_model = LitModel_Vaishnav(model=model, loss=loss, weight_decay=args.wd, lr=args.lr, print_each=1)
        checkpoint_callback = ModelCheckpoint(monitor="val_pearson", mode="max", save_top_k=1, save_last=False)

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
        seq_model = LitModel_Vaishnav.load_from_checkpoint(best_model_path,model=model, loss=nn.MSELoss(), weight_decay=args.wd, lr=args.lr, print_each=1)

        for type in ["native","drift","paired"]:

            paired = True if type=="paired" else False
            test_forw = VaishnavDataset(split="test", dataset_env_type=env, test_dataset_type=type, transform=forw_transform,root=args.root,)
            test_rev = VaishnavDataset(split="test", dataset_env_type=env, test_dataset_type=type, transform=rev_transform,root=args.root,)

            forw = DataLoader(dataset=test_forw,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True,)
            rev = DataLoader(dataset=test_rev,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True,)

            r = meaned_prediction(forw, rev, trainer, seq_model, type, paired).numpy()
            r_array.append(r)
            print(r)
    
    results.loc[len(results)] = r_array

    results.to_csv(args.result_dir, sep = "\t", index = False)