import torch
import torch.nn as nn
import os
import pandas as pd

from mpramnist.deAlmeida2022.dataset import deAlmeidaDataset
from mpramnist.deAlmeida2022.trainer import LitModel_deAlmeida

from mpramnist.models import HumanLegNet
from mpramnist.models import initialize_weights

from mpramnist.models import BassetBranched
from mpramnist.models import L1KLmixed

from mpramnist.models import MPRAnn

from mpramnist.models import PARM

from mpramnist.models import DeepStarr

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
                     default = "./deAlmeida.tsv")
general.add_argument("--device", 
                     type=int,
                     default=0)
general.add_argument("--num_workers",
                     type=int, 
                     default=103)
general.add_argument("--runs",
                     type=int, 
                     default=5)
general.add_argument("--model",
                     type=str, 
                     default="MPRALegNet") # or Malinois/MPRAnn/PARM/DeepStarr

dataset_args =  parser.add_argument_group('dataset args', 
                                'dataset arguments')

dataset_args.add_argument("--root", 
                     type=str, 
                     default="../data/")
dataset_args.add_argument("--promoter_types",
                     nargs='+',            # accepts one or more values
                     default=["Dev_log2", "Hk_log2"],
                     help="list of promoter types")

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

if isinstance(args.promoter_types, str):
    args.promoter_types = [args.promoter_types]

if os.path.exists(args.result_dir):
    results = pd.read_csv(args.result_dir, sep = "\t")
else:
    results = pd.DataFrame(columns = args.promoter_types)


for run in list(range(args.runs)):
    print(args.promoter_types)
    # Read the MPRAdata, preprocess them and encapsulate them into dataloader form.
    train_transform = t.Compose([t.ReverseComplement(0.5),t.Seq2Tensor(),])
    val_test_transform = t.Compose([t.ReverseComplement(0),t.Seq2Tensor(), ])

    train_dataset = deAlmeidaDataset(cell_type=args.promoter_types,use_original_reverse_complement=False,split="train",transform=train_transform,root=args.root)
    val_dataset = deAlmeidaDataset(cell_type=args.promoter_types,split="val",transform=val_test_transform,root=args.root)
    test_dataset = deAlmeidaDataset(cell_type=args.promoter_types,split="test",transform=val_test_transform,root=args.root)

    # encapsulate data into dataloader form
    train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1024, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False, num_workers=args.num_workers)

    if args.model == "MPRALegNet":
        model = HumanLegNet(
            in_ch=len(train_dataset[0][0]),
            output_dim=len(args.promoter_types),
            stem_ch=64,
            stem_ks=11,
            ef_ks=9,
            ef_block_sizes=[80, 96, 112, 128],
            pool_sizes=[2, 2, 2, 2],
            resize_factor=4)
        model.apply(initialize_weights)
        loss =nn.MSELoss()
    elif args.model == "MPRAnn":
        model = MPRAnn(output_dim=len(args.promoter_types))
        loss = nn.MSELoss()
    elif args.model == "Malinois":
        length = len(train_dataset[0][0][0])
        model = BassetBranched(input_len=length, n_outputs=len(args.promoter_types))
        loss = nn.MSELoss()
    elif args.model == "PARM":
        model = PARM(n_block=5, type_loss="mse", output_dim=len(args.promoter_types))
        loss = nn.MSELoss()
    elif args.model == "DeepStarr":
        model = DeepStarr(len(args.promoter_types))
        loss = nn.MSELoss()
    elif args.model =="DREAM-RNN" or args.model == "DREAM_RNN":
            length = len(train_dataset[0][0][0])
            model = DREAM_RNN(in_channels=len(train_dataset[0][0]), seqsize=length, out_channels=len(args.promoter_types))
            loss = nn.MSELoss()

    seq_model = LitModel_deAlmeida(model=model, loss=loss, weight_decay=args.wd, lr=args.lr, cell_types=args.promoter_types, print_each=1)
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
    seq_model = LitModel_deAlmeida.load_from_checkpoint(best_model_path,model=model, loss=nn.MSELoss(), weight_decay=args.wd, lr=args.lr, cell_types=args.promoter_types, print_each=1)

    forw_transform = t.Compose([t.Seq2Tensor()])
    rev_transform = t.Compose([t.ReverseComplement(1),t.Seq2Tensor(),])

    test_forw = deAlmeidaDataset(cell_type=args.promoter_types,split="test",transform=forw_transform,root=args.root,)
    test_rev = deAlmeidaDataset(cell_type=args.promoter_types,split="test",transform=rev_transform,root=args.root,)

    forw = DataLoader(dataset=test_forw,batch_size=1024,shuffle=False,num_workers=args.num_workers,pin_memory=True,)
    rev = DataLoader(dataset=test_rev,batch_size=1024,shuffle=False,num_workers=args.num_workers,pin_memory=True,)

    predictions_forw = trainer.predict(seq_model, dataloaders=forw)
    targets = torch.cat([pred["target"] for pred in predictions_forw])
    y_preds_forw = torch.cat([pred["predicted"] for pred in predictions_forw])

    predictions_rev = trainer.predict(seq_model, dataloaders=rev)
    y_preds_rev = torch.cat([pred["predicted"] for pred in predictions_rev])

    mean_forw = torch.mean(torch.stack([y_preds_forw, y_preds_rev]), dim=0)

    pears = PearsonCorrCoef(num_outputs=len(args.promoter_types))
    print(args.promoter_types, " Pearson correlation")

    corr_pearson = pears(mean_forw, targets)
    
    print(corr_pearson.numpy())
    results.loc[len(results)] = corr_pearson.numpy()

    results.to_csv(args.result_dir, sep = "\t", index = False)