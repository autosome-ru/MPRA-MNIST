import torch
import torch.nn as nn
import os
import pandas as pd

from mpramnist.BarbadillaMartinez2026 import BarbadillaMartinezDataset
from mpramnist.BarbadillaMartinez2026 import LitModel_BarbadillaMartinez

from mpramnist.models import HumanLegNet
from mpramnist.models import initialize_weights

from mpramnist.models import BassetBranched
from mpramnist.models import L1KLmixed

from mpramnist.models import MPRAnn

from mpramnist.models import PARM

from mpramnist.models import DREAM_RNN
import mpramnist.transforms as t

from torch.utils.data import DataLoader
from scipy.stats import pearsonr

import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint


import argparse 
parser = argparse.ArgumentParser()

general = parser.add_argument_group('general args', 
                                    'general_argumens')

general.add_argument("--result_dir",
                     type=str,
                     default = "./barbadillamartinez.tsv")
general.add_argument("--device", 
                     type=int,
                     default=0)
general.add_argument("--num_workers",
                     type=int, 
                     default=103)
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
dataset_args.add_argument("--genomes",
                     type = str,
                     default="all",
                     help="which genomes should be used, applicable for genomewide libraries only")
dataset_args.add_argument("--cell_types",
                     nargs='+',            # accepts one or more values
                     default=['HepG2','K562', 'MCF7', 'U2OS', 'HCT116', 'HEK293','LNCaP'],
                     help="List of cell_types")

trainer_args =  parser.add_argument_group('trainer args', 
                                'trainer arguments')

trainer_args.add_argument("--lr",
                     type=float,
                     default=0.005)
trainer_args.add_argument("--wd",
                     type=float,
                     default=2e-1)
trainer_args.add_argument("--epoch_num",
                            type=int,
                            default=50)

args = parser.parse_args()

if isinstance(args.cell_types, str):
    args.cell_types = [args.cell_types]

if os.path.exists(args.result_dir):
    results = pd.read_csv(args.result_dir, sep = "\t")
else:
    results = pd.DataFrame(columns = args.cell_types)

train_transform = t.Compose(
    [
        t.AddFlanks(BarbadillaMartinezDataset.LEFT_FLANK, BarbadillaMartinezDataset.RIGHT_FLANK),
        t.RandomPadding(600),
        t.RightCrop(600, 600),
        t.Seq2Tensor(),
    ]
)
test_transform = t.Compose(
    [
        t.AddFlanks(BarbadillaMartinezDataset.LEFT_FLANK, BarbadillaMartinezDataset.RIGHT_FLANK),
        t.Padding(600),
        t.RightCrop(600, 600),
        t.Seq2Tensor(),
    ]
)

for run in list(range(args.runs)):
    print(args.cell_types)
    pears = []

    # load the data
    train_dataset = BarbadillaMartinezDataset(split=[0,1,2,3], transform=train_transform, genomes=args.genomes, cell_type=args.cell_types, root=args.root)
    val_dataset = BarbadillaMartinezDataset(split=[4], transform=test_transform, genomes=args.genomes, cell_type=args.cell_types, root=args.root)
    test_dataset = BarbadillaMartinezDataset(split='test', transform=test_transform, genomes=args.genomes, cell_type=args.cell_types, root=args.root)

    # encapsulate data into dataloader form
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.model == "MPRALegNet":
        model = HumanLegNet(
            in_ch=len(train_dataset[0][0]),
            output_dim=len(args.cell_types),
            stem_ch=64,
            stem_ks=11,
            ef_ks=9,
            ef_block_sizes=[80, 96, 112, 128],
            pool_sizes=[2, 2, 2, 2],
            resize_factor=4)
        model.apply(initialize_weights)
        loss =nn.MSELoss()
    elif args.model == "MPRAnn":
        model = MPRAnn(output_dim=len(args.cell_types))
        loss =nn.MSELoss()
    elif args.model == "Malinois":
        model = BassetBranched(input_len=600, n_outputs=len(args.cell_types))
        loss =nn.MSELoss()
    elif args.model == "PARM":
        model = PARM(n_block=5, type_loss="mse", output_dim=len(args.cell_types))
        loss =nn.MSELoss()
    elif args.model =="DREAM-RNN" or args.model == "DREAM_RNN":
        length = 600
        model = DREAM_RNN(in_channels=len(train_dataset[0][0]), seqsize=length, out_channels=len(args.cell_types))
        loss = nn.MSELoss()

    seq_model = LitModel_BarbadillaMartinez(model=model, 
                                            cell_types=args.cell_types,
                                            loss=nn.MSELoss(),
                                            weight_decay=args.wd, 
                                            lr=args.lr,
                                            print_each=1,)

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

    # Validate best model
    best_model_path = checkpoint_callback.best_model_path
    seq_model = LitModel_BarbadillaMartinez.load_from_checkpoint(best_model_path, model=model, cell_types=args.cell_types, loss=nn.MSELoss(), weight_decay=args.wd, lr=args.lr, print_each=1)

    test_dl  = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    answer = trainer.predict(seq_model, dataloaders=test_dl)
    a = [f['predicted'] for f in answer]
    b = torch.cat(a)
    b = b.numpy()
    target_columns = test_dataset.target_columns
    cols = target_columns + ['FEAT']
    real = test_dataset._data[cols].copy()
    pred_columns = []
    for ind, ta in enumerate(target_columns):
        c = f'{ta}_pred'
        real[c] = b[:, ind]
        pred_columns.append(c)
        
    ag = real.groupby('FEAT')[pred_columns + target_columns].mean()
    for r_name, p_name in zip(target_columns, pred_columns):
        p= pearsonr(ag[r_name], ag[p_name])
        pears.append(float(p[0]))

    # Write to file
    results.loc[len(results)] = pears

    results.to_csv(args.result_dir, sep = "\t", index = False)
