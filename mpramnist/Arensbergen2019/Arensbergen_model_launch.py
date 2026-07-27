import torch
import torch.nn as nn
import os
import pandas as pd

from mpramnist.Arensbergen2019.dataset import ArensbergenDataset
from mpramnist.Arensbergen2019.trainer import LitModel_Arensbergen_Reg

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

from torch.nn.utils.rnn import pad_sequence


import argparse 
parser = argparse.ArgumentParser()

general = parser.add_argument_group('general args', 
                                    'general_argumens')

general.add_argument("--result_dir",
                     type=str,
                     default = "./arensbergen.tsv")
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
dataset_args.add_argument("--genome_ids",
                     nargs='+',            # accepts one or more values
                     default=['SuRE42_HG02601', 'SuRE43_GM18983', 'SuRE44_HG01241', 'SuRE45_HG03464'],
                     help="List of genome ids")
dataset_args.add_argument("--cell_types",
                     nargs='+',            # accepts one or more values
                     default=["K562", "HepG2"],
                     help="List of genome ids")

trainer_args =  parser.add_argument_group('trainer args', 
                                'trainer arguments')

trainer_args.add_argument("--lr",
                     type=float,
                     default=5e-3)
trainer_args.add_argument("--wd",
                     type=float,
                     default=1e-2)
trainer_args.add_argument("--epoch_num",
                            type=int,
                            default=50)

args = parser.parse_args()

if isinstance(args.genome_ids, str):
    args.genome_ids = [args.genome_ids]
if isinstance(args.cell_types, str):
    args.cell_types = [args.cell_types]

if os.path.exists(args.result_dir):
    results = pd.read_csv(args.result_dir, sep = "\t")
else:
    cell_types = []
    for genome_id in args.genome_ids:
        for cell in args.cell_types:
            cell_types.append(genome_id + "_" + cell)
    results = pd.DataFrame(columns = cell_types)

if args.model == "Malinois" or  args.model == "DREAM_RNN" or args.model == "DREAM-RNN":
    train_transform = t.Compose([t.Padding(600), t.LeftCrop(600, 600), t.ReverseComplement(0.5),t.Seq2Tensor(sequence_first=True),])
    test_transform = t.Compose([t.Padding(600), t.LeftCrop(600, 600), t.Seq2Tensor(sequence_first=True),])
    forw_transform = t.Compose([t.Padding(600), t.LeftCrop(600, 600), t.Seq2Tensor(sequence_first=True)])
    rev_transform = t.Compose([t.Padding(600), t.LeftCrop(600, 600), t.ReverseComplement(1),t.Seq2Tensor(sequence_first=True)])
else:
    train_transform = t.Compose([t.ReverseComplement(0.5),t.Seq2Tensor(sequence_first=True),])
    test_transform = t.Compose([t.Seq2Tensor(sequence_first=True),])
    forw_transform = t.Compose([t.Seq2Tensor(sequence_first=True)])
    rev_transform = t.Compose([t.ReverseComplement(1),t.Seq2Tensor(sequence_first=True)])

def pad_collate(batch):  # required, because length of sequences is different
    (seq, targets) = zip(*batch)

    seq = pad_sequence(seq, batch_first=True, padding_value=0.25)

    return seq, torch.vstack(targets)

def meaned_prediction(forw, rev, trainer, seq_model, name, out_channels):
    predictions_forw = trainer.predict(seq_model, dataloaders=forw)
    targets = torch.cat([pred["target"] for pred in predictions_forw])
    y_preds_forw = torch.cat([pred["predicted"] for pred in predictions_forw])
    predictions_rev = trainer.predict(seq_model, dataloaders=rev)
    y_preds_rev = torch.cat([pred["predicted"] for pred in predictions_rev])
    mean_forw = torch.mean(torch.stack([y_preds_forw, y_preds_rev]), dim=0)
    pears = PearsonCorrCoef(num_outputs=out_channels)
    
    pearson = pears(mean_forw, targets)
    print("===========")
    print(name, " Pearson correlation")
    print(pearson)
    print("===========")
    return pearson

for run in list(range(args.runs)):
    print(args.cell_types)
    pears = []
    for genome_id in args.genome_ids:
        task = "regression"

        # load the data
        train_dataset = ArensbergenDataset(task=task, cell_type=args.cell_types, genome_id=genome_id, split="train", transform=train_transform, root="../data/",)
        val_dataset = ArensbergenDataset(task=task, cell_type=args.cell_types, genome_id=genome_id, split="val", transform=test_transform, root="../data/",)
        test_dataset = ArensbergenDataset(task=task, cell_type=args.cell_types, genome_id=genome_id, split="test", transform=test_transform, root="../data/",)

        # encapsulate data into dataloader form
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=pad_collate)
        val_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=pad_collate)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=pad_collate)

        if args.model == "MPRALegNet":
            model = HumanLegNet(
                in_ch=len(train_dataset[0][0][0]),
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
            model = DREAM_RNN(in_channels=len(train_dataset[0][0][0]), seqsize=length, out_channels=len(args.cell_types))
            loss = nn.MSELoss()

        seq_model = LitModel_Arensbergen_Reg(model=model, cell_types=args.cell_types, loss=nn.MSELoss(), weight_decay=args.wd, lr=args.lr, print_each=1)

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
        seq_model = LitModel_Arensbergen_Reg.load_from_checkpoint(best_model_path, model=model, cell_types=args.cell_types, loss=nn.MSELoss(), weight_decay=args.wd, lr=args.lr, print_each=1)

        # Load forward and reversed test sequences
        test_forw = ArensbergenDataset(task=task, cell_type=args.cell_types, genome_id=genome_id, split="test", transform=forw_transform, root=args.root)
        test_rev = ArensbergenDataset(task=task, cell_type=args.cell_types, genome_id=genome_id, split="test", transform=rev_transform, root=args.root)

        # Encapsulate test sequences
        forw = DataLoader(dataset=test_forw, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=pad_collate)
        rev = DataLoader(dataset=test_rev, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=pad_collate)

        corr_pearson = meaned_prediction(forw, rev, trainer, seq_model, genome_id + str(args.cell_types), len(args.cell_types)).numpy()

        if corr_pearson.ndim == 0:
            corr_pearson = [corr_pearson.item()]
        pears.extend(corr_pearson)
    # Write to file
    results.loc[len(results)] = pears

    results.to_csv(args.result_dir, sep = "\t", index = False)
