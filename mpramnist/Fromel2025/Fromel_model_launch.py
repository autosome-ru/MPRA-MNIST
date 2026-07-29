import torch
import torch.nn as nn
import os
import pandas as pd

# multi
from mpramnist.Fromel2025 import FromelDataset
from mpramnist.Fromel2025 import LitModel_Fromel, MaskedMSE, MaskedPearsonCorrCoef

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
                     default = "./fromel.tsv")
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
dataset_args.add_argument("--cell_types",
                     type=str, 
                     default="HSPC", # can be K562
                     help="Cell type used")
dataset_args.add_argument("--targets",
                     nargs='+',            # accepts one or more values
                     default=['State_1M','State_2D','State_3E','State_4M','State_5M','State_6N','State_7M',],
                     help="List of cell types")
dataset_args.add_argument("--test_type",
                     nargs='+',            # accepts one or more values
                     default=['synthetic', 'test', 'genome', 'generated'],
                     help="List of test subdatasets")

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

test_dfs = {}
if args.cell_types == "K562":
    args.targets = ["State_9K"]
    args.test_type = ['test']
    result_dir = args.result_dir.split(".tsv")[0] + "_K562_synthetic" + ".tsv"
    if os.path.exists(result_dir):
        test_dfs['test'] = pd.read_csv(result_dir, sep = "\t")
    else:
        test_dfs['test'] = pd.DataFrame(columns = args.targets)
else:
    for test in args.test_type:
        result_dir = args.result_dir.split(".tsv")[0] + "_HSPC_" + test + ".tsv"
        if os.path.exists(result_dir):
            test_dfs[test] = pd.read_csv(result_dir, sep = "\t")
        else:
            test_dfs[test] = pd.DataFrame(columns = args.targets)


    
    

train_transform = t.Compose(
    [
        t.AddFlanks(FromelDataset.CONSTANT_LEFT_FLANK, FromelDataset.CONSTANT_RIGHT_FLANK),
        t.AddFlanks("", FromelDataset.RIGHT_FLANK),  # this is original parameters for human_legnet
        t.RightCrop(245, 270),  # this is using for shifting
        t.LeftCrop(245, 245),
        t.ReverseComplement(0.5),
        t.AddFeatureChannels(['batch']),
        t.Seq2Tensor(),
    ]
)
test_transform = t.Compose(
    [
        t.AddFlanks(FromelDataset.CONSTANT_LEFT_FLANK, FromelDataset.CONSTANT_RIGHT_FLANK),
        t.LeftCrop(245, 245),
        t.ReverseComplement(0),
        t.AddFeatureChannels(['batch']),
        t.Seq2Tensor(),
    ]
)
test_transform_rev = t.Compose(
    [
        t.AddFlanks(FromelDataset.CONSTANT_LEFT_FLANK, FromelDataset.CONSTANT_RIGHT_FLANK),
        t.LeftCrop(245, 245),
        t.ReverseComplement(1),
        t.AddFeatureChannels(['batch']),
        t.Seq2Tensor(),
    ]
)

def meaned_prediction(forw, rev, trainer, seq_model, name, num):
    predictions_forw = trainer.predict(seq_model, dataloaders=forw)
    targets = torch.cat([pred["target"] for pred in predictions_forw])
    y_preds_forw = torch.cat([pred["predicted"] for pred in predictions_forw])

    predictions_rev = trainer.predict(seq_model, dataloaders=rev)
    y_preds_rev = torch.cat([pred["predicted"] for pred in predictions_rev])

    mean_forw = torch.mean(torch.stack([y_preds_forw, y_preds_rev]), dim=0)

    pears = MaskedPearsonCorrCoef(num_outputs=num)
    pearson = pears(mean_forw, targets)
    print("===========")
    print(name, " Pearson correlation")
    print(pearson)
    print("===========")

    return pears(mean_forw, targets)


for run in list(range(args.runs)):
    print(args.cell_types)
    print(args.targets)
    # Read the MPRAdata, preprocess them and encapsulate them into dataloader form.
    train_dataset = FromelDataset(cell_type=args.cell_types, targets=args.targets, split="train", transform=train_transform, root=args.root,)
    val_dataset = FromelDataset(cell_type=args.cell_types, targets=args.targets, split="val", transform=test_transform, root=args.root,)

    # encapsulate data into dataloader form
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.model == "MPRALegNet":
        model = HumanLegNet(
            in_ch=len(train_dataset[0][0]),
            output_dim=len(args.targets),
            stem_ch=64,
            stem_ks=11,
            ef_ks=9,
            ef_block_sizes=[80, 96, 112, 128],
            pool_sizes=[2, 2, 2, 2],
            resize_factor=4)
        model.apply(initialize_weights)
        loss =MaskedMSE()
    elif args.model == "MPRAnn":
        model = MPRAnn(in_channels=len(train_dataset[0][0]), output_dim=len(args.targets))
        loss =MaskedMSE()
    elif args.model == "Malinois":
        length = len(train_dataset[0][0][0])
        model = BassetBranched(input_len=length, n_channels=len(train_dataset[0][0]), n_outputs=len(args.targets))
        loss =MaskedMSE()
    elif args.model == "PARM":
        model = PARM(n_block=5, type_loss="mse", output_dim=len(args.targets), vocab = len(train_dataset[0][0]))
        loss =MaskedMSE()
    elif args.model =="DREAM-RNN" or args.model == "DREAM_RNN":
            length = len(train_dataset[0][0][0])
            model = DREAM_RNN(in_channels=len(train_dataset[0][0]), seqsize=length, out_channels=len(args.targets))
            loss = nn.MSELoss()

    seq_model = LitModel_Fromel(model=model, loss=loss, weight_decay=args.wd, lr=args.lr, activity_columns=args.targets, print_each=1)

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
    seq_model = LitModel_Fromel.load_from_checkpoint(best_model_path,model=model, loss=loss, weight_decay=args.wd, lr=args.lr, 
                                                              activity_columns=args.targets, print_each=1)

    forw_transform = t.Compose([t.Seq2Tensor()])
    rev_transform = t.Compose([t.ReverseComplement(1),t.Seq2Tensor(),])

    for test in args.test_type:
        test_forw = FromelDataset(cell_type=args.cell_types, targets=args.targets, split=test, transform=test_transform, root=args.root)
        test_rev = FromelDataset(cell_type=args.cell_types, targets=args.targets, split=test, transform=test_transform_rev, root=args.root)

        forw = DataLoader(dataset=test_forw, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,)
        rev = DataLoader(dataset=test_rev, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,)

        corr_pearson = meaned_prediction(forw, rev, trainer, seq_model, args.cell_types, len(args.targets))
        
        test_dfs[test].loc[len(test_dfs[test])] = corr_pearson.numpy()
        if args.cell_types == "K562":
            result_dir = args.result_dir.split(".tsv")[0] + "_K562_synthetic" + ".tsv"
        else:
            result_dir = args.result_dir.split(".tsv")[0] + "_HSPC_" + test + ".tsv"
            
        test_dfs[test].to_csv(result_dir, sep = "\t", index = False)