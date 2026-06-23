import torch
import torch.nn as nn
import os
import pandas as pd

# multi
from mpramnist.Gosai2024.dataset import GosaiDataset
from mpramnist.Gosai2024.trainer import LitModel_Gosai

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
                     default = "./gosai.tsv")
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
dataset_args.add_argument("--cell_types",
                     nargs='+',            # accepts one or more values
                     default=["HepG2", "K562", "SKNSH"],
                     help="List of cell types")

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

if isinstance(args.cell_types, str):
    args.cell_types = [args.cell_types]

if os.path.exists(args.result_dir):
    results = pd.read_csv(args.result_dir, sep = "\t")
else:
    results = pd.DataFrame(columns = args.cell_types)

forw_transform = t.Compose([t.AddFlanks(GosaiDataset.LEFT_FLANK, GosaiDataset.RIGHT_FLANK), t.CenterCrop(600), t.Seq2Tensor()])
rev_transform = t.Compose([t.AddFlanks(GosaiDataset.LEFT_FLANK, GosaiDataset.RIGHT_FLANK),t.CenterCrop(600),t.ReverseComplement(1),t.Seq2Tensor(),])

def meaned_prediction(forw, rev, trainer, seq_model, name, num_outputs):
    predictions_forw = trainer.predict(seq_model, dataloaders=forw)
    targets = torch.cat([pred["target"] for pred in predictions_forw])
    y_preds_forw = torch.cat([pred["predicted"] for pred in predictions_forw])

    predictions_rev = trainer.predict(seq_model, dataloaders=rev)
    y_preds_rev = torch.cat([pred["predicted"] for pred in predictions_rev])

    mean_forw = torch.mean(torch.stack([y_preds_forw, y_preds_rev]), dim=0)

    pears = PearsonCorrCoef(num_outputs=num_outputs)
    pearson = pears(mean_forw, targets)
    print("===========")
    print(name, " Pearson correlation")
    print(pearson)
    print("===========")
    return pearson
std_err = [cell + "_lfcSE" for cell in args.cell_types]
for run in list(range(args.runs)):
    print(args.cell_types)

    train_transform = t.Compose([t.AddFlanks(GosaiDataset.LEFT_FLANK, GosaiDataset.RIGHT_FLANK), t.CenterCrop(600), t.ReverseComplement(0.5), t.Seq2Tensor(),])
    val_test_transform = t.Compose([t.AddFlanks(GosaiDataset.LEFT_FLANK, GosaiDataset.RIGHT_FLANK), t.CenterCrop(600), t.Seq2Tensor()])

    # load the data
    train_dataset_own = GosaiDataset(
        split="train",
        transform=train_transform,
        filtration="own",
        cell_types=args.cell_types,
        stderr_columns=std_err,  
        stderr_threshold=1.0,  
        std_multiple_cut=6.0,  
        up_cutoff_move=3.0,  
        duplication_cutoff=0.5,  
        root=args.root
    )
    # Use the same parameters to valid and test
    val_dataset_own = GosaiDataset(split="val", filtration="own", cell_types=args.cell_types, stderr_columns=std_err, stderr_threshold=1.0, std_multiple_cut=6.0, up_cutoff_move=3.0, transform=val_test_transform, root=args.root)
    test_dataset_own = GosaiDataset(split="test", filtration="own", cell_types=args.cell_types, stderr_columns=std_err, stderr_threshold=1.0, std_multiple_cut=6.0, up_cutoff_move=3.0, transform=val_test_transform, root=args.root)

    train_loader = DataLoader(dataset=train_dataset_own, batch_size=args.batch_size, shuffle=True, pin_memory=True, prefetch_factor=4, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_dataset_own, batch_size=args.batch_size, shuffle=False, pin_memory=True, prefetch_factor=4, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_dataset_own, batch_size=args.batch_size, shuffle=False, pin_memory=True, prefetch_factor=4, num_workers=args.num_workers)

    use_one_cycle = True
    if args.model == "MPRALegNet":
        model = HumanLegNet(
            in_ch=len(train_dataset_own[0][0]),
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
        model = BassetBranched(input_len=len(train_dataset_own[0][0][0]), n_outputs=len(args.cell_types))
        loss =L1KLmixed()
        use_one_cycle = False
    elif args.model == "PARM":
        model = PARM(n_block=5, type_loss="mse", output_dim=len(args.cell_types))
        loss =nn.MSELoss()
    elif args.model =="DREAM-RNN" or args.model == "DREAM_RNN":
            model = DREAM_RNN(in_channels=len(train_dataset_own[0][0]), seqsize=600, out_channels=len(args.cell_types))
            loss = nn.MSELoss()

    seq_model = LitModel_Gosai(model=model, loss=nn.MSELoss(), weight_decay=args.wd, lr=args.lr, cell_types=args.cell_types, print_each=1, use_one_cycle=use_one_cycle)

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
    seq_model = LitModel_Gosai.load_from_checkpoint(best_model_path,model=model, loss=nn.MSELoss(), weight_decay=args.wd, lr=args.lr, cell_types=args.cell_types, print_each=1, use_one_cycle=use_one_cycle)

    test_forw = GosaiDataset(split="test", filtration="own", cell_types=args.cell_types, stderr_columns=std_err, stderr_threshold=1.0, std_multiple_cut=6.0, up_cutoff_move=3.0, transform=forw_transform, root=args.root)
    test_rev = GosaiDataset(split="test", filtration="own", cell_types=args.cell_types, stderr_columns=std_err, stderr_threshold=1.0, std_multiple_cut=6.0, up_cutoff_move=3.0, transform=rev_transform, root=args.root)

    forw = DataLoader(dataset=test_forw, batch_size=1024, shuffle=False, num_workers=args.num_workers, pin_memory=True,)
    rev = DataLoader(dataset=test_rev, batch_size=1024, shuffle=False, num_workers=args.num_workers, pin_memory=True,)

    corr_pearson = meaned_prediction(forw, rev, trainer, seq_model, args.cell_types, len(args.cell_types))

    results.loc[len(results)] = corr_pearson.numpy()

    results.to_csv(args.result_dir, sep = "\t", index = False)