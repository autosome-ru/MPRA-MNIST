import torch
import torch.nn as nn
import os
import pandas as pd

# multi
from mpramnist.Gosai2024.dataset import GosaiDataset
from mpramnist.Chen2025.dataset import ChenMultiDataset

from mpramnist.Chen2025.trainer import LitModel_Chen

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
                     default = "./guo.tsv")
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

dataset_args.add_argument("--cell_types_train",
                     nargs='+',            # accepts one or more values
                     default=["HepG2", "K562", "SKNSH"],
                     help="List of cell types for training from Gosai dataset")

dataset_args.add_argument("--cell_types_variant",
                     nargs='+',            # accepts one or more values
                     default=None,
                     help="List of cell types")

dataset_args.add_argument("--states_variant",
                     nargs='+',            # accepts one or more values
                     default=None,
                     help="List of states for each cell type.")

dataset_args.add_argument("--interval_type",
                     nargs='?',            # accepts one or more values
                     default=None,
                     help="Interval type (SNPCENTER or PEAKCENTER) to be used.")


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

if isinstance(args.cell_types_train, str):
    args.cell_types_train = [args.cell_types_train]

if isinstance(args.cell_types_variant, str):
    args.cell_types_variant = [args.cell_types_variant]

forw_transform = t.Compose([t.AddFlanks(GosaiDataset.LEFT_FLANK, GosaiDataset.RIGHT_FLANK), t.CenterCrop(600), t.Seq2Tensor()])
revcomp_transform = t.Compose([t.AddFlanks(GosaiDataset.LEFT_FLANK, GosaiDataset.RIGHT_FLANK), t.CenterCrop(600), t.ReverseComplement(1), t.Seq2Tensor()])

def get_variant_predictions(forw_preds, revcomp_preds, cell_types):

    targets = torch.cat([pred["target"] for pred in forw_preds])
    fdrs = torch.cat([pred["fdr"] for pred in forw_preds])
    reverse_prediction = torch.cat([pred["reverse_prediction"] for pred in forw_preds])

    y_preds_forw_ref = torch.cat([pred["ref_predicted"] for pred in forw_preds])
    y_preds_forw_alt = torch.cat([pred["alt_predicted"] for pred in forw_preds])


    y_preds_revcomp_ref = torch.cat([pred["ref_predicted"] for pred in revcomp_preds])
    y_preds_revcomp_alt = torch.cat([pred["alt_predicted"] for pred in revcomp_preds])

    y_preds_ref = torch.mean(torch.stack([y_preds_forw_ref, y_preds_revcomp_ref]), dim=0)
    y_preds_alt = torch.mean(torch.stack([y_preds_forw_alt, y_preds_revcomp_alt]), dim=0)

    variant_prediction = (y_preds_alt - y_preds_ref).squeeze() * reverse_prediction.squeeze()
    
    results = []
    pears = PearsonCorrCoef()

    for i in range(len(cell_types)):

        pearsonr_all = pears(variant_prediction.squeeze(), targets[:, i].squeeze())

        mask_daSNP = fdrs[:, i].squeeze() < 0.05     # remove non-significant variants with fdr > 0.05
        pearsonr_daSNP = pears(variant_prediction.squeeze()[mask_daSNP], targets[:, i].squeeze()[mask_daSNP])

        results.append({
            'cell_type': cell_types[i],
            'pearsonr_all': pearsonr_all.item(),
            'n_daSNP': mask_daSNP.sum().item(),
            'pearsonr_daSNP': pearsonr_daSNP.item()
        })

    df = pd.DataFrame(results)
    return df

CELL_TYPE = {'THP1': ['aggregated', 'Naive', 'IFNB', 'IFNG', 'LPSIFNG'],
             'HMC3': ['aggregated','Naive', 'IFNB', 'IFNG', 'LPSIFNG'],
             'Brain': ['aggregated','Cortex', 'Hippocampus', 'Striatum']}


std_err = [cell + "_lfcSE" for cell in args.cell_types_train]

for run in list(range(args.runs)):

    train_transform = t.Compose([t.AddFlanks(GosaiDataset.LEFT_FLANK, GosaiDataset.RIGHT_FLANK), t.CenterCrop(600), t.ReverseComplement(0.5), t.Seq2Tensor(),])
    val_test_transform = t.Compose([t.AddFlanks(GosaiDataset.LEFT_FLANK, GosaiDataset.RIGHT_FLANK), t.CenterCrop(600), t.Seq2Tensor()])

    # load the data
    train_dataset_own = GosaiDataset(
        split="train",
        transform=train_transform,
        filtration="own",
        cell_types=args.cell_types_train,
        stderr_columns=std_err,  
        stderr_threshold=1.0,  
        std_multiple_cut=6.0,  
        up_cutoff_move=3.0,  
        duplication_cutoff=0.5,  
        root=args.root
    )

    # Use the same parameters to valid and test
    val_dataset_own = GosaiDataset(split="val", filtration="own", cell_types=args.cell_types_train, stderr_columns=std_err, stderr_threshold=1.0, std_multiple_cut=6.0, up_cutoff_move=3.0, transform=val_test_transform, root=args.root)
    test_dataset_own = GosaiDataset(split="test", filtration="own", cell_types=args.cell_types_train, stderr_columns=std_err, stderr_threshold=1.0, std_multiple_cut=6.0, up_cutoff_move=3.0, transform=val_test_transform, root=args.root)

    train_loader = DataLoader(dataset=train_dataset_own, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_dataset_own, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_dataset_own, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    use_one_cycle = True

    if args.model == "MPRALegNet":
        model = HumanLegNet(
            in_ch=len(train_dataset_own[0][0]),
            output_dim=len(args.cell_types_train),
            stem_ch=64,
            stem_ks=11,
            ef_ks=9,
            ef_block_sizes=[80, 96, 112, 128],
            pool_sizes=[2, 2, 2, 2],
            resize_factor=4)
        model.apply(initialize_weights)
        loss =nn.MSELoss()

    elif args.model == "MPRAnn":
        model = MPRAnn(output_dim=len(args.cell_types_train))
        loss =nn.MSELoss()

    elif args.model == "Malinois":
        model = BassetBranched(input_len=len(train_dataset_own[0][0][0]), n_outputs=len(args.cell_types_train))
        loss =L1KLmixed()
        use_one_cycle = False

    elif args.model == "PARM":
        model = PARM(n_block=5, type_loss="mse", output_dim=len(args.cell_types_train))
        loss =nn.MSELoss()

    elif args.model =="DREAM-RNN" or args.model == "DREAM_RNN":
        model = DREAM_RNN(in_channels=len(train_dataset_own[0][0]), seqsize=600, out_channels=len(args.cell_types_train))
        loss = nn.MSELoss()

    seq_model = LitModel_Chen(model=model, loss=nn.MSELoss(), weight_decay=args.wd, lr=args.lr, print_each=1, use_one_cycle=use_one_cycle)

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
    seq_model = LitModel_Chen.load_from_checkpoint(best_model_path,model=model, loss=nn.MSELoss(), weight_decay=args.wd, lr=args.lr,  print_each=1, use_one_cycle=use_one_cycle)

    predict_forward_dataset = ChenMultiDataset(split = 'test', length = 227, cell_types = args.cell_types_variant, states = args.states_variant, interval_type = args.interval_type, transform=forw_transform, root =args.root)
    predict_forward_dataloader = DataLoader(dataset=predict_forward_dataset, batch_size=1024, shuffle=False, num_workers=args.num_workers, pin_memory=True,)

    predict_revcomp_dataset = ChenMultiDataset(split = 'test', length = 227, cell_types = args.cell_types_variant, states = args.states_variant, interval_type = args.interval_type, transform=revcomp_transform, root =args.root)
    predict_revcomp_dataloader = DataLoader(dataset=predict_forward_dataset, batch_size=1024, shuffle=False, num_workers=args.num_workers, pin_memory=True,)

    forw_preds = trainer.predict(seq_model, dataloaders=predict_forward_dataloader)
    revcomp_preds = trainer.predict(seq_model, dataloaders=predict_revcomp_dataloader)

    results = get_variant_predictions(forw_preds, revcomp_preds, ['THP1_aggregated', 'THP1_Naive', 'THP1_IFNB', 'THP1_IFNG', 'THP1_LPSIFNG', 'HMC3_aggregated','HMC3_Naive', 'HMC3_IFNB', 'HMC3_IFNG', 'HMC3_LPSIFNG', 'Brain_aggregated','Brain_Cortex', 'Brain_Hippocampus', 'Brain_Striatum'])

    output_file = f"{args.result_dir}/{args.model}_run{run}.tsv"

    results.to_csv(output_file, sep="\t", index=False)