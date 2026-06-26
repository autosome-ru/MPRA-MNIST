import torch
import torch.nn as nn
import os
import pandas as pd

# Single
from mpramnist.Agarwal2025.dataset import AgarwalSingleDataset
from mpramnist.Siraj2026.dataset import SirajMPRADataset, SirajSatMutDataset

from mpramnist.Siraj2026.trainer import LitModel_Siraj

from mpramnist.models import HumanLegNet
from mpramnist.models import initialize_weights

from mpramnist.models import BassetBranched
from mpramnist.models import L1KLmixed

from mpramnist.models import MPRAnn

from mpramnist.models import PARM

import mpramnist.transforms as t

from torch.utils.data import DataLoader
from torchmetrics import PearsonCorrCoef

import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint

import argparse

parser = argparse.ArgumentParser()

general = parser.add_argument_group("general args", "general_argumens")

general.add_argument("--result_dir", type=str, default="./Siraj.tsv")
general.add_argument("--device", type=int, default=0)
general.add_argument("--num_workers", type=int, default=103)
general.add_argument("--runs", type=int, default=5)
general.add_argument(
    "--model", type=str, default="MPRALegNet"
)  # or Malinois/MPRAnn/PARM

dataset_args = parser.add_argument_group("dataset args", "dataset arguments")

dataset_args.add_argument("--root", type=str, default="../data/")
dataset_args.add_argument("--cell_type", type=str, default="K562")

trainer_args = parser.add_argument_group("trainer args", "trainer arguments")

trainer_args.add_argument("--lr", type=float, default=0.01)
trainer_args.add_argument("--wd", type=float, default=0.1)
trainer_args.add_argument("--epoch_num", type=int, default=50)


args = parser.parse_args()


if os.path.exists(args.result_dir):
    results = pd.read_csv(args.result_dir, sep="\t")
else:
    results = pd.DataFrame(columns=[args.cell_type])

train_transform = t.Compose([
    t.AddFlanks(
        AgarwalSingleDataset.CONSTANT_LEFT_FLANK,
        AgarwalSingleDataset.CONSTANT_RIGHT_FLANK,
    ),
    t.AddFlanks("", AgarwalSingleDataset.RIGHT_FLANK),
    t.RightCrop(230, 260),
    t.LeftCrop(230, 230),
    t.ReverseComplement(0.5),
    t.Seq2Tensor(),
])
test_transform = t.Compose([
    t.AddFlanks(
        AgarwalSingleDataset.CONSTANT_LEFT_FLANK,
        AgarwalSingleDataset.CONSTANT_RIGHT_FLANK,
    ),
    t.ReverseComplement(0),
    t.Seq2Tensor(),
])

forw_transform = t.Compose([
    t.AddFlanks(
        AgarwalSingleDataset.CONSTANT_LEFT_FLANK,
        AgarwalSingleDataset.CONSTANT_RIGHT_FLANK,
    ),
    t.Seq2Tensor(),
])
rev_transform = t.Compose([
    t.AddFlanks(
        AgarwalSingleDataset.CONSTANT_LEFT_FLANK,
        AgarwalSingleDataset.CONSTANT_RIGHT_FLANK,
    ),
    t.ReverseComplement(1),
    t.Seq2Tensor(),
])


def meaned_prediction(forw, rev, trainer, seq_model, name, is_Siraj=False):
    predictions_forw = trainer.predict(seq_model, dataloaders=forw)
    targets = torch.cat([pred["target"] for pred in predictions_forw])
    y_preds_forw = torch.cat([pred["ref_predicted"] for pred in predictions_forw])

    predictions_rev = trainer.predict(seq_model, dataloaders=rev)
    y_preds_rev = torch.cat([pred["ref_predicted"] for pred in predictions_rev])

    mean_forw = torch.mean(torch.stack([y_preds_forw, y_preds_rev]), dim=0)

    pears = PearsonCorrCoef()
    print(name, " Pearson correlation")

    if is_Siraj:
        y_preds_forw_alt = torch.cat([
            pred["alt_predicted"] for pred in predictions_forw
        ])
        y_preds_rev_alt = torch.cat([pred["alt_predicted"] for pred in predictions_rev])
        mean_alt = torch.mean(torch.stack([y_preds_forw_alt, y_preds_rev_alt]), dim=0)
        pred = mean_alt - mean_forw
        return pears(pred, targets)

    return pears(mean_forw, targets)


for run in list(range(args.runs)):
    # Read the MPRAdata, preprocess them and encapsulate them into dataloader form.
    train_dataset = AgarwalSingleDataset(
        cell_type=args.cell_types,
        split="train",
        transform=train_transform,
        root=args.root,
    )
    val_dataset = AgarwalSingleDataset(
        cell_type=args.cell_types,
        split="val",
        transform=test_transform,
        root=args.root,
    )
    test_dataset = AgarwalSingleDataset(
        cell_type=args.cell_types,
        split="test",
        transform=test_transform,
        root=args.root,
    )

    # encapsulate data into dataloader form
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=args.num_workers,
    )

    if args.model == "MPRALegNet":
        model = HumanLegNet(
            in_ch=len(train_dataset[0][0]),
            output_dim=1,
            stem_ch=64,
            stem_ks=11,
            ef_ks=9,
            ef_block_sizes=[80, 96, 112, 128],
            pool_sizes=[2, 2, 2, 2],
            resize_factor=4,
        )
        model.apply(initialize_weights)
        loss = nn.MSELoss()
    elif args.model == "MPRAnn":
        model = MPRAnn(output_dim=1)
        loss = nn.MSELoss()
    elif args.model == "Malinois":
        length = len(train_dataset[0][0][0])
        model = BassetBranched(input_len=length, n_outputs=1, loss_criterion=L1KLmixed)
        loss = L1KLmixed()
    elif args.model == "PARM":
        model = PARM(n_block=5, type_loss="mse", output_dim=1)
        loss = nn.MSELoss()

    seq_model = LitModel_Siraj(
        model=model, loss=loss, weight_decay=args.wd, lr=args.lr, print_each=1
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_pearson", mode="max", save_top_k=1, save_last=False
    )

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
    seq_model = LitModel_Siraj.load_from_checkpoint(
        best_model_path,
        model=model,
        loss=nn.MSELoss(),
        weight_decay=args.wd,
        lr=args.lr,
        print_each=1,
    )

    test_forw = SirajDataset(
        cell_type=args.cell_type, length=200, transform=forw_transform, root=args.root
    )
    test_rev = SirajDataset(
        cell_type=args.cell_type, length=200, transform=rev_transform, root=args.root
    )

    forw_single = DataLoader(
        dataset=test_forw,
        batch_size=1024,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    rev_single = DataLoader(
        dataset=test_rev,
        batch_size=1024,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    corr_pearson = meaned_prediction(
        forw_single,
        rev_single,
        trainer,
        seq_model,
        name=args.cell_type,
        is_Siraj=True,
    )

    print(corr_pearson.numpy())
    results.loc[len(results)] = corr_pearson.numpy()

    results.to_csv(args.result_dir, sep="\t", index=False)
