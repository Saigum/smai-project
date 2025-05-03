import os
import argparse
import pandas as pd
from PIL import Image

import torch
import timm
import wandb
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import DBSCAN

# --- Helpers ---
@rank_zero_only
def make_logger(args):
    return WandbLogger(
        project="angle-prediction",
        name=f"{args.model_name}_bs{args.batch_size}_lr{args.lr}"
    )

@rank_zero_only
def init_wandb(args, model):
    wandb.init(
        project="angle-prediction",
        name=f"{args.model_name}_bs{args.batch_size}_lr{args.lr}",
        config=vars(args),
    )
    wandb.watch(model, log="all", log_freq=50)

@rank_zero_only
def finish_wandb():
    wandb.finish()

# --- Data Processing ---
def data_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df['angle'].between(0, 360)].copy()

    # MAD-based outlier removal
    med = df['angle'].median()
    mad = (df['angle'] - med).abs().median()
    df = df[(df['angle'] - med).abs() <= 4 * mad]

    # DBSCAN filtering on location (keep clusters with ≥2 points)
    coords = df[['longitude', 'latitude']].values
    db = DBSCAN(eps=50, min_samples=10).fit(coords)
    df['cluster'] = db.labels_
    counts = pd.Series(db.labels_).value_counts()
    valid = counts[counts >= 2].index
    df = df[df['cluster'].isin(valid)].drop(columns='cluster')

    return df

# --- Transforms & Augmentations ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# --- Angle Encoding / Decoding ---
def angle_to_sincos(angle: float):
    rad = torch.deg2rad(torch.tensor(angle))
    return torch.stack([torch.sin(rad), torch.cos(rad)])

def sincos_to_angle(sincos_pred: torch.Tensor):
    # sincos_pred: [batch, 2]
    rad = torch.atan2(sincos_pred[:, 0], sincos_pred[:, 1])
    deg = torch.rad2deg(rad) % 360
    return deg

# --- Dataset & DataModule ---
class DFDataroot:
    def __init__(self, root_dir: str, dataframe: pd.DataFrame):
        self.root_dir = root_dir
        self.dataframe = dataframe

class AngleDataset(Dataset):
    def __init__(self,
                 root_df: DFDataroot,
                 transform,
                 target_transform=angle_to_sincos,
                 target_str: str = "angle"):
        self.root_dir = root_df.root_dir
        self.df = root_df.dataframe.reset_index(drop=True)
        self.transform = transform
        self.target_transform = target_transform
        self.target_str = target_str

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.root_dir, row["filename"])).convert("RGB")
        img = self.transform(img)
        angle = float(row[self.target_str])
        target = self.target_transform(angle)
        return img, target

class AngleDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_root: DFDataroot,
                 val_root: DFDataroot,
                 batch_size: int = 32):
        super().__init__()
        self.train_root = train_root
        self.val_root = val_root
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_ds = AngleDataset(self.train_root, transform=train_transform)
        self.val_ds   = AngleDataset(self.val_root,   transform=val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          num_workers=4,
                          pin_memory=True,
                          persistent_workers=True)

# --- Model ---
class AnglePredictionModel(pl.LightningModule):
    def __init__(self, pretrained: bool,
                 lr: float,
                 model_name: str):
        super().__init__()
        self.save_hyperparameters()

        # Backbone selection
        if 'vit' in model_name:
            self.backbone = timm.create_model(model_name, pretrained=pretrained)
            in_feats = self.backbone.head.in_features
            self.backbone.head = torch.nn.Linear(in_feats, 2)

        elif 'efficientnet' in model_name:
            self.backbone = timm.create_model(model_name, pretrained=pretrained)
            in_feats = self.backbone.classifier.in_features
            self.backbone.classifier = torch.nn.Linear(in_feats, 2)

        elif 'convnext' in model_name:
            self.backbone = timm.create_model(model_name, pretrained=pretrained)
            in_feats = self.backbone.head.fc.in_features
            self.backbone.head.fc = torch.nn.Linear(in_feats, 2)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, target = batch
        pred = self(x)
        loss = self.criterion(pred, target)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        pred = self(x)
        pred_angle = sincos_to_angle(pred)
        true_angle = sincos_to_angle(target)
        delta = (pred_angle - true_angle + 180) % 360 - 180
        ang_mae = torch.mean(torch.abs(delta))
        self.log('val_loss', ang_mae, prog_bar=True, sync_dist=True)
        return ang_mae

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
        return [opt], [sched]

# --- Main ---
def main(args):
    train_df = data_preprocess(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)

    train_root = DFDataroot(args.train_root, train_df)
    val_root   = DFDataroot(args.val_root,   val_df)
    dm = AngleDataModule(train_root, val_root, batch_size=args.batch_size)

    model = AnglePredictionModel(pretrained=True,
                                 lr=args.lr,
                                 model_name=args.model_name)

    init_wandb(args, model)
    wandb_logger = make_logger(args)

    early_stop = EarlyStopping(monitor="val_loss",
                               patience=7,
                               mode="min",
                               verbose=True)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename=f"best-{args.model_name}_bs{args.batch_size}_lr{args.lr}",
        save_top_k=3,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.devices,
        logger=wandb_logger,
        callbacks=[early_stop, ckpt_cb],
        log_every_n_steps=10,
        strategy="ddp",  # ensure distributed sync_dist works
    )

    trainer.fit(model, dm)
    finish_wandb()

if __name__ == "__main__":
    p = argparse.ArgumentParser("Angle Prediction with Sincos + Arctan")
    p.add_argument("--train_root", type=str, required=True)
    p.add_argument("--val_root",   type=str, required=True)
    p.add_argument("--train_csv",  type=str, required=True)
    p.add_argument("--val_csv",    type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--devices",    type=int, default=1)
    p.add_argument("--model_name", type=str, default="vit_base_patch16_224",
                   help="e.g., vit_base_patch16_224, efficientnet_b3.ra2_in1k, convnext_base")
    args = p.parse_args()
    main(args)
