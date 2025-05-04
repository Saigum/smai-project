#!/usr/bin/env python
# latlon_train_yolo.py
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import wandb


# ──────────────────────────── filtering utilities ────────────────────────────
def remove_outliers_mad(df: pd.DataFrame,
                        cols: list[str],
                        thresh: float = 4.0) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        med = out[col].median()
        mad = np.median(np.abs(out[col] - med))
        if mad == 0:
            continue
        out = out[np.abs(out[col] - med) <= thresh * mad]
    return out

def filter_coords(df: pd.DataFrame,
                  lat_col: str = "latitude",
                  lon_col: str = "longitude",
                  mad_thresh: float = 4.0,
                  dbscan_eps: float = 50.0,
                  dbscan_min_samples: int = 10
                  ) -> pd.DataFrame:
    mad_df = remove_outliers_mad(df, [lat_col, lon_col], thresh=mad_thresh)
    print(f"After MAD filtering: {len(mad_df):,} rows")

    coords = mad_df[[lon_col, lat_col]].to_numpy()
    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(coords)
    mad_df = mad_df.assign(dbscan_label=labels)

    final_df = mad_df[mad_df.dbscan_label != -1].copy()
    print(f"After DBSCAN filtering: {len(final_df):,} rows")

    return final_df


# ─────────────────────────────── scaling utility ──────────────────────────────
def scale_df(df: pd.DataFrame,
             lat_col: str = "latitude",
             lon_col: str = "longitude",
             train_mm: tuple[float, float] | None = None,
             train_lm: tuple[float, float] | None = None
             ) -> tuple[pd.DataFrame, tuple[float,float], tuple[float,float]]:
    """
    Min-max scale df in-place. If train_mm/lm provided, reuse them for lat/lon.
    Returns (scaled_df, (lat_min,lat_max), (lon_min,lon_max))
    """
    df = df.copy()
    if train_mm is None:
        lat_min, lat_max = df[lat_col].min(), df[lat_col].max()
    else:
        lat_min, lat_max = train_mm

    if train_lm is None:
        lon_min, lon_max = df[lon_col].min(), df[lon_col].max()
    else:
        lon_min, lon_max = train_lm

    df[lat_col] = (df[lat_col] - lat_min) / (lat_max - lat_min)
    df[lon_col] = (df[lon_col] - lon_min) / (lon_max - lon_min)

    return df, (lat_min, lat_max), (lon_min, lon_max)


# ─────────────────────────────── W&B helpers ────────────────────────────────
@rank_zero_only
def make_logger(args):
    return WandbLogger(
        project="latlon-prediction",
        name=f"{Path(args.backbone_ckpt).stem}_bs{args.batch_size}_lr{args.lr}",
        log_model=True,
        save_dir="checkpoints",
    )

@rank_zero_only
def init_wandb(args, model):
    wandb.init(
        project="latlon-prediction",
        name=f"{Path(args.backbone_ckpt).stem}_bs{args.batch_size}_lr{args.lr}",
        config=vars(args),
    )
    wandb.watch(model, log="all", log_freq=50)

@rank_zero_only
def finish_wandb(best_ckpt):
    art = wandb.Artifact("latlon-regressor", type="model")
    art.add_file(best_ckpt)
    wandb.log_artifact(art)
    wandb.finish()


# ────────────────────────────────── data ─────────────────────────────────────
image_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, .456, .406), (.229, .224, .225)),
])

class RootedDF:
    def __init__(self, root_dir, df):
        self.root_dir = root_dir
        self.df = df.reset_index(drop=True)

class LatLonDS(Dataset):
    def __init__(self, rooted_df: RootedDF, tf=image_tf):
        self.root = rooted_df.root_dir
        self.df = rooted_df.df
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(os.path.join(self.root, r.filename)).convert("RGB")
        img = self.tf(img)
        tgt = torch.tensor([r.latitude, r.longitude], dtype=torch.float32)
        return img, tgt

class LatLonDM(pl.LightningDataModule):
    def __init__(self, train_root, val_root, bs):
        super().__init__()
        self.train_root, self.val_root, self.bs = train_root, val_root, bs

    def setup(self, stage=None):
        self.train_ds = LatLonDS(self.train_root)
        self.val_ds   = LatLonDS(self.val_root)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.bs,
                          shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.bs,
                          num_workers=4, pin_memory=True)


# ─────────────────────────────── Lightning model ─────────────────────────────
class LatLonRegressor(pl.LightningModule):
    def __init__(self, backbone_ckpt: str, lat_mm, lon_mm, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        ckpt = torch.load(backbone_ckpt, map_location="cpu")
        yolo_net = ckpt["model"].model
        self.encoder = nn.Sequential(*yolo_net[:-1]).float()

        with torch.no_grad():
            f = self.encoder(torch.zeros(1, 3, 224, 224))
            if f.dim()>2: f = torch.flatten(f,1)
            feat_dim = f.size(1)

        self.reg_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256),      nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        self.criterion = nn.SmoothL1Loss()
        self.lr = lr

        self.lat_min, self.lat_max = lat_mm
        self.lon_min, self.lon_max = lon_mm

    def forward(self, x):
        return self.reg_head(self.encoder(x))

    def training_step(self, batch, _):
        x,y = batch
        loss = self.criterion(self(x), y)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        x,y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        inv = lambda v,mn,mx: v*(mx-mn)+mn
        p_orig = torch.stack([
            inv(pred[:,0], self.lat_min, self.lat_max),
            inv(pred[:,1], self.lon_min, self.lon_max)
        ], dim=1)
        y_orig = torch.stack([
            inv(y[:,0], self.lat_min, self.lat_max),
            inv(y[:,1], self.lon_min, self.lon_max)
        ], dim=1)
        mae_orig = nn.L1Loss()(p_orig, y_orig)
        self.log("val/mae_orig", mae_orig, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.1)
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val/loss"}


# ─────────────────────────────────── main ────────────────────────────────────
def main(args):
    # 1) read & filter train, then scale
    raw_train = pd.read_csv(args.train_csv)
    filt_train = filter_coords(raw_train)
    train_df, lat_mm, lon_mm = scale_df(filt_train)

    # 2) read & filter val, then scale using train's min/max
    raw_val = pd.read_csv(args.val_csv)
    # filt_val = filter_coords(raw_val)
    filt_val = raw_val.copy()
    val_df, _, _ = scale_df(filt_val, train_mm=lat_mm, train_lm=lon_mm)

    # 3) datamodule
    dm = LatLonDM(
        RootedDF(args.train_root, train_df),
        RootedDF(args.val_root,   val_df),
        args.batch_size
    )

    # 4) model
    model = LatLonRegressor(args.backbone_ckpt, lat_mm, lon_mm, lr=args.lr)
    if args.head_only:
        for p in model.encoder.parameters():
            p.requires_grad = False

    # 5) W&B
    init_wandb(args, model)
    logger = make_logger(args)

    # 6) Trainer
    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="reg-{epoch:02d}-{val/loss:.3f}",
        monitor="val/loss", mode="min", save_top_k=3
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.devices,
        precision="16-mixed",
        logger=logger,
        callbacks=[ckpt_cb, EarlyStopping("val/loss", 5, mode="min")],
        log_every_n_steps=10,
    )
    trainer.fit(model, dm)
    finish_wandb(ckpt_cb.best_model_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_root",    required=True)
    p.add_argument("--val_root",      required=True)
    p.add_argument("--train_csv",     required=True)
    p.add_argument("--val_csv",       required=True)
    p.add_argument("--backbone_ckpt", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--epochs",     type=int,   default=25)
    p.add_argument("--devices",    type=int,   default=1)
    p.add_argument("--head_only",  action="store_true")
    args = p.parse_args()

    for k in ("train_root","val_root","train_csv","val_csv","backbone_ckpt"):
        setattr(args, k, str(Path(getattr(args, k)).expanduser().resolve()))
    main(args)
