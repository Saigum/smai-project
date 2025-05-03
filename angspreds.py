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
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import DBSCAN

# ─── Helpers ────────────────────────────────────────────────────────────────
@rank_zero_only
def make_logger(args):
    return WandbLogger(
        project="angle-prediction",
        name=f"vit_bs{args.batch_size}_lr{args.lr}_img{args.imgsz}"
    )

@rank_zero_only
def init_wandb(args, model):
    wandb.init(
        project="angle-prediction",
        name=f"vit_bs{args.batch_size}_lr{args.lr}_img{args.imgsz}",
        config=vars(args),
    )
    wandb.watch(model, log="all", log_freq=50)

@rank_zero_only
def finish_wandb():
    wandb.finish()

class LabelAwareRotate:
    def __init__(self, max_deg=25):
        self.max_deg = max_deg
    def __call__(self, img, angle):
        delta = torch.empty(1).uniform_(-self.max_deg, self.max_deg).item()
        return F.rotate(img, delta), (angle + delta) % 360

# Convert to unit circle

def to_unit(vec):
    return vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)

# ─── Data Handling ─────────────────────────────────────────────────────────
class DFDataroot:
    def __init__(self, root_dir: str, dataframe: pd.DataFrame):
        self.root_dir = root_dir
        self.dataframe = dataframe

class AngleDataset(Dataset):
    def __init__(self,
                 root_df: DFDataroot,
                 imgsz: int,
                 augment=False):
        self.root_dir = root_df.root_dir
        self.df = root_df.dataframe.reset_index(drop=True)
        self.imgsz = imgsz
        self.augment = augment
        # base transforms
        self.norm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])
        if augment:
            self.rotate = LabelAwareRotate(max_deg=25)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.root_dir, row['filename'])).convert('RGB')
        angle = float(row['angle'])
        if self.augment:
            img, angle = self.rotate(img, angle)
        img = self.norm(img)
        # target sin/cos
        target = torch.tensor([
            torch.sin(torch.deg2rad(torch.tensor(angle))),
            torch.cos(torch.deg2rad(torch.tensor(angle)))
        ], dtype=torch.float32)
        return img, target

# ─── Model ──────────────────────────────────────────────────────────────────
class AnglePredictionModel(pl.LightningModule):
    def __init__(self, model_name: str, lr: float):
        super().__init__()
        self.save_hyperparameters()
        # backbone
        self.backbone = timm.create_model(model_name, pretrained=True)
        in_feats = self.backbone.head.in_features
        # 2D head for sin/cos
        self.backbone.head = torch.nn.Linear(in_feats, 2)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        return to_unit(self.backbone(x))

    def step(self, batch, stage: str):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # compute circular MAE
        pred_ang = torch.rad2deg(torch.atan2(y_hat[:,0], y_hat[:,1])) % 360
        true_ang = torch.rad2deg(torch.atan2(y[:,0], y[:,1])) % 360
        maae = torch.abs((pred_ang - true_ang + 180) % 360 - 180).mean()
        self.log(f'{stage}_loss', loss, prog_bar=False)
        self.log(f'{stage}_maae', maae, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=2, eta_min=self.hparams.lr * 1e-2
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {'scheduler': sched, 'interval': 'epoch'}
        }

# ─── DataModule ────────────────────────────────────────────────────────────
class AngleDataModule(pl.LightningDataModule):
    def __init__(self, train_root, val_root, batch_size, imgsz):
        super().__init__()
        self.train_root = train_root
        self.val_root = val_root
        self.batch_size = batch_size
        self.imgsz = imgsz

    def setup(self, stage=None):
        self.train_ds = AngleDataset(self.train_root, self.imgsz, augment=True)
        self.val_ds   = AngleDataset(self.val_root,   self.imgsz, augment=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=8)

# ─── Clustering ─────────────────────────────────────────────────────────────
def cluster_data(df, eps_degrees=0.0001, min_samples=2):
    coords = df[['latitude', 'longitude']].values
    labels = DBSCAN(eps=eps_degrees, min_samples=min_samples,
                    metric='euclidean', n_jobs=-1).fit_predict(coords)
    df2 = df.copy()
    df2['cluster_id'] = labels
    df2 = df2[df2['cluster_id'] != -1]
    df2['cluster_size'] = df2.groupby('cluster_id')['cluster_id'].transform('count')
    df2['duplicate_count'] = df2['cluster_size'] - 1
    return df2.drop(columns='cluster_size')

# ─── Main ──────────────────────────────────────────────────────────────────
def main(args):
    # Data prep
    train_df = pd.read_csv(args.train_csv).query('0<=angle<=360').copy()
    val_df   = pd.read_csv(args.val_csv).query('0<=angle<=360').copy()
    clustered_train_df = cluster_data(train_df)
    clustered_val_df   = cluster_data(val_df)

    train_root = DFDataroot(args.train_root, clustered_train_df)
    val_root   = DFDataroot(args.val_root,   clustered_val_df)

    dm = AngleDataModule(train_root, val_root,
                         batch_size=args.batch_size,
                         imgsz=args.imgsz)
    model = AnglePredictionModel(model_name=args.model_name,
                                  lr=args.lr)

    # W&B
    init_wandb(args, model)
    wandb_logger = make_logger(args)

    # Callbacks
    early_stop = EarlyStopping(monitor="val_maae", patience=3, mode="min")
    ckpt_cb = ModelCheckpoint(monitor="val_maae", dirpath="checkpoints/",
                               filename="best-{epoch:02d}-{val_maae:.4f}",
                               save_top_k=3, mode="min")

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu", devices=args.devices,
        logger=wandb_logger,
        callbacks=[early_stop, ckpt_cb],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        precision='16-mixed'
    )
    trainer.fit(model, dm)
    finish_wandb()

if __name__ == "__main__":
    p = argparse.ArgumentParser("Angle Prediction")
    p.add_argument("--train_root",   type=str, required=True)
    p.add_argument("--val_root",     type=str, required=True)
    p.add_argument("--train_csv",    type=str, required=True)
    p.add_argument("--val_csv",      type=str, required=True)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--epochs",       type=int, default=60)
    p.add_argument("--devices",      type=int, default=1)
    p.add_argument("--model_name",   type=str, default="vit_base_patch16_224")
    p.add_argument("--imgsz",        type=int, default=224)
    args = p.parse_args()
    main(args)
