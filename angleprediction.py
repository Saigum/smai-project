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


@rank_zero_only
def make_logger(args):
    return WandbLogger(
        project="angle-prediction",
        name=f"vit_bs{args.batch_size}_lr{args.lr}"
    )



@rank_zero_only
def init_wandb(args,model):
    wandb.init(
        project="angle-prediction",
        name=f"vit_bs{args.batch_size}_lr{args.lr}",
        config=vars(args),
    )
    wandb.watch(model, log="all", log_freq=50)

@rank_zero_only
def finish_wandb():
    wandb.finish()


def angle_transform(angle: float) -> float:
    """Convert angles >180 into their smaller complement."""
    return min(angle, 360.0 - angle)


# image normalization for ViT
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

class DFDataroot:
    def __init__(self, root_dir: str, dataframe: pd.DataFrame):
        self.root_dir = root_dir
        self.dataframe = dataframe

class AngleDataset(Dataset):
    def __init__(self,
                 root_df: DFDataroot,
                 transform=image_transform,
                 target_transform=angle_transform):
        self.root_dir = root_df.root_dir
        self.df = root_df.dataframe.reset_index(drop=True)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        angle = float(row["angle"])
        if self.target_transform:
            angle = self.target_transform(angle)
        # return a float tensor for regression
        return img, torch.tensor(angle, dtype=torch.float32)

class AnglePredictionModel(pl.LightningModule):
    def __init__(self, pretrained: bool = True,
                  lr: float = 1e-4,
                  model_name: str = "vit_base_patch16_224"):
        super().__init__()
        self.save_hyperparameters()

        if model_name == "vit_base_patch16_224":
            # ViT backbone with single-output head
            self.backbone = timm.create_model(
                'vit_base_patch16_224', pretrained=pretrained)
            in_feats = self.backbone.head.in_features
            self.backbone.head = torch.nn.Linear(in_feats, 1)
        elif model_name == "efficientnet_b3.ra2_in1k":
            # EfficientNet backbone with single-output head
            self.backbone = timm.create_model(
                'efficientnet_b3.ra2_in1k', pretrained=pretrained)
            in_feats = self.backbone.classifier.in_features
            self.backbone.classifier = torch.nn.Linear(in_feats, 1)

        self.criterion = torch.nn.L1Loss()

    def forward(self, x):
        return self.backbone(x).squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
        return [opt], [sched]

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
        self.train_ds = AngleDataset(self.train_root)
        self.val_ds = AngleDataset(self.val_root)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          num_workers=4)

def data_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df['angle'].between(0, 360)].copy()

def main(args):
    # Prep data
    train_df = data_preprocess(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    train_root = DFDataroot(args.train_root, train_df)
    val_root   = DFDataroot(args.val_root,   val_df)
    dm = AngleDataModule(train_root, val_root, batch_size=args.batch_size)

    # Model
    model = AnglePredictionModel(pretrained=True,
                                 lr=args.lr,
                                 model_name=args.model_name)

    # W&B logger
    init_wandb(args, model)
    # Initialize W&B logger
    wandb_logger = make_logger(args)

    # Callbacks
    early_stop = EarlyStopping(monitor="val_loss",
                               patience=3,
                               mode="min",
                               verbose=True)
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
        prefix=f"{args.model_name}_",
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.devices,
        logger=wandb_logger,
        callbacks=[early_stop, ckpt_cb],
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)

    finish_wandb()

if __name__ == "__main__":
    p = argparse.ArgumentParser("Angle Prediction")
    p.add_argument("--train_root", type=str, required=True,
                   help="Folder of training images")
    p.add_argument("--val_root",   type=str, required=True,
                   help="Folder of validation images")
    p.add_argument("--train_csv",  type=str, required=True,
                   help="CSV file with train filenames & angles")
    p.add_argument("--val_csv",    type=str, required=True,
                   help="CSV file with val filenames & angles")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size")
    p.add_argument("--lr",         type=float, default=1e-4,
                   help="Learning rate")
    p.add_argument("--epochs",     type=int, default=10,
                   help="Number of epochs")
    p.add_argument("--devices",    type=int, default=1,
                   help="Number of GPUs to use")
    p.add_argument("--model_name", type=str, default="vit_base_patch16_224",
                   help="Model name (e.g., vit_base_patch16_224, efficientnet_b3.ra2_in1k)")
    args = p.parse_args()
    main(args)

