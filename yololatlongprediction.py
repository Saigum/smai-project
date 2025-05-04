import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from PIL import Image
import torch
import timm
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.cluster import DBSCAN
import wandb


# ---------------------------------------------------------------------
@rank_zero_only
def make_logger(args):
    return WandbLogger(
        project="latlon-prediction",
        name=f"{args.model_name}_bs{args.batch_size}_lr{args.lr}",
        log_model="all",
        save_dir="checkpoints",
    )

@rank_zero_only
def init_wandb(args, model):
    wandb.init(
        project="latlon-prediction",
        name=f"{args.model_name}_bs{args.batch_size}_lr{args.lr}",
        config=vars(args),
    )
    wandb.watch(model, log="all", log_freq=50)

@rank_zero_only
def finish_wandb():
    wandb.finish()
# ----------------------------------------------------------------------
# Data utilities
# ----------------------------------------------------------------------
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

class DFDataroot:
    def __init__(self, root_dir: str, dataframe: pd.DataFrame):
        self.root_dir = root_dir
        self.dataframe = dataframe.reset_index(drop=True)

# ------------- datasets ------------------------------------------------
class LatLonDataset(Dataset):
    def __init__(self, root_df: DFDataroot, transform=image_transform):
        self.root   = root_df.root_dir
        self.df     = root_df.dataframe
        self.transform = transform

    def __len__(self):  # noqa: D401
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.root, row["filename"])
        img      = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        target = torch.tensor(
            [row["latitude"], row["longitude"]], dtype=torch.float32
        )
        return img, target



### load pretrained model
def load_model(model_path):
    # Load the YOLOv8 model
    model = YOLO(model_path)
    return model

# ightning module
# ----------------------------------------------------------------------
class LatLonModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        lr: float = 1e-4,
        use_contrastive: bool = True,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.encoder.num_features
        self.reg_head = torch.nn.Linear(feat_dim, 2)

        self.use_contrastive = use_contrastive
        if use_contrastive:
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(feat_dim, proj_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(proj_dim, proj_dim),
            )
            self.triplet = torch.nn.TripletMarginWithDistanceLoss(
                distance_function=torch.nn.PairwiseDistance(),
                margin=1.0,
            )

        self.reg_loss = torch.nn.SmoothL1Loss()
        self.lr = lr

    # ------------- helpers --------------------------------------------------

    def forward(self, x):
        return self.reg_head(self.encode(x))

    # ------------- training / validation ------------------------------------
    def training_step(self, batch, _):
        x, t = batch
        loss = self.reg_loss(self(x), t)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        x, t = batch
        loss = self.reg_loss(self(x), t)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    # ------------- optim -----------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.1)
        return [opt], [sch]
# ----------------------------------------------------------------------
# Lightning data modules
# ----------------------------------------------------------------------
class LatLonDataModule(pl.LightningDataModule):
    def __init__(self, train_root, val_root, batch_size=8, contrastive=False):
        super().__init__()
        self.train_root, self.val_root = train_root, val_root
        self.bs, self.contrastive = batch_size, contrastive

    def setup(self, stage=None):
        DS = LatLonDataset
        self.train_ds = DS(self.train_root)
        self.val_ds   = DS(self.val_root)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.bs,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.bs,
            num_workers=4,
            pin_memory=True,
        )
# ----------------------------------------------------------------------
# Data-frame sanity
# ----------------------------------------------------------------------
def load_and_sanitize(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["angle"]<360]  # drop invalid angles
    return df
# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main(args):
    # 1) load data
    train_df = load_and_sanitize(args.train_csv)
    val_df   = load_and_sanitize(args.val_csv)
    print(f" Train : {len(train_df)}", f"{len(val_df)} Val", sep="\n")

    # 3) dataroot objects
    train_root      = DFDataroot(args.train_root, train_df)
    val_root        = DFDataroot(args.val_root  , val_df)
    # 4) lightning data-modules
    dm      = LatLonDataModule(train_root,      val_root,      args.batch_size)

    # 5) model
    model = LatLonModel(
        model_name=args.model_name,
        pretrained=True,
        lr=args.lr,
        use_contrastive=True,
    )

    # 6) W&B
    init_wandb(args, model)
    logger = make_logger(args)

    # 7) callbacks
    early_stop = EarlyStopping("val/loss", patience=5, verbose=True, mode="min")
    ckpt = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{args.model_name}-{{epoch:02d}}-{{val/loss:.3f}}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
    )



    # 9) head-only fine-tune?
    if args.head_only:
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.reg_head.parameters():
            p.requires_grad = True

    # 10) full fine-tuning
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.devices,
        logger=logger,
        callbacks=[early_stop, ckpt],
        log_every_n_steps=10,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else None,
    )
    trainer.fit(model, dm)

    finish_wandb()

# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Latitude & Longitude Prediction")
    parser.add_argument("--train_root", type=str, required=True,
                        help="Folder containing training images")
    parser.add_argument("--val_root",   type=str, required=True,
                        help="Folder containing validation images")
    parser.add_argument("--train_csv",  type=str, required=True,
                        help="CSV with filename, latitude, longitude")
    parser.add_argument("--val_csv",    type=str, required=True,
                        help="CSV with filename, latitude, longitude")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--head_only",  action="store_true",
                        help="Freeze encoder and train only regression head")
    parser.add_argument("--devices",    type=int,   default=1)
    parser.add_argument("--model_name", type=str,   default="vit_base_patch16_224")
    parser.add_argument("--eps_meters", type=float, default=20.0,
                        help="DBSCAN ε in metres for clustering")
    args = parser.parse_args()

    # absolute paths are safer inside SLURM / W&B jobs
    for pth in ("train_root", "val_root", "train_csv", "val_csv"):
        setattr(args, pth, str(Path(getattr(args, pth)).expanduser().resolve()))

    main(args)
