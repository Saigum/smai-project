import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import torch
import timm
import wandb
import joblib
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

@rank_zero_only
def make_logger(args):
    return WandbLogger(project="angle-prediction", name=f"{args.model_name}_bs{args.batch_size}_lr{args.lr}", log_model=False, save_dir=args.model_save_dir)

tfm_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tfm_train = transforms.Compose([
    transforms.Resize(280),
    transforms.RandomResizedCrop(256, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

BASE_DIR = "."
TRAIN_DIR = os.path.join(BASE_DIR, "images_train/images_train")
VAL_DIR = os.path.join(BASE_DIR, "images_val/images_val")
TRAIN_CSV = os.path.join(BASE_DIR, "labels_train.csv")
VAL_CSV = os.path.join(BASE_DIR, "labels_val.csv")
MODEL_NAME = "convnext_small"
MODEL_SAVE = os.path.join(BASE_DIR, "latlongcheckpoints")
SCALER_TYPE = "robust"

def remove_outliers_mad(df, cols, thresh=4.0):
    out = df.copy()
    for c in cols:
        med = out[c].median()
        mad = np.median(np.abs(out[c] - med))
        if mad == 0:
            continue
        out = out[np.abs(out[c] - med) <= thresh * mad]
    return out

def data_preprocess(df):
    df = remove_outliers_mad(df, ["longitude", "latitude"], thresh=4.0)
    coords = df[["longitude", "latitude"]].values
    db = DBSCAN(eps=50, min_samples=10, metric="euclidean").fit(coords)
    df["dbscan_label"] = db.labels_
    df = df[df["dbscan_label"] != -1].drop(columns="dbscan_label")
    return df

def region_specific_dataframes(df):
    return {r: df[df["Region_ID"] == r].copy() for r in df["Region_ID"].unique()}

class DFDataroot:
    def __init__(self, root_dir, dataframe):
        self.root_dir = root_dir
        self.dataframe = dataframe

class LatLongDataset(Dataset):
    def __init__(self, root_df, transform, target_cols=["lat_scaled", "lon_scaled"]):
        self.root_dir = root_df.root_dir
        self.df = root_df.dataframe.reset_index(drop=True)
        self.transform = transform
        self.target_cols = target_cols
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["filename"])
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            img = Image.new("RGB", (256, 256))
        img = self.transform(img)
        target = torch.tensor([row[c] for c in self.target_cols], dtype=torch.float32)
        return img, target

class LatLongDataModule(pl.LightningDataModule):
    def __init__(self, train_root, val_root, batch_size=32, num_workers=4):
        super().__init__()
        self.train_root = train_root
        self.val_root = val_root
        self.batch_size = batch_size
        self.num_workers = num_workers
    def setup(self, stage=None):
        self.train_ds = LatLongDataset(self.train_root, transform=tfm_train)
        self.val_ds = LatLongDataset(self.val_root, transform=tfm_val)
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

class CoordPredictor(pl.LightningModule):
    def __init__(self, model_name="convnext_small", scaler=None, lr=1e-5, weight_decay=1e-4, t_0=10, t_mult=1):
        super().__init__()
        if "convnext" in model_name:
            m = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
            in_features = m.classifier[2].in_features
            m.classifier[2] = torch.nn.Linear(in_features, 2)
        elif "resnet" in model_name:
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features = m.fc.in_features
            m.fc = torch.nn.Linear(in_features, 2)
        elif "efficientnet" in model_name:
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = m.classifier[1].in_features
            m.classifier[1] = torch.nn.Linear(in_features, 2)
        else:
            m = timm.create_model(model_name, pretrained=True)
            m.head = torch.nn.Linear(m.head.in_features, 2)
        self.model = m
        self.scaler = scaler
        self.lr = lr
        self.weight_decay = weight_decay
        self.t_0 = t_0
        self.t_mult = t_mult
        self.criterion = torch.nn.MSELoss()
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = CosineAnnealingWarmRestarts(opt, T_0=self.t_0, T_mult=self.t_mult, eta_min=self.lr * 0.01)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch", "monitor": "val_loss"}}

@torch.no_grad()
def evaluate(model, loader, scaler, device):
    model.eval()
    mse_errs = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        if scaler is not None:
            y_hat_u = torch.tensor(scaler.inverse_transform(y_hat.cpu().numpy()), device=device)
            y_u = torch.tensor(scaler.inverse_transform(y.cpu().numpy()), device=device)
        else:
            y_hat_u = y_hat
            y_u = y
        mse_errs.append(torch.nn.functional.mse_loss(y_hat_u, y_u, reduction="none"))
    mse_errs = torch.cat(mse_errs, dim=0)
    mse = mse_errs.mean(dim=0)
    mse_val = mse.mean().item()
    # print(f"Value of MSE obtained is {mse_val}")
    return mse,mse_val

@rank_zero_only
def region_info(region,df):
    print(f"Processing region {region} with {len(df)} samples")
    print(f"Maximum latitude: {df['latitude'].max()}, Minimum latitude: {df['latitude'].min()}")
    print(f"Maximum longitude: {df['longitude'].max()}, Minimum longitude: {df['longitude'].min()}")

@rank_zero_only
def val_info(region,train_df_r, val_df_r, scaler):
    print(f"Training on region {region} with {len(train_df_r)} samples and validating on {len(val_df_r)} samples")
    print(f"Scaler Information: {scaler}")

@rank_zero_only
def evaluation_info(region, mse, mse_val):
    print(f"Region{region} , obrained mse value of {mse_val}")
    print(f"Region {region}: global validation MSE of [lat,long] = {mse}")
    

def main(args):
    os.makedirs(args.model_save_dir, exist_ok=True)
    train_df = pd.read_csv(args.train_csv)
    val_df_global = pd.read_csv(args.val_csv)

    exclude_rows= [95, 145, 146, 158, 159, 160, 161]
    ## the 4 digit version of the image name is used in the csv
    exclude_filenames = [f"image_{i:04d}.jpg" for i in exclude_rows]
    val_df_global = val_df_global[~val_df_global["filename"].isin(exclude_filenames)]
    

    train_df = data_preprocess(train_df)
    region_dfs = region_specific_dataframes(train_df)

    for region, df in region_dfs.items():
        lat_p1, lat_p99 = np.percentile(df["latitude"].values, [1, 99])
        lon_p1, lon_p99 = np.percentile(df["longitude"].values, [1, 99])
        mask = (df.latitude.between(lat_p1, lat_p99)) & (df.longitude.between(lon_p1, lon_p99))
        df = df[mask].reset_index(drop=True)
        if len(df) < 20:
            continue
        scaler = None
        if args.scaler == "robust":
            scaler = RobustScaler()
        elif args.scaler == "standard":
            scaler = StandardScaler()
        elif args.scaler == "minmax":
            scaler = MinMaxScaler(feature_range=(-1, 1))
        if scaler is not None:
            coords = df[["latitude", "longitude"]].values
            coords_scaled = scaler.fit_transform(coords)
            joblib.dump(scaler, os.path.join(os.path.join(args.model_save_dir,"scalers"), f"scaler_region_{region}.joblib"))
            df["lat_scaled"] = coords_scaled[:, 0]
            df["lon_scaled"] = coords_scaled[:, 1]
        else:
            df["lat_scaled"] = df["latitude"].values
            df["lon_scaled"] = df["longitude"].values
        train_df_r, val_df_r = train_test_split(df, test_size=0.15, random_state=42)

        dm = LatLongDataModule(train_root=DFDataroot(args.train_dir, train_df_r), val_root=DFDataroot(args.train_dir, val_df_r), batch_size=args.batch_size, num_workers=args.num_workers)
        dm.setup()
        model = CoordPredictor(model_name=args.model_name,
                                scaler=scaler, lr=args.lr, weight_decay=args.weight_decay, t_0=args.t_0, t_mult=args.t_mult)
        wandb_logger = make_logger(args)
        early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min", verbose=True)
        ckpt_dir = os.path.join(args.model_save_dir, f"region_{region}")
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, filename="region_{region}-{epoch}-{val_loss:.4f}", monitor="val_loss", mode="min", save_top_k=1)
        trainer = Trainer(max_epochs=args.epochs, accelerator="gpu" if args.num_gpus else "cpu", devices=args.num_gpus, strategy="ddp" if args.num_gpus > 1 else None, precision="16-mixed" if args.use_amp else "32-true", logger=wandb_logger, callbacks=[early_stop, checkpoint_callback], log_every_n_steps=50, enable_checkpointing=True, default_root_dir=args.model_save_dir)
        trainer.fit(model, dm)
        best_ckpt = checkpoint_callback.best_model_path
        if best_ckpt:
            model = CoordPredictor.load_from_checkpoint(best_ckpt, model_name=args.model_name, scaler=scaler, lr=args.lr, weight_decay=args.weight_decay, t_0=args.t_0, t_mult=args.t_mult)

        val_region = val_df_global[val_df_global["Region_ID"] == region].copy()
        if len(val_region) == 0:
            continue
        if scaler is not None:
            coords_val = val_region[["latitude", "longitude"]].values
            coords_val_scaled = scaler.transform(coords_val)
            val_region["lat_scaled"] = coords_val_scaled[:, 0]
            val_region["lon_scaled"] = coords_val_scaled[:, 1]
        else:
            val_region["lat_scaled"] = val_region["latitude"].values
            val_region["lon_scaled"] = val_region["longitude"].values
            
        val_ds = LatLongDataset(DFDataroot(args.val_dir, val_region), transform=tfm_val)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
        device = "cuda" if torch.cuda.is_available() and args.num_gpus else "cpu"
        model.to(device)
        mse,mse_val = evaluate(model, val_loader, scaler, device)

        evaluation_info(region, mse, mse_val)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--scaler", type=str, default=SCALER_TYPE)
    parser.add_argument("--train_dir", type=str, default=TRAIN_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)
    parser.add_argument("--train_csv", type=str, default=TRAIN_CSV)
    parser.add_argument("--val_csv", type=str, default=VAL_CSV)
    parser.add_argument("--model_save_dir", type=str, default=MODEL_SAVE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--t_0", type=int, default=10)
    parser.add_argument("--t_mult", type=int, default=1)
    args = parser.parse_args()
    main(args)