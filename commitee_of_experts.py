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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import DBSCAN # Keep import in case needed later, but commented out in logic
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

@rank_zero_only
def make_logger(args):
    # Make sure the log directory exists before creating logger
    os.makedirs(args.model_save_dir, exist_ok=True)
    log_dir = os.path.join(args.model_save_dir, "wandb_logs")
    os.makedirs(log_dir, exist_ok=True)
    return WandbLogger(project="angle-prediction", name=f"{args.model_name}_bs{args.batch_size}_lr{args.lr}", log_model=False, save_dir=log_dir) # Changed save_dir

tfm_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Modified augmentations: slightly reduced ColorJitter, removed RandomErasing
tfm_train = transforms.Compose([
    transforms.Resize(280),
    transforms.RandomResizedCrop(256, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15), # Slightly less rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05), # Slightly less jitter
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)), # Removed RandomErasing initially
])

BASE_DIR = "."
TRAIN_DIR = os.path.join(BASE_DIR, "images_train/images_train")
VAL_DIR = os.path.join(BASE_DIR, "images_val/images_val")
TRAIN_CSV = os.path.join(BASE_DIR, "labels_train.csv")
VAL_CSV = os.path.join(BASE_DIR, "labels_val.csv")
MODEL_NAME = "convnext_small"
MODEL_SAVE = os.path.join(BASE_DIR, "latlongcheckpoints")
SCALER_TYPE = "robust"

def remove_outliers_mad(df, cols, thresh=3.0):
    out = df.copy()
    for c in cols:
        med = out[c].median()
        mad = np.median(np.abs(out[c] - med))
        if mad == 0:
            continue
        out = out[np.abs(out[c] - med) <= thresh * mad].copy() # Use .copy() to avoid SettingWithCopyWarning
    return out

# Modified data_preprocess: Removed hardcoded lat/lon filtering
def data_preprocess(df):
    df = remove_outliers_mad(df, ["longitude", "latitude"], thresh=3.0)
    # DBSCAN commented out as in original, keep it commented unless needed
    # coords = df[["longitude", "latitude"]].values
    # db = DBSCAN(eps=50, min_samples=10, metric="euclidean").fit(coords)
    # df["dbscan_label"] = db.labels_
    # df = df[df["dbscan_label"] != -1].drop(columns="dbscan_label")
    df = df[df["angle"].between(0, 360)].copy() # Use .copy()
    # Removed explicit lat/lon range filtering here. This will now be handled per region if needed.
    print(f"Dataframe shape after general preprocessing: {df.shape}")
    return df

def region_specific_dataframes(df):
    """Modified to match File 1's filtering approach and ensure copy"""
    regions = {}
    IQR_MULTIPLIER = 1.5
    REGIONS_IQR_FILTER = [2, 3, 10, 15] # Regions needing special IQR filtering

    for r in df["Region_ID"].unique():
        dfr = df[df["Region_ID"] == r].copy() # Ensure copy immediately after filtering by Region_ID

        # Special handling for Region 8
        if r == 8:
            LAT_THRESHOLD_MIN_R8 = 180000
            LON_THRESHOLD_MIN_R8 = 100000
            dfr = dfr[
                (dfr['latitude'] > LAT_THRESHOLD_MIN_R8) &
                (dfr['longitude'] > LON_THRESHOLD_MIN_R8)
            ].copy() # Ensure copy

        # IQR filtering for specific regions
        elif r in REGIONS_IQR_FILTER:
            try:
                coords = dfr[['latitude', 'longitude']].values.astype(np.float64)
                if len(coords) > 1: # Ensure enough data for percentiles
                    q1_lat, q3_lat = np.nanpercentile(coords[:, 0], [25, 75])
                    q1_lon, q3_lon = np.nanpercentile(coords[:, 1], [25, 75])
                    iqr_lat = q3_lat - q1_lat
                    iqr_lon = q3_lon - q1_lon

                    lower_lat = q1_lat - IQR_MULTIPLIER * iqr_lat
                    upper_lat = q3_lat + IQR_MULTIPLIER * iqr_lat
                    lower_lon = q1_lon - IQR_MULTIPLIER * iqr_lon
                    upper_lon = q3_lon + IQR_MULTIPLIER * iqr_lon

                    dfr = dfr[
                        dfr['latitude'].between(lower_lat, upper_lat) &
                        dfr['longitude'].between(lower_lon, upper_lon)
                    ].copy() # Ensure copy
                else:
                    print(f"Warning: Skipping IQR filtering for Region {r} due to insufficient data ({len(coords)} samples).")
            except Exception as e:
                print(f"Warning: IQR filtering failed for Region {r}: {e}")

        # Standard 1-99% percentile clipping for all regions
        try:
            if len(dfr) > 1: # Ensure enough data for percentiles
                lat_p1, lat_p99 = np.percentile(dfr['latitude'], [1, 99])
                lon_p1, lon_p99 = np.percentile(dfr['longitude'], [1, 99])
                dfr = dfr[
                    dfr['latitude'].between(lat_p1, lat_p99) &
                    dfr['longitude'].between(lon_p1, lon_p99)
                ].copy() # Ensure copy
            else:
                 print(f"Warning: Skipping percentile clipping for Region {r} due to insufficient data ({len(dfr)} samples).")
        except Exception as e:
            print(f"Warning: Percentile clipping failed for Region {r}: {e}")

        if len(dfr) >= 20: # Only keep regions with sufficient samples
            regions[r] = dfr
        else:
             print(f"Warning: Region {r} has only {len(dfr)} samples after filtering, skipping.")

    return regions

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
        # Ensure image path is correctly constructed - assumes filename is relative to root_dir
        img_path = os.path.join(self.root_dir, row["filename"])
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Handle missing images gracefully, e.g., return a black image or skip
            # Returning a black image might affect training, skipping might be better
            # For now, keeping the black image behavior
            print(f"Warning: Image not found at {img_path}. Returning black image.")
            img = Image.new("RGB", (256, 256)) # Use a dummy image
        except Exception as e:
             print(f"Error loading image {img_path}: {e}. Returning black image.")
             img = Image.new("RGB", (256, 256)) # Handle other image loading errors

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
        self.save_hyperparameters(ignore=["train_root", "val_root"]) # Log arguments except data roots

    def setup(self, stage=None):
        # Check if dataframes are not empty before creating datasets
        if not self.train_root.dataframe.empty:
            self.train_ds = LatLongDataset(self.train_root, transform=tfm_train)
        else:
            self.train_ds = None
            print("Warning: Training dataframe is empty. Cannot create training dataset.")

        if not self.val_root.dataframe.empty:
             self.val_ds = LatLongDataset(self.val_root, transform=tfm_val)
        else:
            self.val_ds = None
            print("Warning: Validation dataframe is empty. Cannot create validation dataset.")


    def train_dataloader(self):
        if self.train_ds is None: return None
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        if self.val_ds is None: return None
        # Use shuffle=False for validation to ensure consistent evaluation order
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

class CoordPredictor(pl.LightningModule):
    def __init__(self, model_name="convnext_small", scaler=None, lr=1e-5, weight_decay=1e-4, t_0=10, t_mult=1, unfreeze_epochs=5):
        super().__init__()
        self.save_hyperparameters() # Saves all init arguments

        if "convnext" in model_name:
            m = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
            self.feature_extractor = m.features # Features part
            in_features = m.classifier[2].in_features
            self.classifier = torch.nn.Linear(in_features, 2) # Classifier part
            # Initial state: Freeze feature extractor
            # for param in self.feature_extractor.parameters():
            #      param.requires_grad = False
        elif "resnet" in model_name:
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feature_extractor = torch.nn.Sequential(*list(m.children())[:-1]) # Features part (remove last layer)
            in_features = m.fc.in_features
            self.classifier = torch.nn.Linear(in_features, 2) # Classifier part
             # Initial state: Freeze feature extractor
            for param in self.feature_extractor.parameters():
                 param.requires_grad = False
        elif "efficientnet" in model_name:
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.feature_extractor = m.features # Features part
            in_features = m.classifier[1].in_features
            self.classifier = torch.nn.Linear(in_features, 2) # Classifier part
             # Initial state: Freeze feature extractor
            for param in self.feature_extractor.parameters():
                 param.requires_grad = False
        else: # Using timm model
            m = timm.create_model(model_name, pretrained=True, num_classes=0) # Load features only
            self.feature_extractor = m
            # Get in_features for the head from timm model's global pool output
            try:
                # This might vary slightly by timm model
                dummy_input = torch.randn(1, 3, 256, 256)
                with torch.no_grad():
                     in_features = self.feature_extractor(dummy_input).shape[-1]
            except Exception as e:
                print(f"Could not automatically determine head input features for {model_name}. Error: {e}")
                # Fallback or raise error - check timm docs for typical head input size
                # Common is 1024 or 1280 or 2048 depending on model size
                # For safety, you might need to hardcode this or find a reliable way
                # Let's assume a common size for now, but this is risky.
                # A better way is to get the feature size from the model config.
                # For convnext_small, timm output feature size before head is 768
                if "convnext_small" in model_name: in_features = 768
                elif "resnet50" in model_name: in_features = 2048
                elif "efficientnet_b0" in model_name: in_features = 1280
                else: raise ValueError(f"Need to manually specify in_features for model {model_name}")
            self.classifier = torch.nn.Linear(in_features, 2) # Classifier part
            # Initial state: Freeze feature extractor
            # for param in self.feature_extractor.parameters():
            #      param.requires_grad = False


        self.scaler = scaler
        self.lr = lr
        self.weight_decay = weight_decay
        self.t_0 = t_0
        self.t_mult = t_mult
        self.unfreeze_epochs = unfreeze_epochs

        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        features = self.feature_extractor(x)
        if features.ndim == 4: # Handle cases where feature extractor might return spatial dims (like ResNet before AvgPool)
             features = torch.nn.functional.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_scaled = batch
        y_hat_scaled = self(x)
        loss_scaled = self.criterion(y_hat_scaled, y_scaled)
        self.log("val_loss_scaled", loss_scaled, prog_bar=True, sync_dist=True)
        return loss_scaled # Primary metric for checkpointing/early stopping remains scaled loss

    def configure_optimizers(self):
        # Only optimize parameters that require gradients
        params_to_optimize = filter(lambda p: p.requires_grad, self.parameters())
        opt = torch.optim.AdamW(params_to_optimize, lr=self.lr, weight_decay=self.weight_decay)
        sched = CosineAnnealingWarmRestarts(opt, T_0=self.t_0, T_mult=self.t_mult, eta_min=self.lr * 0.01)
        # Monitor the unscaled validation loss for the scheduler if scaler is used, otherwise scaled
        monitor_metric = "val_loss_scaled" 
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch", "monitor": monitor_metric}}


    # def on_train_epoch_start(self):
    #     # Unfreeze backbone after specified epochs
    #     if self.current_epoch == self.unfreeze_epochs:
    #         print(f"Epoch {self.current_epoch}: Unfreezing feature extractor.")
    #         for param in self.feature_extractor.parameters():
    #             param.requires_grad = True
            
            ## confirm as to whether th


@torch.no_grad()
def evaluate(model, loader, scaler, device):
    model.eval()
    mse_errs = []
    # Ensure model is on the correct device
    model.to(device)
    for x, y_scaled in loader:
        x = x.to(device)
        y_hat_scaled = model(x)

        if scaler is not None:
            # Convert predictions and targets back to original scale on CPU before calculating error
            y_hat_u = scaler.inverse_transform(y_hat_scaled.cpu().numpy())
            y_u = scaler.inverse_transform(y_scaled.cpu().numpy())

            mse_errs.append(np.mean((y_hat_u - y_u)**2, axis=1)) # MSE per sample
        else:
            # If no scaler, calculate MSE directly
            y_hat_u = y_hat_scaled.cpu().numpy()
            y_u = y_scaled.cpu().numpy()
            mse_errs.append(np.mean((y_hat_u - y_u)**2, axis=1)) # MSE per sample

    # Concatenate results from all batches
    mse_errs_per_sample = np.concatenate(mse_errs, axis=0) # shape (num_samples,)
    # Calculate mean MSE across all samples
    mse_val = np.mean(mse_errs_per_sample)
    model.eval()
    mse_errs_coords = []
    for x, y_scaled in loader:
        x = x.to(device)
        y_hat_scaled = model(x)
        if scaler is not None:
            y_hat_u = scaler.inverse_transform(y_hat_scaled.cpu().numpy())
            y_u = scaler.inverse_transform(y_scaled.cpu().numpy())
            mse_errs_coords.append((y_hat_u - y_u)**2) # Squared error per coordinate per sample
        else:
            y_hat_u = y_hat_scaled.cpu().numpy()
            y_u = y_scaled.cpu().numpy()
            mse_errs_coords.append((y_hat_u - y_u)**2)

    mse_errs_coords_arr = np.concatenate(mse_errs_coords, axis=0) # shape (num_samples, 2)
    mse_per_coord = np.mean(mse_errs_coords_arr, axis=0) # shape (2,) -> [MSE_lat, MSE_lon]

    return mse_per_coord, mse_val # Return both per-coordinate and overall mean MSE

@rank_zero_only
def region_info(region,df):
    print(f"Processing region {region} with {len(df)} samples")
    if not df.empty:
        print(f"Maximum latitude: {df['latitude'].max()}, Minimum latitude: {df['latitude'].min()}")
        print(f"Maximum longitude: {df['longitude'].max()}, Minimum longitude: {df['longitude'].min()}")
    else:
         print("Dataframe is empty.")

@rank_zero_only
def val_info(region,train_df_r, val_df_r, scaler):
    print(f"Training on region {region} with {len(train_df_r)} samples and validating on {len(val_df_r)} samples")
    # print(f"Scaler Information: {scaler}") # Scaler info can be long, maybe skip printing object

@rank_zero_only
def evaluation_info(region, mse_per_coord, mse_val, val_len):
    print(f"Region {region}: Overall validation MSE = {mse_val:.4f}")
    print(f"Region {region}: Validation MSE [lat, lon] = [{mse_per_coord[0]:.4f}, {mse_per_coord[1]:.4f}]")
    print("-" * 20)
    ##writing this to a text file
    eval_file_path = os.path.join(MODEL_SAVE, "evaluation.txt")
    with open(eval_file_path, "a") as f:
        f.write(f"Region {region}: Overall validation MSE = {mse_val:.4f} (Samples: {val_len})\n")
        f.write(f"Region {region}: Validation MSE [lat, lon] = [{mse_per_coord[0]:.4f}, {mse_per_coord[1]:.4f}]\n")
        f.write("-" * 20 + "\n")


def main(args):
    # Ensure model save dir exists and create scaler specific directory
    os.makedirs(args.model_save_dir, exist_ok=True)
    scaler_save_dir = os.path.join(args.model_save_dir, "scalers")
    os.makedirs(scaler_save_dir, exist_ok=True)
    # Clear previous evaluation file
    eval_file_path = os.path.join(args.model_save_dir, "evaluation.txt")
    if os.path.exists(eval_file_path):
        os.remove(eval_file_path)
    print("Loading and preprocessing data...")
    train_df = pd.read_csv(args.train_csv)
    val_df_global = pd.read_csv(args.val_csv)
    exclude_rows= [95, 145, 146, 158, 159, 160, 161]
    exclude_filenames = [f"image_{i:04d}.jpg" for i in exclude_rows]
    val_df_global = val_df_global[~val_df_global["filename"].isin(exclude_filenames)].copy() # Use .copy()
    train_df_processed = data_preprocess(train_df)
    print("Splitting data into regions and applying region-specific filtering...")
    region_dfs = region_specific_dataframes(train_df_processed)
    overall_weighted_mse_sum = 0
    total_val_samples = 0

    print(f"Starting training loop for {len(region_dfs)} regions...")

    # Train and evaluate models for each region
    for region, df in region_dfs.items():
        region_info(region, df)
        train_df_r, val_df_r = train_test_split(df, test_size=0.15, random_state=42, shuffle=True) # Use shuffle=True
        print(f"Region {region}: Train samples = {len(train_df_r)}, Validation samples (regional split) = {len(val_df_r)}")


        # Fit scaler *only* on the regional training data
        scaler = None
        if args.scaler == "robust":
            scaler = RobustScaler()
        elif args.scaler == "standard":
            scaler = StandardScaler()
        elif args.scaler == "minmax":
            scaler = MinMaxScaler(feature_range=(-1, 1))

        if scaler is not None:
            if not train_df_r.empty:
                coords_train = train_df_r[["latitude", "longitude"]].values
                coords_train_scaled = scaler.fit_transform(coords_train)
                train_df_r["lat_scaled"] = coords_train_scaled[:, 0]
                train_df_r["lon_scaled"] = coords_train_scaled[:, 1]
                # Transform the regional validation set using the scaler fitted on train_df_r
                if not val_df_r.empty:
                    coords_val_r = val_df_r[["latitude", "longitude"]].values
                    coords_val_r_scaled = scaler.transform(coords_val_r)
                    val_df_r["lat_scaled"] = coords_val_r_scaled[:, 0]
                    val_df_r["lon_scaled"] = coords_val_r_scaled[:, 1]
                joblib.dump(scaler, os.path.join(scaler_save_dir, f"scaler_region_{region}.joblib"))
                print(f"Region {region}: Scaler ({args.scaler}) fitted on {len(train_df_r)} training samples and saved.")
            else:
                print(f"Warning: Region {region} training data is empty, skipping scaler fitting.")
                scaler = None # Ensure scaler is None if fitting failed
                train_df_r["lat_scaled"] = train_df_r["latitude"].values
                train_df_r["lon_scaled"] = train_df_r["longitude"].values
                val_df_r["lat_scaled"] = val_df_r["latitude"].values
                val_df_r["lon_scaled"] = val_df_r["longitude"].values

        else:
            # If no scaler, use original coordinates as targets
            train_df_r["lat_scaled"] = train_df_r["latitude"].values
            train_df_r["lon_scaled"] = train_df_r["longitude"].values
            val_df_r["lat_scaled"] = val_df_r["latitude"].values
            val_df_r["lon_scaled"] = val_df_r["longitude"].values
            print(f"Region {region}: No scaler used.")

        val_info(region, train_df_r, val_df_r, scaler)

        # Create DataModule for the current region's train/validation split
        dm = LatLongDataModule(train_root=DFDataroot(args.train_dir, train_df_r),
                               val_root=DFDataroot(args.train_dir, val_df_r), # Use train_dir as root for regional val data
                               batch_size=args.batch_size,
                               num_workers=args.num_workers)
        dm.setup()

        # Skip training if training data is empty
        if dm.train_ds is None or len(dm.train_ds) == 0:
            print(f"Region {region}: Skipping training due to no training samples.")
            continue
        # Skip training if regional validation data is empty (for early stopping/checkpointing)
        if dm.val_ds is None or len(dm.val_ds) == 0:
             print(f"Region {region}: Skipping training due to no regional validation samples.")
             continue


        model = CoordPredictor(model_name=args.model_name,
                               scaler=scaler, # Pass the fitted scaler to the model
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               t_0=args.t_0,
                               t_mult=args.t_mult,
                               unfreeze_epochs=args.unfreeze_epochs)

        wandb_logger = make_logger(args)
        # Monitor unscaled loss for early stopping and checkpointing if scaler is used
        monitor_metric = "val_loss_scaled"
        early_stop = EarlyStopping(monitor=monitor_metric, patience=args.patience, mode="min", verbose=True) # Set verbose=True
        ckpt_dir = os.path.join(args.model_save_dir, f"region_{region}")
        os.makedirs(ckpt_dir, exist_ok=True)
        # Filename includes the monitored metric
        checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, filename=f"region_{region}-{{epoch}}-{{{monitor_metric}:.4f}}", monitor=monitor_metric, mode="min", save_top_k=1)
        lr_monitor = LearningRateMonitor(logging_interval='epoch') # Monitor learning rate

        trainer = Trainer(max_epochs=args.epochs,
                          accelerator="gpu" if args.num_gpus else "cpu",
                          devices=args.num_gpus,
                          strategy="ddp" if args.num_gpus > 1 else "auto", # Use "auto" for single GPU or CPU
                          precision="16-mixed" if args.use_amp and args.num_gpus else "32-true", # Only use AMP with GPU
                          logger=wandb_logger,
                          callbacks=[early_stop, checkpoint_callback, lr_monitor],
                          log_every_n_steps=args.log_steps, # Log more frequently
                          enable_checkpointing=True,
                          default_root_dir=args.model_save_dir,
                          # Add gradient clipping to prevent exploding gradients
                          gradient_clip_val=args.gradient_clip_val,
                          gradient_clip_algorithm="norm" if args.gradient_clip_val > 0 else "value" # "norm" is more common
                         )

        print(f"Region {region}: Starting training...")
        trainer.fit(model, dm)
        print(f"Region {region}: Training finished.")

        # --- Evaluation on Global Validation Data (filtered by region) ---
        # This evaluates the best model trained on region R's train data on the
        # corresponding subset of the *original* global validation data.

        # Load the best model checkpoint
        best_ckpt = checkpoint_callback.best_model_path
        if not best_ckpt:
             print(f"Region {region}: No best checkpoint found for region {region}. Skipping global validation.")
             continue # Skip evaluation if training failed or no checkpoint saved

        print(f"Region {region}: Loading best model from {best_ckpt}")
        # Load the model with the scaler object (it was saved in self.hparams)
        model = CoordPredictor.load_from_checkpoint(best_ckpt, scaler=scaler, strict=False) # strict=False allows loading when hparams mismatch slightly

        # Prepare the validation data for the *global* validation set, filtered by region
        val_region_global = val_df_global[val_df_global["Region_ID"] == region].copy() # Ensure copy
        print(f"Region {region}: Evaluating on {len(val_region_global)} global validation samples.")

        if len(val_region_global) == 0:
            print(f"Region {region}: No samples in global validation set for this region. Skipping evaluation.")
            continue

        # Scale the global validation data using the scaler *fitted on the regional training data*
        if scaler is not None:
            coords_val_global = val_region_global[["latitude", "longitude"]].values
            coords_val_global_scaled = scaler.transform(coords_val_global)
            val_region_global["lat_scaled"] = coords_val_global_scaled[:, 0]
            val_region_global["lon_scaled"] = coords_val_global_scaled[:, 1]
        else:
             val_region_global["lat_scaled"] = val_region_global["latitude"].values
             val_region_global["lon_scaled"] = val_region_global["longitude"].values


        val_ds_global = LatLongDataset(DFDataroot(args.val_dir, val_region_global), transform=tfm_val)
        val_loader_global = DataLoader(val_ds_global, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False) # No shuffle for evaluation

        device = "cuda" if torch.cuda.is_available() and args.num_gpus else "cpu"
        model.to(device)

        # Evaluate the model on the global validation data for this region
        print(f"Region {region}: Running evaluation on global validation data...")
        # evaluate function now returns mse_per_coord and overall mse_val
        mse_per_coord, mse_val = evaluate(model, val_loader_global, scaler, device)

        # Log evaluation results
        evaluation_info(region, mse_per_coord, mse_val, val_len=len(val_region_global))

        # Accumulate for overall weighted average
        overall_weighted_mse_sum += mse_val * len(val_region_global)
        total_val_samples += len(val_region_global)

    # Calculate and print overall validation MSE (weighted by sample count per region)
    if total_val_samples > 0:
        overall_mse = overall_weighted_mse_sum / total_val_samples
        print("\n" + "=" * 30)
        print(f"Overall weighted validation MSE across all regions: {overall_mse:.4f}")
        print("=" * 30)
        # Write overall MSE to evaluation file
        with open(eval_file_path, "a") as f:
             f.write("\n" + "=" * 30 + "\n")
             f.write(f"Overall weighted validation MSE across all regions: {overall_mse:.4f} (Total Samples: {total_val_samples})\n")
             f.write("=" * 30 + "\n")
    else:
        print("\nNo validation samples processed across any region.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Model architecture from torchvision or timm.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation.")
    # Increased default LR slightly
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer.")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum number of epochs to train.") # Increased epochs
    parser.add_argument("--patience", type=int, default=15, help="Patience for early stopping.") # Increased patience
    parser.add_argument("--scaler", type=str, default=SCALER_TYPE, choices=["robust", "standard", "minmax", None], help="Type of scaler to use for coordinates.")
    parser.add_argument("--train_dir", type=str, default=TRAIN_DIR, help="Directory containing training images.")
    parser.add_argument("--val_dir", type=str, default=VAL_DIR, help="Directory containing validation images.")
    parser.add_argument("--train_csv", type=str, default=TRAIN_CSV, help="Path to the training CSV file.")
    parser.add_argument("--val_csv", type=str, default=VAL_CSV, help="Path to the validation CSV file.")
    parser.add_argument("--model_save_dir", type=str, default=MODEL_SAVE, help="Directory to save model checkpoints and logs.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (0 for CPU).")
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision (AMP).")
    # Adjusted scheduler parameters
    parser.add_argument("--t_0", type=int, default=15, help="T_0 parameter for CosineAnnealingWarmRestarts.") # Slightly increased T_0
    parser.add_argument("--t_mult", type=int, default=2, help="T_mult parameter for CosineAnnealingWarmRestarts.") # Increased T_mult
    # Added unfreeze epochs parameter
    parser.add_argument("--unfreeze_epochs", type=int, default=5, help="Number of epochs to train head only before unfreezing backbone.")
    parser.add_argument("--log_steps", type=int, default=20, help="Log training loss every N steps.") # Log more frequently
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value (0 for no clipping).") # Added gradient clipping

    args = parser.parse_args()

    # Print effective settings
    print("\n--- Effective Settings ---")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("--------------------------\n")

    main(args)
