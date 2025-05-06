**LatLong Region-wise Coordinate Prediction**

This project implements a region-specific latitude-longitude prediction pipeline using PyTorch Lightning and ConvNeXt (or other backbones) for regression. The script reads train and validation CSVs with filenames and coordinates, preprocesses data by removing outliers (MAD and DBSCAN), scales coordinates per region, trains separate models per region, and evaluates on a held-out validation set.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Data Preparation](#data-preparation)
6. [Usage](#usage)
7. [Arguments](#arguments)
8. [Output](#output)
9. [Evaluation](#evaluation)
10. [Logging](#logging)
11. [License](#license)

---

## Project Structure

```
├── images_train/            # Training images directory
│   └── images_train/       # Nested image files
├── images_val/              # Validation images directory
│   └── images_val/         # Nested image files
├── labels_train.csv         # CSV with ['filename', 'latitude', 'longitude', 'Region_ID']
├── labels_val.csv           # Global validation CSV
├── latlong_predict.py       # Main training and evaluation script
├── latlongcheckpoints/      # Output directory for region-specific checkpoints and scalers
└── README.md                # This file
```

## Conceptual Overview

This project takes a **region-wise** approach for predicting geographic coordinates (latitude, longitude) from satellite-style images. Key ideas:

* **Outlier Filtering**: Remove extreme coordinate values with Median Absolute Deviation (MAD) and then cluster with DBSCAN to drop noise points.
* **Region Partitioning**: Data is grouped by `Region_ID`, enabling separate models tailored to local distributions and geographic characteristics.
* **Coordinate Scaling**: Coordinates in each region are normalized independently (Robust, Standard, or MinMax scaling) to stabilize model training.
* **Backbone Flexibility**: Modern convolutional or transformer backbones (ConvNeXt, ResNet, EfficientNet, or any timm model) output two regression targets \[lat\_scaled, lon\_scaled].
* **Training & Evaluation Loop**: For each region:

  1. Filter and scale coordinates.
  2. Split into train/validation subsets.
  3. Instantiate a PyTorch Lightning `CoordPredictor` with chosen backbone.
  4. Train with mixed precision or DDP if available.
  5. Log metrics & save best checkpoint to Weights & Biases.
  6. Evaluate on held-out validation set for that region.

---

## Code Structure

```
latlong_predict.py            # Main script containing all classes and logic
 ├── remove_outliers_mad()    # MAD-based filtering utility
 ├── data_preprocess()        # Applies MAD + DBSCAN, drops noisy samples
 ├── region_specific_dataframes() # Splits DataFrame by Region_ID
 ├── DFDataroot               # Simple container for root_dir + DataFrame
 ├── LatLongDataset           # PyTorch Dataset reading images + scaled targets
 ├── LatLongDataModule        # LightningDataModule for train/val loaders
 ├── CoordPredictor           # LightningModule wrapping backbone + MSE loss
 ├── evaluate()               # Offline evaluation with optional inverse scaling
 ├── region_info / val_info / evaluation_info  # Logging helpers
 └── main()                   # Orchestrates data load, region loop, training & eval
```

* **Utilities**: Filtering, scaling (joblib dump), logging functions.
* **Data Classes**: `LatLongDataset` and `LatLongDataModule` handle image transforms and batching.
* **Model Class**: `CoordPredictor` dynamically adapts to chosen backbone and defines training/validation steps.
* **Runner**: `main()` parses arguments, preprocesses global CSVs, loops per region for training and evaluation.

---

## Features

* **Outlier Removal**: Median Absolute Deviation (MAD) filtering and DBSCAN to remove coordinate outliers.
* **Region-wise Modeling**: Splits data by `Region_ID` and trains separate models for each region.
* **Coordinate Scaling**: Supports RobustScaler, StandardScaler, and MinMaxScaler for latitude-longitude normalization.
* **Backbone Flexibility**: Use ConvNeXt, ResNet, EfficientNet or any Timm-supported model.
* **Mixed Precision & DDP**: Optional AMP training and distributed data parallel (DDP) support.
* **Logging & Checkpointing**: Integration with Weights & Biases (wandb) for metrics and model artifacts.

## Prerequisites

* Python 3.8+
* CUDA-enabled GPU (optional, for acceleration)

### Python Packages

Install dependencies via `pip`:

```bash
pip install torch torchvision pytorch-lightning timm wandb joblib scikit-learn pandas numpy pillow
```

> If you plan to use mixed precision or DDP, ensure appropriate versions of PyTorch and CUDA are installed.

## Data Preparation

1. **Organize Images**:

   * Place training images under `images_train/images_train/`.
   * Place validation images under `images_val/images_val/`.

2. **CSV Files**:

   * `labels_train.csv` and `labels_val.csv` must contain at least the columns:

     * `filename` (e.g. `image_0001.jpg`)
     * `latitude` (float)
     * `longitude` (float)
     * `Region_ID` (integer or categorical identifier)

3. **Exclusions**:

   * The script automatically excludes specific filename indices (e.g. low-quality samples) via a built-in list.

## Usage

Run the main script with desired arguments:

```bash
python latlong_predict.py \
  --model_name convnext_small \
  --batch_size 32 \
  --lr 1e-5 \
  --weight_decay 1e-4 \
  --epochs 60 \
  --patience 10 \
  --scaler robust \
  --train_dir images_train/images_train \
  --val_dir images_val/images_val \
  --train_csv labels_train.csv \
  --val_csv labels_val.csv \
  --model_save_dir latlongcheckpoints \
  --num_workers 4 \
  --num_gpus 1 \
  --use_amp
```

### Example (CPU fallback)

```bash
python latlong_predict.py --num_gpus 0  # uses CPU if no GPU available
```

## Arguments

| Argument           | Type   | Default                     | Description                                                   |
| ------------------ | ------ | --------------------------- | ------------------------------------------------------------- |
| `--model_name`     | string | `convnext_small`            | Backbone model name (ConvNeXt, ResNet, EfficientNet, or timm) |
| `--batch_size`     | int    | `32`                        | Training and validation batch size                            |
| `--lr`             | float  | `1e-5`                      | Learning rate for optimizer                                   |
| `--weight_decay`   | float  | `1e-4`                      | Weight decay for AdamW                                        |
| `--epochs`         | int    | `60`                        | Maximum number of training epochs                             |
| `--patience`       | int    | `10`                        | EarlyStopping patience on `val_loss`                          |
| `--scaler`         | string | `robust`                    | Coordinate scaler: `robust`, `standard`, or `minmax`          |
| `--train_dir`      | string | `images_train/images_train` | Path to training images                                       |
| `--val_dir`        | string | `images_val/images_val`     | Path to validation images                                     |
| `--train_csv`      | string | `labels_train.csv`          | Training CSV file path                                        |
| `--val_csv`        | string | `labels_val.csv`            | Validation CSV file path                                      |
| `--model_save_dir` | string | `latlongcheckpoints`        | Directory to save region-specific models and scalers          |
| `--num_workers`    | int    | `4`                         | Number of DataLoader worker processes                         |
| `--num_gpus`       | int    | `1`                         | Number of GPUs to use (0 for CPU)                             |
| `--use_amp`        | flag   | `False`                     | Enable mixed precision training (FP16)                        |
| `--t_0`            | int    | `10`                        | T\_0 parameter for CosineAnnealingWarmRestarts                |
| `--t_mult`         | int    | `1`                         | T\_mult parameter for CosineAnnealingWarmRestarts             |

## Output

* **Checkpoints**: Saved under `latlongcheckpoints/region_<ID>/` with best `val_loss`.
* **Scalers**: Saved under `latlongcheckpoints/scalers/scaler_region_<ID>.joblib`.
* **WandB Logs**: Training metrics and model artifacts logged to Weights & Biases project `angle-prediction`.

## Evaluation

After training, the script evaluates each region model on the global validation set (filtered by region) and prints:

* Region-specific MSE on scaled and unscaled coordinates.
* Overall validation MSE per region.

## Logging

* Uses **Weights & Biases** (wandb) for experiment tracking.
* EarlyStopping on `val_loss` with configurable patience.
* ModelCheckpoint saves the top-1 model per region.

## License

This project is released under the MIT License.

---

*Happy Predicting!*
