**Angle Prediction with Sincos Encoding**

This project implements an image-based angle regression pipeline using PyTorch Lightning and modern backbones (ViT, ConvNeXt, EfficientNet). It encodes angles as sine and cosine targets to handle circularity, filters outliers, and logs experiments with Weights & Biases.

---

## Table of Contents

1. [Conceptual Overview](#conceptual-overview)
2. [Loss Function Explanation](#loss-function-explanation)
3. [Code Structure](#code-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Data Preparation](#data-preparation)
7. [Usage](#usage)
8. [Arguments](#arguments)
9. [Output & Logging](#output--logging)

---

## Conceptual Overview

Predicting angles directly (0–360°) with regression suffers discontinuity at the wrap-around point (e.g., 359° → 0°). To address this:

1. **Sincos Encoding**: Represent each angle $\theta$ as $[\sin(\theta), \cos(\theta)]$.
2. **Network Output**: Model predicts two values ,$\hat{s}, \hat{c}\,$ corresponding to sine and cosine.
3. **Decoding**: Convert $\hat{s}, \hat{c}$ back to angle via $\mathrm{atan2}(\hat{s}, \hat{c})$, mapping to \[0, 360°) domain.

This encoding ensures smoothness around the circular boundary and avoids large regression errors when angles cross 0°.

## Loss Function Explanation

We use **Mean Squared Error (MSE)** between predicted and target sincos vectors:

$$
L = \| [\hat{s},\hat{c}] - [\sin(\theta),\cos(\theta)] \|^2_2
$$

This encourages the model to approximate the unit circle representation. At inference, angular error is computed as:

1. Decode predictions:  $\hat{\theta} = \mathrm{mod}(\deg(\mathrm{atan2}(\hat{s},\hat{c})), 360)$
2. Compute circular difference:

   $$
   \Delta = ((\hat{\theta} - \theta + 180) \bmod 360) - 180
   $$
3. Report **Mean Absolute Error (MAE)** of $\Delta$.

## Code Structure

```text
angle_predict.py            # Main script with all logic
 ├── Helpers                # W&B logger init and finish functions
 ├── data_preprocess()      # CSV load, angle & coordinate outlier filtering
 ├── Transforms            # Train/val augmentations with torchvision
 ├── angle_to_sincos()      # Float → [sin, cos] tensor encoding
 ├── sincos_to_angle()      # Prediction → angle degrees decoding
 ├── DFDataroot             # Simple container for root directory + DataFrame
 ├── AngleDataset           # Dataset applying transforms & target encoding
 ├── AngleDataModule        # LightningDataModule wrapping DataLoaders
 ├── AnglePredictionModel   # LightningModule: backbone + MSE loss + angular MAE logging
 └── main()                 # Orchestrates preprocessing, DM, model init, training
```

## Prerequisites

* Python 3.8+
* GPU with CUDA (optional but recommended)

### Python Packages

Install dependencies:

```bash
pip install torch torchvision timm pytorch-lightning wandb scikit-learn pandas numpy pillow
```

## Data Preparation

Your datasets should include:

* **Images**: a directory with image files referenced in CSVs.
* **CSV Files** (`train.csv`, `val.csv`): columns at minimum:

  * `filename` (string)
  * `angle` (float degrees)
  * `latitude`, `longitude` (optional, used for DBSCAN filtering)

The script filters angles outside \[0, 360] and removes extreme outliers via MAD. It also clusters geographic coords with DBSCAN to remove noisy samples.

## Usage

Run training with:

```bash
python angle_predict.py \
  --train_root /path/to/train/images \
  --val_root   /path/to/val/images \
  --train_csv  train.csv \
  --val_csv    val.csv \
  --batch_size 16 \
  --lr 1e-4 \
  --epochs 20 \
  --devices 1 \
  --model_name vit_base_patch16_224
```

After training, W\&B dashboard will display training loss and validation angular MAE.

## Arguments

| Argument       | Type   | Default                | Description                                                                       |
| -------------- | ------ | ---------------------- | --------------------------------------------------------------------------------- |
| `--train_root` | path   | **required**           | Directory for training images                                                     |
| `--val_root`   | path   | **required**           | Directory for validation images                                                   |
| `--train_csv`  | file   | **required**           | CSV file with training data                                                       |
| `--val_csv`    | file   | **required**           | CSV file with validation data                                                     |
| `--batch_size` | int    | 8                      | Batch size                                                                        |
| `--lr`         | float  | 1e-4                   | Learning rate                                                                     |
| `--epochs`     | int    | 10                     | Number of epochs                                                                  |
| `--devices`    | int    | 1                      | Number of GPUs (0 for CPU)                                                        |
| `--model_name` | string | `vit_base_patch16_224` | Timm model name (e.g. `vit_base_patch16_224`, `convnext_base`, `efficientnet_b3`) |

## Output & Logging

* **Model**: Trained checkpoint saved in `checkpoints/` via W\&B logger.
* **Metrics**: Training MSE loss and validation angular MAE logged to Weights & Biases.

---

