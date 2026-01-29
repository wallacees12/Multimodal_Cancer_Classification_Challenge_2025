# Multimodal Cancer Classification Challenge 2025

Binary classification of cancer cells using multimodal microscopy imaging (Bright-Field + Fluorescence) with ConvNeXt deep learning models.

## Overview

This project classifies microscopy cell images as **cancerous** or **healthy** by combining two imaging modalities:
- **Bright-Field (BF):** Standard optical microscopy
- **Fluorescence (FL):** Fluorescent marker imaging

The multimodal approach concatenates both image types into 6-channel tensors, providing richer features for classification.

## Model Architecture

**ConvNeXt-Large** (primary model) with transfer learning:
- Pretrained on ImageNet, fine-tuned on cancer cell data
- Backbone frozen, classifier head + Stage 3 unfrozen
- Input: 384×384 RGB images (or 6-channel multimodal)
- Output: Binary classification (cancerous/healthy)

## Training Features

- **Mixed Precision Training** (FP16) for memory efficiency
- **Weighted Random Sampling** to handle class imbalance
- **Cosine Annealing** learning rate schedule with warm restarts
- **Early Stopping** with patience monitoring
- **Gradient Clipping** (max_norm=1.0) for stability

### Data Augmentation
- Random rotation (±40°)
- Horizontal flip
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transforms
- Random resized crop (0.8-1.0 scale)

## Usage

### Training

```bash
# Quick training on subset (100 images)
python train_convnext.py

# Full dataset training (recommended)
python train_convex_full.py
```

### Inference

```bash
python inference_convnext.py
```

Outputs `submission_probs_convnext.csv` with cancer probabilities.

## Project Structure

```
├── train_convnext.py       # Fast prototyping (subset)
├── train_convex_full.py    # Production training (full data)
├── inference_convnext.py   # Test set predictions
├── load_data.py            # MultiModalCellDataset class
├── data_analysis.ipynb     # EDA notebook
├── utils/
│   └── dataloader.py       # Data utilities
├── checkpoints/            # Saved models
└── data/
    ├── BF/                 # Bright-field images
    ├── FL/                 # Fluorescence images
    └── train.csv           # Labels
```

## Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 4 |
| Learning Rate | 1e-4 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Image Size | 384×384 |
| Train/Val Split | 90/10 |

## Requirements

- PyTorch
- torchvision
- timm
- pandas
- PIL
- numpy
- tqdm
- tensorboard (optional)

## Results

Model checkpoints saved to `checkpoints/`. Training metrics logged to:
- `metrics_full.csv` (CSV format)
- `runs/convnext_full_train` (TensorBoard)

## Author

Samuel Wallace - MSc Artificial Intelligence, University of Zurich
