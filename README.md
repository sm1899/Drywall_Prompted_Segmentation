# Prompted Segmentation for Drywall QA

A text-conditioned segmentation model for detecting cracks and drywall taping areas using CLIPSeg fine-tuning.

## Overview

This project fine-tunes CLIPSeg to perform binary segmentation on two tasks:
- **Crack segmentation**: Detects cracks in walls using the prompt "segment crack"
- **Drywall taping area segmentation**: Detects drywall joints/taping areas using the prompt "segment taping area"

## Dataset

- **Cracks Dataset**: 5,164 training samples, validation split available
- **Drywall-Join-Detect Dataset**: 820 training samples, validation split available
- Both datasets converted from YOLOv5 format to binary masks (PNG, single-channel, {0, 255})

## Setup

### Requirements

```bash
conda activate comfy  # or your preferred environment
pip install torch torchvision transformers pillow numpy opencv-python matplotlib tqdm
```

### Data Preparation

1. Place datasets in `data/` directory:
   - `data/cracks.v1i.yolov5pytorch/`
   - `data/Drywall-Join-Detect.v2i.yolov5pytorch/`

2. Convert annotations to binary masks:
```bash
python prepare_dataset.py
```

This creates `label_mask/` folders with binary segmentation masks for both datasets.

## Usage

### Training

```bash
python train.py
```

**Configuration** (in `train.py` main function):
- Batch size: 128
- Learning rate: 1e-4
- Image size: 352x352
- Loss weights:
  - BCE weight: 1.0
  - Dice weight: 1.0
  - Cracks pos_weight: 2.0 (for class imbalance)
  - Drywall pos_weight: 20.0 (for class imbalance)
  - Cracks loss weight: 1.0
  - Drywall loss weight: 2.0

**Training features:**
- Balanced batch sampling (50/50 split per batch)
- Data augmentation (horizontal flip, rotation, color jitter)
- Early stopping (patience=10 epochs)
- Learning rate scheduling (ReduceLROnPlateau)
- Combined BCE + Dice loss
- Only decoder fine-tuning (CLIP encoder frozen)

### Inference

```bash
python inference.py
```

Evaluates the model on validation sets and saves predictions to `results/` folder.

## Model Architecture

- **Base Model**: CLIPSeg (CIDAS/clipseg-rd64-refined)
- **Fine-tuning**: Only decoder parameters (CLIP encoder frozen)
- **Input**: RGB images (352x352) + text prompts
- **Output**: Binary segmentation masks (352x352)

## Results

See `TRAINING_REPORT.md` for detailed results and analysis.

**Best Validation Performance:**
- **Cracks**: mIoU: 0.493, Dice: 0.639
- **Drywall**: mIoU: 0.099, Dice: 0.155
- **Overall**: mIoU: 0.303, Dice: 0.441

## Files

- `train.py`: Training script with balanced sampling and loss weighting
- `inference.py`: Evaluation script for validation sets
- `prepare_dataset.py`: Converts YOLO annotations to binary masks
- `checkpoints/best/`: Best model checkpoint (saved based on validation loss)
- `metrics.json`: Training metrics history
- `training_curves.png`: Visualization of training progress

## Notes

- Model uses fixed-length text tokenization (max_length=77)
- Balanced batch sampler ensures equal samples from both datasets per batch
- Early stopping prevents overfitting
- Training metrics tracked separately for each dataset

