# Dynamic Learning Rate Implementation

## Overview
This document describes the implementation of dynamic learning rate scheduling and resume functionality for the 3D medical segmentation project.

## Changes Made

### 1. Updated Trainer Class (`src/training/trainer.py`)
- **Added scheduler support**: New `scheduler` parameter to accept any PyTorch LR scheduler
- **Added save_every_epoch option**: New `save_every_epoch` parameter to control checkpoint frequency
- **Enhanced checkpoint saving**: Now saves scheduler state in checkpoints
- **Dynamic LR stepping**: Automatically steps scheduler based on validation performance

### 2. Updated Training Script (`scripts/train_model.py`)
- **Added scheduler argument**: `--scheduler` with options: `none`, `reduce_on_plateau`, `cosine`, `onecycle`, `polynomial`
- **Added save_every_epoch argument**: `--save_every_epoch` flag to save checkpoints every epoch
- **Enhanced resume logic**: Now loads scheduler state when resuming from checkpoint

### 3. Created Resume Script (`scripts/resume_training_dynamic.py`)
- **Resume from epoch 50**: Automatically finds and resumes from epoch 50 checkpoints
- **Dynamic LR scheduling**: Uses `ReduceLROnPlateau` scheduler for adaptive learning rates
- **Continue to epoch 100**: Extends training from 50 to 100 epochs
- **Save every epoch**: Enables frequent checkpoint saving for better recovery

### 4. Updated Batch Scripts
All batch scripts now use:
- **50 epochs instead of 100**: Reduced training time while maintaining quality
- **Dynamic learning rate**: `ReduceLROnPlateau` scheduler for adaptive training
- **Save every epoch**: More frequent checkpointing for better recovery
- **Updated time estimates**: Realistic time estimates based on 50 epochs

## Learning Rate Schedulers Available

### 1. ReduceLROnPlateau (Recommended)
- **How it works**: Reduces LR when validation metric plateaus
- **Parameters**: `factor=0.5`, `patience=10`, `mode='max'`
- **Best for**: Medical segmentation where validation Dice is the key metric

### 2. CosineAnnealingLR
- **How it works**: Cosine annealing schedule from max LR to min LR
- **Parameters**: `T_max=epochs`, `eta_min=lr*0.01`
- **Best for**: When you want smooth LR decay

### 3. OneCycleLR
- **How it works**: One cycle policy with warmup and decay
- **Parameters**: `max_lr=lr*10`, `total_steps=epochs`
- **Best for**: Fast convergence with high learning rates

### 4. PolynomialLR
- **How it works**: Polynomial decay schedule
- **Parameters**: `total_iters=epochs`, `power=0.9`
- **Best for**: Gradual LR reduction

## Usage Examples

### Resume from Epoch 50 with Dynamic LR
```bash
python scripts/resume_training_dynamic.py
```

### Train with Dynamic LR from Scratch
```bash
python scripts/train_model.py \
    --dataset brats \
    --architecture unet \
    --max_epochs 50 \
    --scheduler reduce_on_plateau \
    --save_every_epoch
```

### Resume Specific Checkpoint
```bash
python scripts/train_model.py \
    --dataset brats \
    --architecture unet \
    --resume_from results/colab_runs/brats_unet/epoch_50.pth \
    --scheduler reduce_on_plateau \
    --save_every_epoch
```

## Benefits of Dynamic Learning Rate

### 1. Adaptive Training
- **Automatic LR adjustment**: Reduces LR when validation performance plateaus
- **Better convergence**: Prevents overfitting and improves final performance
- **Reduced manual tuning**: Less need to manually adjust learning rates

### 2. Improved Recovery
- **Frequent checkpoints**: Save every epoch for better recovery from interruptions
- **Scheduler state preservation**: Resume with correct LR schedule
- **Better monitoring**: Track LR changes during training

### 3. Time Efficiency
- **50 epochs instead of 100**: Faster training while maintaining quality
- **Early stopping potential**: Can stop early if performance plateaus
- **Resource optimization**: Better use of compute resources

## File Structure
```
scripts/
├── train_model.py                    # Enhanced with dynamic LR support
├── resume_training_dynamic.py        # Resume from epoch 50 with dynamic LR
├── run_batch_1_brats.py             # Updated for 50 epochs + dynamic LR
├── run_batch_2_msd_liver.py        # Updated for 50 epochs + dynamic LR
├── run_batch_3_totalsegmentator.py  # Updated for 50 epochs + dynamic LR
└── test_resume.py                   # Test script for resume functionality

src/training/
└── trainer.py                       # Enhanced with scheduler support
```

## Next Steps

1. **Test the implementation**: Run `python scripts/test_resume.py` to verify functionality
2. **Resume training**: Use `python scripts/resume_training_dynamic.py` to continue from epoch 50
3. **Monitor performance**: Watch for LR adjustments and improved convergence
4. **Evaluate results**: Compare dynamic LR results with static LR baseline

## Expected Improvements

- **Better convergence**: Dynamic LR should lead to better final performance
- **Faster training**: 50 epochs with dynamic LR vs 100 epochs with static LR
- **More robust training**: Better handling of plateaus and local minima
- **Better recovery**: Frequent checkpoints and scheduler state preservation
