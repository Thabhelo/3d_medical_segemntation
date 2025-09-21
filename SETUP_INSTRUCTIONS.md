# 3D Medical Segmentation - Setup Instructions

## Project Overview
This is a comparative analysis framework for 3D medical image segmentation using MONAI and PyTorch. The project evaluates 3D U-Net, UNETR, and SegResNet on three datasets: BraTS, MSD Liver, and TotalSegmentator.

## Current Status
✅ **Completed:**
- Dataset loaders for BraTS, MSD Liver, TotalSegmentator
- Model architectures (UNet, UNETR, SegResNet)
- Training pipeline with MONAI integration
- Configuration system with YAML files
- Evaluation metrics and visualization tools

## File Structure
```
3d_medical_segmentation/
├── src/
│   ├── data/           # Dataset loaders, transforms, dataloaders
│   ├── models/         # Model architectures, factory, losses
│   ├── training/       # Trainer and evaluator
│   ├── analysis/       # Metrics, statistics, visualization
│   └── utils/          # Logging utilities
├── configs/            # YAML configuration files
├── scripts/            # Entry point scripts
├── notebooks/           # Jupyter notebooks for Colab
├── tests/              # Test files
├── setup_git.sh        # Git initialization script
└── requirements.txt    # Python dependencies
```

## Setup Instructions

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Or install from setup.py
pip install -e .
```

### 2. Initialize Git Repository
```bash
# Run the git setup script
./setup_git.sh

# Set up remote repository (update URL as needed)
git remote add origin https://github.com/thabhelo/3d_medical_segmentation.git
git branch -M main
git push -u origin main
```

### 3. Install GitHub CLI (Optional)
```bash
# Install GitHub CLI
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# Authenticate with GitHub
gh auth login
```

### 4. Dataset Setup
Place datasets in the following structure:
```
datasets/
├── BraTS/              # BraTS 2021 dataset
├── MSD/                # MSD Liver dataset
└── TotalSegmentator/   # TotalSegmentator dataset
```

### 5. Running Experiments
```bash
# Train a model
python scripts/train_model.py --dataset brats --architecture unet --max_epochs 10

# Run all experiments
python scripts/run_all_experiments.py

# Evaluate a model
python scripts/evaluate_model.py --model_path results/best_model.pth
```

## Configuration Files
- `configs/base_config.yaml` - Base configuration
- `configs/unet_brats.yaml` - UNet on BraTS
- `configs/unetr_brats.yaml` - UNETR on BraTS
- `configs/segresnet_brats.yaml` - SegResNet on BraTS

## Key Features
- **Multi-dataset support**: BraTS, MSD Liver, TotalSegmentator
- **Multiple architectures**: UNet, UNETR, SegResNet
- **MONAI integration**: Medical imaging transforms and metrics
- **Flexible configuration**: YAML-based experiment setup
- **Comprehensive evaluation**: Dice, IoU, Hausdorff distance metrics

## Next Steps
1. Fix repository name from "segemntation" to "segmentation"
2. Set up proper git remote
3. Continue development based on specific requirements
4. Add more evaluation metrics and visualization tools

## Notes
- The project is designed to work with Google Colab (A100 GPU)
- All paths are configured for Colab environment
- For local development, update paths in configuration files
