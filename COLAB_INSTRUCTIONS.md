# Google Colab Instructions - 3D Medical Segmentation

## 🚀 Quick Start (Open in Colab)

### Option 1: Direct Notebook Link
1. Open `notebooks/00_environment_setup.ipynb` in Colab
2. Run all cells sequentially
3. Training and evaluation will execute automatically

### Option 2: Manual Setup
Follow the steps below for manual control.

---

## 📝 Step-by-Step Colab Workflow

### 1. **Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. **Clone/Pull Repository**
```python
import os
import subprocess
from pathlib import Path

repo_dir = Path('/content/drive/MyDrive/3d_medical_segemntation')

if not repo_dir.exists():
    # First time: clone
    subprocess.run([
        'git', 'clone', '-q',
        'https://github.com/Thabhelo/3d_medical_segemntation.git',
        str(repo_dir)
    ], check=True)
else:
    # Update existing repo
    subprocess.run(['git', '-C', str(repo_dir), 'pull', '--ff-only'], check=True)

os.chdir(repo_dir)
print(f"Working directory: {Path.cwd()}")
```

### 3. **Install Dependencies** (Python 3.12 Compatible)
```python
import subprocess

# Upgrade base packages
subprocess.run(['pip', 'install', '-q', '--upgrade', 'pip', 'setuptools', 'wheel'], check=True)

# Install PyTorch with CUDA
subprocess.run([
    'pip', 'install', '-q',
    'torch==2.4.0', 'torchvision==0.19.0',
    '--index-url', 'https://download.pytorch.org/whl/cu121'
], check=True)

# Install medical imaging packages
subprocess.run([
    'pip', 'install', '-q',
    'monai-weekly', 'numpy>=1.26.4', 'scipy>=1.12',
    'nibabel', 'SimpleITK', 'PyYAML', 'tqdm',
    'tensorboard', 'matplotlib>=3.7', 'seaborn>=0.12',
    'scikit-learn>=1.3', 'pandas>=2.0'
], check=True)

print("✅ Dependencies installed")
```

### 4. **Verify Environment**
```python
import torch
import sys

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 5. **Check Datasets**
```python
from pathlib import Path

datasets_dir = Path('/content/drive/MyDrive/datasets')
print(f"Datasets directory: {datasets_dir}")

for dataset in ['BraTS', 'MSD/Task03_Liver', 'TotalSegmentator']:
    path = datasets_dir / dataset
    status = "✅ FOUND" if path.exists() else "❌ MISSING"
    print(f"  {dataset}: {status}")
```

**Note**: If datasets are missing, ensure they are uploaded to `/content/drive/MyDrive/datasets/` in your Google Drive.

---

## 🏃 Running Experiments

### Option A: Run All Experiments (Automated)
```python
import sys
import subprocess

# This will train all 9 model combinations
# Skip logic prevents re-running completed experiments
subprocess.run([sys.executable, '-u', 'scripts/run_experiments.py'], check=False)
```

**What this does**:
- Trains 3 datasets × 3 architectures = 9 models
- Skips already-completed runs (checks for `best.pth`)
- Streams logs with ETA estimation
- Saves checkpoints to `results/colab_runs/`

### Option B: Run Single Experiment
```python
import subprocess

# Example: Train UNet on BraTS
subprocess.run([
    'python', '-u', 'scripts/train_model.py',
    '--dataset', 'brats',
    '--architecture', 'unet',
    '--in_channels', '4',
    '--out_channels', '4',
    '--max_epochs', '100',
    '--batch_size', '2',
    '--output_dir', 'results/my_experiment'
], check=True)
```

---

## 📊 Model Evaluation

### Run Comprehensive Evaluation
```python
import subprocess
import sys

# Evaluate all trained models
subprocess.run([sys.executable, '-u', 'scripts/evaluate_models.py'], check=False)
```

**Output**:
- Validation Dice scores for all models
- Model complexity (parameters, file size)
- Comparative performance tables
- JSON results saved to `results/colab_runs/evaluation_results.json`

### View Evaluation Results
```python
import json
import pandas as pd

# Load results
with open('results/colab_runs/evaluation_results.json') as f:
    results = json.load(f)

# Create DataFrame for easy viewing
df = pd.DataFrame(results)

# Pivot table: Dice scores
pivot = df.pivot(index='dataset', columns='architecture', values='val_dice')
print("Validation Dice Scores:")
print(pivot.round(4))

# Model parameters (millions)
param_pivot = df.pivot(index='dataset', columns='architecture', values='num_parameters')
print("\nModel Parameters (M):")
print((param_pivot / 1e6).round(2))
```

---

## 📂 File Organization

### Repository Structure
```
/content/drive/MyDrive/3d_medical_segemntation/
├── notebooks/
│   └── 00_environment_setup.ipynb  # Main notebook (run this)
├── scripts/
│   ├── train_model.py              # Single training script
│   ├── run_experiments.py          # Multi-experiment orchestrator
│   └── evaluate_models.py          # Evaluation script
├── src/
│   ├── data/                       # Dataset loaders
│   ├── models/                     # Architecture definitions
│   └── training/                   # Trainer implementation
└── results/
    └── colab_runs/                 # Training outputs
        ├── brats_unet/
        │   ├── best.pth
        │   └── train_*.log
        ├── brats_unetr/
        └── ...
```

### Dataset Structure
```
/content/drive/MyDrive/datasets/
├── BraTS/
│   ├── sub-001/
│   │   ├── t1.nii.gz
│   │   ├── t1ce.nii.gz
│   │   ├── t2.nii.gz
│   │   ├── flair.nii.gz
│   │   └── seg.nii.gz
│   └── ...
├── MSD/
│   └── Task03_Liver/
│       ├── imagesTr/
│       └── labelsTr/
└── TotalSegmentator/
    ├── sub-001/
    │   ├── ct.nii.gz
    │   └── segmentations/
    └── ...
```

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"
**Solution**: Ensure you're in the repo directory
```python
import os
os.chdir('/content/drive/MyDrive/3d_medical_segemntation')
```

### Issue: "CalledProcessError" during git pull
**Solution**: Hard reset to remote
```python
import subprocess
repo_dir = '/content/drive/MyDrive/3d_medical_segemntation'
subprocess.run(['git', '-C', repo_dir, 'fetch', 'origin'], check=True)
subprocess.run(['git', '-C', repo_dir, 'reset', '--hard', 'origin/main'], check=True)
```

### Issue: "ValueError: No training samples found"
**Solution**: Check dataset path
```python
from pathlib import Path

# For MSD Liver, path must include Task03_Liver
msd_path = Path('/content/drive/MyDrive/datasets/MSD/Task03_Liver')
print(f"MSD Liver exists: {msd_path.exists()}")
print(f"imagesTr exists: {(msd_path / 'imagesTr').exists()}")
```

### Issue: CUDA out of memory
**Solutions**:
1. Reduce batch size: `--batch_size 1`
2. Reduce patch size: modify `create_dataloaders()` to use `patch_size=(96, 96, 96)`
3. Use gradient accumulation (requires code modification)

### Issue: Session timeout during long training
**Solutions**:
1. Use Colab Pro for longer runtimes
2. Implement checkpoint resume (already supported)
3. Run experiments in batches
4. Use `screen` or `tmux` if SSH access available

---

## ⚙️ Configuration Options

### Adjust Training Epochs
Edit `scripts/run_experiments.py`:
```python
MAX_EPOCHS = 100  # Change from 2 to 100 for production
```

### Change Batch Size
```python
BATCH_SIZE = 2  # Reduce to 1 if OOM errors
```

### Select Specific Experiments
Edit `scripts/run_experiments.py`:
```python
DATASETS = ['brats']  # Run only BraTS
ARCHITECTURES = ['unet', 'unetr']  # Skip SegResNet
```

### Custom Output Directory
```python
OUTPUT_BASE = Path('results/my_custom_runs')
```

---

## 📈 Monitoring Training

### Check Completed Runs
```bash
!find results/colab_runs -name "best.pth" | sort
```

### View Training Logs
```bash
!tail -n 20 results/colab_runs/brats_unet/train_*.log
```

### Monitor GPU Usage
```python
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)
```

---

## 💾 Saving Results

All results are automatically saved to Google Drive:
- **Checkpoints**: `results/colab_runs/{dataset}_{arch}/best.pth`
- **Training logs**: `results/colab_runs/{dataset}_{arch}/train_*.log`
- **Evaluation**: `results/colab_runs/evaluation_results.json`

**Important**: Results persist across Colab sessions because they're stored in Drive!

---

## 🎯 Next Steps After Training

1. ✅ **Evaluate models**: Run `scripts/evaluate_models.py`
2. 📊 **Analyze results**: Create visualizations and comparisons
3. 📝 **Document findings**: Update `DOCUMENTATION.md` with results
4. 🚀 **Extended training**: Re-run with 50-100 epochs for production metrics
5. 🔬 **Advanced analysis**: Learning curves, statistical tests, clinical insights

---

**Last Updated**: September 30, 2025  
**Repository**: https://github.com/Thabhelo/3d_medical_segemntation  
**Support**: See `NEXT_STEPS.md` for detailed guidance
