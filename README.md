# 3D Medical Image Segmentation

Comparative analysis framework for 3D medical image segmentation using MONAI and PyTorch. We evaluate 3D U-Net, UNETR, and SegResNet on three datasets: BraTS, MSD Liver, and TotalSegmentator.

## Quick Start (Google Colab)
1. Open `notebooks/00_environment_setup.ipynb` in Colab and Run all. It will:
   - Mount Drive
   - Clone/pull this repo into `/content/drive/MyDrive/3d_medical_segemntation` (current name)
   - Install compatible dependencies for Colab (Python 3.12)
   - Run all selected experiments
2. Ensure datasets are under `/content/drive/MyDrive/datasets` with folders:
   - `BraTS/` (or `brats`, `BraTS2021`)
   - `MSD/` (or `MSD_Liver`, `Task03_Liver`)
   - `TotalSegmentator/` (or `TotalSeg`)

## Project Structure
- `src/data/`: dataset loaders, transforms, dataloaders
- `src/models/`: model factory and architectures (TBD)
- `src/training/`: trainer and evaluator (TBD)
- `configs/`: YAML configs (base provided)
- `scripts/`: entry points
- `scripts/dev/`: developer utilities (`setup_git.sh`, `setup_github_cli.sh`)
- `notebooks/`: Colab environment setup and unified experiments runner

## Datasets
Place datasets in Drive at `/content/drive/MyDrive/datasets`. Folder synonyms are supported. See `src/data/utils.py:normalize_dataset_name`.
Synonyms examples:
- BraTS: `BraTS`, `brats`, `BraTS2021`
- MSD Liver: `MSD`, `MSD_Liver`, `Task03_Liver`
- TotalSegmentator: `TotalSegmentator`, `TotalSegmentor`, `TotalSegmantator`, `TotalSeg`
  (We now standardize on `TotalSegmentator` only.)

## Status

### ✅ Completed Infrastructure
- **Datasets**: BraTS, MSD Liver, TotalSegmentator loaders with robust path resolution
- **Models**: UNet (BasicUNet), UNETR, SegResNet with proper MONAI integration
- **Training**: Comprehensive pipeline with mixed precision, checkpointing, streaming logs
- **Environment**: Colab-ready setup with Drive persistence and auto-detection
- **Evaluation**: Dice metric computation with proper one-hot encoding

### ✅ Training Results (All 9 Combinations Complete)
**Experiment Matrix**: 3 datasets × 3 architectures = 9 trained models

| Dataset | UNet | UNETR | SegResNet |
|---------|------|-------|-----------|
| BraTS (4→4 channels) | ✅ | ✅ | ✅ |
| MSD Liver (1→3 channels) | ✅ | ✅ | ✅ |
| TotalSegmentator (1→2 channels) | ✅ | ✅ | ✅ |

**Training Performance**:
- BraTS: ~17s/epoch (4-channel input, 4-class output)
- MSD Liver: ~1500s/epoch (single-channel CT, 3-class liver segmentation)  
- TotalSegmentator: ~2000s/epoch (single-channel CT, 118-class→2-class simplified)
- All models: CUDA acceleration, mixed precision, proper convergence

### ✅ New Features
- **2D Slice Visualization**: Generate triptych views (input, ground truth, prediction) and GIF animations
- **Learning Rate Scheduler Experiments**: Compare schedulers (none, reduce_on_plateau, cosine, onecycle, polynomial, dlrs)
- **DLRS Integration**: Dynamic Learning Rate Scheduler support for adaptive training
- **Visualization CLI**: `scripts/visualize_predictions.py` for generating publication-quality visuals
- **Scheduler Experiments**: `scripts/run_scheduler_experiments.py` for quick scheduler comparison

See `SCHEDULER_EXPERIMENTS.md` for details on running scheduler experiments and generating visualizations.

### 🔄 Next Phase: Evaluation & Analysis
- **Model Evaluation**: Comprehensive metrics (Dice, IoU, Hausdorff distance)
- **Results Analysis**: Performance comparison across architectures and datasets
- **Visualization**: Sample predictions, confusion matrices, learning curves
- **Documentation**: Technical report with findings and reproducibility guide

## Inference Speed Benchmarking
Measure per-volume latency and throughput across GPUs to compare deployment efficiency.

### Steps
1. Ensure you have a trained checkpoint and a sample input tensor matching your model's IO shape.
2. Run the following minimal snippet to time forward passes with CUDA synchronization:
3. Log results in a table (dataset, architecture, GPU, num GPUs, latency, throughput, Torch, CUDA).

Running `python scripts/evaluate_models.py` now performs both validation evaluation and inference benchmarking by default and writes a single unified summary at:
- `results/colab_runs/evaluation_full.json` (also mirrored to `results/evaluation_full.json`).

For the full protocol and reporting guidance, see "Inference Efficiency Benchmarking" in `DOCUMENTATION.md`.

## License
MIT
