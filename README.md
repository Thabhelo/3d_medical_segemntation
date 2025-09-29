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
- Datasets: BraTS, MSD Liver, TotalSegmentator loaders implemented
- Transforms & Dataloaders: implemented
- Model Factory: UNet, UNETR, SegResNet implemented
- Training Infrastructure: implemented with checkpointing
- Experiments Progress: 5/9 completed
  - Completed: UNet+BraTS, UNet+MSD, UNETR+BraTS, UNETR+MSD, UNet+BraTS_test
  - Remaining: SegResNet experiments (2) + TotalSegmentator experiments (3)
- Next: Complete remaining experiments and results analysis

## License
MIT
