# 3D Medical Image Segmentation

Comparative analysis framework for 3D medical image segmentation using MONAI and PyTorch. We evaluate 3D U-Net, UNETR, and SegResNet on three datasets: BraTS, MSD Liver, and TotalSegmentator.

## Quick Start (Google Colab)
1. Open a Colab notebook and run:
   - Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
   - Run notebook `notebooks/01_colab_setup_and_eda.ipynb` to clone this repo into Drive and install requirements.
2. Ensure datasets are under `/content/drive/MyDrive/datasets` with folders:
   - `BraTS/` (or `brats`, `BraTS2021`)
   - `MSD/` (or `MSD_Liver`, `Task03_Liver`)
   - `TotalSegmentator/` (or `TotalSeg`)

## Project Structure
- `src/data/`: dataset loaders, transforms, dataloaders
- `src/models/`: model factory and architectures (TBD)
- `src/training/`: trainer and evaluator (TBD)
- `configs/`: YAML configs (base provided)
- `scripts/`: entry points (stubs provided)
- `notebooks/`: Colab setup and analysis notebooks

## Datasets
Place datasets in Drive at `/content/drive/MyDrive/datasets`. Folder synonyms are supported. See `src/data/utils.py:normalize_dataset_name`.
Synonyms examples:
- BraTS: `BraTS`, `brats`, `BraTS2021`
- MSD Liver: `MSD`, `MSD_Liver`, `Task03_Liver`
- TotalSegmentator: `TotalSegmentator`, `TotalSegmentor`, `TotalSegmantator`, `TotalSeg`

## Status
- Datasets: BraTS, MSD Liver, TotalSegmentator loaders implemented
- Transforms & Dataloaders: implemented
- Next: model factory (UNet, UNETR, SegResNet)

## License
MIT
