# 3D Medical Image Segmentation: Comprehensive Documentation

Author: Thabhelo Duve  
Supervisor: William (Liam) Oswald  
Organization: Analytical AI  
Duration: 12 weeks (Fall 2025)

This document serves as the authoritative record of the project: design decisions, software choices and justification, experimental procedures, results, and materials for the conference paper.

## 1. Introduction and Objectives
- Comparative analysis of 3D U-Net, UNETR, SegResNet across BraTS, MSD Liver, TotalSegmentator.
- Goals: performance benchmarking, best-practices, and recommendations for 3D medical segmentation.

## 2. Software and Platform Choices
- Compute: Google Colab (GPU), storage via Google Drive.
- Frameworks: PyTorch + MONAI. Justification: mature 3D tooling, extensive transforms/metrics, active community.
- Experiment tracking: Git commits + on-disk results; future: lightweight CSV/JSON aggregation.

### Libraries In Use
- PyTorch (core DL), MONAI (medical imaging transforms, metrics, networks)
- NumPy/SciPy/Pandas (data ops), Matplotlib/Seaborn (plots)
- nibabel/SimpleITK (NIfTI IO)

### Dataset Handling
- Canonical dataset keys: `brats`, `msd_liver`, `totalsegmentator`
- Synonyms supported via `src/data/utils.py:normalize_dataset_name`
- Current folder names (local Drive): `BraTS/`, `MSD/`, `TotalSegmentator/`.

## 3. Data Management
- Datasets located at: `/content/drive/MyDrive/datasets`.
- Expected subfolders: `BraTS/`, `MSD_Liver/`, `TotalSegmentator/`.
- File formats: NIfTI (`.nii`, `.nii.gz`).

## 4. Methodology
- 3×3 experiment matrix (architectures × datasets).
- Config-driven training (see `configs/`).
- Preprocessing and augmentation using MONAI.
- Evaluation metrics: Dice, IoU, Hausdorff, surface distance, sensitivity, specificity.

## 5. Reproducibility
- All scripts and configs under version control.
- Random seeds documented per run; checkpoints stored under Drive `results/`.

## 6. Progress Timeline (Engineering Log)
- 2025-08-29: Repo initialized; structure scaffolded; requirements and setup added.
- 2025-08-29: Base config added for Colab paths; script stubs created.
- 2025-08-29: Dataset loaders implemented for BraTS, MSD Liver, TotalSegmentator.
- 2025-08-29: Preprocessing/augmentation transforms and dataloader builders implemented.
- 2025-08-29: Colab bootstrap fixed to clone repo to Drive and install requirements.

## 7. Results (to be populated)
- Record per-experiment metrics, resource usage, and training curves.
- Comparative tables and plots.

## 8. Discussion (to be populated)
- Interpretation of results, dataset/architecture interactions, limitations.

## 9. Conclusion (to be populated)
- Key findings and recommendations.

## 10. References
- Include citations for datasets and key methods.

---
Guidelines: no emojis; formal tone; keep this document up to date after each major milestone.
