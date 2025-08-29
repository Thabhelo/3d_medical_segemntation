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

## 6. Results (to be populated)
- Record per-experiment metrics, resource usage, and training curves.
- Comparative tables and plots.

## 7. Discussion (to be populated)
- Interpretation of results, dataset/architecture interactions, limitations.

## 8. Conclusion (to be populated)
- Key findings and recommendations.

## 9. References
- Include citations for datasets and key methods.

---
Guidelines: no emojis; formal tone; keep this document up to date after each major milestone.
