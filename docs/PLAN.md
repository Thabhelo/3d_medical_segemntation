# Working Plan and Commit Strategy

This plan follows small, complete commits to build the 3D medical segmentation framework and execute experiments on Google Colab with datasets in Google Drive (`My Drive/datasets`). The GitHub repository is `github.com/thabhelo/3d_medical_segemntation`.

## Milestone 0: Repo & Scaffolding (Week 1)
- Initialize repo and GitHub remote
- README, LICENSE, .gitignore
- Scaffold `src/`, `configs/`, `scripts/`, `tests/`, `notebooks/`
- Add `requirements.txt`, `setup.py`
- Base config for Colab paths

## Milestone 1: Data Pipeline (Weeks 1-2)
- Implement dataset interfaces for BraTS, MSD Liver, TotalSegmentator
- Preprocessing & augmentation with MONAI
- Unit tests for data loaders
- Colab notebook: EDA and sanity checks

## Milestone 2: Model Factory (Weeks 3-4)
- Implement `UNet`, `UNETR`, `SegResNet` constructors via factory
- Loss factory and metrics utilities
- Smoke tests on tiny volumes

## Milestone 3: Training Engine (Weeks 5-6)
- Trainer with checkpointing, logging, evaluation
- Config-driven runs
- Early stopping and LR scheduling

## Milestone 4: Experiments (Weeks 7-10)
- Generate 9 configs (3 arch × 3 data)
- Run on Colab, store outputs in Drive
- Track metrics and artifacts

## Milestone 5: Analysis & Reporting (Weeks 11-12)
- Aggregate results, statistical analysis (ANOVA)
- Plots and comparative tables
- Final report notebook and README summary

## Commit Cadence
- One concern per commit
- Prefer files added with minimal code first (stubs), then implementations
- Keep commits under ~200 LOC where possible

## Colab Workflow
1. Mount Drive
2. `pip install -r requirements.txt`
3. `cd /content/drive/MyDrive/3d_medical_segmentation`
4. Run scripts/notebooks

## Dataset Paths
- Base: `/content/drive/MyDrive/datasets`
- Expect subfolders: `BraTS/`, `MSD_Liver/`, `TotalSegmentator/`

## Next Steps
- Implement `src/data/datasets.py` classes (BraTS/MSD/TotalSeg)
- Implement `src/data/transforms.py` preprocessing/augmentation
- Add unit tests in `tests/`
