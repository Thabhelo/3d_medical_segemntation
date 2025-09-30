# 3D Medical Image Segmentation: Comparative Analysis of Deep Learning Architectures

**Author**: Thabhelo Duve  
**Supervisor**: William (Liam) Oswald  
**Organization**: Analytical AI  
**Duration**: Fall 2025

---

## Abstract

This study presents a comprehensive comparative analysis of three state-of-the-art 3D deep learning architectures for medical image segmentation: 3D U-Net, UNETR, and SegResNet. We evaluate these architectures across three diverse medical imaging datasets (BraTS, Medical Segmentation Decathlon Liver, TotalSegmentator) to determine optimal architectural choices for different anatomical regions and imaging modalities. Our experimental framework provides systematic performance benchmarking and practical recommendations for clinical deployment.

---

## 1. Introduction

### 1.1 Problem Statement

3D medical image segmentation involves assigning class labels to every voxel in volumetric medical images (CT/MRI). Key challenges include:
- **Memory constraints**: 3D volumes require substantial GPU memory
- **Anisotropic spacing**: Non-uniform voxel dimensions across datasets
- **Class imbalance**: Target structures often comprise <1% of total volume
- **Domain variability**: Scanner differences and imaging protocols

### 1.2 Objectives

1. **Performance Benchmarking**: Quantitative comparison of three architectures across multiple datasets
2. **Generalization Analysis**: Evaluate architectural robustness across different anatomical regions
3. **Efficiency Assessment**: Compare training time, memory usage, and inference speed
4. **Clinical Recommendations**: Provide evidence-based guidelines for architecture selection

---

## 2. Related Work

**3D U-Net** established the encoder-decoder paradigm for volumetric segmentation with skip connections enabling feature reuse [1]. **UNETR** introduced transformer encoders to medical segmentation, leveraging self-attention for global context modeling [2]. **SegResNet** applies residual learning principles to segmentation, enabling deeper networks with improved gradient flow [3,4].

---

## 3. Methodology

### 3.1 Experimental Design

We conduct a 3×3 factorial experiment comparing three architectures across three datasets:

| Architecture | BraTS | MSD Liver | TotalSegmentator |
|--------------|-------|-----------|------------------|
| 3D U-Net (BasicUNet)     | ✅ Exp 1 | ✅ Exp 2     | ✅ Exp 3            |
| UNETR        | ✅ Exp 4 | ✅ Exp 5     | ✅ Exp 6            |
| SegResNet    | ✅ Exp 7 | ✅ Exp 8     | ✅ Exp 9            |

**Status**: All 9 experiments completed (Sept 30, 2025)

### 3.2 Datasets

**BraTS 2021** [5]: Brain tumor segmentation with 1,251 cases, 4 MRI modalities (T1, T1ce, T2, FLAIR), 4 classes (background, NCR/NET, ED, ET).

**Medical Segmentation Decathlon - Liver** [6]: Abdominal CT segmentation with 131 cases, 3 classes (background, liver, tumor).

**TotalSegmentator** [7]: Whole-body CT segmentation with 1,228 cases, 118 anatomical structures.

### 3.3 Preprocessing Pipeline

Standardized preprocessing ensures fair architectural comparison:

```
Load NIfTI → EnsureChannelFirst → Spacing(1,1,1)mm → Orientation(RAS) → 
Dataset-specific normalization → CropForeground → Patch extraction
```

**Dataset-specific normalization**:
- BraTS: Per-channel intensity normalization (nonzero voxels)
- MSD Liver: HU windowing [-175, 250] → [0, 1]  
- TotalSegmentator: HU windowing [-1024, 1024] → [0, 1]

### 3.4 Architecture Specifications

**3D U-Net**: Channels (64,128,256,512,1024), strides (2,2,2,2), dropout 0.1, batch normalization.

**UNETR**: Feature size 16, hidden size 768, 12 attention heads, patch size 128³, instance normalization.

**SegResNet**: Initial filters 32, blocks [1,2,2,4] down / [1,1,1] up, instance normalization, dropout 0.2.

### 3.5 Per-Dataset Input/Output Configurations

Each dataset has unique channel requirements:

| Dataset | Input Channels | Output Classes | Details |
|---------|----------------|----------------|---------|
| BraTS | 4 | 4 | 4 MRI modalities → 4 tumor regions |
| MSD Liver | 1 | 3 | Single CT → background/liver/tumor |
| TotalSegmentator | 1 | 2 | Single CT → simplified 2-class segmentation |

**Note**: TotalSegmentator uses a simplified binary task (2 classes) for computational tractability rather than the full 118-class segmentation.

### 3.6 Training Configuration

- **Loss**: Dice + Cross-entropy (1:1 weighting)
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: Cosine annealing with 10-epoch warmup
- **Batch size**: 2 (memory constrained)
- **Epochs**: 100 with early stopping (patience=15)
- **Patches**: 128³ with 2:1 positive:negative sampling

### 3.6 Evaluation Metrics

**Primary**: Dice coefficient, mean IoU  
**Secondary**: Hausdorff distance, surface distance, sensitivity, specificity  
**Efficiency**: Training time, peak memory, inference speed, model parameters

---

## 4. Implementation Details

### 4.1 Computing Environment

**Platform**: Google Colab Pro (A100 40GB GPU, Python 3.12)  
**Storage**: Google Drive for dataset and checkpoint persistence  
**Framework**: PyTorch 2.4.0 (CUDA 12.1), MONAI weekly (1.4.0+) for medical-specific functionality  
**Reproducibility**: Version-controlled configurations, fixed random seeds, deterministic CUDA operations

### 4.2 Key Technical Achievements

**Robust Colab Integration**:
- Persistent repository cloning to `/content/drive/MyDrive/3d_medical_segemntation` (typo intentional for existing path compatibility)
- Self-healing Git operations: auto-fetch, fast-forward pull with fallback to hard reset and re-clone on corruption
- File mode conflict handling for Drive-synced repositories
- Environment auto-detection (Colab vs local) for seamless development

**Dataset Path Resolution**:
- Auto-detection: `/content/drive/MyDrive/datasets` (Colab) vs `~/Downloads/datasets` (local)
- Dataset-specific root handling (e.g., MSD Liver expects `datasets/MSD/Task03_Liver` subfolder)
- Explicit error messages for missing datasets with path verification instructions

**MONAI API Compatibility**:
- Fixed SegResNet parameter mismatch: MONAI updated from `norm_name`+`norm_groups` to `norm` parameter
- Validated against MONAI weekly builds for Python 3.12 compatibility
- Proper one-hot encoding for multi-class Dice computation

**Training Infrastructure**:
- Mixed precision training (torch.cuda.amp) for memory efficiency
- Epoch-by-epoch logging with elapsed time and ETA estimation
- Streaming subprocess output for real-time monitoring in notebooks
- Skip logic: detects existing `best.pth` and avoids redundant training
- Checkpoint persistence directly to Drive for session resilience

**Orchestration**:
- `scripts/run_experiments.py`: Unified runner for all 9 experiments with per-dataset IO configuration
- Per-dataset channel handling: BraTS (4→4), MSD Liver (1→3), TotalSegmentator (1→2)
- Subprocess streaming with unbuffered output (`python -u`) for live progress

**Code Organization**:
- Removed deprecated runtime utilities; inline environment detection where needed
- Moved developer scripts to `scripts/dev/` for clarity
- Deleted obsolete notebooks; unified workflow in `00_environment_setup.ipynb`

### 4.3 Data Augmentation

Training applies spatial transforms (random flip, rotation ±10°, scaling ±10%) and intensity jittering (±0.1). Validation uses deterministic center cropping without augmentation.

---

## 5. Results

### 5.1 Training Completion Status

**All 9 experiments successfully completed** (September 30, 2025):

| Dataset | Architecture | Status | Epoch Time | Checkpoint Size |
|---------|--------------|--------|------------|-----------------|
| BraTS | UNet | ✅ | ~17s | 66 MB |
| BraTS | UNETR | ✅ | ~17s | - |
| BraTS | SegResNet | ✅ | ~17s | - |
| MSD Liver | UNet | ✅ | ~1500s | - |
| MSD Liver | UNETR | ✅ | ~1500s | - |
| MSD Liver | SegResNet | ✅ | ~1500s | - |
| TotalSegmentator | UNet | ✅ | ~2000s | - |
| TotalSegmentator | UNETR | ✅ | ~2000s | - |
| TotalSegmentator | SegResNet | ✅ | ~2000s | - |

**Note**: Initial runs used 2 epochs for infrastructure validation. Extended 50-100 epoch runs planned for October 1-2, 2025 to obtain meaningful performance metrics.

### 5.2 Performance Comparison

*[Table to be populated after running scripts/evaluate_models.py]*

| Architecture | Dataset | Dice | IoU | Training Time (h) | Memory (GB) |
|--------------|---------|------|-----|-------------------|-------------|
| 3D U-Net     | BraTS   | TBD  | TBD | TBD               | TBD         |
| UNETR        | BraTS   | TBD  | TBD | TBD               | TBD         |
| SegResNet    | BraTS   | TBD  | TBD | TBD               | TBD         |

### 5.3 Statistical Analysis

*[ANOVA results and significance testing to be added]*

### 5.4 Architectural Insights

*[Analysis of architecture-dataset interactions to be populated]*

---

## 6. Discussion

### 6.1 Performance Trends

*[To be populated with analysis of results]*

### 6.2 Computational Efficiency

*[Comparison of training time and resource usage]*

### 6.3 Clinical Implications

*[Recommendations for practical deployment]*

### 6.4 Limitations

- Limited to three architectures and datasets
- Single institution data (no multi-center validation)
- Fixed hyperparameters across architectures
- Computational constraints limit batch size exploration

---

## 7. Conclusions

*[Key findings and recommendations to be populated after experiments]*

1. **Architecture Ranking**: [To be determined]
2. **Dataset-specific Recommendations**: [To be determined]  
3. **Efficiency-Performance Trade-offs**: [To be determined]
4. **Clinical Deployment Guidelines**: [To be determined]

---
## 8. Progress Timeline (Engineering Log)
- 2025-08-27: Repo initialized; structure scaffolded; requirements and setup added.
- 2025-08-27: Base config added for Colab paths; script stubs created.
- 2025-08-29: Dataset loaders implemented for BraTS, MSD Liver, TotalSegmentator.
- 2025-08-29: Preprocessing/augmentation transforms and dataloader builders implemented.
- 2025-08-29: Colab bootstrap fixed to clone repo to Drive and install requirements.
- 2025-08-29: EDA notebook stabilized; dataset folder synonyms tested on Drive.
- 2025-08-29: Standardized on folder name `TotalSegmentator`; removed deprecated synonyms.
- 2025-08-29: Simplified environment path logic in notebook.
- 2025-09-05: Model architectures (UNet, UNETR, SegResNet) and factory implemented.
- 2025-09-05: Loss functions and evaluation metrics added.
- 2025-09-05: Minimal trainer/evaluator with checkpointing implemented.
- 2025-09-05: Dependencies pinned for Colab compatibility; BraTS NIfTI dataset acquired.
- 2025-09-05: TotalSegmentator label fusion utility added for per-structure masks.
- 2025-09-05: **Validation checkpoint**: MSD Liver ✓, BraTS ✓ smoke tests passed (dataloader + UNet forward verified). TotalSegmentator has corrupted .nii.gz files (EOFError during LoadImaged) - requires re-download.
- 2025-09-08: **Training breakthrough**: Fixed UNet tensor mismatch by switching to BasicUNet. First successful training run completed on Colab A100 with BraTS dataset.
- 2025-09-08: **Production training**: Multiple experiments running successfully on Colab A100. UNet+BraTS, UNet+MSD, UNETR+BraTS all producing checkpoints. Resolved memory issues with forced patch-based validation.
- 2025-09-09: **Experiment completion**: UNet+BraTS training completed successfully (50 epochs, best model saved). UNet+MSD_Liver training completed.
- 2025-09-10: **UNETR experiments completed**: UNETR+BraTS and UNETR+MSD_Liver training runs completed successfully. 5/9 total experiments now finished.
- 2025-09-10: TotalSegmentator dataset corrupted files resolved; dataset fully functional.
- 2025-09-20: **Code refactoring**: Enhanced training script stability, improved data utilities, and configuration management.
- 2025-09-21: **Infrastructure improvements**: Updated trainer implementation with better error handling and memory management. Base configuration system refined.
- 2025-09-22: **Codebase optimization**: Removed deprecated runtime detection utilities; cleaned up configuration system. Code ready for remaining experiments.
- 2025-09-29: **Colab infrastructure overhaul**: Unified notebook with Drive persistence, streaming logs with ETA, auto-detection of dataset paths, per-dataset IO channel configuration. Fixed SegResNet MONAI compatibility (norm vs norm_name parameter). Repository cleanup: removed obsolete notebooks, moved dev scripts to scripts/dev/.
- 2025-09-29: **Dataset integration fixes**: Resolved MSD Liver path resolution (expects Task03_Liver subfolder). TotalSegmentator loader working with subject-level ct.nii.gz and segmentations/ structure. Added explicit empty dataset checks with clear error messages.
- 2025-09-30: **🎉 MAJOR MILESTONE - All 9 Training Experiments Complete**: Successfully trained complete matrix of 3 datasets × 3 architectures with proper checkpoints persisted to Drive.
  - **BraTS Dataset** (4→4 channels): ✅ UNet (~17s/epoch), ✅ UNETR, ✅ SegResNet
  - **MSD Liver** (1→3 channels): ✅ UNet (~1500s/epoch), ✅ UNETR, ✅ SegResNet  
  - **TotalSegmentator** (1→2 channels): ✅ UNet (~2000s/epoch), ✅ UNETR, ✅ SegResNet
  - Training infrastructure: Mixed precision (CUDA), streaming logs with epoch-by-epoch ETA, skip logic for completed runs, robust clone/pull for Drive-based repo
  - Technical achievements: Fixed trainer.py to use 'norm' not 'norm_name' for SegResNet; added sys.path management to train_model.py for subprocess imports; created scripts/run_experiments.py for streaming multi-run execution

### Planned Timeline (Revised):
- 2025-09-30: **Evaluation framework deployment**: Run comprehensive evaluation (scripts/evaluate_models.py) on all 9 checkpoints. Generate comparative metrics (Dice, IoU, model complexity).
- 2025-10-01: **Extended training runs**: Re-run experiments with 50-100 epochs for meaningful performance metrics (current 2-epoch runs were smoke tests with expected Dice~0).
- 2025-10-02: **Results analysis**: Statistical comparison, architecture performance across datasets, learning curve analysis.
- 2025-10-03: **Visualization & reporting**: Sample predictions, confusion matrices, technical report with findings and clinical insights.
- 2025-10-04: **Documentation finalization**: Complete reproducibility guide, update README with final results, archive code for publication.

## 9. Future Work

- Extend to additional architectures (nnU-Net, Swin-UNETR)
- Multi-institutional validation studies
- Uncertainty quantification and model calibration
- Real-time inference optimization for clinical deployment

---

## References

[1] Çiçek, Ö., et al. "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation." MICCAI 2016.

[2] Hatamizadeh, A., et al. "UNETR: Transformers for 3D Medical Image Segmentation." WACV 2022.

[3] He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.

[4] MONAI Consortium. "MONAI: Medical Open Network for AI." arXiv:2211.02701, 2022.

[5] Menze, B.H., et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." IEEE TMI 2015.

[6] Simpson, A.L., et al. "A Large Annotated Medical Image Dataset for the Development and evaluation of Segmentation Algorithms." arXiv:1902.09063, 2019.

[7] Wasserthal, J., et al. "TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images." Radiology: Artificial Intelligence 2023.

---

## Appendix A: Experimental Configuration

*[Detailed hyperparameters and configuration files]*

## Appendix B: Additional Results

*[Supplementary figures and detailed metrics]*