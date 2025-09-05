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
| 3D U-Net     | Exp 1 | Exp 2     | Exp 3            |
| UNETR        | Exp 4 | Exp 5     | Exp 6            |
| SegResNet    | Exp 7 | Exp 8     | Exp 9            |

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

**SegResNet**: Initial filters 32, blocks [1,2,2,4] down / [1,1,1] up, group normalization (8 groups), dropout 0.2.

### 3.5 Training Configuration

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

**Platform**: Google Colab (A100 GPU), Google Drive storage  
**Framework**: PyTorch 2.0, MONAI 1.3+ for medical-specific transforms and metrics  
**Reproducibility**: Fixed random seeds, version-controlled configurations  

Data augmentation includes spatial transforms (flip, rotation, scaling) and intensity jittering during training. Validation uses center cropping without augmentation.

---

## 5. Results

### 5.1 Performance Comparison

*[Table to be populated with experimental results]*

| Architecture | Dataset | Dice | IoU | Training Time (h) | Memory (GB) |
|--------------|---------|------|-----|-------------------|-------------|
| 3D U-Net     | BraTS   | -    | -   | -                 | -           |
| UNETR        | BraTS   | -    | -   | -                 | -           |
| SegResNet    | BraTS   | -    | -   | -                 | -           |

### 5.2 Statistical Analysis

*[ANOVA results and significance testing to be added]*

### 5.3 Architectural Insights

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
- 2025-09-05: **Validation checkpoint**: MSD Liver ✓, BraTS ✓ smoke tests passed (dataloader + UNet forward verified). TotalSegmentator pending transform debug.

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