# Brain Tumor Segmentation and Survival Prediction

An end-to-end classical machine learning pipeline for automated brain tumor segmentation and survival prediction from multi-modal MRI, built on the BraTS 2020 dataset. 

## Overview

This project implements a four-stage pipeline that:
1. Preprocesses multi-modal MRI volumes (T1, T1ce, T2, FLAIR)
2. Segments brain tumors using multi-modal intensity thresholding
3. Extracts 114 radiomic features per patient across all modalities
4. Predicts patient survival (short/medium/long term) using an ensemble of classical ML models

## Key Results

| Metric | Baseline (FLAIR only) | Final Pipeline |
|--------|------------------------|-----------------|
| Mean Dice | 0.104 | **0.482** |
| Mean IoU | 0.064 | **0.348** |
| Survival Test Accuracy | 38% | **70%** |
| Survival Test ROC-AUC | 0.522 | **0.863** |
| Medium-class Recall | 0.00 | **0.58** |

**Top survival predictor:** Patient age (importance 0.1035), nearly double the second-ranked feature, confirming that clinical variables carry strong prognostic signal alongside imaging features.

## Dataset

[BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/) — 369 multi-modal MRI volumes (293 HGG, 76 LGG) with expert-annotated tumor masks. Survival data available for 235 patients (age 19-87, survival 5-1,767 days).

## Pipeline

### Stage 1 - Preprocessing
- Z-score normalization per modality over non-zero brain pixels
- Filtering to retain only tumor-containing slices (>=0.1% tumor pixel fraction)

### Stage 2 - Segmentation
- Multi-modal thresholding (FLAIR + T1ce + T2, lambda=1.5 sigma) combined via union
- Morphological post-processing: closing, hole filling, small object removal
- Top-2 connected component filtering to eliminate vascular false positives

### Stage 3 - Feature Extraction
- 22 radiomic features x 4 modalities = 88 features (geometric, shape, GLCM texture, intensity)
- Patient age, enhancing core ratio, tumor sub-region ratios (necrotic/edema/enhancing)
- 13 3D volumetric features (tumor volume, bounding box, aspect ratios)
- 114 total features, reduced to top 30 via Random Forest importance-based selection

### Stage 4 - Survival Prediction
- SVM, Random Forest, Gradient Boosting, and soft-voting ensemble
- SMOTE oversampling to address class imbalance (89/59/87 -> 89/89/89)
- 5-fold stratified cross-validation + held-out test set evaluation

## Setup

```bash
pip install numpy pandas scikit-learn scikit-image scipy h5py matplotlib imbalanced-learn
```

Place BraTS 2020 data (pre-sliced HDF5 format) in `data/slices/`, with `meta_data.csv`, `name_mapping.csv`, and `survival_info.csv` in `data/`.

## Usage

```bash
# Full pipeline run
python main.py

# Quick test on a subset of volumes
python main.py --max_volumes 20

# Use watershed segmentation instead of multi-modal thresholding
python main.py --seg_method watershed

# Binary survival classification (short/long) instead of 3-class
python main.py --binary_survival

# Generate all visualization plots from saved results
python visualize.py
```

## Results Visualizations

Running `visualize.py` produces:
- `segmentation_performance.png` - Dice/IoU distributions across all volumes
- `slice_examples.png` - High/medium/low Dice example segmentations
- `feature_distributions.png` - Radiomic feature box plots
- `feature_importance.png` - Top 15 Random Forest feature importances
- `survival_analysis.png` - Survival cohort analysis and class-wise violin plots
- `confusion_matrices.png` - Test set confusion matrices for all four classifiers

## Future Work

- U-Net deep learning segmentation (expected Dice > 0.85)
- Survival regression / Cox proportional hazards instead of discrete classes
- Additional clinical variables (extent of resection, MGMT methylation, IDH status)
- 3D volumetric deep learning end-to-end pipeline

## Acknowledgments

BraTS 2020 dataset (Menze et al. 2015, Bakas et al. 2017)