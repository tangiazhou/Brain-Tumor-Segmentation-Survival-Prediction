"""
main.py — End-to-end pipeline for Brain Tumor Segmentation & Survival Prediction

Usage:
    python main.py
    python main.py --max_volumes 20
    python main.py --seg_method multimodal
    python main.py --seg_method watershed
    python main.py --binary_survival
"""

import os
import argparse
import numpy as np
import pandas as pd

from src.preprocessing import (load_volume_slices, binarize_mask,
                                get_channel, list_slices, parse_filename,
                                CHANNEL_NAMES)
from src.segmentation import segment_multimodal, segment_watershed
from src.feature_extraction import extract_all_features
from src.evaluation import segmentation_report
from src.survival_prediction import (bin_survival, apply_smote,
                                      train_and_evaluate,
                                      fit_final_model, evaluate_on_test,
                                      feature_importances, SURVIVAL_LABELS)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier as _QuickRF

OUTPUT_DIR = "outputs"

# Verified channel indices (from channel_check.png)
T1CE_IDX  = 0
T1_IDX    = 1
FLAIR_IDX = 2
T2_IDX    = 3


def load_metadata(data_dir):
    meta     = pd.read_csv(os.path.join(data_dir, "meta_data.csv"))
    name_map = pd.read_csv(os.path.join(data_dir, "name_mapping.csv"))
    survival = pd.read_csv(os.path.join(data_dir, "survival_info.csv"))
    return meta, name_map, survival


def build_volume_survival_map(name_map, survival):
    """
    Volume ID extracted from BraTS20 ID suffix: BraTS20_Training_042 -> 42
    Includes Age as a feature for survival prediction.
    """
    name_map = name_map.copy()
    name_map["volume_id"] = (name_map["BraTS_2020_subject_ID"]
                              .str.extract(r"_(\d+)$")[0].astype(int))

    merged = name_map[["volume_id", "BraTS_2020_subject_ID"]].merge(
        survival[["Brats20ID", "Survival_days", "Age"]],
        left_on="BraTS_2020_subject_ID",
        right_on="Brats20ID",
        how="inner"
    )
    merged = merged.rename(columns={
        "Survival_days": "survival_days",
        "Age":           "age"
    })
    merged["survival_days"] = pd.to_numeric(merged["survival_days"],
                                             errors="coerce")
    merged["age"] = pd.to_numeric(merged["age"], errors="coerce")
    result = merged[["volume_id", "survival_days", "age"]].dropna(
        subset=["survival_days"])
    print(f"Matched {len(result)} volumes to survival labels")
    return result


def get_t1ce_core_mask(image: np.ndarray,
                        pred_mask: np.ndarray) -> np.ndarray:
    """
    Extract enhancing core region from T1ce channel within predicted mask.
    Uses pixels above mean + 0.5 std of T1ce brain intensity.
    Falls back to full mask if core is too small.
    """
    t1ce  = image[..., T1CE_IDX]
    brain = t1ce[t1ce > 0]
    if len(brain) == 0 or pred_mask.sum() == 0:
        return pred_mask
    threshold = brain.mean() + 0.5 * brain.std()
    core = ((t1ce > threshold) & pred_mask.astype(bool)).astype(np.uint8)
    return core if core.sum() >= 10 else pred_mask


def extract_subregion_ratios(mask_gt: np.ndarray) -> dict:
    """
    Compute ratio of each tumor sub-region to total tumor volume.
        channel 0 = Necrotic core (NCR/NET)
        channel 1 = Peritumoral edema (ED)
        channel 2 = Enhancing tumor (ET)
    These are among the strongest clinical survival predictors.
    """
    total = (mask_gt.max(axis=-1) > 0).sum()
    if total == 0:
        return {"necrotic_ratio": 0.0, "edema_ratio": 0.0,
                "enhancing_ratio": 0.0}
    return {
        "necrotic_ratio":  float(mask_gt[..., 0].sum() / total),
        "edema_ratio":     float(mask_gt[..., 1].sum() / total),
        "enhancing_ratio": float(mask_gt[..., 2].sum() / total),
    }


def process_volume(slices_dir, volume_id, seg_method="multimodal"):
    """
    Run preprocessing, segmentation, and feature extraction for one volume.

    Feature strategy:
    - 88 features: 22 per modality across all 4 channels
    - T1ce features use enhancing core sub-mask (more prognostically specific)
    - Additional: enhancing_core_ratio, necrotic/edema/enhancing sub-region ratios
    """
    slices = load_volume_slices(slices_dir, volume_id,
                                normalize=True, tumor_only=True)
    if not slices:
        return {}

    slice_features, seg_metrics = [], []

    for entry in slices:
        image   = entry["image"]   # (240, 240, 4)
        mask_gt = entry["mask"]    # (240, 240, 3)
        gt_binary = binarize_mask(mask_gt)

        if seg_method == "watershed":
            pred_mask = segment_watershed(image)
        else:
            pred_mask = segment_multimodal(image)

        seg_metrics.append(segmentation_report(gt_binary, pred_mask))

        t1ce_core = get_t1ce_core_mask(image, pred_mask)

        feats = {}
        for i, modality in enumerate(CHANNEL_NAMES):
            channel     = image[..., i]
            mask_to_use = t1ce_core if modality == "t1ce" else pred_mask
            mod_feats   = extract_all_features(channel, mask_to_use)
            feats.update({f"{modality}_{k}": v
                          for k, v in mod_feats.items()})

        feats["enhancing_core_ratio"] = float(
            t1ce_core.sum() / (pred_mask.sum() + 1e-8)
        ) if pred_mask.sum() > 0 else 0.0

        feats.update(extract_subregion_ratios(mask_gt))
        slice_features.append(feats)

    if not slice_features:
        return {}

    # 2D slice-level aggregation (mean across slices)
    agg = {k: float(np.mean([f[k] for f in slice_features]))
           for k in slice_features[0]}
    agg["volume_id"]      = volume_id
    agg["n_tumor_slices"] = len(slice_features)
    agg["mean_dice"]      = float(np.mean([m["dice"] for m in seg_metrics]))
    agg["mean_iou"]       = float(np.mean([m["iou"]  for m in seg_metrics]))

    # 3D volumetric features — capture spatial context lost by 2D averaging
    # Collect all predicted masks and images across slices
    all_pred_masks = []
    all_gt_masks   = []
    all_flair      = []
    all_t1ce       = []
    for entry in slices:
        image   = entry["image"]
        mask_gt = entry["mask"]
        if seg_method == "watershed":
            pred = segment_watershed(image)
        else:
            pred = segment_multimodal(image)
        all_pred_masks.append(pred)
        all_gt_masks.append(binarize_mask(mask_gt))
        all_flair.append(image[..., FLAIR_IDX])
        all_t1ce.append(image[..., T1CE_IDX])

    pred_vol = np.stack(all_pred_masks, axis=0)   # (n_slices, H, W)
    gt_vol   = np.stack(all_gt_masks,   axis=0)
    flair_vol= np.stack(all_flair,      axis=0)
    t1ce_vol = np.stack(all_t1ce,       axis=0)

    # Total tumor volume (voxel count across all slices)
    agg["vol_3d_tumor_volume"]    = float(pred_vol.sum())
    agg["vol_3d_gt_volume"]       = float(gt_vol.sum())

    # 3D bounding box dimensions
    nonzero = np.argwhere(pred_vol > 0)
    if len(nonzero) > 0:
        agg["vol_3d_bbox_depth"]  = float(nonzero[:, 0].max() - nonzero[:, 0].min() + 1)
        agg["vol_3d_bbox_height"] = float(nonzero[:, 1].max() - nonzero[:, 1].min() + 1)
        agg["vol_3d_bbox_width"]  = float(nonzero[:, 2].max() - nonzero[:, 2].min() + 1)
        # Aspect ratios
        depth = agg["vol_3d_bbox_depth"]
        agg["vol_3d_aspect_dh"]   = float(depth / (agg["vol_3d_bbox_height"] + 1e-8))
        agg["vol_3d_aspect_dw"]   = float(depth / (agg["vol_3d_bbox_width"]  + 1e-8))
    else:
        for k in ["vol_3d_bbox_depth", "vol_3d_bbox_height", "vol_3d_bbox_width",
                  "vol_3d_aspect_dh", "vol_3d_aspect_dw"]:
            agg[k] = 0.0

    # 3D intensity statistics within predicted tumor volume
    tumor_flair = flair_vol[pred_vol > 0]
    tumor_t1ce  = t1ce_vol[pred_vol > 0]
    if len(tumor_flair) > 0:
        agg["vol_3d_flair_mean"]  = float(tumor_flair.mean())
        agg["vol_3d_flair_std"]   = float(tumor_flair.std())
        agg["vol_3d_flair_max"]   = float(tumor_flair.max())
        agg["vol_3d_t1ce_mean"]   = float(tumor_t1ce.mean())
        agg["vol_3d_t1ce_std"]    = float(tumor_t1ce.std())
        agg["vol_3d_t1ce_max"]    = float(tumor_t1ce.max())
    else:
        for k in ["vol_3d_flair_mean", "vol_3d_flair_std", "vol_3d_flair_max",
                  "vol_3d_t1ce_mean", "vol_3d_t1ce_std", "vol_3d_t1ce_max"]:
            agg[k] = 0.0

    return agg


def main(data_dir, max_volumes=None, seg_method="multimodal",
         binary_survival=False):

    slices_dir = os.path.join(data_dir, "slices")
    os.makedirs(os.path.join(OUTPUT_DIR, "features"), exist_ok=True)

    all_paths  = list_slices(slices_dir)
    volume_ids = sorted(set(parse_filename(p)[0] for p in all_paths))
    if max_volumes:
        volume_ids = volume_ids[:max_volumes]
    print(f"Found {len(volume_ids)} volume(s) in {slices_dir}")
    print(f"Segmentation method : {seg_method}")
    print(f"Survival mode       : {'binary' if binary_survival else '3-class'}")

    meta, name_map, survival = load_metadata(data_dir)
    vol_surv_df = build_volume_survival_map(name_map, survival)

    print("\nProcessing volumes...")
    all_features = []
    for vol_id in volume_ids:
        print(f"  Volume {vol_id:>3} ...", end=" ", flush=True)
        try:
            feats = process_volume(slices_dir, vol_id,
                                   seg_method=seg_method)
            if feats:
                all_features.append(feats)
                print(f"Dice={feats['mean_dice']:.3f}"
                      f"  slices={feats['n_tumor_slices']}")
            else:
                print("skipped (no tumor slices)")
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_features:
        print("No features extracted.")
        return

    df_features   = pd.DataFrame(all_features)
    features_path = os.path.join(OUTPUT_DIR, "features", "volume_features.csv")
    df_features.to_csv(features_path, index=False)
    print(f"\nFeatures saved -> {features_path}")
    print(f"Mean Dice : {df_features['mean_dice'].mean():.3f}")
    print(f"Mean IoU  : {df_features['mean_iou'].mean():.3f}")

    # -----------------------------------------------------------------------
    # Stage 4: Survival Prediction
    # -----------------------------------------------------------------------
    print("\n--- Stage 4: Survival Prediction ---")

    df = df_features.merge(vol_surv_df, on="volume_id", how="inner")

    if binary_survival:
        df["survival_class"] = pd.cut(
            df["survival_days"],
            bins=[0, 365, np.inf],
            labels=["short", "long"],
            right=False
        )
        class_names = ["short", "long"]
    else:
        df["survival_class"] = bin_survival(df["survival_days"])
        class_names = SURVIVAL_LABELS

    df = df.dropna(subset=["survival_class"])
    print(f"Patients with survival labels : {len(df)}")
    print(f"Class distribution:\n{df['survival_class'].value_counts().to_string()}")

    if len(df) < 10 or df["survival_class"].nunique() < 2:
        print("Insufficient labelled samples for survival prediction.")
        return

    le = LabelEncoder()
    y  = le.fit_transform(df["survival_class"])

    drop_cols    = ["volume_id", "survival_days", "survival_class",
                    "mean_dice", "mean_iou", "n_tumor_slices"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].fillna(0).values

    print(f"Feature matrix: {X.shape[0]} patients x {X.shape[1]} features")

    # Apply SMOTE to balance classes
    print("\nApplying SMOTE oversampling...")
    X_resampled, y_resampled = apply_smote(X, y)

    # ---------------------------------------------------------------------------
    # Feature selection — keep top N features to reduce redundancy
    # Rule of thumb: ~10 patients per feature; 235 patients -> ~30 features
    # Uses a quick Random Forest importance ranking on SMOTE-balanced data
    # ---------------------------------------------------------------------------
    N_FEATURES = 30
    print(f"\nSelecting top {N_FEATURES} features from {X.shape[1]}...")
    quick_rf = _QuickRF(n_estimators=50, random_state=42,
                        class_weight="balanced")
    quick_rf.fit(StandardScaler().fit_transform(X_resampled), y_resampled)
    importances  = quick_rf.feature_importances_
    top_indices  = np.argsort(importances)[-N_FEATURES:]
    top_indices  = np.sort(top_indices)   # keep original column order

    X            = X[:, top_indices]
    X_resampled  = X_resampled[:, top_indices]
    feature_cols = [feature_cols[i] for i in top_indices]
    print(f"Selected features: {feature_cols[:5]} ... (and {N_FEATURES-5} more)")

    # Cross-validated evaluation on resampled data
    for model_name in ["svm", "random_forest", "gradient_boosting", "ensemble"]:
        results = train_and_evaluate(X_resampled, y_resampled,
                                     model_name=model_name)
        print(f"\n{model_name.upper()} (5-fold CV, SMOTE):")
        print(f"  Accuracy : {results['accuracy_mean']:.3f}"
              f" ± {results['accuracy_std']:.3f}")
        print(f"  F1 Macro : {results['f1_mean']:.3f}")
        print(f"  ROC-AUC  : {results['roc_auc_mean']:.3f}")

    # Final test on original (unaugmented) held-out set
    # Train on SMOTE-resampled, test on original to avoid data leakage
    X_train_s, _, y_train_s, _ = train_test_split(
        X_resampled, y_resampled, test_size=0.2,
        stratify=y_resampled, random_state=42
    )
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n--- Test Set Results (trained on SMOTE, tested on original) ---")
    best_auc   = -1
    best_name  = "random_forest"
    all_models = {}
    for model_name in ["svm", "random_forest", "gradient_boosting", "ensemble"]:
        model        = fit_final_model(X_train_s, y_train_s,
                                       model_name=model_name)
        test_results = evaluate_on_test(model, X_test, y_test,
                                         class_names=class_names)
        all_models[model_name] = model
        print(f"\n{model_name.upper()} — Test Set:")
        print(test_results["classification_report"])
        auc = test_results["roc_auc"]
        if not np.isnan(auc):
            print(f"ROC-AUC (test): {auc:.3f}")
            if auc > best_auc:
                best_auc  = auc
                best_name = model_name
        else:
            print("ROC-AUC (test): nan (insufficient classes in test set)")

    if best_auc > 0:
        print(f"\nBest model: {best_name} (ROC-AUC = {best_auc:.3f})")
    else:
        print("\nROC-AUC undefined for test set (run full dataset for valid metrics)")
        best_name = "random_forest"

    # Feature importances — use RF or GB (not SVM which has no importances)
    fi_model_name = best_name if best_name not in ["svm"] else "random_forest"
    fi_model = all_models[fi_model_name]
    try:
        fi_df = feature_importances(fi_model, feature_cols)
        print(f"\nTop 10 Feature Importances ({fi_model_name}):")
        print(fi_df.head(10).to_string(index=False))
        fi_df.to_csv(
            os.path.join(OUTPUT_DIR, "features", "feature_importances.csv"),
            index=False
        )
    except AttributeError:
        print("Feature importances not available for selected model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Brain Tumor Segmentation & Survival Pipeline")
    parser.add_argument("--data_dir",        type=str,  default="data")
    parser.add_argument("--max_volumes",     type=int,  default=None)
    parser.add_argument("--seg_method",      type=str,  default="multimodal",
                        choices=["multimodal", "watershed"])
    parser.add_argument("--binary_survival", action="store_true",
                        help="Binary (short/long) instead of 3-class survival")
    args = parser.parse_args()
    main(args.data_dir, args.max_volumes,
         args.seg_method, args.binary_survival)