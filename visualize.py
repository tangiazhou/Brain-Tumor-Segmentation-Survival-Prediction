"""
visualize.py — Generate result plots for all pipeline stages.

Usage:
    python visualize.py
    python visualize.py --features_dir outputs/features --data_dir data

Produces plots saved to outputs/plots/:
    1. segmentation_performance.png  — Dice & IoU distributions across volumes
    2. feature_distributions.png     — Box plots of key extracted features
    3. feature_importance.png        — Random Forest feature importances
    4. survival_analysis.png         — Class balance + per-feature violin plots
    5. slice_examples.png            — Sample MRI slices with GT vs predicted masks
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

PLOT_DIR = os.path.join("outputs", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

C_BLUE   = "#2E86AB"
C_ORANGE = "#E84855"
C_GREEN  = "#3BB273"
C_PURPLE = "#7B2D8B"
C_GRAY   = "#8D99AE"
PALETTE  = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_GRAY]

# Verified channel order from channel_check.png
FLAIR_IDX = 2


# ---------------------------------------------------------------------------
# 1. Segmentation performance
# ---------------------------------------------------------------------------

def plot_segmentation_performance(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Segmentation Performance Across Volumes",
                 fontsize=14, fontweight="bold")

    axes[0].hist(df["mean_dice"], bins=20, color=C_BLUE,
                 edgecolor="white", linewidth=0.5)
    axes[0].axvline(df["mean_dice"].mean(), color=C_ORANGE, linestyle="--",
                    linewidth=2, label=f'Mean = {df["mean_dice"].mean():.3f}')
    axes[0].set_title("Dice Coefficient Distribution")
    axes[0].set_xlabel("Dice Score")
    axes[0].set_ylabel("Number of Volumes")
    axes[0].legend()

    axes[1].hist(df["mean_iou"], bins=20, color=C_GREEN,
                 edgecolor="white", linewidth=0.5)
    axes[1].axvline(df["mean_iou"].mean(), color=C_ORANGE, linestyle="--",
                    linewidth=2, label=f'Mean = {df["mean_iou"].mean():.3f}')
    axes[1].set_title("IoU Score Distribution")
    axes[1].set_xlabel("IoU Score")
    axes[1].set_ylabel("Number of Volumes")
    axes[1].legend()

    axes[2].scatter(df["mean_dice"], df["mean_iou"], alpha=0.6, color=C_PURPLE,
                    edgecolors="white", linewidth=0.3, s=40)
    lim = max(df["mean_dice"].max(), df["mean_iou"].max()) * 1.05
    axes[2].plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1, label="y = x")
    axes[2].set_title("Dice vs IoU per Volume")
    axes[2].set_xlabel("Dice Score")
    axes[2].set_ylabel("IoU Score")
    axes[2].legend()

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "segmentation_performance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 2. Feature distributions — uses modality-prefixed column names
# ---------------------------------------------------------------------------

def plot_feature_distributions(df: pd.DataFrame):
    """
    Plot box plots for each feature group using FLAIR-prefixed features
    as representative, since features are now extracted per modality.
    """
    feature_groups = {
        "Geometric": [
            "flair_area_px", "flair_perimeter",
            "flair_bbox_height", "flair_bbox_width"
        ],
        "Shape": [
            "flair_compactness", "flair_eccentricity",
            "flair_solidity", "flair_extent"
        ],
        "Intensity": [
            "flair_intensity_mean", "flair_intensity_std",
            "flair_intensity_skew", "flair_intensity_kurt"
        ],
        "Texture": [
            "flair_glcm_contrast", "flair_glcm_homogeneity",
            "flair_glcm_energy", "flair_entropy"
        ],
    }

    # Short display labels (strip flair_ prefix for readability)
    short_labels = {
        "flair_area_px": "area\npx",
        "flair_perimeter": "perimeter",
        "flair_bbox_height": "bbox\nheight",
        "flair_bbox_width": "bbox\nwidth",
        "flair_compactness": "compactness",
        "flair_eccentricity": "eccentricity",
        "flair_solidity": "solidity",
        "flair_extent": "extent",
        "flair_intensity_mean": "intensity\nmean",
        "flair_intensity_std": "intensity\nstd",
        "flair_intensity_skew": "intensity\nskew",
        "flair_intensity_kurt": "intensity\nkurt",
        "flair_glcm_contrast": "glcm\ncontrast",
        "flair_glcm_homogeneity": "glcm\nhomogeneity",
        "flair_glcm_energy": "glcm\nenergy",
        "flair_entropy": "entropy",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Feature Distributions Across Volumes (FLAIR channel)",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for ax, (group_name, features) in zip(axes, feature_groups.items()):
        present = [f for f in features if f in df.columns]
        if not present:
            continue
        data = [df[f].dropna().values for f in present]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="white", linewidth=2))
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels([short_labels.get(f, f) for f in present], fontsize=8)
        ax.set_title(f"{group_name} Features (FLAIR)")
        ax.set_ylabel("Value")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "feature_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 3. Feature importances
# ---------------------------------------------------------------------------

def plot_feature_importance(fi_path: str):
    if not os.path.exists(fi_path):
        print("  Skipping feature importance plot (file not found)")
        return

    fi_df = pd.read_csv(fi_path).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(fi_df))]
    bars = ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_title("Top 15 Feature Importances (Random Forest)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, fi_df["importance"][::-1]):
        ax.text(bar.get_width() + 0.0005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 4. Survival analysis
# ---------------------------------------------------------------------------

def plot_survival_analysis(df: pd.DataFrame, survival_path: str):
    if not os.path.exists(survival_path):
        print("  Skipping survival analysis plot (survival_info.csv not found)")
        return

    survival = pd.read_csv(survival_path)
    survival["Survival_days"] = pd.to_numeric(survival["Survival_days"],
                                               errors="coerce")
    survival = survival.dropna(subset=["Survival_days"])

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Survival Analysis", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Survival days histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(survival["Survival_days"], bins=30, color=C_BLUE,
             edgecolor="white", linewidth=0.5)
    ax1.axvline(300, color=C_ORANGE, linestyle="--", linewidth=1.5,
                label="short/med (300d)")
    ax1.axvline(450, color=C_GREEN, linestyle="--", linewidth=1.5,
                label="med/long (450d)")
    ax1.set_title("Survival Days Distribution")
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8)

    # Class balance pie
    ax2 = fig.add_subplot(gs[0, 1])
    bins   = [0, 300, 450, np.inf]
    labels = ["Short\n(<300d)", "Medium\n(300-450d)", "Long\n(>450d)"]
    counts = pd.cut(survival["Survival_days"], bins=bins, labels=labels,
                    right=False).value_counts().reindex(labels)
    wedges, texts, autotexts = ax2.pie(
        counts, labels=labels, autopct="%1.1f%%",
        colors=[C_ORANGE, C_BLUE, C_GREEN],
        wedgeprops=dict(edgecolor="white", linewidth=2)
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax2.set_title("Survival Class Balance")

    # Age vs survival scatter
    if "Age" in survival.columns:
        ax3 = fig.add_subplot(gs[0, 2])
        age = pd.to_numeric(survival["Age"], errors="coerce")
        sc = ax3.scatter(age, survival["Survival_days"], alpha=0.5,
                         c=survival["Survival_days"], cmap="viridis",
                         edgecolors="white", linewidth=0.3, s=30)
        plt.colorbar(sc, ax=ax3, label="Survival Days")
        ax3.set_title("Age vs Survival Days")
        ax3.set_xlabel("Age (years)")
        ax3.set_ylabel("Survival Days")

    # Violin plots by survival class — use modality-prefixed features
    if "survival_days" in df.columns:
        df = df.copy()
        df["survival_class"] = pd.cut(
            pd.to_numeric(df["survival_days"], errors="coerce"),
            bins=bins, labels=["Short", "Medium", "Long"], right=False
        )
        # Use FLAIR features for violin plots
        violin_features = [
            "flair_intensity_mean",
            "flair_area_px",
            "flair_glcm_contrast"
        ]
        violin_features = [f for f in violin_features if f in df.columns]
        violin_titles   = ["Intensity Mean by Class",
                           "Area Px by Class",
                           "Glcm Contrast by Class"]

        for idx, (feat, title) in enumerate(zip(violin_features[:3],
                                                 violin_titles[:3])):
            ax = fig.add_subplot(gs[1, idx])
            groups = [df[df["survival_class"] == cls][feat].dropna().values
                      for cls in ["Short", "Medium", "Long"]]
            groups = [g for g in groups if len(g) > 1]
            if groups:
                parts = ax.violinplot(groups, showmedians=True)
                for i, pc in enumerate(parts["bodies"]):
                    pc.set_facecolor([C_ORANGE, C_BLUE, C_GREEN][i % 3])
                    pc.set_alpha(0.7)
                ax.set_xticks(range(1, len(groups) + 1))
                ax.set_xticklabels(["Short", "Medium", "Long"][:len(groups)])
                ax.set_title(title)
                ax.set_ylabel("Value")
                ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "survival_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 5. Slice examples — uses segment_multimodal with correct channel indices
# ---------------------------------------------------------------------------

def plot_slice_examples(data_dir: str, features_csv: str = None):
    """
    Show three representative volumes: one high-Dice, one medium-Dice,
    one low-Dice — selected from volume_features.csv so the examples
    are actually representative of the range of segmentation quality.
    Falls back to volumes 1, 2, 3 if features CSV is not available.
    """
    try:
        from src.preprocessing import load_volume_slices, binarize_mask
        from src.segmentation import segment_multimodal
    except ImportError as e:
        print(f"  Skipping slice examples: {e}")
        return

    slices_dir = os.path.join(data_dir, "slices")
    if not os.path.exists(slices_dir):
        print("  Skipping slice examples (slices dir not found)")
        return

    # Select representative volumes from Dice distribution
    if features_csv and os.path.exists(features_csv):
        df_feats = pd.read_csv(features_csv)[["volume_id", "mean_dice"]]
        df_feats = df_feats.dropna().sort_values("mean_dice")
        n = len(df_feats)
        # Pick low (10th percentile), medium (50th), high (90th)
        low_id  = int(df_feats.iloc[int(n * 0.10)]["volume_id"])
        med_id  = int(df_feats.iloc[int(n * 0.50)]["volume_id"])
        high_id = int(df_feats.iloc[int(n * 0.90)]["volume_id"])
        vol_ids = [high_id, med_id, low_id]
        labels  = [
            f"Vol {high_id} — High Dice ({df_feats.iloc[int(n*0.90)]['mean_dice']:.3f})",
            f"Vol {med_id} — Med Dice ({df_feats.iloc[int(n*0.50)]['mean_dice']:.3f})",
            f"Vol {low_id} — Low Dice ({df_feats.iloc[int(n*0.10)]['mean_dice']:.3f})",
        ]
        print(f"  Slice examples: high={high_id}, med={med_id}, low={low_id}")
    else:
        vol_ids = [1, 2, 3]
        labels  = [f"Volume {v}" for v in vol_ids]

    n_volumes = len(vol_ids)
    fig, axes = plt.subplots(n_volumes, 4, figsize=(14, 4 * n_volumes))
    fig.suptitle("Sample Slices: FLAIR | Ground Truth | Prediction | Overlay\n"
                 "(High / Medium / Low Dice)",
                 fontsize=13, fontweight="bold")
    col_titles = ["FLAIR Input", "Ground Truth Mask",
                  "Predicted Mask", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for row, (vol_id, row_label) in enumerate(zip(vol_ids, labels)):
        slices = load_volume_slices(slices_dir, vol_id,
                                    normalize=True, tumor_only=True)
        if not slices:
            for ax in axes[row]:
                ax.axis("off")
            axes[row, 0].set_ylabel(row_label, fontsize=8)
            continue

        entry     = slices[len(slices) // 2]
        image     = entry["image"]
        flair     = image[..., FLAIR_IDX]
        gt_binary = binarize_mask(entry["mask"])
        pred_mask = segment_multimodal(image)

        axes[row, 0].imshow(flair, cmap="gray")
        axes[row, 0].set_ylabel(row_label, fontsize=8)

        axes[row, 1].imshow(flair, cmap="gray")
        axes[row, 1].imshow(
            np.ma.masked_where(gt_binary == 0, gt_binary),
            cmap="Greens", alpha=0.6, vmin=0, vmax=1)

        axes[row, 2].imshow(flair, cmap="gray")
        axes[row, 2].imshow(
            np.ma.masked_where(pred_mask == 0, pred_mask),
            cmap="Reds", alpha=0.6, vmin=0, vmax=1)

        overlay = np.zeros((*flair.shape, 4))
        tp = (gt_binary > 0) & (pred_mask > 0)
        fp = (gt_binary == 0) & (pred_mask > 0)
        fn = (gt_binary > 0) & (pred_mask == 0)
        overlay[tp] = [1.0, 1.0, 0.0, 0.7]
        overlay[fp] = [1.0, 0.0, 0.0, 0.6]
        overlay[fn] = [0.0, 1.0, 0.0, 0.6]
        axes[row, 3].imshow(flair, cmap="gray")
        axes[row, 3].imshow(overlay)
        axes[row, 3].legend(
            handles=[Patch(color="yellow", label="TP"),
                     Patch(color="red",    label="FP"),
                     Patch(color="green",  label="FN")],
            loc="lower right", fontsize=7, framealpha=0.7
        )

    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "slice_examples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")



# ---------------------------------------------------------------------------
# 6. Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(features_path: str, survival_path: str,
                             data_dir: str):
    """
    Train all four classifiers and plot their confusion matrices
    on the held-out test set.
    """
    if not os.path.exists(survival_path):
        print("  Skipping confusion matrices (survival_info.csv not found)")
        return

    from src.survival_prediction import (bin_survival, apply_smote,
                                          fit_final_model, SURVIVAL_LABELS)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix as sk_cm

    # Load and merge data
    df = pd.read_csv(features_path)
    name_map = pd.read_csv(os.path.join(data_dir, "name_mapping.csv"))
    survival = pd.read_csv(survival_path)
    name_map["volume_id"] = (name_map["BraTS_2020_subject_ID"]
                              .str.extract(r"_(\d+)$")[0].astype(int))
    merged = name_map[["volume_id", "BraTS_2020_subject_ID"]].merge(
        survival[["Brats20ID", "Survival_days", "Age"]],
        left_on="BraTS_2020_subject_ID", right_on="Brats20ID", how="inner"
    ).rename(columns={"Survival_days": "survival_days", "Age": "age"})
    merged["survival_days"] = pd.to_numeric(merged["survival_days"],
                                             errors="coerce")
    merged["age"] = pd.to_numeric(merged["age"], errors="coerce")
    df = df.merge(merged[["volume_id", "survival_days", "age"]],
                  on="volume_id", how="inner")
    df["survival_class"] = bin_survival(df["survival_days"])
    df = df.dropna(subset=["survival_class"])

    if len(df) < 10:
        print("  Skipping confusion matrices (insufficient data)")
        return

    le = LabelEncoder()
    y  = le.fit_transform(df["survival_class"])
    drop_cols    = ["volume_id", "survival_days", "survival_class",
                    "mean_dice", "mean_iou", "n_tumor_slices"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].fillna(0).values

    # Apply SMOTE and split
    X_res, y_res = apply_smote(X, y)
    X_train_s, _, y_train_s, _ = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model_names = ["svm", "random_forest", "gradient_boosting", "ensemble"]
    class_labels = SURVIVAL_LABELS

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Confusion Matrices — Test Set (Trained on SMOTE)",
                 fontsize=13, fontweight="bold")

    for ax, model_name in zip(axes, model_names):
        model  = fit_final_model(X_train_s, y_train_s, model_name=model_name)
        y_pred = model.predict(X_test)

        # Build confusion matrix — handle missing classes in test set
        present = np.unique(np.concatenate([y_test, y_pred]))
        cm = sk_cm(y_test, y_pred, labels=present)
        labels = [class_labels[i] for i in present]

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(model_name.replace("_", " ").title(), fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        # Annotate cells
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=12,
                        color="white" if cm[i, j] > thresh else "black")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(features_dir: str, data_dir: str):
    features_path = os.path.join(features_dir, "volume_features.csv")
    fi_path       = os.path.join(features_dir, "feature_importances.csv")
    survival_path = os.path.join(data_dir, "survival_info.csv")

    if not os.path.exists(features_path):
        print(f"ERROR: {features_path} not found. Run main.py first.")
        return

    df = pd.read_csv(features_path)
    print(f"Loaded features for {len(df)} volumes.")
    print("Generating plots...")

    # Merge survival days for violin plots
    if os.path.exists(survival_path):
        name_map = pd.read_csv(os.path.join(data_dir, "name_mapping.csv"))
        survival = pd.read_csv(survival_path)
        name_map["volume_id"] = (name_map["BraTS_2020_subject_ID"]
                                  .str.extract(r"_(\d+)$")[0].astype(int))
        merged = name_map[["volume_id", "BraTS_2020_subject_ID"]].merge(
            survival[["Brats20ID", "Survival_days"]],
            left_on="BraTS_2020_subject_ID",
            right_on="Brats20ID",
            how="inner"
        ).rename(columns={"Survival_days": "survival_days"})
        merged["survival_days"] = pd.to_numeric(merged["survival_days"],
                                                 errors="coerce")
        df = df.merge(merged[["volume_id", "survival_days"]],
                      on="volume_id", how="left")

    plot_segmentation_performance(df)
    plot_feature_distributions(df)
    plot_feature_importance(fi_path)
    plot_survival_analysis(df, survival_path)
    plot_slice_examples(data_dir, features_csv=features_path)
    plot_confusion_matrices(features_path, survival_path, data_dir)

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize brain tumor pipeline results")
    parser.add_argument("--features_dir", type=str, default="outputs/features")
    parser.add_argument("--data_dir",     type=str, default="data")
    args = parser.parse_args()
    main(args.features_dir, args.data_dir)