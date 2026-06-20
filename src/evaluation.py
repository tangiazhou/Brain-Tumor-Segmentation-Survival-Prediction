"""
Evaluation metrics for segmentation (Dice, IoU) and classification
(accuracy, precision, recall, ROC-AUC).
"""

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                               recall_score, f1_score, roc_auc_score)


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray,
                     smooth: float = 1e-6) -> float:
    """
    Sørensen–Dice coefficient between two binary masks.

    DSC = 2|A ∩ B| / (|A| + |B|)

    Args:
        y_true: Ground-truth binary mask (0/1).
        y_pred: Predicted binary mask (0/1).
        smooth: Laplace smoothing to avoid division by zero.

    Returns:
        Dice score in [0, 1].
    """
    y_true = y_true.astype(bool).flatten()
    y_pred = y_pred.astype(bool).flatten()
    intersection = (y_true & y_pred).sum()
    return float((2.0 * intersection + smooth) /
                 (y_true.sum() + y_pred.sum() + smooth))


def iou_score(y_true: np.ndarray, y_pred: np.ndarray,
              smooth: float = 1e-6) -> float:
    """
    Intersection over Union (Jaccard index).

    IoU = |A ∩ B| / |A ∪ B|
    """
    y_true = y_true.astype(bool).flatten()
    y_pred = y_pred.astype(bool).flatten()
    intersection = (y_true & y_pred).sum()
    union = (y_true | y_pred).sum()
    return float((intersection + smooth) / (union + smooth))


def pixel_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly classified pixels."""
    return float((y_true.flatten() == y_pred.flatten()).mean())


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray,
                smooth: float = 1e-6) -> float:
    """True positive rate (recall) for binary segmentation masks."""
    y_true = y_true.astype(bool).flatten()
    y_pred = y_pred.astype(bool).flatten()
    tp = (y_true & y_pred).sum()
    fn = (y_true & ~y_pred).sum()
    return float((tp + smooth) / (tp + fn + smooth))


def specificity(y_true: np.ndarray, y_pred: np.ndarray,
                smooth: float = 1e-6) -> float:
    """True negative rate for binary segmentation masks."""
    y_true = y_true.astype(bool).flatten()
    y_pred = y_pred.astype(bool).flatten()
    tn = (~y_true & ~y_pred).sum()
    fp = (~y_true & y_pred).sum()
    return float((tn + smooth) / (tn + fp + smooth))


def segmentation_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all segmentation metrics and return as dict."""
    return {
        "dice": dice_coefficient(y_true, y_pred),
        "iou": iou_score(y_true, y_pred),
        "pixel_accuracy": pixel_accuracy(y_true, y_pred),
        "sensitivity": sensitivity(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Classification metrics (survival prediction)
# ---------------------------------------------------------------------------

def classification_metrics(y_true, y_pred, y_prob=None,
                             average: str = "macro") -> dict:
    """
    Compute accuracy, precision, recall, F1, and optionally ROC-AUC.

    Args:
        y_true:   Ground-truth class labels.
        y_pred:   Predicted class labels.
        y_prob:   Predicted probabilities (n_samples x n_classes). Optional.
        average:  Averaging strategy for multiclass ('macro', 'weighted').

    Returns:
        Dict of metric scores.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average,
                                      zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average,
                                zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    if y_prob is not None:
        n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                 average=average)
        metrics["roc_auc"] = float(auc)
    return metrics
