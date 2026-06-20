"""
Stage 4: Survival Prediction
- Bins continuous survival days into short / medium / long categories
- Trains SVM, Random Forest, and Gradient Boosting classifiers
- SMOTE oversampling to address class imbalance
- Returns trained models and evaluation metrics
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix)


# ---------------------------------------------------------------------------
# Survival binning
# ---------------------------------------------------------------------------

SURVIVAL_BINS   = [0, 300, 450, np.inf]
SURVIVAL_LABELS = ["short", "medium", "long"]


def bin_survival(days: pd.Series,
                 bins: list = None,
                 labels: list = None) -> pd.Series:
    """Convert continuous survival days to categorical labels."""
    if bins is None:
        bins = SURVIVAL_BINS
    if labels is None:
        labels = SURVIVAL_LABELS
    days = pd.to_numeric(days, errors="coerce")
    return pd.cut(days, bins=bins, labels=labels, right=False)


# ---------------------------------------------------------------------------
# SMOTE oversampling
# ---------------------------------------------------------------------------

def apply_smote(X: np.ndarray, y: np.ndarray):
    """
    Apply SMOTE to balance class distribution.
    Safely handles cases where a class has too few samples for SMOTE.

    Returns:
        X_resampled, y_resampled — balanced arrays
    """
    try:
        from imblearn.over_sampling import SMOTE
        min_samples = int(np.bincount(y).min())
        k = min(5, min_samples - 1)
        if k < 1:
            print("  SMOTE skipped: not enough samples in smallest class")
            return X, y
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X, y)
        print(f"  SMOTE applied: {len(y)} -> {len(y_res)} samples")
        print(f"  Class counts after SMOTE: {np.bincount(y_res).tolist()}")
        return X_res, y_res
    except ImportError:
        print("  SMOTE skipped: imbalanced-learn not installed")
        print("  Install with: pip install imbalanced-learn")
        return X, y
    except Exception as e:
        print(f"  SMOTE failed ({e}), using original data")
        return X, y


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_svm(C: float = 1.0, kernel: str = "rbf", **kwargs) -> Pipeline:
    """SVM with StandardScaler preprocessing."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=C, kernel=kernel, probability=True,
                    class_weight="balanced", random_state=42, **kwargs))
    ])


def build_random_forest(n_estimators: int = 100, max_depth=None,
                        **kwargs) -> Pipeline:
    """Random Forest with class balancing."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        class_weight="balanced",
                                        random_state=42, **kwargs))
    ])


def build_gradient_boosting(n_estimators: int = 100,
                             max_depth: int = 3,
                             learning_rate: float = 0.1,
                             **kwargs) -> Pipeline:
    """
    Gradient Boosting classifier — often outperforms Random Forest
    on tabular medical data with class imbalance.
    Note: does not support class_weight natively; SMOTE handles imbalance.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42, **kwargs))
    ])


def build_ensemble(**kwargs) -> Pipeline:
    """
    Soft-voting ensemble of SVM, Random Forest, and Gradient Boosting.
    Each model votes with probability estimates; final prediction is the
    class with the highest average predicted probability across all three.
    This typically outperforms any individual model on tabular medical data.
    """
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=1.0, kernel="rbf", probability=True,
                    class_weight="balanced", random_state=42))
    ])
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                        random_state=42))
    ])
    gb = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                            learning_rate=0.1, random_state=42))
    ])
    return VotingClassifier(
        estimators=[("svm", svm), ("rf", rf), ("gb", gb)],
        voting="soft"
    )


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(X: np.ndarray, y: np.ndarray,
                        model_name: str = "random_forest",
                        cv_folds: int = 5) -> dict:
    """
    Train a model using stratified k-fold cross-validation.

    Args:
        X:          Feature matrix (n_samples x n_features).
        y:          Integer class labels.
        model_name: 'svm', 'random_forest', or 'gradient_boosting'.
        cv_folds:   Number of cross-validation folds.
    """
    if model_name == "svm":
        model = build_svm()
    elif model_name == "random_forest":
        model = build_random_forest()
    elif model_name == "gradient_boosting":
        model = build_gradient_boosting()
    elif model_name == "ensemble":
        model = build_ensemble()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision_macro", "recall_macro",
               "f1_macro", "roc_auc_ovr_weighted"]

    scores = cross_validate(model, X, y, cv=cv, scoring=scoring,
                             return_train_score=False)

    return {
        "model":          model_name,
        "cv_folds":       cv_folds,
        "accuracy_mean":  scores["test_accuracy"].mean(),
        "accuracy_std":   scores["test_accuracy"].std(),
        "precision_mean": scores["test_precision_macro"].mean(),
        "recall_mean":    scores["test_recall_macro"].mean(),
        "f1_mean":        scores["test_f1_macro"].mean(),
        "roc_auc_mean":   scores["test_roc_auc_ovr_weighted"].mean(),
    }


def fit_final_model(X_train: np.ndarray, y_train: np.ndarray,
                     model_name: str = "random_forest"):
    """Fit a model on the full training set and return it."""
    if model_name == "svm":
        model = build_svm()
    elif model_name == "random_forest":
        model = build_random_forest()
    elif model_name == "gradient_boosting":
        model = build_gradient_boosting()
    elif model_name == "ensemble":
        model = build_ensemble()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.fit(X_train, y_train)
    return model


def evaluate_on_test(model, X_test: np.ndarray, y_test: np.ndarray,
                      class_names: list = None) -> dict:
    """Generate a full evaluation report on held-out test data."""
    if class_names is None:
        class_names = SURVIVAL_LABELS

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    present_labels = np.unique(np.concatenate([y_test, y_pred]))
    report = classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=[class_names[i] for i in present_labels],
        zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    n_classes = len(np.unique(y_test))
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test, y_prob,
                                multi_class="ovr", average="weighted")
    except ValueError:
        auc = float("nan")

    return {
        "classification_report": report,
        "confusion_matrix":      cm,
        "roc_auc":               float(auc),
    }


def feature_importances(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importances from a tree-based pipeline or ensemble.
    For ensembles, uses the Gradient Boosting component importances.
    """
    # Handle VotingClassifier ensemble
    if hasattr(model, "estimators_"):
        # Get GB pipeline from ensemble
        for name, est in zip(model.estimators, model.estimators_):
            if name[0] == "gb":
                clf = est.named_steps["clf"]
                if hasattr(clf, "feature_importances_"):
                    importances = clf.feature_importances_
                    return (pd.DataFrame({"feature": feature_names,
                                          "importance": importances})
                            .sort_values("importance", ascending=False)
                            .reset_index(drop=True))
        raise AttributeError("Could not extract importances from ensemble.")
    # Handle regular pipeline
    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        raise AttributeError("Model does not expose feature_importances_.")
    importances = clf.feature_importances_
    return (pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))
