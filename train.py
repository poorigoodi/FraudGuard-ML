"""
FraudGuard-ML — model training script.

Trains Logistic Regression and XGBoost, selects the best model by
validation Average Precision, tunes a decision threshold on validation
data (maximize F1), evaluates once on the held-out test set, and saves
the final artifact to models/fraud_model.pkl.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODELS_DIR = ROOT / "models"
ARTIFACT_PATH = MODELS_DIR / "fraud_model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
FIGURES_DIR = MODELS_DIR / "figures"

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------
def load_and_validate_data(path: Path) -> pd.DataFrame:
    """Load creditcard.csv and validate schema, types, and labels."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download creditcard.csv from:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place it in the data/ directory."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Ensure features are numeric
    for col in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[FEATURE_COLUMNS].isna().any().any():
        n_bad = int(df[FEATURE_COLUMNS].isna().any(axis=1).sum())
        raise ValueError(
            f"Found {n_bad} rows with non-numeric or missing feature values."
        )

    # Handle missing values (drop if any remain in features/target)
    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing values.")

    # Validate binary target
    unique_labels = set(df[TARGET_COLUMN].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Target column 'Class' must contain only 0/1 labels. "
            f"Found: {sorted(unique_labels)}"
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified 70 / 15 / 15 train / validation / test split."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print(
        f"Split sizes - train: {len(X_train):,}, "
        f"val: {len(X_val):,}, test: {len(X_test):,}"
    )
    print(
        f"Fraud rates  - train: {y_train.mean():.4%}, "
        f"val: {y_val.mean():.4%}, test: {y_test.mean():.4%}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_logistic_regression() -> Pipeline:
    """Logistic Regression with StandardScaler and balanced class weights."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgboost(y_train: pd.Series) -> XGBClassifier:
    """XGBoost with scale_pos_weight from training class distribution only."""
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    print(f"XGBoost scale_pos_weight (train only): {scale_pos_weight:.2f}")

    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics at a given probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }


def select_threshold_by_f1(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """
    Choose the decision threshold that maximizes F1 on validation data.

    Returns (best_threshold, best_f1).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thresholds) = len(precisions) - 1
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (
        precisions[:-1] + recalls[:-1] + 1e-12
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def print_metrics(title: str, metrics: dict[str, float], threshold: float) -> None:
    print(f"\n{title} (threshold={threshold:.4f})")
    print("-" * 50)
    for name, value in metrics.items():
        print(f"  {name:20s}: {value:.4f}")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def save_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    title: str,
    path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_pr_curve(
    results: dict[str, dict[str, Any]],
    y_val: pd.Series,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, info in results.items():
        precision, recall, _ = precision_recall_curve(y_val, info["y_prob_val"])
        ap = info["val_metrics"]["average_precision"]
        ax.plot(recall, precision, label=f"{name} (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve (Validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_roc_curve(
    results: dict[str, dict[str, Any]],
    y_val: pd.Series,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, info in results.items():
        fpr, tpr, _ = roc_curve(y_val, info["y_prob_val"])
        auc = info["val_metrics"]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_comparison_chart(
    results: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]
    model_names = list(results.keys())
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(model_names):
        values = [results[name]["val_metrics"][m] for m in metric_names]
        ax.bar(x + i * width, values, width, label=name)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "AP"],
        rotation=15,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (Validation, threshold=0.5)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main training workflow
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("FraudGuard-ML - Training")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_validate_data(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    models: dict[str, Any] = {
        "Logistic Regression": build_logistic_regression(),
        "XGBoost": build_xgboost(y_train),
    }

    results: dict[str, dict[str, Any]] = {}

    print("\n--- Training models ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_prob_val = model.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_prob_val, threshold=0.5)
        print_metrics(f"{name} - Validation", val_metrics, 0.5)
        results[name] = {
            "model": model,
            "y_prob_val": y_prob_val,
            "val_metrics": val_metrics,
        }

    # Select best model by validation Average Precision
    best_name = max(
        results,
        key=lambda n: results[n]["val_metrics"]["average_precision"],
    )
    best_model = results[best_name]["model"]
    best_prob_val = results[best_name]["y_prob_val"]

    print("\n" + "=" * 60)
    print(f"Selected model (highest validation AP): {best_name}")
    print("=" * 60)

    # Threshold tuning on validation only (maximize F1)
    best_threshold, best_f1 = select_threshold_by_f1(y_val, best_prob_val)
    print(
        f"Selected threshold (max validation F1={best_f1:.4f}): "
        f"{best_threshold:.4f}"
    )

    tuned_val_metrics = compute_metrics(y_val, best_prob_val, best_threshold)
    print_metrics(
        f"{best_name} - Validation (tuned threshold)",
        tuned_val_metrics,
        best_threshold,
    )

    # Final evaluation on untouched test set
    y_prob_test = best_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_prob_test, best_threshold)
    print_metrics(
        f"{best_name} - Final Test Evaluation",
        test_metrics,
        best_threshold,
    )

    y_pred_test = (y_prob_test >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_test)
    print("\nFinal Test Confusion Matrix:")
    print(f"  TN={cm[0, 0]:,}  FP={cm[0, 1]:,}")
    print(f"  FN={cm[1, 0]:,}  TP={cm[1, 1]:,}")

    print("\n--- Concise Evaluation Summary ---")
    print(f"  Model     : {best_name}")
    print(f"  Threshold : {best_threshold:.4f} (chosen on validation F1)")
    print(f"  Test AP   : {test_metrics['average_precision']:.4f}")
    print(f"  Test F1   : {test_metrics['f1']:.4f}")
    print(f"  Test Recall: {test_metrics['recall']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")

    # Visualizations
    print("\n--- Saving figures ---")
    for name, info in results.items():
        y_pred = (info["y_prob_val"] >= 0.5).astype(int)
        save_confusion_matrix(
            y_val,
            y_pred,
            f"{name} - Validation Confusion Matrix (t=0.5)",
            FIGURES_DIR / f"cm_val_{name.lower().replace(' ', '_')}.png",
        )

    save_confusion_matrix(
        y_test,
        y_pred_test,
        f"{best_name} - Test Confusion Matrix (t={best_threshold:.3f})",
        FIGURES_DIR / "cm_test_final.png",
    )
    save_pr_curve(results, y_val, FIGURES_DIR / "pr_curve_validation.png")
    save_roc_curve(results, y_val, FIGURES_DIR / "roc_curve_validation.png")
    save_comparison_chart(results, FIGURES_DIR / "model_comparison.png")
    print(f"Figures saved to {FIGURES_DIR}")

    # Persist artifact
    artifact = {
        "model": best_model,
        "threshold": best_threshold,
        "feature_names": FEATURE_COLUMNS,
        "model_name": best_name,
    }
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"\nSaved model artifact -> {ARTIFACT_PATH}")

    # Persist metrics for README / documentation
    metrics_payload = {
        "selection_metric": "validation_average_precision",
        "selected_model": best_name,
        "selected_threshold": best_threshold,
        "threshold_selection": "maximize_validation_f1",
        "validation": {
            name: info["val_metrics"] for name, info in results.items()
        },
        "validation_tuned": tuned_val_metrics,
        "test": test_metrics,
        "test_confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved metrics         -> {METRICS_PATH}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
