"""
Model Evaluation & Comparison Module
====================================
Milestone 5.2: Baseline Evaluation & Comparison Table
Owner: ML Lead (Person B)
Branch: feature/ml-models

Evaluates trained baseline classifiers on the untouched test split:
- Metrics: Accuracy, Precision (Macro), Recall (Macro), Macro F1, Weighted F1,
  Per-class metrics, and Confusion Matrices.
- Primary selection criterion: Macro F1 (pre-declared in Docs/model_selection_criterion.md).
- Output: reports/model_comparison.csv and reports/figures/confusion_matrices/*.png.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger("water_quality.models.evaluate")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON_CSV = _REPO_ROOT / "reports" / "model_comparison.csv"
DEFAULT_CONFUSION_DIR = _REPO_ROOT / "reports" / "figures" / "confusion_matrices"

# Canonical ordered class labels
CANONICAL_CLASSES: List[str] = [
    "Unsuitable",
    "Very Poor",
    "Poor",
    "Good",
    "Excellent",
]


def evaluate_model(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate a single classifier on the test split.

    Parameters
    ----------
    model : fitted estimator
        A fitted classifier exposing a .predict(X) method.
    X_test : pd.DataFrame or np.ndarray
        Preprocessed test feature matrix.
    y_test : pd.Series or np.ndarray
        Ground-truth class labels.
    labels : list[str], optional
        Class labels to order the confusion matrix and per-class reports.
        Defaults to sorted unique values in y_test.

    Returns
    -------
    dict
        Comprehensive evaluation metrics dictionary.
    """
    y_true = np.asarray(y_test)
    y_pred = model.predict(X_test)

    if labels is None:
        if hasattr(model, "classes_") and model.classes_ is not None:
            labels = list(model.classes_)
        else:
            labels = sorted(list(np.unique(y_true)))

    # 1. Global aggregate metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # 2. Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # 3. Per-class metrics
    prec_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class_metrics = {}
    for idx, cls_name in enumerate(labels):
        per_class_metrics[cls_name] = {
            "precision": float(prec_per_class[idx]),
            "recall": float(rec_per_class[idx]),
            "f1": float(f1_per_class[idx]),
            "support": int(np.sum(y_true == cls_name)),
        }

    training_time = getattr(model, "training_time_", None)

    return {
        "accuracy": acc,
        "precision": prec_macro,
        "precision_macro": prec_macro,
        "recall": rec_macro,
        "recall_macro": rec_macro,
        "macro_f1": f1_macro,
        "f1_macro": f1_macro,
        "weighted_f1": f1_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "labels": labels,
        "per_class": per_class_metrics,
        "per_class_precision": {k: v["precision"] for k, v in per_class_metrics.items()},
        "per_class_recall": {k: v["recall"] for k, v in per_class_metrics.items()},
        "per_class_f1": {k: v["f1"] for k, v in per_class_metrics.items()},
        "training_time": training_time,
        "predictions": y_pred,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    model_name: str,
    output_path: Union[str, Path],
) -> Path:
    """Generate and save an aesthetically styled confusion matrix plot.

    Parameters
    ----------
    cm : np.ndarray
        2D confusion matrix array.
    labels : list[str]
        Class label names corresponding to axes.
    model_name : str
        Human-readable model name for the figure title.
    output_path : str or Path
        Destination filepath for the saved image (.png).

    Returns
    -------
    Path
        Absolute path to the saved figure.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Normalized proportions for color intensity
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

    cax = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row Normalized Ratio", rotation=270, labelpad=15)

    n_classes = len(labels)
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells with count and percentage
    thresh = cm_normalized.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            count = cm[i, j]
            pct = cm_normalized[i, j] * 100
            color = "white" if cm_normalized[i, j] > thresh else "black"
            ax.text(
                j, i,
                f"{count}\n({pct:.1f}%)",
                ha="center", va="center",
                color=color, fontsize=8,
            )

    ax.set_ylabel("Actual Ground Truth", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("  Saved confusion matrix plot: %s", out_file)
    return out_file


def evaluate_all_models(
    models: Dict[str, Any],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    output_csv_path: Optional[Union[str, Path]] = None,
    figures_dir: Optional[Union[str, Path]] = None,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """Evaluate all models, build comparison table, and save confusion matrices.

    Parameters
    ----------
    models : dict[str, fitted estimator]
        Mapping from model name to fitted estimator.
    X_test : pd.DataFrame or np.ndarray
        Test feature matrix.
    y_test : pd.Series or np.ndarray
        Test target labels.
    output_csv_path : str or Path, optional
        Path to save reports/model_comparison.csv.
    figures_dir : str or Path, optional
        Directory to save confusion matrix plots.
    save_outputs : bool, default True
        Whether to write CSV and image files to disk.

    Returns
    -------
    pd.DataFrame
        Model comparison table matching spec section 26.
    """
    csv_file = Path(output_csv_path) if output_csv_path is not None else DEFAULT_COMPARISON_CSV
    fig_dir = Path(figures_dir) if figures_dir is not None else DEFAULT_CONFUSION_DIR

    logger.info("===== M5.2 BASELINE MODEL EVALUATION START =====")
    logger.info("  Evaluating %d models on test set (n=%d)", len(models), len(X_test))

    rows: List[Dict[str, Any]] = []

    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        training_time = metrics["training_time"]
        time_str = f"{training_time:.4f}" if training_time is not None else "N/A"

        row = {
            "Model": name,
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "Macro F1": round(metrics["macro_f1"], 4),
            "Weighted F1": round(metrics["weighted_f1"], 4),
            "Training Time (s)": round(training_time, 4) if training_time is not None else np.nan,
        }
        rows.append(row)

        logger.info(
            "  %-20s | Acc: %.4f | Prec: %.4f | Rec: %.4f | Macro F1: %.4f | W-F1: %.4f | Time: %ss",
            name,
            row["Accuracy"],
            row["Precision"],
            row["Recall"],
            row["Macro F1"],
            row["Weighted F1"],
            time_str,
        )

        if save_outputs:
            slug = name.lower().replace(" ", "_")
            out_img = fig_dir / f"confusion_matrix_{slug}.png"
            plot_confusion_matrix(
                metrics["confusion_matrix"],
                metrics["labels"],
                name,
                out_img,
            )

    df_comparison = pd.DataFrame(rows)

    if save_outputs:
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        df_comparison.to_csv(csv_file, index=False)
        logger.info("  Saved model comparison table: %s", csv_file)

    logger.info("===== M5.2 EVALUATION COMPLETE =====")
    return df_comparison


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.models.train import perform_split, train_baseline_models

    split_res = perform_split(save_artifacts=False)
    trained_models = train_baseline_models(
        split_res.X_train_processed,
        split_res.y_train,
        random_seed=split_res.random_seed,
        save_models=True,
    )

    comparison_df = evaluate_all_models(
        trained_models,
        split_res.X_test_processed,
        split_res.y_test,
        save_outputs=True,
    )

    print("\n[Baseline Model Comparison Table (Spec Section 26)]")
    print(comparison_df.to_string(index=False))
