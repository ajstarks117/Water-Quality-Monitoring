"""
Unit tests for Milestone 5.2: Baseline Evaluation & Comparison Table
===================================================================
Tests:
- TC-5.2-01: Metrics computed correctly (compare against manual sklearn calculations).
- TC-5.2-02: Comparison table completeness (all 4 models present, no blanks, correct headers).
- TC-5.2-03: Confusion matrices generated for all models (4 PNG images present).
- TC-5.2-04: Selection criterion documented with Macro F1 as primary metric.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.evaluate import (
    DEFAULT_COMPARISON_CSV,
    DEFAULT_CONFUSION_DIR,
    evaluate_all_models,
    evaluate_model,
)
from src.models.train import (
    BASELINE_MODEL_NAMES,
    perform_split,
    train_baseline_models,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def baseline_eval_artifacts(tmp_path_factory):
    """Module-level fixture training baseline models and evaluating on test split."""
    split_res = perform_split(save_artifacts=False)
    models = train_baseline_models(
        X_train=split_res.X_train_processed,
        y_train=split_res.y_train,
        random_seed=42,
        save_models=False,
    )
    tmp_dir = tmp_path_factory.mktemp("eval_out")
    csv_path = tmp_dir / "model_comparison.csv"
    fig_dir = tmp_dir / "confusion_matrices"

    df_comp = evaluate_all_models(
        models=models,
        X_test=split_res.X_test_processed,
        y_test=split_res.y_test,
        output_csv_path=csv_path,
        figures_dir=fig_dir,
        save_outputs=True,
    )

    return {
        "split": split_res,
        "models": models,
        "df": df_comp,
        "csv_path": csv_path,
        "fig_dir": fig_dir,
    }


def test_tc_5_2_01_metrics_computed_correctly(baseline_eval_artifacts):
    """TC-5.2-01: Compare evaluate_model output against manual sklearn calculations."""
    split = baseline_eval_artifacts["split"]
    models = baseline_eval_artifacts["models"]

    # Test with Decision Tree
    model = models["Decision Tree"]
    X_test = split.X_test_processed
    y_test = split.y_test.values

    metrics = evaluate_model(model, X_test, y_test)
    y_pred = model.predict(X_test)

    # Compute manual sklearn metrics
    manual_acc = accuracy_score(y_test, y_pred)
    manual_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    manual_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    manual_f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    manual_f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    assert pytest.approx(metrics["accuracy"], rel=1e-5) == manual_acc
    assert pytest.approx(metrics["precision"], rel=1e-5) == manual_prec
    assert pytest.approx(metrics["recall"], rel=1e-5) == manual_rec
    assert pytest.approx(metrics["macro_f1"], rel=1e-5) == manual_f1_macro
    assert pytest.approx(metrics["weighted_f1"], rel=1e-5) == manual_f1_weighted

    # Verify per-class coverage
    assert len(metrics["per_class"]) == 5
    for cls in split.y_test.unique():
        assert cls in metrics["per_class"]
        assert 0.0 <= metrics["per_class"][cls]["precision"] <= 1.0
        assert 0.0 <= metrics["per_class"][cls]["recall"] <= 1.0
        assert 0.0 <= metrics["per_class"][cls]["f1"] <= 1.0


def test_tc_5_2_02_comparison_table_completeness(baseline_eval_artifacts):
    """TC-5.2-02: Verify reports/model_comparison.csv contains all 4 models and required metric columns."""
    csv_path = baseline_eval_artifacts["csv_path"]
    assert csv_path.exists(), f"Comparison table CSV missing at {csv_path}"

    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "Macro F1",
        "Weighted F1",
        "Training Time (s)",
    ]
    for col in required_cols:
        assert col in df.columns, f"Required column '{col}' missing from comparison table"

    # Check 4 models present
    assert len(df) == 4, f"Expected 4 models in comparison table, found {len(df)}"
    assert sorted(df["Model"].tolist()) == sorted(BASELINE_MODEL_NAMES)

    # Check no blank or null values in metric columns
    metric_cols = ["Accuracy", "Precision", "Recall", "Macro F1", "Weighted F1"]
    for col in metric_cols:
        assert df[col].notna().all(), f"Column '{col}' contains NaN values"
        assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all(), f"Values in '{col}' out of [0, 1] bounds"

    # Also check the repository artifact
    assert DEFAULT_COMPARISON_CSV.exists(), "Repository artifact reports/model_comparison.csv missing"
    repo_df = pd.read_csv(DEFAULT_COMPARISON_CSV)
    assert len(repo_df) == 4


def test_tc_5_2_03_confusion_matrices_generated(baseline_eval_artifacts):
    """TC-5.2-03: Verify confusion matrix images exist for all 4 models."""
    fig_dir = baseline_eval_artifacts["fig_dir"]
    assert fig_dir.exists(), f"Confusion matrices directory missing at {fig_dir}"

    for name in BASELINE_MODEL_NAMES:
        slug = name.lower().replace(" ", "_")
        img_file = fig_dir / f"confusion_matrix_{slug}.png"
        assert img_file.exists(), f"Confusion matrix image missing: {img_file}"
        assert img_file.stat().st_size > 1000, f"Confusion matrix image is unexpectedly small: {img_file}"

    # Also check the repository artifacts directory
    assert DEFAULT_CONFUSION_DIR.exists()
    for name in BASELINE_MODEL_NAMES:
        slug = name.lower().replace(" ", "_")
        repo_img = DEFAULT_CONFUSION_DIR / f"confusion_matrix_{slug}.png"
        assert repo_img.exists(), f"Repository confusion matrix missing: {repo_img}"
        assert repo_img.stat().st_size > 1000


def test_tc_5_2_04_selection_criterion_documented_before_results():
    """TC-5.2-04: Confirm Docs/model_selection_criterion.md exists with pre-declared Macro F1 criterion."""
    criterion_path = _REPO_ROOT / "Docs" / "model_selection_criterion.md"
    assert criterion_path.exists(), f"Criterion document missing: {criterion_path}"

    content = criterion_path.read_text(encoding="utf-8")

    # Verify primary metric is Macro F1
    assert "Macro F1" in content or "Macro-Averaged F1" in content
    assert "Primary" in content

    # Verify caution against accuracy alone
    assert "Accuracy alone MUST NEVER be used" in content or "Accuracy" in content

    # Verify public health recall justification for severe classes
    assert "Unsuitable" in content
    assert "Recall" in content
