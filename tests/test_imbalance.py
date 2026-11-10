"""
Unit tests for Milestone 6.1: Class Imbalance Handling
=======================================================
Tests:
- TC-6.1-01: Resampling is applied ONLY on training data, never on test data.
- TC-6.1-02: Minority class recall comparison — post-resampling vs baseline.
- TC-6.1-03: Imbalance handling decision is documented in Docs/model_selection_criterion.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.train import (
    perform_split,
    resample_training_data,
    train_baseline_models,
)
from src.models.evaluate import evaluate_model


_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def split_data():
    """Module-level fixture providing the frozen split without re-saving artifacts."""
    return perform_split(save_artifacts=False)


# -----------------------------------------------------------------------
# TC-6.1-01: Leakage Check — Resampling Only on Training Data
# -----------------------------------------------------------------------
class TestTC_6_1_01_LeakageCheck:
    """TC-6.1-01: Resampling MUST be applied strictly to X_train/y_train.
    The test set (X_test, y_test) must remain untouched."""

    def test_resample_none_passthrough(self, split_data):
        """When method='none', X_train and y_train are returned unchanged."""
        X_train = split_data.X_train_processed
        y_train = split_data.y_train

        X_out, y_out = resample_training_data(X_train, y_train, method="none")

        # Must be the exact same objects (passthrough, no copy)
        assert X_out is X_train, "Passthrough should return the exact same X_train object"
        assert y_out is y_train, "Passthrough should return the exact same y_train object"

    def test_resample_smote_only_changes_train_size(self, split_data):
        """SMOTE resampling changes training set size but not the structure."""
        X_train = split_data.X_train_processed
        y_train = split_data.y_train
        n_original = len(y_train)

        X_res, y_res = resample_training_data(
            X_train, y_train, method="smote", random_seed=42
        )

        # SMOTE creates synthetic samples -> resampled train must be >= original
        assert len(y_res) >= n_original, (
            f"SMOTE should increase training size: got {len(y_res)} < original {n_original}"
        )
        # Feature dimensionality must be preserved
        assert X_res.shape[1] == X_train.shape[1], (
            f"Feature columns changed after SMOTE: {X_res.shape[1]} vs {X_train.shape[1]}"
        )

    def test_resample_ros_only_changes_train_size(self, split_data):
        """RandomOverSampler duplicates minority samples -> increases training set."""
        X_train = split_data.X_train_processed
        y_train = split_data.y_train
        n_original = len(y_train)

        X_res, y_res = resample_training_data(
            X_train, y_train, method="random_oversampler", random_seed=42
        )

        assert len(y_res) >= n_original
        assert X_res.shape[1] == X_train.shape[1]

    def test_test_set_unchanged_after_resampled_training(self, split_data):
        """Training with resampling must NOT modify X_test or y_test."""
        X_test = split_data.X_test_processed.copy()
        y_test = split_data.y_test.copy()

        # Train with SMOTE
        train_baseline_models(
            split_data.X_train_processed,
            split_data.y_train,
            random_seed=42,
            save_models=False,
            resampling_method="smote",
        )

        # Test set must be bit-identical
        np.testing.assert_array_equal(
            split_data.X_test_processed, X_test,
            err_msg="X_test was modified after resampled training"
        )
        np.testing.assert_array_equal(
            split_data.y_test.values, y_test.values,
            err_msg="y_test was modified after resampled training"
        )


# -----------------------------------------------------------------------
# TC-6.1-02: Minority Class Recall Comparison
# -----------------------------------------------------------------------
class TestTC_6_1_02_MinorityRecallComparison:
    """TC-6.1-02: Compare minority class recall for baseline vs resampled models.
    The experiment result must be documented regardless of whether resampling improves recall."""

    def test_baseline_recall_computed(self, split_data):
        """Baseline models (class_weight='balanced', no resampling) produce valid per-class recall."""
        models = train_baseline_models(
            split_data.X_train_processed,
            split_data.y_train,
            random_seed=42,
            save_models=False,
            resampling_method="none",
        )

        for name, model in models.items():
            metrics = evaluate_model(model, split_data.X_test_processed, split_data.y_test)
            macro_f1 = metrics["macro_f1"]
            per_class_recall = metrics["per_class_recall"]

            # Macro F1 must be a valid number
            assert 0.0 <= macro_f1 <= 1.0, f"{name}: Macro F1 out of range: {macro_f1}"

            # Per-class recall must exist for all classes present in test
            unique_test_classes = set(split_data.y_test.unique())
            for cls in unique_test_classes:
                assert cls in per_class_recall, f"{name}: Missing recall for class '{cls}'"
                assert 0.0 <= per_class_recall[cls] <= 1.0, (
                    f"{name}: Recall for '{cls}' out of range: {per_class_recall[cls]}"
                )

    def test_resampling_experiment_results_saved(self):
        """The M6.1 experiment results JSON must exist with comparison data."""
        results_path = _REPO_ROOT / "reports" / "m61_imbalance_experiment.json"
        assert results_path.exists(), (
            f"M6.1 experiment results not found at {results_path}. "
            "Run the M6.1 imbalance experiment first."
        )

        import json
        with open(results_path) as f:
            results = json.load(f)

        # Must contain at least baseline and one resampling method
        assert len(results) >= 2, (
            f"Expected at least 2 resampling methods compared, got {len(results)}"
        )

        # Each method must contain results for all 4 models
        for method_name, method_results in results.items():
            assert len(method_results) >= 4, (
                f"Method '{method_name}' missing model results: {list(method_results.keys())}"
            )
            for model_name, model_results in method_results.items():
                assert "macro_f1" in model_results, (
                    f"{method_name}/{model_name}: Missing 'macro_f1' in results"
                )


# -----------------------------------------------------------------------
# TC-6.1-03: Documentation Verification
# -----------------------------------------------------------------------
class TestTC_6_1_03_DocumentationVerification:
    """TC-6.1-03: The imbalance handling decision must be documented in
    Docs/model_selection_criterion.md with justification."""

    def test_model_selection_doc_contains_imbalance_section(self):
        """model_selection_criterion.md must document the M6.1 decision."""
        doc_path = _REPO_ROOT / "Docs" / "model_selection_criterion.md"
        assert doc_path.exists(), f"Document not found: {doc_path}"

        content = doc_path.read_text(encoding="utf-8").lower()

        # Must mention class imbalance handling
        assert "imbalance" in content or "class imbalance" in content, (
            "model_selection_criterion.md missing 'imbalance' discussion"
        )

        # Must mention the decision (SMOTE/none/class_weight)
        assert "class_weight" in content or "balanced" in content, (
            "model_selection_criterion.md missing class_weight/balanced discussion"
        )

        # Must reference M6.1 or Milestone 6.1
        assert "6.1" in content, (
            "model_selection_criterion.md missing M6.1 / Milestone 6.1 reference"
        )

    def test_config_resampling_method_matches_decision(self):
        """config.yaml resampling_method must match the documented M6.1 decision."""
        import yaml
        config_path = _REPO_ROOT / "config" / "config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        method = cfg.get("resampling_method", "none")
        # The experiment showed baseline class_weight='balanced' is best,
        # so resampling_method should remain 'none'
        assert method == "none", (
            f"config.yaml resampling_method is '{method}', expected 'none' "
            "based on M6.1 experiment showing class_weight='balanced' is optimal"
        )
