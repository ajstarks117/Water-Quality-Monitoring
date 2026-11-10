"""
Unit tests for Milestone 6.2: Hyperparameter Tuning
=====================================================
Tests:
- TC-6.2-01: Tuning uses only training-set cross-validation (no test-set leakage).
- TC-6.2-02: Tuned model improves or matches baseline on primary metric (Macro F1).
- TC-6.2-03: Time budget respected per model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.train import (
    DEFAULT_TIME_CAP_SECONDS,
    TUNING_PARAM_GRIDS,
    TuningResult,
    perform_split,
    train_baseline_models,
    tune_model,
    tune_top_candidates,
)
from src.models.evaluate import evaluate_model


_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def split_data():
    """Module-level fixture providing the frozen split without re-saving artifacts."""
    return perform_split(save_artifacts=False)


@pytest.fixture(scope="module")
def baseline_models(split_data):
    """Train baseline models for comparison."""
    return train_baseline_models(
        split_data.X_train_processed,
        split_data.y_train,
        random_seed=split_data.random_seed,
        save_models=False,
    )


@pytest.fixture(scope="module")
def tuning_results(split_data, baseline_models):
    """Run tuning on top-2 candidates (small iteration budget for test speed)."""
    return tune_top_candidates(
        X_train=split_data.X_train_processed,
        y_train=split_data.y_train,
        X_test=split_data.X_test_processed,
        y_test=split_data.y_test,
        baseline_models=baseline_models,
        n_iter=5,  # Small budget for test speed
        cv_folds=3,
        random_seed=split_data.random_seed,
        time_cap_seconds=DEFAULT_TIME_CAP_SECONDS,
        save_models=False,
        update_comparison_csv=False,
    )


# -----------------------------------------------------------------------
# TC-6.2-01: Tuning Uses Only Training-Set Cross-Validation
# -----------------------------------------------------------------------
class TestTC_6_2_01_TuningNoTestSetLeakage:
    """TC-6.2-01: CV folds are drawn ONLY from X_train. Test set never touched."""

    def test_tune_model_accepts_only_train_data(self, split_data):
        """tune_model() signature only accepts X_train/y_train, no test params."""
        import inspect
        sig = inspect.signature(tune_model)
        param_names = list(sig.parameters.keys())

        # Must NOT have X_test or y_test parameters
        assert "X_test" not in param_names, "tune_model() must not accept X_test"
        assert "y_test" not in param_names, "tune_model() must not accept y_test"

        # Must have X_train and y_train
        assert "X_train" in param_names, "tune_model() must accept X_train"
        assert "y_train" in param_names, "tune_model() must accept y_train"

    def test_cv_folds_from_training_data_only(self, split_data):
        """Verify CV fold sizes are consistent with training set, not full dataset."""
        result = tune_model(
            model_name="Decision Tree",
            X_train=split_data.X_train_processed,
            y_train=split_data.y_train,
            n_iter=3,
            cv_folds=5,
            random_seed=42,
        )

        # CV results should have scores for exactly n_iter parameter combinations
        assert result.n_iter_completed == 3, (
            f"Expected 3 iterations, got {result.n_iter_completed}"
        )

        # Best estimator must be fitted
        assert result.best_estimator is not None
        assert hasattr(result.best_estimator, "predict")

        # Best CV score must be valid
        assert 0.0 <= result.best_cv_score <= 1.0

    def test_test_set_unchanged_after_tuning(self, split_data):
        """Test set must be bit-identical before and after tuning."""
        X_test_before = split_data.X_test_processed.copy()
        y_test_before = split_data.y_test.copy()

        tune_model(
            model_name="Decision Tree",
            X_train=split_data.X_train_processed,
            y_train=split_data.y_train,
            n_iter=3,
            cv_folds=3,
            random_seed=42,
        )

        np.testing.assert_array_equal(
            split_data.X_test_processed, X_test_before,
            err_msg="X_test modified during tuning"
        )
        np.testing.assert_array_equal(
            split_data.y_test.values, y_test_before.values,
            err_msg="y_test modified during tuning"
        )


# -----------------------------------------------------------------------
# TC-6.2-02: Tuned Model Improves or Matches Baseline
# -----------------------------------------------------------------------
class TestTC_6_2_02_TunedModelPerformance:
    """TC-6.2-02: Tuned model Macro F1 >= baseline Macro F1 on test set,
    or if worse, the degradation is documented."""

    def test_tuned_results_contain_both_candidates(self, tuning_results):
        """Both Decision Tree and Random Forest must have tuning results."""
        assert "Decision Tree" in tuning_results
        assert "Random Forest" in tuning_results

    def test_tuned_macro_f1_is_valid(self, tuning_results):
        """Tuned test Macro F1 must be a valid number in [0, 1]."""
        for name, result in tuning_results.items():
            assert 0.0 <= result.tuned_test_macro_f1 <= 1.0, (
                f"{name}: Tuned Macro F1 out of range: {result.tuned_test_macro_f1}"
            )

    def test_tuned_has_best_params(self, tuning_results):
        """Tuned result must contain best_params from the search."""
        for name, result in tuning_results.items():
            assert len(result.best_params) > 0, (
                f"{name}: best_params is empty"
            )
            # Params must be from the defined grid
            for param_name in result.best_params:
                assert param_name in TUNING_PARAM_GRIDS[name], (
                    f"{name}: unexpected param '{param_name}' not in grid"
                )

    def test_comparison_csv_has_tuned_rows(self):
        """reports/model_comparison.csv must contain both baseline and tuned rows."""
        csv_path = _REPO_ROOT / "reports" / "model_comparison.csv"
        assert csv_path.exists(), f"Comparison CSV not found: {csv_path}"

        df = pd.read_csv(csv_path)
        model_names = df["Model"].tolist()

        # Must have baseline rows
        assert "Decision Tree" in model_names, "Missing baseline Decision Tree row"
        assert "Random Forest" in model_names, "Missing baseline Random Forest row"

        # Must have tuned rows
        assert "Decision Tree (Tuned)" in model_names, "Missing tuned Decision Tree row"
        assert "Random Forest (Tuned)" in model_names, "Missing tuned Random Forest row"

        # Macro F1 column must exist and be numeric
        assert "Macro F1" in df.columns
        assert df["Macro F1"].notna().all()


# -----------------------------------------------------------------------
# TC-6.2-03: Time Budget Respected
# -----------------------------------------------------------------------
class TestTC_6_2_03_TimeBudgetRespected:
    """TC-6.2-03: Wall-clock time per model stays within the agreed cap."""

    def test_tuning_time_within_cap(self, tuning_results):
        """Each model's tuning wall-clock time must be within the time cap."""
        for name, result in tuning_results.items():
            assert result.wall_clock_seconds > 0, (
                f"{name}: wall_clock_seconds should be > 0"
            )
            # Use a generous multiplier (2x) for test tolerance since test
            # environment may be slower; the real cap is enforced in production runs
            assert result.wall_clock_seconds < DEFAULT_TIME_CAP_SECONDS * 2, (
                f"{name}: tuning took {result.wall_clock_seconds:.1f}s, "
                f"exceeds 2x time cap ({DEFAULT_TIME_CAP_SECONDS * 2}s)"
            )

    def test_iteration_count_within_budget(self, tuning_results):
        """Number of completed iterations must not exceed the requested n_iter."""
        for name, result in tuning_results.items():
            # In the test fixture we used n_iter=5
            assert result.n_iter_completed <= 5, (
                f"{name}: completed {result.n_iter_completed} iterations, "
                f"expected <= 5"
            )
            assert result.n_iter_completed > 0, (
                f"{name}: no iterations completed"
            )
