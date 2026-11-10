"""
Unit tests for Milestone 5.1: Baseline Model Training
======================================================
Tests:
- TC-5.1-01: All 4 models train successfully and checkpoints saved.
- TC-5.1-02: Identical input across models (same preprocessed X_train, y_train).
- TC-5.1-03: Prediction sanity check (valid classes, shape (5,) on 5 test rows).
- TC-5.1-04: Reproducibility across runs with the same random_seed.
"""

from pathlib import Path
import numpy as np
import pytest

from src.models.train import (
    BASELINE_MODEL_NAMES,
    perform_split,
    train_baseline_models,
)


@pytest.fixture(scope="module")
def split_data():
    """Module-level fixture providing the frozen split data without re-saving artifacts."""
    return perform_split(save_artifacts=False)


def test_tc_5_1_01_all_4_models_train_successfully(split_data, tmp_path):
    """TC-5.1-01: All 4 models train successfully and return fitted estimators."""
    X_train = split_data.X_train_processed
    y_train = split_data.y_train

    save_dir = tmp_path / "experiments"
    models = train_baseline_models(
        X_train=X_train,
        y_train=y_train,
        random_seed=42,
        save_dir=save_dir,
        save_models=True,
    )

    # 1. Exact 4 expected model names
    assert sorted(models.keys()) == sorted(BASELINE_MODEL_NAMES), (
        f"Expected models {BASELINE_MODEL_NAMES}, got {list(models.keys())}"
    )

    # 2. Check each model is fitted and has training_time_ logged
    for name, model in models.items():
        assert hasattr(model, "classes_"), f"Model {name} missing classes_ attribute"
        assert len(model.classes_) == 5, f"Model {name} should have 5 classes, got {len(model.classes_)}"
        assert hasattr(model, "predict"), f"Model {name} missing predict method"
        assert hasattr(model, "training_time_"), f"Model {name} missing training_time_ attribute"
        assert model.training_time_ > 0.0, f"Model {name} training time should be > 0"

        # 3. Checkpoint saved to experiments dir
        slug = name.lower().replace(" ", "_")
        expected_file = save_dir / f"baseline_{slug}.pkl"
        assert expected_file.exists(), f"Checkpoint file missing: {expected_file}"
        assert expected_file.stat().st_size > 0, f"Checkpoint file is empty: {expected_file}"


def test_tc_5_1_02_identical_input_across_models(split_data, monkeypatch):
    """TC-5.1-02: Ensure every model consumes the EXACT same preprocessed X_train and y_train."""
    X_train = split_data.X_train_processed
    y_train = split_data.y_train

    # Capture X and y shapes and memory hashes seen by each model's fit() call
    seen_inputs = []

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from src.models.train import XGBoostWaterQualityClassifier

    orig_lr_fit = LogisticRegression.fit
    orig_dt_fit = DecisionTreeClassifier.fit
    orig_rf_fit = RandomForestClassifier.fit
    orig_xgb_fit = XGBoostWaterQualityClassifier.fit

    def track_fit(self, X, y, *args, **kwargs):
        seen_inputs.append((X.shape, len(y), np.array(X, copy=True), np.array(y, copy=True)))

    def tracking_wrapper(orig_fit):
        def wrapped(self, X, y, *args, **kwargs):
            track_fit(self, X, y, *args, **kwargs)
            return orig_fit(self, X, y, *args, **kwargs)
        return wrapped

    monkeypatch.setattr(LogisticRegression, "fit", tracking_wrapper(orig_lr_fit))
    monkeypatch.setattr(DecisionTreeClassifier, "fit", tracking_wrapper(orig_dt_fit))
    monkeypatch.setattr(RandomForestClassifier, "fit", tracking_wrapper(orig_rf_fit))
    monkeypatch.setattr(XGBoostWaterQualityClassifier, "fit", tracking_wrapper(orig_xgb_fit))

    train_baseline_models(
        X_train=X_train,
        y_train=y_train,
        random_seed=42,
        save_models=False,
    )

    # Exactly 4 models fitted
    assert len(seen_inputs) == 4, f"Expected 4 fit invocations, got {len(seen_inputs)}"

    base_shape, base_len, base_X, base_y = seen_inputs[0]
    for i, (shape, y_len, X_arr, y_arr) in enumerate(seen_inputs[1:], start=2):
        assert shape == base_shape, f"Model {i} input shape {shape} != base shape {base_shape}"
        assert y_len == base_len, f"Model {i} target length {y_len} != base length {base_len}"
        np.testing.assert_array_equal(X_arr, base_X, err_msg=f"Model {i} X input differs from Model 1")
        np.testing.assert_array_equal(y_arr, base_y, err_msg=f"Model {i} y input differs from Model 1")


def test_tc_5_1_03_prediction_sanity_check(split_data):
    """TC-5.1-03: Prediction sanity check on 5 rows of X_test for all 4 models."""
    X_train = split_data.X_train_processed
    y_train = split_data.y_train
    X_test_5 = split_data.X_test_processed[:5]

    models = train_baseline_models(
        X_train=X_train,
        y_train=y_train,
        random_seed=42,
        save_models=False,
        X_test_sanity=X_test_5,
    )

    expected_classes = set(y_train.unique())

    for name, model in models.items():
        preds = model.predict(X_test_5)

        # 1. Output shape must be (5,)
        assert preds.shape == (5,), f"{name}: expected prediction shape (5,), got {preds.shape}"

        # 2. Output labels must be valid class strings from the target distribution
        for label in preds:
            assert label in expected_classes, (
                f"{name}: predicted invalid class '{label}'. Expected one of {expected_classes}"
            )

        # 3. Predict proba shape must be (5, 5) with normalized row sums
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_5)
            assert probs.shape == (5, len(expected_classes)), (
                f"{name}: expected proba shape (5, 5), got {probs.shape}"
            )
            np.testing.assert_allclose(
                probs.sum(axis=1),
                np.ones(5),
                rtol=1e-5,
                err_msg=f"{name}: predicted probabilities do not sum to 1.0",
            )


def test_tc_5_1_04_reproducibility(split_data):
    """TC-5.1-04: Re-running training twice with identical random_seed yields identical predictions."""
    X_train = split_data.X_train_processed
    y_train = split_data.y_train
    X_test_5 = split_data.X_test_processed[:5]

    run1 = train_baseline_models(
        X_train=X_train,
        y_train=y_train,
        random_seed=42,
        save_models=False,
    )

    run2 = train_baseline_models(
        X_train=X_train,
        y_train=y_train,
        random_seed=42,
        save_models=False,
    )

    for name in BASELINE_MODEL_NAMES:
        m1 = run1[name]
        m2 = run2[name]

        preds1 = m1.predict(X_test_5)
        preds2 = m2.predict(X_test_5)
        np.testing.assert_array_equal(
            preds1,
            preds2,
            err_msg=f"{name}: predictions differ between two deterministic runs with random_seed=42",
        )

        probs1 = m1.predict_proba(X_test_5)
        probs2 = m2.predict_proba(X_test_5)
        np.testing.assert_allclose(
            probs1,
            probs2,
            rtol=1e-5,
            err_msg=f"{name}: predicted probabilities differ between two deterministic runs with random_seed=42",
        )
