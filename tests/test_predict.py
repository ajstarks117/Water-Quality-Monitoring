"""
Unit & Integration Tests for Milestone 6.3: Final Model Selection & Artifact Freeze
=====================================================================================
Test Cases:
- TC-6.3-01: Artifact loads and predicts on raw input
- TC-6.3-02: predict_proba works and sums to 1
- TC-6.3-03: Selection criterion applied, not retrofitted
- TC-6.3-04: Negative: malformed input handling
"""

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data.preprocessing import FROZEN_CANDIDATE_FEATURES
from src.models.predict import (
    DEFAULT_MODEL_PATH,
    FROZEN_RAW_FEATURE_COLUMNS,
    TARGET_CLASSES,
    compute_artifact_hash,
    load_model,
    predict,
    predict_proba,
    validate_raw_features,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_PATH = _REPO_ROOT / "models" / "best_model.pkl"
_CRITERION_DOC = _REPO_ROOT / "Docs" / "model_selection_criterion.md"
_CONTRACT_DOC = _REPO_ROOT / "Docs" / "interface_contracts.md"
_COMPARISON_CSV = _REPO_ROOT / "reports" / "model_comparison.csv"


@pytest.fixture(scope="module")
def sample_raw_dataframe() -> pd.DataFrame:
    """Fixture providing raw, unprocessed water quality observations."""
    return pd.DataFrame(
        [
            {
                "Potential of Hydrogen (pH)": 7.42,
                "Dissolved oxygen (mg/L)": 6.8,
                "Biochemical Oxygen Demand (mg/L)": 1.9,
                "Fecal Coliform (MPN/100mL)": 120.0,
            },
            {
                "Potential of Hydrogen (pH)": 8.1,
                "Dissolved oxygen (mg/L)": 3.2,
                "Biochemical Oxygen Demand (mg/L)": 8.5,
                "Fecal Coliform (MPN/100mL)": 4500.0,
            },
            {
                "Potential of Hydrogen (pH)": 6.5,
                "Dissolved oxygen (mg/L)": 1.1,
                "Biochemical Oxygen Demand (mg/L)": 28.0,
                "Fecal Coliform (MPN/100mL)": 95000.0,
            },
        ]
    )


@pytest.fixture(scope="module")
def loaded_artifact() -> Pipeline:
    """Load the frozen artifact fresh."""
    assert _MODEL_PATH.exists(), f"Artifact must exist at {_MODEL_PATH}"
    artifact = joblib.load(_MODEL_PATH)
    assert isinstance(artifact, Pipeline), "Frozen artifact must be an sklearn Pipeline"
    return artifact


# ===========================================================================
# TC-6.3-01: Artifact loads and predicts on raw input
# ===========================================================================
def test_tc_6_3_01_artifact_loads_and_predicts_raw_input(sample_raw_dataframe):
    """
    TC-6.3-01: Artifact loads fresh in a new process; call .predict() with
    a DataFrame of raw, unprocessed sample values.
    Expected: Returns valid class predictions with no manual preprocessing
    step required by the caller.
    """
    assert _MODEL_PATH.exists(), f"Frozen artifact {_MODEL_PATH} not found"

    # Fresh load
    artifact = joblib.load(_MODEL_PATH)
    assert hasattr(artifact, "predict"), "Artifact must expose .predict()"

    # Predict on raw 4-column input
    preds = artifact.predict(sample_raw_dataframe)

    assert isinstance(preds, np.ndarray), "Predictions must be a numpy ndarray"
    assert len(preds) == len(sample_raw_dataframe), "Prediction length must match input"

    # All predicted classes must belong to canonical classes
    for pred in preds:
        assert pred in TARGET_CLASSES, f"Unexpected class label '{pred}' not in {TARGET_CLASSES}"

    # Also test via src.models.predict wrapper
    wrapper_preds = predict(sample_raw_dataframe, model=artifact)
    np.testing.assert_array_equal(preds, wrapper_preds)


def test_tc_6_3_01_pipeline_structure(loaded_artifact):
    """
    Verify pipeline structure is a single combined pipeline containing:
    1. feature_engineering
    2. preprocessor
    3. classifier (DecisionTreeClassifier)
    """
    step_names = [name for name, _ in loaded_artifact.steps]
    assert "feature_engineering" in step_names, "Pipeline must contain 'feature_engineering' step"
    assert "preprocessor" in step_names, "Pipeline must contain 'preprocessor' step"
    assert "classifier" in step_names, "Pipeline must contain 'classifier' step"

    classifier = loaded_artifact.named_steps["classifier"]
    assert isinstance(classifier, DecisionTreeClassifier), (
        f"Classifier must be DecisionTreeClassifier, got {type(classifier)}"
    )


# ===========================================================================
# TC-6.3-02: predict_proba works and sums to 1
# ===========================================================================
def test_tc_6_3_02_predict_proba_sums_to_one(loaded_artifact, sample_raw_dataframe):
    """
    TC-6.3-02: Call .predict_proba() on a raw sample.
    Expected: Returns a probability array per class summing to 1.0
    within floating point tolerance, all values between 0.0 and 1.0.
    """
    assert hasattr(loaded_artifact, "predict_proba"), "Artifact must expose .predict_proba()"

    probas = loaded_artifact.predict_proba(sample_raw_dataframe)

    assert isinstance(probas, np.ndarray), "Probabilities must be a numpy ndarray"
    assert probas.shape == (len(sample_raw_dataframe), len(TARGET_CLASSES)), (
        f"Probability shape {probas.shape} must match (n_samples, {len(TARGET_CLASSES)})"
    )

    # Probabilities in range [0, 1]
    assert np.all(probas >= 0.0) and np.all(probas <= 1.0), (
        "All probability values must be bounded in [0.0, 1.0]"
    )

    # Sum of probabilities across classes equals 1.0
    sums = probas.sum(axis=1)
    np.testing.assert_allclose(
        sums,
        np.ones(len(sample_raw_dataframe)),
        rtol=1e-5,
        atol=1e-5,
        err_msg="Class probabilities must sum to 1.0 per sample",
    )

    # Also test via src.models.predict wrapper
    wrapper_probas = predict_proba(sample_raw_dataframe, model=loaded_artifact)
    np.testing.assert_allclose(probas, wrapper_probas)


# ===========================================================================
# TC-6.3-03: Selection criterion applied, not retrofitted
# ===========================================================================
def test_tc_6_3_03_selection_criterion_consistency():
    """
    TC-6.3-03: Compare final selection reasoning against pre-declared criterion.
    Expected: Final selection is consistent with pre-declared primary/secondary
    criteria; any deviation is explicitly justified.
    """
    # 1. Comparison table exists and has results
    assert _COMPARISON_CSV.exists(), f"Comparison table {_COMPARISON_CSV} not found"
    df_comp = pd.read_csv(_COMPARION_PATH if False else _COMPARISON_CSV)
    assert not df_comp.empty, "Comparison table must not be empty"

    # Find highest Macro F1
    assert "Macro F1" in df_comp.columns, "Comparison table must contain 'Macro F1'"
    df_sorted = df_comp.sort_values(by="Macro F1", ascending=False)
    best_model_name = df_sorted.iloc[0]["Model"]
    best_macro_f1 = df_sorted.iloc[0]["Macro F1"]

    # Winner in table must be Decision Tree
    assert "Decision Tree" in best_model_name, (
        f"Expected Decision Tree to have highest Macro F1, got {best_model_name}"
    )
    assert best_macro_f1 > 0.85, f"Best Macro F1 must be high, got {best_macro_f1}"

    # 2. Frozen artifact must encapsulate the winning model type
    artifact = joblib.load(_MODEL_PATH)
    classifier = artifact.named_steps["classifier"]
    assert isinstance(classifier, DecisionTreeClassifier), (
        "Frozen model classifier must match the winning Decision Tree"
    )

    # 3. model_selection_criterion.md must document the selection explicitly
    assert _CRITERION_DOC.exists(), f"Criterion document {_CRITERION_DOC} not found"
    criterion_text = _CRITERION_DOC.read_text(encoding="utf-8")

    assert "Milestone 6.3: Final Model Selection & Freeze" in criterion_text, (
        "Criterion doc must contain Section 9 with M6.3 record"
    )
    assert "Random Forest" in criterion_text, "Criterion doc must discuss runner-up Random Forest"
    assert "Macro F1" in criterion_text, "Criterion doc must reference primary metric Macro F1"


def test_tc_6_3_03_contract_records_frozen_artifact():
    """
    Verify Docs/interface_contracts.md records the frozen artifact contract,
    version, and exact SHA-256 hash.
    """
    assert _CONTRACT_DOC.exists(), f"Interface contracts {_CONTRACT_DOC} not found"
    contract_text = _CONTRACT_DOC.read_text(encoding="utf-8")

    actual_hash = compute_artifact_hash(_MODEL_PATH)
    assert actual_hash in contract_text, (
        f"Artifact SHA-256 hash {actual_hash} must be recorded in Docs/interface_contracts.md"
    )
    assert "FROZEN & CONTRACT LOCKED" in contract_text
    assert "models/best_model.pkl" in contract_text


# ===========================================================================
# TC-6.3-04: Negative: malformed input
# ===========================================================================
def test_tc_6_3_04_missing_required_column_raises_error(loaded_artifact):
    """
    TC-6.3-04: Call .predict() with a DataFrame missing a required column.
    Expected: Raises a clear, specific error rather than a cryptic internal
    sklearn exception.
    """
    # Missing 'Fecal Coliform (MPN/100mL)'
    bad_df_1 = pd.DataFrame(
        [
            {
                "Potential of Hydrogen (pH)": 7.5,
                "Dissolved oxygen (mg/L)": 6.0,
                "Biochemical Oxygen Demand (mg/L)": 2.5,
            }
        ]
    )

    with pytest.raises(ValueError) as excinfo:
        loaded_artifact.predict(bad_df_1)

    err_msg = str(excinfo.value)
    assert "Fecal Coliform" in err_msg or "Missing" in err_msg, (
        f"Error message must be clear and mention the missing feature, got: {err_msg}"
    )

    # Missing 'Potential of Hydrogen (pH)'
    bad_df_2 = pd.DataFrame(
        [
            {
                "Dissolved oxygen (mg/L)": 6.0,
                "Biochemical Oxygen Demand (mg/L)": 2.5,
                "Fecal Coliform (MPN/100mL)": 150.0,
            }
        ]
    )

    with pytest.raises(ValueError) as excinfo2:
        loaded_artifact.predict(bad_df_2)

    err_msg2 = str(excinfo2.value)
    assert "Potential of Hydrogen (pH)" in err_msg2 or "Missing" in err_msg2, (
        f"Error message must be clear, got: {err_msg2}"
    )


def test_tc_6_3_04_wrapper_input_validation():
    """
    Test input validation in src.models.predict wrapper:
    - Empty DataFrame raises ValueError
    - None input raises ValueError
    - Missing column raises ValueError with specific column name
    """
    with pytest.raises(ValueError, match="Input data cannot be None"):
        validate_raw_features(None)

    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        validate_raw_features(pd.DataFrame())

    malformed = pd.DataFrame([{"pH": 7.0}])
    with pytest.raises(ValueError, match="Missing required raw feature column"):
        validate_raw_features(malformed)


def test_tc_6_3_04_single_dict_prediction():
    """
    Test predict() and predict_proba() handle single dict inputs gracefully.
    """
    sample_dict = {
        "Potential of Hydrogen (pH)": 7.2,
        "Dissolved oxygen (mg/L)": 5.5,
        "Biochemical Oxygen Demand (mg/L)": 3.0,
        "Fecal Coliform (MPN/100mL)": 200.0,
    }

    pred = predict(sample_dict)
    assert len(pred) == 1
    assert pred[0] in TARGET_CLASSES

    proba = predict_proba(sample_dict)
    assert proba.shape == (1, len(TARGET_CLASSES))
    np.testing.assert_allclose(proba.sum(), 1.0, rtol=1e-5)
