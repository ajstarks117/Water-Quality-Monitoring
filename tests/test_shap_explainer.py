"""
Unit & Integration Tests for Milestone 7.1: SHAP Global Explanation Module
===========================================================================
Test Cases:
- TC-7.1-01: SHAP values computed without error on small samples and full sets
- TC-7.1-02: Feature names match frozen contract exactly
- TC-7.1-03: Global importance is directionally sane and cross-checked with M3.2 EDA
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.features.engineering import FINAL_EXTENDED_FEATURE_COLUMNS
from src.explainability.shap_explainer import (
    CAUSATION_CAVEAT_STATEMENT,
    compare_shap_with_eda,
    explain_samples,
    generate_shap_local_plot,
    generate_shap_summary_plot,
    get_global_importance,
    get_local_explanation,
    get_tree_classifier,
    get_tree_explainer,
    load_frozen_pipeline,
    prepare_features_for_explainer,
)
from src.models.predict import FROZEN_RAW_FEATURE_COLUMNS

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = _REPO_ROOT / "Docs" / "interface_contracts.md"
_SUMMARY_PLOT_PATH = _REPO_ROOT / "reports" / "figures" / "shap_summary.png"
_LOCAL_PLOT_PATH = _REPO_ROOT / "reports" / "figures" / "shap_local_example.png"


@pytest.fixture(scope="module")
def frozen_model() -> Pipeline:
    """Fixture providing the frozen pipeline artifact."""
    model = load_frozen_pipeline()
    assert isinstance(model, Pipeline), "Frozen model must be an sklearn Pipeline"
    return model


@pytest.fixture(scope="module")
def raw_test_samples() -> pd.DataFrame:
    """Fixture providing a small DataFrame of raw candidate features."""
    return pd.DataFrame(
        [
            {
                "Potential of Hydrogen (pH)": 7.5,
                "Dissolved oxygen (mg/L)": 6.8,
                "Biochemical Oxygen Demand (mg/L)": 2.1,
                "Fecal Coliform (MPN/100mL)": 150.0,
            },
            {
                "Potential of Hydrogen (pH)": 8.2,
                "Dissolved oxygen (mg/L)": 3.4,
                "Biochemical Oxygen Demand (mg/L)": 9.5,
                "Fecal Coliform (MPN/100mL)": 4800.0,
            },
            {
                "Potential of Hydrogen (pH)": 6.9,
                "Dissolved oxygen (mg/L)": 1.5,
                "Biochemical Oxygen Demand (mg/L)": 25.0,
                "Fecal Coliform (MPN/100mL)": 75000.0,
            },
            {
                "Potential of Hydrogen (pH)": 7.1,
                "Dissolved oxygen (mg/L)": 8.9,
                "Biochemical Oxygen Demand (mg/L)": 1.2,
                "Fecal Coliform (MPN/100mL)": 25.0,
            },
            {
                "Potential of Hydrogen (pH)": 7.8,
                "Dissolved oxygen (mg/L)": 5.2,
                "Biochemical Oxygen Demand (mg/L)": 4.5,
                "Fecal Coliform (MPN/100mL)": 850.0,
            },
        ]
    )


# ===========================================================================
# TC-7.1-01: SHAP values computed without error
# ===========================================================================
def test_tc_7_1_01_tree_explainer_initialization(frozen_model):
    """Verify that TreeExplainer initializes successfully on the frozen model."""
    classifier = get_tree_classifier(frozen_model)
    assert isinstance(classifier, DecisionTreeClassifier), (
        f"Underlying model must be DecisionTreeClassifier, got {type(classifier)}"
    )

    explainer = get_tree_explainer(frozen_model)
    assert isinstance(explainer, shap.TreeExplainer), (
        f"Explainer must be shap.TreeExplainer, got {type(explainer)}"
    )


def test_tc_7_1_01_compute_global_importance_on_small_sample(frozen_model, raw_test_samples):
    """
    TC-7.1-01: Run get_global_importance on a small sample of raw features.
    Expected: Returns a ranked feature importance DataFrame with no exceptions.
    """
    df_imp = get_global_importance(model=frozen_model, X_sample=raw_test_samples)

    assert isinstance(df_imp, pd.DataFrame), "Output must be a pandas DataFrame"
    assert not df_imp.empty, "Importance DataFrame must not be empty"

    # Verify required columns for dashboard consumption
    required_cols = ["feature", "importance", "normalized_importance", "rank"]
    for col in required_cols:
        assert col in df_imp.columns, f"Missing required column '{col}' in importance table"

    # All 5 features present
    assert len(df_imp) == 5, f"Expected 5 features, got {len(df_imp)}"

    # Ranks must be 1 through 5 sequentially
    assert df_imp["rank"].tolist() == [1, 2, 3, 4, 5], "Ranks must be [1, 2, 3, 4, 5]"

    # Importance values must be non-negative and monotonically decreasing
    importances = df_imp["importance"].values
    assert np.all(importances >= 0.0), "Importance values must be >= 0"
    assert (np.diff(importances) <= 1e-9).all(), "Table must be sorted descending by importance"

    # Normalized importance must sum to 1.0
    np.testing.assert_allclose(df_imp["normalized_importance"].sum(), 1.0, rtol=1e-5)


def test_tc_7_1_01_default_arguments_execution():
    """Verify get_global_importance runs cleanly with default arguments (auto-loading)."""
    df_imp = get_global_importance(sample_size=20, random_state=42)
    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 5
    assert df_imp["rank"].iloc[0] == 1


def test_tc_7_1_01_single_sample_dict(frozen_model):
    """Verify get_global_importance handles a single dictionary input."""
    single_dict = {
        "Potential of Hydrogen (pH)": 7.4,
        "Dissolved oxygen (mg/L)": 6.5,
        "Biochemical Oxygen Demand (mg/L)": 2.0,
        "Fecal Coliform (MPN/100mL)": 100.0,
    }
    df_imp = get_global_importance(model=frozen_model, X_sample=single_dict)
    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 5


# ===========================================================================
# TC-7.1-02: Feature names match model features exactly
# ===========================================================================
def test_tc_7_1_02_feature_names_match_frozen_contract(frozen_model, raw_test_samples):
    """
    TC-7.1-02: Compare SHAP feature names against the frozen feature list in Docs/interface_contracts.md.
    Expected: Exact match, no mismatch, missing, or renamed columns.
    """
    df_imp = get_global_importance(model=frozen_model, X_sample=raw_test_samples)
    shap_features = set(df_imp["feature"].tolist())
    expected_features = set(FINAL_EXTENDED_FEATURE_COLUMNS)

    assert shap_features == expected_features, (
        f"Feature mismatch between SHAP and frozen contract:\n"
        f"SHAP features: {sorted(shap_features)}\n"
        f"Expected features: {sorted(expected_features)}"
    )

    # Check Docs/interface_contracts.md confirms these features
    assert _CONTRACT_PATH.exists(), f"Contract file {_CONTRACT_PATH} missing"
    contract_text = _CONTRACT_PATH.read_text(encoding="utf-8")
    for feat in expected_features:
        assert feat in contract_text, f"Feature '{feat}' not documented in Docs/interface_contracts.md"


def test_tc_7_1_02_feature_engineering_consistency(frozen_model, raw_test_samples):
    """
    Verify that prepare_features_for_explainer correctly generates Log10 Fecal Coliform
    and outputs exactly the 5 extended features in correct order.
    """
    preprocessed_df, _ = prepare_features_for_explainer(frozen_model, raw_test_samples)
    assert list(preprocessed_df.columns) == FINAL_EXTENDED_FEATURE_COLUMNS, (
        f"Column order mismatch: {list(preprocessed_df.columns)} vs {FINAL_EXTENDED_FEATURE_COLUMNS}"
    )
    assert preprocessed_df.shape == (len(raw_test_samples), 5)


# ===========================================================================
# TC-7.1-03: Global importance is directionally sane
# ===========================================================================
def test_tc_7_1_03_directionally_sane_importance(frozen_model):
    """
    TC-7.1-03: Compare SHAP top-3 features against the visually separating parameters from EDA.
    Expected: Reasonable overlap; any divergence investigated and explained.
    """
    df_imp = get_global_importance(model=frozen_model, sample_size=100, random_state=42)
    top_3 = df_imp.head(3)["feature"].tolist()

    # BOD must be #1 feature based on dominant class separation in EDA
    assert df_imp.iloc[0]["feature"] == "Biochemical Oxygen Demand (mg/L)", (
        f"Expected BOD to be top feature, got {df_imp.iloc[0]['feature']}"
    )

    # Dissolved Oxygen must be in top 3
    assert "Dissolved oxygen (mg/L)" in top_3, (
        f"Expected Dissolved oxygen in top 3, got {top_3}"
    )

    # Log10 Fecal Coliform must have higher importance than raw Fecal Coliform
    fc_raw_imp = df_imp.loc[df_imp["feature"] == "Fecal Coliform (MPN/100mL)", "importance"].values[0]
    fc_log_imp = df_imp.loc[df_imp["feature"] == "Log10 Fecal Coliform", "importance"].values[0]
    assert fc_log_imp >= fc_raw_imp, (
        f"Engineered Log10 FC ({fc_log_imp:.4f}) should have >= importance than raw FC ({fc_raw_imp:.4f})"
    )


def test_tc_7_1_03_structured_eda_comparison_documented():
    """Verify compare_shap_with_eda returns structured agreement/divergence records."""
    comp = compare_shap_with_eda()
    assert "findings" in comp
    assert len(comp["findings"]) >= 4

    statuses = [f["status"] for f in comp["findings"]]
    assert "STRONG AGREEMENT" in statuses
    assert "DOMAIN DIVERGENCE EXPLAINED" in statuses
    assert "FEATURE ENGINEERING VALIDATED" in statuses


def test_tc_7_1_03_summary_plot_generation(tmp_path):
    """Verify that generate_shap_summary_plot creates a non-empty, readable image file."""
    test_plot = tmp_path / "test_shap_summary.png"
    out_path = generate_shap_summary_plot(output_path=test_plot, sample_size=20, random_state=42)

    assert out_path.exists(), f"Plot was not created at {out_path}"
    assert out_path.stat().st_size > 5000, f"Plot file is too small ({out_path.stat().st_size} bytes)"

    # Also verify repo default plot exists
    assert _SUMMARY_PLOT_PATH.exists(), f"Repository plot {_SUMMARY_PLOT_PATH} missing"
    assert _SUMMARY_PLOT_PATH.stat().st_size > 5000


# ===========================================================================
# Milestone 7.2: Local Explanation Test Suite
# ===========================================================================

# ===========================================================================
# TC-7.2-01: Local explanation computed for a single sample
# ===========================================================================
def test_tc_7_2_01_local_explanation_structure_and_additive_property(frozen_model, raw_test_samples):
    """
    TC-7.2-01: Call get_local_explanation on one raw sample.
    Expected: Returns per-feature contribution values that sum consistently
    with the prediction (within SHAP's additive property tolerance).
    """
    single_sample = raw_test_samples.iloc[[0]]
    explanation = get_local_explanation(model=frozen_model, single_input_row=single_sample)

    assert isinstance(explanation, dict), "Local explanation must be a dictionary"

    # Verify essential fields
    required_keys = [
        "predicted_class",
        "predicted_probability",
        "base_value",
        "feature_contributions",
        "contributions_sum",
        "additive_check_passed",
        "class_probabilities",
        "per_class_contributions",
        "causation_caveat",
    ]
    for key in required_keys:
        assert key in explanation, f"Missing required key '{key}' in local explanation"

    # Additive property check
    assert explanation["additive_check_passed"] is True, (
        f"Additive property check failed: base_value ({explanation['base_value']}) + "
        f"contributions_sum ({explanation['contributions_sum']}) != "
        f"predicted_probability ({explanation['predicted_probability']})"
    )
    expected_prob = explanation["base_value"] + explanation["contributions_sum"]
    np.testing.assert_allclose(
        expected_prob,
        explanation["predicted_probability"],
        rtol=1e-5,
        atol=1e-5,
        err_msg="Base value + sum of contributions must equal predicted probability",
    )

    # Verify feature contributions list
    contributions = explanation["feature_contributions"]
    assert len(contributions) == 5, f"Expected 5 feature contributions, got {len(contributions)}"

    # Check rank order
    ranks = [c["rank"] for c in contributions]
    assert ranks == [1, 2, 3, 4, 5], "Contributions must be ranked 1 through 5"

    # Check each contribution item
    for item in contributions:
        assert "feature" in item
        assert "feature_value" in item
        assert "contribution" in item
        assert "absolute_contribution" in item
        assert "direction" in item
        assert item["direction"] in ["increases_probability", "decreases_probability", "neutral"]
        assert item["absolute_contribution"] >= 0.0

    # Test with single dict input
    dict_sample = {
        "Potential of Hydrogen (pH)": 7.4,
        "Dissolved oxygen (mg/L)": 6.5,
        "Biochemical Oxygen Demand (mg/L)": 2.0,
        "Fecal Coliform (MPN/100mL)": 100.0,
    }
    exp_dict = get_local_explanation(model=frozen_model, single_input_row=dict_sample)
    assert exp_dict["additive_check_passed"] is True
    assert len(exp_dict["feature_contributions"]) == 5


# ===========================================================================
# TC-7.2-02: Performance target met
# ===========================================================================
def test_tc_7_2_02_performance_target_met(frozen_model, raw_test_samples):
    """
    TC-7.2-02: Time 20 consecutive calls to get_local_explanation.
    Expected: Average latency is well under the agreed 1-2 second budget per call.
    """
    import time

    single_sample = raw_test_samples.iloc[[0]]
    durations = []

    # Warm-up call
    _ = get_local_explanation(model=frozen_model, single_input_row=single_sample)

    for _ in range(20):
        t0 = time.perf_counter()
        _ = get_local_explanation(model=frozen_model, single_input_row=single_sample)
        durations.append(time.perf_counter() - t0)

    avg_time = sum(durations) / len(durations)
    max_time = max(durations)

    # Dashboard SLA target: under 1.0 second
    assert avg_time < 1.0, f"Average explanation time {avg_time:.4f}s exceeds 1.0s budget"
    assert max_time < 2.0, f"Max explanation time {max_time:.4f}s exceeds 2.0s budget"


# ===========================================================================
# TC-7.2-03: Negative: malformed single input
# ===========================================================================
def test_tc_7_2_03_negative_missing_features(frozen_model):
    """
    TC-7.2-03: Call with a row missing required features.
    Expected: Raises a clear, specific ValueError rather than a silent wrong answer.
    """
    # Missing 3 features
    bad_sample_1 = {"Potential of Hydrogen (pH)": 7.5}
    with pytest.raises(ValueError) as excinfo:
        get_local_explanation(model=frozen_model, single_input_row=bad_sample_1)
    err_str = str(excinfo.value)
    assert "Missing required" in err_str or "Dissolved oxygen" in err_str

    # Missing BOD
    bad_sample_2 = pd.DataFrame(
        [
            {
                "Potential of Hydrogen (pH)": 7.5,
                "Dissolved oxygen (mg/L)": 6.0,
                "Fecal Coliform (MPN/100mL)": 150.0,
            }
        ]
    )
    with pytest.raises(ValueError) as excinfo2:
        get_local_explanation(model=frozen_model, single_input_row=bad_sample_2)
    assert "Biochemical Oxygen Demand (mg/L)" in str(excinfo2.value) or "Missing" in str(excinfo2.value)

    # Empty input
    with pytest.raises(ValueError):
        get_local_explanation(model=frozen_model, single_input_row=pd.DataFrame())

    # None input
    with pytest.raises(ValueError):
        get_local_explanation(model=frozen_model, single_input_row=None)

    # Multiple rows passed
    multi_sample = pd.DataFrame([bad_sample_1, bad_sample_1])
    with pytest.raises(ValueError) as excinfo_multi:
        get_local_explanation(model=frozen_model, single_input_row=multi_sample)
    assert "exactly 1" in str(excinfo_multi.value)


# ===========================================================================
# TC-7.2-04: Output language check
# ===========================================================================
def test_tc_7_2_04_causation_safe_language(frozen_model, raw_test_samples):
    """
    TC-7.2-04: Review field names/labels in the returned structure.
    Expected: No field or label claims causation; language is limited to 'contribution' / 'attribution'.
    Enforces Specification Section 31 & 48 policy.
    """
    single_sample = raw_test_samples.iloc[[0]]
    explanation = get_local_explanation(model=frozen_model, single_input_row=single_sample)

    forbidden_terms = ["cause", "causal", "reason", "responsible"]

    # Check top-level keys
    for key in explanation.keys():
        if key == "causation_caveat":
            continue
        for term in forbidden_terms:
            assert term not in key.lower(), f"Forbidden causal term '{term}' found in key: '{key}'"

    # Check feature contribution item keys and direction values
    for item in explanation["feature_contributions"]:
        for k, v in item.items():
            for term in forbidden_terms:
                assert term not in k.lower(), f"Forbidden causal term '{term}' found in item key: '{k}'"
                if isinstance(v, str):
                    assert term not in v.lower(), f"Forbidden causal term '{term}' found in value: '{v}'"

    # Check causation caveat exists and explicitly states SHAP does not prove causation
    caveat = explanation["causation_caveat"]
    assert "does not prove" in caveat.lower() or "not prove" in caveat.lower()
    assert "causation" in caveat.lower()
    assert "attribution" in caveat.lower() or "contribution" in caveat.lower()


def test_tc_7_2_local_plot_generation(tmp_path, frozen_model, raw_test_samples):
    """Verify generate_shap_local_plot generates a valid local explanation image."""
    single_sample = raw_test_samples.iloc[[0]]
    temp_plot = tmp_path / "temp_local_plot.png"
    out_path = generate_shap_local_plot(
        model=frozen_model,
        single_input_row=single_sample,
        output_path=temp_plot,
        show=False,
    )

    assert out_path.exists(), f"Local plot missing at {out_path}"
    assert out_path.stat().st_size > 5000, "Local plot file is too small"

    # Verify repository reference plot exists
    assert _LOCAL_PLOT_PATH.exists(), f"Repository local plot {_LOCAL_PLOT_PATH} missing"
    assert _LOCAL_PLOT_PATH.stat().st_size > 5000

