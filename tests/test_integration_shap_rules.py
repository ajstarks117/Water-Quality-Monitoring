"""Integration Tests for Milestone 8.3: SHAP-to-Rules Integration.

Owner: Explainability & Decision Support Lead (Person C)
Branch: feature/explainability-decision-support

Test Suite:
- TC-8.3-01: End-to-end chain produces all 4 labeled sections (WHAT, WHY, WHAT DOES IT MEAN, WHAT CAN BE DONE).
- TC-8.3-02: SHAP-important-but-normal-range case handled explicitly (in shap_important_normal_parameters).
- TC-8.3-03: Abnormal-but-not-top-SHAP case handled explicitly (in additional_abnormal_parameters).
- TC-8.3-04: No causal-claim language in output (anti-causation phrasing guardrail).
- TC-8.3-05: Priority is present and strictly consistent with Milestone 8.2 logic.
"""

from pathlib import Path
import re
import pytest

from src.models.predict import load_model
from src.explainability.shap_explainer import get_local_explanation
from src.decision_support.rules import (
    evaluate_parameters,
    generate_decision_support,
    get_overall_priority,
    load_thresholds,
)


@pytest.fixture(scope="module")
def frozen_model():
    """Load the frozen scikit-learn pipeline artifact."""
    return load_model()


@pytest.fixture(scope="module")
def thresholds_cfg():
    """Load default thresholds configuration."""
    return load_thresholds()


@pytest.fixture
def poor_water_sample():
    """Sample exhibiting clear contamination (Poor/Unsuitable classification)."""
    return {
        "Potential of Hydrogen (pH)": 7.2,
        "Dissolved oxygen (mg/L)": 3.2,
        "Biochemical Oxygen Demand (mg/L)": 6.8,
        "Fecal Coliform (MPN/100mL)": 2200.0,
        "Turbidity (NTU)": 15.0,  # Abnormal secondary parameter
    }


@pytest.fixture
def clean_water_sample():
    """Pristine sample where all parameters are within normal ranges."""
    return {
        "Potential of Hydrogen (pH)": 7.4,
        "Dissolved oxygen (mg/L)": 8.2,
        "Biochemical Oxygen Demand (mg/L)": 1.4,
        "Fecal Coliform (MPN/100mL)": 40.0,
        "Turbidity (NTU)": 2.5,
    }


# ===========================================================================
# TC-8.3-01: End-to-End Chain Produces All 4 Labeled Sections
# ===========================================================================
def test_tc_8_3_01_all_four_sections_present(frozen_model, poor_water_sample, thresholds_cfg):
    """TC-8.3-01: End-to-end chain generates the four distinct labeled sections."""
    # 1. SHAP local explanation
    shap_output = get_local_explanation(frozen_model, poor_water_sample)
    assert shap_output["predicted_class"] in ["Poor", "Very Poor", "Unsuitable"]

    # 2. Integration call
    ds_output = generate_decision_support(
        prediction=shap_output["predicted_class"],
        local_shap_output=shap_output,
        parameter_values=poor_water_sample,
        top_n=3,
        thresholds_config=thresholds_cfg,
    )

    # Verify presence of the four core sections
    assert "prediction" in ds_output, "Missing 'prediction' section (WHAT)"
    assert "model_level_evidence" in ds_output, "Missing 'model_level_evidence' section (WHY)"
    assert "parameter_interpretation" in ds_output, "Missing 'parameter_interpretation' section (WHAT DOES IT MEAN)"
    assert "recommended_actions" in ds_output, "Missing 'recommended_actions' section (WHAT CAN BE DONE)"

    # Verify stage labels
    assert "Level 0" in ds_output["prediction"]["stage"]
    assert "Level 1" in ds_output["model_level_evidence"]["stage"]
    assert "Level 2" in ds_output["parameter_interpretation"]["stage"]
    assert "Level 3" in ds_output["recommended_actions"]["stage"]

    # Verify section contents
    assert ds_output["prediction"]["predicted_class"] == shap_output["predicted_class"]
    assert len(ds_output["model_level_evidence"]["top_contributing_features"]) == 3
    assert len(ds_output["recommended_actions"]["action_items"]) > 0


# ===========================================================================
# TC-8.3-02: SHAP-Important-But-Normal-Range Case Handled
# ===========================================================================
def test_tc_8_3_02_shap_important_normal_parameter_surfaced(frozen_model, thresholds_cfg):
    """TC-8.3-02: When a top SHAP feature is within normal statutory range, it is explicitly cataloged."""
    # Craft a synthetic SHAP output where pH is top-ranked contributor, but pH=7.4 is within normal range (6.5 - 8.5)
    synthetic_shap = {
        "predicted_class": "Good",
        "predicted_probability": 0.88,
        "base_value": 0.20,
        "feature_contributions": [
            {
                "feature": "Potential of Hydrogen (pH)",
                "feature_value": 7.4,
                "contribution": 0.35,
                "absolute_contribution": 0.35,
                "direction": "increases_probability",
                "rank": 1,
            },
            {
                "feature": "Biochemical Oxygen Demand (mg/L)",
                "feature_value": 2.0,
                "contribution": 0.25,
                "absolute_contribution": 0.25,
                "direction": "increases_probability",
                "rank": 2,
            },
            {
                "feature": "Dissolved oxygen (mg/L)",
                "feature_value": 7.5,
                "contribution": 0.08,
                "absolute_contribution": 0.08,
                "direction": "increases_probability",
                "rank": 3,
            },
        ],
        "contributions_sum": 0.68,
        "additive_check_passed": True,
    }

    sample_values = {
        "Potential of Hydrogen (pH)": 7.4,
        "Biochemical Oxygen Demand (mg/L)": 2.0,
        "Dissolved oxygen (mg/L)": 7.5,
    }

    ds_output = generate_decision_support(
        prediction="Good",
        local_shap_output=synthetic_shap,
        parameter_values=sample_values,
        top_n=3,
        thresholds_config=thresholds_cfg,
    )

    interp = ds_output["parameter_interpretation"]
    shap_normal = interp["shap_important_normal_parameters"]

    # All three parameters are in normal ranges, so all 3 top SHAP features should be in shap_normal
    assert len(shap_normal) == 3, f"Expected 3 normal SHAP features, got {len(shap_normal)}"
    normal_features = [item["feature"] for item in shap_normal]
    assert "Potential of Hydrogen (pH)" in normal_features

    # Verify documented explanation note is present rather than silent dropping
    ph_entry = next(item for item in shap_normal if item["feature"] == "Potential of Hydrogen (pH)")
    assert "Within standard regulatory limits" in ph_entry["status"]
    assert "statistical attribution" in ph_entry["interpretation"]
    assert "statutory limits" in ph_entry["interpretation"]


# ===========================================================================
# TC-8.3-03: Abnormal-But-Not-Top-SHAP Case Handled
# ===========================================================================
def test_tc_8_3_03_abnormal_not_in_top_shap_surfaced(frozen_model, poor_water_sample, thresholds_cfg):
    """TC-8.3-03: Parameters abnormal per rule engine but not in top-N SHAP contributors are surfaced as secondary alerts."""
    # In poor_water_sample, Turbidity (NTU) = 15.0 is abnormal (> 5.0 NTU).
    # However, the ML model only accepts the 4 candidate features, so Turbidity is NOT in SHAP features.
    shap_output = get_local_explanation(frozen_model, poor_water_sample)

    ds_output = generate_decision_support(
        prediction=shap_output["predicted_class"],
        local_shap_output=shap_output,
        parameter_values=poor_water_sample,
        top_n=2,  # deliberately restrict to top 2 SHAP features
        thresholds_config=thresholds_cfg,
    )

    interp = ds_output["parameter_interpretation"]
    additional_abnormal = interp["additional_abnormal_parameters"]

    assert len(additional_abnormal) > 0, "Expected secondary abnormal parameters to be surfaced"
    additional_params = [item["parameter"] for item in additional_abnormal]
    assert "Turbidity (NTU)" in additional_params

    # Verify explicit explanatory note is attached
    turbidity_entry = next(item for item in additional_abnormal if item["parameter"] == "Turbidity (NTU)")
    assert "exceeds statutory limits" in turbidity_entry["note"]
    assert "was not among the top" in turbidity_entry["note"]


# ===========================================================================
# TC-8.3-04: No Causal-Claim Language in Output (Spec Section 31 & 48)
# ===========================================================================
def test_tc_8_3_04_no_causal_claim_language(frozen_model, poor_water_sample, thresholds_cfg):
    """TC-8.3-04: Scan all text fields across output dictionary to verify zero causal claim language."""
    shap_output = get_local_explanation(frozen_model, poor_water_sample)
    ds_output = generate_decision_support(
        prediction=shap_output["predicted_class"],
        local_shap_output=shap_output,
        parameter_values=poor_water_sample,
        top_n=4,
        thresholds_config=thresholds_cfg,
    )

    # Collect all string values recursively
    all_strings = []

    def _extract_strings(obj):
        if isinstance(obj, str):
            all_strings.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _extract_strings(v)
        elif isinstance(obj, list):
            for item in obj:
                _extract_strings(item)

    _extract_strings(ds_output)

    # Forbidden causal phrases (Spec Section 31/48)
    forbidden_patterns = [
        r"\bcaused\s+by\b",
        r"\bthe\s+model\s+found\s+.*\s+caused\b",
        r"\bcauses\s+pollution\b",
        r"\bproven\s+source\b",
        r"\bproven\s+cause\b",
        r"\bguilty\b",
        r"\bculprit\b",
    ]

    for text in all_strings:
        lower_text = text.lower()
        for pattern in forbidden_patterns:
            assert not re.search(pattern, lower_text), (
                f"Forbidden causal language detected matching '{pattern}' in text: '{text}'"
            )

    # Verify causation caveat is explicitly embedded
    assert "causation_caveat" in ds_output
    assert "not constitute physical" in ds_output["causation_caveat"] or "do not prove" in ds_output["causation_caveat"]


# ===========================================================================
# TC-8.3-05: Priority is Present and Consistent with M8.2 Logic
# ===========================================================================
def test_tc_8_3_05_priority_consistency(frozen_model, poor_water_sample, thresholds_cfg):
    """TC-8.3-05: Priority in integrated decision support matches standalone rule engine priority."""
    shap_output = get_local_explanation(frozen_model, poor_water_sample)
    ds_output = generate_decision_support(
        prediction=shap_output["predicted_class"],
        local_shap_output=shap_output,
        parameter_values=poor_water_sample,
        top_n=3,
        thresholds_config=thresholds_cfg,
    )

    # Evaluate standalone priority
    standalone_rules = evaluate_parameters(
        poor_water_sample,
        thresholds_config=thresholds_cfg,
        predicted_class=shap_output["predicted_class"],
    )
    standalone_priority = get_overall_priority(
        standalone_rules,
        predicted_class=shap_output["predicted_class"],
    )

    # Must match exactly
    assert ds_output["overall_priority"] == standalone_priority
    assert ds_output["recommended_actions"]["overall_priority"] == standalone_priority
    assert ds_output["overall_priority"] == "High"  # Multiple severe violations
