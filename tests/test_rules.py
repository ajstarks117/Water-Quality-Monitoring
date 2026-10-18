"""Test Suite for Milestone 8.2: Decision-Support Rule Engine Implementation.

Owner: Explainability & Decision Support Lead (Person C)
Branch: feature/explainability-decision-support

Test Cases:
- TC-8.2-01: No abnormal parameters produces no triggered rules (empty list).
- TC-8.2-02: Single abnormal parameter triggers exactly one rule matching that parameter.
- TC-8.2-03: Multiple abnormal parameters trigger multiple rules and elevate priority to High.
- TC-8.2-04: Boundary values handled strictly per documented decision (at-threshold = NORMAL).
- TC-8.2-05: Thresholds are config-driven, not hard-coded (editing config alters rule triggers).
- Additional tests:
  - Exact Spec Section 33 schema verification (all 6 required keys present with correct types).
  - Alias / shorthand parameter name resolution.
  - Overall priority calculation and summary dictionary structure.
"""

import copy
from pathlib import Path
import pytest
import yaml

from src.decision_support.rules import (
    DEFAULT_CONFIG_PATH,
    evaluate_parameters,
    get_overall_priority,
    load_thresholds,
    summarize_decision_support,
)


@pytest.fixture(scope="module")
def default_config():
    """Load default thresholds configuration."""
    return load_thresholds()


@pytest.fixture
def pristine_sample():
    """Water quality sample where all parameters are strictly within normal regulatory limits."""
    return {
        "Potential of Hydrogen (pH)": 7.4,
        "Dissolved oxygen (mg/L)": 8.0,
        "Biochemical Oxygen Demand (mg/L)": 1.5,
        "Fecal Coliform (MPN/100mL)": 50.0,
        "Turbidity (NTU)": 2.0,
        "Electric Conductivity (μS/cm)": 350.0,
        "Total Dissolved Solids (mg/L)": 220.0,
    }


# ===========================================================================
# TC-8.2-01: No Abnormal Parameters Produces No Triggered Rules
# ===========================================================================
def test_tc_8_2_01_all_normal_returns_empty_list(pristine_sample, default_config):
    """TC-8.2-01: When all parameters are within normal limits, no rules are triggered."""
    triggered = evaluate_parameters(pristine_sample, thresholds_config=default_config)
    assert isinstance(triggered, list)
    assert len(triggered) == 0, f"Expected 0 triggered rules, got {len(triggered)}: {triggered}"

    summary = summarize_decision_support(pristine_sample, thresholds_config=default_config)
    assert summary["abnormal_parameter_count"] == 0
    assert summary["overall_priority"] == "None"


# ===========================================================================
# TC-8.2-02: Single Abnormal Parameter Triggers Exactly One Rule
# ===========================================================================
def test_tc_8_2_02_single_abnormal_bod(pristine_sample, default_config):
    """TC-8.2-02: Setting BOD above 3.0 mg/L triggers exactly one BOD rule."""
    sample = copy.deepcopy(pristine_sample)
    sample["Biochemical Oxygen Demand (mg/L)"] = 4.5  # Standard is <= 3.0

    triggered = evaluate_parameters(sample, thresholds_config=default_config)
    assert len(triggered) == 1, f"Expected exactly 1 rule, got {len(triggered)}"

    rule = triggered[0]
    assert "Biochemical Oxygen Demand" in rule["parameter"]
    assert rule["priority"] == "Medium"  # Single standard violation without poor class
    assert "4.5" in rule["condition"] or "4.50" in rule["condition"]
    assert len(rule["possible_factors"]) > 0
    assert len(rule["recommendations"]) > 0


def test_tc_8_2_02_single_abnormal_ph_acidic(pristine_sample, default_config):
    """TC-8.2-02: Setting pH below 6.5 triggers exactly one acidic pH rule."""
    sample = copy.deepcopy(pristine_sample)
    sample["Potential of Hydrogen (pH)"] = 5.8  # Standard is 6.5 - 8.5

    triggered = evaluate_parameters(sample, thresholds_config=default_config)
    assert len(triggered) == 1
    rule = triggered[0]
    assert "pH" in rule["parameter"]
    assert "Acidic" in rule["condition"] or "5.8" in rule["condition"]
    assert rule["priority"] == "Medium"


def test_tc_8_2_02_single_abnormal_do_depleted(pristine_sample, default_config):
    """TC-8.2-02: Setting DO below 5.0 mg/L triggers exactly one DO rule."""
    sample = copy.deepcopy(pristine_sample)
    sample["Dissolved oxygen (mg/L)"] = 4.2  # Standard is >= 5.0

    triggered = evaluate_parameters(sample, thresholds_config=default_config)
    assert len(triggered) == 1
    rule = triggered[0]
    assert "Dissolved oxygen" in rule["parameter"]
    assert "Oxygen depletion" in rule["condition"] or "4.2" in rule["condition"]
    assert rule["priority"] == "Medium"


# ===========================================================================
# TC-8.2-03: Multiple Abnormal Parameters Trigger Multiple Rules and High Priority
# ===========================================================================
def test_tc_8_2_03_multiple_abnormal_triggers_multiple_rules_and_high_priority(pristine_sample, default_config):
    """TC-8.2-03: Multiple abnormal parameters trigger respective rules and elevate priority to High."""
    sample = copy.deepcopy(pristine_sample)
    sample["Dissolved oxygen (mg/L)"] = 3.5  # Abnormal (depleted)
    sample["Biochemical Oxygen Demand (mg/L)"] = 5.8  # Abnormal (high organic load)
    sample["Fecal Coliform (MPN/100mL)"] = 1500.0  # Abnormal (high fecal coliform)

    triggered = evaluate_parameters(sample, thresholds_config=default_config)
    assert len(triggered) == 3, f"Expected 3 triggered rules, got {len(triggered)}"

    # Priority logic check: when multiple abnormal parameters exist, priority elevates to High
    for rule in triggered:
        assert rule["priority"] == "High", f"Rule for {rule['parameter']} did not elevate to High"

    overall_priority = get_overall_priority(triggered)
    assert overall_priority == "High"


def test_tc_8_2_03_critical_single_violation_triggers_high_priority(pristine_sample, default_config):
    """TC-8.2-03: Severe single parameter violation (e.g. FC > 2500 MPN/100mL) triggers High priority."""
    sample = copy.deepcopy(pristine_sample)
    sample["Fecal Coliform (MPN/100mL)"] = 3000.0  # Critical pathogen alert

    triggered = evaluate_parameters(sample, thresholds_config=default_config)
    assert len(triggered) == 1
    assert triggered[0]["priority"] == "High"
    assert get_overall_priority(triggered) == "High"


def test_tc_8_2_03_poor_class_prediction_elevates_single_violation(pristine_sample, default_config):
    """TC-8.2-03: Single moderate violation elevates to High priority if model predicts Poor class."""
    sample = copy.deepcopy(pristine_sample)
    sample["Biochemical Oxygen Demand (mg/L)"] = 4.2  # Moderate standard violation

    # Without poor class -> Medium
    triggered_normal = evaluate_parameters(sample, thresholds_config=default_config, predicted_class="Good")
    assert triggered_normal[0]["priority"] == "Medium"

    # With poor class -> High
    triggered_poor = evaluate_parameters(sample, thresholds_config=default_config, predicted_class="Poor")
    assert triggered_poor[0]["priority"] == "High"
    assert get_overall_priority(triggered_poor, predicted_class="Poor") == "High"


# ===========================================================================
# TC-8.2-04: Boundary Value Handled Per Documented Decision (At-Threshold = Normal)
# ===========================================================================
def test_tc_8_2_04_exact_boundary_values_are_normal(default_config):
    """TC-8.2-04: Verify that readings set EXACTLY at statutory thresholds do not trigger violations.

    Per Docs/threshold_standards.md:
    - Upper limits (S_i): value <= threshold is NORMAL (e.g., BOD=3.0, FC=250.0, Turbidity=5.0).
    - Lower limits: value >= threshold is NORMAL (DO=5.0).
    - Range limits: min <= value <= max is NORMAL (pH=6.5, pH=8.5).
    """
    boundary_sample = {
        "Potential of Hydrogen (pH)": 6.5,  # Exactly at lower acceptable bound
        "Dissolved oxygen (mg/L)": 5.0,  # Exactly at lower acceptable bound
        "Biochemical Oxygen Demand (mg/L)": 3.0,  # Exactly at upper permissible limit
        "Fecal Coliform (MPN/100mL)": 250.0,  # Exactly at upper permissible limit
        "Turbidity (NTU)": 5.0,  # Exactly at upper permissible limit
        "Electric Conductivity (μS/cm)": 750.0,  # Exactly at upper permissible limit
        "Total Dissolved Solids (mg/L)": 500.0,  # Exactly at upper permissible limit
        "Chemical Oxygen Demand (mg/L)": 10.0,  # Exactly at upper permissible limit
        "Total Coliform (MPN/100mL)": 500.0,  # Exactly at upper permissible limit
        "Nitrate N (mgN/L)": 10.0,  # Exactly at upper permissible limit
    }

    triggered = evaluate_parameters(boundary_sample, thresholds_config=default_config)
    assert len(triggered) == 0, (
        f"Boundary values at exact thresholds triggered unexpected rules: "
        f"{[r['parameter'] for r in triggered]}"
    )

    # Test other boundary bound for pH (8.5)
    boundary_ph_high = {"Potential of Hydrogen (pH)": 8.5}
    triggered_ph = evaluate_parameters(boundary_ph_high, thresholds_config=default_config)
    assert len(triggered_ph) == 0, f"pH=8.5 triggered unexpected rule: {triggered_ph}"


def test_tc_8_2_04_marginal_delta_triggers_rule(default_config):
    """TC-8.2-04: A value just marginally past the boundary does trigger the rule."""
    # Just above BOD threshold (3.01 > 3.00)
    triggered_bod = evaluate_parameters({"BOD": 3.01}, thresholds_config=default_config)
    assert len(triggered_bod) == 1

    # Just below DO threshold (4.99 < 5.00)
    triggered_do = evaluate_parameters({"DO": 4.99}, thresholds_config=default_config)
    assert len(triggered_do) == 1

    # Just below pH lower bound (6.49 < 6.50)
    triggered_ph_low = evaluate_parameters({"pH": 6.49}, thresholds_config=default_config)
    assert len(triggered_ph_low) == 1

    # Just above pH upper bound (8.51 > 8.50)
    triggered_ph_high = evaluate_parameters({"pH": 8.51}, thresholds_config=default_config)
    assert len(triggered_ph_high) == 1


# ===========================================================================
# TC-8.2-05: Thresholds are Config-Driven, Not Hard-Coded
# ===========================================================================
def test_tc_8_2_05_custom_config_alters_rule_behavior(pristine_sample, default_config):
    """TC-8.2-05: Modifying configuration values changes rule triggering without altering python code."""
    # Deep copy config and modify BOD threshold from 3.0 to 1.0
    custom_cfg = copy.deepcopy(default_config)
    custom_cfg["parameters"]["BOD"]["max_acceptable"] = 1.0

    # With pristine sample (BOD = 1.5):
    # Under default config (3.0): 1.5 is normal -> 0 rules
    default_res = evaluate_parameters(pristine_sample, thresholds_config=default_config)
    assert len(default_res) == 0

    # Under custom config (1.0): 1.5 is abnormal -> 1 rule triggered
    custom_res = evaluate_parameters(pristine_sample, thresholds_config=custom_cfg)
    assert len(custom_res) == 1
    assert "Biochemical Oxygen Demand" in custom_res[0]["parameter"]
    assert "1.0" in custom_res[0]["condition"] or "1.00" in custom_res[0]["condition"]


# ===========================================================================
# Spec Section 33 Schema & Data Structure Conformance
# ===========================================================================
def test_spec_section_33_exact_schema_keys(default_config):
    """Verify that all triggered rules contain the exact 6 keys specified in Section 33."""
    sample = {
        "pH": 9.2,
        "DO": 3.0,
        "BOD": 8.0,
        "FC": 5000.0,
    }
    triggered = evaluate_parameters(sample, thresholds_config=default_config)
    assert len(triggered) == 4

    required_keys = {
        "parameter",
        "condition",
        "interpretation",
        "possible_factors",
        "recommendations",
        "priority",
    }

    for rule in triggered:
        # All 6 required keys must be present
        assert required_keys.issubset(rule.keys()), (
            f"Rule missing required Spec Section 33 keys: {required_keys - set(rule.keys())}"
        )
        assert isinstance(rule["parameter"], str) and len(rule["parameter"]) > 0
        assert isinstance(rule["condition"], str) and len(rule["condition"]) > 0
        assert isinstance(rule["interpretation"], str) and len(rule["interpretation"]) > 0
        assert isinstance(rule["possible_factors"], list) and len(rule["possible_factors"]) > 0
        assert isinstance(rule["recommendations"], list) and len(rule["recommendations"]) > 0
        assert rule["priority"] in ["High", "Medium", "Low"]


# ===========================================================================
# Alias / Shorthand Parameter Resolution
# ===========================================================================
def test_shorthand_parameter_resolution(default_config):
    """Verify that short codes ('pH', 'DO', 'BOD', 'FC', 'turbidity') resolve equivalently to full names."""
    short_sample = {"pH": 9.0, "do": 3.5, "bod": 4.0, "fc": 300.0}
    full_sample = {
        "Potential of Hydrogen (pH)": 9.0,
        "Dissolved oxygen (mg/L)": 3.5,
        "Biochemical Oxygen Demand (mg/L)": 4.0,
        "Fecal Coliform (MPN/100mL)": 300.0,
    }

    rules_short = evaluate_parameters(short_sample, thresholds_config=default_config)
    rules_full = evaluate_parameters(full_sample, thresholds_config=default_config)

    assert len(rules_short) == 4
    assert len(rules_full) == 4

    short_params = {r["parameter"] for r in rules_short}
    full_params = {r["parameter"] for r in rules_full}
    assert short_params == full_params


# ===========================================================================
# Decision-Support Summary & Causation Caveat
# ===========================================================================
def test_summarize_decision_support_structure(pristine_sample, default_config):
    """Verify structure and causation caveat of summarize_decision_support."""
    summary = summarize_decision_support(pristine_sample, thresholds_config=default_config)
    assert "predicted_class" in summary
    assert "parameters_evaluated" in summary
    assert "abnormal_parameter_count" in summary
    assert "overall_priority" in summary
    assert "triggered_rules" in summary
    assert "causation_caveat" in summary
    assert "not legal proof" in summary["causation_caveat"]
