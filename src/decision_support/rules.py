"""Decision-Support Rule Engine.

This module implements the parameter -> condition -> interpretation -> recommendation -> priority
rule engine as specified in Specification Section 33, Section 34, and Section 35.

Key Design Principles:
1. Exact Spec Section 33 Schema:
   Each rule dictionary contains:
     - parameter: str
     - condition: str
     - interpretation: str
     - possible_factors: list[str]
     - recommendations: list[str]
     - priority: str ("High", "Medium", "Low")
   Additional diagnostic keys (e.g., observed_value, threshold_value, unit) are
   provided for downstream dashboard integration.

2. Fully Config-Driven (No Hard-Coded Numbers):
   Thresholds, alert limits, and rule templates are loaded directly from `config/thresholds.yaml`
   (derived from `Docs/threshold_standards.md` / `Docs/wqi_standard.md`). Any threshold change
   can be made via configuration without touching Python code.

3. Explicit Boundary Condition Semantics (Spec Section 55 & M8.1):
   - Upper permissible limit (S_i): value <= threshold is NORMAL; value > threshold is ABNORMAL.
   - Lower required limit (DO): value >= threshold is NORMAL; value < threshold is ABNORMAL.
   - Range-bounded (pH): min_acceptable <= value <= max_acceptable is NORMAL.
   Exact values at statutory thresholds are defined as NORMAL / PERMISSIBLE.

4. Explicit Priority Classification Architecture (Spec Section 35):
   - High Priority: Critical biological/chemical violations (FC > 2500 MPN/100mL, DO < 4.0 mg/L,
     BOD > 6.0 mg/L, extreme pH < 4.0 or > 9.5), OR multiple (>= 2) abnormal parameters,
     OR poor-quality model prediction ("Poor", "Very Poor", "Unsuitable") with parameter violations.
   - Medium Priority: Exactly one standard violation (e.g., 3.0 < BOD <= 6.0, 250 < FC <= 2500,
     4.0 <= DO < 5.0, pH outside 6.5-8.5).
   - Low Priority: Minor secondary deviations (e.g., Turbidity 5-10 NTU, TDS 500-1000 mg/L).

5. Standalone Architecture:
   Runs independently of any trained machine learning model or artifact.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import yaml

logger = logging.getLogger("decision_support.rules")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "thresholds.yaml"

# Parameter alias lookup mapping alternative / raw column names to canonical config keys
CANONICAL_KEY_LOOKUP: Dict[str, str] = {
    # pH
    "ph": "pH",
    "potential of hydrogen (ph)": "pH",
    "potential of hydrogen": "pH",
    # Dissolved Oxygen
    "do": "DO",
    "dissolved oxygen": "DO",
    "dissolved oxygen (mg/l)": "DO",
    "dissolved_oxygen": "DO",
    # BOD
    "bod": "BOD",
    "biochemical oxygen demand": "BOD",
    "biochemical oxygen demand (mg/l)": "BOD",
    # Fecal Coliform
    "fc": "FC",
    "fecal coliform": "FC",
    "fecal coliform (mpn/100ml)": "FC",
    "faecal coliform": "FC",
    "faecal coliform (mpn/100ml)": "FC",
    "fecal_coliform": "FC",
    # Log10 Fecal Coliform
    "log10_fc": "Log10_FC",
    "log10 fecal coliform": "Log10_FC",
    "log10_fecal_coliform": "Log10_FC",
    # Turbidity
    "turbidity": "Turbidity",
    "turbidity (ntu)": "Turbidity",
    # Electrical Conductivity
    "ec": "EC",
    "conductivity": "EC",
    "electric conductivity (μs/cm)": "EC",
    "electric conductivity (us/cm)": "EC",
    "electrical conductivity": "EC",
    # Total Dissolved Solids
    "tds": "TDS",
    "solids": "TDS",
    "total dissolved solids": "TDS",
    "total dissolved solids (mg/l)": "TDS",
    # COD
    "cod": "COD",
    "chemical oxygen demand": "COD",
    "chemical oxygen demand (mg/l)": "COD",
    # Total Coliform
    "tc": "TC",
    "total coliform": "TC",
    "total coliform (mpn/100ml)": "TC",
    # Nitrate
    "nitrate": "Nitrate",
    "nitrate n": "Nitrate",
    "nitrate n (mgn/l)": "Nitrate",
    "nitrate (mg/l)": "Nitrate",
    # Ammonia
    "ammonia": "Ammonia",
    "amonia n (mgn/l)": "Ammonia",
    "ammonia n": "Ammonia",
    # Temperature
    "temperature": "Temperature",
    "temp": "Temperature",
    "temperature (ºc)": "Temperature",
    "temperature (°c)": "Temperature",
}


def load_thresholds(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load threshold limits, alert boundaries, and rule templates from a YAML configuration file.

    Args:
        config_path: Optional path to the YAML thresholds file. Defaults to `config/thresholds.yaml`.

    Returns:
        Dictionary containing configuration metadata and parameter rules.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Thresholds configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config or "parameters" not in config:
        raise ValueError(f"Invalid thresholds config at {path}: missing 'parameters' section")

    return config


def _normalize_key(key: str) -> str:
    """Normalize input parameter string to its canonical config key."""
    cleaned = str(key).strip().lower()
    return CANONICAL_KEY_LOOKUP.get(cleaned, key)


def evaluate_parameters(
    parameter_values: Dict[str, Any],
    thresholds_config: Optional[Dict[str, Any]] = None,
    predicted_class: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Evaluate a dictionary of water quality parameters against regulatory thresholds.

    Generates triggered decision-support rules strictly conforming to Specification Section 33.

    Args:
        parameter_values: Dictionary mapping parameter names (raw or short) to numeric readings.
        thresholds_config: Optional loaded thresholds configuration dictionary. If None,
                           loads dynamically from `config/thresholds.yaml`.
        predicted_class: Optional model prediction label (e.g., 'Good', 'Poor', 'Unsuitable')
                         used for compounding priority classification per Spec Section 35.

    Returns:
        List of triggered rule dictionaries. Each dictionary contains exactly the 6 required keys:
            - "parameter": Human-readable parameter name
            - "condition": Description of the triggering condition
            - "interpretation": Ecological/chemical interpretation
            - "possible_factors": List of potential contributing factors
            - "recommendations": List of actionable recommendations
            - "priority": "High" | "Medium" | "Low"
        Along with optional diagnostic metadata (`observed_value`, `threshold_value`, `unit`).
    """
    if thresholds_config is None:
        thresholds_config = load_thresholds()

    params_cfg = thresholds_config.get("parameters", {})
    triggered_rules: List[Dict[str, Any]] = []

    # Map inputs to canonical keys
    normalized_inputs: Dict[str, float] = {}
    for raw_key, raw_val in parameter_values.items():
        if raw_val is None:
            continue
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            continue
        canonical_key = _normalize_key(raw_key)
        normalized_inputs[canonical_key] = val

    # Evaluate each parameter present in inputs
    for canonical_key, val in normalized_inputs.items():
        if canonical_key not in params_cfg:
            continue

        p_info = params_cfg[canonical_key]
        p_name = p_info.get("column_name", canonical_key)
        unit = p_info.get("unit", "")
        rules_cfg = p_info.get("rules", {})

        # -------------------------------------------------------------------
        # Range-Bounded: pH (6.5 - 8.5)
        # -------------------------------------------------------------------
        if canonical_key == "pH":
            min_acc = float(p_info.get("min_acceptable", 6.5))
            max_acc = float(p_info.get("max_acceptable", 8.5))
            crit_low = float(p_info.get("critical_low", 4.0))
            crit_high = float(p_info.get("critical_high", 9.5))

            # Boundary: exactly at 6.5 or 8.5 is NORMAL
            if val < min_acc:
                r_tmpl = rules_cfg.get("low", {})
                is_crit = val < crit_low
                priority = r_tmpl.get("critical_priority", "High") if is_crit else r_tmpl.get("base_priority", "Medium")
                triggered_rules.append({
                    "parameter": p_name,
                    "condition": f"pH = {val:.2f} < {min_acc:.1f} (Acidic water condition)",
                    "interpretation": r_tmpl.get("interpretation", "Acidic water condition below permissible range."),
                    "possible_factors": list(r_tmpl.get("possible_factors", [])),
                    "recommendations": list(r_tmpl.get("recommendations", [])),
                    "priority": priority,
                    "observed_value": val,
                    "threshold_value": min_acc,
                    "unit": unit,
                })
            elif val > max_acc:
                r_tmpl = rules_cfg.get("high", {})
                is_crit = val > crit_high
                priority = r_tmpl.get("critical_priority", "High") if is_crit else r_tmpl.get("base_priority", "Medium")
                triggered_rules.append({
                    "parameter": p_name,
                    "condition": f"pH = {val:.2f} > {max_acc:.1f} (Alkaline water condition)",
                    "interpretation": r_tmpl.get("interpretation", "Alkaline water condition exceeding permissible range."),
                    "possible_factors": list(r_tmpl.get("possible_factors", [])),
                    "recommendations": list(r_tmpl.get("recommendations", [])),
                    "priority": priority,
                    "observed_value": val,
                    "threshold_value": max_acc,
                    "unit": unit,
                })

        # -------------------------------------------------------------------
        # Lower-Limit: Dissolved Oxygen (>= 5.0 mg/L)
        # -------------------------------------------------------------------
        elif canonical_key == "DO":
            min_acc = float(p_info.get("min_acceptable", 5.0))
            crit_low = float(p_info.get("critical_low", 4.0))
            supersat = float(p_info.get("supersaturation", 14.6))

            # Boundary: exactly 5.0 is NORMAL
            if val < min_acc:
                r_tmpl = rules_cfg.get("low", {})
                is_crit = val < crit_low
                priority = r_tmpl.get("critical_priority", "High") if is_crit else r_tmpl.get("base_priority", "Medium")
                triggered_rules.append({
                    "parameter": p_name,
                    "condition": f"Dissolved oxygen = {val:.2f} mg/L < {min_acc:.1f} mg/L (Oxygen depletion)",
                    "interpretation": r_tmpl.get("interpretation", "Dissolved oxygen depletion below regulatory minimum."),
                    "possible_factors": list(r_tmpl.get("possible_factors", [])),
                    "recommendations": list(r_tmpl.get("recommendations", [])),
                    "priority": priority,
                    "observed_value": val,
                    "threshold_value": min_acc,
                    "unit": unit,
                })
            elif val > supersat:
                r_tmpl = rules_cfg.get("supersaturated", {})
                triggered_rules.append({
                    "parameter": p_name,
                    "condition": f"Dissolved oxygen = {val:.2f} mg/L > {supersat:.1f} mg/L (Photosynthetic supersaturation)",
                    "interpretation": r_tmpl.get("interpretation", "Dissolved oxygen exceeds cold-water physical saturation."),
                    "possible_factors": list(r_tmpl.get("possible_factors", [])),
                    "recommendations": list(r_tmpl.get("recommendations", [])),
                    "priority": r_tmpl.get("base_priority", "Low"),
                    "observed_value": val,
                    "threshold_value": supersat,
                    "unit": unit,
                })

        # -------------------------------------------------------------------
        # Range-Bounded: Temperature (15.0 - 32.0 ºC)
        # -------------------------------------------------------------------
        elif canonical_key == "Temperature":
            min_acc = float(p_info.get("min_acceptable", 15.0))
            max_acc = float(p_info.get("max_acceptable", 32.0))
            crit_high = float(p_info.get("critical_high", 40.0))

            if val > max_acc:
                r_tmpl = rules_cfg.get("high", {})
                is_crit = val > crit_high
                priority = r_tmpl.get("critical_priority", "Medium") if is_crit else r_tmpl.get("base_priority", "Low")
                triggered_rules.append({
                    "parameter": p_name,
                    "condition": f"Water temperature = {val:.1f} ºC > {max_acc:.1f} ºC (Elevated thermal regime)",
                    "interpretation": r_tmpl.get("interpretation", "Water temperature exceeds normal seasonal baseline."),
                    "possible_factors": list(r_tmpl.get("possible_factors", [])),
                    "recommendations": list(r_tmpl.get("recommendations", [])),
                    "priority": priority,
                    "observed_value": val,
                    "threshold_value": max_acc,
                    "unit": unit,
                })

        # -------------------------------------------------------------------
        # Standard Upper Permissible Limits (BOD, FC, Log10_FC, Turbidity, EC, TDS, COD, TC, Nitrate, Ammonia)
        # -------------------------------------------------------------------
        else:
            max_acc = float(p_info.get("max_acceptable", float("inf")))
            crit_high = float(p_info.get("critical_high", max_acc * 2.0))

            # Boundary: exactly at max_acceptable is NORMAL
            if val > max_acc:
                r_tmpl = rules_cfg.get("high", {})
                is_crit = val > crit_high
                priority = r_tmpl.get("critical_priority", "High") if is_crit else r_tmpl.get("base_priority", "Medium")
                triggered_rules.append({
                    "parameter": p_name,
                    "condition": f"{p_name} = {val:.2f} {unit} > {max_acc:.2f} {unit}",
                    "interpretation": r_tmpl.get("interpretation", f"{p_name} exceeds regulatory permissible limit."),
                    "possible_factors": list(r_tmpl.get("possible_factors", [])),
                    "recommendations": list(r_tmpl.get("recommendations", [])),
                    "priority": priority,
                    "observed_value": val,
                    "threshold_value": max_acc,
                    "unit": unit,
                })

    # -----------------------------------------------------------------------
    # Dynamic Priority Elevation Logic (Spec Section 35)
    # -----------------------------------------------------------------------
    # High Priority triggers if:
    # 1. Multiple abnormal parameters (>= 2 rules triggered)
    # 2. OR poor-quality class ("Poor", "Very Poor", "Unsuitable") with any violation
    # 3. OR any individual rule is already classified as High
    is_poor_class = predicted_class in ["Poor", "Very Poor", "Unsuitable"] if predicted_class else False
    multiple_abnormal = len(triggered_rules) >= 2

    for rule in triggered_rules:
        if multiple_abnormal:
            rule["priority"] = "High"
        elif is_poor_class and rule["priority"] == "Medium":
            rule["priority"] = "High"

    return triggered_rules


def get_overall_priority(
    triggered_rules: List[Dict[str, Any]],
    predicted_class: Optional[str] = None,
) -> str:
    """Compute the overall assessment priority level from a set of triggered rules.

    Args:
        triggered_rules: List of triggered rule dictionaries from `evaluate_parameters`.
        predicted_class: Optional predicted class label from the ML model.

    Returns:
        "High" | "Medium" | "Low" | "None"
    """
    if not triggered_rules:
        return "None"

    priorities = {r.get("priority", "Low") for r in triggered_rules}
    if "High" in priorities:
        return "High"

    if len(triggered_rules) >= 2:
        return "High"

    if predicted_class in ["Poor", "Very Poor", "Unsuitable"]:
        return "High"

    if "Medium" in priorities:
        return "Medium"

    return "Low"


def _keys_match(rule_param_name: str, feat_name: str) -> bool:
    """Check if a rule parameter name and a SHAP feature name refer to the same physical parameter."""
    if rule_param_name == feat_name:
        return True
    k1 = _normalize_key(rule_param_name)
    k2 = _normalize_key(feat_name)
    if k1 == k2:
        return True
    # Fecal coliform raw vs log-engineered
    if k1 in ("FC", "Log10_FC") and k2 in ("FC", "Log10_FC"):
        return True
    return False


def generate_decision_support(
    prediction: Optional[Union[str, Dict[str, Any]]] = None,
    local_shap_output: Optional[Dict[str, Any]] = None,
    parameter_values: Optional[Union[Dict[str, Any], Any]] = None,
    top_n: int = 3,
    thresholds_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Wire model-level SHAP explanation to rule-based domain interpretation and recommendations.

    Implements the 4-level chain of evidence mandated by Specification Sections 32 & 48:
        Level 0: Water Quality Prediction (WHAT)
        Level 1: Model-Level Evidence / SHAP Attribution (WHY)
        Level 2: Parameter Interpretation / Domain Rules (WHAT DOES IT MEAN)
        Level 3: Recommended Actions (WHAT CAN BE DONE)

    Edge Case Handling:
        - SHAP-important but normal-range: When a top-N SHAP feature is within statutory limits,
          it is explicitly cataloged under `shap_important_normal_parameters` with an explanatory
          note rather than silently dropped.
        - Abnormal but not in top-N SHAP: When a parameter exceeds statutory limits but did not
          dominate the model's top mathematical split path, it is surfaced under
          `additional_abnormal_parameters` as a secondary diagnostic alert.

    Args:
        prediction: Predicted class string (e.g. "Poor") or dict containing prediction metadata.
        local_shap_output: Dictionary returned by `get_local_explanation(model, input_row)`.
        parameter_values: Dictionary (or DataFrame/Series) of observed water quality parameters.
        top_n: Number of top SHAP contributors to cross-reference (defaults to 3).
        thresholds_config: Optional custom thresholds configuration dictionary.

    Returns:
        Structured dictionary containing four distinct labeled sections:
            - "prediction": Level 0 prediction metadata
            - "model_level_evidence": Level 1 SHAP attributions and directional contributions
            - "parameter_interpretation": Level 2 domain rule interpretations (abnormal & normal)
            - "recommended_actions": Level 3 prioritized actionable recommendations
        Plus top-level convenience accessors (`overall_priority`, `causation_caveat`).
    """
    # -----------------------------------------------------------------------
    # 1. Resolve Prediction (Level 0: WHAT)
    # -----------------------------------------------------------------------
    pred_class: Optional[str] = None
    pred_prob: Optional[float] = None
    class_probs: Optional[Dict[str, float]] = None

    if isinstance(prediction, str):
        pred_class = prediction
    elif isinstance(prediction, dict):
        pred_class = prediction.get("predicted_class") or prediction.get("prediction")
        pred_prob = prediction.get("predicted_probability")
        class_probs = prediction.get("class_probabilities")

    if pred_class is None and local_shap_output:
        pred_class = local_shap_output.get("predicted_class")
    if pred_prob is None and local_shap_output:
        pred_prob = local_shap_output.get("predicted_probability")
    if class_probs is None and local_shap_output:
        class_probs = local_shap_output.get("class_probabilities")

    resolved_pred_class = pred_class if pred_class else "Unknown"

    section_prediction = {
        "stage": "Level 0: Water Quality Prediction (WHAT)",
        "predicted_class": resolved_pred_class,
        "predicted_probability": pred_prob,
        "class_probabilities": class_probs,
        "summary": f"Predicted Water Quality: {resolved_pred_class}",
    }

    # -----------------------------------------------------------------------
    # 2. Resolve Model-Level Evidence from SHAP (Level 1: WHY)
    # -----------------------------------------------------------------------
    all_contributions: List[Dict[str, Any]] = []
    base_val: Optional[float] = None
    additive_passed: Optional[bool] = None

    if local_shap_output:
        all_contributions = list(local_shap_output.get("feature_contributions", []))
        base_val = local_shap_output.get("base_value")
        additive_passed = local_shap_output.get("additive_check_passed")

    top_n_features = all_contributions[:top_n] if top_n > 0 else all_contributions

    section_model_evidence = {
        "stage": "Level 1: Model-Level Evidence (WHY)",
        "top_contributing_features": top_n_features,
        "all_feature_contributions": all_contributions,
        "base_value": base_val,
        "additive_check_passed": additive_passed,
        "causation_caveat": (
            "SHAP values reflect statistical attribution toward model output and do not "
            "prove physical, chemical, or biological causation (Specification Section 31/48)."
        ),
    }

    # -----------------------------------------------------------------------
    # 3. Resolve Parameter Values & Run Rule Engine Evaluation
    # -----------------------------------------------------------------------
    param_dict: Dict[str, Any] = {}
    if parameter_values is not None:
        if hasattr(parameter_values, "to_dict"):
            if hasattr(parameter_values, "iloc"):
                param_dict = parameter_values.iloc[0].to_dict()
            else:
                param_dict = parameter_values.to_dict()
        elif isinstance(parameter_values, dict):
            param_dict = dict(parameter_values)
    elif local_shap_output and all_contributions:
        # Fallback: recover parameter values from SHAP contribution records
        param_dict = {
            item["feature"]: item["feature_value"]
            for item in all_contributions
            if "feature" in item and "feature_value" in item
        }

    all_triggered_rules = evaluate_parameters(
        parameter_values=param_dict,
        thresholds_config=thresholds_config,
        predicted_class=resolved_pred_class,
    )

    # -----------------------------------------------------------------------
    # 4. Integrate SHAP with Rules (Level 2: WHAT DOES IT MEAN)
    # -----------------------------------------------------------------------
    abnormal_contributing: List[Dict[str, Any]] = []
    shap_normal: List[Dict[str, Any]] = []
    matched_rule_indices: Set[int] = set()

    for feat in top_n_features:
        feat_name = feat.get("feature", "")
        # Look for matching triggered rules
        matching_rules = [
            (idx, r) for idx, r in enumerate(all_triggered_rules)
            if _keys_match(r["parameter"], feat_name)
        ]

        if matching_rules:
            for idx, rule in matching_rules:
                matched_rule_indices.add(idx)
                abnormal_contributing.append({
                    "feature": feat_name,
                    "observed_value": feat.get("feature_value"),
                    "shap_rank": feat.get("rank"),
                    "shap_contribution": feat.get("contribution"),
                    "shap_direction": feat.get("direction"),
                    "condition": rule["condition"],
                    "interpretation": rule["interpretation"],
                    "possible_factors": rule["possible_factors"],
                    "priority": rule["priority"],
                    "recommendations": rule["recommendations"],
                    "rule_details": rule,
                })
        else:
            # Edge Case 1: SHAP-important but within normal range
            shap_normal.append({
                "feature": feat_name,
                "observed_value": feat.get("feature_value"),
                "shap_rank": feat.get("rank"),
                "shap_contribution": feat.get("contribution"),
                "shap_direction": feat.get("direction"),
                "status": "Within standard regulatory limits",
                "interpretation": (
                    f"Parameter '{feat_name}' provided statistical attribution toward the model prediction, "
                    f"but its observed value is currently within standard statutory limits. "
                    f"This indicates a baseline supporting factor or subtle multi-parameter interaction."
                ),
                "possible_factors": [
                    "Baseline parameter within permissible ecological range",
                    "Multi-parameter interaction path in decision tree structure",
                ],
            })

    # Edge Case 2: Abnormal parameters NOT among top-N SHAP contributors
    additional_abnormal: List[Dict[str, Any]] = []
    for idx, rule in enumerate(all_triggered_rules):
        if idx not in matched_rule_indices:
            additional_abnormal.append({
                "parameter": rule["parameter"],
                "condition": rule["condition"],
                "interpretation": rule["interpretation"],
                "possible_factors": rule["possible_factors"],
                "priority": rule["priority"],
                "recommendations": rule["recommendations"],
                "note": (
                    f"Parameter '{rule['parameter']}' exceeds statutory limits and is flagged as an abnormal condition, "
                    f"although it was not among the top {top_n} mathematical attribution contributors for this specific prediction."
                ),
                "rule_details": rule,
            })

    section_parameter_interpretation = {
        "stage": "Level 2: Parameter Interpretation (WHAT DOES IT MEAN)",
        "abnormal_contributing_parameters": abnormal_contributing,
        "shap_important_normal_parameters": shap_normal,
        "additional_abnormal_parameters": additional_abnormal,
    }

    # -----------------------------------------------------------------------
    # 5. Compile Prioritized Recommendations (Level 3: WHAT CAN BE DONE)
    # -----------------------------------------------------------------------
    high_actions: List[str] = []
    med_actions: List[str] = []
    low_actions: List[str] = []
    all_action_items: List[Dict[str, Any]] = []

    combined_rules = [item["rule_details"] for item in abnormal_contributing] + [
        item["rule_details"] for item in additional_abnormal
    ]

    for rule in combined_rules:
        prio = rule.get("priority", "Medium")
        param_label = rule.get("parameter", "Unknown")
        for rec in rule.get("recommendations", []):
            action_entry = {
                "parameter": param_label,
                "priority": prio,
                "recommendation": rec,
            }
            if action_entry not in all_action_items:
                all_action_items.append(action_entry)

            if prio == "High" and rec not in high_actions:
                high_actions.append(rec)
            elif prio == "Medium" and rec not in med_actions:
                med_actions.append(rec)
            elif prio == "Low" and rec not in low_actions:
                low_actions.append(rec)

    overall_priority = get_overall_priority(all_triggered_rules, predicted_class=resolved_pred_class)

    section_recommended_actions = {
        "stage": "Level 3: Recommended Actions (WHAT CAN BE DONE)",
        "overall_priority": overall_priority,
        "action_items": all_action_items,
        "high_priority_actions": high_actions,
        "medium_priority_actions": med_actions,
        "low_priority_actions": low_actions,
    }

    return {
        # The 4 distinct labeled sections
        "prediction": section_prediction,
        "model_level_evidence": section_model_evidence,
        "parameter_interpretation": section_parameter_interpretation,
        "recommended_actions": section_recommended_actions,
        # Top-level convenience accessors
        "predicted_class": resolved_pred_class,
        "overall_priority": overall_priority,
        "abnormal_parameter_count": len(all_triggered_rules),
        "causation_caveat": (
            "Decision-support outputs provide diagnostic interpretations and investigative recommendations "
            "based on statistical models and environmental standards. They indicate potential associations "
            "and do not constitute physical, chemical, or legal proof of pollution origin or causation."
        ),
    }


def summarize_decision_support(
    parameter_values: Dict[str, Any],
    thresholds_config: Optional[Dict[str, Any]] = None,
    predicted_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Provide a comprehensive diagnostic decision-support summary dictionary.

    Args:
        parameter_values: Dictionary of water quality parameter readings.
        thresholds_config: Optional custom thresholds config dictionary.
        predicted_class: Optional ML predicted water quality class.

    Returns:
        Structured dictionary containing triggered rules, overall priority, parameter count,
        and mandatory causation caveat.
    """
    rules = evaluate_parameters(
        parameter_values=parameter_values,
        thresholds_config=thresholds_config,
        predicted_class=predicted_class,
    )
    overall_prio = get_overall_priority(rules, predicted_class=predicted_class)

    return {
        "predicted_class": predicted_class,
        "parameters_evaluated": len(parameter_values),
        "abnormal_parameter_count": len(rules),
        "overall_priority": overall_prio,
        "triggered_rules": rules,
        "causation_caveat": (
            "Decision-support rules provide diagnostic interpretations and recommended "
            "investigations based on statutory water quality standards. They indicate "
            "potential environmental associations, not legal proof of pollution origin or causality."
        ),
    }


if __name__ == "__main__":
    # Standalone demonstration
    print("=" * 70)
    print("Decision-Support Rule Engine - Standalone Execution Demonstration")
    print("=" * 70)

    sample_inputs = {
        "Potential of Hydrogen (pH)": 6.2,
        "Dissolved oxygen (mg/L)": 3.8,
        "Biochemical Oxygen Demand (mg/L)": 5.5,
        "Fecal Coliform (MPN/100mL)": 1200.0,
        "Turbidity (NTU)": 12.0,
    }

    summary = summarize_decision_support(sample_inputs, predicted_class="Poor")
    print(f"Parameters Evaluated: {summary['parameters_evaluated']}")
    print(f"Abnormal Parameters: {summary['abnormal_parameter_count']}")
    print(f"Overall Priority:     {summary['overall_priority']}")
    print("-" * 70)
    for idx, rule in enumerate(summary["triggered_rules"], 1):
        print(f"Rule #{idx}: [{rule['priority']} Priority] {rule['parameter']}")
        print(f"  Condition:       {rule['condition']}")
        print(f"  Interpretation:  {rule['interpretation']}")
        print(f"  Recommendations: {', '.join(rule['recommendations'][:2])}")
    print("=" * 70)

