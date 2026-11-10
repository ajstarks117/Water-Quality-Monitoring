"""Live prediction and local explanation for the same validated observation."""
from src.decision_support.rules import generate_decision_support, load_thresholds

from app.dashboard_input import assessment_frame, frozen_model
from src.explainability.shap_explainer import get_local_explanation

# Dashboard display policy, independent of training and probability calibration.
UNCERTAINTY_THRESHOLD = 0.60


def predict_assessment(values):
    frame = assessment_frame(values)
    # Phase 7 calls both predict and predict_proba on this frame and pipeline.
    explanation = get_local_explanation(frozen_model(), frame)
    config = load_thresholds()
    contributions = explanation["feature_contributions"]
    values = {item["feature"]: item["feature_value"] for item in contributions}
    decision = generate_decision_support(local_shap_output=explanation, parameter_values=values,
                                         thresholds_config=config)
    interpretations = decision["parameter_interpretation"]
    triggered = {
        item["rule_details"]["parameter"]
        for group in ("abnormal_contributing_parameters", "additional_abnormal_parameters")
        for item in interpretations[group]
    }
    parameters = []
    for spec in config["parameters"].values():
        name = spec["column_name"]
        if name in values:
            parameters.append({
                "name": name, "value": values[name], "unit": spec["unit"],
                "minimum": spec.get("min_acceptable"),
                "maximum": spec.get("max_acceptable", spec.get("supersaturation")),
                "source": spec["standard_source"], "abnormal": name in triggered,
            })
    return {
        "parameters": parameters, "decision": decision,
        "measurements": frame.iloc[0].to_dict(),
        "predicted_class": explanation["predicted_class"],
        "model_probability": explanation["predicted_probability"],
        "class_probabilities": explanation["class_probabilities"],
        "uncertain": explanation["predicted_probability"] < UNCERTAINTY_THRESHOLD,
        "base_value": explanation["base_value"],
        "contributions": explanation["feature_contributions"],
        "caveat": explanation["causation_caveat"],
    }
