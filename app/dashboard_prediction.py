"""Live prediction and local explanation for the same validated observation."""
import pandas as pd

from src.decision_support.rules import generate_decision_support, load_thresholds

from app.dashboard_input import BOUNDS, assessment_frame, frozen_model
from src.explainability.shap_explainer import get_local_explanation

# Dashboard display policy, independent of training and probability calibration.
UNCERTAINTY_THRESHOLD = 0.60

# Marginal extrema of the M4.2 X_train_v1 split, in raw measurement units.
# Probe steps are dashboard display policy, not laboratory error estimates.
TRAINING_RANGES_AND_STEPS = {
    "Potential of Hydrogen (pH)": (5.45, 9.2, 0.01),
    "Dissolved oxygen (mg/L)": (0.3, 28, 0.1),
    "Biochemical Oxygen Demand (mg/L)": (1.1, 127, 0.1),
    "Fecal Coliform (MPN/100mL)": (2, 22_000_000, 1),
}


def screening_warnings(frame, predicted_class):
    sample = frame.iloc[0].to_dict()
    probes = []
    changes = []
    messages = []
    for name, (minimum, maximum, step) in TRAINING_RANGES_AND_STEPS.items():
        value = sample[name]
        if not minimum <= value <= maximum:
            messages.append(f"{name} is outside the observed training range ({minimum:g}–{maximum:g}).")
        low, high = BOUNDS[name]
        for nearby in (value - step, value + step):
            if low <= nearby <= high:
                probes.append({**sample, name: nearby})
                changes.append((name, nearby))
    labels = frozen_model().predict(pd.DataFrame(probes, columns=frame.columns))
    for (name, nearby), label in zip(changes, labels):
        if label != predicted_class:
            messages.append(f"Prediction is sensitive to small measurement changes. "
                            f"Changing only {name} to {nearby:g} gives {label}. Verify the measurement.")
            break
    return messages


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
        "screening_warnings": screening_warnings(frame, explanation["predicted_class"]),
        "base_value": explanation["base_value"],
        "contributions": explanation["feature_contributions"],
        "caveat": explanation["causation_caveat"],
    }
