"""Live prediction and local explanation for the same validated observation."""
from app.dashboard_input import assessment_frame, frozen_model
from src.explainability.shap_explainer import get_local_explanation

# Dashboard display policy, independent of training and probability calibration.
UNCERTAINTY_THRESHOLD = 0.60


def predict_assessment(values):
    frame = assessment_frame(values)
    # Phase 7 calls both predict and predict_proba on this frame and pipeline.
    explanation = get_local_explanation(frozen_model(), frame)
    return {
        "measurements": frame.iloc[0].to_dict(),
        "predicted_class": explanation["predicted_class"],
        "model_probability": explanation["predicted_probability"],
        "class_probabilities": explanation["class_probabilities"],
        "uncertain": explanation["predicted_probability"] < UNCERTAINTY_THRESHOLD,
        "base_value": explanation["base_value"],
        "contributions": explanation["feature_contributions"],
        "caveat": explanation["causation_caveat"],
    }
