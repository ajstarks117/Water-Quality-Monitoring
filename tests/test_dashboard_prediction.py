"""M9.3 parity with the frozen artifact and Phase 7 explanation contract."""
from unittest.mock import patch

import pytest

from app.dashboard_input import BOUNDS, assessment_frame, frozen_model
from app.dashboard_prediction import predict_assessment
from src.explainability.shap_explainer import get_local_explanation
from src.models.predict import load_model

SAMPLE = dict(zip(BOUNDS, [7.2, 4.8, 6.2, 320]))


def test_dashboard_matches_direct_model_and_shap():
    frame = assessment_frame(SAMPLE)
    model = load_model()
    direct = get_local_explanation(model, frame)
    result = predict_assessment(SAMPLE)
    assert result["measurements"] == SAMPLE
    assert result["predicted_class"] == model.predict(frame)[0]
    assert result["class_probabilities"] == dict(zip(model.classes_, model.predict_proba(frame)[0]))
    assert result["model_probability"] == direct["predicted_probability"]
    assert result["contributions"] == direct["feature_contributions"]
    assert result["base_value"] == direct["base_value"]
    assert frozen_model() is frozen_model()


@pytest.mark.parametrize(("probability", "uncertain"), [(0.59, True), (0.60, False)])
def test_uncertainty_display_boundary(probability, uncertain):
    record = get_local_explanation(frozen_model(), assessment_frame(SAMPLE))
    record["predicted_probability"] = probability
    with patch("app.dashboard_prediction.get_local_explanation", return_value=record):
        assert predict_assessment(SAMPLE)["uncertain"] is uncertain
