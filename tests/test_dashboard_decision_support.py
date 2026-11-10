"""Milestone 9.4 contract checks against the frozen rules and labeled observations."""
from app.dashboard_input import BOUNDS
from app.dashboard_history import historical_dashboard
from app.dashboard_prediction import predict_assessment


def test_two_abnormal_and_three_normal_values():
    sample = dict(zip(BOUNDS, [6.5, 4.9, 3.1, 250]))
    result = predict_assessment(sample)
    rows = result["parameters"]
    assert [r["abnormal"] for r in rows] == [False, True, True, False, False]
    assert rows[1]["minimum"] == 5
    assert rows[1]["maximum"] == 14.6
    assert rows[-1]["maximum"] == 2.398
    decision = result["decision"]
    assert decision["overall_priority"] == "High"
    assert decision["prediction"]["predicted_class"] == result["predicted_class"]
    assert decision["model_level_evidence"]["top_contributing_features"] == result["contributions"][:3]
    assert decision["recommended_actions"]["action_items"]


def test_historical_coverage_matches_labeled_records():
    history = historical_dashboard()
    records = history["records"]
    assert sum(history["distribution"].values()) == len(records)
    assert len({r["month"] for r in records if r["month"]}) > 1
    assert any(r[f] is None for r in records for f in history["features"])
    assert {s["Station"] for s in history["stations"]} == {r["station"] for r in records}
