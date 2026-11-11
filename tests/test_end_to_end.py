"""Exercise the public dashboard API with five complete CPCB holdout observations."""
import json
import math
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread

import pytest

from app.api import DashboardHandler


@pytest.fixture(scope="module")
def dashboard_address():
    with ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler) as server:
        worker = Thread(target=server.serve_forever)
        worker.start()
        try:
            yield server.server_address
        finally:
            server.shutdown()
            worker.join()


@pytest.mark.parametrize(("expected", "ph", "oxygen", "bod", "coliform"), [
    ("Excellent", 6.94, 6.4, 3.9, 2),
    ("Good", 7.3, 9.6, 1.3, 320),
    ("Poor", 8.22, 9.9, 2.4, 1100),
    ("Very Poor", 7.42, 7.0, 3.3, 5000),
    ("Unsuitable", 7.4, 4.2, 11.0, 350),
], ids=["excellent", "good", "poor", "very-poor", "unsuitable"])
def test_raw_measurements_to_dashboard_decision(dashboard_address, expected, ph, oxygen, bod, coliform):
    measurements = {
        "Potential of Hydrogen (pH)": ph,
        "Dissolved oxygen (mg/L)": oxygen,
        "Biochemical Oxygen Demand (mg/L)": bod,
        "Fecal Coliform (MPN/100mL)": coliform,
    }
    connection = HTTPConnection(*dashboard_address, timeout=15)
    try:
        connection.request("POST", "/api/assessment/predict", json.dumps(measurements),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        body = json.load(response)
        assert response.status == 200, body
    finally:
        connection.close()

    assert body["measurements"] == measurements
    assert body["predicted_class"] == expected
    probabilities = body["class_probabilities"]
    assert set(probabilities) == {"Excellent", "Good", "Poor", "Very Poor", "Unsuitable"}
    assert sum(probabilities.values()) == pytest.approx(1)
    assert probabilities[expected] == body["model_probability"] == max(probabilities.values())
    contributions = {item["feature"]: item for item in body["contributions"]}
    assert contributions.keys() == measurements.keys() | {"Log10 Fecal Coliform"}
    assert contributions["Log10 Fecal Coliform"]["feature_value"] == pytest.approx(math.log10(coliform))
    assert body["base_value"] + sum(item["contribution"] for item in contributions.values()) == pytest.approx(probabilities[expected])

    decision = body["decision"]
    assert decision["prediction"]["predicted_class"] == expected
    assert decision["model_level_evidence"]["top_contributing_features"] == body["contributions"][:3]
    assert decision["parameter_interpretation"]["abnormal_contributing_parameters"] or decision["parameter_interpretation"]["additional_abnormal_parameters"]
    flagged = {p["name"] for p in body["parameters"] if p["abnormal"]}
    actions = decision["recommended_actions"]["action_items"]
    assert actions and all(a["parameter"] in flagged and a["recommendation"] for a in actions)
    assert decision["recommended_actions"]["overall_priority"] == decision["overall_priority"]
