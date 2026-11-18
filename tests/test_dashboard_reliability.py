"""HTTP validation, safe failure handling and private request log regression."""
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread
from unittest.mock import patch

import pytest

from app.api import DashboardHandler
from app.dashboard_input import BOUNDS


@pytest.fixture
def server_address():
    with ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler) as server:
        worker = Thread(target=server.serve_forever)
        worker.start()
        try:
            yield server.server_address
        finally:
            server.shutdown()
            worker.join()


def request(address, body, path="/api/assessment/predict", method="POST"):
    connection = HTTPConnection(*address, timeout=10)
    try:
        connection.request(method, path, body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.load(response)
    finally:
        connection.close()


@pytest.mark.parametrize("body,status", [
    ("{", 400), ("", 400), ("x" * 8193, 400),
    ("null", 422), ("[]", 422), ('"hello"', 422), ("{}", 422),
    *[(json.dumps(dict(zip(BOUNDS, [value, 6.3, 2.6, 4]))), 422)
      for value in [None, "", True, -1, 15, "7.8", "eval(1)", [], {}, float("nan"), float("inf")]],
    (json.dumps({**dict(zip(BOUNDS, [7.8, 6.3, 2.6, 4])), "extra": 1}), 422),
])
def test_invalid_request_never_reaches_model(server_address, body, status):
    with patch("app.dashboard_input.frozen_model") as model:
        actual, payload = request(server_address, body)
        assert actual == status
        assert set(payload) == {"error"}
        model.assert_not_called()


@pytest.mark.parametrize("method,path,target,value,status", [
    ("POST", "/api/assessment/predict", "app.api.predict_assessment",
     RuntimeError("PRIVATE_SENTINEL C:/private/key"), 500),
    ("POST", "/api/assessment/predict", "app.api.predict_assessment",
     {"bad": float("nan")}, 500),
    ("GET", "/api/dashboard", "app.api.historical_dashboard",
     RuntimeError("PRIVATE_SENTINEL C:/private/key"), 503),
    ("GET", "/api/dashboard", "app.api.historical_dashboard",
     {"bad": float("nan")}, 503),
])
def test_internal_failures_are_safe(server_address, capsys, method, path, target, value, status):
    behavior = {"side_effect": value} if isinstance(value, Exception) else {"return_value": value}
    with patch(target, **behavior):
        actual, payload = request(server_address, "{}", path + "?PRIVATE_QUERY=secret", method)
    assert actual == status
    assert set(payload) == {"error"}
    text = json.dumps(payload) + capsys.readouterr().err
    for private in ["PRIVATE_SENTINEL", "PRIVATE_QUERY", "C:/private", "Traceback"]:
        assert private not in text
