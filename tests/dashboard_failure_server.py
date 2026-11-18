"""Test-only backend for verifying safe errors in the real browser.

The production handler and prediction path run unchanged. A failing threshold
loader simulates an internal configuration failure in this isolated process.
"""
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from app.api import DashboardHandler


if __name__ == "__main__":
    with patch("app.dashboard_prediction.load_thresholds",
               side_effect=ValueError("PRIVATE_SENTINEL C:/private/config.yaml")):
        with ThreadingHTTPServer(("127.0.0.1", 18742), DashboardHandler) as server:
            server.serve_forever()
