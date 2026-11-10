"""Local dashboard API. Run from the repository root with python -m app.api."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from app.dashboard_prediction import predict_assessment
from app.dashboard_input import frozen_model
from app.dashboard_history import historical_dashboard
from src.data.load_data import DataIngestionError
from app.dashboard_input import FIELDS, InputError, preprocess_assessment


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if urlsplit(self.path).path == "/api/assessment/schema":
            self.send_json(200, FIELDS)
        elif urlsplit(self.path).path == "/api/dashboard":
            try:
                history = historical_dashboard()
            except (OSError, DataIngestionError):
                self.send_json(503, {"error": "Historical observations are unavailable."})
                return
            self.send_json(200, history)
        else:
            self.send_error(404)

    def do_POST(self):
        routes = {"/api/assessment/preprocess": preprocess_assessment,
                  "/api/assessment/predict": predict_assessment}
        handler = routes.get(urlsplit(self.path).path)
        if handler is None:
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError
            values = json.loads(self.rfile.read(length))
        except (ValueError, RecursionError):
            self.send_json(400, {"error": "Invalid assessment request."})
            return
        try:
            result = handler(values)
        except InputError as exc:
            self.send_json(422, {"error": str(exc)})
            return
        except Exception:
            self.log_error("Assessment processing failed")
            self.send_json(500, {"error": "Assessment processing is unavailable. Please try again."})
            return
        self.send_json(200, result)

    def send_json(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    frozen_model()
    with ThreadingHTTPServer(("127.0.0.1", 18741), DashboardHandler) as server:
        server.serve_forever()
