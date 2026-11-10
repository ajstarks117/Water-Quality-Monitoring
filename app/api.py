"""Local M9.1 dashboard API. Run from the repository root with python -m app.api."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from app.mock_data import DASHBOARD


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if urlsplit(self.path).path != "/api/dashboard":
            self.send_error(404)
            return
        body = json.dumps(DASHBOARD, allow_nan=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    with ThreadingHTTPServer(("127.0.0.1", 18741), DashboardHandler) as server:
        server.serve_forever()
