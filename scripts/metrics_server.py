"""Prometheus exporter for the local demo.  python scripts/metrics_server.py [--port 9108] [--window 24h] [--once]

Serves GET /metrics (text exposition format) and GET /healthz. Read-only: it only reads the local trace / review databases
and data files. Bind address defaults to 127.0.0.1 - put it behind your network controls before exposing it."""
import argparse
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config  # noqa: E402
from src.monitoring.prometheus import render_prometheus  # noqa: E402
from src.monitoring.snapshot import WINDOWS, build_snapshot  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9108)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--window", default="24h", choices=list(WINDOWS))
    ap.add_argument("--once", action="store_true", help="print the metrics once and exit")
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.once:
        print(render_prometheus(build_snapshot(cfg, a.window)), end="")
        return 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.split("?")[0] == "/metrics":
                body, ctype = render_prometheus(build_snapshot(cfg, a.window)).encode(), "text/plain; version=0.0.4; charset=utf-8"
            elif self.path == "/healthz":
                body, ctype = b"ok\n", "text/plain"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep the console quiet
            pass

    print(f"serving http://{a.host}:{a.port}/metrics (window {a.window})")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
