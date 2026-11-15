#!/usr/bin/env python3
"""Local test server that never lets the browser keep old files.

    python3 tools/serve.py          then open http://localhost:8000

The standard `python3 -m http.server` lets Chrome reuse older copies of some
scripts after an ordinary reload, so the page can end up with a mix of old and
new files and stop working. This server tells the browser not to store anything.
"""
import http.server
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
with http.server.ThreadingHTTPServer(("", port), Handler) as srv:
    print(f"http://localhost:{port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
