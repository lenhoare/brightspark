#!/usr/bin/env python3
"""Local web app for browsing speculative-decoding runs.

    python3 viz/serve.py            # http://127.0.0.1:8081

Serves every run under runs/ and builds each replay on demand, so a run
recorded five minutes ago shows up on reload with no extra step. Replays are
cached next to the trace and rebuilt whenever the trace is newer.

Binds to localhost only.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from anticipation import export_replay, schema  # noqa: E402

RUNS = ROOT / "runs"
APP = Path(__file__).resolve().parent / "app.html"

_build_lock = threading.Lock()


def list_runs() -> list[dict]:
    out = []
    if not RUNS.exists():
        return out
    candidates = [d for d in RUNS.iterdir() if d.is_dir() and (d / "trace.jsonl").exists()]
    # order by when the run was recorded, not by directory mtime -- writing the
    # replay cache touches the directory and would otherwise reshuffle the list
    candidates.sort(key=lambda d: (d / "trace.jsonl").stat().st_mtime, reverse=True)
    for d in candidates:
        trace = d / "trace.jsonl"
        takes = 0
        cached = d / "replay.json"
        if cached.exists():
            try:
                takes = len(json.loads(cached.read_text())["takes"])
            except Exception:
                takes = 0
        out.append({"run": d.name, "takes": takes,
                    "mtime": int(trace.stat().st_mtime)})
    return out


def replay_for(name: str) -> bytes:
    """Cached replay payload, rebuilt when the trace is newer than the cache."""
    run = (RUNS / name).resolve()
    if RUNS.resolve() not in run.parents or not run.is_dir():
        raise FileNotFoundError(f"no such run: {name}")

    trace = run / "trace.jsonl"
    if not trace.exists():
        raise FileNotFoundError(f"{name} has no trace.jsonl")

    # the payload is built from the trace AND the manifest/responses, which the
    # harness writes only after the server stops. Invalidating on the trace alone
    # would permanently cache a prompt-less replay built while a run was still
    # in flight.
    def newest_input() -> float:
        return max((run / f).stat().st_mtime
                   for f in ("trace.jsonl", "manifest.jsonl", "responses.json", "run.json")
                   if (run / f).exists())

    cache = run / "replay.json"
    if cache.exists() and cache.stat().st_mtime >= newest_input():
        return cache.read_bytes()

    with _build_lock:
        if cache.exists() and cache.stat().st_mtime >= newest_input():
            return cache.read_bytes()
        payload = export_replay.build(run)
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        cache.write_bytes(data)
        return data


class Handler(BaseHTTPRequestHandler):
    server_version = "anticipation/1.0"

    def log_message(self, fmt, *args):  # quieter than the default
        if "api/replay" in (args[0] if args else ""):
            sys.stderr.write(f"  {args[0]}\n")

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)

        if path in ("/", "/index.html"):
            if not APP.exists():
                self._send(b"app.html is missing", "text/plain; charset=utf-8", 500)
                return
            self._send(APP.read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/api/runs":
            self._send(json.dumps(list_runs()).encode(), "application/json")
            return

        if path.startswith("/api/replay/"):
            name = path[len("/api/replay/"):]
            try:
                self._send(replay_for(name), "application/json")
            except FileNotFoundError as e:
                self._send(str(e).encode(), "text/plain; charset=utf-8", 404)
            except Exception as e:
                self._send(f"{type(e).__name__}: {e}".encode(),
                           "text/plain; charset=utf-8", 500)
            return

        self._send(b"not found", "text/plain; charset=utf-8", 404)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="open a browser window")
    args = ap.parse_args()

    runs = list_runs()
    url = f"http://{args.host}:{args.port}/"

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Draft Graveyard -> {url}")
    print(f"  {len(runs)} run(s) in {RUNS}"
          + (f": {', '.join(r['run'] for r in runs[:6])}" if runs else " (record one first)"))
    print("  ctrl-c to stop", flush=True)

    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
