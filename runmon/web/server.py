"""
Browser view of a run: a small local HTTP server (standard library only).

``runmon web TARGET`` serves one page on ``http://127.0.0.1:<port>/`` that
updates live.  It is a *client* of a run — the same sources as ``runmon
watch`` (:mod:`runmon.client.sources`: local files, the live server, or a
cluster run over SSH) — so it runs on your machine, never inside the job.

Endpoints (all GET, read-only):

``/``               the page (``index.html``)
``/api/snapshot``   the latest snapshot + the source's label / state
``/api/stream``     Server-Sent Events: a ``snapshot`` message whenever it changes
``/api/history``    CPU / memory totals over the whole run (from heartbeats)
``/api/query``      search the event log (``kind``, ``level``, ``grep``, ``since``,
                    ``task``, ``limit``) — see :mod:`runmon.query`

Security: binds 127.0.0.1 by default and accepts only requests whose ``Host``
is the loopback address (so a web page elsewhere cannot read it through DNS
rebinding).  It exposes what the run's token protects, to *your* machine only.
"""

from __future__ import annotations

import json
import re
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse

from runmon import query as q

LOOPBACK = {"127.0.0.1", "localhost", "[::1]", "::1"}

#: ``() -> iterable of raw JSON lines`` of the run's events.jsonl.
LineReader = Callable[[], Iterable[bytes]]


def _page() -> bytes:
    return resources.files("runmon.web").joinpath("index.html").read_bytes()


class WebApp:
    """State shared by the request handlers.

    :param source: A client source (``snapshot()``, ``label``, ``state``, ``close()``).
    :param lines: Reader of the run's event log (for history and search), or ``None``.
    """

    def __init__(self, source, lines: Optional[LineReader] = None):
        self.source = source
        self.lines = lines
        self._lock = threading.Lock()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap = self.source.snapshot()
        return {"source": self.source.label, "state": self.source.state, "snapshot": snap}

    def history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Totals from the run's heartbeats (one every ~30 s) and the stage starts /
        ends (timeline markers), oldest first — one pass over the event log."""
        if self.lines is None:
            return {"history": [], "stages": []}
        out, stages = [], []
        try:
            for e in q.iter_matching(self.lines(), kinds=["heartbeat", "stage_start", "stage_end"]):
                f = e.get("f") or {}
                if e.get("kind") != "heartbeat":
                    stages.append({"ts": e.get("ts"), "kind": e.get("kind"), "stage": f.get("stage"),
                                   "status": f.get("status")})
                    continue
                t = f.get("totals") or {}
                if t:
                    out.append({"ts": e.get("ts"), "cpu_percent": t.get("cpu_percent"),
                                "rss_mb": t.get("rss_mb"), "allocation": f.get("allocation"),
                                "pool_workers": (f.get("pool_workers") or {}).get("workers")})
        except OSError:
            pass
        return {"history": out, "stages": stages}

    def query(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Search the event log with the ``runmon query`` filters."""
        if self.lines is None:
            return {"error": "no event log for this source", "events": []}
        try:
            since = q.parse_when(params.get("since") or None)
        except ValueError as e:
            return {"error": str(e), "events": []}
        kinds = [k for k in (params.get("kind") or "").split(",") if k]
        grep = params.get("grep") or None
        try:
            pattern = re.compile(grep, re.I) if grep else None
        except re.error as e:
            return {"error": f"bad pattern: {e}", "events": []}
        limit = max(1, min(5000, int(params.get("limit") or 500)))
        found: List[Dict[str, Any]] = []
        for e in q.iter_matching(self.lines(), kinds=kinds, level=params.get("level") or None,
                                 pattern=pattern, since=since, task=params.get("task") or None):
            found.append(e)
            if len(found) > 4 * limit:
                found = found[-limit:]
        return {"events": found[-limit:], "total": len(found)}


def _handler(app: WebApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "runmon"

        def log_message(self, fmt, *args):      # quiet
            pass

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").strip()
            name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
            return name in LOOPBACK

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self):                        # noqa: N802 (http.server API)
            if not self._host_ok():
                self._send(HTTPStatus.FORBIDDEN, b"forbidden host", "text/plain")
                return
            url = urlparse(self.path)
            params = {k: v[-1] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, _page(), "text/html; charset=utf-8")
                elif url.path == "/api/snapshot":
                    self._json(app.snapshot())
                elif url.path == "/api/history":
                    self._json(app.history())
                elif url.path == "/api/query":
                    self._json(app.query(params))
                elif url.path == "/api/stream":
                    self._stream()
                else:
                    self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last, quiet = None, 0.0
            while True:
                data = app.snapshot()
                snap = data.get("snapshot") or {}
                events = snap.get("events") or []
                key = (snap.get("ts"), len(events), events[-1].get("seq") if events else None,
                       data["state"])
                if key != last:
                    last = key
                    payload = json.dumps(data, default=str)
                    self.wfile.write(f"event: snapshot\ndata: {payload}\n\n".encode())
                    self.wfile.flush()
                    quiet = 0.0
                elif quiet > 15:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    quiet = 0.0
                time.sleep(0.5)
                quiet += 0.5

    return Handler


def serve(source, *, lines: Optional[LineReader] = None, host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True) -> None:
    """Serve the page until Ctrl-C (tries the next ports if ``port`` is taken)."""
    app = WebApp(source, lines)
    httpd = None
    for candidate in [port + i for i in range(20)] + [0]:
        try:
            httpd = ThreadingHTTPServer((host, candidate), _handler(app))
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit("no free port for the web view")
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"runmon web: {url}  (source: {source.label}; Ctrl-C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        source.close()


def make_server(source, lines: Optional[LineReader] = None, host: str = "127.0.0.1", port: int = 0):
    """A server object for tests / embedding (call ``serve_forever`` yourself)."""
    httpd = ThreadingHTTPServer((host, port), _handler(WebApp(source, lines)))
    httpd.daemon_threads = True
    return httpd
