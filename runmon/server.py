"""
The live monitor server — runs **inside the hub thread** (no extra thread or process).

Protocol: newline-delimited JSON over TCP, read-only.

1. The client connects and sends ``{"type": "hello", "v": 1, "token": "<token>"}``
   within :data:`HELLO_TIMEOUT` seconds; a wrong / missing token closes the
   connection.  The token is in ``monitor.json`` (mode 0600, readable only by
   the run's owner), so on a shared cluster other users cannot watch.
2. The server answers ``{"type": "hello", "v": 1, "run": {...}}`` followed at
   once by a full ``{"type": "snapshot", ...}`` (which includes recent events).
3. Then: a snapshot every ``snapshot_s`` seconds and every new event as
   ``{"type": "event", "event": {...}}`` as soon as it is published.
4. When the run ends: ``{"type": "bye", "reason": "..."}``.

Robustness: sockets are non-blocking; each client has a bounded output buffer.
A client that does not keep up (buffer above :data:`MAX_BUFFER`) has its
backlog dropped — the next snapshot resynchronises it — so a slow or stuck
client can never stall the monitored run.  Snapshots are only built while at
least one client is connected.

Binding: ``host="auto"`` binds 127.0.0.1 normally and, inside a batch job
(PBS / Slurm), all interfaces with the node's hostname advertised in
``monitor.json`` — compute nodes are reached through a port forward made on the
cluster's login node (``ssh -L local:<node>:<port> login``), which needs the
port to be open on the node's cluster address.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import selectors
import socket
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from runmon.events import dumps

PROTOCOL_VERSION = 1
HELLO_TIMEOUT = 5.0
MAX_BUFFER = 8 << 20            # bytes queued per client before its backlog is dropped
MAX_LINE = 1 << 16              # longest client line accepted
INFO_FILE = "monitor.json"


def resolve_bind(host: str) -> tuple:
    """``(bind_address, advertised_host)`` for a configured host (``"auto"`` → see module doc)."""
    if host and host != "auto":
        return host, (socket.gethostname() if host in ("0.0.0.0", "::") else host)
    in_job = bool(os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID"))
    if in_job:
        return "0.0.0.0", socket.gethostname()
    return "127.0.0.1", "127.0.0.1"


class _Client:
    __slots__ = ("sock", "addr", "inbuf", "outbuf", "authed", "since", "dropped")

    def __init__(self, sock, addr):
        self.sock = sock
        self.addr = addr
        self.inbuf = b""
        self.outbuf = bytearray()
        self.authed = False
        self.since = time.monotonic()
        self.dropped = 0


class MonitorServer:
    """Non-blocking server driven by the hub's loop (see module docstring).

    :param hub: The :class:`~runmon.hub.Hub` whose state is served.
    :param host: Bind host (``"auto"``, ``"127.0.0.1"``, ``"0.0.0.0"`` ...).
    :param port: TCP port (0 = any free port).
    :param snapshot_s: Seconds between snapshots pushed to clients.
    """

    def __init__(self, hub, host: str = "auto", port: int = 0, snapshot_s: float = 1.0):
        self.hub = hub
        self.snapshot_s = snapshot_s
        self.token = secrets.token_urlsafe(24)
        bind, self.advertised = resolve_bind(host)
        self.sel = selectors.DefaultSelector()
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((bind, int(port)))
        self.listener.listen(16)
        self.listener.setblocking(False)
        self.bind = bind
        self.port = self.listener.getsockname()[1]
        self.sel.register(self.listener, selectors.EVENT_READ, "listen")
        self._wake_r, self._wake_w = socket.socketpair()
        self._wake_r.setblocking(False)
        self._wake_w.setblocking(False)
        self.sel.register(self._wake_r, selectors.EVENT_READ, "wake")
        self.clients: Dict[int, _Client] = {}
        self._events: Deque[Dict[str, Any]] = deque(maxlen=10_000)
        self._last_snapshot = 0.0
        self.closed = False
        hub.subscribe(self._events.append)

    # ------------------------------------------------------------------
    # monitor.json
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """Connection details written to ``monitor.json``."""
        return {"v": PROTOCOL_VERSION, "host": self.advertised, "bind": self.bind, "port": self.port,
                "token": self.token, "pid": os.getpid(), "run": self.hub.run_id,
                "started": self.hub.started, "node": socket.gethostname(), "stopped": False}

    def write_info(self, **extra: Any) -> str:
        """Write ``monitor.json`` (owner-only permissions) atomically; returns its path."""
        data = self.info()
        data.update(extra)
        path = os.path.join(self.hub.run_dir, INFO_FILE)
        tmp = f"{path}.tmp{os.getpid()}"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
        return path

    # ------------------------------------------------------------------
    # Loop integration
    # ------------------------------------------------------------------

    def wake(self) -> None:
        """Interrupt :meth:`poll` (any thread)."""
        try:
            self._wake_w.send(b"x")
        except OSError:
            pass

    def poll(self, timeout: float) -> None:
        """Serve the network for up to ``timeout`` seconds (called by the hub loop)."""
        if self.closed:
            time.sleep(max(0.0, timeout))
            return
        self._flush_pending()
        for key, mask in self.sel.select(max(0.0, timeout)):
            if key.data == "listen":
                self._accept()
            elif key.data == "wake":
                try:
                    while self._wake_r.recv(4096):
                        pass
                except (BlockingIOError, OSError):
                    pass
            else:
                client = key.data
                if mask & selectors.EVENT_READ:
                    self._read(client)
                if mask & selectors.EVENT_WRITE and client.sock.fileno() in self.clients:
                    self._write(client)
        self._flush_pending()

    def _flush_pending(self) -> None:
        now = time.monotonic()
        for c in list(self.clients.values()):           # hello timeouts
            if not c.authed and now - c.since > HELLO_TIMEOUT:
                self._drop(c)
        authed = [c for c in self.clients.values() if c.authed]
        events = []
        while self._events:
            try:
                events.append(self._events.popleft())
            except IndexError:
                break
        if authed and events:
            data = b"".join(self._line({"type": "event", "event": e}) for e in events)
            for c in authed:
                self._queue(c, data)
        if authed and now - self._last_snapshot >= self.snapshot_s:
            self._last_snapshot = now
            data = self._line(self.hub.snapshot())
            for c in authed:
                self._queue(c, data)

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    @staticmethod
    def _line(obj: Dict[str, Any]) -> bytes:
        return (dumps(obj) + "\n").encode("utf-8")

    def _accept(self) -> None:
        try:
            sock, addr = self.listener.accept()
        except (BlockingIOError, OSError):
            return
        sock.setblocking(False)
        c = _Client(sock, addr)
        self.clients[sock.fileno()] = c
        self.sel.register(sock, selectors.EVENT_READ, c)

    def _read(self, c: _Client) -> None:
        try:
            data = c.sock.recv(65536)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            self._drop(c)
            return
        if not data:
            self._drop(c)
            return
        c.inbuf += data
        if len(c.inbuf) > MAX_LINE:
            self._drop(c)
            return
        while b"\n" in c.inbuf:
            line, c.inbuf = c.inbuf.split(b"\n", 1)
            self._handle(c, line)
            if c.sock.fileno() not in self.clients:
                return

    def _handle(self, c: _Client, line: bytes) -> None:
        try:
            msg = json.loads(line)
        except ValueError:
            self._drop(c)
            return
        if not c.authed:
            token = str(msg.get("token", "")) if isinstance(msg, dict) else ""
            if msg.get("type") != "hello" or not hmac.compare_digest(token, self.token):
                self._queue(c, self._line({"type": "error", "error": "bad hello or token"}))
                self._write(c)
                self._drop(c)
                return
            c.authed = True
            self.hub.event("client_connected", f"monitor client connected from {c.addr[0]}", "DEBUG",
                           peer=f"{c.addr[0]}:{c.addr[1]}")
            hello = {"type": "hello", "v": PROTOCOL_VERSION,
                     "run": {"id": self.hub.run_id, "host": self.hub.host, "pid": self.hub.pid,
                             "dir": self.hub.run_dir, "started": self.hub.started}}
            self._queue(c, self._line(hello) + self._line(self.hub.snapshot()))
        # Authenticated clients may send nothing else meaningful (read-only); ignore.

    def _queue(self, c: _Client, data: bytes) -> None:
        if len(c.outbuf) + len(data) > MAX_BUFFER:
            c.outbuf.clear()                 # slow client: drop its backlog, resync by snapshot
            c.dropped += 1
            self._last_snapshot = 0.0
            return
        c.outbuf += data
        self._write(c)

    def _write(self, c: _Client) -> None:
        if c.sock.fileno() not in self.clients:
            return
        while c.outbuf:
            try:
                n = c.sock.send(c.outbuf)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                self._drop(c)
                return
            if n <= 0:
                break
            del c.outbuf[:n]
        try:
            self.sel.modify(c.sock, selectors.EVENT_READ | (selectors.EVENT_WRITE if c.outbuf else 0), c)
        except (KeyError, ValueError, OSError):
            pass

    def _drop(self, c: _Client) -> None:
        fd = c.sock.fileno()
        self.clients.pop(fd, None)
        try:
            self.sel.unregister(c.sock)
        except (KeyError, ValueError, OSError):
            pass
        try:
            c.sock.close()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self, reason: str = "exit") -> None:
        """Send ``bye`` (best effort, ≤1 s), close everything, mark ``monitor.json`` stopped."""
        if self.closed:
            return
        self._flush_pending()
        bye = self._line({"type": "bye", "reason": reason})
        deadline = time.monotonic() + 1.0
        for c in list(self.clients.values()):
            if c.authed:
                c.outbuf += bye
            while c.outbuf and time.monotonic() < deadline and c.sock.fileno() in self.clients:
                self._write(c)
                if c.outbuf:
                    time.sleep(0.01)
            self._drop(c)
        self.closed = True
        for s in (self.listener, self._wake_r, self._wake_w):
            try:
                self.sel.unregister(s)
            except (KeyError, ValueError, OSError):
                pass
            try:
                s.close()
            except OSError:
                pass
        self.sel.close()
        try:
            self.write_info(stopped=True, reason=reason)
        except OSError:
            pass
