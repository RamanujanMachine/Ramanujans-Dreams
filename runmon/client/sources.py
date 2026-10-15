"""
Where the terminal view gets its snapshots from.

Every source offers ``snapshot() -> dict | None``, ``label``, ``state`` and
``close()``; the view (:mod:`runmon.client.tui`) does not care which it is.

* :class:`FileSource` — ``status.json`` in a run directory (same machine or a
  shared filesystem; refreshed by the hub every few seconds).
* :class:`SocketSource` — the live server (:mod:`runmon.server`): snapshots
  every second plus every event as it happens; reconnects if the connection
  drops and stops at the server's ``bye``.
* :class:`SshFileSource` — ``status.json`` streamed over one SSH connection
  (no port needed; the fallback when a tunnel cannot be made).
* :func:`open_ssh` — a run on a cluster reached through its login node:
  reads ``monitor.json`` over SSH, opens ``ssh -L <local>:<node>:<port> login``
  (the **login node** forwards to the compute node — no SSH into the node) and
  connects; falls back to :class:`SshFileSource`.
* :func:`open_local` — a local run directory: the live server if it answers,
  else ``status.json``.
"""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence

from runmon.server import INFO_FILE, PROTOCOL_VERSION

#: Record separator between streamed status.json copies (SshFileSource).
_RS = "\x1e"


class FileSource:
    """Snapshots from ``<run_dir>/status.json``."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.label = f"file {run_dir}"
        self.state = "file"
        self._last: Optional[Dict[str, Any]] = None

    def snapshot(self) -> Optional[Dict[str, Any]]:
        try:
            with open(os.path.join(self.run_dir, "status.json"), encoding="utf-8") as f:
                self._last = json.load(f)
        except (OSError, ValueError):
            pass                                   # missing or mid-replace: keep the last one
        if self._last and self._last.get("run", {}).get("stopped"):
            self.state = "run ended"
        return self._last

    def close(self) -> None:
        pass


class SocketSource:
    """Live snapshots + events from a :class:`~runmon.server.MonitorServer`.

    :param host: Server host (or the local end of a tunnel).
    :param port: Server port.
    :param token: The run's token (from ``monitor.json``).
    :param label: Text shown as the data source.
    :param reconnect: Keep reconnecting after a dropped connection (until ``bye``).
    """

    def __init__(self, host: str, port: int, token: str, label: Optional[str] = None,
                 reconnect: bool = True):
        self.host, self.port, self.token = host, int(port), token
        self.label = label or f"live {host}:{port}"
        self.reconnect = reconnect
        self.state = "connecting"
        self.run: Optional[Dict[str, Any]] = None
        self.bye: Optional[str] = None
        self._snap: Optional[Dict[str, Any]] = None
        self._events: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._lock = threading.Lock()
        self._sock: Optional[socket.socket] = None
        self._closed = False
        self._thread: Optional[threading.Thread] = None

    # -- connection ---------------------------------------------------------

    def _handshake(self, timeout: float) -> socket.socket:
        sock = socket.create_connection((self.host, self.port), timeout=timeout)
        sock.settimeout(timeout)
        sock.sendall((json.dumps({"type": "hello", "v": PROTOCOL_VERSION, "token": self.token}) + "\n").encode())
        f = sock.makefile("rb")
        line = f.readline()
        msg = json.loads(line) if line else {}
        if msg.get("type") != "hello":
            sock.close()
            raise ConnectionError(msg.get("error") or "no hello from server")
        self.run = msg.get("run")
        sock.settimeout(None)
        self._file = f
        return sock

    def start(self, timeout: float = 10.0) -> bool:
        """Connect and authenticate now; ``True`` on success (then reads in the background)."""
        try:
            self._sock = self._handshake(timeout)
        except (OSError, ValueError, ConnectionError) as e:
            self.state = f"cannot connect: {e}"
            return False
        self.state = "live"
        self._thread = threading.Thread(target=self._reader, name="runmon-client", daemon=True)
        self._thread.start()
        return True

    def _reader(self) -> None:
        backoff = 1.0
        while not self._closed:
            try:
                for raw in self._file:
                    self._on_message(json.loads(raw))
                    backoff = 1.0
                    if self.bye is not None:
                        return
            except (OSError, ValueError):
                pass
            if self._closed or self.bye is not None or not self.reconnect:
                if self.bye is None and not self._closed:
                    self.state = "disconnected"
                return
            self.state = "reconnecting"
            time.sleep(backoff)
            backoff = min(30.0, backoff * 2)
            try:
                self._sock = self._handshake(10.0)
                self.state = "live"
            except (OSError, ValueError, ConnectionError):
                continue

    def _on_message(self, msg: Dict[str, Any]) -> None:
        kind = msg.get("type")
        with self._lock:
            if kind == "snapshot":
                self._snap = msg
                self._events = deque(msg.get("events") or [], maxlen=200)
            elif kind == "event":
                self._events.append(msg.get("event") or {})
            elif kind == "bye":
                self.bye = str(msg.get("reason", ""))
                self.state = f"run ended ({self.bye})"

    def snapshot(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._snap is None:
                return None
            snap = dict(self._snap)
            snap["events"] = list(self._events)       # newest events, even between snapshots
            if self.bye is not None:
                snap["run"] = dict(snap.get("run") or {}, stopped=True)
            return snap

    def close(self) -> None:
        self._closed = True
        try:
            if self._sock is not None:
                self._sock.shutdown(socket.SHUT_RDWR)
                self._sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# SSH (cluster login node)
# ---------------------------------------------------------------------------

def remote_path(path: str) -> str:
    """Shell expression for a remote path, keeping ``~`` expansion but quoting the rest."""
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def _ssh_base(ssh: str, target: str, opts: Sequence[str]) -> List[str]:
    return [ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", *opts, target]


def read_remote_info(target: str, run_dir: str, *, ssh: str = "ssh", opts: Sequence[str] = (),
                     runner=subprocess.run) -> Optional[Dict[str, Any]]:
    """``monitor.json`` of a remote run directory (``latest`` links resolved), or ``None``."""
    cmd = f'cat "$(readlink -f {remote_path(run_dir)})/{INFO_FILE}"'
    try:
        out = runner(_ssh_base(ssh, target, opts) + [cmd], capture_output=True, text=True, timeout=45)
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(out, "returncode", 1) != 0:
        return None
    try:
        return json.loads(out.stdout)
    except ValueError:
        return None


def free_local_port() -> int:
    """A currently free TCP port on 127.0.0.1."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tunnel_command(target: str, node: str, port: int, local_port: int, *, ssh: str = "ssh",
                   opts: Sequence[str] = ()) -> List[str]:
    """``ssh -N -L 127.0.0.1:<local>:<node>:<port> <login>`` (the login node forwards)."""
    return [ssh, "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=15", *opts,
            "-L", f"127.0.0.1:{local_port}:{node}:{port}", target]


class _Tunneled(SocketSource):
    """A socket source that owns its ``ssh -L`` process."""

    def __init__(self, proc: subprocess.Popen, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._proc = proc

    def close(self) -> None:
        super().close()
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(5)
            except subprocess.TimeoutExpired:
                self._proc.kill()


class SshFileSource:
    """``status.json`` streamed over one SSH connection (every ``interval`` s)."""

    def __init__(self, target: str, run_dir: str, *, interval: float = 2.0, ssh: str = "ssh",
                 opts: Sequence[str] = ()):
        self.label = f"ssh {target}:{run_dir} (status file)"
        self.state = "connecting"
        self._last: Optional[Dict[str, Any]] = None
        loop = (f'd="$(readlink -f {remote_path(run_dir)})"; '
                f'while :; do cat "$d/status.json" 2>/dev/null; printf "\\n{_RS}\\n"; sleep {interval}; done')
        self._proc = subprocess.Popen(_ssh_base(ssh, target, opts) + [loop], stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._reader, name="runmon-ssh-file", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        buf: List[str] = []
        for line in self._proc.stdout:
            if line.rstrip("\r\n") == _RS:           # not strip(): Python counts \x1e as whitespace
                text = "".join(buf).strip()
                buf = []
                if text:
                    try:
                        self._last = json.loads(text)
                        self.state = "ssh stream"
                    except ValueError:
                        pass
            else:
                buf.append(line)
        self.state = "ssh stream closed"

    def snapshot(self) -> Optional[Dict[str, Any]]:
        if self._last and self._last.get("run", {}).get("stopped"):
            self.state = "run ended"
        return self._last

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()


def open_ssh(target: str, run_dir: str, *, ssh: str = "ssh", opts: Sequence[str] = (),
             wait: float = 20.0):
    """Best source for a run on a remote cluster (see module docstring)."""
    info = read_remote_info(target, run_dir, ssh=ssh, opts=opts)
    if info and not info.get("stopped") and info.get("port") and info.get("token"):
        node = info.get("host") or info.get("node")
        if node in ("127.0.0.1", "localhost"):
            node = "127.0.0.1"                    # server bound to the login node itself
        local = free_local_port()
        proc = subprocess.Popen(tunnel_command(target, node, info["port"], local, ssh=ssh, opts=opts),
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
        deadline = time.time() + wait
        while time.time() < deadline and proc.poll() is None:
            try:
                socket.create_connection(("127.0.0.1", local), timeout=1).close()
                break
            except OSError:
                time.sleep(0.3)
        if proc.poll() is None:
            src = _Tunneled(proc, "127.0.0.1", local, info["token"],
                            label=f"live {node}:{info['port']} via {target}")
            if src.start(timeout=wait):
                return src
            src.close()
        else:
            proc.wait()
    return SshFileSource(target, run_dir, ssh=ssh, opts=opts)


def open_local(run_dir: str):
    """Live server of a local run if it answers, else its ``status.json``."""
    try:
        with open(os.path.join(run_dir, INFO_FILE), encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError):
        info = None
    if info and not info.get("stopped"):
        host = info.get("host") or "127.0.0.1"
        for h in dict.fromkeys([host, "127.0.0.1"]):
            src = SocketSource(h, info["port"], info["token"], label=f"live {h}:{info['port']}")
            if src.start(timeout=3.0):
                return src
    return FileSource(run_dir)
