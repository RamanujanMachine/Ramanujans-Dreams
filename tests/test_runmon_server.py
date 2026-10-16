"""Tests for the live server (:mod:`runmon.server`) and the client sources.

Server (inside the hub thread):
  - ``monitor.json``: owner-only (0600), port + token; marked stopped at the end.
  - Hello/token handshake; wrong token and silent clients are dropped.
  - Snapshot on connect, then events pushed live (well before the next snapshot).
  - A client that never reads cannot stall the hub; its backlog stays bounded.
  - ``bye`` at the end; binding is 127.0.0.1 normally and all interfaces in a
    batch job (advertising the node name).

Sources:
  - ``open_local``: live server when it answers, ``status.json`` after the run.
  - SSH path end to end with a fake ``ssh`` (a local "login node": runs remote
    commands here and implements ``-L`` as a real port forwarder): reads
    ``monitor.json`` over SSH, tunnels, connects; falls back to streaming
    ``status.json`` over SSH when the run has no live server.
"""

import json
import os
import socket
import stat
import sys
import textwrap
import time

import pytest

from runmon import server as srv
from runmon.client import sources
from runmon.hub import Hub, HubConfig

FAST = dict(sample_s=0.05, drain_s=0.02, worker_tail_s=0.02, status_s=0.05, heartbeat_s=1.0,
            walltime_check_s=0.05)


def _wait(pred, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def hub(tmp_path):
    h = Hub(str(tmp_path / "run"), "srv-test", HubConfig(server=True, host="127.0.0.1",
                                                          snapshot_s=0.2, **FAST)).start()
    yield h
    h.stop("teardown")


def _info(h):
    return json.load(open(os.path.join(h.run_dir, srv.INFO_FILE)))


def test_info_file_is_private(hub):
    path = os.path.join(hub.run_dir, srv.INFO_FILE)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    info = _info(hub)
    assert info["port"] == hub.server.port and len(info["token"]) >= 24
    assert info["host"] == "127.0.0.1" and info["stopped"] is False


def test_snapshot_events_and_bye(tmp_path):
    h = Hub(str(tmp_path / "r"), "live", HubConfig(server=True, host="127.0.0.1",
                                                    snapshot_s=5.0, **FAST)).start()
    info = _info(h)
    src = sources.SocketSource("127.0.0.1", info["port"], info["token"])
    assert src.start(), src.state
    assert src.run["id"] == "live"
    assert _wait(lambda: src.snapshot() is not None)            # snapshot right after hello
    t0 = time.time()
    h.event("worker_died", "pushed live", "WARNING", pool="eval")
    assert _wait(lambda: any(e.get("kind") == "worker_died" for e in src.snapshot()["events"]), 2)
    assert time.time() - t0 < 2.0                               # not waiting for the 5 s snapshot
    assert _wait(lambda: h.snapshot()["monitor"]["clients"] == 1)
    h.stop("SIGTERM")
    assert _wait(lambda: src.bye == "SIGTERM")
    assert src.state.startswith("run ended")
    assert src.snapshot()["run"]["stopped"] is True
    assert any(e.get("kind") == "run_end" for e in src.snapshot()["events"])
    assert _info(h)["stopped"] is True and _info(h)["reason"] == "SIGTERM"
    src.close()


def test_wrong_token_rejected(hub):
    src = sources.SocketSource("127.0.0.1", hub.server.port, "not-the-token", reconnect=False)
    assert not src.start(timeout=3)
    assert "cannot connect" in src.state


def test_silent_client_dropped(hub, monkeypatch):
    monkeypatch.setattr(srv, "HELLO_TIMEOUT", 0.3)
    s = socket.create_connection(("127.0.0.1", hub.server.port))
    s.settimeout(5)
    assert s.recv(10) == b""                                    # closed by the server
    s.close()


def test_slow_client_cannot_stall_the_hub(hub, monkeypatch):
    monkeypatch.setattr(srv, "MAX_BUFFER", 64 * 1024)
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    s.connect(("127.0.0.1", hub.server.port))
    s.sendall((json.dumps({"type": "hello", "token": hub.server.token}) + "\n").encode())
    assert _wait(lambda: any(c.authed for c in hub.server.clients.values()))
    ticks = hub.ticks
    for i in range(3000):                                        # never read by the client
        hub.event("spam", "x" * 500, n=i)
    assert _wait(lambda: hub.ticks > ticks + 20, 10)            # the hub keeps ticking
    client = next(c for c in hub.server.clients.values() if c.authed)
    assert client.dropped > 0 and len(client.outbuf) <= 64 * 1024
    s.close()


def test_bind_in_batch_job(monkeypatch):
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert srv.resolve_bind("auto") == ("127.0.0.1", "127.0.0.1")
    monkeypatch.setenv("PBS_JOBID", "1.zeus-master")
    assert srv.resolve_bind("auto") == ("0.0.0.0", socket.gethostname())
    assert srv.resolve_bind("10.0.0.5") == ("10.0.0.5", "10.0.0.5")


def test_open_local_live_then_file(tmp_path):
    h = Hub(str(tmp_path / "r"), "loc", HubConfig(server=True, host="127.0.0.1", **FAST)).start()
    src = sources.open_local(h.run_dir)
    assert isinstance(src, sources.SocketSource) and src.state == "live"
    src.close()
    h.stop()
    after = sources.open_local(h.run_dir)
    assert isinstance(after, sources.FileSource)
    assert after.snapshot()["run"]["stopped"] is True and after.state == "run ended"


# ---------------------------------------------------------------------------
# SSH path with a fake ssh
# ---------------------------------------------------------------------------

FAKE_SSH = textwrap.dedent('''\
    #!{python}
    """Stand-in for ssh: this machine is the "login node"."""
    import os, socket, sys, threading
    args, rest, fwd, i = sys.argv[1:], [], None, 0
    while i < len(args):
        a = args[i]
        if a == "-o":
            i += 2; continue
        if a == "-N":
            i += 1; continue
        if a == "-L":
            fwd = args[i + 1]; i += 2; continue
        rest.append(a); i += 1
    with open(os.environ["FAKE_SSH_LOG"], "a") as log:
        log.write(" ".join(sys.argv[1:]) + "\\n")
    if fwd is None:
        os.execvp("sh", ["sh", "-c", " ".join(rest[1:])])
    lhost, lport, rhost, rport = fwd.split(":")
    lst = socket.socket(); lst.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lst.bind((lhost, int(lport))); lst.listen(8)
    def pipe(a, b):
        try:
            while True:
                d = a.recv(65536)
                if not d: break
                b.sendall(d)
        except OSError:
            pass
        finally:
            for s in (a, b):
                try: s.close()
                except OSError: pass
    while True:
        c, _ = lst.accept()
        r = socket.create_connection((rhost, int(rport)))
        threading.Thread(target=pipe, args=(c, r), daemon=True).start()
        threading.Thread(target=pipe, args=(r, c), daemon=True).start()
''')


@pytest.fixture
def fake_ssh(tmp_path, monkeypatch):
    path = tmp_path / "fake-ssh"
    path.write_text(FAKE_SSH.format(python=sys.executable))
    path.chmod(0o755)
    log = tmp_path / "ssh.log"
    monkeypatch.setenv("FAKE_SSH_LOG", str(log))
    return str(path), log


def test_remote_path_quoting():
    assert sources.remote_path("~/work/run dir") == '"$HOME"/\'work/run dir\''
    assert sources.remote_path("/abs/p") == "/abs/p"
    cmd = sources.tunnel_command("u@login", "n017", 42221, 47123)
    assert cmd[-3:] == ["-L", "127.0.0.1:47123:n017:42221", "u@login"] and "-N" in cmd


def test_open_ssh_tunnels_to_the_live_server(tmp_path, fake_ssh):
    ssh, log = fake_ssh
    h = Hub(str(tmp_path / "runs" / "r1"), "ssh-live",
            HubConfig(server=True, host="127.0.0.1", **FAST)).start()
    os.symlink("r1", tmp_path / "runs" / "latest")
    try:
        src = sources.open_ssh("user@login", str(tmp_path / "runs" / "latest"), ssh=ssh, wait=15)
        assert isinstance(src, sources.SocketSource) and src.state == "live", src.state
        assert "via user@login" in src.label
        assert _wait(lambda: src.snapshot() is not None)
        assert src.snapshot()["run"]["id"] == "ssh-live"
        src.close()
    finally:
        h.stop()
    calls = log.read_text()
    assert "monitor.json" in calls and "-L 127.0.0.1:" in calls


def test_open_ssh_falls_back_to_streaming_status(tmp_path, fake_ssh):
    ssh, _ = fake_ssh
    h = Hub(str(tmp_path / "r"), "ended", HubConfig(server=True, host="127.0.0.1", **FAST)).start()
    h.stop()                                                     # no live server any more
    src = sources.open_ssh("user@login", h.run_dir, ssh=ssh)
    try:
        assert isinstance(src, sources.SshFileSource)
        assert _wait(lambda: src.snapshot() is not None)
        assert src.snapshot()["run"]["id"] == "ended" and src.state == "run ended"
    finally:
        src.close()
