"""Tests for :mod:`runmon.lifecycle` — a final record however the run ends.

Each case runs a small monitored program in a subprocess and ends it a
different way:

  - normal exit → ``run_end`` reason ``exit``, exit code 0;
  - ``SIGTERM`` (``qdel`` / scheduler kill) → reason ``SIGTERM``, the program's
    ``finally`` block still runs, exit code 143;
  - ``SIGINT`` → reason ``SIGINT`` and the usual ``KeyboardInterrupt``;
  - a ``fork``-ed child that is terminated must not write the parent's
    ``run_end`` (only the parent's own exit is recorded).

Plus the package boundary: ``runmon`` never imports ``dreamer``.
"""

import ast
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

from runmon import events as ev

ROOT = pathlib.Path(__file__).resolve().parents[1]

PROGRAM = r"""
import os, sys, time
sys.path.insert(0, {root!r})
import runmon
run_dir = sys.argv[1]
runmon.start(run_dir, sample_s=0.05, drain_s=0.02, status_s=0.05)
mode = sys.argv[2]
try:
    with runmon.task("search", total=10, kind="stage"):
        if mode == "fork":
            pid = os.fork()
            if pid == 0:
                time.sleep(30)
                os._exit(0)
            time.sleep(0.3)   # CPython drops signals that arrive during its post-fork reset
            os.kill(pid, 15)
            os.waitpid(pid, 0)
        else:
            print("READY", flush=True)
            if mode == "wait":
                time.sleep(30)
finally:
    open(os.path.join(run_dir, "finally.txt"), "w").write("ran")
"""


def _run(tmp_path, mode, sig=None):
    run_dir = str(tmp_path / "run")
    script = tmp_path / "prog.py"
    script.write_text(PROGRAM.format(root=str(ROOT)))
    p = subprocess.Popen([sys.executable, str(script), run_dir, mode],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if sig is not None:
        assert p.stdout.readline().strip() == "READY"
        time.sleep(0.3)
        p.send_signal(sig)
    out, err = p.communicate(timeout=60)
    events = list(ev.iter_events(os.path.join(run_dir, "events.jsonl")))
    return p.returncode, events, run_dir, err


def _ends(events):
    return [e for e in events if e["kind"] == "run_end"]


def test_normal_exit(tmp_path):
    code, events, run_dir, err = _run(tmp_path, "quick")
    assert code == 0, err
    ends = _ends(events)
    assert len(ends) == 1 and ends[0]["f"]["reason"] == "exit"
    assert json.load(open(os.path.join(run_dir, "status.json")))["run"]["stopped"] is True


def test_sigterm(tmp_path):
    code, events, run_dir, err = _run(tmp_path, "wait", signal.SIGTERM)
    assert code == 128 + signal.SIGTERM, err
    ends = _ends(events)
    assert len(ends) == 1 and ends[0]["f"]["reason"] == "SIGTERM"
    assert open(os.path.join(run_dir, "finally.txt")).read() == "ran"
    assert json.load(open(os.path.join(run_dir, "status.json")))["run"]["stopped"] is True


def test_sigint(tmp_path):
    code, events, run_dir, err = _run(tmp_path, "wait", signal.SIGINT)
    assert code != 0 and "KeyboardInterrupt" in err
    ends = _ends(events)
    assert len(ends) == 1 and ends[0]["f"]["reason"] == "SIGINT"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_forked_child_does_not_finish_the_run(tmp_path):
    code, events, run_dir, err = _run(tmp_path, "fork")
    assert code == 0, err
    ends = _ends(events)
    assert len(ends) == 1 and ends[0]["f"]["reason"] == "exit"
    assert ends[0]["pid"] != 0


def test_runmon_never_imports_dreamer():
    for path in (ROOT / "runmon").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not any(n.split(".")[0] == "dreamer" for n in names), f"{path} imports {names}"
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); import runmon, runmon.hub, runmon.lifecycle; "
         "print(any(m.split('.')[0] == 'dreamer' for m in sys.modules))" % str(ROOT)],
        capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "False", out.stderr
