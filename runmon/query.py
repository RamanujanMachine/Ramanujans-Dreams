"""
Search a run's event log (``runmon query``).

Filters combine with AND; ``--kind`` may repeat (OR among kinds)::

    runmon query runs/latest --level WARNING
    runmon query runs/latest --kind worker_died --kind pool_shrink --since 2h
    runmon query runs/latest --grep "shard 3f2a" --task search/annealing
    runmon query ~/work/run/runs/latest --ssh uri.kasher@zeus.technion.ac.il --level ERROR
    runmon query events.jsonl --json | jq .

Heartbeats are hidden unless asked for (``--kind heartbeat``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from runmon.events import level_value


def parse_when(text: Optional[str], now: Optional[float] = None) -> Optional[float]:
    """Epoch seconds from ``30s`` / ``15m`` / ``2h`` / ``3d`` ago, an epoch number, or ISO-8601."""
    if not text:
        return None
    now = time.time() if now is None else now
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd])\s*", text)
    if m:
        return now - float(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        raise ValueError(f"cannot read time {text!r} (use 30m / 2h / 3d, epoch, or ISO-8601)")


def matches(e: Dict[str, Any], *, kinds: Sequence[str] = (), level: Optional[str] = None,
            pattern: Optional["re.Pattern"] = None, since: Optional[float] = None,
            until: Optional[float] = None, pid: Optional[int] = None, role: Optional[str] = None,
            task: Optional[str] = None) -> bool:
    """Does event ``e`` pass every given filter?"""
    kind = e.get("kind")
    if kinds:
        if kind not in kinds:
            return False
    elif kind == "heartbeat":
        return False
    if level and level_value(e.get("level", "INFO")) < level_value(level):
        return False
    ts = e.get("ts", 0)
    if since is not None and ts < since:
        return False
    if until is not None and ts > until:
        return False
    if pid is not None and e.get("pid") != pid:
        return False
    if role and e.get("role") != role:
        return False
    if task and task not in str((e.get("ctx") or {}).get("task", "")):
        return False
    if pattern is not None and not pattern.search(e.get("msg", "") + " " + json.dumps(e.get("f", {}), default=str)):
        return False
    return True


def format_event(e: Dict[str, Any]) -> str:
    """One readable line: time, level, kind, process, message, task context."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.get("ts", 0)))
    who = f"{e.get('role', '?')}:{e.get('pid', '?')}"
    task = (e.get("ctx") or {}).get("task")
    return (f"{ts}  {e.get('level', ''):<8} {e.get('kind', ''):<18} {who:<14} {e.get('msg', '')}"
            + (f"   [{task}]" if task else ""))


def _lines_local(path: str) -> Iterator[bytes]:
    if os.path.isdir(path):
        path = os.path.join(path, "events.jsonl")
    with open(path, "rb") as f:
        yield from f


def _lines_ssh(target: str, path: str, ssh: str = "ssh") -> Iterator[bytes]:
    from runmon.client.sources import remote_path
    cmd = (f'p="$(readlink -f {remote_path(path)})"; '
           f'if [ -d "$p" ]; then p="$p/events.jsonl"; fi; cat "$p"')
    proc = subprocess.Popen([ssh, "-o", "BatchMode=yes", target, cmd], stdout=subprocess.PIPE)
    try:
        yield from proc.stdout
    finally:
        proc.stdout.close()
        proc.wait()


def iter_matching(lines: Iterable[bytes], **filters: Any) -> Iterator[Dict[str, Any]]:
    """Parse JSON lines and yield the events passing ``filters`` (see :func:`matches`)."""
    for raw in lines:
        if not raw.strip():
            continue
        try:
            e = json.loads(raw)
        except ValueError:
            continue
        if matches(e, **filters):
            yield e


def run(path: str, *, ssh_target: Optional[str] = None, kinds: Sequence[str] = (),
        level: Optional[str] = None, grep: Optional[str] = None, since: Optional[str] = None,
        until: Optional[str] = None, pid: Optional[int] = None, role: Optional[str] = None,
        task: Optional[str] = None, as_json: bool = False, limit: Optional[int] = None,
        out=None) -> int:
    """CLI body: print matching events; returns how many matched."""
    import sys
    out = out or sys.stdout
    lines = _lines_ssh(ssh_target, path) if ssh_target else _lines_local(path)
    found: List[Dict[str, Any]] = []
    for e in iter_matching(lines, kinds=list(kinds), level=level,
                           pattern=re.compile(grep, re.I) if grep else None,
                           since=parse_when(since), until=parse_when(until),
                           pid=pid, role=role, task=task):
        found.append(e)
        if limit is None and not as_json:
            out.write(format_event(e) + "\n")
    shown = found[-limit:] if limit else found
    if limit is not None or as_json:
        for e in shown:
            out.write((json.dumps(e, default=str) if as_json else format_event(e)) + "\n")
    return len(found)
