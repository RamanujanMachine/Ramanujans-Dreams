"""
Terminal view of a run's snapshot (``rich``).

:func:`render` is a pure function of one snapshot dict (what ``status.json``
holds and the server will send), so it is testable without a terminal and
shared by every data source.  :func:`watch_files` is the file-mode client: it
re-reads ``status.json`` in a run directory and redraws.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

STAGES = ("loading", "extraction", "analysis", "search", "postprocess")
LEVEL_STYLE = {"DEBUG": "dim", "INFO": "white", "WARNING": "yellow", "ERROR": "bold red", "CRITICAL": "bold red"}


def _dur(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    s = int(max(0, seconds))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d{h:02d}h"
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _bar(fraction: Optional[float], width: int = 20, estimated: bool = False) -> Text:
    if fraction is None:
        return Text("·" * width, style="dim")
    filled = int(round(fraction * width))
    t = Text("█" * filled, style="cyan" if not estimated else "magenta")
    t.append("░" * (width - filled), style="dim")
    t.append(f" {'~' if estimated else ' '}{100 * fraction:5.1f}%")
    return t


def _header(s: Dict[str, Any], source: str) -> Panel:
    run = s.get("run", {})
    t = Text()
    t.append(f"{run.get('id', '?')}", style="bold")
    t.append(f"  host {run.get('host', '?')}  pid {run.get('pid', '?')}  up {_dur(run.get('uptime'))}")
    age = s.get("sample_age")
    stale = (time.time() - s.get("ts", time.time()))
    t.append(f"\nsource {source}  snapshot age {_dur(stale)}", style="dim")
    if age is not None:
        t.append(f"  sample age {age:.1f}s", style="dim" if age < 10 else "bold yellow")
    if run.get("stopped"):
        t.append("   RUN ENDED", style="bold red")
    wall = s.get("walltime")
    if wall:
        left = wall.get("remaining")
        style = "green" if left > 3600 else "yellow" if left > 900 else "bold red"
        t.append(f"\nwalltime left {_dur(left)}", style=style)
        if wall.get("limit"):
            t.append(f" of {_dur(wall['limit'])}", style="dim")
        t.append(f" ({wall.get('source')})", style="dim")
    return Panel(t, title="Run", border_style="blue")


def _stages(s: Dict[str, Any]) -> Text:
    current = (s.get("stage") or "").lower()
    seen_names = {n.get("name", "").lower(): n for n in s.get("tasks", [])}
    t = Text()
    for name in STAGES:
        node = seen_names.get(name)
        if name == current:
            t.append(f" ▶ {name} ", style="bold black on cyan")
        elif node is not None and node.get("status") == "done":
            t.append(f" ✓ {name} ", style="green")
        elif node is not None and node.get("status") in ("failed", "stopped"):
            t.append(f" ✗ {name} ", style="red")
        else:
            t.append(f" · {name} ", style="dim")
    return t


def _progress(s: Dict[str, Any], max_rows: int = 14) -> Table:
    table = Table(box=None, expand=True, show_header=True, header_style="bold")
    table.add_column("task", ratio=3, no_wrap=True)
    table.add_column("progress", width=29, no_wrap=True)
    table.add_column("done", justify="right", width=11)
    table.add_column("ETA", justify="right", width=7)
    table.add_column("metrics", ratio=3)                 # wraps: metrics matter more than width
    rows: List[tuple] = []

    def walk(node, depth):
        if node.get("status") != "running" and depth > 0:
            return
        done, total = node.get("done"), node.get("total")
        count = f"{done:g}/{total:g}" if total else (f"{done:g}" if done else "")
        metrics = "  ".join(f"{k}={_fmt(v)}" for k, v in (node.get("metrics") or {}).items())
        rows.append((("  " * depth) + ("└ " if depth else "") + node.get("name", ""),
                     _bar(node.get("fraction"), estimated=node.get("estimated", False)),
                     count, _dur(node.get("eta")) if node.get("eta") is not None else "",
                     metrics))
        for child in node.get("children", []):
            walk(child, depth + 1)

    running = [n for n in s.get("tasks", []) if n.get("status") == "running"]
    for root in running or s.get("tasks", [])[-1:]:
        walk(root, 0)
    for r in rows[:max_rows]:
        table.add_row(*r)
    return table


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _resources(s: Dict[str, Any], max_procs: int = 12) -> Group:
    res = s.get("resources") or {}
    tot, alloc, mach = res.get("totals") or {}, res.get("allocation") or {}, res.get("machine") or {}
    head = Text()
    if tot:
        head.append(f"CPU {tot['cpu_percent']:.0f}% of {100 * alloc.get('cores', 1)}% "
                    f"({alloc.get('cores')} cores, {alloc.get('cores_source')})  ")
        head.append_text(_bar(min(1.0, tot.get("cpu_util", 0)), 15))
        head.append(f"\nRSS {tot['rss_mb']:.0f} MB")
        if alloc.get("memory_mb"):
            head.append(f" of {alloc['memory_mb']:.0f} MB  ")
            head.append_text(_bar(min(1.0, tot.get("mem_util") or 0), 15))
        head.append(f"\nmachine: {mach.get('cores')} cores, {mach.get('mem_available_mb')} / "
                    f"{mach.get('mem_total_mb')} MB available", style="dim")
    table = Table(box=None, expand=True, header_style="bold")
    for col, kw in (("pid", {"justify": "right"}), ("role", {}), ("cpu %", {"justify": "right"}),
                    ("RSS MB", {"justify": "right"}), ("state", {})):
        table.add_column(col, **kw)
    procs = sorted(res.get("processes", []), key=lambda p: (p["role"] != "main", -p["cpu"]))
    for p in procs[:max_procs]:
        cpu_style = "green" if p["cpu"] > 50 else "yellow" if p["cpu"] > 5 else "dim"
        table.add_row(str(p["pid"]), p["role"], Text(f"{p['cpu']:.0f}", style=cpu_style),
                      f"{p['rss_mb']:.0f}", p["status"])
    if len(procs) > max_procs:
        table.add_row("…", f"{len(procs) - max_procs} more", "", "", "")
    return Group(head, table)


def _pools(s: Dict[str, Any]) -> Table:
    table = Table(box=None, expand=True, header_style="bold")
    for col in ("pool", "workers", "in flight", "done", "deaths", "shrinks", "grows", "peak MB"):
        table.add_column(col, justify="right" if col != "pool" else "left")
    for p in s.get("pools", []):
        table.add_row(str(p.get("name")), f"{p.get('workers', '?')}/{p.get('requested', p.get('workers', '?'))}",
                      str(p.get("in_flight", "")), str(p.get("completed", "")),
                      Text(str(p.get("deaths", 0)), style="bold red" if p.get("deaths") else ""),
                      str(p.get("shrinks", "")), str(p.get("grows", "")),
                      f"{p['peak_mb']:.0f}" if p.get("peak_mb") else "")
    return table


def _events(s: Dict[str, Any], n: int = 10) -> Table:
    table = Table(box=None, expand=True, show_header=False)
    table.add_column("time", width=8)
    table.add_column("level", width=7)
    table.add_column("kind", width=16, no_wrap=True)
    table.add_column("message", ratio=1, no_wrap=True)
    evs = [e for e in s.get("events", []) if e.get("kind") != "heartbeat" and e.get("level") != "DEBUG"]
    for e in evs[-n:]:
        style = LEVEL_STYLE.get(e.get("level"), "white")
        table.add_row(time.strftime("%H:%M:%S", time.localtime(e.get("ts", 0))),
                      Text(e.get("level", ""), style=style), e.get("kind", ""),
                      Text(e.get("msg", ""), style=style))
    return table


def render(snapshot: Optional[Dict[str, Any]], source: str = "") -> Group:
    """Renderable for one snapshot (``None`` → waiting screen)."""
    if not snapshot:
        return Group(Panel(Text(f"waiting for a run status ({source}) …", style="dim"), title="runmon"))
    parts = [_header(snapshot, source), _stages(snapshot)]
    alerts = snapshot.get("alerts") or {}
    if alerts:
        parts.append(Panel(Text("\n".join(f"{k}: {v}" for k, v in alerts.items()), style="bold yellow"),
                           title="Alerts", border_style="yellow"))
    parts.append(Panel(_progress(snapshot), title=f"Progress — {snapshot.get('current') or ''}",
                       border_style="cyan"))
    parts.append(Panel(_resources(snapshot), title="Processes & memory", border_style="green"))
    if snapshot.get("pools"):
        parts.append(Panel(_pools(snapshot), title="Pools", border_style="green"))
    parts.append(Panel(_events(snapshot), title="Recent events", border_style="white"))
    return Group(*parts)


def read_status(run_dir: str) -> Optional[Dict[str, Any]]:
    """The run's latest ``status.json`` (``None`` if missing / mid-replace)."""
    try:
        with open(os.path.join(run_dir, "status.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def watch(source, refresh: float = 0.5, until_ended: bool = False) -> None:
    """Redraw from ``source`` (see :mod:`runmon.client.sources`) until Ctrl-C.

    :param until_ended: Return a few seconds after the run ended.
    """
    with Live(render(None, source.label), refresh_per_second=4, screen=True) as live:
        try:
            ended_at = None
            while True:
                snap = source.snapshot()
                live.update(render(snap, f"{source.label} [{source.state}]"))
                if snap and snap.get("run", {}).get("stopped"):
                    ended_at = ended_at or time.time()
                    if until_ended and time.time() - ended_at > 3:
                        break
                time.sleep(refresh)
        except KeyboardInterrupt:
            pass
        finally:
            source.close()


def watch_files(run_dir: str, refresh: float = 1.0, until_ended: bool = False) -> None:
    """File-mode client: redraw from ``run_dir/status.json`` until Ctrl-C."""
    from runmon.client.sources import FileSource
    watch(FileSource(run_dir), refresh, until_ended)
