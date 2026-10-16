"""
Hierarchical progress: run → stage → method → shard → inner loop.

A :class:`TaskNode` is opened for each unit of work and nested under the node
that was open when it started (per thread).  Its completed fraction is

* the explicit estimate, when the code supplied one (:meth:`TaskNode.estimate`
  — e.g. simulated annealing, whose length is not a fixed count);
* else ``(done + fraction of the open child) / total`` when ``total`` is known,
  so a stage with 12 shards, the 4th of which is 50 % done, reads 3.5 / 12;
* else unknown (``None``).

Closing a child does **not** advance its parent — the caller counts (e.g. a
``for shard in shards`` loop calls ``stage.advance()``), so wrapping an
existing counted loop never double-counts.

Cost model: updates (``advance``, ``set``, ``estimate``, ``metric``) only assign
attributes — no lock, no I/O, no allocation beyond the metrics dict — so they
are safe in tight loops.  Opening/closing a node takes a lock (structure
change).  :meth:`ProgressTree.snapshot` runs on the monitor thread and copies
what it reads; a racing update makes a snapshot at most one update stale.

Memory is bounded: each node keeps only its last :data:`KEEP_FINISHED`
finished children (a stage over 10 000 shards does not keep 10 000 nodes).
"""

from __future__ import annotations

import itertools
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

#: Finished children kept per node for display/history.
KEEP_FINISHED = 3

_ids = itertools.count(1)


class TaskNode:
    """One unit of work in the progress tree (see module docstring)."""

    __slots__ = ("id", "name", "kind", "parent", "children", "total", "done",
                 "_estimate", "metrics", "started", "finished", "status",
                 "updated", "_tree", "__weakref__")

    def __init__(self, tree: "ProgressTree", name: str, kind: str = "task",
                 total: Optional[float] = None, parent: Optional["TaskNode"] = None):
        self.id = next(_ids)
        self.name = str(name)
        self.kind = kind
        self.parent = parent
        self.children: List[TaskNode] = []
        self.total = total
        self.done = 0.0
        self._estimate: Optional[float] = None
        self.metrics: Dict[str, Any] = {}
        self.started = time.time()
        self.finished: Optional[float] = None
        self.status = "running"
        self.updated = self.started
        self._tree = tree

    # -- hot path (attribute assignment only) -------------------------------

    def advance(self, n: float = 1) -> None:
        """Mark ``n`` more units done."""
        self.done += n
        self.updated = time.time()
        self._tree.last_progress = self.updated

    def set(self, done: Optional[float] = None, total: Optional[float] = None) -> None:
        """Set the done count and/or the total."""
        if done is not None:
            self.done = done
        if total is not None:
            self.total = total
        self.updated = time.time()
        self._tree.last_progress = self.updated

    def estimate(self, fraction: Optional[float], **metrics: Any) -> None:
        """Supply an estimated completed fraction in ``[0, 1]`` (``None`` clears it) plus metrics."""
        self._estimate = None if fraction is None else min(1.0, max(0.0, float(fraction)))
        if metrics:
            self.metrics.update(metrics)
        self.updated = time.time()
        self._tree.last_progress = self.updated

    def metric(self, **metrics: Any) -> None:
        """Attach display metrics (temperature, best objective, ...) without changing progress."""
        self.metrics.update(metrics)
        self.updated = time.time()

    # -- lifecycle ----------------------------------------------------------

    def close(self, status: str = "done") -> None:
        """Finish this node (``done`` / ``failed`` / ``stopped``); idempotent."""
        self._tree._close(self, status)

    def __enter__(self) -> "TaskNode":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close("failed" if exc_type is not None and not issubclass(exc_type, GeneratorExit) else "done")
        return False

    # -- derived ------------------------------------------------------------

    @property
    def estimated(self) -> bool:
        """``True`` when the fraction is an estimate rather than a count."""
        return self._estimate is not None

    def fraction(self) -> Optional[float]:
        """Completed fraction in ``[0, 1]``, or ``None`` if unknown (see module docstring)."""
        if self.status == "done":
            return 1.0
        if self._estimate is not None:
            return self._estimate
        if self.total:
            active = 0.0
            for child in list(self.children):
                if child.status == "running":
                    f = child.fraction()
                    active = f if f is not None else 0.0
                    break
            return min(1.0, max(0.0, (self.done + active) / float(self.total)))
        return None

    def eta_seconds(self, now: Optional[float] = None) -> Optional[float]:
        """Remaining seconds from elapsed time and fraction (``None`` if unknown)."""
        f = self.fraction()
        if not f or f <= 0 or self.status != "running":
            return None
        elapsed = (now or time.time()) - self.started
        return elapsed * (1.0 - f) / f

    def path(self) -> str:
        """``stage/method/shard`` style path of names from the root."""
        parts, node = [], self
        while node is not None:
            parts.append(node.name)
            node = node.parent
        return "/".join(reversed(parts))

    def to_dict(self, now: Optional[float] = None, depth: int = 8) -> Dict[str, Any]:
        """JSON-ready snapshot of this node and (down to ``depth``) its children."""
        now = now or time.time()
        d = {
            "id": self.id, "name": self.name, "kind": self.kind, "status": self.status,
            "done": self.done, "total": self.total, "fraction": self.fraction(),
            "estimated": self.estimated, "eta": self.eta_seconds(now),
            "elapsed": (self.finished or now) - self.started,
            "started": self.started, "finished": self.finished,
            "metrics": dict(self.metrics),
        }
        if depth > 0:
            d["children"] = [c.to_dict(now, depth - 1) for c in list(self.children)]
        return d


class ProgressTree:
    """The run's progress tree with a per-thread stack of open nodes."""

    def __init__(self, on_change: Optional[Any] = None):
        """:param on_change: optional ``(event, node)`` callback, ``event`` in
        ``{"open", "close"}`` (the hub publishes stage start/end from it)."""
        self.on_change = on_change
        self.roots: List[TaskNode] = []
        self._lock = threading.RLock()   # re-entrant: shutdown may run in a signal handler
        self._local = threading.local()
        self.last_progress = time.time()

    def _stack(self) -> List[TaskNode]:
        st = getattr(self._local, "stack", None)
        if st is None:
            st = self._local.stack = []
        return st

    def current(self) -> Optional[TaskNode]:
        """The innermost open node of the calling thread."""
        st = self._stack()
        return st[-1] if st else None

    def open(self, name: str, total: Optional[float] = None, kind: str = "task") -> TaskNode:
        """Open a node under the calling thread's current node."""
        st = self._stack()
        parent = st[-1] if st else None
        node = TaskNode(self, name, kind, total, parent)
        with self._lock:
            (parent.children if parent is not None else self.roots).append(node)
        st.append(node)
        self.last_progress = node.started
        if self.on_change is not None:
            self.on_change("open", node)
        return node

    def _close(self, node: TaskNode, status: str) -> None:
        if node.status != "running":
            return
        node.status = status
        node.finished = node.updated = time.time()
        self.last_progress = node.finished
        st = self._stack()
        if node in st:                         # close dangling inner nodes too
            while st:
                top = st.pop()
                if top is node:
                    break
                if top.status == "running":
                    top.status, top.finished = "stopped", node.finished
        if self.on_change is not None:
            self.on_change("close", node)
        with self._lock:
            siblings = node.parent.children if node.parent is not None else self.roots
            finished = [c for c in siblings if c.status != "running"]
            for old in finished[:-KEEP_FINISHED]:
                siblings.remove(old)

    @contextmanager
    def task(self, name: str, total: Optional[float] = None, kind: str = "task") -> Iterator[TaskNode]:
        """``with tree.task("search", total=n): ...`` — open, yield, close."""
        node = self.open(name, total, kind)
        try:
            yield node
        except BaseException:
            node.close("failed")
            raise
        else:
            node.close("done")

    def active_path(self) -> str:
        """Path of the deepest running node (any thread), for event context."""
        with self._lock:
            roots = list(self.roots)
        best = ""
        for r in roots:
            node = r
            if node.status != "running":
                continue
            while True:
                running = [c for c in list(node.children) if c.status == "running"]
                if not running:
                    break
                node = running[-1]
            p = node.path()
            if len(p) > len(best):
                best = p
        return best

    def stage(self) -> Optional[str]:
        """Name of the running ``kind="stage"`` node, if any."""
        with self._lock:
            roots = list(self.roots)
        todo = list(roots)
        while todo:
            n = todo.pop()
            if n.status == "running" and n.kind == "stage":
                return n.name
            todo.extend(c for c in list(n.children) if c.status == "running")
        return None

    def snapshot(self, depth: int = 8) -> List[Dict[str, Any]]:
        """JSON-ready copy of the whole tree."""
        now = time.time()
        with self._lock:
            roots = list(self.roots)
        return [r.to_dict(now, depth) for r in roots]
