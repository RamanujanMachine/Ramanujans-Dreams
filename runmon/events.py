"""
Structured events and the append-only JSON-Lines event log.

Every event is one JSON object on one line (``events.jsonl``), so the log is
greppable with ``grep``/``jq`` and replayable by the monitor client::

    {"v": 1, "ts": 1760542800.12, "seq": 17, "pid": 4242, "role": "main",
     "level": "WARNING", "kind": "worker_died", "msg": "...",
     "ctx": {"task": "search/annealing/shard 3"}, "f": {"pool": "eval", "workers": 8}}

Keys
----
``v``      schema version (:data:`SCHEMA_VERSION`).
``ts``     wall-clock time (``time.time()``).
``seq``    per-process sequence number (orders events of one process).
``pid`` / ``role``  emitting process and its role (``main`` / ``worker`` / ...).
``level``  ``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR`` / ``CRITICAL``.
``kind``   machine-readable type: ``log`` for ordinary log lines, otherwise a
           named event (``stage_start``, ``pool_created``, ``worker_died``,
           ``heartbeat``, ``walltime_warning``, ``run_end`` ...).
``msg``    human-readable text.
``ctx``    where the run was (filled in by the hub: the active task path).
``f``      free-form structured fields of this event.

Durability: :class:`EventWriter` flushes every line it writes (an OS-level
write — survives a crash of the process, not of the machine) and offers
:meth:`EventWriter.sync` (``fsync``) for the moments that matter (shutdown,
the walltime grace period).  Workers write their **own** files
(``events.w<pid>.jsonl``) — no shared lock, so a worker killed mid-write can
never block another process; :class:`FileTailer` reads them incrementally and
tolerates a torn last line.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

SCHEMA_VERSION = 1

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

_seq = itertools.count()
_role = "main"


def set_role(role: str) -> None:
    """Set the role stamped on events from this process (``main``, ``worker``, ...)."""
    global _role
    _role = role


def level_name(level: Any) -> str:
    """Normalise a level (``logging`` int, name, or custom level) to one of :data:`LEVELS`."""
    if isinstance(level, str):
        up = level.upper()
        return up if up in LEVELS else "INFO"
    try:
        n = int(level)
    except (TypeError, ValueError):
        return "INFO"
    if n >= logging.CRITICAL:
        return "CRITICAL"
    if n >= logging.ERROR:
        return "ERROR"
    if n >= logging.WARNING:
        return "WARNING"
    if n >= logging.INFO:
        return "INFO"
    return "DEBUG"


def level_value(name: str) -> int:
    """``logging`` integer for a level name (unknown → INFO)."""
    return getattr(logging, str(name).upper(), logging.INFO)


def make_event(kind: str, msg: str = "", level: Any = "INFO", **fields: Any) -> Dict[str, Any]:
    """Build an event dict (see module docstring); ``fields`` go under ``f``."""
    return {
        "v": SCHEMA_VERSION,
        "ts": time.time(),
        "seq": next(_seq),
        "pid": os.getpid(),
        "role": _role,
        "level": level_name(level),
        "kind": kind,
        "msg": str(msg),
        "ctx": {},
        "f": fields,
    }


def dumps(event: Dict[str, Any]) -> str:
    """One JSON line (no newline); values that are not JSON become ``str``."""
    return json.dumps(event, default=str, separators=(",", ":"), ensure_ascii=False)


class EventWriter:
    """Append-only JSONL file; thread-safe; every write is flushed to the OS.

    :param path: File to append to (parent directories are created).
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._lock = threading.RLock()   # re-entrant: shutdown may run in a signal handler
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, event: Dict[str, Any]) -> None:
        """Append one event and flush it."""
        self.write_many([event])

    def write_many(self, events: List[Dict[str, Any]]) -> None:
        """Append several events with one flush."""
        if not events:
            return
        text = "".join(dumps(e) + "\n" for e in events)
        with self._lock:
            if self._f.closed:
                return
            self._f.write(text)
            self._f.flush()

    def sync(self) -> None:
        """Force the file to stable storage (``fsync``)."""
        with self._lock:
            if not self._f.closed:
                self._f.flush()
                try:
                    os.fsync(self._f.fileno())
                except OSError:
                    pass

    def close(self) -> None:
        """Flush, sync and close (idempotent)."""
        self.sync()
        with self._lock:
            if not self._f.closed:
                self._f.close()


class FileTailer:
    """Incrementally read complete JSON lines appended to a file.

    Keeps the byte offset of the last complete line; a partial last line (a
    writer mid-write, or killed mid-write) is left for the next call.  Lines
    that are not valid JSON are skipped.
    """

    def __init__(self, path: str):
        self.path = path
        self._offset = 0
        self._partial = b""

    def read_new(self, max_bytes: int = 4 << 20) -> List[Dict[str, Any]]:
        """Events appended since the last call (at most ``max_bytes`` read per call)."""
        try:
            with open(self.path, "rb") as f:
                f.seek(self._offset)
                chunk = f.read(max_bytes)
        except OSError:
            return []
        if not chunk:
            return []
        self._offset += len(chunk)
        data = self._partial + chunk
        lines = data.split(b"\n")
        self._partial = lines.pop()          # incomplete (or empty) tail
        out = []
        for line in lines:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out


def iter_events(path: str) -> Iterator[Dict[str, Any]]:
    """All valid events of a JSONL file, in file order (for ``query`` / ``replay``)."""
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


class JsonlHandler(logging.Handler):
    """``logging`` handler that turns log records into events.

    Structured data rides on the record's ``extra``: ``event`` (the kind,
    default ``log``) and ``fields`` (a dict)::

        logger.warning("worker died", extra={"event": "worker_died", "fields": {"pool": "eval"}})

    :param sink: Callable receiving each event dict (e.g. ``Hub.publish`` or
        ``EventWriter.write``).
    """

    def __init__(self, sink: Callable[[Dict[str, Any]], None], level: int = logging.DEBUG):
        super().__init__(level)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            fields = dict(getattr(record, "fields", None) or {})
            fields.setdefault("src", f"{record.filename}:{record.funcName}")
            if record.exc_info:
                fields["exc"] = logging.Formatter().formatException(record.exc_info)
            event = make_event(
                getattr(record, "event", None) or "log",
                record.getMessage(),
                record.levelno,
                **fields,
            )
            event["ts"] = record.created
            self._sink(event)
        except Exception:  # a monitoring failure must never break the program
            self.handleError(record)
