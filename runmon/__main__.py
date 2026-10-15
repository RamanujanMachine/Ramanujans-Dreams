"""Command line: ``runmon {watch,query,demo} ...`` (also ``python -m runmon``)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile


def _watch_source(args):
    from runmon.client import sources
    if args.ssh:
        return sources.open_ssh(args.ssh, args.target or "runs/latest")
    target = args.target or "runs/latest"
    if not os.path.exists(target) and ":" in target:            # host:port
        host, _, port = target.rpartition(":")
        token = args.token or os.environ.get("RUNMON_TOKEN")
        if not token:
            raise SystemExit("host:port needs --token (or RUNMON_TOKEN); it is in the run's monitor.json")
        src = sources.SocketSource(host, int(port), token)
        if not src.start():
            raise SystemExit(f"cannot connect to {target}: {src.state}")
        return src
    if not os.path.isdir(target):
        raise SystemExit(f"no run directory {target!r}")
    return sources.open_local(target)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="runmon", description="Live run monitor.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("watch", help="live view of a run",
                       description="TARGET: a run directory (default runs/latest), host:port, or — "
                                   "with --ssh — a run directory on the remote cluster.")
    w.add_argument("target", nargs="?")
    w.add_argument("--ssh", metavar="USER@LOGIN",
                   help="reach the run through this SSH login host (tunnels to the compute node)")
    w.add_argument("--token", help="token for host:port targets")
    w.add_argument("--refresh", type=float, default=0.5)

    wb = sub.add_parser("web", help="live view of a run in your browser (local HTTP server)",
                        description="Same TARGET forms as `watch`. Serves http://127.0.0.1:PORT/ "
                                    "on this machine only.")
    wb.add_argument("target", nargs="?")
    wb.add_argument("--ssh", metavar="USER@LOGIN", help="reach the run through this SSH login host")
    wb.add_argument("--token", help="token for host:port targets")
    wb.add_argument("--port", type=int, default=8765, help="local port (next free one if taken)")
    wb.add_argument("--no-browser", action="store_true", help="do not open a browser tab")

    q = sub.add_parser("query", help="search a run's event log")
    q.add_argument("path", help="run directory or events.jsonl")
    q.add_argument("--ssh", metavar="USER@LOGIN", help="read the log on this SSH host")
    q.add_argument("--kind", action="append", default=[], help="event kind (repeatable)")
    q.add_argument("--level", help="minimum level (DEBUG / INFO / WARNING / ERROR)")
    q.add_argument("--grep", help="regular expression over message and fields")
    q.add_argument("--since", help="30m / 2h / 3d ago, epoch, or ISO time")
    q.add_argument("--until")
    q.add_argument("--pid", type=int)
    q.add_argument("--role", help="main / worker")
    q.add_argument("--task", help="substring of the task path (e.g. search/annealing)")
    q.add_argument("--json", action="store_true", help="print raw JSON events")
    q.add_argument("--limit", type=int, help="only the last N matches")

    d = sub.add_parser("demo", help="run a simulated run and watch it")
    d.add_argument("--dir", default=None, help="run directory (default: a new temp dir)")
    d.add_argument("--speed", type=float, default=1.0, help=">1 runs faster (default 1 ≈ 2 min)")
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--no-watch", action="store_true", help="only run the simulation")

    args = parser.parse_args(argv)
    if args.cmd == "watch":
        from runmon.client.tui import watch
        watch(_watch_source(args), args.refresh)
        return 0

    if args.cmd == "web":
        from runmon import query as qmod
        from runmon.web.server import serve
        source = _watch_source(args)
        run_dir = args.target or "runs/latest"
        if args.ssh:
            lines = lambda: qmod._lines_ssh(args.ssh, run_dir)              # noqa: E731
        elif os.path.isdir(run_dir):
            lines = lambda: qmod._lines_local(run_dir)                      # noqa: E731
        else:
            lines = None                                                    # host:port only
        serve(source, lines=lines, port=args.port, open_browser=not args.no_browser)
        return 0

    if args.cmd == "query":
        from runmon import query
        n = query.run(args.path, ssh_target=args.ssh, kinds=args.kind, level=args.level, grep=args.grep,
                      since=args.since, until=args.until, pid=args.pid, role=args.role, task=args.task,
                      as_json=args.json, limit=args.limit)
        return 0 if n else 1

    run_dir = args.dir or tempfile.mkdtemp(prefix="runmon-demo-")
    if args.no_watch:
        from runmon.demo import run
        run(run_dir, args.speed, args.workers)
        return 0
    log = open(os.path.join(run_dir, "demo.stdout"), "w")
    proc = subprocess.Popen([sys.executable, "-m", "runmon", "demo", "--no-watch", "--dir", run_dir,
                             "--speed", str(args.speed), "--workers", str(args.workers)],
                            stdout=log, stderr=subprocess.STDOUT)
    from runmon.client.sources import FileSource, open_local
    from runmon.client.tui import watch
    import time
    for _ in range(50):                          # wait for the run's server to come up
        if os.path.exists(os.path.join(run_dir, "monitor.json")) or proc.poll() is not None:
            break
        time.sleep(0.1)
    source = open_local(run_dir) if os.path.exists(os.path.join(run_dir, "monitor.json")) else FileSource(run_dir)
    try:
        watch(source, 0.5, until_ended=True)
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.wait()
    print(f"run directory: {run_dir}\n  events: {os.path.join(run_dir, 'events.jsonl')}")
    return proc.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
