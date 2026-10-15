"""Command line: ``python -m runmon {watch,demo} ...``"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="runmon", description="Live run monitor.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("watch", help="live view of a run (reads RUN_DIR/status.json)")
    w.add_argument("run_dir")
    w.add_argument("--refresh", type=float, default=1.0)

    d = sub.add_parser("demo", help="run a simulated run and watch it")
    d.add_argument("--dir", default=None, help="run directory (default: a new temp dir)")
    d.add_argument("--speed", type=float, default=1.0, help=">1 runs faster (default 1 ≈ 3 min)")
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--no-watch", action="store_true", help="only run the simulation")

    args = parser.parse_args(argv)
    if args.cmd == "watch":
        from runmon.client.tui import watch_files
        watch_files(args.run_dir, args.refresh)
        return 0

    run_dir = args.dir or tempfile.mkdtemp(prefix="runmon-demo-")
    if args.no_watch:
        from runmon.demo import run
        run(run_dir, args.speed, args.workers)
        return 0
    log = open(os.path.join(run_dir, "demo.stdout"), "w")
    proc = subprocess.Popen([sys.executable, "-m", "runmon", "demo", "--no-watch", "--dir", run_dir,
                             "--speed", str(args.speed), "--workers", str(args.workers)],
                            stdout=log, stderr=subprocess.STDOUT)
    from runmon.client.tui import watch_files
    try:
        watch_files(run_dir, 0.5, until_ended=True)
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.wait()
    print(f"run directory: {run_dir}\n  events: {os.path.join(run_dir, 'events.jsonl')}")
    return proc.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
