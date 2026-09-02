"""Command line for the GPU lease: pause, resume, status, run.

    python -m rl.pace.ctl status
    python -m rl.pace.ctl pause [--reason TEXT] [--wait] [--timeout S]
    python -m rl.pace.ctl resume
    python -m rl.pace.ctl run --name NAME -- CMD [ARGS...]

``pause`` sets the hold and signals live workers to checkpoint and
exit; with ``--wait`` it blocks until they are gone (after ``--timeout``
seconds it escalates to SIGTERM, then SIGKILL). ``resume`` clears the
hold; supervisors waiting on it relaunch their worker. ``run`` wraps a
worker command in a supervisor that restarts it after every yield.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

from rl.pace.lease import EXIT_YIELD, GpuLease, iter_status, run_supervised


def cmd_status(lease: GpuLease, args: argparse.Namespace) -> int:
    for line in iter_status(lease):
        print(line)
    return 0


def cmd_pause(lease: GpuLease, args: argparse.Namespace) -> int:
    record = lease.request_hold(reason=args.reason, by=args.by)
    print(f"hold set: {record['reason']} (by {record['by']})")
    n = lease.signal_workers()
    if n:
        print(f"signalled {n} worker(s) to checkpoint and exit")
    if not args.wait:
        return 0
    deadline = time.monotonic() + args.timeout
    while lease.gpu_workers():
        if time.monotonic() > deadline:
            break
        time.sleep(1.0)
    if lease.gpu_workers():
        print("workers still alive after timeout; sending SIGTERM")
        lease.signal_workers(signal.SIGTERM)
        lease.wait_until_idle(poll=1.0, timeout=30.0)
    if lease.gpu_workers():
        print("workers still alive; sending SIGKILL")
        lease.signal_workers(signal.SIGKILL)
        lease.wait_until_idle(poll=1.0, timeout=10.0)
    left = lease.gpu_workers()
    if left:
        print(f"ERROR: {len(left)} worker(s) survived", file=sys.stderr)
        return 1
    print("GPU is free")
    return 0


def cmd_resume(lease: GpuLease, args: argparse.Namespace) -> int:
    hold = lease.hold()
    if args.only_if_by and (hold is None or hold.get("by") != args.only_if_by):
        who = hold.get("by") if hold else "nobody"
        print(f"hold is held by {who}, not {args.only_if_by}; leaving it")
        return 0
    if lease.release_hold():
        print("hold released; supervisors will relaunch their workers")
    else:
        print("no hold was set")
    return 0


def cmd_run(lease: GpuLease, args: argparse.Namespace) -> int:
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("run: missing command after --", file=sys.stderr)
        return 2

    def launch() -> int:
        proc = subprocess.Popen(cmd, env=os.environ.copy())
        try:
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            return 130

    return run_supervised(
        args.name, launch, lease=lease, max_restarts=args.max_restarts
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rl.pace.ctl", description=__doc__)
    parser.add_argument("--state-dir", default=None, help="lease directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show the hold and live workers")

    p = sub.add_parser("pause", help="reserve the GPU for a person")
    p.add_argument("--reason", default="interactive use")
    p.add_argument("--by", default=os.environ.get("USER", ""))
    p.add_argument("--wait", action="store_true", help="block until workers exit")
    p.add_argument("--timeout", type=float, default=300.0)

    s = sub.add_parser("resume", help="release the GPU back to training")
    s.add_argument(
        "--only-if-by", default=None, help="release only a hold set by this name"
    )

    r = sub.add_parser("run", help="supervise a worker command")
    r.add_argument("--name", required=True)
    r.add_argument("--max-restarts", type=int, default=1000)
    r.add_argument("cmd", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)
    lease = GpuLease(args.state_dir)
    handler = {
        "status": cmd_status,
        "pause": cmd_pause,
        "resume": cmd_resume,
        "run": cmd_run,
    }[args.command]
    return handler(lease, args)


if __name__ == "__main__":
    sys.exit(main())
