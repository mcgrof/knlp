"""Supervise fixed disjoint GPU seed blocks or verified CPU endpoints."""

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

from .stage_c_common import plan_guard, read, write


def worker(root, role, index):
    root = Path(root)
    r = root / "records"
    plan, auth = plan_guard(root)
    jobs = (
        plan["jobs"]
        if role == "cpu"
        else [j for j in plan["jobs"] if j["worker"] == index]
    )
    state = r / f"STAGE-C-WORKER-{role}-{index}.json"
    record = {
        "status": "running",
        "role": role,
        "index": index,
        "completed": [],
        "current": None,
        "pid": os.getpid(),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if state.exists():
        old = read(state)
        assert (
            role == "cpu" and old["status"] == "running"
        ), "Never relaunch a GPU worker or silently retry failed CPU work"
        record["completed"] = old["completed"]
    write(state, record)

    def command(module, jid):
        log = root / "logs" / f'{role}-{jid}-{module.rsplit(".",1)[-1]}.log'
        with log.open("ab") as f:
            subprocess.run(
                [sys.executable, "-m", module, "--root", str(root), "--job-id", jid],
                cwd=root / "code",
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )

    try:
        for job in jobs:
            jid = job["id"]
            if jid in record["completed"]:
                continue
            record["current"] = jid
            write(state, record)
            if role == "gpu":
                command("research.mo_variants.stage_c_job", jid)
            else:
                while True:
                    receipt = r / "export" / (jid + ".json")
                    if receipt.exists() and read(receipt)["verified"]:
                        break
                    status = r / "STAGE-C-GPU-STATUS.json"
                    if status.exists() and read(status)["status"] in (
                        "failed",
                        "stopped",
                    ):
                        raise RuntimeError("GPU execution stopped before " + jid)
                    time.sleep(20)
                for module, directory in (
                    ("stage_c_numerics", "numerics"),
                    ("stage_c_quantize", "quantization"),
                ):
                    path = r / directory / (jid + ".json")
                    if path.exists():
                        assert (
                            read(path)["status"] == "complete"
                        ), "Partial CPU record retained; manual review required"
                    else:
                        command("research.mo_variants." + module, jid)
            record["completed"].append(jid)
            record["current"] = None
            record["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            write(state, record)
        if role == "cpu":
            with (root / "logs/final-analysis.log").open("ab") as f:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "research.mo_variants.stage_c_analyze",
                        "--root",
                        str(root),
                    ],
                    cwd=root / "code",
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        record["status"] = "complete"
    except BaseException as e:
        record["status"] = "failed"
        record["error"] = repr(e)
        raise
    finally:
        record["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write(state, record)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--role", choices=("cpu", "gpu"), required=True)
    p.add_argument("--index", type=int, default=0)
    a = p.parse_args()
    worker(a.root, a.role, a.index)
