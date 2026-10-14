"""Utilities for study records, manifest ordering and learning-rate selection."""

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def append(path, event):
    event = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), **event}
    with Path(path).open("a") as f:
        f.write(json.dumps(event, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(event), flush=True)


def command(args, cwd=None):
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


def ordered_jobs(registry):
    jobs = [j for j in registry["jobs"] if j["stage"].startswith("A-")]
    assert len(jobs) == 256 and len({j["id"] for j in jobs}) == 256
    qualification = sorted(
        (j for j in jobs if j["stage"] == "A-qualification"),
        key=lambda j: j["variant"],
    )
    pilots = sorted(
        (j for j in jobs if j["stage"] == "A-pilot"),
        key=lambda j: (j["lr"], j["variant"]),
    )
    discovery = sorted(
        (j for j in jobs if j["stage"] == "A-discovery"),
        key=lambda j: (j["seed"], j["variant"]),
    )
    assert (len(qualification), len(pilots), len(discovery)) == (32, 128, 96)
    return qualification, pilots, discovery


def choose_lrs(jobs, results):
    selected = {}
    for variant in sorted({j["variant"] for j in jobs}):
        cells = []
        for j in sorted(
            (j for j in jobs if j["variant"] == variant), key=lambda j: j["lr"]
        ):
            r = results[j["id"]]
            values = {v["step"]: v["ce"] for v in r.get("validation", [])}
            eligible = r["status"] == "complete" and all(
                s in values and math.isfinite(values[s]) for s in (300, 400, 500)
            )
            cells.append(
                {
                    "job_id": j["id"],
                    "lr": j["lr"],
                    "status": r["status"],
                    "eligible": eligible,
                    "score": (
                        sum(values[s] for s in (300, 400, 500)) / 3
                        if eligible
                        else None
                    ),
                    "error": r.get("error"),
                }
            )
        valid = [c for c in cells if c["eligible"]]
        best = min(valid, key=lambda c: (c["score"], c["lr"])) if valid else None
        selected[variant] = {
            "lr": best["lr"] if best else None,
            "score": best["score"] if best else None,
            "status": "selected" if best else "no eligible LR",
            "pilots": cells,
            "boundary_best_on_grid": bool(best and best["lr"] in (0.0003, 0.01)),
            "rule": "minimum mean clean CE at 300/400/500; lower LR breaks exact ties",
        }
    return selected


def matching_runs(history, jobid):
    found = []
    for r in history:
        cmd = r.get("command", [])
        if "research.mo_variants.job" in cmd and "--job-id" in cmd:
            if cmd[cmd.index("--job-id") + 1] == jobid:
                found.append(r)
    assert len(found) <= 1, f"duplicate job payload for {jobid}"
    return found


def allowed_failure(job, result):
    # Only numerical LR divergence is recoverable. Infrastructure, invariant,
    # provenance, disk, cancellation, or missing-result failures stop the study.
    text = result.get("error", "").lower()
    return (
        job["stage"] == "A-pilot"
        and result["status"] == "failed"
        and "nonfinite" in text
        and "invariant" not in text
    )
