"""Single-owner Stage A supervisor. GPU payloads always use production gpuq."""

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

EXPERIMENT = "depth-code-discovery"
TERMINAL = {"completed", "failed", "cancelled", "canceled"}


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
    assert len(found) <= 1, f"duplicate GPUQ payload for {jobid}"
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


class Supervisor:
    def __init__(self, root):
        self.root = Path(root)
        self.records = self.root / "records"
        self.events = self.records / "STAGE-A-EVENTS.jsonl"
        self.status = self.records / "STAGE-A-STATUS.json"
        self.code = self.root / "code"

    def record(self, **event):
        append(self.events, event)

    def history(self):
        return json.loads(
            command(["gpuq", "history", "--experiment", EXPERIMENT, "--json"])
        )

    def progress(self, **fields):
        prior = read(self.status) if self.status.exists() else {}
        atomic(
            self.status,
            {
                **prior,
                **fields,
                "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )

    def run_one(self, job):
        jobid = job["id"]
        history = self.history()
        found = matching_runs(history, jobid)
        events = (
            [json.loads(l) for l in self.events.read_text().splitlines()]
            if self.events.exists()
            else []
        )
        previous = [e for e in events if e.get("job_id") == jobid]
        if not found:
            assert (
                not previous and not (self.records / "runs" / jobid).exists()
            ), f"ambiguous previous attempt for {jobid}; inspect, never blindly resubmit"
            assert shutil.disk_usage(self.root).free >= 50 * 2**30, "disk watermark"
            self.record(event="submission_intent", job_id=jobid, stage=job["stage"])
            expected = max(
                2,
                math.ceil((job["steps"] * 0.5 + job["steps"] // 100 * 4 + 30) / 60),
            )
            output = command(
                [
                    "gpuq",
                    "submit",
                    "--experiment",
                    EXPERIMENT,
                    "--description",
                    jobid,
                    "--expected-minutes",
                    str(expected),
                    "--cwd",
                    str(self.code),
                    "--",
                    sys.executable,
                    "-u",
                    "-m",
                    "research.mo_variants.job",
                    "--root",
                    str(self.root),
                    "--job-id",
                    jobid,
                ]
            )
            found = matching_runs(self.history(), jobid)
            assert len(found) == 1, {"submission_output": output}
            run = found[0]
            self.record(
                event="submitted",
                job_id=jobid,
                stage=job["stage"],
                run_id=run["run_id"],
                run_dir=run["run_dir"],
            )
        else:
            run = found[0]
        self.progress(
            status="running",
            current_job=jobid,
            current_gpuq_run=run["run_id"],
            stage=job["stage"],
        )
        while run["state"] not in TERMINAL:
            time.sleep(5)
            run = json.loads(command(["gpuq", "show", run["run_id"]]))["run"]
        path = self.records / "runs" / jobid / "RESULT.json"
        if path.exists():
            result = read(path)
        else:
            result = {
                "status": "failed",
                "error": "GPU job ended without RESULT.json",
                "completed_step": 0,
            }
        if not any(e["event"] == "terminal" for e in previous):
            self.record(
                event="terminal",
                job_id=jobid,
                stage=job["stage"],
                run_id=run["run_id"],
                queue_state=run["state"],
                result_status=result["status"],
                completed_step=result.get("completed_step", 0),
                error=result.get("error"),
                result_sha256=sha(path) if path.exists() else None,
                started_at=run["started_at"],
                ended_at=run["ended_at"],
                reservation_seconds=(
                    (
                        dt.datetime.fromisoformat(run["ended_at"])
                        - dt.datetime.fromisoformat(run["started_at"])
                    ).total_seconds()
                    if run["started_at"] and run["ended_at"]
                    else 0
                ),
                training_seconds=result.get("training_seconds", 0),
                scheduled_eval_seconds=result.get("scheduled_eval_seconds", 0),
                gpuq_telemetry_summary=str(
                    Path(run["run_dir"]) / "telemetry/summary.json"
                ),
            )
        command(["gpuq", "pin", run["run_id"]]) if not run["pinned"] else None
        assert (run["state"] == "completed" and result["status"] == "complete") or (
            run["state"] == "failed" and allowed_failure(job, result)
        ), f"stop: {jobid}: {result.get('error', run['state'])}"
        if result["status"] == "complete":
            target = self.records / "numerics" / (jobid + ".json")
            if not target.exists():
                self.progress(stage="CPU routing diagnostics", current_gpuq_run=None)
                before = time.perf_counter()
                command(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "research.mo_variants.diagnostics",
                        "--root",
                        str(self.root),
                        "--job-id",
                        jobid,
                    ],
                    cwd=self.code,
                )
                self.record(
                    event="cpu_diagnostics",
                    job_id=jobid,
                    seconds=time.perf_counter() - before,
                    path=str(target),
                    sha256=sha(target),
                )
            assert read(target)["status"] == "complete"
            if job["stage"] == "A-discovery":
                target = self.records / "quantization" / (jobid + ".json")
                if not target.exists():
                    self.progress(
                        stage="CPU endpoint quantization", current_gpuq_run=None
                    )
                    before = time.perf_counter()
                    command(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "research.mo_variants.quantize",
                            "--root",
                            str(self.root),
                            "--job-id",
                            jobid,
                        ],
                        cwd=self.code,
                    )
                    self.record(
                        event="cpu_quantization",
                        job_id=jobid,
                        seconds=time.perf_counter() - before,
                        path=str(target),
                        sha256=sha(target),
                    )
                assert (
                    read(target).get("status") == "complete"
                ), "partial precision record; retain and inspect"
        return result

    def run(self):
        self.records.mkdir(exist_ok=True)
        with (self.records / ".stage-a.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            authorization = read(self.records / "RESOURCE-DECISION.json")
            assert (
                authorization["status"] == "approved"
                and authorization["scope"] == "A"
                and authorization["user_message"]
            )
            source = read(self.records / "SOURCE.json")
            for name, expected in source["implemented_files"].items():
                assert sha(self.code / name) == expected, name
            qualification, pilots, discovery = ordered_jobs(
                read(self.records / "JOBS.json")
            )
            self.progress(
                status="running",
                authorized_scope="A",
                planned_gpu_jobs=256,
                stages_b_c_d_enabled=False,
            )
            try:
                for job in qualification:
                    self.run_one(job)
                results = {job["id"]: self.run_one(job) for job in pilots}
                selected = choose_lrs(pilots, results)
                dest = self.records / "SELECTED-LRS.json"
                if dest.exists():
                    assert read(dest) == selected, "frozen LR selection changed"
                else:
                    atomic(dest, selected)
                    self.record(
                        event="lr_selection_frozen",
                        sha256=sha(dest),
                        path=str(dest),
                    )
                for job in discovery:
                    if selected[job["variant"]]["lr"] is None:
                        self.record(
                            event="missing_cell",
                            job_id=job["id"],
                            reason="all four LR pilots ineligible",
                        )
                    else:
                        self.run_one(job)
                self.progress(
                    stage="Stage A analysis",
                    current_job=None,
                    current_gpuq_run=None,
                )
                command(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "research.mo_variants.analyze",
                        "--root",
                        str(self.root),
                    ],
                    cwd=self.code,
                )
                self.record(
                    event="stage_a_complete",
                    results_sha256=sha(self.records / "STAGE-A-RESULTS.json"),
                )
                self.progress(
                    status="complete",
                    stage="Stage A complete; no automatic extension",
                    current_job=None,
                    current_gpuq_run=None,
                )
            except BaseException as exc:
                self.record(event="coordinator_stopped", error=repr(exc))
                self.progress(status="stopped", stop_reason=repr(exc))
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    Supervisor(args.root).run()
