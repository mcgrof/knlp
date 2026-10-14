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
    if (
        args
        and args[0] == sys.executable
        and any(
            module in args
            for module in (
                "research.mo_variants.stage_b_numerics",
                "research.mo_variants.quantize",
                "research.mo_variants.stage_b_analyze",
            )
        )
    ):
        return cpu_slot(args, cwd)
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


def cpu_slot(args, cwd):
    """ROCm CPU imports hold device handles; reserve them cooperatively too."""
    root = Path(args[args.index("--root") + 1])
    records = root / "records"
    phase = args[args.index("-m") + 1]
    jobid = args[args.index("--job-id") + 1] if "--job-id" in args else None
    payload = [
        "/usr/bin/env",
        "HIP_VISIBLE_DEVICES=",
        "CUDA_VISIBLE_DEVICES=",
        "ROCR_VISIBLE_DEVICES=",
        *args,
    ]
    history = json.loads(
        command(["gpuq", "history", "--experiment", EXPERIMENT, "--json"])
    )
    found = [run for run in history if run["command"] == payload]
    assert len(found) <= 1, "duplicate CPU reservation"
    if not found:
        submission = [
            "gpuq",
            "submit",
            "--experiment",
            EXPERIMENT,
            "--description",
            f"B CPU {phase.rsplit('.',1)[-1]} {jobid or 'final-report'}",
            "--expected-minutes",
            "2",
            "--cwd",
            str(cwd),
        ]
        if jobid:
            parent = read(records / "runs" / jobid / "RESULT.json")["metadata"][
                "gpuq_run_id"
            ]
            submission.extend(["--parent-run", parent])
        command([*submission, "--", *payload])
        history = json.loads(
            command(["gpuq", "history", "--experiment", EXPERIMENT, "--json"])
        )
        found = [run for run in history if run["command"] == payload]
        assert len(found) == 1
    run = found[0]
    while run["state"] not in TERMINAL:
        time.sleep(2)
        run = json.loads(command(["gpuq", "show", run["run_id"]]))["run"]
    path = records / "STAGE-B-CPU-SLOTS.json"
    slots = (
        read(path)
        if path.exists()
        else {
            "scope": "GPU-aware CPU diagnostics, precision screens and report; not extra training trajectories",
            "runs": [],
        }
    )
    if not any(x["run_id"] == run["run_id"] for x in slots["runs"]):
        slots["runs"].append(
            {
                "job_id": jobid,
                "phase": phase,
                "run_id": run["run_id"],
                "state": run["state"],
                "run_dir": run["run_dir"],
                "reservation_seconds": (
                    (
                        dt.datetime.fromisoformat(run["ended_at"])
                        - dt.datetime.fromisoformat(run["started_at"])
                    ).total_seconds()
                    if run["started_at"]
                    else 0
                ),
                "queue_wait_seconds": (
                    (
                        dt.datetime.fromisoformat(run["started_at"])
                        - dt.datetime.fromisoformat(run["submitted_at"])
                    ).total_seconds()
                    if run["started_at"]
                    else None
                ),
            }
        )
        atomic(path, slots)
    if not run["pinned"]:
        command(["gpuq", "pin", run["run_id"]])
    assert (
        run["state"] == "completed"
    ), f"CPU phase failed; retain logs and stop: {run['run_id']}"
    if jobid is None:
        # The analyzer finishes before its reservation receipt exists. Add the
        # complete auxiliary ledger using stdlib only, without rerunning it.
        cost = read(records / "STAGE-B-COST.json")
        cost["gpu_aware_cpu_slots"] = slots
        cost["cpu_slots_reservation_seconds"] = sum(
            x["reservation_seconds"] for x in slots["runs"]
        )
        cost["stage_b_training_and_cpu_slots_reserved_seconds"] = (
            cost["stage_b_reservation_seconds"] + cost["cpu_slots_reservation_seconds"]
        )
        atomic(records / "STAGE-B-COST.json", cost)
        results = read(records / "STAGE-B-RESULTS.json")
        results["cost"] = cost
        atomic(records / "STAGE-B-RESULTS.json", results)
    return (Path(run["run_dir"]) / "logs/stdout.log").read_text()


def cpu_timing(records, jobid, phase):
    runs = read(records / "STAGE-B-CPU-SLOTS.json")["runs"]
    return next(x for x in runs if x["job_id"] == jobid and x["phase"].endswith(phase))[
        "reservation_seconds"
    ]


def matching_runs(history, jobid):
    found = []
    for r in history:
        cmd = r.get("command", [])
        if "research.mo_variants.stage_b_job" in cmd and "--job-id" in cmd:
            if cmd[cmd.index("--job-id") + 1] == jobid:
                found.append(r)
    assert len(found) <= 1, f"duplicate GPUQ payload for {jobid}"
    return found


def allowed_failure(job, result):
    return False


class Supervisor:
    def __init__(self, root):
        self.root = Path(root)
        self.records = self.root / "records"
        self.events = self.records / "STAGE-B-EVENTS.jsonl"
        self.status = self.records / "STAGE-B-STATUS.json"
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
                math.ceil(((job["steps"] - 1000) * 0.5 + 10 * 4 + 30) / 60),
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
                    "research.mo_variants.stage_b_job",
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
                        "research.mo_variants.stage_b_numerics",
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
                    seconds=cpu_timing(self.records, jobid, "stage_b_numerics"),
                    path=str(target),
                    sha256=sha(target),
                )
            assert read(target)["status"] == "complete"
            if job["stage"] == "B":
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
                        seconds=cpu_timing(self.records, jobid, "quantize"),
                        path=str(target),
                        sha256=sha(target),
                    )
                assert (
                    read(target).get("status") == "complete"
                ), "partial precision record; retain and inspect"
        return result

    def run(self):
        with (self.records / ".stage-b.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            authorization = read(self.records / "STAGE-B-RESOURCE-DECISION.json")
            assert (
                authorization["scope"] == "B" and authorization["status"] == "approved"
            )
            # Keep this process standard-library-only. Importing the model
            # runtime can acquire KFD/render ownership before a queued payload.
            original = read(self.records / "SOURCE.json")
            extension = read(self.records / "STAGE-B-SOURCE.json")
            assert (
                sha(self.records / "SOURCE.json")
                == extension["stage_a_source_manifest_sha256"]
            )
            for name, expected in {
                **original["implemented_files"],
                **extension["files"],
            }.items():
                assert sha(self.code / name) == expected, name
            selection = read(self.records / "STAGE-B-SELECTION.json")
            assert len(selection["jobs"]) == 24
            assert (
                read(self.records / "STAGE-B-RESUME-FIXTURE.json")["status"]
                == "complete"
            )
            assert read(self.records / "NUMERICAL-REPLAY.json")["status"] == "complete"
            self.progress(
                status="running",
                authorized_scope="B",
                planned_gpu_jobs=24,
                stages_c_d_enabled=False,
            )
            try:
                for job in selection["jobs"]:
                    self.run_one(job)
                self.progress(
                    stage="Stage B analysis", current_job=None, current_gpuq_run=None
                )
                command(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "research.mo_variants.stage_b_analyze",
                        "--root",
                        str(self.root),
                    ],
                    cwd=self.code,
                )
                self.record(
                    event="stage_b_complete",
                    results_sha256=sha(self.records / "STAGE-B-RESULTS.json"),
                )
                self.progress(
                    status="complete",
                    stage="Stage B complete; C/D disabled",
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
    Supervisor(parser.parse_args().root).run()
