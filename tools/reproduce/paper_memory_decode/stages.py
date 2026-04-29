"""Stage definitions and resume engine.

Each stage is identified by name, takes the run context, and writes
results to results/<run_id>/stages/<name>/.  A stage is considered done
when results/<run_id>/stages/<name>/DONE exists.
"""
from __future__ import annotations
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


# Profile → ordered stage list
PROFILE_STAGES: dict[str, list[str]] = {
    "decode": [
        "00_doctor",
        "01_fetch_repos",
        "02_build_flashinfer",
        "03_build_vllm",
        "04_build_lmcache",
        "05_flashinfer_standalone_gates",
        "06_vllm_writer_gate",
        "07_qwen25_fullstack_quality",
        "08_qwen25_smoke_throughput",
        "09_lmcache_codec_quality",
        "10_lmcache_split_tier_microbench",
        "11_report",
    ],
    "decode-sat": [
        "00_doctor",
        "01_fetch_repos",
        "02_build_flashinfer",
        "03_build_vllm",
        "sat_h100_sweep",
        "sat_hill_fit",
        "sat_b_vs_bt_model_comparison",
        "sat_figures",
        "11_report",
    ],
    "decode-full": [
        # superset; cross-GPU lanes are skipped with reason if hardware absent
        "00_doctor",
        "01_fetch_repos",
        "02_build_flashinfer",
        "03_build_vllm",
        "04_build_lmcache",
        "05_flashinfer_standalone_gates",
        "06_vllm_writer_gate",
        "07_qwen25_fullstack_quality",
        "08_qwen25_smoke_throughput",
        "09_lmcache_codec_quality",
        "10_lmcache_split_tier_microbench",
        "sat_h100_sweep",
        "sat_hill_fit",
        "sat_b_vs_bt_model_comparison",
        "static_calibration_battery",
        "niah_16k_32k",
        "turboquant_stress",
        "spec_decode_interaction",
        "large_model_lanes",
        "cross_gpu_lanes",
        "full_storage_grid",
        "11_report",
    ],
}


@dataclass
class StageResult:
    name: str
    status: str            # "passed", "failed", "skipped", "warned"
    reason: str = ""
    started_at: float = 0.0
    duration_s: float = 0.0
    metrics: dict = None  # type: ignore[assignment]


class StageContext:
    """Per-stage I/O bundle — passed to each stage callable."""

    def __init__(self, name: str, run_dir: Path, cfg, host, telemetry):
        self.name = name
        self.run_dir = run_dir
        self.stage_dir = run_dir / "stages" / name
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.host = host
        self.telemetry = telemetry
        self.metrics_path = self.stage_dir / "metrics.jsonl"
        self.stdout_path = self.stage_dir / "stdout.log"
        self.stderr_path = self.stage_dir / "stderr.log"
        self.done_path = self.stage_dir / "DONE"
        self.skip_path = self.stage_dir / "SKIPPED"

    def already_done(self) -> bool:
        return self.done_path.exists() or self.skip_path.exists()

    def mark_done(self, payload: dict | None = None) -> None:
        self.done_path.write_text(json.dumps(payload or {"status": "passed"}, indent=2))

    def mark_skipped(self, reason: str) -> None:
        self.skip_path.write_text(json.dumps({"status": "skipped", "reason": reason}, indent=2))

    def log_metric(self, name: str, value, **labels) -> None:
        self.telemetry.log_metric(f"{self.name}/{name}", value, **labels)
        with self.metrics_path.open("a") as f:
            f.write(json.dumps({
                "t": time.time(), "stage": self.name,
                "metric": name, "value": value, "labels": labels
            }, default=str) + "\n")

    def run_subprocess(self, cmd: list[str], cwd: str | None = None,
                       env: dict[str, str] | None = None,
                       extra_env: dict[str, str] | None = None,
                       timeout: int | None = None) -> int:
        """Run a subprocess, mirror stdout/stderr to log files, return rc."""
        merged = dict(os.environ)
        if env is not None:
            merged = dict(env)
        if extra_env:
            merged.update(extra_env)
        with self.stdout_path.open("a") as out, self.stderr_path.open("a") as err:
            out.write(f"\n+ {' '.join(cmd)}\n")
            out.flush()
            try:
                proc = subprocess.run(
                    cmd, cwd=cwd, env=merged, stdout=out, stderr=err,
                    timeout=timeout
                )
                return proc.returncode
            except subprocess.TimeoutExpired:
                err.write(f"TIMEOUT after {timeout}s: {' '.join(cmd)}\n")
                return 124


# Stage registry — each entry is a callable taking StageContext and
# returning StageResult.  A stage that is "not yet implemented" returns
# status="skipped" with reason="not implemented in this scaffold".

def _not_yet(ctx: StageContext) -> StageResult:
    ctx.mark_skipped("scaffold: stage script not yet implemented")
    return StageResult(
        name=ctx.name, status="skipped",
        reason="scaffold: stage script not yet implemented",
    )


# Each stage registered here can be replaced as we wire scripts in.
# The doctor and report stages are inline.

def stage_doctor(ctx: StageContext) -> StageResult:
    from . import doctor as _doctor
    issues = _doctor.run_checks(ctx.cfg, ctx.host)
    ctx.log_metric("issue_count", len(issues))
    if issues:
        with ctx.stderr_path.open("a") as f:
            for i in issues:
                f.write(f"DOCTOR: {i}\n")
        return StageResult(name=ctx.name, status="failed",
                           reason="; ".join(issues[:3]))
    return StageResult(name=ctx.name, status="passed")


REGISTRY: dict[str, Callable[[StageContext], StageResult]] = {
    "00_doctor": stage_doctor,
    # Stages 01–11 and the saturation/full ones default to scaffold-skip
    # until each is wired up.  They show up in the report as 'skipped'
    # rather than disappearing silently.
}


def get_callable(name: str) -> Callable[[StageContext], StageResult]:
    return REGISTRY.get(name, _not_yet)
