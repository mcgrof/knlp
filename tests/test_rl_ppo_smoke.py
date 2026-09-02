"""PPO smoke: learns on the simulator on CPU, checkpoints, yields, resumes."""

import csv
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from rl.pace.lease import EXIT_YIELD, GpuLease


def _run(args, env):
    return subprocess.run(
        [sys.executable, "-m", "rl.ppo", *args], env=env, capture_output=True, text=True
    )


def test_ppo_smoke_learns_then_yields_and_resumes(tmp_path):
    runs = tmp_path / "runs"
    lease_dir = tmp_path / "lease"
    env = {
        **os.environ,
        "PYTHONPATH": os.getcwd(),
        "KNLP_GPU_LEASE_DIR": str(lease_dir),
        "OMP_NUM_THREADS": "2",
    }
    common = [
        "--env",
        "sim:sim_flat",
        "--run-name",
        "smoke",
        "--runs-dir",
        str(runs),
        "--device",
        "cpu",
        "--num-envs",
        "4",
        "--num-steps",
        "64",
        "--total-timesteps",
        "5120",
        "--checkpoint-every",
        "1",
        "--max-seconds",
        "40",
        "--torch-threads",
        "2",
    ]
    # 1. first session: a hold is set up front, so the trainer must yield
    #    before its first update and leave a checkpoint
    GpuLease(lease_dir).request_hold("pytest")
    out = _run(common, env)
    assert out.returncode == EXIT_YIELD, out.stdout + out.stderr
    assert (runs / "smoke" / "checkpoint.pt").exists()
    GpuLease(lease_dir).release_hold()
    assert GpuLease(lease_dir).gpu_workers() == []

    # 2. resume and finish all updates
    out = _run(common + ["--resume"], env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "resumed smoke at update 0" in out.stdout
    assert (runs / "smoke" / "final_agent.pt").exists()
    rows = list(csv.DictReader(open(runs / "smoke" / "metrics.csv")))
    assert len(rows) == 5120 // (4 * 64)
    assert all(float(r["approx_kl"]) >= 0 for r in rows)
    assert int(rows[-1]["global_step"]) == 5120

    # 3. resuming a finished run is a no-op that exits cleanly
    out = _run(common + ["--resume"], env)
    assert out.returncode == 0, out.stdout + out.stderr
