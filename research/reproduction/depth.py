"""Execute the fixed TinyStories depth screen after operator qualification."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from research.reproduction.run import REPO, write


def run(root, config):
    destination = root / "depth-screen"
    destination.mkdir()
    plan = json.loads(
        (REPO / "research/reproduction/plans/depth-screen.json").read_text()
    )
    plan["frozen_before_outcomes"] = True
    write(destination / "PLAN.json", plan)
    write(destination / "QUALIFICATION.json", {"status": "pending"})
    checkpoints = (
        Path(config["checkpoint_dir"]).expanduser() / root.name / "depth-screen"
        if config["checkpoint_dir"]
        else destination / "checkpoints"
    )
    env = {
        **os.environ,
        "KNLP_SUPERVISED_SOURCE_JSON": json.dumps(
            {
                "code": "public checkout",
                "plan": str(destination / "PLAN.json"),
            }
        ),
        "KNLP_GPU_REMAINING_SECONDS": str(config["max_seconds"] or 604800),
        "KNLP_GPU_COMMAND_STARTED_MONOTONIC": str(time.monotonic()),
    }

    def command(module, *args, cpu=False):
        subprocess.run(
            [
                config["cpu_python"] if cpu else sys.executable,
                "-m",
                module,
                "--root",
                str(destination),
                *map(str, args),
            ],
            cwd=REPO,
            env=env,
            check=True,
        )

    for device in ("cpu", "cuda"):
        command(
            "research.tinystories_depth.qualify",
            "--device",
            device,
            cpu=device == "cpu",
        )
    shared = ["--data", root / "data", "--checkpoints", checkpoints]
    command("research.tinystories_depth.train", *shared, "--qualify")
    qualification = json.loads((destination / "QUALIFICATION.json").read_text())
    if qualification["status"] != "qualified":
        raise RuntimeError("Depth-screen training qualification failed")
    command("research.tinystories_depth.train", *shared)
    payloads = sorted(checkpoints.glob("diagnostic-*.pt"))
    command("research.tinystories_depth.diagnostics", "--actual", *payloads, cpu=True)
    command("research.tinystories_depth.quantize", "--data", root / "data", cpu=True)
    command("research.tinystories_depth.analyze", cpu=True)
    status = json.loads((destination / "STATUS.json").read_text())
    if status["status"] != "complete" or status["missing"]:
        raise RuntimeError("Depth screen is incomplete; see STATUS.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, json.loads(args.config_json.read_text()))
