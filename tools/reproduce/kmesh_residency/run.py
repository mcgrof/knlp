# SPDX-License-Identifier: MIT
"""Select CPU metadata qualification or a controlled model capture."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(".config"))
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    config = {}
    if args.config.exists():
        for line in args.config.read_text().splitlines():
            if line.startswith("CONFIG_KNLP_KMESH_RESIDENCY_"):
                key, value = line.split("=", 1)
                config[key.removeprefix("CONFIG_KNLP_KMESH_RESIDENCY_")] = json.loads(
                    value
                )

    def value(key, default=""):
        return os.environ.get("KMESH_RESIDENCY_" + key, config.get(key, default))

    mode = value("MODE", "cpu")
    if mode not in ("cpu", "serving"):
        parser.error("mode must be cpu or serving")
    plan = {
        "mode": mode,
        "source": value("SOURCE"),
        "model": value("MODEL"),
        "output": value("OUTPUT", "results/kmesh-family-residency"),
        "gpu_execution": mode == "serving",
    }
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if not plan["source"] or (mode == "serving" and not plan["model"]):
        parser.error(
            "set KMESH_RESIDENCY_SOURCE and, for serving, KMESH_RESIDENCY_MODEL"
        )
    command = [
        sys.executable,
        "-m",
        "tools.reproduce.kmesh_residency." + mode,
        "--source",
        plan["source"],
        "--output",
        plan["output"],
    ]
    environment = os.environ.copy()
    if mode == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    else:
        command += ["--model", plan["model"]]
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
