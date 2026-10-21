# SPDX-License-Identifier: MIT
"""Select CPU qualification, controlled serving or a live search capture."""

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
    if mode not in ("cpu", "serving", "tot"):
        parser.error("mode must be cpu, serving or tot")
    plan = {
        "mode": mode,
        "source": value("SOURCE"),
        "model": value("MODEL"),
        "output": value("OUTPUT", "results/kmesh-family-residency"),
        "gpu_execution": mode != "cpu",
        "workload_source": value("WORKLOAD_SOURCE"),
    }
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if not plan["source"] or (mode != "cpu" and not plan["model"]):
        parser.error(
            "set KMESH_RESIDENCY_SOURCE and, for GPU modes, KMESH_RESIDENCY_MODEL"
        )
    if mode == "tot" and not plan["workload_source"]:
        parser.error("set KMESH_RESIDENCY_WORKLOAD_SOURCE for the pinned search")
    module = "tot_workload" if mode == "tot" else mode
    command = [
        sys.executable,
        "-m",
        "tools.reproduce.kmesh_residency." + module,
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
    if mode == "tot":
        command += ["--workload-source", plan["workload_source"]]
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
