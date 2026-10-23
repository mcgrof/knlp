# SPDX-License-Identifier: MIT
"""Plan, run or verify the pinned KMesh CPU ownership controls."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tools.reproduce.kmesh_contracts.contract import profile, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "verify"))
    parser.add_argument("--config", type=Path, default=Path(".config"))
    args = parser.parse_args()
    settings = {"SOURCE": "", "OUTPUT": ""}
    if args.config.is_file():
        for line in args.config.read_text().splitlines():
            if line.startswith("CONFIG_KNLP_KMESH_CONTRACTS_"):
                key, value = line.split("=", 1)
                settings[key.removeprefix("CONFIG_KNLP_KMESH_CONTRACTS_")] = json.loads(
                    value
                )
    settings = {
        key: os.environ.get("KMESH_CONTRACTS_" + key, value)
        for key, value in settings.items()
    }
    if args.action == "plan":
        print(json.dumps(profile() | {"settings": settings}, indent=2))
        return
    if not settings["OUTPUT"]:
        parser.error("set KMESH_CONTRACTS_OUTPUT to an external result directory")
    output = Path(settings["OUTPUT"]).resolve()
    checkout = Path(__file__).resolve().parents[3]
    if output == checkout or checkout in output.parents:
        parser.error("experiment output must be outside the knlp source checkout")
    if args.action == "verify":
        print(json.dumps(verify(output), indent=2))
        return
    if not settings["SOURCE"]:
        parser.error("set KMESH_CONTRACTS_SOURCE to the pinned LMCache checkout")
    source = Path(settings["SOURCE"]).resolve()
    if output == source or source in output.parents:
        parser.error("experiment output must be outside the LMCache source checkout")
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        LMCACHE_TRACK_USAGE="false",
    )
    # CPU is a fallback in the pinned platform registry, not an explicit backend.
    env.pop("LMCACHE_DEVICE_BACKEND", None)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(source), str(checkout), env.get("PYTHONPATH", ""))
    )
    with (output / "execution.log").open("x") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.reproduce.kmesh_contracts.cell",
                "--source",
                str(source),
                "--output",
                str(output),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=180,
        )
    result = verify(output)
    (output / "verified.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
