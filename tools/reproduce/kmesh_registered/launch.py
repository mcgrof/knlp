# SPDX-License-Identifier: MIT
"""Plan, execute or verify the registered serial model replay from a defconfig."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FIELDS = (
    "ROUTE",
    "VLLM_SOURCE",
    "LMCACHE_SOURCE",
    "WHEEL_PACKAGE",
    "MODEL_DIR",
    "TRACE",
    "MANIFEST",
    "OUTPUT",
    "REFERENCE",
    "VERIFICATION",
    "ORDINARY_RESULT",
    "ORDINARY_VERIFICATION",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "verify", "controls"))
    parser.add_argument("--config", type=Path, default=Path(".config"))
    args = parser.parse_args()
    settings = dict.fromkeys(FIELDS, "")
    settings["ROUTE"] = "registered"
    prefix = "CONFIG_KNLP_KMESH_REGISTERED_"
    if args.config.is_file():
        for line in args.config.read_text().splitlines():
            if line.startswith(prefix) and "=" in line:
                key, value = line.split("=", 1)
                if key.removeprefix(prefix) in settings:
                    settings[key.removeprefix(prefix)] = json.loads(value)
    settings = {
        k: os.environ.get("KMESH_REGISTERED_" + k, v) for k, v in settings.items()
    }
    profile = json.loads((ROOT / "profile.json").read_text())
    if args.action == "plan":
        print(json.dumps({"profile": profile, "settings": settings}, indent=2))
        return
    if args.action == "verify":
        from tools.reproduce.kmesh_registered.verify import compare

        if any(not settings[k] for k in ("OUTPUT", "REFERENCE", "VERIFICATION")):
            parser.error("set OUTPUT, REFERENCE and VERIFICATION")
        result = compare(
            settings["OUTPUT"], settings["REFERENCE"], settings["VERIFICATION"]
        )
        path = Path(settings["OUTPUT"]) / "comparison.json"
        with path.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit("registered replay differs from the serial reference")
        return
    names = {
        "VLLM_SOURCE": "vllm-source",
        "LMCACHE_SOURCE": "lmcache-source",
        "WHEEL_PACKAGE": "wheel-vllm-package",
        "MODEL_DIR": "model-dir",
        "TRACE": "trace",
        "MANIFEST": "manifest",
        "OUTPUT": "output-dir",
    }
    if any(not settings[k] for k in names):
        parser.error(
            "set all source, runtime, model, trace and output paths shown by plan"
        )
    output = Path(settings["OUTPUT"]).resolve()
    for source in (
        ROOT.parents[2],
        Path(settings["LMCACHE_SOURCE"]),
        Path(settings["VLLM_SOURCE"]),
    ):
        if output == source.resolve() or source.resolve() in output.parents:
            parser.error("capture output must remain outside source checkouts")
    argv = [sys.executable, str(ROOT / "run.py"), "--route", settings["ROUTE"]]
    for key, option in names.items():
        if key != "OUTPUT":
            argv.extend(("--" + option, settings[key]))
    if args.action == "controls":
        if settings["ROUTE"] != "registered" or any(
            not settings[k] for k in ("ORDINARY_RESULT", "ORDINARY_VERIFICATION")
        ):
            parser.error(
                "controls require registered route and verified ordinary result paths"
            )
        output.mkdir(parents=True, exist_ok=False)
        for mode in (
            "partial_miss",
            "corrupt",
            "evict",
            "post_enqueue",
            "transient_drain",
            "persistent_drain",
            "worker_drain",
        ):
            with (output / (mode + ".log")).open("x") as log:
                subprocess.run(
                    argv
                    + [
                        "--control",
                        mode,
                        "--output-dir",
                        str(output / mode),
                        "--ordinary-result",
                        settings["ORDINARY_RESULT"],
                        "--ordinary-verification",
                        settings["ORDINARY_VERIFICATION"],
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
    else:
        subprocess.run(argv + ["--output-dir", str(output)], check=True)


if __name__ == "__main__":
    main()
