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
    "PROFILE",
    "MODEL_EXECUTION",
    "STORAGE_ROUTE",
    "STORAGE_FILE",
    "TRIALS",
    "COMPARISON",
    "CHECKSUM_LIBRARY",
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
    parser.add_argument(
        "action",
        choices=("plan", "run", "verify", "controls", "storage-pair", "group-failure"),
    )
    parser.add_argument("--config", type=Path, default=Path(".config"))
    args = parser.parse_args()
    settings = dict.fromkeys(FIELDS, "")
    settings["ROUTE"] = "registered"
    settings["PROFILE"] = str(ROOT / "profile.json")
    settings["STORAGE_ROUTE"] = "cpu-l1"
    settings["MODEL_EXECUTION"] = "serial"
    settings["TRIALS"] = 7
    settings["COMPARISON"] = "storage"
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
    profile = json.loads(Path(settings["PROFILE"]).read_text())
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
    if args.action == "storage-pair":
        if any(not settings[k] for k in ("REFERENCE", "VERIFICATION")):
            parser.error(
                "storage pair requires independently verified historical reference paths"
            )
        argv = [
            sys.executable,
            str(ROOT / "storage_pair.py"),
            "--profile",
            settings["PROFILE"],
            "--trials",
            str(settings["TRIALS"]),
            "--historical-reference",
            settings["REFERENCE"],
            "--historical-verification",
            settings["VERIFICATION"],
            "--output-dir",
            str(output),
        ]
        argv += ["--comparison", settings["COMPARISON"]]
    else:
        argv = [
            sys.executable,
            str(ROOT / "run.py"),
            "--route",
            settings["ROUTE"],
            "--profile",
            settings["PROFILE"],
            "--storage-route",
            settings["STORAGE_ROUTE"],
            "--model-execution",
            settings["MODEL_EXECUTION"],
        ]
        if settings["STORAGE_FILE"]:
            argv += ["--storage-file", settings["STORAGE_FILE"]]
    if settings["CHECKSUM_LIBRARY"]:
        argv += ["--checksum-library", settings["CHECKSUM_LIBRARY"]]
    for key, option in names.items():
        if key != "OUTPUT":
            argv.extend(("--" + option, settings[key]))
    if args.action == "controls":
        if (
            settings["ROUTE"] != "registered"
            or settings["STORAGE_ROUTE"] != "cpu-l1"
            or settings["MODEL_EXECUTION"] != "serial"
            or any(
                not settings[k] for k in ("ORDINARY_RESULT", "ORDINARY_VERIFICATION")
            )
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
    elif args.action == "group-failure":
        if (
            settings["MODEL_EXECUTION"] != "collective-lockstep"
            or settings["ROUTE"] != "registered"
            or settings["STORAGE_ROUTE"] != "cpu-l1"
            or any(
                not settings[k]
                for k in (
                    "REFERENCE",
                    "VERIFICATION",
                    "ORDINARY_RESULT",
                    "ORDINARY_VERIFICATION",
                )
            )
        ):
            parser.error(
                "group member control requires registered CPU-L1 and verified ordinary/serial references"
            )
        subprocess.run(
            argv
            + [
                "--group-failure",
                "--output-dir",
                str(output),
                "--ordinary-result",
                settings["ORDINARY_RESULT"],
                "--ordinary-verification",
                settings["ORDINARY_VERIFICATION"],
                "--serial-result",
                str(Path(settings["REFERENCE"]) / "result.json"),
                "--serial-comparison",
                settings["VERIFICATION"],
            ],
            check=True,
        )
    elif args.action == "storage-pair":
        subprocess.run(argv, check=True)
    else:
        subprocess.run(argv + ["--output-dir", str(output)], check=True)


if __name__ == "__main__":
    main()
