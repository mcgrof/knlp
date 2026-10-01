# SPDX-License-Identifier: MIT
"""Create one isolated, source-pinned baseline environment (plan by default)."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .live import MINI_REVISION, SWEBENCH_REVISION
from .replay import EFFICIENTAGENT_REVISION

SOURCES = {
    "mini": ("https://github.com/SWE-agent/mini-swe-agent.git", MINI_REVISION),
    "efficientagent": (
        "https://github.com/KunmingSHAO/efficientagent_release.git",
        EFFICIENTAGENT_REVISION,
    ),
    "swebench": ("https://github.com/SWE-bench/SWE-bench.git", SWEBENCH_REVISION),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=SOURCES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    source, revision = SOURCES[args.component]
    checkout = out / "source"
    python = out / "venv/bin/python"
    install = str(checkout) + ("[test]" if args.component == "efficientagent" else "")
    # Pin the transport packages exercised by the local HTTP contract test.
    extra = ["litellm==1.103.2", "openai==2.54.0"] if args.component == "mini" else []
    commands = [
        ["git", "clone", "--no-checkout", source, str(checkout)],
        ["git", "-C", str(checkout), "checkout", "--detach", revision],
        [sys.executable, "-m", "venv", str(out / "venv")],
        [str(python), "-m", "pip", "install", "-e", install, *extra],
    ]
    plan = {
        "component": args.component,
        "revision": revision,
        "commands": commands,
        "gpu_dependencies": "not installed; reference serving stack is separate",
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    out.mkdir(parents=True)
    (out / "bootstrap.json").write_text(json.dumps(plan, indent=2) + "\n")
    with (out / "bootstrap.log").open("w") as log:
        for command in commands:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    with (out / "requirements-resolved.txt").open("w") as stream:
        subprocess.run(
            [str(python), "-m", "pip", "freeze", "--all"], check=True, stdout=stream
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
