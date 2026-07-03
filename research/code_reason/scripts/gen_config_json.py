#!/usr/bin/env python3
"""Generate the code-reason runner config artifact from a Kconfig .config.

All experimental policy flows Kconfig -> .config -> this JSON -> the runner.
Nothing is hard-coded in Python constants; the runner reads only this file.

Usage:
    python3 -m research.code_reason.scripts.gen_config_json \
        --config .config --out build/code_reason/config.json
"""

import argparse
import json
import os
import re

CONFIG_LINE = re.compile(r"^(CONFIG_[A-Z0-9_]+)=(.*)$")
PREFIX = "CONFIG_CODE_REASON"


def _coerce(raw):
    raw = raw.strip()
    if raw == "y":
        return True
    if raw == "n" or raw == "":
        return False
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        return raw


def parse_config(path):
    """Return the CODE_REASON flag dict from a .config or defconfig file."""
    flags = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = CONFIG_LINE.match(line)
            if not m:
                # kconfig writes "# CONFIG_X is not set" for disabled bools
                m2 = re.match(r"^# (CONFIG_[A-Z0-9_]+) is not set$", line)
                if m2 and m2.group(1).startswith(PREFIX):
                    flags[m2.group(1)] = False
                continue
            key, val = m.group(1), m.group(2)
            if key.startswith(PREFIX):
                flags[key] = _coerce(val)
    return flags


def build_artifact(flags, source):
    return {
        "schema": "code_reason.run_config/v0",
        "source": source,
        "flags": dict(sorted(flags.items())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=".config")
    ap.add_argument("--out", default="build/code_reason/config.json")
    args = ap.parse_args()
    flags = parse_config(args.config)
    if not flags.get("CONFIG_CODE_REASON"):
        raise SystemExit(
            f"{args.config}: CONFIG_CODE_REASON is not enabled; "
            "load a code-reason defconfig first "
            "(make defconfig-code-reason-paper)"
        )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(build_artifact(flags, args.config), fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.out} ({len(flags)} CODE_REASON flags)")


if __name__ == "__main__":
    main()
