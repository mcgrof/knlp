#!/usr/bin/env python3
"""Check a maintenance proposal's shape without executing its contents."""

import argparse
import json
import math
from pathlib import Path
import re

MAX_BYTES = 65536
RESOURCE_KEYS = {
    "kind",
    "alias",
    "grant_ref",
    "capabilities",
    "wall_seconds",
    "network",
}


def read_json(path):
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("input exceeds the proposal size limit")
    return json.loads(raw)


def matches(value, template):
    """Check required keys and scalar types against the shipped proposal."""
    if isinstance(template, dict):
        return (
            isinstance(value, dict)
            and value.keys() == template.keys()
            and all(matches(value[key], item) for key, item in template.items())
        )
    if isinstance(template, list):
        return isinstance(value, list) and all(
            isinstance(item, str) and item.strip() for item in value
        )
    if template is None:
        return value is None or isinstance(value, str) and bool(value.strip())
    if type(template) is int:
        return type(value) is int and value >= 0
    if type(template) is float:
        return (
            type(value) in (int, float)
            and value >= 0
            and (type(value) is int or math.isfinite(value))
        )
    return type(value) is type(template)


def validate(profile):
    template = read_json(Path(__file__).with_name("policy.json"))
    resources = profile.get("resources") if isinstance(profile, dict) else None
    if not isinstance(resources, list) or len(resources) > 32:
        return False
    copy = dict(profile, resources=[])
    if not matches(copy, template):
        return False
    if profile["schema_version"] != 1 or profile["status"] != "proposal":
        return False
    if (
        profile["schedule"]["enabled"]
        or profile["model_access"]["personal_account_fallback"]
    ):
        return False
    coordination = profile["coordination"]
    if (
        coordination["admission"]
        not in {"manual-single-worker", "external-coordinator"}
        or coordination["single_writer_per_branch"] is not True
        or coordination["stale_results"] != "reject"
        or profile["delegation"]["on_no_response"] != "pause-affected-task"
    ):
        return False
    revision = profile["repository"]["revision"]
    if revision is not None and not re.fullmatch(
        r"[0-9a-f]{40}|[0-9a-f]{64}", revision
    ):
        return False
    for item in resources:
        if not isinstance(item, dict) or item.keys() != RESOURCE_KEYS:
            return False
        if (
            item["kind"] not in ("cpu", "qemu", "bare-metal", "real-gpu")
            or not isinstance(item["alias"], str)
            or not item["alias"].strip()
            or not matches(item["grant_ref"], None)
            or not matches(item["capabilities"], [])
            or not matches(item["wall_seconds"], 0)
            or item["network"] not in ("deny", "approved-egress-only")
        ):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    try:
        valid = validate(read_json(args.profile))
    except (OSError, ValueError, UnicodeError, RecursionError):
        valid = False
    if not valid:
        # Input values may contain an accidentally pasted secret.
        print("Invalid proposal. Compare with the local template; no action was taken.")
        return 1
    print("Valid proposal shape. Incomplete fields and permissions need owner review.")
    print("No model call, test, claim, schedule or authorization was created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
