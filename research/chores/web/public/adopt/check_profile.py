#!/usr/bin/env python3
"""Compatibility entrypoint for fixed-schema proposal checking."""

import argparse
from pathlib import Path

from maintenance import check_profile, read_json


def validate(profile):
    try:
        return check_profile(profile)["schema_valid"]
    except (ValueError, RecursionError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    try:
        valid = validate(read_json(args.profile))
    except (OSError, ValueError, RecursionError):
        valid = False
    if not valid:
        print("Invalid proposal; no action was taken.")
        return 1
    print("Valid schema; completeness and authority are separate checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
