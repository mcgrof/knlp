#!/usr/bin/env python3
"""Package validated BPA paper results into a clean export tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import write_json
from manifest import load_manifest, validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Package BPA paper results")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(error)
        return 1

    payload = {
        "mode": "dry-run" if args.dry_run else "plan-only",
        "source_root": args.source_root,
        "export_root": args.export_root,
        "required_subtrees": [
            "raw",
            "derived",
            "figures",
            "manifests",
            "logs",
            "system",
            "reports",
        ],
        "rules": [
            "export only paper-usable points",
            "preserve checksums in export manifest",
            "redact secrets from environment captures",
            "keep private hodge-podge tree out of public-facing docs",
        ],
    }
    out = Path(args.export_root) / "manifests" / "export_manifest.json"
    write_json(out, payload)
    print(f"wrote export plan to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
