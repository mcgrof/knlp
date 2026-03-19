#!/usr/bin/env python3
"""Focused self-test for BPA paper manifest/config helpers."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import load_config
from manifest import empty_manifest, validate_manifest


def main() -> int:
    manifest = empty_manifest()
    assert validate_manifest(manifest) == []

    manifest["points"].append(
        {
            "gpu": "A100",
            "lane": "matched-scaling",
            "batch": 1,
            "context": 1024,
            "status": "bogus",
        }
    )
    errors = validate_manifest(manifest)
    assert any("invalid point status" in error for error in errors)

    cfg = load_config(ROOT / "configs" / "a100.yaml")
    assert cfg.gpu == "A100"
    assert 4096 in cfg.contexts
    print("bpa_paper manifest/config self-test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
