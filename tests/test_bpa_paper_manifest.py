#!/usr/bin/env python3

from pathlib import Path
import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts" / "paper" / "bpa_paper")
)

from common import load_config
from manifest import empty_manifest, validate_manifest


def test_empty_manifest_is_valid() -> None:
    manifest = empty_manifest()
    assert validate_manifest(manifest) == []


def test_invalid_status_is_rejected() -> None:
    manifest = empty_manifest()
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


def test_config_loads() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "paper"
        / "bpa_paper"
        / "configs"
        / "a100.yaml"
    )
    cfg = load_config(path)
    assert cfg.gpu == "A100"
    assert 4096 in cfg.contexts
