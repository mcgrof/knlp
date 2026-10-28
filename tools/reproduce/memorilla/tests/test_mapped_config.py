# SPDX-License-Identifier: MIT
"""Mapped serving choices must not silently change the author control."""

from pathlib import Path

import pytest

from tools.reproduce.memorilla.run import configuration


def test_mapped_defconfig_selects_borrowed_bf16_and_explicit_trust(tmp_path):
    root = Path(__file__).resolve().parents[4]
    text = (root / "defconfigs/memorilla-lmcache-mapped").read_text()
    path = tmp_path / "config"
    path.write_text(text)
    cfg = configuration(path)
    assert cfg["artifact_backing"] == "mmap"
    assert cfg["artifact_verification"] == "published"
    assert cfg["input_pipeline"] == "leased-bf16"
    path.write_text(text.replace('VERIFICATION="published"', 'VERIFICATION="on-map"'))
    assert configuration(path)["artifact_verification"] == "on-map"


@pytest.mark.parametrize(
    "extra,match",
    [
        ('ARTIFACT_BACKING="unknown"', "backing"),
        ('ARTIFACT_VERIFICATION="unknown"', "verification"),
        ('ARTIFACT_VERIFICATION="published"', "arena"),
        ("MAX_MAPPINGS=0", "positive"),
    ],
)
def test_invalid_policies_refuse(tmp_path, extra, match):
    path = tmp_path / "config"
    path.write_text(
        "CONFIG_KNLP_REPRODUCE_MEMORILLA=y\nCONFIG_KNLP_MEM_LMCACHE=y\n"
        f"CONFIG_KNLP_MEM_{extra}\n"
    )
    with pytest.raises(ValueError, match=match):
        configuration(path)
