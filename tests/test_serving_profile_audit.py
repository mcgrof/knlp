# SPDX-License-Identifier: MIT
"""A diagnostic must remain bound to its exact completed parent experiment."""
import json
import sys

import pytest

from tools.kv.native import serving_profile_audit as profile


def parent_fixture(tmp_path, monkeypatch):
    arms = ["candidate", "native_fi", "native_fa3"]
    confirmation = tmp_path / "confirmation.json"
    confirmation.write_text(
        json.dumps(
            {
                "complete": True,
                "pending": [],
                "design_sha256": "design",
                "source_sha": "measured",
                "blocks": [
                    {
                        "block": 0,
                        "arms": {
                            arm: {"protocol_sha256": arm + "-protocol"} for arm in arms
                        },
                    }
                ],
            }
        )
    )
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    plan = freeze / "PROFILE-PLAN.json"
    plan.write_text(
        json.dumps(
            {
                "parent_result_sha256": profile.sha(confirmation),
                "parent_design_sha256": "design",
            }
        )
    )
    output = tmp_path / "output.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--root",
            str(tmp_path),
            "--freeze",
            str(freeze),
            "--confirmation",
            str(confirmation),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        profile,
        "audit",
        lambda root, freeze, arm: {
            "protocol_sha256": arm + "-protocol",
            "source_sha": "measured",
        },
    )
    return confirmation, plan, output


def test_rejects_changed_parent_before_profile_analysis(tmp_path, monkeypatch):
    confirmation, _, output = parent_fixture(tmp_path, monkeypatch)
    confirmation.write_text(confirmation.read_text() + " ")
    monkeypatch.setattr(
        profile, "audit", lambda *args: pytest.fail("must not read profiles")
    )
    with pytest.raises(AssertionError):
        profile.main()
    assert not output.exists()


def test_rejects_changed_runtime_protocol(tmp_path, monkeypatch):
    _, _, output = parent_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        profile,
        "audit",
        lambda root, freeze, arm: {
            "protocol_sha256": "changed",
            "source_sha": "measured",
        },
    )
    with pytest.raises(AssertionError):
        profile.main()
    assert not output.exists()


def test_retains_parent_identity_for_matched_protocols(tmp_path, monkeypatch):
    confirmation, _, output = parent_fixture(tmp_path, monkeypatch)
    profile.main()
    result = json.loads(output.read_text())
    assert result["matched_parent_protocols"]
    assert result["parent_confirmation_sha256"] == profile.sha(confirmation)
    assert result["excluded_from_confirmation"]
