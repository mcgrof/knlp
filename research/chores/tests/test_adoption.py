"""Check the downloadable maintenance starter and local site references."""

import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
from urllib.parse import urlsplit

import pytest

PUBLIC = Path(__file__).resolve().parents[1] / "web" / "public"
STARTER = PUBLIC / "adopt"
CHECKER = STARTER / "check_profile.py"
validate = runpy.run_path(str(CHECKER))["validate"]


def proposal():
    return json.loads((STARTER / "policy.json").read_text())


def test_starter_has_no_execution_or_publication_grant():
    value = proposal()
    assert validate(value)
    assert value["status"] == "proposal"
    assert value["repository"]["include"] == []
    assert value["authority"]["read_paths"] == []
    assert value["authority"]["write_paths"] == []
    assert all(
        item is False
        for key, item in value["authority"].items()
        if key not in {"read_paths", "write_paths"}
    )
    assert value["schedule"]["enabled"] is False
    assert value["resources"] == []


@pytest.mark.parametrize("kind", ["cpu", "qemu", "bare-metal", "real-gpu"])
def test_each_resource_kind_is_a_proposal_not_a_runner(kind):
    value = proposal()
    value["repository"]["revision"] = "a" * 40
    value["model_access"]["cost_limit_per_task"] = 0.25
    value["resources"] = [
        {
            "kind": kind,
            "alias": "test-worker",
            "grant_ref": None,
            "capabilities": ["contract-check"],
            "wall_seconds": 60,
            "network": "deny",
        }
    ]
    assert validate(value)
    assert value["authority"]["execute_tests"] is False


@pytest.mark.parametrize(
    "section,key,bad",
    [
        ("authority", "execute_tests", "yes"),
        ("model_access", "request_limit_per_task", True),
        ("model_access", "token_limit_per_task", -1),
        ("model_access", "cost_limit_per_task", float("inf")),
        ("model_access", "cost_limit_per_task", float("nan")),
        ("model_access", "personal_account_fallback", True),
        ("schedule", "enabled", True),
        ("repository", "revision", "main"),
        ("coordination", "single_writer_per_branch", False),
        ("coordination", "stale_results", "accept"),
    ],
)
def test_invalid_or_activating_options_are_rejected(section, key, bad):
    value = proposal()
    value[section][key] = bad
    assert not validate(value)


def test_unknown_credential_field_and_missing_field_are_rejected():
    value = proposal()
    value["model_access"]["token"] = "synthetic-secret"
    assert not validate(value)
    value = proposal()
    del value["delegation"]["incident_route_ref"]
    assert not validate(value)
    assert not validate([])


def test_cli_never_echoes_invalid_contents_or_executes_them(tmp_path):
    value = proposal()
    marker = tmp_path / "must-not-exist"
    value["run"] = [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]
    value["token"] = "synthetic-secret"
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(value))
    completed = subprocess.run(
        [sys.executable, str(CHECKER), str(path)], capture_output=True, text=True
    )
    assert completed.returncode == 1
    assert "synthetic-secret" not in completed.stdout + completed.stderr
    assert not marker.exists()
    path.write_text("{" + " " * 65536)
    assert (
        subprocess.run(
            [sys.executable, str(CHECKER), str(path)], capture_output=True
        ).returncode
        == 1
    )


def test_skill_references_resolve():
    for filename in ("SKILL.md", "README.md", "REVIEW.md"):
        for target in re.findall(r"\]\(([^)]+)\)", (STARTER / filename).read_text()):
            parsed = urlsplit(target)
            if not parsed.scheme and parsed.path:
                assert (STARTER / parsed.path).is_file(), (filename, target)
