"""Strict proposal inputs, advisory gates and bounded inventory regressions."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit

import pytest

PUBLIC = Path(__file__).resolve().parents[1] / "web" / "public"
STARTER = PUBLIC / "adopt"
spec = importlib.util.spec_from_file_location("maintenance", STARTER / "maintenance.py")
maintenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maintenance)
NOW, BEFORE, AFTER = (
    "2030-01-01T00:10:00Z",
    "2030-01-01T00:09:00Z",
    "2030-01-02T00:00:00Z",
)
HEAD, TARGET, COMMAND = "a" * 40, "b" * 40, "c" * 40


@pytest.fixture
def documents():
    p = maintenance.read_json(STARTER / "policy.json")
    classification = {"method": "deterministic", "reviewer_ref": None}
    p.update(
        status="approved",
        repository="example/library",
        target_branch="main",
        approved_at="2030-01-01T00:00:00Z",
        expires_at=AFTER,
        endpoint="verify",
        observation_max_age_seconds=300,
        classes={
            "documentation": {
                "paths": ["docs/**"],
                "actions": [
                    "investigate",
                    "edit",
                    "test",
                    "review",
                    "publish",
                    "branch_update",
                    "merge",
                    "verify",
                ],
                "required_tests": ["links"],
                "human_review": "on_exception",
                "classification": classification,
            }
        },
        tests={
            "links": {
                "command_id": "documentation-links",
                "command_revision": COMMAND,
                "environment_id": "cpu-v1",
                "environment_digest": "d" * 64,
                "producer_id": "trusted-runner",
                "required_assertions": ["links-resolve"],
                "capabilities": ["cpu"],
                "requires_resource": True,
                "max_age_seconds": 300,
            }
        },
        post_merge_tests=["links"],
        limits={
            "repair_attempts": 3,
            "elapsed_seconds": 1800,
            "requests": 10,
            "tokens": 10000,
            "cost": 1,
            "ci_runs": 5,
        },
        retry={
            "limit": 1,
            "transient_reasons": ["RUNNER_INTERRUPTED", "MODEL_UNAVAILABLE"],
        },
        delegation={"route": "project-maintainer", "response_seconds": 600},
    )
    p["model_access"].update(
        route="managed-session",
        data_classes=["public-source"],
        request_limit=10,
        token_limit=10000,
        cost_limit=1,
    )
    change = {
        "path": "docs/guide.md",
        "previous_path": None,
        "kind": "modified",
        "file_type": "regular",
        "file_mode": "100644",
        "behavior": "ordinary",
        "classes": ["documentation"],
        "classification": dict(
            classification, receipt="inventory:1", rationale="Plain prose fixture."
        ),
    }
    task = {
        "schema_version": 1,
        "task_id": "repair-1",
        "repository": p["repository"],
        "classes": ["documentation"],
        "next_action": "merge",
        "endpoint": "verify",
        "head": HEAD,
        "target": TARGET,
        "policy_digest": maintenance.digest(p),
        "attempt": 1,
        "started_at": "2030-01-01T00:00:00Z",
        "changes": [change],
    }
    evidence = {
        "schema_version": 1,
        "task_id": "repair-1",
        "results": [
            {
                "test_id": "links",
                "head": HEAD,
                "target": TARGET,
                "tested_revision": HEAD,
                "command_revision": COMMAND,
                "environment_id": "cpu-v1",
                "environment_digest": "d" * 64,
                "producer_id": "trusted-runner",
                "artifact_digest": "e" * 64,
                "runner_id": "runner-1",
                "resource_grant": "cpu-grant",
                "receipt": "run:1",
                "completed_at": BEFORE,
                "outcome": "passed",
                "assertions": {"links-resolve": "passed"},
                "applicability_rule": None,
                "transient_reason": None,
            }
        ],
    }
    values = {
        "policy": {
            "digest": task["policy_digest"],
            "repository": p["repository"],
            "target_branch": "main",
            "baseline_revision": TARGET,
            "authority_id": "owner-policy",
            "revoked": False,
            "expires_at": AFTER,
        },
        "candidate": {
            "head": HEAD,
            "target": TARGET,
            "changes": deepcopy(task["changes"]),
            "merge_method": "ff-only",
            "tested_revision": HEAD,
            "queue_id": None,
        },
        "review": {
            "head": HEAD,
            "target": TARGET,
            "policy_digest": task["policy_digest"],
            "decision": "pending",
            "approved_actions": [],
            "blockers": [],
            "expires_at": AFTER,
        },
        "protections": {
            "head": HEAD,
            "target": TARGET,
            "required_checks": ["links"],
            "producers": ["trusted-runner"],
            "allowed_methods": ["ff-only"],
            "atomic_freshness": "compare-and-swap",
            "native_reviews_satisfied": True,
            "blockers": [],
        },
        "resources": {
            "capabilities": ["cpu"],
            "unavailable_tests": [],
            "grants": [
                {
                    "id": "cpu-grant",
                    "capabilities": ["cpu"],
                    "expires_at": AFTER,
                    "revoked": False,
                }
            ],
            "usage": {"requests": 1, "tokens": 100, "cost": 0, "ci_runs": 1},
            "worker_generation": 1,
            "lease_generation": 1,
            "lease_expires_at": AFTER,
        },
        "effects": {
            "operation_id": "merge-1",
            "state": "none",
            "head": HEAD,
            "target": TARGET,
            "merged_revision": None,
            "receipt": None,
        },
    }
    snapshot = {
        "schema_version": 1,
        "task_id": "repair-1",
        "observations": {
            k: {
                "state": "known",
                "observed_at": BEFORE,
                "source": "fixture-authority",
                "receipt": "observation:1",
                "value": v,
            }
            for k, v in values.items()
        },
    }
    return [p, task, evidence, snapshot]


def value(d, group):
    return d[3]["observations"][group]["value"]


def refresh_policy(d):
    pd = maintenance.digest(d[0])
    d[1]["policy_digest"] = value(d, "policy")["digest"] = value(d, "review")[
        "policy_digest"
    ] = pd


def assess(d):
    report = maintenance.assess(*d, NOW)
    assert report["authority_verified"] is False
    return report


def codes(d):
    return {r["code"] for r in assess(d)["reasons"]}


def test_draft_is_valid_but_incomplete():
    report = maintenance.check_profile(
        maintenance.read_json(STARTER / "policy.json"), True
    )
    assert report["schema_valid"] and not report["complete"]
    assert not report["strict_check_passed"] and not report["authority_verified"]


@pytest.mark.parametrize(
    "endpoint,action,state",
    [
        ("review", "review", "reviewed"),
        ("publish", "publish", "published"),
        ("verify", "verify", "verified"),
    ],
)
def test_each_endpoint_requires_a_matching_receipt(documents, endpoint, action, state):
    d = documents
    d[0]["endpoint"] = d[1]["endpoint"] = endpoint
    d[1]["next_action"] = action
    refresh_policy(d)
    value(d, "effects").update(state=state, receipt="native:1", merged_revision=HEAD)
    assert assess(d)["status"] == "complete"
    value(d, "effects")["receipt"] = None
    assert assess(d)["status"] == "ready"


@pytest.mark.parametrize(
    "field",
    [
        "head",
        "target",
        "tested_revision",
        "command_revision",
        "environment_digest",
        "environment_id",
        "producer_id",
    ],
)
def test_evidence_binds_all_identities(documents, field):
    result = documents[2]["results"][0]
    result[field] = (
        "f" * len(result[field])
        if field not in {"environment_id", "producer_id"}
        else "other"
    )
    assert "STALE_EVIDENCE" in codes(documents)


@pytest.mark.parametrize(
    "outcome", ["failed", "skipped", "inconclusive", "cancelled", "unavailable"]
)
def test_nonpassing_is_not_success(documents, outcome):
    documents[2]["results"][0]["outcome"] = outcome
    assert "REQUIRED_TEST_FAILED" in codes(documents)


def test_assertions_and_explicit_applicability(documents):
    result = documents[2]["results"][0]
    result["assertions"] = {}
    assert "ASSERTION_MISSING" in codes(documents)
    result["outcome"] = "not_applicable"
    assert "APPLICABILITY_UNPROVEN" in codes(documents)
    documents[0]["applicability_rules"]["fixture-exclusion"] = {
        "test_id": "links",
        "classes": ["documentation"],
        "reason": "Fixture-only exclusion.",
    }
    result["applicability_rule"] = "fixture-exclusion"
    refresh_policy(documents)
    assert assess(documents)["status"] == "ready"


def test_resource_wait_and_missing_test_are_distinct(documents):
    documents[2]["results"] = []
    value(documents, "resources")["unavailable_tests"] = ["links"]
    assert assess(documents)["status"] == "waiting_resource"
    assert "REQUIRED_TEST_MISSING" not in codes(documents)
    value(documents, "resources")["unavailable_tests"] = []
    assert "REQUIRED_TEST_MISSING" in codes(documents)


@pytest.mark.parametrize(
    "group", ["policy", "candidate", "review", "protections", "resources", "effects"]
)
def test_missing_unknown_and_stale_observations(documents, group):
    observation = documents[3]["observations"].pop(group)
    original = deepcopy(observation["value"])
    assert "OBSERVATION_MISSING" in codes(documents)
    documents[3]["observations"][group] = observation
    observation.update(state="unknown", value=None)
    assert "OBSERVATION_UNKNOWN" in codes(documents)
    observation.update(
        state="known", value=original, observed_at="2029-01-01T00:00:00Z"
    )
    assert "OBSERVATION_STALE" in codes(documents)


def test_human_handoff_and_bound_resume(documents):
    documents[0]["classes"]["documentation"]["human_review"] = "before_merge"
    refresh_policy(documents)
    report = assess(documents)
    assert report["status"] == "needs_human"
    assert (
        report["handoff"]["route"]
        and report["handoff"]["response_deadline"] == "2030-01-01T00:20:00+00:00"
    )
    value(documents, "review").update(decision="approved", approved_actions=["merge"])
    assert assess(documents)["status"] == "ready"
    value(documents, "review")["target"] = "f" * 40
    assert "STALE_APPROVAL" in codes(documents)


@pytest.mark.parametrize(
    "behavior", ["executable_example", "generated_api", "test_removal", "unknown"]
)
def test_reserved_behavior_needs_review(documents, behavior):
    documents[1]["changes"][0]["behavior"] = behavior
    value(documents, "candidate")["changes"] = deepcopy(documents[1]["changes"])
    assert assess(documents)["status"] == "needs_human"
    value(documents, "review").update(decision="approved", approved_actions=["merge"])
    assert assess(documents)["status"] == (
        "needs_human" if behavior == "unknown" else "ready"
    )


@pytest.mark.parametrize(
    "path", ["maintenance/policy.json", ".github/workflows/tests.yml"]
)
def test_policy_and_ci_cannot_self_authorize(documents, path):
    documents[1]["changes"][0].update(path=path, behavior="policy_change")
    value(documents, "candidate")["changes"] = deepcopy(documents[1]["changes"])
    value(documents, "review").update(decision="approved", approved_actions=["merge"])
    assert "PROTECTED_CHANGE" in codes(documents)


def test_mixed_classes_union_tests_and_intersect_rights(documents):
    p, task, _, _ = documents
    p["classes"]["api"] = deepcopy(p["classes"]["documentation"])
    p["classes"]["api"]["required_tests"] = ["api-test"]
    p["tests"]["api-test"] = deepcopy(p["tests"]["links"])
    task["classes"].append("api")
    task["changes"][0]["classes"].append("api")
    value(documents, "candidate")["changes"] = deepcopy(task["changes"])
    refresh_policy(documents)
    assert "api-test" in assess(documents)["missing_evidence"]
    p["classes"]["api"]["actions"].remove("merge")
    refresh_policy(documents)
    assert "ACTION_DENIED" in codes(documents)


@pytest.mark.parametrize(
    "group,key,bad,code",
    [
        ("candidate", "head", "f" * 40, "CANDIDATE_CHANGED"),
        ("candidate", "target", "f" * 40, "CANDIDATE_CHANGED"),
        ("policy", "digest", "f" * 64, "POLICY_MISMATCH"),
        ("resources", "lease_generation", 2, "LEASE_EXPIRED"),
        ("protections", "producers", ["other"], "CHECK_PRODUCER_MISMATCH"),
        ("protections", "atomic_freshness", "none", "ATOMIC_FRESHNESS_UNSUPPORTED"),
        ("effects", "state", "unknown", "EXTERNAL_RESULT_UNKNOWN"),
        ("effects", "state", "post_merge_failed", "POST_MERGE_FAILED"),
    ],
)
def test_runtime_gate_changes(documents, group, key, bad, code):
    value(documents, group)[key] = bad
    assert code in codes(documents)


def test_expired_grants_budgets_and_bounded_retry(documents):
    value(documents, "resources")["grants"][0]["revoked"] = True
    assert "RESOURCE_GRANT_INVALID" in codes(documents)
    value(documents, "resources")["grants"][0]["revoked"] = False
    value(documents, "resources")["usage"]["tokens"] = 10001
    assert "BUDGET_EXHAUSTED" in codes(documents)
    value(documents, "resources")["usage"]["tokens"] = 100
    result = documents[2]["results"][0]
    result.update(outcome="cancelled", transient_reason="RUNNER_INTERRUPTED")
    assert assess(documents)["status"] == "retryable"
    documents[1]["attempt"] = 2
    assert assess(documents)["status"] == "blocked"
    documents[1]["attempt"] = 1
    result["outcome"] = "failed"
    assert assess(documents)["status"] == "blocked"


def test_endpoint_and_postmerge_identity(documents):
    documents[0]["endpoint"] = documents[1]["endpoint"] = "review"
    refresh_policy(documents)
    assert "ACTION_BEYOND_ENDPOINT" in codes(documents)
    documents[0]["endpoint"] = documents[1]["endpoint"] = "verify"
    documents[1]["next_action"] = "verify"
    refresh_policy(documents)
    value(documents, "effects").update(
        state="verified", merged_revision="f" * 40, receipt="native:1"
    )
    assert "STALE_EVIDENCE" in codes(documents)


def test_model_limit_is_independent_and_zero_spend_can_finish(documents):
    documents[0]["model_access"]["token_limit"] = 50
    refresh_policy(documents)
    assert "BUDGET_EXHAUSTED" in codes(documents)
    documents[0]["model_access"]["token_limit"] = 10000
    documents[0]["model_access"]["cost_limit"] = documents[0]["limits"]["cost"] = 0
    refresh_policy(documents)
    assert assess(documents)["status"] == "ready"


def test_queue_identity_and_native_reviews_are_required(documents):
    documents[0]["merge_method"] = value(documents, "candidate")["merge_method"] = (
        "merge-queue"
    )
    value(documents, "protections").update(
        allowed_methods=["merge-queue"], atomic_freshness="merge-queue"
    )
    refresh_policy(documents)
    assert "ATOMIC_FRESHNESS_UNSUPPORTED" in codes(documents)
    value(documents, "candidate")["queue_id"] = "queue-entry-1"
    assert assess(documents)["status"] == "ready"
    value(documents, "protections")["native_reviews_satisfied"] = False
    assert "NATIVE_PROTECTION_BLOCKED" in codes(documents)


def test_rename_cannot_escape_protected_source(documents):
    documents[1]["changes"][0].update(
        kind="renamed", previous_path="maintenance/policy.json"
    )
    value(documents, "candidate")["changes"] = deepcopy(documents[1]["changes"])
    assert "PROTECTED_CHANGE" in codes(documents)


@pytest.mark.parametrize(
    "path", ["../outside", "/absolute", "docs/../policy", "docs//guide"]
)
def test_noncanonical_paths_are_rejected(documents, path):
    documents[1]["changes"][0]["path"] = path
    with pytest.raises(ValueError):
        assess(documents)


def test_model_unavailability_is_an_explicit_bounded_retry(documents):
    documents[2]["results"][0].update(
        outcome="unavailable", transient_reason="MODEL_UNAVAILABLE"
    )
    assert assess(documents)["status"] == "retryable"


def test_hash_cannot_smuggle_a_trailing_newline(documents):
    documents[1]["head"] += "\n"
    with pytest.raises(ValueError):
        assess(documents)


@pytest.mark.parametrize(
    "raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}', "{"]
)
def test_strict_json_parser(tmp_path, raw):
    path = tmp_path / "input.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        maintenance.read_json(path)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("schema_version", 999),
        ("endpoint", "run-shell"),
        ("approved_at", "yesterday"),
        ("token", "synthetic-secret"),
    ],
)
def test_strict_schema_fields(key, bad):
    p = maintenance.read_json(STARTER / "policy.json")
    p[key] = bad
    with pytest.raises(ValueError):
        maintenance.check_profile(p)


def test_schema_independent_of_editable_example(tmp_path, monkeypatch):
    draft = maintenance.read_json(STARTER / "policy.json")
    (tmp_path / "policy.json").write_text('{"token":"synthetic-secret"}')
    (tmp_path / "schemas").symlink_to(STARTER / "schemas", target_is_directory=True)
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    assert maintenance.check_profile(draft)["schema_valid"]
    with pytest.raises(ValueError):
        maintenance.check_profile({"token": "synthetic-secret"})


def test_inventory_never_executes_follows_symlinks_or_overwrites(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    marker = tmp_path / "not-created"
    (repo / "Makefile").write_text(f"all:\n\ttouch {marker}\n")
    (repo / "AGENTS.md").symlink_to(tmp_path / "secret")
    (tmp_path / "secret").write_text("synthetic-secret")
    os.mkfifo(repo / "CONTRIBUTING")
    out = tmp_path / "proposal"
    maintenance.initialize(repo, out)
    assert not marker.exists()
    inventory = json.loads((out / "inventory.json").read_text())
    assert [f["path"] for f in inventory["files"]] == ["Makefile"]
    assert "synthetic-secret" not in (out / "inventory.json").read_text()
    with pytest.raises(FileExistsError):
        maintenance.initialize(repo, out)


def test_cli_bounds_and_secret_redaction(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"token":"synthetic-secret"}')
    result = subprocess.run(
        [sys.executable, str(STARTER / "maintenance.py"), "check-profile", str(path)],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 2
        and "synthetic-secret" not in result.stdout + result.stderr
    )
    path.write_text(" " * (maintenance.MAX_BYTES + 1))
    with pytest.raises(ValueError):
        maintenance.read_json(path)


def test_inventory_does_not_attribute_dirty_commands_to_head(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["/usr/bin/git", "init", str(repo)], capture_output=True, check=True)
    (repo / "Makefile").write_text("all:\n\ttrue\n")
    subprocess.run(["/usr/bin/git", "-C", str(repo), "add", "Makefile"], check=True)
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-m",
            "Fixture",
        ],
        capture_output=True,
        check=True,
    )
    for name, dirty in [("clean", False), ("dirty", True)]:
        if dirty:
            (repo / "Makefile").write_text("all:\n\tfalse\n")
        output = tmp_path / name
        maintenance.initialize(repo, output)
        inventory = maintenance.read_json(output / "inventory.json")
        source = inventory["files"][0]
        assert bool(source["revision"]) is not dirty
        assert source["command_candidates"][0]["source_revision"] == source["revision"]


def test_local_references_resolve():
    for filename in ("SKILL.md", "README.md", "REVIEW.md", "ASSESSMENT.md"):
        for target in re.findall(r"\]\(([^)]+)\)", (STARTER / filename).read_text()):
            parsed = urlsplit(target)
            if not parsed.scheme and parsed.path:
                assert (STARTER / parsed.path).is_file(), (filename, target)
