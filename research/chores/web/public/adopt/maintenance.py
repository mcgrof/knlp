#!/usr/bin/env python3
"""Prepare maintenance proposals and assess supplied facts without executing work."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parent
SCHEMA_URI = "https://chores.knlp.io/adopt/schemas/"
SCHEMAS = {
    "common": "common-v1.json",
    "policy": "policy-v2.json",
    "task": "task-v1.json",
    "evidence": "evidence-v1.json",
    "snapshot": "snapshot-v1.json",
    "assessment": "assessment-v1.json",
}
MAX_BYTES = 262144
MAX_INVENTORY_FILES = 64
MAX_FILE_BYTES = 32768
MAX_INVENTORY_BYTES = 524288
GIT_ENV = {
    "PATH": "/usr/bin:/bin",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_NO_LAZY_FETCH": "1",
}
FORMATS = FormatChecker()


class InputError(ValueError):
    """Reject malformed input without echoing its contents."""


def parse_time(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,6})?(?:Z|[+-]\d\d:\d\d)", value
    ):
        raise InputError("invalid timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError as error:
        raise InputError("invalid timestamp") from error


@FORMATS.checks("date-time", raises=ValueError)
def valid_time(value):
    parse_time(value)
    return True


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError("duplicate JSON key")
        result[key] = value
    return result


def reject_constant(_):
    raise InputError("nonfinite JSON number")


def check_finite(value):
    if isinstance(value, float) and not (-float("inf") < value < float("inf")):
        raise InputError("nonfinite JSON number")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or any(ord(char) < 32 for char in key):
                raise InputError("invalid JSON member name")
            check_finite(item)
    if isinstance(value, list):
        for item in value:
            check_finite(item)


def read_json(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise InputError("document exceeds size limit")
    try:
        value = json.loads(
            raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant
        )
        check_finite(value)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise InputError("invalid JSON document") from error
    return value


def digest(value):
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_document(kind, value):
    check_finite(value)
    schemas = {
        name: read_json(ROOT / "schemas" / filename)
        for name, filename in SCHEMAS.items()
    }
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in schemas.values()
    )
    schema = schemas[kind]
    Draft202012Validator.check_schema(schema)
    if next(
        Draft202012Validator(
            schema, registry=registry, format_checker=FORMATS
        ).iter_errors(value),
        None,
    ):
        raise InputError("document does not match its versioned schema")


def profile_gaps(policy):
    gaps = []
    for key in ("repository", "target_branch", "approved_at", "expires_at"):
        if not policy[key]:
            gaps.append(key)
    if policy["status"] != "approved":
        gaps.append("status")
    if (
        not policy["model_access"]["route"]
        or not policy["model_access"]["data_classes"]
    ):
        gaps.append("model_access")
    if not policy["classes"]:
        gaps.append("classes")
    if (
        not policy["delegation"]["route"]
        or not policy["delegation"]["response_seconds"]
        or not policy["observation_max_age_seconds"]
    ):
        gaps.append("delegation_or_freshness")
    if (
        not policy["limits"]["repair_attempts"]
        or not policy["limits"]["elapsed_seconds"]
    ):
        gaps.append("limits")
    for class_id, rule in policy["classes"].items():
        if not rule["paths"] or not rule["actions"]:
            gaps.append("class_scope:" + class_id)
        if (
            rule["classification"]["method"] == "reviewer"
            and not rule["classification"]["reviewer_ref"]
        ):
            gaps.append("classification_reviewer:" + class_id)
        for test in rule["required_tests"]:
            if test not in policy["tests"]:
                gaps.append("unknown_test:" + test)
    for test in policy["post_merge_tests"]:
        if test not in policy["tests"]:
            gaps.append("unknown_post_merge_test:" + test)
    if (
        any("merge" in rule["actions"] for rule in policy["classes"].values())
        and not policy["post_merge_tests"]
    ):
        gaps.append("post_merge_tests")
    for rule in policy["applicability_rules"].values():
        if (
            rule["test_id"] not in policy["tests"]
            or not set(rule["classes"]) <= policy["classes"].keys()
        ):
            gaps.append("applicability_reference")
    if (
        policy["approved_at"]
        and policy["expires_at"]
        and parse_time(policy["approved_at"]) >= parse_time(policy["expires_at"])
    ):
        gaps.append("policy_validity_window")
    for test_id, test in policy["tests"].items():
        if not test["required_assertions"] or not test["max_age_seconds"]:
            gaps.append("test_assertions_or_freshness:" + test_id)
    return sorted(set(gaps))


def check_profile(policy, complete=False):
    validate_document("policy", policy)
    gaps = profile_gaps(policy)
    return {
        "schema_valid": True,
        "complete": not gaps,
        "missing": gaps,
        "authority_verified": False,
        "strict_check_passed": not complete or not gaps,
    }


def matches_path(path, selector):
    if selector.endswith("/**"):
        return path.startswith(selector[:-2])
    return path == selector


def assess(policy, task, evidence, snapshot, now):
    for kind, value in (
        ("policy", policy),
        ("task", task),
        ("evidence", evidence),
        ("snapshot", snapshot),
    ):
        validate_document(kind, value)
    clock = parse_time(now)
    reasons, missing = [], []

    def add(code, detail, status="blocked"):
        record = (status, code, detail)
        if record not in reasons:
            reasons.append(record)

    def observation(group):
        item = snapshot["observations"].get(group)
        if not item:
            add("OBSERVATION_MISSING", group)
            missing.append(group)
            return None
        if item["state"] != "known":
            add("OBSERVATION_UNKNOWN", group)
            missing.append(group)
            return None
        age = (clock - parse_time(item["observed_at"])).total_seconds()
        if age < 0 or age > policy["observation_max_age_seconds"]:
            add("OBSERVATION_STALE", group)
            return None
        return item["value"]

    pd = digest(policy)
    if profile_gaps(policy):
        add("PROFILE_INCOMPLETE", "policy needs explicit decisions")
    if pd != task["policy_digest"] or task["repository"] != policy["repository"]:
        add("POLICY_MISMATCH", "task is not bound to this policy")
    if evidence["task_id"] != task["task_id"] or snapshot["task_id"] != task["task_id"]:
        add("TASK_MISMATCH", "input task identities differ")
    if task["endpoint"] != policy["endpoint"]:
        add("ENDPOINT_MISMATCH", "task endpoint differs from policy")
    if (
        task["endpoint"] == "review"
        and task["next_action"]
        in {"publish", "branch_update", "merge", "verify", "revert"}
        or task["endpoint"] == "publish"
        and task["next_action"] in {"merge", "verify", "revert"}
    ):
        add("ACTION_BEYOND_ENDPOINT", "action exceeds the selected completion point")
    if policy["approved_at"] and parse_time(policy["approved_at"]) > clock:
        add("POLICY_NOT_YET_VALID", "policy approval is in the future")
    if policy["expires_at"] and parse_time(policy["expires_at"]) <= clock:
        add("POLICY_EXPIRED", "policy validity has ended")
    elapsed = (clock - parse_time(task["started_at"])).total_seconds()
    if (
        elapsed < 0
        or elapsed >= policy["limits"]["elapsed_seconds"]
        or not 1 <= task["attempt"] <= policy["limits"]["repair_attempts"]
    ):
        add("BUDGET_EXHAUSTED", "task time or repair attempts exhausted")
    authority = observation("policy")
    candidate = observation("candidate")
    resources = observation("resources")
    if authority:
        if (
            authority["digest"] != pd
            or authority["repository"] != policy["repository"]
            or authority["target_branch"] != policy["target_branch"]
        ):
            add("POLICY_MISMATCH", "effective policy observation differs")
        if authority["revoked"] or parse_time(authority["expires_at"]) <= clock:
            add("POLICY_REVOKED_OR_EXPIRED", "effective policy grant is unavailable")
    if candidate:
        if any(candidate[key] != task[key] for key in ("head", "target", "changes")):
            add(
                "CANDIDATE_CHANGED",
                "candidate head, target or full change inventory differs",
            )
        if candidate["merge_method"] != policy["merge_method"]:
            add("MERGE_METHOD_DENIED", "candidate method differs from policy")
    if resources:
        if (
            resources["worker_generation"] != resources["lease_generation"]
            or parse_time(resources["lease_expires_at"]) <= clock
        ):
            add("LEASE_EXPIRED", "worker no longer owns current authority")
        model_limits = {
            "requests": policy["model_access"]["request_limit"],
            "tokens": policy["model_access"]["token_limit"],
            "cost": policy["model_access"]["cost_limit"],
        }
        for key, used in resources["usage"].items():
            limit = min(
                policy["limits"][key], model_limits.get(key, policy["limits"][key])
            )
            if used > limit:
                add("BUDGET_EXHAUSTED", "cumulative usage limit reached: " + key)
    if not task["classes"] or not task["changes"]:
        add("CLASS_AMBIGUOUS", "classify the full candidate", "needs_human")
    reserved_behavior = any(
        change["behavior"] != "ordinary"
        or change["file_type"] != "regular"
        or change["file_mode"] not in {"100644", "100664"}
        for change in task["changes"]
    )
    external = task["next_action"] in {"publish", "branch_update", "merge", "revert"}
    review = observation("review") if external or reserved_behavior else None
    current_approval = bool(
        review
        and (review["head"], review["target"], review["policy_digest"])
        == (task["head"], task["target"], pd)
        and parse_time(review["expires_at"]) > clock
        and review["decision"] == "approved"
        and not review["blockers"]
        and task["next_action"] in review["approved_actions"]
    )
    declared_classes = set()
    for change in task["changes"]:
        paths = [change["path"]] + (
            [change["previous_path"]] if change["previous_path"] else []
        )
        if any(
            matches_path(path, pattern)
            for path in paths
            for pattern in policy["protected_paths"]
        ):
            add("PROTECTED_CHANGE", "reserved path requires trusted policy review")
        if (
            change["behavior"] == "unknown"
            or not current_approval
            and (
                change["behavior"] != "ordinary"
                or change["file_type"] != "regular"
                or change["file_mode"] not in {"100644", "100664"}
            )
        ):
            add(
                "CLASS_AMBIGUOUS",
                "reserved or unknown behavior needs review",
                "needs_human",
            )
        if not change["classes"]:
            add("CLASS_AMBIGUOUS", "a changed item has no class", "needs_human")
        for class_id in change["classes"]:
            declared_classes.add(class_id)
            rule = policy["classes"].get(class_id)
            if not rule or not all(
                any(matches_path(path, pattern) for pattern in rule["paths"])
                for path in paths
            ):
                add("ACTION_DENIED", "change lies outside its declared class")
                continue
            if (
                change["classification"]["method"] != rule["classification"]["method"]
                or change["classification"]["reviewer_ref"]
                != rule["classification"]["reviewer_ref"]
            ):
                add(
                    "CLASS_AMBIGUOUS",
                    "classification producer differs from policy",
                    "needs_human",
                )
    if declared_classes != set(task["classes"]):
        add(
            "CLASS_AMBIGUOUS",
            "task classes do not cover the full change inventory",
            "needs_human",
        )
    rules = [
        policy["classes"][name] for name in task["classes"] if name in policy["classes"]
    ]
    if len(rules) != len(task["classes"]) or any(
        task["next_action"] not in rule["actions"] for rule in rules
    ):
        add("ACTION_DENIED", "every applicable class must allow the requested action")
    effects = (
        observation("effects")
        if external or task["next_action"] in {"review", "verify"}
        else None
    )
    if effects:
        if effects["head"] != task["head"] or effects["target"] != task["target"]:
            add("STALE_EFFECT", "operation receipt concerns another candidate")
        if effects["state"] == "unknown":
            add("EXTERNAL_RESULT_UNKNOWN", "reconcile the operation before retrying")
        if effects["state"] in {"cancelled", "post_merge_failed"}:
            add(
                (
                    "POST_MERGE_FAILED"
                    if effects["state"] == "post_merge_failed"
                    else "TASK_CANCELLED"
                ),
                "affected scope needs its recovery decision",
            )
        if task["next_action"] == "verify" and (
            not effects["merged_revision"]
            or effects["state"] not in {"merged", "verified"}
        ):
            add(
                "MERGE_NOT_OBSERVED",
                "post-merge verification needs the actual merged revision",
            )
    if review:
        if (review["head"], review["target"], review["policy_digest"]) != (
            task["head"],
            task["target"],
            pd,
        ) or parse_time(review["expires_at"]) <= clock:
            add("STALE_APPROVAL", "review no longer applies")
        if review["blockers"] or review["decision"] == "denied":
            add("REVIEW_BLOCKED", "blocking review findings remain")
        if review["decision"] == "conflict":
            add("OWNER_CONFLICT", "resolve conflicting decisions", "needs_human")
        required = any(
            rule["human_review"] == "before_publish"
            or rule["human_review"] == "before_merge"
            and task["next_action"] in {"merge", "revert"}
            for rule in rules
        )
        if required and (
            review["decision"] != "approved"
            or task["next_action"] not in review["approved_actions"]
        ):
            add(
                "REVIEW_REQUIRED",
                "a current revision-bound human decision is required",
                "needs_human",
            )
    protections = observation("protections") if task["next_action"] == "merge" else None
    if protections:
        if (protections["head"], protections["target"]) != (
            task["head"],
            task["target"],
        ):
            add("TARGET_CHANGED", "native integration state differs")
        if protections["blockers"] or not protections["native_reviews_satisfied"]:
            add("NATIVE_PROTECTION_BLOCKED", "repository protections are not satisfied")
        if (
            policy["merge_method"] not in protections["allowed_methods"]
            or protections["atomic_freshness"] == "none"
            or policy["merge_method"] == "merge-queue"
            and protections["atomic_freshness"] != "merge-queue"
            or policy["merge_method"] == "merge-queue"
            and candidate
            and not candidate["queue_id"]
        ):
            add(
                "ATOMIC_FRESHNESS_UNSUPPORTED",
                "writer cannot enforce the expected integration state",
            )
    required_tests = set().union(*(set(rule["required_tests"]) for rule in rules))
    if protections:
        required_tests.update(protections["required_checks"])
    if task["next_action"] == "verify":
        required_tests = set(policy["post_merge_tests"])
    if task["next_action"] in {
        "review",
        "publish",
        "branch_update",
        "merge",
        "verify",
        "revert",
    }:
        for test_id in sorted(required_tests):
            definition = policy["tests"].get(test_id)
            if not definition:
                add(
                    "UNKNOWN_TEST",
                    "native or class test is not in trusted test definitions",
                )
                continue
            results = [
                result for result in evidence["results"] if result["test_id"] == test_id
            ]
            if not results:
                if resources and test_id in resources["unavailable_tests"]:
                    add(
                        "REQUIRED_RESOURCE_UNAVAILABLE",
                        "required test environment is unavailable",
                        "waiting_resource",
                    )
                else:
                    add("REQUIRED_TEST_MISSING", "required test has no result")
                    missing.append(test_id)
                continue
            for result in results:
                tested = (
                    effects["merged_revision"]
                    if task["next_action"] == "verify" and effects
                    else (candidate["tested_revision"] if candidate else task["head"])
                )
                if any(
                    result[key] != expected
                    for key, expected in (
                        ("head", task["head"]),
                        ("target", task["target"]),
                        ("tested_revision", tested),
                        ("command_revision", definition["command_revision"]),
                        ("environment_id", definition["environment_id"]),
                        ("environment_digest", definition["environment_digest"]),
                        ("producer_id", definition["producer_id"]),
                    )
                ):
                    add(
                        "STALE_EVIDENCE",
                        "test identity, command, producer or environment differs",
                    )
                age = (clock - parse_time(result["completed_at"])).total_seconds()
                if age < 0 or age > definition["max_age_seconds"]:
                    add(
                        "STALE_EVIDENCE",
                        "test completion is outside its validity window",
                    )
                if (
                    protections
                    and result["producer_id"] not in protections["producers"]
                ):
                    add("CHECK_PRODUCER_MISMATCH", "native producer is not recognized")
                if definition["requires_resource"]:
                    grants = [
                        grant
                        for grant in (resources or {}).get("grants", [])
                        if grant["id"] == result["resource_grant"]
                    ]
                    if not grants or any(
                        grant["revoked"]
                        or parse_time(grant["expires_at"]) <= clock
                        or not set(definition["capabilities"])
                        <= set(grant["capabilities"])
                        for grant in grants
                    ):
                        add(
                            "RESOURCE_GRANT_INVALID",
                            "test needs a current matching resource grant",
                        )
                if result["outcome"] == "not_applicable":
                    exclusion = policy["applicability_rules"].get(
                        result["applicability_rule"]
                    )
                    if (
                        not exclusion
                        or exclusion["test_id"] != test_id
                        or not set(task["classes"]) <= set(exclusion["classes"])
                    ):
                        add(
                            "APPLICABILITY_UNPROVEN",
                            "no trusted rule excludes this required test",
                        )
                elif (
                    result["outcome"] in {"cancelled", "unavailable"}
                    and result["transient_reason"]
                    in policy["retry"]["transient_reasons"]
                    and task["attempt"] <= policy["retry"]["limit"]
                ):
                    add(
                        result["transient_reason"],
                        "bounded infrastructure retry is available",
                        "retryable",
                    )
                elif result["outcome"] != "passed":
                    add("REQUIRED_TEST_FAILED", "required test did not pass")
                elif any(
                    result["assertions"].get(assertion) != "passed"
                    for assertion in definition["required_assertions"]
                ):
                    add(
                        "ASSERTION_MISSING",
                        "exit success does not supply all required assertions",
                    )
    status = next(
        (
            state
            for state in ("blocked", "needs_human", "waiting_resource", "retryable")
            if any(item[0] == state for item in reasons)
        ),
        "ready",
    )
    if status == "ready" and effects and effects["receipt"]:
        endpoint_state = {
            "review": "reviewed",
            "publish": "published",
            "verify": "verified",
        }[task["endpoint"]]
        if effects["state"] == endpoint_state:
            status = "complete"
    if not reasons:
        reasons.append(
            (
                status,
                (
                    "ENDPOINT_VERIFIED"
                    if status == "complete"
                    else "DECLARED_GATES_SATISFIED"
                ),
                "supplied facts only; runtime authority remains unverified",
            )
        )
    handoff = None
    if status not in {"ready", "complete"}:
        first_gap = reasons[0][1] + ": " + reasons[0][2]
        handoff = {
            "completed_work": "Candidate "
            + task["head"]
            + "; "
            + str(len(evidence["results"]))
            + " recorded test attempts; retain artifacts.",
            "decision": (task["next_action"] + " is waiting: " + first_gap)[:256],
            "recommendation": {
                "blocked": "Resolve the failed gate and obtain fresh trusted observations; preserve the candidate meanwhile.",
                "needs_human": "Ask the named project route for the revision-bound decision; reuse an existing standing grant if it covers this case.",
                "waiting_resource": "Queue the named required test in its approved environment; retain completed evidence.",
                "retryable": "Use only the declared bounded infrastructure retry; do not rerun deterministic failures unchanged.",
            }[status],
            "alternatives": [
                "Keep this action paused; continue independent authorized work."
            ],
            "evidence": sorted(
                {item[1] for item in reasons}
                | {"head:" + task["head"], "target:" + task["target"]}
            ),
            "route": policy["delegation"]["route"],
            "response_deadline": (
                clock + timedelta(seconds=policy["delegation"]["response_seconds"])
            ).isoformat(),
        }
    result = {
        "schema_version": 1,
        "status": status,
        "next_action": task["next_action"],
        "reasons": [{"code": code, "detail": detail} for _, code, detail in reasons],
        "missing_evidence": sorted(set(missing)),
        "policy_digest": pd,
        "head": task["head"],
        "target": task["target"],
        "authority_verified": False,
        "summary": (
            "Selected endpoint verified in supplied observations."
            if status == "complete"
            else (
                "Declared gates satisfied in supplied observations."
                if status == "ready"
                else "Dependent action must wait for the listed conditions."
            )
        ),
        "handoff": handoff,
    }
    validate_document("assessment", result)
    return result


def bounded_file(repo, relative):
    parts = Path(relative).parts
    if not parts or any(part in {".", ".."} for part in parts):
        raise InputError("invalid inventory path")
    directory = os.open(repo, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            os.close(directory)
            directory = child
        fd = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise InputError("inventory accepts regular files only")
            return stream.read(MAX_FILE_BYTES + 1)
    finally:
        os.close(directory)


def initialize(repo, output):
    repo = Path(repo).absolute()
    if repo.is_symlink() or not repo.is_dir():
        raise InputError("inventory requires a real repository directory")
    revision = None
    try:
        revision = (
            subprocess.run(
                [
                    "/usr/bin/git",
                    "-c",
                    "core.fsmonitor=false",
                    "-C",
                    str(repo),
                    "rev-parse",
                    "--verify",
                    "HEAD",
                ],
                capture_output=True,
                timeout=5,
                check=True,
                env=GIT_ENV,
            )
            .stdout.decode()
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        pass
    candidates = [
        "AGENTS.md",
        "CLAUDE.md",
        "CONTRIBUTING",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "MAINTAINERS",
        "CODEOWNERS",
        ".github/CODEOWNERS",
        "Makefile",
        "meson.build",
        "Cargo.toml",
        "pyproject.toml",
        "package.json",
    ]
    try:
        fd = os.open(repo / ".github", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            wfd = os.open(
                "workflows", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            try:
                with os.scandir(wfd) as entries:
                    for index, entry in enumerate(entries):
                        if index >= MAX_INVENTORY_FILES:
                            break
                        if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}\.ya?ml", entry.name):
                            candidates.append(".github/workflows/" + entry.name)
            finally:
                os.close(wfd)
        finally:
            os.close(fd)
    except OSError:
        pass
    candidates = sorted(set(candidates))[:MAX_INVENTORY_FILES]
    baseline_blobs = {}
    if revision:
        try:
            tree = subprocess.run(
                [
                    "/usr/bin/git",
                    "-c",
                    "core.fsmonitor=false",
                    "-C",
                    str(repo),
                    "ls-tree",
                    "-z",
                    revision,
                    "--",
                    *candidates,
                ],
                capture_output=True,
                check=True,
                timeout=5,
                env=GIT_ENV,
            ).stdout
            for entry in tree.split(b"\0"):
                if not entry:
                    continue
                metadata, path = entry.split(b"\t", 1)
                _, kind, oid = metadata.split()
                if kind == b"blob":
                    baseline_blobs[path.decode()] = oid.decode()
        except (OSError, ValueError, UnicodeError, subprocess.SubprocessError):
            baseline_blobs = {}
    files, skipped, total = [], [], 0
    for relative in candidates:
        try:
            raw = bounded_file(repo, relative)
        except (OSError, InputError):
            skipped.append(relative)
            continue
        if len(raw) > MAX_FILE_BYTES or total + len(raw) > MAX_INVENTORY_BYTES:
            skipped.append(relative)
            continue
        total += len(raw)
        blob = b"blob " + str(len(raw)).encode() + b"\0" + raw
        blob_hash = hashlib.new(
            "sha256" if revision and len(revision) == 64 else "sha1", blob
        ).hexdigest()
        source_revision = (
            revision if baseline_blobs.get(relative) == blob_hash else None
        )
        commands = []
        for number, line in enumerate(
            raw.decode("utf-8", errors="replace").splitlines(), 1
        ):
            if line.startswith("\t") or re.match(r"\s*(?:-\s*)?run:", line):
                commands.append(
                    {
                        "source_line": number,
                        "command_digest": hashlib.sha256(line.encode()).hexdigest(),
                        "source_revision": source_revision,
                    }
                )
        files.append(
            {
                "path": relative,
                "digest": hashlib.sha256(raw).hexdigest(),
                "revision": source_revision,
                "command_candidates": commands[:64],
            }
        )
    proposal = read_json(ROOT / "policy.json")
    inventory = {
        "schema_version": 1,
        "revision": revision,
        "files": files,
        "skipped_or_missing": skipped,
        "limits": {
            "files": MAX_INVENTORY_FILES,
            "per_file_bytes": MAX_FILE_BYTES,
            "total_bytes": MAX_INVENTORY_BYTES,
        },
        "authority_verified": False,
    }
    integration = """# Maintenance integration decisions

No repository code ran. Inventory command candidates are source references,
not approved commands. Missing files or ambiguous behavior need inspection.

1. Which maintenance classes, paths and protected behaviors apply?
2. Which independent edit/test/publish/branch-update/merge/revert actions
   already have a standing grant, and which require a person?
3. Which assertions, command revisions and environments must each class pass?
4. Which existing branch, PR, mailing-list or queue process carries decisions?
5. What time, model/CI/spending/retry limits and incident routes apply?

Keep the policy draft until the owner supplies decisions. Use the existing
agent, approved model route and runner. No scheduler, credential, provider
enrollment or live repository adapter has been installed.
"""
    output = Path(output)
    output.mkdir(mode=0o700)
    for name, content in (
        ("inventory.json", json.dumps(inventory, indent=2) + "\n"),
        ("policy.proposed.json", json.dumps(proposal, indent=2) + "\n"),
        ("INTEGRATION.md", integration),
    ):
        fd = os.open(
            output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
    return {
        "schema_version": 1,
        "files_inventoried": len(files),
        "authority_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--repo", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check-profile")
    check.add_argument("profile", type=Path)
    check.add_argument("--complete", action="store_true")
    assess_parser = commands.add_parser("assess-task")
    for name in ("policy", "task", "evidence", "snapshot"):
        assess_parser.add_argument("--" + name, type=Path, required=True)
    assess_parser.add_argument("--now", required=True)
    args = parser.parse_args()
    try:
        if args.command == "init":
            result = initialize(args.repo, args.output)
        elif args.command == "check-profile":
            result = check_profile(read_json(args.profile), args.complete)
        else:
            result = assess(
                *(
                    read_json(getattr(args, name))
                    for name in ("policy", "task", "evidence", "snapshot")
                ),
                args.now
            )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 1 if result.get("strict_check_passed") is False else 0
    except (OSError, ValueError, RecursionError):
        print(
            json.dumps(
                {
                    "status": "invalid",
                    "code": "INPUT_INVALID",
                    "authority_verified": False,
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
