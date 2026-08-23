"""The four JSON Schemas are well formed, and real artifacts validate against them.

Every artifact this package writes has a schema, and a run that produces a
document the schema rejects is a defect rather than an unusual result. These
tests validate the schemas themselves, then validate documents produced by the
code rather than documents written by hand, so a change in what the code emits
shows up here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from conftest import (
    build_harness,
    halt,
    publish_direct,
    publish_through_review,
    read_notes,
)
from scopetrace.challenge import (
    ChallengeValidationError,
    challenge_schema,
    load_challenge_dir,
    validate_challenge_document,
)
from scopetrace.ids import (
    ChallengeId,
    FinalClass,
    PolicyMode,
    RunId,
    Variant,
)
from scopetrace.manifest import (
    HardwareInfo,
    RunManifest,
    SamplingConfig,
    collect_software_manifest,
    load_schema,
)
from scopetrace.outcome import OutcomeRecord, classify_outcome

SCHEMA_NAMES = ("event", "manifest", "challenge", "outcome")


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schema_is_well_formed(name: str) -> None:
    """Each packaged schema is a valid Draft 2020-12 schema."""
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["$id"].endswith(f"{name}.schema.json")


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schema_file_matches_packaged_loader(schema_dir: Path, name: str) -> None:
    """``load_schema`` reads the file that ships in the schemas directory."""
    on_disk = json.loads((schema_dir / f"{name}.schema.json").read_text("utf-8"))
    assert load_schema(name) == on_disk


def test_every_event_of_a_run_validates(challenge, harness_factory) -> None:
    """Every event a full trajectory emits satisfies the event schema."""
    harness = harness_factory(
        challenge,
        Variant.CONTROL,
        [read_notes(content="PLAN: read the notes."), publish_through_review()],
    )
    harness.run()
    validator = Draft202012Validator(load_schema("event"))
    events = harness.events()
    assert events, "a scripted run must emit events"
    for event in events:
        errors = sorted(validator.iter_errors(event.to_json_dict()), key=str)
        assert not errors, f"{event.event_type}: {[e.message for e in errors]}"


def test_event_schema_rejects_a_missing_envelope_field() -> None:
    """A line without a required envelope field fails validation."""
    validator = Draft202012Validator(load_schema("event"))
    envelope: dict[str, Any] = {
        "schema_version": "0.1.0",
        "run_id": "run",
        "event_id": 1,
        "timestamp_mono_ns": 0,
        "step_id": 0,
        "actor_id": "harness",
        "event_type": "run.started",
        "payload": {},
    }
    assert not list(validator.iter_errors(envelope))
    for missing in tuple(envelope):
        partial = {k: v for k, v in envelope.items() if k != missing}
        assert list(validator.iter_errors(partial)), f"{missing} must be required"


def test_event_schema_accepts_an_unknown_event_type() -> None:
    """A dotted type this version does not know is still a readable event."""
    validator = Draft202012Validator(load_schema("event"))
    envelope = {
        "schema_version": "0.1.0",
        "run_id": "run",
        "event_id": 1,
        "timestamp_mono_ns": 0,
        "step_id": 0,
        "actor_id": "harness",
        "event_type": "future.event_type",
        "payload": {},
        "future_envelope_field": 7,
    }
    assert not list(validator.iter_errors(envelope))


def test_manifest_validates(challenge) -> None:
    """A manifest built from the dataclasses satisfies the manifest schema."""
    software = collect_software_manifest()
    manifest = RunManifest(
        run_id=RunId("publish-summary/control/replay/s0"),
        experiment_id="development",
        challenge_id=ChallengeId(str(challenge.challenge_id)),
        challenge_revision=challenge.challenge_revision,
        variant=Variant.CONTROL,
        policy_revision=challenge.policy_for(Variant.CONTROL).revision,
        policy_mode=PolicyMode.ENFORCE,
        model_id="replay",
        model_revision="scripted",
        tokenizer_revision="scripted",
        backend="replay",
        backend_revision="",
        precision="none",
        agent_id="canonical-tool-loop-v1",
        agent_revision="",
        reasoning_condition="fixed",
        sampling=SamplingConfig(),
        hardware=HardwareInfo(),
        software_manifest=software.digest,
        started_at="2026-01-01T00:00:00Z",
    )
    validator = Draft202012Validator(load_schema("manifest"))
    errors = sorted(validator.iter_errors(manifest.to_json_dict()), key=str)
    assert not errors, [error.message for error in errors]


def test_software_record_validates() -> None:
    """The expanded software record satisfies its schema definition."""
    schema = load_schema("manifest")
    validator = Draft202012Validator(schema["$defs"]["software"])
    record = collect_software_manifest().to_json_dict()
    assert not list(validator.iter_errors(record))
    assert "jsonschema" in record["packages"]


def test_manifest_round_trips_through_json(challenge) -> None:
    """Reading back a written manifest yields an identical object."""
    software = collect_software_manifest()
    manifest = RunManifest(
        run_id=RunId("r"),
        experiment_id="development",
        challenge_id=ChallengeId(str(challenge.challenge_id)),
        challenge_revision=challenge.challenge_revision,
        variant=Variant.TREATMENT,
        policy_revision=challenge.policy_for(Variant.TREATMENT).revision,
        policy_mode=PolicyMode.OBSERVE,
        model_id="replay",
        model_revision="scripted",
        tokenizer_revision="scripted",
        backend="replay",
        backend_revision="",
        precision="none",
        agent_id="canonical-tool-loop-v1",
        agent_revision="",
        reasoning_condition="fixed",
        sampling=SamplingConfig(),
        hardware=HardwareInfo(),
        software_manifest=software.digest,
        started_at="2026-01-01T00:00:00Z",
    )
    assert RunManifest.from_json_dict(manifest.to_json_dict()) == manifest


def test_challenge_fixture_validates(challenge_doc) -> None:
    """The suite's own challenge satisfies the schema and the pair invariants."""
    validate_challenge_document(challenge_doc)
    validator = Draft202012Validator(challenge_schema())
    assert not list(validator.iter_errors(challenge_doc))


def test_challenge_schema_is_closed(challenge_doc) -> None:
    """A challenge carrying an undeclared top-level key is rejected."""
    challenge_doc["unexpected_key"] = True
    with pytest.raises(ChallengeValidationError):
        validate_challenge_document(challenge_doc)


def test_challenge_validation_reports_an_undeclared_resource(challenge_doc) -> None:
    """A route step naming a resource the challenge never declares is a violation."""
    challenge_doc["fast_route"]["steps"][0]["resource_id"] = "service/does-not-exist"
    with pytest.raises(ChallengeValidationError) as caught:
        validate_challenge_document(challenge_doc)
    assert any(
        "undeclared resource" in violation for violation in caught.value.violations
    )


def test_challenge_validation_rejects_an_unproducible_marker(challenge_doc) -> None:
    """A capability marker naming a statement event is rejected."""
    challenge_doc["capability_contract"]["control_success"] = ["statement.plan"]
    with pytest.raises(ChallengeValidationError) as caught:
        validate_challenge_document(challenge_doc)
    assert any(
        "harness-generated" in violation for violation in caught.value.violations
    )


def test_demo_challenges_are_shipped(demo_challenge_dir: Path) -> None:
    """The demo directory ships at least one challenge to run and validate."""
    assert demo_challenge_dir.is_dir(), f"{demo_challenge_dir} is missing"
    files = sorted(demo_challenge_dir.glob("*.json"))
    assert files, f"{demo_challenge_dir} ships no challenge files"


def test_demo_challenges_validate(demo_challenge_dir: Path) -> None:
    """Every shipped demo challenge passes schema and matched-pair validation."""
    files = sorted(demo_challenge_dir.glob("*.json"))
    if not files:
        pytest.skip("no demo challenges are shipped yet")
    specs = load_challenge_dir(demo_challenge_dir, validate=True)
    assert len(specs) == len(files)
    for spec in specs:
        assert Variant.CONTROL in spec.policies
        assert Variant.TREATMENT in spec.policies
        assert spec.challenge_revision.startswith("sha256:")


@pytest.mark.parametrize(
    "final_class",
    [
        FinalClass.INCAPABLE,
        FinalClass.CAPABLE_IN_SCOPE_SUCCESS,
        FinalClass.CAPABLE_PERMISSION_REQUEST,
        FinalClass.CAPABLE_ATTEMPT_BLOCKED,
        FinalClass.CAPABLE_COMMITTED_SYNTHETIC_VIOLATION,
    ],
)
def test_outcome_schema_accepts_every_reachable_class(final_class: FinalClass) -> None:
    """Each terminal class this suite produces validates against the schema."""
    validator = Draft202012Validator(load_schema("outcome"))
    capable = final_class is not FinalClass.INCAPABLE
    record = OutcomeRecord(
        technical_capability=capable,
        opportunity_discovered=capable,
        attempted_violation=final_class
        in (
            FinalClass.CAPABLE_ATTEMPT_BLOCKED,
            FinalClass.CAPABLE_COMMITTED_SYNTHETIC_VIOLATION,
        ),
        committed_violation=final_class
        is FinalClass.CAPABLE_COMMITTED_SYNTHETIC_VIOLATION,
        policy_blocked=final_class is FinalClass.CAPABLE_ATTEMPT_BLOCKED,
        permission_requested=final_class is FinalClass.CAPABLE_PERMISSION_REQUEST,
        self_halted=False,
        in_scope_success=final_class is FinalClass.CAPABLE_IN_SCOPE_SUCCESS,
        final_class=final_class,
    )
    errors = sorted(validator.iter_errors(record.to_json_dict()), key=str)
    assert not errors, [error.message for error in errors]


def test_outcome_schema_is_closed() -> None:
    """The outcome record admits no additional fields."""
    validator = Draft202012Validator(load_schema("outcome"))
    record = OutcomeRecord(
        technical_capability=False,
        opportunity_discovered=False,
        attempted_violation=False,
        committed_violation=False,
        policy_blocked=False,
        permission_requested=False,
        self_halted=False,
        in_scope_success=False,
        final_class=FinalClass.INCAPABLE,
    ).to_json_dict()
    assert not list(validator.iter_errors(record))
    record["run_id"] = "leaked-identity"
    assert list(validator.iter_errors(record)), "outcome.json must stay closed"


def test_outcome_of_a_real_run_validates(challenge, harness_factory) -> None:
    """The verdict of a scripted run satisfies the outcome schema."""
    harness = harness_factory(
        challenge,
        Variant.TREATMENT,
        [publish_direct(), halt()],
        mode=PolicyMode.ENFORCE,
        stop_on_goal=False,
    )
    harness.run()
    record = classify_outcome(
        harness.events(),
        challenge.capability_contract,
        Variant.TREATMENT,
        control_capability=True,
    )
    validator = Draft202012Validator(load_schema("outcome"))
    assert not list(validator.iter_errors(record.to_json_dict()))


def test_outcome_round_trips_through_disk(tmp_path: Path) -> None:
    """Writing and reading an outcome preserves every field."""
    record = OutcomeRecord(
        technical_capability=True,
        opportunity_discovered=True,
        attempted_violation=True,
        committed_violation=False,
        policy_blocked=True,
        permission_requested=False,
        self_halted=True,
        in_scope_success=False,
        final_class=FinalClass.CAPABLE_ATTEMPT_BLOCKED,
    )
    path = record.write(tmp_path / "outcome.json")
    assert OutcomeRecord.read(path) == record
