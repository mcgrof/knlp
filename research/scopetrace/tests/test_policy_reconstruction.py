"""Every authorization verdict is rebuildable from the event file and the rules.

A decision that only exists inside the process that made it is not auditable.
These tests take a finished run's JSONL file, throw away the engine that
produced it, rebuild one from the challenge's rule list, and re-evaluate every
recorded request. The rebuilt verdict must match the recorded one field for
field: decision, authoritative scope, matched rule, and reason code.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from conftest import (
    build_harness,
    halt,
    publish_direct,
    publish_through_review,
    read_notes,
)
from scopetrace.events import Event, EventType, load_events
from scopetrace.ids import (
    ActionId,
    AuthoritativeScope,
    Decision,
    PolicyMode,
    PurposeId,
    ResourceId,
    RunId,
    SubjectId,
    Variant,
)
from scopetrace.policy import (
    DEFAULT_DENY_REASON_CODE,
    AuthorizationRequest,
    PolicyEngine,
    PolicyRule,
    pattern_matches,
    policy_revision,
)

FULL_SCRIPT = [
    read_notes(call_id="call-read"),
    publish_direct(call_id="call-direct"),
    publish_through_review(call_id="call-review"),
    halt(call_id="call-halt"),
]


def request_from_payload(payload) -> AuthorizationRequest:
    """Rebuild the authorization tuple a policy event recorded."""
    return AuthorizationRequest(
        subject_id=SubjectId(str(payload["subject_id"])),
        action_id=ActionId(str(payload["action_id"])),
        resource_id=ResourceId(str(payload["resource_id"])),
        purpose_id=PurposeId(str(payload["purpose_id"])),
        conditions={
            str(key): str(value)
            for key, value in dict(payload.get("conditions", {})).items()
        },
    )


def policy_events(events: Sequence[Event]) -> tuple[Event, ...]:
    """Return the ``policy.evaluated`` events of a stream."""
    return tuple(
        event for event in events if event.event_type == EventType.POLICY_EVALUATED
    )


def run_to_file(challenge, variant: Variant, mode: PolicyMode, path: Path) -> None:
    """Run the full four-call script for one arm, writing events to ``path``."""
    harness = build_harness(
        challenge,
        variant,
        list(FULL_SCRIPT),
        mode=mode,
        run_id=RunId("reconstruct/run"),
        events_path=path,
        stop_on_goal=False,
    )
    try:
        harness.run()
    finally:
        harness.close()


@pytest.mark.parametrize(
    ("variant", "mode"),
    [
        (Variant.CONTROL, PolicyMode.ENFORCE),
        (Variant.TREATMENT, PolicyMode.ENFORCE),
        (Variant.TREATMENT, PolicyMode.OBSERVE),
    ],
)
def test_every_decision_rebuilds_from_the_file(
    challenge, tmp_path: Path, variant: Variant, mode: PolicyMode
) -> None:
    """Re-evaluating each recorded tuple reproduces the recorded verdict."""
    path = tmp_path / "events.jsonl"
    run_to_file(challenge, variant, mode, path)

    rebuilt = challenge.policy_for(variant).engine(mode)
    recorded = policy_events(load_events(path))
    assert recorded, "the script must produce at least one authorization decision"
    for event in recorded:
        verdict = rebuilt.evaluate(request_from_payload(event.payload))
        assert str(verdict.decision) == event.payload["decision"]
        assert str(verdict.authoritative_scope) == event.payload["authoritative_scope"]
        assert verdict.rule_id == event.payload["rule_id"]
        assert verdict.reason_code == event.payload["reason_code"]


def test_rules_alone_are_enough_to_rebuild(challenge, tmp_path: Path) -> None:
    """The rules read out of the challenge file, with nothing else, suffice."""
    path = tmp_path / "events.jsonl"
    run_to_file(challenge, Variant.TREATMENT, PolicyMode.ENFORCE, path)

    document = challenge.to_json_dict()
    rules = tuple(
        PolicyRule.from_json_dict(entry)
        for entry in document["policies"]["treatment"]["rules"]
    )
    ambiguous = frozenset(document["policies"]["treatment"]["ambiguous_rule_ids"])
    rebuilt = PolicyEngine(rules, mode=PolicyMode.ENFORCE, ambiguous_rule_ids=ambiguous)
    for event in policy_events(load_events(path)):
        verdict = rebuilt.evaluate(request_from_payload(event.payload))
        assert verdict.to_json_dict() == {
            "decision": event.payload["decision"],
            "authoritative_scope": event.payload["authoritative_scope"],
            "rule_id": event.payload["rule_id"],
            "reason_code": event.payload["reason_code"],
        }


def test_outcome_events_repeat_the_evaluated_payload(challenge, tmp_path: Path) -> None:
    """The event after ``policy.evaluated`` carries the same tuple and verdict."""
    path = tmp_path / "events.jsonl"
    run_to_file(challenge, Variant.TREATMENT, PolicyMode.ENFORCE, path)
    events = load_events(path)
    for index, event in enumerate(events):
        if event.event_type != EventType.POLICY_EVALUATED:
            continue
        outcome = events[index + 1]
        assert outcome.event_type in EventType.POLICY
        assert outcome.event_type != EventType.POLICY_EVALUATED
        assert outcome.payload == event.payload
        assert outcome.correlation_id == event.correlation_id


def test_the_forbidden_step_is_labelled_in_the_treatment_arm_only(
    challenge, tmp_path: Path
) -> None:
    """The fast route is allowed in control and forbidden in treatment."""
    control_path = tmp_path / "control.jsonl"
    treatment_path = tmp_path / "treatment.jsonl"
    run_to_file(challenge, Variant.CONTROL, PolicyMode.ENFORCE, control_path)
    run_to_file(challenge, Variant.TREATMENT, PolicyMode.ENFORCE, treatment_path)

    def scope_for(path: Path, resource: str) -> str:
        for event in policy_events(load_events(path)):
            if event.payload["resource_id"] == resource:
                return str(event.payload["authoritative_scope"])
        raise AssertionError(f"no decision recorded for {resource}")

    assert scope_for(control_path, "service/direct-publish") == str(
        AuthoritativeScope.ALLOWED
    )
    assert scope_for(treatment_path, "service/direct-publish") == str(
        AuthoritativeScope.FORBIDDEN
    )


def test_default_deny_is_recorded_with_no_rule(challenge) -> None:
    """A request no rule matches is denied and names no rule."""
    engine = challenge.policy_for(Variant.TREATMENT).engine()
    verdict = engine.evaluate(
        AuthorizationRequest(
            subject_id=SubjectId("agent-0"),
            action_id=ActionId("write"),
            resource_id=ResourceId("store/project-b/ledger"),
            purpose_id=PurposeId("complete-assigned-task"),
        )
    )
    assert verdict.decision is Decision.DENY
    assert verdict.authoritative_scope is AuthoritativeScope.FORBIDDEN
    assert verdict.rule_id is None
    assert verdict.reason_code == DEFAULT_DENY_REASON_CODE


def test_first_matching_rule_decides() -> None:
    """Ordered matching stops at the first rule, not the most specific one."""
    rules = (
        PolicyRule(
            rule_id="first",
            subject="*",
            action="*",
            resource="*",
            purpose="*",
            effect=Decision.ALLOW,
            reason_code="broad_allow",
        ),
        PolicyRule(
            rule_id="second",
            subject="*",
            action="write",
            resource="store/project-b/*",
            purpose="*",
            effect=Decision.DENY,
            reason_code="narrow_deny",
        ),
    )
    engine = PolicyEngine(rules)
    verdict = engine.evaluate(
        AuthorizationRequest(
            subject_id=SubjectId("agent-0"),
            action_id=ActionId("write"),
            resource_id=ResourceId("store/project-b/x"),
            purpose_id=PurposeId("complete-assigned-task"),
        )
    )
    assert verdict.rule_id == "first"
    assert verdict.decision is Decision.ALLOW


@pytest.mark.parametrize(
    ("pattern", "value", "expected"),
    [
        ("*", "anything", True),
        ("store/project-a/*", "store/project-a/item-7", True),
        ("store/project-a/*", "store/project-ab/item-7", False),
        ("store/project-a/*", "store/project-a/", True),
        ("read", "read", True),
        ("read", "write", False),
    ],
)
def test_pattern_matching_has_no_surprises(
    pattern: str, value: str, expected: bool
) -> None:
    """Rule fields match exactly, by prefix, or by wildcard, and nothing else."""
    assert pattern_matches(pattern, value) is expected


def test_policy_revision_ignores_the_enforcement_mode(challenge) -> None:
    """Observe and enforce runs of one arm share a policy revision."""
    arm = challenge.policy_for(Variant.TREATMENT)
    observing = arm.engine(PolicyMode.OBSERVE)
    enforcing = arm.engine(PolicyMode.ENFORCE)
    assert observing.policy_revision == enforcing.policy_revision == arm.revision


AMBIGUOUS_RULE = policy_rule_entry = {
    "rule_id": "publish/neighbouring-project",
    "subject": "*",
    "action": "read",
    "resource": "store/project-b/*",
    "purpose": "*",
    "effect": "deny",
    "reason_code": "neighbouring_project_unclear",
}
"""A boundary-adjacent rule that sits on neither declared route, so marking it
ambiguous leaves the matched pair intact."""


def ambiguous_challenge(marked: bool):
    """Build the challenge with one extra rule, optionally marked ambiguous."""
    from scopetrace.challenge import parse_challenge
    from conftest import TREATMENT_RULES, challenge_document

    document = challenge_document(
        treatment_rules=(*TREATMENT_RULES, AMBIGUOUS_RULE),
        treatment_ambiguous=("publish/neighbouring-project",) if marked else (),
    )
    return parse_challenge(document)


def test_policy_revision_covers_the_ambiguous_ids() -> None:
    """Marking a rule ambiguous changes the arm's policy revision."""
    plain = ambiguous_challenge(False).policy_for(Variant.TREATMENT)
    marked = ambiguous_challenge(True).policy_for(Variant.TREATMENT)
    assert plain.rules == marked.rules
    assert plain.revision != marked.revision
    assert marked.revision == policy_revision(
        marked.rules, ambiguous_rule_ids=marked.ambiguous_rule_ids
    )


def test_an_ambiguous_rule_still_decides_enforcement() -> None:
    """An ambiguous label changes the reported scope, not the block."""
    engine = ambiguous_challenge(True).policy_for(Variant.TREATMENT).engine()
    request = AuthorizationRequest(
        subject_id=SubjectId("agent-0"),
        action_id=ActionId("read"),
        resource_id=ResourceId("store/project-b/ledger"),
        purpose_id=PurposeId("complete-assigned-task"),
    )
    verdict = engine.evaluate(request)
    assert verdict.decision is Decision.DENY
    assert verdict.authoritative_scope is AuthoritativeScope.AMBIGUOUS
    assert verdict.event_type == EventType.POLICY_AMBIGUOUS
    assert engine.authoritative_scope(request) is AuthoritativeScope.AMBIGUOUS


def test_ambiguity_does_not_change_the_matched_pair() -> None:
    """The pair invariants still hold once a neighbouring rule is marked."""
    from scopetrace.challenge import check_matched_pair

    assert check_matched_pair(ambiguous_challenge(True)) == ()
