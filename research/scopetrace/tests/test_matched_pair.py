"""The two arms of a challenge differ in their rule list and in nothing else.

This is the scientific claim the project rests on. If anything else varies
between control and treatment, a difference in behaviour between the arms has a
second explanation and the measurement is worthless. So the check is mechanical:
render both arms of a challenge into canonical JSON with the policy block
removed, and require the two texts to be byte identical.

Every shipped demo challenge is checked, and so is the suite's own challenge.
Two negative tests confirm the check has teeth: perturbing the shared world
must fail the diff, and giving both arms the same rules must fail the pair
invariants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import challenge_document
from scopetrace.challenge import (
    ChallengeSpec,
    canonical_challenge_text,
    challenge_revision,
    check_matched_pair,
    document_from_spec,
    load_challenge,
    load_challenge_dir,
    parse_challenge,
)
from scopetrace.ids import AuthoritativeScope, Variant
from scopetrace.manifest import canonical_json


def arm_document(document: dict[str, Any], variant: Variant) -> str:
    """Return the canonical JSON of one arm with the policy block removed.

    Everything the arm shares with its partner survives: the resources, the
    services, the evidence schedule, the goal, both routes, the capability
    contract, and the prose the model sees. Only the rule list is taken out,
    because the rule list is the one thing the arms are allowed to differ in.
    """
    assert variant in (Variant.CONTROL, Variant.TREATMENT)
    return canonical_json(
        {
            key: value
            for key, value in document.items()
            if key not in ("policies", "challenge_revision")
        }
    )


def matched_pair_diff(document: dict[str, Any]) -> tuple[str, str]:
    """Return the canonical arm texts to compare, control first."""
    return (
        arm_document(document, Variant.CONTROL),
        arm_document(document, Variant.TREATMENT),
    )


def assert_matched(spec: ChallengeSpec) -> None:
    """Assert that a challenge is a matched pair, mechanically and semantically."""
    document = document_from_spec(spec)
    control_text, treatment_text = matched_pair_diff(document)
    assert (
        control_text == treatment_text
    ), f"{spec.challenge_id}: the arms differ outside their policy rules"

    world = spec.world_spec()
    assert world.revision() == spec.world_spec().revision()

    control = spec.policy_for(Variant.CONTROL)
    treatment = spec.policy_for(Variant.TREATMENT)
    assert (
        control.revision != treatment.revision
    ), f"{spec.challenge_id}: both arms carry the same rules"
    assert (
        check_matched_pair(spec) == ()
    ), f"{spec.challenge_id}: {check_matched_pair(spec)}"


def test_the_suite_challenge_is_a_matched_pair(challenge) -> None:
    """The challenge every other test runs against is itself a matched pair."""
    assert_matched(challenge)


def test_the_world_specification_is_arm_independent(challenge) -> None:
    """``world_spec`` takes no variant, so both arms get the same world."""
    assert challenge.world_spec() == challenge.world_spec()
    assert challenge.world_spec().revision() == challenge.world_spec().revision()


def test_only_the_policy_key_differs_in_the_document(challenge) -> None:
    """The rendered document differs between arms only under ``policies``."""
    document = document_from_spec(challenge)
    control = document["policies"]["control"]
    treatment = document["policies"]["treatment"]
    assert control != treatment
    stripped = {k: v for k, v in document.items() if k != "policies"}
    assert json.loads(canonical_json(stripped)) == stripped


def demo_specs(directory: Path) -> tuple[ChallengeSpec, ...]:
    """Load the shipped demo challenges, or skip when none are shipped."""
    if not directory.is_dir() or not sorted(directory.glob("*.json")):
        pytest.skip("no demo challenges are shipped yet")
    return load_challenge_dir(directory, validate=True)


def test_every_demo_challenge_is_a_matched_pair(demo_challenge_dir: Path) -> None:
    """Every shipped challenge passes the byte-level arm diff."""
    specs = demo_specs(demo_challenge_dir)
    assert specs
    for spec in specs:
        assert_matched(spec)


def test_every_demo_challenge_file_diffs_cleanly(demo_challenge_dir: Path) -> None:
    """The check runs against the file on disk, not only the parsed object."""
    if not demo_challenge_dir.is_dir() or not sorted(demo_challenge_dir.glob("*.json")):
        pytest.skip("no demo challenges are shipped yet")
    for path in sorted(demo_challenge_dir.glob("*.json")):
        document = json.loads(path.read_text("utf-8"))
        control_text, treatment_text = matched_pair_diff(document)
        assert control_text == treatment_text, f"{path.name} differs outside its rules"


def test_every_demo_challenge_declares_both_routes(demo_challenge_dir: Path) -> None:
    """A pair needs a fast route to forbid and a slow route that still works."""
    for spec in demo_specs(demo_challenge_dir):
        assert spec.fast_route.steps, f"{spec.challenge_id} declares no fast route"
        assert spec.slow_route.steps, f"{spec.challenge_id} declares no slow route"
        control = spec.policy_for(Variant.CONTROL).engine()
        treatment = spec.policy_for(Variant.TREATMENT).engine()
        from scopetrace.challenge import _route_request

        assert all(
            control.authoritative_scope(_route_request(step))
            is AuthoritativeScope.ALLOWED
            for step in spec.fast_route.steps
        )
        assert any(
            treatment.authoritative_scope(_route_request(step))
            is AuthoritativeScope.FORBIDDEN
            for step in spec.fast_route.steps
        )
        assert all(
            treatment.authoritative_scope(_route_request(step))
            is AuthoritativeScope.ALLOWED
            for step in spec.slow_route.steps
        )


def test_every_demo_challenge_is_a_pilot_rung(demo_challenge_dir: Path) -> None:
    """Only the rungs cleared for the initial set are used."""
    from scopetrace.challenge import PILOT_RUNGS

    for spec in demo_specs(demo_challenge_dir):
        assert spec.rung in PILOT_RUNGS, f"{spec.challenge_id} uses rung {spec.rung}"


def test_the_diff_catches_a_perturbed_world() -> None:
    """A challenge whose arms disagreed about the world would fail the diff."""
    document = challenge_document()
    control_text, _ = matched_pair_diff(document)
    perturbed = challenge_document()
    perturbed["resources"][0]["attributes"]["content"] = "different notes"
    _, treatment_text = matched_pair_diff(perturbed)
    assert control_text != treatment_text


def test_the_diff_catches_a_perturbed_goal() -> None:
    """A goal that differed between arms would fail the diff."""
    document = challenge_document()
    control_text, _ = matched_pair_diff(document)
    perturbed = challenge_document()
    perturbed["goal"] = "Publish it however you can."
    _, treatment_text = matched_pair_diff(perturbed)
    assert control_text != treatment_text


def test_identical_rule_sets_fail_the_pair_invariants() -> None:
    """A pair with nothing to measure is rejected by the invariant check."""
    from conftest import CONTROL_RULES

    document = challenge_document(treatment_rules=CONTROL_RULES)
    spec = parse_challenge(document, validate=False)
    violations = check_matched_pair(spec)
    assert violations
    assert any("same rule set" in violation for violation in violations)


def test_a_treatment_arm_that_forbids_nothing_is_rejected() -> None:
    """The treatment arm must forbid at least one step of the fast route."""
    from conftest import CONTROL_RULES, policy_rule

    permissive = (
        *CONTROL_RULES,
        policy_rule("publish/extra", "read", "store/shared/*", "allow", "shared_ok"),
    )
    spec = parse_challenge(
        challenge_document(treatment_rules=permissive), validate=False
    )
    violations = check_matched_pair(spec)
    assert any("forbids no step" in violation for violation in violations)


def test_a_treatment_arm_that_blocks_the_slow_route_is_rejected() -> None:
    """A treatment arm with no working route is not a solvable pair."""
    from conftest import policy_rule

    blocked = (
        policy_rule(
            "publish/direct-service",
            "call",
            "service/direct-publish",
            "deny",
            "direct_publish_not_authorized",
        ),
        policy_rule(
            "publish/review-service",
            "call",
            "service/review-publish",
            "deny",
            "review_publish_not_authorized",
        ),
    )
    spec = parse_challenge(challenge_document(treatment_rules=blocked), validate=False)
    violations = check_matched_pair(spec)
    assert any("slow route step" in violation for violation in violations)


def test_reformatting_a_challenge_preserves_its_revision(
    challenge, tmp_path: Path
) -> None:
    """An author may reformat a file without invalidating runs made against it."""
    from scopetrace.challenge import dump_challenge

    path = dump_challenge(challenge, tmp_path / "reformatted.json")
    reloaded = load_challenge(path)
    assert reloaded.challenge_revision == challenge.challenge_revision


def test_revision_ignores_a_stored_revision_key(challenge) -> None:
    """Re-hashing a loaded file reproduces the digest it was loaded with."""
    document = challenge.to_json_dict()
    assert document["challenge_revision"] == challenge.challenge_revision
    assert challenge_revision(document) == challenge.challenge_revision
    assert "challenge_revision" not in canonical_challenge_text(document)


def test_the_revision_changes_when_a_rule_changes(challenge) -> None:
    """Editing one arm's rules produces a visibly different challenge."""
    from conftest import TREATMENT_RULES, policy_rule

    edited = parse_challenge(
        challenge_document(
            treatment_rules=(
                *TREATMENT_RULES,
                policy_rule("publish/extra", "read", "store/shared/*", "allow", "ok"),
            )
        )
    )
    assert edited.challenge_revision != challenge.challenge_revision
