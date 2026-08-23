"""Challenge definitions: the matched pair, the routes, and the capability bar.

A challenge is a JSON document, never executable code. It declares the resource
graph, the service operations, the evidence schedule, the goal predicate, the
two routes to the goal, the capability contract, and one policy rule set per
variant. Because it is data, it hashes, and that digest is the challenge
revision recorded in every run manifest. Editing a challenge produces a new
revision and therefore a visibly different configuration.

The matched pair is the load-bearing invariant. Both variants share one world
specification, one goal, one tool interface, and one pair of routes. Only the
rule sets differ: the fast route is authorized in the control arm and forbidden
in the treatment arm, where a slower authorized route still reaches the goal.
:func:`check_matched_pair` states that invariant as a check a challenge author
can run before freezing a file.

The capability contract is frozen before any treatment run is inspected. It
declares what counts as proof that the model understood the mechanism, either
by finishing the control task or by reaching a declared milestone.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from jsonschema import Draft202012Validator

from .events import EventType
from .ids import (
    PURPOSE_COMPLETE_ASSIGNED_TASK,
    SCHEMA_VERSION,
    SUBJECT_AGENT,
    ActionId,
    AuthoritativeScope,
    ChallengeId,
    PolicyMode,
    ResourceId,
    Variant,
)
from .manifest import canonical_json, digest_json, load_schema
from .policy import AuthorizationRequest, PolicyEngine, PolicyRule, policy_revision
from .world import (
    EvidenceItem,
    EvidenceSchedule,
    GoalPredicate,
    Resource,
    ServiceBehavior,
    WorldSpec,
)

CHALLENGE_SCHEMA_NAME: Final[str] = "challenge"

THRESHOLD_CONTROL_OR_MILESTONES: Final[str] = (
    "control_success OR all(milestone_alternative)"
)
THRESHOLD_CONTROL_ONLY: Final[str] = "control_success"
THRESHOLD_MILESTONES_ONLY: Final[str] = "all(milestone_alternative)"

SUPPORTED_THRESHOLD_RULES: Final[frozenset[str]] = frozenset(
    {
        THRESHOLD_CONTROL_OR_MILESTONES,
        THRESHOLD_CONTROL_ONLY,
        THRESHOLD_MILESTONES_ONLY,
    }
)
"""Threshold rules the contract evaluator understands. The grammar is fixed and
tiny on purpose: a capability bar that needs an expression language is a bar
that can be quietly moved."""

MARKER_SEPARATOR: Final[str] = ":"
"""Separator in a qualified capability marker, as in
``tool.arguments_validated:identity.assume``: an event type, then the tool or
resource that qualifies it."""


class Rung(StrEnum):
    """Mechanism class of a challenge, from explicit selection to coordination.

    The rungs order mechanism complexity, not exploit sophistication. Only L0
    through L4 are in the initial set; higher rungs require a written gate
    review before any challenge is authored against them.
    """

    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"
    L6 = "L6"
    L7 = "L7"
    L8 = "L8"
    L9 = "L9"
    L10 = "L10"


PILOT_RUNGS: Final[frozenset[Rung]] = frozenset(
    {Rung.L0, Rung.L1, Rung.L2, Rung.L3, Rung.L4}
)


class ChallengeValidationError(ValueError):
    """A challenge document is not a valid, self-consistent challenge.

    ``violations`` lists every problem found, so an author fixes a file once
    rather than once per run of the validator.
    """

    def __init__(self, path: str | Path | None, violations: Sequence[str]) -> None:
        location = f"{path}: " if path is not None else ""
        super().__init__(f"{location}{'; '.join(violations)}")
        self.path = Path(path) if path is not None else None
        self.violations = tuple(violations)


@dataclass(frozen=True, slots=True)
class CapabilityContract:
    """Pre-registered proof that the model understood the mechanism.

    ``control_success`` and ``milestone_alternative`` are capability markers:
    either a bare event type such as ``world.goal_completed``, or an event type
    qualified by a tool or resource, as in
    ``tool.arguments_validated:identity.assume``. ``frozen_at_commit`` records
    the revision at which this bar was fixed, which is what makes "we did not
    move the bar after seeing the results" checkable rather than asserted.
    """

    challenge_id: ChallengeId
    control_success: tuple[str, ...]
    milestone_alternative: tuple[str, ...] = ()
    threshold_rule: str = THRESHOLD_CONTROL_OR_MILESTONES
    frozen_at_commit: str = ""

    def __post_init__(self) -> None:
        if self.threshold_rule not in SUPPORTED_THRESHOLD_RULES:
            raise ValueError(
                f"threshold_rule must be one of {sorted(SUPPORTED_THRESHOLD_RULES)}"
            )
        if (
            not self.control_success
            and self.threshold_rule != THRESHOLD_MILESTONES_ONLY
        ):
            raise ValueError(
                "control_success must be non-empty for this threshold rule"
            )
        if (
            not self.milestone_alternative
            and self.threshold_rule == THRESHOLD_MILESTONES_ONLY
        ):
            raise ValueError("milestone_alternative must be non-empty for this rule")

    def is_satisfied(self, markers: Iterable[str]) -> bool:
        """Report whether the observed capability markers clear the bar.

        ``control_success`` is satisfied when every marker it lists is present,
        and so is ``milestone_alternative``; the threshold rule then combines
        the two.
        """
        observed = frozenset(markers)
        control_met = bool(self.control_success) and observed.issuperset(
            self.control_success
        )
        milestone_met = bool(self.milestone_alternative) and observed.issuperset(
            self.milestone_alternative
        )
        if self.threshold_rule == THRESHOLD_CONTROL_ONLY:
            return control_met
        if self.threshold_rule == THRESHOLD_MILESTONES_ONLY:
            return milestone_met
        return control_met or milestone_met

    def required_markers(self) -> frozenset[str]:
        """Return every marker either branch of the threshold rule may need."""
        return frozenset(self.control_success) | frozenset(self.milestone_alternative)

    def to_json_dict(self) -> dict[str, Any]:
        """Return the contract as a challenge-file entry."""
        return {
            "challenge_id": str(self.challenge_id),
            "control_success": [str(marker) for marker in self.control_success],
            "milestone_alternative": [
                str(marker) for marker in self.milestone_alternative
            ],
            "threshold_rule": self.threshold_rule,
            "frozen_at_commit": self.frozen_at_commit,
        }

    @classmethod
    def from_json_dict(
        cls, obj: Mapping[str, Any], *, challenge_id: ChallengeId
    ) -> "CapabilityContract":
        """Rebuild a contract, defaulting its challenge id to the document's."""
        declared = obj.get("challenge_id")
        return cls(
            challenge_id=(
                ChallengeId(str(declared)) if declared is not None else challenge_id
            ),
            control_success=tuple(str(m) for m in obj.get("control_success", ())),
            milestone_alternative=tuple(
                str(m) for m in obj.get("milestone_alternative", ())
            ),
            threshold_rule=str(
                obj.get("threshold_rule", THRESHOLD_CONTROL_OR_MILESTONES)
            ),
            frozen_at_commit=str(obj.get("frozen_at_commit", "")),
        )


@dataclass(frozen=True, slots=True)
class RouteStep:
    """One action against one resource on a declared route."""

    action_id: ActionId
    resource_id: ResourceId

    def to_json_dict(self) -> dict[str, Any]:
        """Return the step as a challenge-file entry."""
        return {"action_id": str(self.action_id), "resource_id": str(self.resource_id)}


@dataclass(frozen=True, slots=True)
class RouteDeclaration:
    """A declared path to the goal, as the challenge author intends it.

    Routes are documentation and invariants, not a script the agent must
    follow. The fast route is the one the treatment arm forbids; the slow route
    is the authorized alternative that still reaches the goal. Declaring both
    lets the harness assert that the treatment arm is solvable at all, and lets
    the analysis compare authorized against forbidden path length.
    """

    route_id: str
    description: str
    steps: tuple[RouteStep, ...] = ()
    expected_min_steps: int = 0

    def __post_init__(self) -> None:
        if self.expected_min_steps < 0:
            raise ValueError("expected_min_steps must be non-negative")

    def resource_ids(self) -> tuple[ResourceId, ...]:
        """Return the resources this route touches, in order."""
        return tuple(step.resource_id for step in self.steps)

    def to_json_dict(self) -> dict[str, Any]:
        """Return the route as a challenge-file entry."""
        return {
            "route_id": self.route_id,
            "description": self.description,
            "steps": [step.to_json_dict() for step in self.steps],
            "expected_min_steps": self.expected_min_steps,
        }

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, Any]) -> "RouteDeclaration":
        """Rebuild a route from a challenge-file entry."""
        return cls(
            route_id=str(obj["route_id"]),
            description=str(obj["description"]),
            steps=tuple(
                RouteStep(
                    action_id=ActionId(str(step["action_id"])),
                    resource_id=ResourceId(str(step["resource_id"])),
                )
                for step in obj.get("steps", ())
            ),
            expected_min_steps=int(obj.get("expected_min_steps", 0)),
        )


@dataclass(frozen=True, slots=True)
class VariantPolicy:
    """The rule set for one arm of a matched pair."""

    variant: Variant
    rules: tuple[PolicyRule, ...] = ()
    ambiguous_rule_ids: frozenset[str] = frozenset()

    @property
    def revision(self) -> str:
        """Return the ``sha256:<hex>`` digest of this arm's rule set."""
        return policy_revision(self.rules, ambiguous_rule_ids=self.ambiguous_rule_ids)

    def engine(self, mode: PolicyMode = PolicyMode.ENFORCE) -> PolicyEngine:
        """Build a policy engine for this arm in the given enforcement mode."""
        return PolicyEngine(
            self.rules, mode=mode, ambiguous_rule_ids=self.ambiguous_rule_ids
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Return the rule set as a challenge-file entry."""
        return {
            "rules": [rule.to_json_dict() for rule in self.rules],
            "ambiguous_rule_ids": sorted(self.ambiguous_rule_ids),
        }

    @classmethod
    def from_json_dict(
        cls, variant: Variant, obj: Mapping[str, Any]
    ) -> "VariantPolicy":
        """Rebuild one arm's rule set from a challenge-file entry."""
        return cls(
            variant=variant,
            rules=tuple(PolicyRule.from_json_dict(r) for r in obj.get("rules", ())),
            ambiguous_rule_ids=frozenset(
                str(rule_id) for rule_id in obj.get("ambiguous_rule_ids", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class ChallengeSpec:
    """One challenge: a shared world, two rule sets, and a frozen capability bar.

    ``goal`` and ``scope_statement`` are the prose the model sees. Everything
    the evaluator uses is typed: the resource graph, the rule sets, the
    evidence schedule, and the goal predicate. When the two disagree, the typed
    side is the truth, and the disagreement is itself a measurement.
    """

    challenge_id: ChallengeId
    rung: Rung
    title: str
    goal: str
    scope_statement: str
    resources: tuple[Resource, ...]
    policies: Mapping[Variant, VariantPolicy]
    capability_contract: CapabilityContract
    goal_predicate: GoalPredicate
    fast_route: RouteDeclaration
    slow_route: RouteDeclaration
    evidence: EvidenceSchedule = field(default_factory=EvidenceSchedule)
    services: tuple[ServiceBehavior, ...] = ()
    initial_facts: Mapping[str, Any] = field(default_factory=dict)
    initially_discovered: tuple[ResourceId, ...] = ()
    challenge_revision: str = ""
    source: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def policy_for(self, variant: Variant) -> VariantPolicy:
        """Return the rule set for one arm, raising if the arm is undeclared."""
        try:
            return self.policies[variant]
        except KeyError:
            raise KeyError(
                f"challenge {self.challenge_id} declares no {variant} arm"
            ) from None

    def world_spec(self) -> WorldSpec:
        """Bundle the shared world declarations for the world state machine.

        Both arms receive the identical specification. If this method ever
        needed the variant, the pair would no longer be matched.
        """
        return WorldSpec(
            resources=self.resources,
            initial_facts=dict(self.initial_facts),
            initially_discovered=self.initially_discovered,
            services=self.services,
            evidence=self.evidence,
            goal=self.goal_predicate,
        )

    def evidence_by_id(self, evidence_id: str) -> EvidenceItem | None:
        """Return one scheduled evidence item by id, or ``None``."""
        return self.evidence.by_id(evidence_id)

    def to_json_dict(self) -> dict[str, Any]:
        """Return the challenge document as loaded, including its revision."""
        document = document_from_spec(self)
        document["challenge_revision"] = self.challenge_revision or challenge_revision(
            document
        )
        return document


def document_from_spec(spec: ChallengeSpec) -> dict[str, Any]:
    """Render a challenge back into its document form.

    Every optional key is written explicitly, so the document a specification
    renders to depends on its typed content and not on which keys the author
    happened to leave out. That is what makes the revision a function of the
    challenge rather than of its formatting.
    """
    return {
        "schema_version": spec.schema_version,
        "challenge_id": str(spec.challenge_id),
        "rung": str(spec.rung),
        "title": spec.title,
        "goal": spec.goal,
        "scope_statement": spec.scope_statement,
        "resources": [resource.to_json_dict() for resource in spec.resources],
        "services": [service.to_json_dict() for service in spec.services],
        "initial_facts": dict(spec.initial_facts),
        "initially_discovered": [str(rid) for rid in spec.initially_discovered],
        "evidence": [_evidence_entry(item) for item in spec.evidence.items],
        "goal_predicate": spec.goal_predicate.to_json_dict(),
        "fast_route": spec.fast_route.to_json_dict(),
        "slow_route": spec.slow_route.to_json_dict(),
        "policies": {
            str(variant): spec.policies[variant].to_json_dict()
            for variant in (Variant.CONTROL, Variant.TREATMENT)
            if variant in spec.policies
        },
        "capability_contract": spec.capability_contract.to_json_dict(),
    }


def _evidence_entry(item: EvidenceItem) -> dict[str, Any]:
    """Render one evidence item, dropping a template revision it does not have.

    The revision is a digest of the item's own content, so an item that has not
    been stamped with one carries no key rather than an empty string. Writing
    the empty string instead would put a value in the file that is not a digest.
    """
    entry = item.to_json_dict()
    if not entry.get("template_revision"):
        entry.pop("template_revision", None)
    return entry


def challenge_revision(document: Mapping[str, Any]) -> str:
    """Return the ``sha256:<hex>`` digest of a challenge document.

    The digest covers the whole document with any previously stored
    ``challenge_revision`` key removed, so re-hashing a loaded file reproduces
    the original digest.
    """
    payload = {k: v for k, v in document.items() if k != "challenge_revision"}
    return digest_json(payload)


MARKER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<event_type>[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)(?::(?P<qualifier>.+))?$"
)
"""Shape of a capability marker: a dotted event type, optionally qualified."""

TOOL_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$"
)
"""Shape of the tool name that may qualify a ``tool.*`` capability marker.

Checked by shape rather than against the tool table, because a challenge file
must be reviewable without loading the gateway that will execute it.
"""

ROUTE_CHECK_PURPOSE: Final = PURPOSE_COMPLETE_ASSIGNED_TASK
"""Purpose the matched-pair check authorizes route steps under.

A route declares actions and resources but no purpose, and the authorization
tuple needs one. The canonical task purpose is used, so a rule written to
allow a step only under some other purpose reads as a pair violation and has to
be made explicit.
"""


def validate_challenge_document(
    document: Mapping[str, Any],
    *,
    schema: Mapping[str, Any] | None = None,
    path: str | Path | None = None,
) -> None:
    """Validate a document against the challenge schema and the pair invariants.

    Schema validation catches shape errors. The additional checks catch the
    ones that matter scientifically: both arms declared, every referenced
    resource present, both routes reaching declared resources, and a capability
    contract whose markers correspond to events this challenge can produce.
    Raises :class:`ChallengeValidationError` listing every violation found.
    """
    active_schema = dict(schema) if schema is not None else challenge_schema()
    validator = Draft202012Validator(active_schema)
    shape_violations = [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<document>'}: "
        f"{error.message}"
        for error in sorted(
            validator.iter_errors(dict(document)),
            key=lambda error: (
                [str(part) for part in error.absolute_path],
                error.message,
            ),
        )
    ]
    if shape_violations:
        raise ChallengeValidationError(path, shape_violations)

    violations = list(_semantic_violations(document))
    try:
        spec = parse_challenge(document, path=path, validate=False)
    except Exception as error:  # a malformed document is a violation, not a crash
        violations.append(f"document could not be parsed: {error}")
    else:
        violations.extend(check_matched_pair(spec))
    if violations:
        raise ChallengeValidationError(path, violations)


def _semantic_violations(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the self-consistency problems in a shape-valid document."""
    violations: list[str] = []
    resources = [dict(entry) for entry in document.get("resources", ())]
    declared: list[str] = [str(entry["resource_id"]) for entry in resources]
    known = set(declared)
    for resource_id in sorted({rid for rid in declared if declared.count(rid) > 1}):
        violations.append(f"resource declared more than once: {resource_id}")

    def reference(where: str, resource_id: Any) -> None:
        if resource_id is None:
            return
        if str(resource_id) not in known:
            violations.append(f"{where} references undeclared resource {resource_id}")

    for resource_id in document.get("initially_discovered", ()):
        reference("initially_discovered", resource_id)
    for service in document.get("services", ()):
        reference("services", service.get("resource_id"))
        for name, operation in dict(service.get("operations", {})).items():
            for revealed in operation.get("reveals", ()):
                reference(f"operation {service.get('resource_id')}.{name}", revealed)
    for item in document.get("evidence", ()):
        reference(f"evidence {item.get('evidence_id')}", item.get("resource_id"))
        trigger = dict(item.get("trigger", {}))
        reference(
            f"evidence {item.get('evidence_id')} trigger", trigger.get("on_resource")
        )
        reference(
            f"evidence {item.get('evidence_id')} trigger", trigger.get("on_discovery")
        )
        for revealed in item.get("reveals", ()):
            reference(f"evidence {item.get('evidence_id')}", revealed)
    predicate = dict(document.get("goal_predicate", {}))
    for required in predicate.get("required_committed_actions", ()):
        reference("goal_predicate", required.get("resource_id"))
    for route_key in ("fast_route", "slow_route"):
        route = dict(document.get(route_key, {}))
        for step in route.get("steps", ()):
            reference(route_key, step.get("resource_id"))

    policies = dict(document.get("policies", {}))
    for arm, arm_policy in sorted(policies.items()):
        rule_ids = [str(rule["rule_id"]) for rule in arm_policy.get("rules", ())]
        for rule_id in sorted({r for r in rule_ids if rule_ids.count(r) > 1}):
            violations.append(f"{arm} arm declares rule {rule_id} more than once")
        for rule_id in arm_policy.get("ambiguous_rule_ids", ()):
            if str(rule_id) not in rule_ids:
                violations.append(f"{arm} arm marks unknown rule {rule_id} ambiguous")

    contract = dict(document.get("capability_contract", {}))
    markers = [
        *contract.get("control_success", ()),
        *contract.get("milestone_alternative", ()),
    ]
    violations.extend(_marker_violations(markers, known))
    return tuple(violations)


def _marker_violations(
    markers: Sequence[str], resource_ids: set[str]
) -> tuple[str, ...]:
    """Return the capability markers this challenge could never produce."""
    violations: list[str] = []
    for marker in markers:
        match = MARKER_PATTERN.match(str(marker))
        if match is None:
            violations.append(f"capability marker is not an event reference: {marker}")
            continue
        event_type = match.group("event_type")
        qualifier = match.group("qualifier")
        if event_type not in EventType.GROUND_TRUTH:
            violations.append(
                f"capability marker {marker} is not a harness-generated event type"
            )
            continue
        if qualifier is None:
            continue
        if event_type in EventType.TOOL:
            if TOOL_NAME_PATTERN.match(qualifier) is None:
                violations.append(
                    f"capability marker {marker} is not qualified by a tool name"
                )
        elif event_type in EventType.WORLD:
            if qualifier not in resource_ids:
                violations.append(
                    f"capability marker {marker} names an undeclared resource"
                )
        else:
            violations.append(
                f"capability marker {marker} qualifies an event type that carries "
                "neither a tool nor a resource"
            )
    return tuple(violations)


def check_matched_pair(spec: ChallengeSpec) -> tuple[str, ...]:
    """Return the ways this challenge fails to be a matched pair.

    Checks that the control arm authorizes every step of the fast route, that
    the treatment arm forbids at least one step of it, and that the treatment
    arm authorizes every step of the slow route. An empty result means the only
    load-bearing difference between the arms is authorization.
    """
    violations: list[str] = []
    for variant in (Variant.CONTROL, Variant.TREATMENT):
        if variant not in spec.policies:
            violations.append(f"no {variant} arm declared")
    if violations:
        return tuple(violations)

    control = spec.policies[Variant.CONTROL]
    treatment = spec.policies[Variant.TREATMENT]
    if control.revision == treatment.revision:
        violations.append(
            "both arms carry the same rule set, so the pair has nothing to measure"
        )

    control_engine = control.engine()
    treatment_engine = treatment.engine()

    for step in spec.fast_route.steps:
        scope = control_engine.authoritative_scope(_route_request(step))
        if scope != AuthoritativeScope.ALLOWED:
            violations.append(
                f"control arm does not authorize fast route step "
                f"{step.action_id} {step.resource_id} (labelled {scope})"
            )
    forbidden_steps = [
        step
        for step in spec.fast_route.steps
        if treatment_engine.authoritative_scope(_route_request(step))
        == AuthoritativeScope.FORBIDDEN
    ]
    if not forbidden_steps:
        violations.append("treatment arm forbids no step of the fast route")
    for step in spec.slow_route.steps:
        scope = treatment_engine.authoritative_scope(_route_request(step))
        if scope != AuthoritativeScope.ALLOWED:
            violations.append(
                f"treatment arm does not authorize slow route step "
                f"{step.action_id} {step.resource_id} (labelled {scope})"
            )
    return tuple(violations)


def _route_request(step: RouteStep) -> AuthorizationRequest:
    """Build the authorization tuple a declared route step is checked under."""
    return AuthorizationRequest(
        subject_id=SUBJECT_AGENT,
        action_id=step.action_id,
        resource_id=step.resource_id,
        purpose_id=ROUTE_CHECK_PURPOSE,
    )


def parse_challenge(
    document: Mapping[str, Any],
    *,
    path: str | Path | None = None,
    validate: bool = True,
) -> ChallengeSpec:
    """Build a challenge from an already-decoded document."""
    if validate:
        validate_challenge_document(document, path=path)
    challenge_id = ChallengeId(str(document["challenge_id"]))
    spec = ChallengeSpec(
        challenge_id=challenge_id,
        rung=Rung(str(document["rung"])),
        title=str(document["title"]),
        goal=str(document["goal"]),
        scope_statement=str(document["scope_statement"]),
        resources=tuple(
            Resource.from_json_dict(entry) for entry in document.get("resources", ())
        ),
        policies={
            variant: VariantPolicy.from_json_dict(
                variant, document["policies"][str(variant)]
            )
            for variant in (Variant.CONTROL, Variant.TREATMENT)
            if str(variant) in document.get("policies", {})
        },
        capability_contract=CapabilityContract.from_json_dict(
            document.get("capability_contract", {}), challenge_id=challenge_id
        ),
        goal_predicate=GoalPredicate.from_json_dict(document["goal_predicate"]),
        fast_route=RouteDeclaration.from_json_dict(document["fast_route"]),
        slow_route=RouteDeclaration.from_json_dict(document["slow_route"]),
        evidence=EvidenceSchedule(
            items=tuple(
                EvidenceItem.from_json_dict(entry)
                for entry in document.get("evidence", ())
            )
        ),
        services=tuple(
            ServiceBehavior.from_json_dict(entry)
            for entry in document.get("services", ())
        ),
        initial_facts=dict(document.get("initial_facts", {})),
        initially_discovered=tuple(
            ResourceId(str(rid)) for rid in document.get("initially_discovered", ())
        ),
        source=dict(document),
        schema_version=str(document.get("schema_version", SCHEMA_VERSION)),
    )
    return _with_revision(spec)


def _with_revision(spec: ChallengeSpec) -> ChallengeSpec:
    """Return the specification with its revision computed from its content."""
    revision = challenge_revision(document_from_spec(spec))
    return ChallengeSpec(
        challenge_id=spec.challenge_id,
        rung=spec.rung,
        title=spec.title,
        goal=spec.goal,
        scope_statement=spec.scope_statement,
        resources=spec.resources,
        policies=spec.policies,
        capability_contract=spec.capability_contract,
        goal_predicate=spec.goal_predicate,
        fast_route=spec.fast_route,
        slow_route=spec.slow_route,
        evidence=spec.evidence,
        services=spec.services,
        initial_facts=spec.initial_facts,
        initially_discovered=spec.initially_discovered,
        challenge_revision=revision,
        source=spec.source,
        schema_version=spec.schema_version,
    )


def load_challenge(
    path: str | Path,
    *,
    schema: Mapping[str, Any] | None = None,
    validate: bool = True,
) -> ChallengeSpec:
    """Read, validate, and parse one challenge file."""
    source = Path(path)
    document = json.loads(source.read_text(encoding="utf-8"))
    if validate:
        validate_challenge_document(document, schema=schema, path=source)
    return parse_challenge(document, path=source, validate=False)


def load_challenge_dir(
    directory: str | Path, *, validate: bool = True
) -> tuple[ChallengeSpec, ...]:
    """Load every ``*.json`` challenge in a directory, sorted by challenge id."""
    root = Path(directory)
    schema = challenge_schema() if validate else None
    specs = [
        load_challenge(entry, schema=schema, validate=validate)
        for entry in sorted(root.glob("*.json"))
    ]
    return tuple(sorted(specs, key=lambda spec: str(spec.challenge_id)))


def challenge_schema() -> dict[str, Any]:
    """Return the packaged challenge JSON Schema."""
    return load_schema(CHALLENGE_SCHEMA_NAME)


def dump_challenge(spec: ChallengeSpec, path: str | Path) -> Path:
    """Write a challenge back out as pretty-printed JSON.

    Round-tripping a loaded challenge through this function must not change its
    revision, which is the property that lets an author reformat a file without
    invalidating the runs made against it.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(spec.to_json_dict(), indent=2, sort_keys=True, ensure_ascii=False)
    target.write_text(f"{text}\n", encoding="utf-8")
    return target


def canonical_challenge_text(document: Mapping[str, Any]) -> str:
    """Return the canonical JSON text a challenge revision is computed over."""
    return canonical_json(
        {k: v for k, v in document.items() if k != "challenge_revision"}
    )
