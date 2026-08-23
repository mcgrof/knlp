"""Deterministic synthetic world: typed resource graph and state machine.

The world is a pure state machine over a declared resource graph. Given the
same specification, seed, and action sequence it produces the same state, the
same observations, and the same events on any machine. It performs no I/O of
any kind: no sockets, no subprocesses, no filesystem access, no clock reads. It
emits events through an injected event log and nothing else leaves it.

The design preference is deterministic typed APIs over realistic services. A
challenge declares resources, service operations, an evidence schedule, and a
goal predicate; the world executes them. Nothing about a challenge is expressed
as executable code, which is what keeps a challenge reviewable and its
revision hashable.

Evidence is scheduled, not improvised. Opportunities and contradictions fire
when their declared trigger conditions hold, so the point at which the agent
first saw a forbidden route, and the point at which it first saw that a
resource does not belong to the exercise, are properties of the challenge
rather than of the transcript.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Final

from .events import EventLog, EventType
from .ids import (
    ACTION_ASSUME,
    ACTION_CALL,
    ACTION_LIST,
    ACTION_READ,
    ACTION_SUBMIT,
    ACTION_WRITE,
    ACTOR_WORLD,
    ActionId,
    EvidenceClass,
    EvidenceKind,
    PurposeId,
    ResourceId,
    ResourceKind,
)
from .manifest import digest_json, short_digest

ERROR_UNKNOWN_RESOURCE: Final[str] = "unknown_resource"
ERROR_UNKNOWN_OPERATION: Final[str] = "unknown_operation"
ERROR_NOT_DISCOVERED: Final[str] = "resource_not_discovered"
ERROR_WRONG_KIND: Final[str] = "resource_wrong_kind"
ERROR_NOT_WRITABLE: Final[str] = "resource_not_writable"
ERROR_OPERATION_FAILED: Final[str] = "operation_failed"

OBSERVATION_RESOURCE_LIST: Final[str] = "resource_list"
OBSERVATION_CONTENT: Final[str] = "content"
OBSERVATION_ACK: Final[str] = "ack"
OBSERVATION_SERVICE_RESULT: Final[str] = "service_result"
OBSERVATION_IDENTITY: Final[str] = "identity"
OBSERVATION_JOB: Final[str] = "job"
OBSERVATION_ERROR: Final[str] = "error"

FACT_ACTIVE_IDENTITY: Final[str] = "active_identity"
"""Fact key holding the identity the agent is currently acting as."""

FACT_JOB_PREFIX: Final[str] = "job:"
"""Prefix of the fact key one submitted job is recorded under, per runner.

A goal predicate names the whole key, as in ``job:runner/build``, so a
challenge can require that a particular job reached a particular runner.
"""

ATTRIBUTE_CONTENT: Final[str] = "content"
ATTRIBUTE_WRITABLE: Final[str] = "writable"
ATTRIBUTE_SECRET: Final[str] = "secret"


@dataclass(slots=True)
class Resource:
    """One node in the synthetic resource graph.

    ``owner`` is a synthetic ownership label such as a project or team name;
    challenges use it to make the boundary between in-scope and out-of-scope
    resources legible to a reader as well as to the policy engine.
    ``discovered`` tracks whether the agent has seen this resource yet, which
    is what makes opportunity discovery measurable.
    """

    resource_id: ResourceId
    kind: ResourceKind
    owner: str
    attributes: dict[str, Any] = field(default_factory=dict)
    discovered: bool = False

    @property
    def is_credential(self) -> bool:
        """Report whether this resource carries a synthetic credential."""
        return self.kind in (ResourceKind.CREDENTIAL, ResourceKind.SECRET_STORE)

    def public_view(self) -> dict[str, Any]:
        """Return the fields shown to the agent, with secret attributes elided."""
        secret = self.attributes.get(ATTRIBUTE_SECRET)
        if isinstance(secret, str):
            hidden = {secret}
        elif isinstance(secret, (list, tuple, set, frozenset)):
            hidden = {str(name) for name in secret}
        elif secret:
            hidden = {name for name in self.attributes if name != ATTRIBUTE_SECRET}
        else:
            hidden = set()
        view: dict[str, Any] = {
            "resource_id": str(self.resource_id),
            "kind": str(self.kind),
            "owner": self.owner,
            "attributes": {
                key: value
                for key, value in sorted(self.attributes.items())
                if key != ATTRIBUTE_SECRET and key not in hidden
            },
        }
        if hidden:
            view["elided_attributes"] = sorted(hidden)
        return view

    def to_json_dict(self) -> dict[str, Any]:
        """Return the resource as a challenge-file entry."""
        return {
            "resource_id": str(self.resource_id),
            "kind": str(self.kind),
            "owner": self.owner,
            "attributes": dict(self.attributes),
            "discovered": self.discovered,
        }

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, Any]) -> "Resource":
        """Rebuild a resource from a challenge-file entry."""
        return cls(
            resource_id=ResourceId(str(obj["resource_id"])),
            kind=ResourceKind(obj["kind"]),
            owner=str(obj["owner"]),
            attributes=deepcopy(dict(obj.get("attributes", {}))),
            discovered=bool(obj.get("discovered", False)),
        )


@dataclass(frozen=True, slots=True)
class ServiceOperation:
    """One named operation on a synthetic service.

    An operation is a declarative transition: it returns a fixed payload, sets
    facts, reveals resources, and optionally fails with a stable error code.
    Nothing is computed from the arguments beyond echoing declared fields, so
    two runs with the same call sequence transition identically.
    """

    name: str
    returns: Mapping[str, Any] = field(default_factory=dict)
    sets_facts: Mapping[str, Any] = field(default_factory=dict)
    reveals: tuple[ResourceId, ...] = ()
    error_code: str | None = None
    requires_identity: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return the operation as a challenge-file entry."""
        return {
            "returns": dict(self.returns),
            "sets_facts": dict(self.sets_facts),
            "reveals": [str(resource_id) for resource_id in self.reveals],
            "error_code": self.error_code,
            "requires_identity": self.requires_identity,
        }

    @classmethod
    def from_json_dict(cls, name: str, obj: Mapping[str, Any]) -> "ServiceOperation":
        """Rebuild an operation from a challenge-file entry."""
        error_code = obj.get("error_code")
        requires_identity = obj.get("requires_identity")
        return cls(
            name=name,
            returns=deepcopy(dict(obj.get("returns", {}))),
            sets_facts=deepcopy(dict(obj.get("sets_facts", {}))),
            reveals=tuple(
                ResourceId(str(resource_id)) for resource_id in obj.get("reveals", ())
            ),
            error_code=None if error_code is None else str(error_code),
            requires_identity=(
                None if requires_identity is None else str(requires_identity)
            ),
        )


@dataclass(frozen=True, slots=True)
class ServiceBehavior:
    """The operation table of one callable synthetic service."""

    resource_id: ResourceId
    operations: Mapping[str, ServiceOperation] = field(default_factory=dict)

    def operation(self, name: str) -> ServiceOperation | None:
        """Return an operation by name, or ``None`` if this service lacks it."""
        return self.operations.get(name)

    def to_json_dict(self) -> dict[str, Any]:
        """Return the service as a challenge-file entry."""
        return {
            "resource_id": str(self.resource_id),
            "operations": {
                name: operation.to_json_dict()
                for name, operation in sorted(self.operations.items())
            },
        }

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, Any]) -> "ServiceBehavior":
        """Rebuild a service from a challenge-file entry."""
        return cls(
            resource_id=ResourceId(str(obj["resource_id"])),
            operations={
                str(name): ServiceOperation.from_json_dict(str(name), entry)
                for name, entry in obj.get("operations", {}).items()
            },
        )


@dataclass(frozen=True, slots=True)
class EvidenceTrigger:
    """Conjunction of conditions under which one evidence item fires.

    Every non-null condition must hold. ``on_step`` fires at or after a step
    index, ``on_action`` and ``on_resource`` match the action just executed,
    and ``on_discovery`` fires once a resource has been discovered. An empty
    trigger fires on the first evaluation, which is how an opportunity that is
    visible from the start is declared.
    """

    on_step: int | None = None
    on_action: ActionId | None = None
    on_resource: ResourceId | None = None
    on_discovery: ResourceId | None = None

    def is_satisfied(
        self,
        *,
        step: int,
        action_id: ActionId | None,
        resource_id: ResourceId | None,
        discovered: frozenset[ResourceId],
    ) -> bool:
        """Report whether every declared condition holds."""
        if self.on_step is not None and step < self.on_step:
            return False
        if self.on_action is not None and action_id != self.on_action:
            return False
        if self.on_resource is not None and resource_id != self.on_resource:
            return False
        if self.on_discovery is not None and self.on_discovery not in discovered:
            return False
        return True

    def to_json_dict(self) -> dict[str, Any]:
        """Return the trigger as a challenge-file entry."""
        return {
            "on_step": self.on_step,
            "on_action": None if self.on_action is None else str(self.on_action),
            "on_resource": None if self.on_resource is None else str(self.on_resource),
            "on_discovery": (
                None if self.on_discovery is None else str(self.on_discovery)
            ),
        }

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, Any]) -> "EvidenceTrigger":
        """Rebuild a trigger from a challenge-file entry."""
        on_step = obj.get("on_step")
        on_action = obj.get("on_action")
        on_resource = obj.get("on_resource")
        on_discovery = obj.get("on_discovery")
        return cls(
            on_step=None if on_step is None else int(on_step),
            on_action=None if on_action is None else ActionId(str(on_action)),
            on_resource=None if on_resource is None else ResourceId(str(on_resource)),
            on_discovery=(
                None if on_discovery is None else ResourceId(str(on_discovery))
            ),
        )


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One pre-registered piece of evidence the world may expose.

    ``strength`` is an ordinal challenge parameter used to order and to build a
    cumulative evidence level in the trace. It ranks items within a challenge
    and carries no claim about how a human or a model would weigh them.
    ``note`` is the short text handed to the agent; it is part of the challenge
    revision, so editing it changes the challenge.
    """

    evidence_id: str
    kind: EvidenceKind
    evidence_class: EvidenceClass
    strength: int
    note: str
    resource_id: ResourceId | None = None
    trigger: EvidenceTrigger = field(default_factory=EvidenceTrigger)
    reveals: tuple[ResourceId, ...] = ()
    template_revision: str = ""

    def __post_init__(self) -> None:
        if self.strength < 0:
            raise ValueError("evidence strength must be non-negative")
        if not self.evidence_id:
            raise ValueError("evidence_id must be non-empty")

    @property
    def event_type(self) -> str:
        """Return the world event type this item is exposed as."""
        return (
            EventType.WORLD_OPPORTUNITY_EXPOSED
            if self.kind is EvidenceKind.OPPORTUNITY
            else EventType.WORLD_CONTRADICTION_EXPOSED
        )

    def _declared_content(self) -> dict[str, Any]:
        """Return the declared fields this item's revision is computed over."""
        return {
            "evidence_id": self.evidence_id,
            "kind": str(self.kind),
            "evidence_class": str(self.evidence_class),
            "strength": self.strength,
            "note": self.note,
            "resource_id": (
                None if self.resource_id is None else str(self.resource_id)
            ),
            "trigger": self.trigger.to_json_dict(),
            "reveals": [str(resource_id) for resource_id in self.reveals],
        }

    def revision(self) -> str:
        """Return the ``sha256:<hex>`` digest of this item's declared content."""
        return digest_json(self._declared_content())

    def to_payload(self) -> dict[str, Any]:
        """Return the evidence event payload for this item."""
        payload = self._declared_content()
        payload["template_revision"] = self.template_revision or self.revision()
        return payload

    def to_json_dict(self) -> dict[str, Any]:
        """Return the item as a challenge-file entry."""
        entry = self._declared_content()
        if self.template_revision:
            entry["template_revision"] = self.template_revision
        return entry

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, Any]) -> "EvidenceItem":
        """Rebuild an item from a challenge-file entry."""
        resource_id = obj.get("resource_id")
        return cls(
            evidence_id=str(obj["evidence_id"]),
            kind=EvidenceKind(obj["kind"]),
            evidence_class=EvidenceClass(obj["evidence_class"]),
            strength=int(obj["strength"]),
            note=str(obj["note"]),
            resource_id=None if resource_id is None else ResourceId(str(resource_id)),
            trigger=EvidenceTrigger.from_json_dict(obj.get("trigger", {})),
            reveals=tuple(
                ResourceId(str(revealed)) for revealed in obj.get("reveals", ())
            ),
            template_revision=str(obj.get("template_revision", "")),
        )


@dataclass(frozen=True, slots=True)
class EvidenceSchedule:
    """Ordered evidence items and the rule for firing them.

    Each item fires at most once per run. Order is the declaration order in the
    challenge file, which the loader preserves, so a challenge author controls
    the sequence in which contradictions accumulate.
    """

    items: tuple[EvidenceItem, ...] = ()

    def due(
        self,
        *,
        step: int,
        action_id: ActionId | None,
        resource_id: ResourceId | None,
        discovered: frozenset[ResourceId],
        already_exposed: frozenset[str],
    ) -> tuple[EvidenceItem, ...]:
        """Return the items whose triggers hold and that have not yet fired."""
        return tuple(
            item
            for item in self.items
            if item.evidence_id not in already_exposed
            and item.trigger.is_satisfied(
                step=step,
                action_id=action_id,
                resource_id=resource_id,
                discovered=discovered,
            )
        )

    def by_id(self, evidence_id: str) -> EvidenceItem | None:
        """Return one item by id, or ``None``."""
        for item in self.items:
            if item.evidence_id == evidence_id:
                return item
        return None

    def cumulative_strength(self, exposed: Iterable[str]) -> int:
        """Return the summed strength of the exposed items."""
        exposed_ids = set(exposed)
        return sum(
            item.strength for item in self.items if item.evidence_id in exposed_ids
        )


@dataclass(frozen=True, slots=True)
class RequiredAction:
    """One action that must be committed for a goal predicate to hold."""

    action_id: ActionId
    resource_id: ResourceId

    def to_json_dict(self) -> dict[str, Any]:
        """Return the requirement as a challenge-file entry."""
        return {"action_id": str(self.action_id), "resource_id": str(self.resource_id)}


@dataclass(frozen=True, slots=True)
class GoalPredicate:
    """Declarative success condition for a challenge.

    The goal holds when every required fact matches the world state and every
    required action has been committed. Both halves are declared in the
    challenge file; there is no callable hook, so a goal cannot quietly depend
    on the model or on the enforcement mode.
    """

    description: str
    required_facts: Mapping[str, Any] = field(default_factory=dict)
    required_committed_actions: tuple[RequiredAction, ...] = ()

    def evaluate(self, state: "WorldState") -> bool:
        """Report whether the world state satisfies the goal."""
        return not self.missing(state)

    def missing(self, state: "WorldState") -> tuple[str, ...]:
        """Return human-readable descriptions of the unmet requirements."""
        gaps: list[str] = []
        for key, expected in self.required_facts.items():
            if state.facts.get(key) != expected:
                gaps.append(f"fact {key!r} is not {expected!r}")
        for required in self.required_committed_actions:
            if not any(
                committed.action_id == required.action_id
                and committed.resource_id == required.resource_id
                for committed in state.committed
            ):
                gaps.append(
                    f"action {required.action_id} on {required.resource_id} "
                    "was never committed"
                )
        return tuple(gaps)

    def to_json_dict(self) -> dict[str, Any]:
        """Return the predicate as a challenge-file entry."""
        return {
            "description": self.description,
            "required_facts": dict(self.required_facts),
            "required_committed_actions": [
                required.to_json_dict() for required in self.required_committed_actions
            ],
        }

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, Any]) -> "GoalPredicate":
        """Rebuild a predicate from a challenge-file entry."""
        return cls(
            description=str(obj["description"]),
            required_facts=deepcopy(dict(obj.get("required_facts", {}))),
            required_committed_actions=tuple(
                RequiredAction(
                    action_id=ActionId(str(entry["action_id"])),
                    resource_id=ResourceId(str(entry["resource_id"])),
                )
                for entry in obj.get("required_committed_actions", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class CommittedAction:
    """A world mutation that actually happened, in commit order."""

    action_id: ActionId
    resource_id: ResourceId
    purpose_id: PurposeId
    step: int


@dataclass(frozen=True, slots=True)
class WorldObservation:
    """What the world hands back for one action.

    The observation is data, not prose: the tool gateway renders it for the
    model and stores it in the event payload unchanged, so an observation the
    agent saw can be compared byte for byte across runs.
    """

    ok: bool
    kind: str
    data: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None

    @classmethod
    def failure(cls, error_code: str, **data: Any) -> "WorldObservation":
        """Build a failed observation carrying a stable error code."""
        return cls(ok=False, kind=OBSERVATION_ERROR, data=data, error_code=error_code)

    def to_json_dict(self) -> dict[str, Any]:
        """Return the observation as an event payload fragment."""
        return {
            "ok": self.ok,
            "kind": self.kind,
            "data": dict(self.data),
            "error_code": self.error_code,
        }


@dataclass(slots=True)
class WorldState:
    """Mutable state of one run of the world.

    ``facts`` is the general-purpose key/value store that service operations
    write and goal predicates read. ``step`` counts committed world actions,
    not model turns; a rejected tool call does not advance it.
    """

    resources: dict[ResourceId, Resource] = field(default_factory=dict)
    discovered: set[ResourceId] = field(default_factory=set)
    facts: dict[str, Any] = field(default_factory=dict)
    committed: list[CommittedAction] = field(default_factory=list)
    exposed_evidence: list[str] = field(default_factory=list)
    step: int = 0
    goal_reached: bool = False

    def resource(self, resource_id: ResourceId) -> Resource | None:
        """Return a resource by id, or ``None`` if the graph lacks it."""
        return self.resources.get(resource_id)

    def snapshot_digest(self) -> str:
        """Return a ``sha256:<hex>`` digest of the observable state.

        Two runs that agree on this digest at every step are identical worlds.
        Determinism tests compare the digest sequence rather than the events,
        so a change in event formatting does not look like a state divergence.
        """
        return digest_json(self.to_json_dict())

    def to_json_dict(self) -> dict[str, Any]:
        """Return the state as a JSON object, for debugging and golden files."""
        return {
            "resources": [
                self.resources[resource_id].to_json_dict()
                for resource_id in sorted(self.resources)
            ],
            "discovered": sorted(str(resource_id) for resource_id in self.discovered),
            "facts": dict(self.facts),
            "committed": [
                {
                    "action_id": str(action.action_id),
                    "resource_id": str(action.resource_id),
                    "purpose_id": str(action.purpose_id),
                    "step": action.step,
                }
                for action in self.committed
            ],
            "exposed_evidence": list(self.exposed_evidence),
            "step": self.step,
            "goal_reached": self.goal_reached,
        }


@dataclass(frozen=True, slots=True)
class WorldSpec:
    """Everything the world needs to run one challenge arm.

    A challenge holds one of these and hands it to the world; the world never
    reads the challenge itself. That keeps the dependency one-way and lets a
    test build a world from a handful of resources without a challenge file.
    """

    resources: tuple[Resource, ...] = ()
    initial_facts: Mapping[str, Any] = field(default_factory=dict)
    initially_discovered: tuple[ResourceId, ...] = ()
    services: tuple[ServiceBehavior, ...] = ()
    evidence: EvidenceSchedule = field(default_factory=EvidenceSchedule)
    goal: GoalPredicate = field(
        default_factory=lambda: GoalPredicate(description="no goal declared")
    )

    def service(self, resource_id: ResourceId) -> ServiceBehavior | None:
        """Return the behavior table for a service resource, or ``None``."""
        for service in self.services:
            if service.resource_id == resource_id:
                return service
        return None

    def revision(self) -> str:
        """Return a ``sha256:<hex>`` digest of the declared world."""
        return digest_json(
            {
                "resources": [r.to_json_dict() for r in self.resources],
                "initial_facts": dict(self.initial_facts),
                "initially_discovered": list(self.initially_discovered),
                "services": [s.to_json_dict() for s in self.services],
                "evidence": [e.to_json_dict() for e in self.evidence.items],
                "goal": self.goal.to_json_dict(),
            }
        )


class World:
    """Deterministic state machine over the declared resource graph.

    Construction builds the initial state from the specification; the seed
    feeds a private generator used for any ordering choice that is not fully
    determined by the specification. Every method that mutates state emits the
    corresponding world events and returns a typed observation.

    The world does not consult the policy engine. Authorization happens one
    layer up in the tool gateway, and a method called here has already been
    authorized. That is deliberate: it keeps the observe-mode path honest,
    since the world executes exactly what it is asked to.
    """

    def __init__(
        self,
        spec: WorldSpec,
        log: EventLog,
        *,
        seed: int = 0,
    ) -> None:
        self.spec = spec
        self.log = log
        self.seed = seed
        self._rng = random.Random(seed)
        self.state = WorldState()
        self.reset()

    def reset(self) -> None:
        """Rebuild the initial state from the specification.

        Resources are deep-copied out of the specification so that a world can
        be re-run without the specification accumulating mutations.
        """
        resources: dict[ResourceId, Resource] = {}
        for declared in self.spec.resources:
            resources[declared.resource_id] = Resource(
                resource_id=declared.resource_id,
                kind=declared.kind,
                owner=declared.owner,
                attributes=deepcopy(dict(declared.attributes)),
                discovered=declared.discovered,
            )
        for resource_id in self.spec.initially_discovered:
            visible = resources.get(resource_id)
            if visible is not None:
                visible.discovered = True
        self.state = WorldState(
            resources=resources,
            discovered={
                resource_id
                for resource_id, resource in resources.items()
                if resource.discovered
            },
            facts=deepcopy(dict(self.spec.initial_facts)),
        )
        self._rng = random.Random(self.seed)

    def list_resources(
        self,
        *,
        prefix: str | None,
        purpose_id: PurposeId,
        step_id: int,
        correlation_id: str,
        parent_event_id: int | None = None,
    ) -> WorldObservation:
        """List discovered resources whose id matches an optional prefix.

        Listing reveals only resources already discovered, so a challenge
        controls discovery through its evidence schedule and service
        operations rather than through a directory dump.
        """
        visible = [
            resource
            for resource in self.state.resources.values()
            if resource.resource_id in self.state.discovered
        ]
        matched = resources_by_prefix(visible, prefix)
        observation = WorldObservation(
            ok=True,
            kind=OBSERVATION_RESOURCE_LIST,
            data={
                "prefix": prefix,
                "count": len(matched),
                "resources": [resource.public_view() for resource in matched],
            },
        )
        return self._finish(
            observation,
            action_id=ACTION_LIST,
            resource_id=None,
            purpose_id=purpose_id,
            step_id=step_id,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
        )

    def read(
        self,
        resource_id: ResourceId,
        *,
        purpose_id: PurposeId,
        step_id: int,
        correlation_id: str,
        parent_event_id: int | None = None,
    ) -> WorldObservation:
        """Return the content of a discovered resource."""
        resource = self.state.resource(resource_id)
        if resource is None:
            observation = WorldObservation.failure(
                ERROR_UNKNOWN_RESOURCE, resource_id=str(resource_id)
            )
        elif resource_id not in self.state.discovered:
            observation = WorldObservation.failure(
                ERROR_NOT_DISCOVERED, resource_id=str(resource_id)
            )
        else:
            view = resource.public_view()
            observation = WorldObservation(
                ok=True,
                kind=OBSERVATION_CONTENT,
                data={**view, "content": view["attributes"].get(ATTRIBUTE_CONTENT)},
            )
        return self._finish(
            observation,
            action_id=ACTION_READ,
            resource_id=resource_id,
            purpose_id=purpose_id,
            step_id=step_id,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
        )

    def write(
        self,
        resource_id: ResourceId,
        content: str,
        *,
        purpose_id: PurposeId,
        step_id: int,
        correlation_id: str,
        parent_event_id: int | None = None,
    ) -> WorldObservation:
        """Replace the content of a writable resource."""
        resource = self.state.resource(resource_id)
        committed = False
        if resource is None:
            observation = WorldObservation.failure(
                ERROR_UNKNOWN_RESOURCE, resource_id=str(resource_id)
            )
        elif resource_id not in self.state.discovered:
            observation = WorldObservation.failure(
                ERROR_NOT_DISCOVERED, resource_id=str(resource_id)
            )
        elif not resource.attributes.get(ATTRIBUTE_WRITABLE, False):
            observation = WorldObservation.failure(
                ERROR_NOT_WRITABLE, resource_id=str(resource_id)
            )
        else:
            before_digest = self.state.snapshot_digest()
            resource.attributes[ATTRIBUTE_CONTENT] = content
            self.commit(
                action_id=ACTION_WRITE,
                resource_id=resource_id,
                purpose_id=purpose_id,
                step_id=step_id,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
            )
            self.emit_state_transition(
                action_id=ACTION_WRITE,
                resource_id=resource_id,
                purpose_id=purpose_id,
                before_digest=before_digest,
                step_id=step_id,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
            )
            observation = WorldObservation(
                ok=True,
                kind=OBSERVATION_ACK,
                data={"resource_id": str(resource_id), "length": len(content)},
            )
            committed = True
        result = self._finish(
            observation,
            action_id=ACTION_WRITE,
            resource_id=resource_id,
            purpose_id=purpose_id,
            step_id=step_id,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
        )
        if committed:
            self.evaluate_goal(
                step_id=step_id,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
            )
        return result

    def call_service(
        self,
        resource_id: ResourceId,
        operation: str,
        arguments: Mapping[str, Any],
        *,
        purpose_id: PurposeId,
        step_id: int,
        correlation_id: str,
        parent_event_id: int | None = None,
    ) -> WorldObservation:
        """Invoke a declared operation on a synthetic service.

        A successful call applies the operation's fact updates, reveals the
        resources it declares, and returns its fixed payload.
        """
        resource = self.state.resource(resource_id)
        service = self.spec.service(resource_id)
        declared = None if service is None else service.operation(operation)
        active_identity = self.state.facts.get(FACT_ACTIVE_IDENTITY)
        committed = False
        if resource is None:
            observation = WorldObservation.failure(
                ERROR_UNKNOWN_RESOURCE, resource_id=str(resource_id)
            )
        elif resource_id not in self.state.discovered:
            observation = WorldObservation.failure(
                ERROR_NOT_DISCOVERED, resource_id=str(resource_id)
            )
        elif service is None:
            observation = WorldObservation.failure(
                ERROR_WRONG_KIND, resource_id=str(resource_id), kind=str(resource.kind)
            )
        elif declared is None:
            observation = WorldObservation.failure(
                ERROR_UNKNOWN_OPERATION,
                resource_id=str(resource_id),
                operation=operation,
                available=sorted(service.operations),
            )
        elif (
            declared.requires_identity is not None
            and active_identity != declared.requires_identity
        ):
            observation = WorldObservation.failure(
                ERROR_OPERATION_FAILED,
                resource_id=str(resource_id),
                operation=operation,
                requires_identity=declared.requires_identity,
            )
        elif declared.error_code is not None:
            observation = WorldObservation.failure(
                declared.error_code, resource_id=str(resource_id), operation=operation
            )
        else:
            before_digest = self.state.snapshot_digest()
            for key, value in declared.sets_facts.items():
                self.state.facts[key] = deepcopy(value)
            for revealed in declared.reveals:
                self.discover(
                    revealed,
                    step_id=step_id,
                    correlation_id=correlation_id,
                    parent_event_id=parent_event_id,
                )
            self.commit(
                action_id=ACTION_CALL,
                resource_id=resource_id,
                purpose_id=purpose_id,
                step_id=step_id,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
            )
            self.emit_state_transition(
                action_id=ACTION_CALL,
                resource_id=resource_id,
                purpose_id=purpose_id,
                before_digest=before_digest,
                step_id=step_id,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
            )
            observation = WorldObservation(
                ok=True,
                kind=OBSERVATION_SERVICE_RESULT,
                data={
                    "resource_id": str(resource_id),
                    "operation": operation,
                    "result": deepcopy(dict(declared.returns)),
                    "revealed_resources": [str(name) for name in declared.reveals],
                },
            )
            committed = True
        result = self._finish(
            observation,
            action_id=ACTION_CALL,
            resource_id=resource_id,
            purpose_id=purpose_id,
            step_id=step_id,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
        )
        if committed:
            self.evaluate_goal(
                step_id=step_id,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
            )
        return result

    def assume_identity(
        self,
        identity_id: ResourceId,
        *,
        purpose_id: PurposeId,
        step_id: int,
        correlation_id: str,
        parent_event_id: int | None = None,
    ) -> WorldObservation:
        """Act as a discovered synthetic identity for subsequent actions.

        The active identity is recorded as a world fact and surfaces in the
        authorization conditions of later requests.
        """
        identity = self.state.resource(identity_id)
        committed = False
        if identity is None:
            observation = WorldObservation.failure(
                ERROR_UNKNOWN_RESOURCE, resource_id=str(identity_id)
            )
        elif identity_id not in self.state.discovered:
            observation = WorldObservation.failure(
                ERROR_NOT_DISCOVERED, resource_id=str(identity_id)
            )
        elif identity.kind is not ResourceKind.IDENTITY and not identity.is_credential:
            observation = WorldObservation.failure(
                ERROR_WRONG_KIND, resource_id=str(identity_id), kind=str(identity.kind)
            )
        else:
            before_digest = self.state.snapshot_digest()
            self.state.facts[FACT_ACTIVE_IDENTITY] = str(identity_id)
            self.commit(
                action_id=ACTION_ASSUME,
                resource_id=identity_id,
                purpose_id=purpose_id,
                step_id=step_id,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
            )
            self.emit_state_transition(
                action_id=ACTION_ASSUME,
                resource_id=identity_id,
                purpose_id=purpose_id,
                before_digest=before_digest,
                step_id=step_id,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
            )
            observation = WorldObservation(
                ok=True,
                kind=OBSERVATION_IDENTITY,
                data={
                    "identity_id": str(identity_id),
                    "active_identity": str(identity_id),
                    "owner": identity.owner,
                },
            )
            committed = True
        result = self._finish(
            observation,
            action_id=ACTION_ASSUME,
            resource_id=identity_id,
            purpose_id=purpose_id,
            step_id=step_id,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
        )
        if committed:
            self.evaluate_goal(
                step_id=step_id,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
            )
        return result

    def submit_job(
        self,
        resource_id: ResourceId,
        job: Mapping[str, Any],
        *,
        purpose_id: PurposeId,
        step_id: int,
        correlation_id: str,
        parent_event_id: int | None = None,
    ) -> WorldObservation:
        """Queue a synthetic job on a job runner and record it as a fact."""
        runner = self.state.resource(resource_id)
        committed = False
        if runner is None:
            observation = WorldObservation.failure(
                ERROR_UNKNOWN_RESOURCE, resource_id=str(resource_id)
            )
        elif resource_id not in self.state.discovered:
            observation = WorldObservation.failure(
                ERROR_NOT_DISCOVERED, resource_id=str(resource_id)
            )
        elif runner.kind not in (ResourceKind.JOB_RUNNER, ResourceKind.WORKER):
            observation = WorldObservation.failure(
                ERROR_WRONG_KIND, resource_id=str(resource_id), kind=str(runner.kind)
            )
        else:
            before_digest = self.state.snapshot_digest()
            submitted = deepcopy(dict(job))
            job_id = short_digest(
                digest_json({"resource_id": str(resource_id), "job": submitted})
            )
            self.state.facts[f"{FACT_JOB_PREFIX}{resource_id}"] = submitted
            self.commit(
                action_id=ACTION_SUBMIT,
                resource_id=resource_id,
                purpose_id=purpose_id,
                step_id=step_id,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
            )
            self.emit_state_transition(
                action_id=ACTION_SUBMIT,
                resource_id=resource_id,
                purpose_id=purpose_id,
                before_digest=before_digest,
                step_id=step_id,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
            )
            observation = WorldObservation(
                ok=True,
                kind=OBSERVATION_JOB,
                data={
                    "resource_id": str(resource_id),
                    "job_id": job_id,
                    "job": submitted,
                    "status": "queued",
                },
            )
            committed = True
        result = self._finish(
            observation,
            action_id=ACTION_SUBMIT,
            resource_id=resource_id,
            purpose_id=purpose_id,
            step_id=step_id,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
        )
        if committed:
            self.evaluate_goal(
                step_id=step_id,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
            )
        return result

    def discover(
        self,
        resource_id: ResourceId,
        *,
        step_id: int,
        correlation_id: str,
        parent_event_id: int | None = None,
    ) -> bool:
        """Mark a resource discovered, emitting the discovery event once.

        Credential-bearing resources also emit ``world.credential_discovered``.
        Returns whether this call was the first discovery.
        """
        resource = self.state.resource(resource_id)
        if resource is None or resource_id in self.state.discovered:
            return False
        resource.discovered = True
        self.state.discovered.add(resource_id)
        payload = {
            "resource_id": str(resource_id),
            "kind": str(resource.kind),
            "owner": resource.owner,
        }
        discovered = self.log.emit(
            EventType.WORLD_RESOURCE_DISCOVERED,
            payload,
            actor_id=WORLD_ACTOR,
            step_id=step_id,
            parent_event_id=parent_event_id,
            correlation_id=correlation_id,
        )
        if resource.is_credential:
            self.log.emit(
                EventType.WORLD_CREDENTIAL_DISCOVERED,
                dict(payload),
                actor_id=WORLD_ACTOR,
                step_id=step_id,
                parent_event_id=discovered.event_id,
                correlation_id=correlation_id,
            )
        return True

    def expose_due_evidence(
        self,
        *,
        action_id: ActionId | None,
        resource_id: ResourceId | None,
        step_id: int,
        correlation_id: str,
        parent_event_id: int | None = None,
    ) -> tuple[EvidenceItem, ...]:
        """Fire every scheduled evidence item whose trigger now holds."""
        fired: list[EvidenceItem] = []
        # An item may reveal a resource that satisfies another item's trigger, so
        # the schedule is re-read until it stops producing. Each item fires at most
        # once, which bounds the loop by the length of the schedule.
        while True:
            due = self.spec.evidence.due(
                step=self.state.step,
                action_id=action_id,
                resource_id=resource_id,
                discovered=frozenset(self.state.discovered),
                already_exposed=frozenset(self.state.exposed_evidence),
            )
            if not due:
                return tuple(fired)
            for item in due:
                if item.evidence_id in self.state.exposed_evidence:
                    continue
                if item.kind is EvidenceKind.OPPORTUNITY:
                    self.opportunity_exposed(
                        item,
                        step_id=step_id,
                        correlation_id=correlation_id,
                        parent_event_id=parent_event_id,
                    )
                else:
                    self.contradiction_exposed(
                        item,
                        step_id=step_id,
                        correlation_id=correlation_id,
                        parent_event_id=parent_event_id,
                    )
                fired.append(item)

    def opportunity_exposed(
        self,
        item: EvidenceItem,
        *,
        step_id: int,
        correlation_id: str,
        parent_event_id: int | None = None,
    ) -> None:
        """Emit ``world.opportunity_exposed`` and reveal what the item reveals.

        Called by the evidence schedule. A challenge may also call it directly
        from a service operation when a route becomes visible mid-run.
        """
        self._expose(
            item,
            step_id=step_id,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
        )

    def contradiction_exposed(
        self,
        item: EvidenceItem,
        *,
        step_id: int,
        correlation_id: str,
        parent_event_id: int | None = None,
    ) -> None:
        """Emit ``world.contradiction_exposed`` for one scheduled item."""
        self._expose(
            item,
            step_id=step_id,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
        )

    def evaluate_goal(
        self,
        *,
        step_id: int,
        correlation_id: str,
        parent_event_id: int | None = None,
    ) -> bool:
        """Evaluate the goal predicate, emitting ``world.goal_completed`` once."""
        if self.state.goal_reached:
            return True
        if not self.spec.goal.evaluate(self.state):
            return False
        self.state.goal_reached = True
        self.log.emit(
            EventType.WORLD_GOAL_COMPLETED,
            {
                "description": self.spec.goal.description,
                "step": self.state.step,
                "committed_actions": len(self.state.committed),
            },
            actor_id=WORLD_ACTOR,
            step_id=step_id,
            parent_event_id=parent_event_id,
            correlation_id=correlation_id,
        )
        return True

    @property
    def goal_reached(self) -> bool:
        """Report whether the goal predicate has already been satisfied."""
        return self.state.goal_reached

    def observation_facts(self) -> Mapping[str, Any]:
        """Return the current facts, for the authorization conditions block."""
        return dict(self.state.facts)

    def emit_state_transition(
        self,
        *,
        action_id: ActionId,
        resource_id: ResourceId,
        purpose_id: PurposeId,
        before_digest: str,
        step_id: int,
        correlation_id: str,
        parent_event_id: int | None = None,
    ) -> None:
        """Emit ``world.state_transition`` with the digests either side of a change."""
        self.log.emit(
            EventType.WORLD_STATE_TRANSITION,
            {
                "action_id": str(action_id),
                "resource_id": str(resource_id),
                "purpose_id": str(purpose_id),
                "before_digest": before_digest,
                "after_digest": self.state.snapshot_digest(),
                "step": self.state.step,
            },
            actor_id=WORLD_ACTOR,
            step_id=step_id,
            parent_event_id=parent_event_id,
            correlation_id=correlation_id,
        )

    def commit(
        self,
        *,
        action_id: ActionId,
        resource_id: ResourceId,
        purpose_id: PurposeId,
        step_id: int,
        correlation_id: str,
        parent_event_id: int | None = None,
    ) -> CommittedAction:
        """Record a mutation as committed and emit ``world.action_committed``.

        This is the event outcome classification treats as proof that an action
        actually happened inside the world, as opposed to merely being
        requested. Read-only actions do not commit.
        """
        self.state.step += 1
        action = CommittedAction(
            action_id=action_id,
            resource_id=resource_id,
            purpose_id=purpose_id,
            step=self.state.step,
        )
        self.state.committed.append(action)
        self.log.emit(
            EventType.WORLD_ACTION_COMMITTED,
            {
                "action_id": str(action_id),
                "resource_id": str(resource_id),
                "purpose_id": str(purpose_id),
                "step": action.step,
            },
            actor_id=WORLD_ACTOR,
            step_id=step_id,
            parent_event_id=parent_event_id,
            correlation_id=correlation_id,
        )
        return action

    def _finish(
        self,
        observation: WorldObservation,
        *,
        action_id: ActionId,
        resource_id: ResourceId | None,
        purpose_id: PurposeId,
        step_id: int,
        correlation_id: str,
        parent_event_id: int | None,
    ) -> WorldObservation:
        """Emit an observation, fire whatever evidence is now due, and return it.

        Every action ends here, successful or not, so the evidence schedule sees
        the same sequence of attempts the agent made. A challenge can therefore
        expose a route on the failure that sends the agent looking for another
        one.
        """
        payload: dict[str, Any] = {
            "action_id": str(action_id),
            "purpose_id": str(purpose_id),
        }
        if resource_id is not None:
            payload["resource_id"] = str(resource_id)
        payload.update(observation.to_json_dict())
        self.log.emit(
            EventType.WORLD_OBSERVATION,
            payload,
            actor_id=WORLD_ACTOR,
            step_id=step_id,
            parent_event_id=parent_event_id,
            correlation_id=correlation_id,
        )
        self.expose_due_evidence(
            action_id=action_id,
            resource_id=resource_id,
            step_id=step_id,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
        )
        return observation

    def _expose(
        self,
        item: EvidenceItem,
        *,
        step_id: int,
        correlation_id: str,
        parent_event_id: int | None,
    ) -> None:
        """Emit one evidence item and discover whatever it reveals.

        The event type comes from the item, so an opportunity and a
        contradiction travel the identical path and differ only in what they are
        recorded as.
        """
        exposed = self.log.emit(
            item.event_type,
            item.to_payload(),
            actor_id=WORLD_ACTOR,
            step_id=step_id,
            parent_event_id=parent_event_id,
            correlation_id=correlation_id,
        )
        if item.evidence_id not in self.state.exposed_evidence:
            self.state.exposed_evidence.append(item.evidence_id)
        for revealed in item.reveals:
            self.discover(
                revealed,
                step_id=step_id,
                correlation_id=correlation_id,
                parent_event_id=exposed.event_id,
            )


WORLD_ACTOR = ACTOR_WORLD
"""Actor id stamped on every event this module emits."""

MUTATING_ACTIONS: Final[frozenset[ActionId]] = frozenset(
    {ACTION_WRITE, ACTION_CALL, ACTION_ASSUME, ACTION_SUBMIT}
)
"""Actions that change the world and therefore commit.

A committed mutation emits ``world.action_committed``, which is the evidence
outcome classification uses to separate a request from a change. Listing and
reading observe without committing. ``call`` counts because a declared
operation may set facts or reveal resources.
"""


def resources_by_prefix(
    resources: Sequence[Resource], prefix: str | None
) -> tuple[Resource, ...]:
    """Return the resources whose id starts with ``prefix``, in declared order."""
    if not prefix:
        return tuple(resources)
    return tuple(
        resource
        for resource in resources
        if str(resource.resource_id).startswith(prefix)
    )
