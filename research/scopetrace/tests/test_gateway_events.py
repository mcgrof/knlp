"""Exact event order and causal chain for one allowed call and one denied call.

An analysis that walks the stream asking what happened to a request depends on
the order of the five gateway events and on the parent chain that links them.
Both are contract, so both are asserted literally rather than by counting.

The world here is built directly from a small specification rather than from a
challenge, so the sequences under test carry nothing but the gateway's own
events: no scheduled evidence, and a goal predicate that stays unmet.
"""

from __future__ import annotations

import pytest

from conftest import event_types
from scopetrace.events import EventLog, EventType, StepClock
from scopetrace.ids import (
    ACTOR_AGENT,
    ACTOR_POLICY,
    ACTOR_WORLD,
    AuthoritativeScope,
    Decision,
    PolicyMode,
    ResourceId,
    ResourceKind,
    RunId,
)
from scopetrace.policy import PolicyEngine, PolicyRule
from scopetrace.tools import (
    ERROR_INVALID_ARGUMENTS,
    ERROR_POLICY_DENIED,
    GATEWAY_EVENT_SEQUENCE,
    ToolCall,
    ToolGateway,
)
from scopetrace.world import GoalPredicate, Resource, World, WorldSpec

OPEN_DOC = ResourceId("store/project-a/doc")
CLOSED_DOC = ResourceId("store/project-b/doc")

ALLOWED_SEQUENCE = (
    EventType.TOOL_REQUESTED,
    EventType.TOOL_ARGUMENTS_VALIDATED,
    EventType.POLICY_EVALUATED,
    EventType.POLICY_ALLOWED,
    EventType.TOOL_EXECUTED,
    EventType.WORLD_OBSERVATION,
    EventType.TOOL_RESULT,
)

DENIED_SEQUENCE = (
    EventType.TOOL_REQUESTED,
    EventType.TOOL_ARGUMENTS_VALIDATED,
    EventType.POLICY_EVALUATED,
    EventType.POLICY_DENIED,
    EventType.TOOL_RESULT,
)

COMMITTING_SEQUENCE = (
    EventType.TOOL_REQUESTED,
    EventType.TOOL_ARGUMENTS_VALIDATED,
    EventType.POLICY_EVALUATED,
    EventType.POLICY_ALLOWED,
    EventType.TOOL_EXECUTED,
    EventType.WORLD_ACTION_COMMITTED,
    EventType.WORLD_STATE_TRANSITION,
    EventType.WORLD_OBSERVATION,
    EventType.TOOL_RESULT,
)

REJECTED_SEQUENCE = (
    EventType.TOOL_REQUESTED,
    EventType.TOOL_ARGUMENTS_REJECTED,
    EventType.TOOL_RESULT,
)

CONTROL_SEQUENCE = (
    EventType.TOOL_REQUESTED,
    EventType.TOOL_ARGUMENTS_VALIDATED,
    EventType.TOOL_EXECUTED,
    EventType.TOOL_RESULT,
)


@pytest.fixture
def gateway_world():
    """Build a two-resource world where project-a is open and project-b is not."""

    def build(mode: PolicyMode = PolicyMode.ENFORCE):
        spec = WorldSpec(
            resources=(
                Resource(
                    resource_id=OPEN_DOC,
                    kind=ResourceKind.DOCUMENT,
                    owner="project-a",
                    attributes={"content": "open", "writable": True},
                ),
                Resource(
                    resource_id=CLOSED_DOC,
                    kind=ResourceKind.DOCUMENT,
                    owner="project-b",
                    attributes={"content": "closed", "writable": True},
                ),
            ),
            initially_discovered=(OPEN_DOC, CLOSED_DOC),
            goal=GoalPredicate(
                description="never satisfied in this fixture",
                required_facts={"unreachable": True},
            ),
        )
        rules = (
            PolicyRule(
                rule_id="gateway/project-a",
                subject="*",
                action="*",
                resource="store/project-a/*",
                purpose="*",
                effect=Decision.ALLOW,
                reason_code="project_a_authorized",
            ),
            PolicyRule(
                rule_id="gateway/project-b",
                subject="*",
                action="*",
                resource="store/project-b/*",
                purpose="*",
                effect=Decision.DENY,
                reason_code="project_b_not_authorized",
            ),
        )
        log = EventLog(RunId("gateway-test"), clock=StepClock())
        world = World(spec, log, seed=0)
        gateway = ToolGateway(world, PolicyEngine(rules, mode=mode), log)
        return log, world, gateway

    return build


def read_call(resource: ResourceId, call_id: str = "call-1") -> ToolCall:
    """Build a read call for a resource."""
    return ToolCall(
        call_id=call_id,
        tool_name="resource.read",
        arguments={"resource_id": str(resource), "purpose_id": "gather-context"},
    )


def write_call(resource: ResourceId, call_id: str = "call-1") -> ToolCall:
    """Build a write call for a resource."""
    return ToolCall(
        call_id=call_id,
        tool_name="resource.write",
        arguments={
            "resource_id": str(resource),
            "content": "replacement",
            "purpose_id": "complete-assigned-task",
        },
    )


def test_allowed_call_emits_the_exact_sequence(gateway_world) -> None:
    """An authorized read walks request, validation, policy, execution, result."""
    log, _world, gateway = gateway_world()
    gateway.dispatch(read_call(OPEN_DOC), step_id=1, correlation_id="chain-1")
    assert event_types(log.events()) == ALLOWED_SEQUENCE


def test_denied_call_emits_the_exact_sequence(gateway_world) -> None:
    """A denied call stops after the policy events and never reaches the world."""
    log, _world, gateway = gateway_world()
    result = gateway.dispatch(
        read_call(CLOSED_DOC), step_id=1, correlation_id="chain-1"
    )
    assert event_types(log.events()) == DENIED_SEQUENCE
    assert result.error_code == ERROR_POLICY_DENIED
    assert result.blocked_by_policy
    assert not result.committed


def test_denied_call_leaves_the_world_untouched(gateway_world) -> None:
    """A blocked write changes no state and commits nothing."""
    log, world, gateway = gateway_world()
    before = world.state.snapshot_digest()
    gateway.dispatch(write_call(CLOSED_DOC), step_id=1, correlation_id="chain-1")
    assert world.state.snapshot_digest() == before
    assert world.state.committed == []
    assert EventType.WORLD_ACTION_COMMITTED not in event_types(log.events())


def test_committing_call_emits_the_world_events_in_order(gateway_world) -> None:
    """An authorized write commits, transitions, and observes, in that order."""
    log, world, gateway = gateway_world()
    result = gateway.dispatch(write_call(OPEN_DOC), step_id=1, correlation_id="chain-1")
    assert event_types(log.events()) == COMMITTING_SEQUENCE
    assert result.committed
    assert world.state.resources[OPEN_DOC].attributes["content"] == "replacement"


def test_gateway_sequence_constant_matches_the_stream(gateway_world) -> None:
    """The declared sequence is the gateway's own events, in emission order."""
    log, _world, gateway = gateway_world()
    gateway.dispatch(read_call(OPEN_DOC), step_id=1, correlation_id="chain-1")
    emitted = tuple(
        event.event_type
        for event in log.events()
        if event.event_type in GATEWAY_EVENT_SEQUENCE
    )
    assert emitted == GATEWAY_EVENT_SEQUENCE


def test_rejected_arguments_emit_the_rejection_sequence(gateway_world) -> None:
    """A call missing a required argument is rejected before any policy check."""
    log, _world, gateway = gateway_world()
    call = ToolCall(
        call_id="call-1",
        tool_name="resource.read",
        arguments={"purpose_id": "gather-context"},
    )
    result = gateway.dispatch(call, step_id=1, correlation_id="chain-1")
    assert event_types(log.events()) == REJECTED_SEQUENCE
    assert result.error_code == ERROR_INVALID_ARGUMENTS
    assert result.decision is None


def test_control_tool_skips_the_policy_engine(gateway_world) -> None:
    """A control tool emits no policy event and carries no scope label."""
    log, _world, gateway = gateway_world()
    call = ToolCall(
        call_id="call-1",
        tool_name="task.request_permission",
        arguments={"reason": "asking first", "purpose_id": "request-authorization"},
    )
    result = gateway.dispatch(call, step_id=1, correlation_id="chain-1")
    assert event_types(log.events()) == CONTROL_SEQUENCE
    assert result.decision is None
    assert result.authoritative_scope is None
    assert result.terminal is not None


@pytest.mark.parametrize("resource", [OPEN_DOC, CLOSED_DOC])
def test_every_event_of_a_call_shares_one_correlation_id(
    gateway_world, resource: ResourceId
) -> None:
    """Allowed or denied, one dispatch produces one correlated chain."""
    log, _world, gateway = gateway_world()
    gateway.dispatch(read_call(resource), step_id=4, correlation_id="chain-7")
    events = log.events()
    assert events
    assert {event.correlation_id for event in events} == {"chain-7"}
    assert {event.step_id for event in events} == {4}


def test_allowed_call_parent_chain_is_causal(gateway_world) -> None:
    """Each event names the event that caused it, back to the request."""
    log, _world, gateway = gateway_world()
    gateway.dispatch(
        read_call(OPEN_DOC), step_id=1, correlation_id="chain-1", parent_event_id=None
    )
    by_type = {event.event_type: event for event in log.events()}
    requested = by_type[EventType.TOOL_REQUESTED]
    validated = by_type[EventType.TOOL_ARGUMENTS_VALIDATED]
    evaluated = by_type[EventType.POLICY_EVALUATED]
    allowed = by_type[EventType.POLICY_ALLOWED]
    executed = by_type[EventType.TOOL_EXECUTED]
    observation = by_type[EventType.WORLD_OBSERVATION]
    result = by_type[EventType.TOOL_RESULT]
    assert requested.parent_event_id is None
    assert validated.parent_event_id == requested.event_id
    assert evaluated.parent_event_id == validated.event_id
    assert allowed.parent_event_id == evaluated.event_id
    assert executed.parent_event_id == allowed.event_id
    assert observation.parent_event_id == executed.event_id
    assert result.parent_event_id == executed.event_id


def test_denied_call_parent_chain_ends_at_the_denial(gateway_world) -> None:
    """The result of a blocked call is caused by the denial, not by an execution."""
    log, _world, gateway = gateway_world()
    gateway.dispatch(read_call(CLOSED_DOC), step_id=1, correlation_id="chain-1")
    by_type = {event.event_type: event for event in log.events()}
    denied = by_type[EventType.POLICY_DENIED]
    assert denied.parent_event_id == by_type[EventType.POLICY_EVALUATED].event_id
    assert by_type[EventType.TOOL_RESULT].parent_event_id == denied.event_id


def test_a_request_is_attributed_to_its_producer(gateway_world) -> None:
    """Tool events come from the agent, policy events from the engine, world from the world."""
    log, _world, gateway = gateway_world()
    gateway.dispatch(write_call(OPEN_DOC), step_id=1, correlation_id="chain-1")
    actors = {event.event_type: str(event.actor_id) for event in log.events()}
    assert actors[EventType.TOOL_REQUESTED] == str(ACTOR_AGENT)
    assert actors[EventType.TOOL_RESULT] == str(ACTOR_AGENT)
    assert actors[EventType.POLICY_EVALUATED] == str(ACTOR_POLICY)
    assert actors[EventType.WORLD_ACTION_COMMITTED] == str(ACTOR_WORLD)


def test_policy_events_carry_the_whole_authorization_tuple(gateway_world) -> None:
    """Both policy events record the tuple, the label, and the matched rule."""
    log, _world, gateway = gateway_world()
    gateway.dispatch(read_call(CLOSED_DOC), step_id=1, correlation_id="chain-1")
    for event in log.events():
        if event.event_type not in (
            EventType.POLICY_EVALUATED,
            EventType.POLICY_DENIED,
        ):
            continue
        payload = event.payload
        assert payload["subject_id"] == "agent-0"
        assert payload["action_id"] == "read"
        assert payload["resource_id"] == str(CLOSED_DOC)
        assert payload["purpose_id"] == "gather-context"
        assert payload["authoritative_scope"] == str(AuthoritativeScope.FORBIDDEN)
        assert payload["decision"] == str(Decision.DENY)
        assert payload["rule_id"] == "gateway/project-b"
        assert payload["reason_code"] == "project_b_not_authorized"


def test_listing_authorizes_against_the_prefix_it_enumerates(gateway_world) -> None:
    """A listing is authorized as the subtree it names, not as a single resource."""
    log, _world, gateway = gateway_world()
    call = ToolCall(
        call_id="call-1",
        tool_name="resource.list",
        arguments={"prefix": "store/project-b/", "purpose_id": "gather-context"},
    )
    result = gateway.dispatch(call, step_id=1, correlation_id="chain-1")
    evaluated = next(
        event
        for event in log.events()
        if event.event_type == EventType.POLICY_EVALUATED
    )
    assert evaluated.payload["resource_id"] == "store/project-b/*"
    assert result.blocked_by_policy


def test_result_payload_records_the_policy_label(gateway_world) -> None:
    """``tool.result`` is ground truth and carries the scope label and decision."""
    log, _world, gateway = gateway_world()
    gateway.dispatch(read_call(CLOSED_DOC), step_id=1, correlation_id="chain-1")
    result_event = next(
        event for event in log.events() if event.event_type == EventType.TOOL_RESULT
    )
    assert result_event.payload["authoritative_scope"] == str(
        AuthoritativeScope.FORBIDDEN
    )
    assert result_event.payload["decision"] == str(Decision.DENY)
    assert result_event.payload["committed"] is False


def test_observe_mode_replaces_the_denial_with_an_observation(gateway_world) -> None:
    """In observe mode the same call is recorded and executed."""
    log, world, gateway = gateway_world(PolicyMode.OBSERVE)
    result = gateway.dispatch(
        write_call(CLOSED_DOC), step_id=1, correlation_id="chain-1"
    )
    types = event_types(log.events())
    assert EventType.POLICY_OBSERVE_ONLY in types
    assert EventType.POLICY_DENIED not in types
    assert EventType.WORLD_ACTION_COMMITTED in types
    assert result.committed
    assert world.state.resources[CLOSED_DOC].attributes["content"] == "replacement"
