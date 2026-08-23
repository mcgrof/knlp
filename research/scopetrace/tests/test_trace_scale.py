"""A few thousand events still export, still validate, and stay a sane size.

The trace is meant to be opened in a viewer and read by a person, so a long run
must not turn into a file nobody can load. This is a smoke test rather than a
benchmark: it asserts that the exporter finishes, that the result is still a
well-formed trace, and that the bytes per event stay within an order of
magnitude of the payloads that produced them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from scopetrace.events import Event, EventType, write_events
from scopetrace.ids import (
    ACTOR_AGENT,
    ACTOR_POLICY,
    ACTOR_WORLD,
    AuthoritativeScope,
    Decision,
    RunId,
)
from scopetrace.trace_export import (
    PH_COMPLETE,
    PH_FLOW_FINISH,
    PH_FLOW_START,
    export_trace,
    iter_trace_events,
    trace_from_jsonl,
    validate_trace,
)

TURNS = 400
EVENTS_PER_TURN = 10
RUN_ID = RunId("scale/run")

MAX_BYTES_PER_EVENT = 3_000
"""Ceiling on the rendered size per source event. Generous, because the point
is to catch a blow-up rather than to pin the encoding."""

MAX_SECONDS = 10.0
"""Ceiling on the export itself. A conversion this size is not a computation."""


def large_stream(turns: int = TURNS) -> tuple[Event, ...]:
    """Build a long, well-formed stream of correlated turns."""
    events: list[Event] = []
    event_id = 0

    def add(step: int, actor, event_type: str, payload, correlation: str | None):
        nonlocal event_id
        event_id += 1
        events.append(
            Event(
                run_id=RUN_ID,
                event_id=event_id,
                timestamp_mono_ns=event_id * 1_000_000,
                step_id=step,
                actor_id=actor,
                event_type=event_type,
                payload=payload,
                correlation_id=correlation,
            )
        )

    add(0, "harness", EventType.RUN_STARTED, {"experiment_id": "scale"}, None)
    for step in range(1, turns + 1):
        chain = f"{RUN_ID}/s{step:04d}/c000"
        forbidden = step % 4 == 0
        scope = (
            str(AuthoritativeScope.FORBIDDEN)
            if forbidden
            else str(AuthoritativeScope.ALLOWED)
        )
        decision = str(Decision.DENY) if forbidden else str(Decision.ALLOW)
        resource = f"store/project-{'b' if forbidden else 'a'}/item-{step}"
        tuple_payload = {
            "subject_id": "agent-0",
            "action_id": "read",
            "resource_id": resource,
            "purpose_id": "gather-context",
            "conditions": {},
            "decision": decision,
            "authoritative_scope": scope,
            "rule_id": "scale/rule",
            "reason_code": "scale_reason",
        }
        request_payload = {
            "tool_name": "resource.read",
            "call_id": f"call-{step}",
            "action_id": "read",
            "resource_id": resource,
            "purpose_id": "gather-context",
            "arguments_hash": "sha256:" + f"{step:064x}",
            "model_declared_scope": "unstated",
        }
        add(step, ACTOR_AGENT, EventType.MODEL_REQUEST, {"attempt": 1}, chain)
        add(step, ACTOR_AGENT, EventType.MODEL_RESPONSE_STARTED, {"attempt": 1}, chain)
        add(
            step,
            ACTOR_AGENT,
            EventType.MODEL_RESPONSE_COMPLETED,
            {
                "attempt": 1,
                "input_tokens": 30,
                "output_tokens": 8,
                "finish_reason": "tool_call",
                "response_hash": "sha256:" + f"{step:064x}",
                "reasoning_present": False,
            },
            chain,
        )
        add(step, ACTOR_AGENT, EventType.TOOL_REQUESTED, request_payload, chain)
        add(
            step,
            ACTOR_AGENT,
            EventType.TOOL_ARGUMENTS_VALIDATED,
            request_payload,
            chain,
        )
        add(step, ACTOR_POLICY, EventType.POLICY_EVALUATED, tuple_payload, chain)
        add(
            step,
            ACTOR_POLICY,
            EventType.POLICY_DENIED if forbidden else EventType.POLICY_ALLOWED,
            tuple_payload,
            chain,
        )
        add(step, ACTOR_AGENT, EventType.TOOL_EXECUTED, request_payload, chain)
        add(
            step,
            ACTOR_WORLD,
            EventType.WORLD_OBSERVATION,
            {
                "action_id": "read",
                "resource_id": resource,
                "ok": True,
                "kind": "content",
            },
            chain,
        )
        add(
            step,
            ACTOR_AGENT,
            EventType.TOOL_RESULT,
            {
                "call_id": f"call-{step}",
                "tool_name": "resource.read",
                "ok": not forbidden,
                "committed": False,
                "decision": decision,
                "authoritative_scope": scope,
            },
            chain,
        )
    return tuple(events)


@pytest.fixture(scope="module")
def stream() -> tuple[Event, ...]:
    """Build the long stream once for the whole module."""
    return large_stream()


def test_the_stream_is_large_enough_to_be_a_test(stream) -> None:
    """The smoke test really does run over a few thousand events."""
    assert len(stream) == 1 + TURNS * EVENTS_PER_TURN
    assert len(stream) >= 4000


def test_export_completes_within_the_budget(stream) -> None:
    """Converting a long run is a linear pass, not a computation."""
    started = time.monotonic()
    trace = export_trace(stream, run_id=RUN_ID)
    elapsed = time.monotonic() - started
    assert elapsed < MAX_SECONDS, f"export took {elapsed:.2f}s"
    assert trace["otherData"]["event_count"] == len(stream)


def test_the_long_trace_validates(stream) -> None:
    """Nothing about scale breaks the Trace Event conventions."""
    assert validate_trace(export_trace(stream, run_id=RUN_ID)) == ()


def test_the_long_trace_stays_a_sane_size(stream, tmp_path: Path) -> None:
    """Rendered bytes per source event stay within a readable bound."""
    path = tmp_path / "events.jsonl"
    write_events(path, stream)
    trace_path = trace_from_jsonl(path, tmp_path / "trace.json")
    size = trace_path.stat().st_size
    per_event = size / len(stream)
    assert per_event < MAX_BYTES_PER_EVENT, f"{per_event:.0f} bytes per event"
    assert size < 40 * 1024 * 1024


def test_the_long_trace_keeps_its_flows_balanced(stream) -> None:
    """Every one of the correlated turns renders as a complete flow."""
    records = list(iter_trace_events(export_trace(stream, run_id=RUN_ID)))
    starts = {r["id"] for r in records if r["ph"] == PH_FLOW_START}
    finishes = {r["id"] for r in records if r["ph"] == PH_FLOW_FINISH}
    assert len(starts) == TURNS
    assert starts == finishes


def test_the_long_trace_draws_every_slice(stream) -> None:
    """Each turn draws one inference slice and one tool slice."""
    records = list(iter_trace_events(export_trace(stream, run_id=RUN_ID)))
    slices = [record for record in records if record["ph"] == PH_COMPLETE]
    assert len(slices) == TURNS * 2
    assert all(record["dur"] >= 0 for record in slices)


def test_the_long_trace_round_trips_through_disk(stream, tmp_path: Path) -> None:
    """A written trace parses back into the same object."""
    path = tmp_path / "events.jsonl"
    write_events(path, stream)
    trace_path = trace_from_jsonl(path, tmp_path / "trace.json")
    reloaded = json.loads(trace_path.read_text("utf-8"))
    assert reloaded == export_trace(stream, run_id=RUN_ID)
