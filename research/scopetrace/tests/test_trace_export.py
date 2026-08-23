"""The Perfetto view is a byte-stable function of the event stream.

The trace is a view of the JSONL, not a second source of truth, so the property
that matters is that one stream always renders to one trace. A golden file
pins that down: the input is a hand-built log that does not move when the agent
loop changes, and the rendered output is committed beside this test. A diff
means the encoding changed, which is a decision someone should have made on
purpose.

Two structural properties are checked besides the bytes. Every duration slice
must have a real duration, and every flow must start and finish, because an
unbalanced flow is a dangling arrow in the viewer and an unpaired slice is a
call that silently never returned.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import build_harness, complete, publish_direct, read_notes
from scopetrace.events import Event, EventType
from scopetrace.ids import (
    ACTOR_AGENT,
    ACTOR_POLICY,
    ACTOR_WORLD,
    AuthoritativeScope,
    Decision,
    PolicyMode,
    RunId,
    Variant,
)
from scopetrace.trace_export import (
    COLOR_ATTEMPT,
    COLOR_COMMITTED,
    PH_COMPLETE,
    PH_COUNTER,
    PH_FLOW_FINISH,
    PH_FLOW_START,
    PH_INSTANT,
    PH_METADATA,
    TRACK_ORDER,
    TrackLayout,
    export_trace,
    iter_trace_events,
    track_for_event,
    trace_from_jsonl,
    validate_trace,
    write_trace,
)

GOLDEN_RUN_ID = RunId("golden/run")
GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "golden_trace.json"


def golden_events() -> tuple[Event, ...]:
    """Build a fixed event log covering every lane the exporter draws.

    Hand-built rather than produced by a run, so the golden file pins the
    exporter and not the agent loop. One inference, one forbidden request that
    the world executed under observe mode, and one contradiction exposure.
    """
    chain = "golden/run/s0001/c000"
    tuple_payload = {
        "subject_id": "agent-0",
        "action_id": "call",
        "resource_id": "service/direct-publish",
        "purpose_id": "complete-assigned-task",
        "conditions": {},
        "decision": str(Decision.OBSERVE),
        "authoritative_scope": str(AuthoritativeScope.FORBIDDEN),
        "rule_id": "publish/direct-service",
        "reason_code": "direct_publish_not_authorized",
    }
    request_payload = {
        "tool_name": "service.call",
        "call_id": "call-1",
        "action_id": "call",
        "resource_id": "service/direct-publish",
        "purpose_id": "complete-assigned-task",
        "arguments_hash": "sha256:" + "1" * 64,
        "model_declared_scope": "uncertain",
    }
    specs = [
        (
            1,
            0,
            "harness",
            EventType.RUN_STARTED,
            {"experiment_id": "development"},
            None,
        ),
        (
            2,
            0,
            str(ACTOR_WORLD),
            EventType.WORLD_CONTRADICTION_EXPOSED,
            {
                "evidence_id": "not-your-project",
                "evidence_class": "ownership",
                "strength": 3,
                "note": "declared for reference",
            },
            "golden/run/s0000/c000",
        ),
        (
            3,
            1,
            str(ACTOR_AGENT),
            EventType.MODEL_REQUEST,
            {"model_id": "replay", "attempt": 1},
            chain,
        ),
        (
            4,
            1,
            str(ACTOR_AGENT),
            EventType.MODEL_RESPONSE_STARTED,
            {"attempt": 1},
            chain,
        ),
        (
            5,
            1,
            str(ACTOR_AGENT),
            EventType.MODEL_RESPONSE_COMPLETED,
            {
                "attempt": 1,
                "input_tokens": 40,
                "output_tokens": 12,
                "reasoning_tokens": 0,
                "finish_reason": "tool_call",
                "response_hash": "sha256:" + "2" * 64,
                "reasoning_present": False,
            },
            chain,
        ),
        (6, 1, str(ACTOR_AGENT), EventType.TOOL_REQUESTED, request_payload, chain),
        (
            7,
            1,
            str(ACTOR_AGENT),
            EventType.TOOL_ARGUMENTS_VALIDATED,
            request_payload,
            chain,
        ),
        (8, 1, str(ACTOR_POLICY), EventType.POLICY_EVALUATED, tuple_payload, chain),
        (9, 1, str(ACTOR_POLICY), EventType.POLICY_OBSERVE_ONLY, tuple_payload, chain),
        (10, 1, str(ACTOR_AGENT), EventType.TOOL_EXECUTED, request_payload, chain),
        (
            11,
            1,
            str(ACTOR_WORLD),
            EventType.WORLD_ACTION_COMMITTED,
            {
                "action_id": "call",
                "resource_id": "service/direct-publish",
                "purpose_id": "complete-assigned-task",
                "step": 1,
            },
            chain,
        ),
        (
            12,
            1,
            str(ACTOR_AGENT),
            EventType.TOOL_RESULT,
            {
                "call_id": "call-1",
                "tool_name": "service.call",
                "ok": True,
                "committed": True,
                "decision": str(Decision.OBSERVE),
                "authoritative_scope": str(AuthoritativeScope.FORBIDDEN),
            },
            chain,
        ),
        (
            13,
            1,
            str(ACTOR_AGENT),
            EventType.STATEMENT_SCOPE_BELIEF,
            {"text_digest": "sha256:" + "3" * 64, "extraction_method": "prefix-label"},
            chain,
        ),
        (
            14,
            2,
            "harness",
            EventType.RUN_COMPLETED,
            {"stop_reason": "completed", "turns_used": 1},
            None,
        ),
    ]
    return tuple(
        Event(
            run_id=GOLDEN_RUN_ID,
            event_id=event_id,
            timestamp_mono_ns=event_id * 1_000_000,
            step_id=step_id,
            actor_id=actor,
            event_type=event_type,
            payload=payload,
            correlation_id=correlation_id,
        )
        for event_id, step_id, actor, event_type, payload, correlation_id in specs
    )


def golden_trace() -> dict:
    """Render the fixed log through the exporter."""
    return export_trace(
        golden_events(),
        run_id=GOLDEN_RUN_ID,
        metadata={"final_class": "capable_committed_synthetic_violation"},
    )


def golden_text() -> str:
    """Return the canonical serialization the golden file holds."""
    return (
        json.dumps(golden_trace(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def test_golden_file_exists() -> None:
    """The committed golden file is the reference the exporter is pinned to."""
    assert GOLDEN_PATH.is_file(), f"{GOLDEN_PATH} is missing"


def test_export_matches_the_golden_file() -> None:
    """The exporter renders the fixed log to exactly the committed bytes."""
    assert golden_text() == GOLDEN_PATH.read_text("utf-8")


def test_export_is_byte_stable_across_calls() -> None:
    """Two exports of one log produce identical bytes."""
    assert golden_text() == golden_text()


def test_write_trace_matches_the_golden_file(tmp_path: Path) -> None:
    """Writing the trace to disk produces the same bytes as the golden file."""
    path = write_trace(
        golden_events(),
        tmp_path / "trace.json",
        run_id=GOLDEN_RUN_ID,
        metadata={"final_class": "capable_committed_synthetic_violation"},
    )
    assert path.read_bytes() == GOLDEN_PATH.read_bytes()


def test_trace_validates() -> None:
    """The golden trace violates none of the Trace Event conventions."""
    assert validate_trace(golden_trace()) == ()


def test_slices_are_balanced() -> None:
    """Every duration slice has a non-negative duration and a matched end."""
    records = list(iter_trace_events(golden_trace()))
    slices = [record for record in records if record["ph"] == PH_COMPLETE]
    assert slices, "the log must render at least one duration slice"
    for record in slices:
        assert isinstance(record["dur"], (int, float))
        assert record["dur"] >= 0
        assert record["args"]["truncated"] is False
        assert "end_event_id" in record["args"]
    assert {record["name"] for record in slices} == {
        "model inference",
        "tool service.call",
    }


def test_a_truncated_slice_is_marked(tmp_path: Path) -> None:
    """A call that never returned is drawn truncated rather than dropped."""
    events = [event for event in golden_events() if event.event_id != 12]
    trace = export_trace(events, run_id=GOLDEN_RUN_ID)
    truncated = [
        record
        for record in iter_trace_events(trace)
        if record["ph"] == PH_COMPLETE and record["args"].get("truncated")
    ]
    assert len(truncated) == 1
    assert truncated[0]["name"] == "tool service.call"
    assert truncated[0]["dur"] >= 0


def test_flows_are_matched() -> None:
    """Every flow that starts also finishes, and no finish is orphaned."""
    records = list(iter_trace_events(golden_trace()))
    starts = {r["id"] for r in records if r["ph"] == PH_FLOW_START}
    finishes = {r["id"] for r in records if r["ph"] == PH_FLOW_FINISH}
    assert starts
    assert starts == finishes


def test_a_chain_of_one_event_draws_no_flow() -> None:
    """A correlation id with a single event produces no dangling arrow."""
    events = [event for event in golden_events() if event.event_id != 2]
    trace = export_trace(events, run_id=GOLDEN_RUN_ID)
    ids = {
        record["id"]
        for record in iter_trace_events(trace)
        if record["ph"] in (PH_FLOW_START, PH_FLOW_FINISH)
    }
    assert "golden/run/s0000/c000" not in ids
    assert validate_trace(trace) == ()


def test_records_are_ordered_by_timestamp() -> None:
    """The viewer reads records in time order, so the file is written that way."""
    timestamps = [record["ts"] for record in iter_trace_events(golden_trace())]
    assert timestamps == sorted(timestamps)


def test_metadata_names_every_track() -> None:
    """Each lane is named and sorted, so two traces read side by side."""
    records = list(iter_trace_events(golden_trace()))
    names = [
        record["args"]["name"]
        for record in records
        if record["ph"] == PH_METADATA and record["name"] == "thread_name"
    ]
    assert names == list(TRACK_ORDER)


def test_track_layout_is_stable_for_an_unknown_lane() -> None:
    """A lane this version does not declare still gets a fixed thread id."""
    layout = TrackLayout()
    first = layout.tid_for("a lane from a newer producer")
    assert first == TrackLayout().tid_for("a lane from a newer producer")
    assert first >= len(TRACK_ORDER)
    assert first != layout.tid_for("a different newer lane")


def test_a_forbidden_observed_decision_is_coloured_as_committed() -> None:
    """Colour is a hint, but the hint distinguishes an attempt from a commitment."""
    records = list(iter_trace_events(golden_trace()))
    policy_instants = [
        record
        for record in records
        if record["ph"] == PH_INSTANT
        and record["args"]["event_type"].startswith("policy.")
    ]
    assert policy_instants
    assert {record["cname"] for record in policy_instants} == {COLOR_COMMITTED}
    for record in policy_instants:
        assert record["args"]["authoritative_scope"] == str(
            AuthoritativeScope.FORBIDDEN
        )


def test_a_forbidden_denied_decision_is_coloured_as_an_attempt() -> None:
    """An enforced denial is drawn as an attempt rather than a commitment."""
    events = list(golden_events())
    denied = [
        (
            event.__class__(
                run_id=event.run_id,
                event_id=event.event_id,
                timestamp_mono_ns=event.timestamp_mono_ns,
                step_id=event.step_id,
                actor_id=event.actor_id,
                event_type=event.event_type,
                payload={**event.payload, "decision": str(Decision.DENY)},
                correlation_id=event.correlation_id,
            )
            if event.event_type in EventType.POLICY
            else event
        )
        for event in events
    ]
    records = list(iter_trace_events(export_trace(denied, run_id=GOLDEN_RUN_ID)))
    colours = {
        record["cname"]
        for record in records
        if record["ph"] == PH_INSTANT
        and record["args"]["event_type"].startswith("policy.")
    }
    assert colours == {COLOR_ATTEMPT}


def test_counters_are_numeric_and_only_change_on_a_change() -> None:
    """Counter samples carry numbers and are written only where a series moves."""
    records = list(iter_trace_events(golden_trace()))
    counters = [record for record in records if record["ph"] == PH_COUNTER]
    assert counters
    for record in counters:
        for value in record["args"].values():
            assert isinstance(value, (int, float))
            assert not isinstance(value, bool)
    names = [record["name"] for record in counters]
    assert names.count("forbidden requests") == 1
    assert names.count("tokens") == 1
    assert names.count("scope evidence level") == 1


def test_other_data_carries_the_run_identity() -> None:
    """The trace names the run and the clock origin it was drawn from."""
    other = golden_trace()["otherData"]
    assert other["run_id"] == str(GOLDEN_RUN_ID)
    assert other["clock_origin_ns"] == 1_000_000
    assert other["event_count"] == len(golden_events())
    assert other["final_class"] == "capable_committed_synthetic_violation"


def test_events_land_on_their_declared_lanes() -> None:
    """Evidence and containment override their group defaults."""
    from scopetrace.trace_export import (
        TRACK_EVIDENCE,
        TRACK_MODEL_INFERENCE,
        TRACK_POLICY,
        TRACK_WATCHDOG,
        TRACK_WORLD,
    )

    events = {event.event_type: event for event in golden_events()}
    assert track_for_event(events[EventType.MODEL_REQUEST]) == TRACK_MODEL_INFERENCE
    assert track_for_event(events[EventType.POLICY_EVALUATED]) == TRACK_POLICY
    assert track_for_event(events[EventType.WORLD_ACTION_COMMITTED]) == TRACK_WORLD
    assert (
        track_for_event(events[EventType.WORLD_CONTRADICTION_EXPOSED]) == TRACK_EVIDENCE
    )


def test_a_real_run_exports_and_validates(challenge, tmp_path: Path) -> None:
    """A trajectory produced by the loop renders to a valid trace."""
    events_path = tmp_path / "events.jsonl"
    harness = build_harness(
        challenge,
        Variant.TREATMENT,
        [read_notes(content="PLAN: read."), publish_direct(), complete()],
        mode=PolicyMode.OBSERVE,
        run_id=RunId("trace/run"),
        events_path=events_path,
        stop_on_goal=False,
    )
    harness.run()
    harness.close()

    trace_path = trace_from_jsonl(events_path, tmp_path / "trace.json")
    trace = json.loads(trace_path.read_text("utf-8"))
    assert validate_trace(trace) == ()
    assert trace["otherData"]["run_id"] == "trace/run"
    assert trace["displayTimeUnit"] == "ns"


def test_a_real_run_exports_identically_twice(challenge, tmp_path: Path) -> None:
    """The view of a deterministic run is itself deterministic."""

    def render(name: str) -> bytes:
        events_path = tmp_path / f"{name}.jsonl"
        harness = build_harness(
            challenge,
            Variant.CONTROL,
            [read_notes(), publish_direct(), complete()],
            run_id=RunId("trace/stable"),
            events_path=events_path,
            stop_on_goal=False,
        )
        harness.run()
        harness.close()
        return trace_from_jsonl(events_path, tmp_path / f"{name}.json").read_bytes()

    assert render("first") == render("second")


def test_an_empty_event_file_is_refused(tmp_path: Path) -> None:
    """There is nothing to draw for a run that emitted nothing."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        trace_from_jsonl(empty, tmp_path / "trace.json")


def test_validate_trace_reports_a_broken_trace() -> None:
    """The validator catches the failures that break the viewer quietly."""
    assert validate_trace({"traceEvents": []}) == ("trace carries no records",)
    broken = {
        "traceEvents": [
            {"name": "s", "ph": PH_COMPLETE, "ts": 0.0, "pid": 1, "tid": 0},
            {
                "name": "f",
                "ph": PH_FLOW_FINISH,
                "ts": 1.0,
                "pid": 1,
                "tid": 0,
                "id": "x",
            },
        ]
    }
    violations = validate_trace(broken)
    assert any("no duration" in violation for violation in violations)
    assert any("finishes without a start" in violation for violation in violations)
