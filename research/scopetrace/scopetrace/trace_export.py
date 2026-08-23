"""Export an event stream as Trace Event JSON for the Perfetto UI.

The JSONL stream is canonical; this is a view of it. The exporter maps one run
onto one process whose threads are the semantic tracks, so a reader sees
inference, tool calls, policy decisions, world transitions, evidence, model
statements, annotations, and the watchdog on parallel lanes with a shared time
axis. The question it is built to answer at a glance is when the agent first
saw a forbidden route, what it had been told by then, and what it did next.

Trace Event JSON is used rather than the newer track-event protobuf because it
is readable in a text editor, diffable in a golden test, and easy to check by
hand while the schema is still settling. The format choice is an
implementation detail of the view, not of the data.

Encoding rules. Inference and tool execution become duration slices. Policy
decisions and evidence exposure become instant events. Cumulative denials,
tokens, and the running evidence level become counters. The chain from a plan
through a request and a policy decision into a world transition becomes flow
events, linked by correlation id. Colour is a hint attached with ``cname`` and
never the only encoding: every slice also carries its meaning in its name and
its arguments, so a colour-blind reader and a text search both work.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .events import Event, EventType, load_events
from .ids import AuthoritativeScope, Decision, RunId

TRACK_MODEL_INFERENCE: Final[str] = "model inference"
TRACK_AGENT_DECISIONS: Final[str] = "agent decisions"
TRACK_TOOL_GATEWAY: Final[str] = "tool gateway"
TRACK_POLICY: Final[str] = "policy"
TRACK_WORLD: Final[str] = "synthetic world"
TRACK_EVIDENCE: Final[str] = "contradiction evidence"
TRACK_STATEMENTS: Final[str] = "model statements"
TRACK_ANNOTATIONS: Final[str] = "derived annotations"
TRACK_WATCHDOG: Final[str] = "watchdog and containment"
TRACK_COUNTERS: Final[str] = "counters"

TRACK_ORDER: Final[tuple[str, ...]] = (
    TRACK_MODEL_INFERENCE,
    TRACK_AGENT_DECISIONS,
    TRACK_TOOL_GATEWAY,
    TRACK_POLICY,
    TRACK_WORLD,
    TRACK_EVIDENCE,
    TRACK_STATEMENTS,
    TRACK_ANNOTATIONS,
    TRACK_WATCHDOG,
    TRACK_COUNTERS,
)
"""Track lanes, top to bottom. The order is part of the golden file."""

PH_COMPLETE: Final[str] = "X"
PH_INSTANT: Final[str] = "i"
PH_COUNTER: Final[str] = "C"
PH_FLOW_START: Final[str] = "s"
PH_FLOW_STEP: Final[str] = "t"
PH_FLOW_FINISH: Final[str] = "f"
PH_METADATA: Final[str] = "M"

SCOPE_THREAD: Final[str] = "t"
SCOPE_PROCESS: Final[str] = "p"
SCOPE_GLOBAL: Final[str] = "g"

COLOR_ALLOWED: Final[str] = "good"
COLOR_AMBIGUOUS: Final[str] = "yellow"
COLOR_DENIED: Final[str] = "olive"
COLOR_ATTEMPT: Final[str] = "bad"
COLOR_COMMITTED: Final[str] = "terrible"
COLOR_STATEMENT: Final[str] = "rail_response"
COLOR_CONTAINMENT: Final[str] = "vsync_highlight_color"
COLOR_NEUTRAL: Final[str] = "grey"

COLOR_HINTS: Final[Mapping[str, str]] = {
    "allowed": COLOR_ALLOWED,
    "ambiguous": COLOR_AMBIGUOUS,
    "denied": COLOR_DENIED,
    "attempt": COLOR_ATTEMPT,
    "committed": COLOR_COMMITTED,
    "statement": COLOR_STATEMENT,
    "containment": COLOR_CONTAINMENT,
    "neutral": COLOR_NEUTRAL,
}
"""Semantic name to ``cname`` hint. A reader must be able to reach the same
conclusion with colour turned off, so nothing is encoded here alone."""

COUNTER_DENIALS: Final[str] = "cumulative denials"
COUNTER_FORBIDDEN_REQUESTS: Final[str] = "forbidden requests"
COUNTER_ALLOWED_REQUESTS: Final[str] = "allowed requests"
COUNTER_TOKENS: Final[str] = "tokens"
COUNTER_EVIDENCE_LEVEL: Final[str] = "scope evidence level"

COUNTER_NAMES: Final[tuple[str, ...]] = (
    COUNTER_DENIALS,
    COUNTER_FORBIDDEN_REQUESTS,
    COUNTER_ALLOWED_REQUESTS,
    COUNTER_TOKENS,
    COUNTER_EVIDENCE_LEVEL,
)

DISPLAY_TIME_UNIT: Final[str] = "ns"

TRACK_BY_EVENT_TYPE: Final[Mapping[str, str]] = {
    **{event_type: TRACK_MODEL_INFERENCE for event_type in EventType.MODEL},
    **{event_type: TRACK_TOOL_GATEWAY for event_type in EventType.TOOL},
    **{event_type: TRACK_POLICY for event_type in EventType.POLICY},
    **{event_type: TRACK_WORLD for event_type in EventType.WORLD},
    **{event_type: TRACK_STATEMENTS for event_type in EventType.STATEMENT},
    **{event_type: TRACK_ANNOTATIONS for event_type in EventType.ANNOTATION},
    EventType.WORLD_OPPORTUNITY_EXPOSED: TRACK_EVIDENCE,
    EventType.WORLD_CONTRADICTION_EXPOSED: TRACK_EVIDENCE,
    EventType.CONTAINMENT_CHECK: TRACK_WATCHDOG,
    EventType.CONTAINMENT_FAILURE: TRACK_WATCHDOG,
    EventType.WATCHDOG_TRIGGERED: TRACK_WATCHDOG,
    EventType.RUN_STARTED: TRACK_AGENT_DECISIONS,
    EventType.RUN_CONFIGURATION_VALIDATED: TRACK_AGENT_DECISIONS,
    EventType.RUN_COMPLETED: TRACK_AGENT_DECISIONS,
    EventType.RUN_INVALIDATED: TRACK_AGENT_DECISIONS,
    EventType.RUN_KILLED: TRACK_AGENT_DECISIONS,
}
"""Which lane each event type is drawn on. Evidence and containment override
the group defaults so the two lanes a reader scans first stay uncluttered."""


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One Trace Event JSON record.

    Fields that do not apply to a phase stay ``None`` and are dropped on
    serialization, so a slice does not carry an empty counter field and a
    golden file stays readable.
    """

    name: str
    ph: str
    ts: float
    pid: int = 1
    tid: int = 0
    cat: str = ""
    dur: float | None = None
    args: Mapping[str, Any] = field(default_factory=dict)
    id: str | None = None
    scope: str | None = None
    cname: str | None = None
    bp: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return the record with unset optional fields omitted."""
        record: dict[str, Any] = {
            "name": self.name,
            "ph": self.ph,
            "ts": self.ts,
            "pid": self.pid,
            "tid": self.tid,
        }
        if self.cat:
            record["cat"] = self.cat
        if self.dur is not None:
            record["dur"] = self.dur
        if self.args:
            record["args"] = dict(self.args)
        if self.id is not None:
            record["id"] = self.id
        if self.scope is not None:
            record["scope"] = self.scope
        if self.cname is not None:
            record["cname"] = self.cname
        if self.bp is not None:
            record["bp"] = self.bp
        return record


@dataclass(frozen=True, slots=True)
class TrackLayout:
    """Assignment of track names to thread ids within the run process.

    Ids are the index in :data:`TRACK_ORDER`, so a lane keeps its position
    across runs and two traces can be read side by side.
    """

    tracks: tuple[str, ...] = TRACK_ORDER
    pid: int = 1

    def tid_for(self, track: str) -> int:
        """Return the thread id for a track.

        A declared track gets its index in :attr:`tracks`. A track this layout
        does not declare gets a stable id past the end of them, so an event
        from a newer producer still lands on a lane of its own instead of
        being folded into an existing one. The undeclared id is derived from
        the track name rather than from the order it was first seen, so two
        runs that met different subsets of the newer producer's lanes still
        draw them in the same place.
        """
        if track in self.tracks:
            return self.tracks.index(track)
        digest = hashlib.sha256(track.encode("utf-8")).hexdigest()
        return len(self.tracks) + 1 + int(digest[:8], 16) % 1000

    def metadata_events(self, *, process_name: str) -> tuple[TraceEvent, ...]:
        """Return the process and thread naming records for the trace header."""
        records = [
            TraceEvent(
                name="process_name",
                ph=PH_METADATA,
                ts=0.0,
                pid=self.pid,
                tid=0,
                args={"name": process_name},
            )
        ]
        for index, track in enumerate(self.tracks):
            tid = self.tid_for(track)
            records.append(
                TraceEvent(
                    name="thread_name",
                    ph=PH_METADATA,
                    ts=0.0,
                    pid=self.pid,
                    tid=tid,
                    args={"name": track},
                )
            )
            records.append(
                TraceEvent(
                    name="thread_sort_index",
                    ph=PH_METADATA,
                    ts=0.0,
                    pid=self.pid,
                    tid=tid,
                    args={"sort_index": index},
                )
            )
        return tuple(records)


class TraceExporter:
    """Converts one run's events into Trace Event JSON records.

    Timestamps are microseconds relative to the first event, which keeps a
    trace readable and makes two runs comparable from zero. The origin is
    recorded in the trace metadata so an absolute time can be recovered.
    """

    def __init__(
        self,
        *,
        run_id: RunId,
        layout: TrackLayout | None = None,
        process_name: str = "run",
        clock_origin_ns: int | None = None,
    ) -> None:
        self.run_id = run_id
        self.layout = layout if layout is not None else TrackLayout()
        self.process_name = process_name
        self.clock_origin_ns = clock_origin_ns
        self._truncate_ns: int | None = None

    def convert(self, events: Sequence[Event]) -> list[TraceEvent]:
        """Convert a whole event stream into trace records, in time order."""
        ordered = sorted(
            events, key=lambda event: (event.timestamp_mono_ns, event.event_id)
        )
        if self.clock_origin_ns is None and ordered:
            self.clock_origin_ns = ordered[0].timestamp_mono_ns
        self._truncate_ns = ordered[-1].timestamp_mono_ns if ordered else None

        records: list[TraceEvent] = list(
            self.layout.metadata_events(process_name=self.process_name)
        )
        consumed: set[int] = set()
        for start, end in _slice_pairs(ordered):
            track = track_for_event(start)
            record = self.slice_for(start, end, track=track)
            if record is not None:
                records.append(record)
            consumed.add(start.event_id)
            if end is not None:
                consumed.add(end.event_id)
        for event in ordered:
            if event.event_id in consumed:
                continue
            records.append(self.instant_for(event, track=track_for_event(event)))
        records.extend(self._flows(ordered))
        records.extend(self.counters(ordered))
        records.sort(
            key=lambda record: (
                record.ts,
                _PHASE_ORDER.get(record.ph, len(_PHASE_ORDER)),
                record.tid,
                record.name,
                record.id or "",
            )
        )
        return records

    def _flows(self, events: Sequence[Event]) -> list[TraceEvent]:
        """Build the flow records linking every correlated chain.

        A chain of one event produces nothing: a flow that starts and never
        finishes is a dangling arrow in the viewer and an unbalanced record in
        the golden file.
        """
        chains: dict[str, list[Event]] = {}
        for event in events:
            if event.correlation_id:
                chains.setdefault(event.correlation_id, []).append(event)
        records: list[TraceEvent] = []
        for correlation_id, chain in sorted(chains.items()):
            if len(chain) < 2:
                continue
            for index, event in enumerate(chain):
                if index == 0:
                    phase = PH_FLOW_START
                elif index == len(chain) - 1:
                    phase = PH_FLOW_FINISH
                else:
                    phase = PH_FLOW_STEP
                records.append(
                    self.flow_for(
                        event,
                        phase=phase,
                        track=track_for_event(event),
                        correlation_id=correlation_id,
                    )
                )
        return records

    def timestamp_us(self, event: Event) -> float:
        """Return an event's timestamp in microseconds from the trace origin."""
        return self._to_us(event.timestamp_mono_ns)

    def _to_us(self, timestamp_mono_ns: int) -> float:
        """Convert a monotonic nanosecond stamp to microseconds from the origin."""
        origin = self.clock_origin_ns if self.clock_origin_ns is not None else 0
        return round((timestamp_mono_ns - origin) / 1000.0, 3)

    def slice_for(
        self, start: Event, end: Event | None, *, track: str
    ) -> TraceEvent | None:
        """Build a duration slice from a start event and its completion.

        Inference spans ``model.request`` to ``model.response_completed``, and
        tool execution spans ``tool.executed`` to ``tool.result``. A start
        without an end, which is what a killed run leaves behind, produces a
        slice truncated at the last event rather than no slice at all.
        """
        start_us = self.timestamp_us(start)
        if end is not None:
            end_us = self.timestamp_us(end)
        elif self._truncate_ns is not None:
            end_us = self._to_us(self._truncate_ns)
        else:
            end_us = start_us
        args = _event_args(start)
        args["truncated"] = end is None
        if end is not None:
            args["end_event_id"] = end.event_id
            for key, value in end.payload.items():
                args.setdefault(key, value)
        return TraceEvent(
            name=_slice_name(start),
            ph=PH_COMPLETE,
            ts=start_us,
            pid=self.layout.pid,
            tid=self.layout.tid_for(track),
            cat=_category(start),
            dur=round(max(end_us - start_us, 0.0), 3),
            args=args,
            cname=self.color_for(end if end is not None else start),
        )

    def instant_for(self, event: Event, *, track: str) -> TraceEvent:
        """Build an instant record for a policy decision, evidence, or statement."""
        return TraceEvent(
            name=event.event_type,
            ph=PH_INSTANT,
            ts=self.timestamp_us(event),
            pid=self.layout.pid,
            tid=self.layout.tid_for(track),
            cat=_category(event),
            args=_event_args(event),
            scope=SCOPE_THREAD,
            cname=self.color_for(event),
        )

    def counter_for(
        self, event: Event, name: str, values: Mapping[str, float]
    ) -> TraceEvent:
        """Build a counter sample."""
        return TraceEvent(
            name=name,
            ph=PH_COUNTER,
            ts=self.timestamp_us(event),
            pid=self.layout.pid,
            tid=self.layout.tid_for(TRACK_COUNTERS),
            cat="counter",
            args={key: value for key, value in sorted(values.items())},
        )

    def flow_for(
        self, event: Event, *, phase: str, track: str, correlation_id: str
    ) -> TraceEvent:
        """Build a flow record linking one step of a correlated chain.

        The chain runs from the plan through the tool request and the policy
        decision into the world transition, so a reader can follow one decision
        across every lane it touched.
        """
        return TraceEvent(
            name=correlation_id,
            ph=phase,
            ts=self.timestamp_us(event),
            pid=self.layout.pid,
            tid=self.layout.tid_for(track),
            cat="flow",
            id=flow_id(correlation_id),
            bp="e" if phase == PH_FLOW_FINISH else None,
        )

    def counters(self, events: Sequence[Event]) -> list[TraceEvent]:
        """Build the running counter series for the whole stream.

        A sample is written only where a series changes, so a long run of
        uneventful turns does not fill the trace with repeated values.
        """
        totals: dict[str, float] = {name: 0.0 for name in COUNTER_NAMES}
        records: list[TraceEvent] = []
        decision_ids = _decision_event_ids(events)
        for event in events:
            changed: set[str] = set()
            if event.event_id in decision_ids:
                scope = str(event.payload.get("authoritative_scope", ""))
                decision = str(event.payload.get("decision", ""))
                if decision == str(Decision.DENY):
                    totals[COUNTER_DENIALS] += 1
                    changed.add(COUNTER_DENIALS)
                if scope == str(AuthoritativeScope.FORBIDDEN):
                    totals[COUNTER_FORBIDDEN_REQUESTS] += 1
                    changed.add(COUNTER_FORBIDDEN_REQUESTS)
                elif scope == str(AuthoritativeScope.ALLOWED):
                    totals[COUNTER_ALLOWED_REQUESTS] += 1
                    changed.add(COUNTER_ALLOWED_REQUESTS)
            if event.event_type == EventType.MODEL_RESPONSE_COMPLETED:
                tokens = int(event.payload.get("input_tokens", 0) or 0) + int(
                    event.payload.get("output_tokens", 0) or 0
                )
                if tokens:
                    totals[COUNTER_TOKENS] += tokens
                    changed.add(COUNTER_TOKENS)
            if event.event_type in (
                EventType.WORLD_OPPORTUNITY_EXPOSED,
                EventType.WORLD_CONTRADICTION_EXPOSED,
            ):
                strength = int(event.payload.get("strength", 0) or 0)
                totals[COUNTER_EVIDENCE_LEVEL] += strength
                changed.add(COUNTER_EVIDENCE_LEVEL)
            for name in sorted(changed):
                records.append(self.counter_for(event, name, {"value": totals[name]}))
        return records

    def color_for(self, event: Event) -> str | None:
        """Return the colour hint for an event, or ``None`` for the default."""
        if event.event_type in (
            EventType.CONTAINMENT_FAILURE,
            EventType.WATCHDOG_TRIGGERED,
            EventType.RUN_KILLED,
        ):
            return COLOR_CONTAINMENT
        if event.event_type in EventType.STATEMENT:
            return COLOR_STATEMENT
        if event.event_type in (
            EventType.WORLD_OPPORTUNITY_EXPOSED,
            EventType.WORLD_CONTRADICTION_EXPOSED,
        ):
            return COLOR_AMBIGUOUS
        if event.event_type in EventType.POLICY:
            scope = str(event.payload.get("authoritative_scope", ""))
            decision = str(event.payload.get("decision", ""))
            if scope == str(AuthoritativeScope.FORBIDDEN):
                if decision == str(Decision.OBSERVE):
                    return COLOR_COMMITTED
                return COLOR_ATTEMPT
            if scope == str(AuthoritativeScope.AMBIGUOUS):
                return COLOR_AMBIGUOUS
            if decision == str(Decision.DENY):
                return COLOR_DENIED
            return COLOR_ALLOWED
        return None


_PHASE_ORDER: Final[Mapping[str, int]] = {
    PH_METADATA: 0,
    PH_COMPLETE: 1,
    PH_FLOW_START: 2,
    PH_FLOW_STEP: 3,
    PH_FLOW_FINISH: 4,
    PH_INSTANT: 5,
    PH_COUNTER: 6,
}
"""Tie-break among records sharing a timestamp, so the output is byte-stable."""

SLICE_SPANS: Final[tuple[tuple[str, str], ...]] = (
    (EventType.MODEL_REQUEST, EventType.MODEL_RESPONSE_COMPLETED),
    (EventType.TOOL_EXECUTED, EventType.TOOL_RESULT),
)
"""The event pairs drawn as duration slices, as start and completion types."""

CATEGORY_BY_GROUP: Final[tuple[tuple[frozenset[str], str], ...]] = (
    (EventType.MODEL, "model"),
    (EventType.TOOL, "tool"),
    (EventType.POLICY, "policy"),
    (EventType.WORLD, "world"),
    (EventType.STATEMENT, "statement"),
    (EventType.ANNOTATION, "annotation"),
    (EventType.RUN_LIFECYCLE, "run"),
)
"""Category string attached to each record, for filtering in the viewer."""


def _decision_event_ids(events: Sequence[Event]) -> frozenset[int]:
    """Return one event id per authorization decision.

    ``policy.evaluated`` is emitted once per request and the outcome event that
    follows repeats it, so counting both would double every request. The
    outcome events are read only when a stream carries no ``policy.evaluated``
    at all.
    """
    evaluated = {
        event.event_id
        for event in events
        if event.event_type == EventType.POLICY_EVALUATED
    }
    if evaluated:
        return frozenset(evaluated)
    return frozenset(
        event.event_id
        for event in events
        if event.event_type in EventType.POLICY
        and event.event_type != EventType.POLICY_EVALUATED
    )


def _slice_pairs(events: Sequence[Event]) -> tuple[tuple[Event, Event | None], ...]:
    """Pair every slice start with its completion, by correlation id.

    A start whose completion never arrived is paired with ``None`` rather than
    dropped, so a killed run still shows the call that was in flight when it
    stopped.
    """
    pairs: list[tuple[Event, Event | None]] = []
    for start_type, end_type in SLICE_SPANS:
        ends = [event for event in events if event.event_type == end_type]
        used: set[int] = set()
        for start in (event for event in events if event.event_type == start_type):
            match = next(
                (
                    end
                    for end in ends
                    if end.event_id not in used
                    and end.event_id > start.event_id
                    and end.correlation_id == start.correlation_id
                ),
                None,
            )
            if match is not None:
                used.add(match.event_id)
            pairs.append((start, match))
    return tuple(sorted(pairs, key=lambda pair: pair[0].event_id))


def _slice_name(start: Event) -> str:
    """Return the label of a duration slice."""
    if start.event_type == EventType.MODEL_REQUEST:
        return "model inference"
    tool_name = start.payload.get("tool_name")
    return f"tool {tool_name}" if tool_name else start.event_type


def _category(event: Event) -> str:
    """Return the viewer category of an event."""
    for group, category in CATEGORY_BY_GROUP:
        if event.event_type in group:
            return category
    return "other"


def _event_args(event: Event) -> dict[str, Any]:
    """Return the arguments carried on a record: the payload plus its envelope.

    The payload is copied verbatim, so anything a reader can search for in the
    JSONL is searchable in the trace under the same key.
    """
    args: dict[str, Any] = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "actor_id": str(event.actor_id),
        "step_id": event.step_id,
    }
    if event.correlation_id:
        args["correlation_id"] = event.correlation_id
    for key, value in event.payload.items():
        args.setdefault(key, value)
    return args


def track_for_event(event: Event) -> str:
    """Return the lane an event belongs on, defaulting to agent decisions."""
    return TRACK_BY_EVENT_TYPE.get(event.event_type, TRACK_AGENT_DECISIONS)


def flow_id(correlation_id: str) -> str:
    """Return a stable flow id for a correlation id.

    The correlation id is used directly: it is already unique within a run and
    keeping it readable makes a trace searchable by the same key as the JSONL.
    """
    return correlation_id


def export_trace(
    events: Sequence[Event],
    *,
    run_id: RunId,
    process_name: str = "run",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full trace object for one run.

    The result has ``traceEvents``, ``displayTimeUnit``, and an ``otherData``
    block carrying the run id, the clock origin, and any caller-supplied
    metadata such as the outcome class.
    """
    exporter = TraceExporter(run_id=run_id, process_name=process_name)
    records = exporter.convert(events)
    other: dict[str, Any] = {
        "run_id": str(run_id),
        "clock_origin_ns": exporter.clock_origin_ns or 0,
        "event_count": len(events),
    }
    for key, value in sorted(dict(metadata or {}).items()):
        other[key] = value
    return {
        "traceEvents": [record.to_json_dict() for record in records],
        "displayTimeUnit": DISPLAY_TIME_UNIT,
        "otherData": other,
    }


def write_trace(
    events: Sequence[Event],
    path: str | Path,
    *,
    run_id: RunId,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a trace to disk as pretty-printed JSON and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    trace = export_trace(events, run_id=run_id, metadata=metadata)
    text = json.dumps(trace, indent=2, sort_keys=True, ensure_ascii=False)
    target.write_text(f"{text}\n", encoding="utf-8")
    return target


def trace_from_jsonl(
    events_path: str | Path,
    trace_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Convert a stored event file into a trace file.

    The run id is read from the events themselves, so a trace can be rebuilt
    from a run directory with no other input.
    """
    events = load_events(events_path)
    if not events:
        raise ValueError(f"{events_path} holds no events to export")
    return write_trace(events, trace_path, run_id=events[0].run_id, metadata=metadata)


def validate_trace(trace: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the ways a trace object violates the Trace Event conventions.

    Checks the ones that break the Perfetto UI quietly: a duration slice with
    no duration, a counter with a non-numeric value, a flow finish with no
    matching start, and records that are not ordered by timestamp.
    """
    violations: list[str] = []
    records = list(iter_trace_events(trace))
    if not records:
        return ("trace carries no records",)
    flow_starts: set[str] = set()
    flow_finishes: set[str] = set()
    previous_ts: float | None = None
    for index, record in enumerate(records):
        phase = record.get("ph")
        timestamp = record.get("ts")
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
            violations.append(f"record {index} has a non-numeric ts")
        elif previous_ts is not None and timestamp < previous_ts:
            violations.append(f"record {index} is out of timestamp order")
        else:
            previous_ts = float(timestamp)
        if phase == PH_COMPLETE:
            duration = record.get("dur")
            if not isinstance(duration, (int, float)) or isinstance(duration, bool):
                violations.append(f"record {index} is a slice with no duration")
            elif duration < 0:
                violations.append(f"record {index} is a slice with a negative duration")
        if phase == PH_COUNTER:
            for key, value in dict(record.get("args", {})).items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    violations.append(
                        f"record {index} counter series {key} is not numeric"
                    )
        if phase in (PH_FLOW_START, PH_FLOW_STEP, PH_FLOW_FINISH):
            identifier = record.get("id")
            if identifier is None:
                violations.append(f"record {index} is a flow record with no id")
            elif phase == PH_FLOW_START:
                flow_starts.add(str(identifier))
            elif phase == PH_FLOW_FINISH:
                flow_finishes.add(str(identifier))
    for identifier in sorted(flow_finishes - flow_starts):
        violations.append(f"flow {identifier} finishes without a start")
    for identifier in sorted(flow_starts - flow_finishes):
        violations.append(f"flow {identifier} starts without a finish")
    return tuple(violations)


def iter_trace_events(trace: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Yield the records of a trace object."""
    return tuple(trace.get("traceEvents", ()))
