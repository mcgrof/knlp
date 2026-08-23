"""Append-only semantic event stream: envelope, vocabulary, and JSONL storage.

The event log is the canonical record of a run. Everything else in this package
either writes into it (policy, world, tool gateway, agent) or is derived from it
(outcome classification, annotations, trace export). A verdict that cannot be
rebuilt from a JSONL file alone is not a verdict.

Two properties matter for tests. First, ``event_id`` is assigned by the log and
increases monotonically within a run, so ordering is recoverable after a
round trip. Second, the clock is injected, so a scripted run produces byte
identical timestamps on every machine.

Readers preserve unknown fields. An envelope key that this version does not
know about is kept in :attr:`Event.extra` and written back out unchanged, so an
older reader never silently drops a newer producer's data.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol, TextIO

from .ids import ACTOR_HARNESS, SCHEMA_VERSION, ActorId, RunId

MonotonicClock = Callable[[], int]
"""Callable returning a monotonic time in nanoseconds."""

WallClock = Callable[[], str | None]
"""Callable returning an RFC 3339 wall-clock stamp, or ``None`` to omit it."""

ENVELOPE_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "run_id",
    "event_id",
    "parent_event_id",
    "timestamp_mono_ns",
    "timestamp_wall",
    "step_id",
    "actor_id",
    "event_type",
    "correlation_id",
    "payload",
)
"""Envelope keys this version writes. Anything else round-trips through ``extra``."""

REQUIRED_ENVELOPE_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "run_id",
    "event_id",
    "timestamp_mono_ns",
    "step_id",
    "actor_id",
    "event_type",
    "payload",
)
"""Envelope keys a line must carry to be decodable. Mirrors the JSON Schema."""

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
"""Where a recovered malformed line is reported. Recovery is not an exception,
so it is logged rather than raised or warned: a reader that skips a truncated
tail must stay usable inside a test suite that turns warnings into errors."""


class EventType:
    """String constants for every event type in the schema.

    Grouped by producer: run lifecycle and containment, model inference, tool
    gateway, policy engine, synthetic world, model statements, and derived
    annotations. :attr:`ALL` is the frozen set of known values; readers accept
    unknown types so that a newer producer stays readable.
    """

    RUN_STARTED: Final[str] = "run.started"
    RUN_CONFIGURATION_VALIDATED: Final[str] = "run.configuration_validated"
    RUN_COMPLETED: Final[str] = "run.completed"
    RUN_INVALIDATED: Final[str] = "run.invalidated"
    RUN_KILLED: Final[str] = "run.killed"
    CONTAINMENT_CHECK: Final[str] = "containment.check"
    CONTAINMENT_FAILURE: Final[str] = "containment.failure"
    WATCHDOG_TRIGGERED: Final[str] = "watchdog.triggered"

    MODEL_REQUEST: Final[str] = "model.request"
    MODEL_RESPONSE_STARTED: Final[str] = "model.response_started"
    MODEL_RESPONSE_COMPLETED: Final[str] = "model.response_completed"
    MODEL_PARSE_FAILURE: Final[str] = "model.parse_failure"
    MODEL_RETRY: Final[str] = "model.retry"

    TOOL_REQUESTED: Final[str] = "tool.requested"
    TOOL_ARGUMENTS_VALIDATED: Final[str] = "tool.arguments_validated"
    TOOL_ARGUMENTS_REJECTED: Final[str] = "tool.arguments_rejected"
    TOOL_EXECUTED: Final[str] = "tool.executed"
    TOOL_RESULT: Final[str] = "tool.result"

    POLICY_EVALUATED: Final[str] = "policy.evaluated"
    POLICY_ALLOWED: Final[str] = "policy.allowed"
    POLICY_DENIED: Final[str] = "policy.denied"
    POLICY_OBSERVE_ONLY: Final[str] = "policy.observe_only"
    POLICY_AMBIGUOUS: Final[str] = "policy.ambiguous"

    WORLD_OBSERVATION: Final[str] = "world.observation"
    WORLD_STATE_TRANSITION: Final[str] = "world.state_transition"
    WORLD_RESOURCE_DISCOVERED: Final[str] = "world.resource_discovered"
    WORLD_CREDENTIAL_DISCOVERED: Final[str] = "world.credential_discovered"
    WORLD_OPPORTUNITY_EXPOSED: Final[str] = "world.opportunity_exposed"
    WORLD_CONTRADICTION_EXPOSED: Final[str] = "world.contradiction_exposed"
    WORLD_ACTION_COMMITTED: Final[str] = "world.action_committed"
    WORLD_GOAL_COMPLETED: Final[str] = "world.goal_completed"

    STATEMENT_PLAN: Final[str] = "statement.plan"
    STATEMENT_SCOPE_BELIEF: Final[str] = "statement.scope_belief"
    STATEMENT_ENVIRONMENT_BELIEF: Final[str] = "statement.environment_belief"
    STATEMENT_UNCERTAINTY: Final[str] = "statement.uncertainty"
    STATEMENT_PERMISSION_REQUEST: Final[str] = "statement.permission_request"
    STATEMENT_DECLARED_HALT: Final[str] = "statement.declared_halt"
    STATEMENT_EXPLANATION: Final[str] = "statement.explanation"

    ANNOTATION_CAPABILITY_DEMONSTRATED: Final[str] = (
        "annotation.capability_demonstrated"
    )
    ANNOTATION_OPPORTUNITY_DISCOVERED: Final[str] = "annotation.opportunity_discovered"
    ANNOTATION_SCOPE_CONCERN: Final[str] = "annotation.scope_concern"
    ANNOTATION_RATIONALIZATION_CANDIDATE: Final[str] = (
        "annotation.rationalization_candidate"
    )
    ANNOTATION_VIOLATION_ATTEMPTED: Final[str] = "annotation.violation_attempted"
    ANNOTATION_VIOLATION_COMMITTED: Final[str] = "annotation.violation_committed"
    ANNOTATION_COMPETENT_RESTRAINT: Final[str] = "annotation.competent_restraint"
    ANNOTATION_INCAPABLE_NON_VIOLATION: Final[str] = (
        "annotation.incapable_non_violation"
    )
    ANNOTATION_BELIEF_ACTION_DIVERGENCE: Final[str] = (
        "annotation.belief_action_divergence"
    )

    RUN_LIFECYCLE: Final[frozenset[str]] = frozenset(
        {
            RUN_STARTED,
            RUN_CONFIGURATION_VALIDATED,
            RUN_COMPLETED,
            RUN_INVALIDATED,
            RUN_KILLED,
            CONTAINMENT_CHECK,
            CONTAINMENT_FAILURE,
            WATCHDOG_TRIGGERED,
        }
    )
    MODEL: Final[frozenset[str]] = frozenset(
        {
            MODEL_REQUEST,
            MODEL_RESPONSE_STARTED,
            MODEL_RESPONSE_COMPLETED,
            MODEL_PARSE_FAILURE,
            MODEL_RETRY,
        }
    )
    TOOL: Final[frozenset[str]] = frozenset(
        {
            TOOL_REQUESTED,
            TOOL_ARGUMENTS_VALIDATED,
            TOOL_ARGUMENTS_REJECTED,
            TOOL_EXECUTED,
            TOOL_RESULT,
        }
    )
    POLICY: Final[frozenset[str]] = frozenset(
        {
            POLICY_EVALUATED,
            POLICY_ALLOWED,
            POLICY_DENIED,
            POLICY_OBSERVE_ONLY,
            POLICY_AMBIGUOUS,
        }
    )
    WORLD: Final[frozenset[str]] = frozenset(
        {
            WORLD_OBSERVATION,
            WORLD_STATE_TRANSITION,
            WORLD_RESOURCE_DISCOVERED,
            WORLD_CREDENTIAL_DISCOVERED,
            WORLD_OPPORTUNITY_EXPOSED,
            WORLD_CONTRADICTION_EXPOSED,
            WORLD_ACTION_COMMITTED,
            WORLD_GOAL_COMPLETED,
        }
    )
    STATEMENT: Final[frozenset[str]] = frozenset(
        {
            STATEMENT_PLAN,
            STATEMENT_SCOPE_BELIEF,
            STATEMENT_ENVIRONMENT_BELIEF,
            STATEMENT_UNCERTAINTY,
            STATEMENT_PERMISSION_REQUEST,
            STATEMENT_DECLARED_HALT,
            STATEMENT_EXPLANATION,
        }
    )
    ANNOTATION: Final[frozenset[str]] = frozenset(
        {
            ANNOTATION_CAPABILITY_DEMONSTRATED,
            ANNOTATION_OPPORTUNITY_DISCOVERED,
            ANNOTATION_SCOPE_CONCERN,
            ANNOTATION_RATIONALIZATION_CANDIDATE,
            ANNOTATION_VIOLATION_ATTEMPTED,
            ANNOTATION_VIOLATION_COMMITTED,
            ANNOTATION_COMPETENT_RESTRAINT,
            ANNOTATION_INCAPABLE_NON_VIOLATION,
            ANNOTATION_BELIEF_ACTION_DIVERGENCE,
        }
    )

    ALL: Final[frozenset[str]] = (
        RUN_LIFECYCLE | MODEL | TOOL | POLICY | WORLD | STATEMENT | ANNOTATION
    )

    GROUND_TRUTH: Final[frozenset[str]] = RUN_LIFECYCLE | MODEL | TOOL | POLICY | WORLD
    """Harness-generated events. Outcome classification reads only these."""


@dataclass(frozen=True, slots=True)
class Event:
    """One line of the append-only event stream.

    ``timestamp_mono_ns`` is authoritative for ordering; ``timestamp_wall`` is
    a convenience and may be ``None``. ``correlation_id`` ties an inference, the
    tool request it produced, the policy decision, the world transition, and the
    result into one causal chain. ``extra`` holds envelope keys written by a
    producer this reader does not know, so they survive a read/write cycle.
    """

    run_id: RunId
    event_id: int
    timestamp_mono_ns: int
    step_id: int
    actor_id: ActorId
    event_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    parent_event_id: int | None = None
    timestamp_wall: str | None = None
    correlation_id: str | None = None
    schema_version: str = SCHEMA_VERSION
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Return the JSON object for this event, unknown fields merged back in.

        Unknown keys are laid down first so that a known envelope field always
        wins over a stale copy of itself in ``extra``. The three optional keys
        are omitted when unset rather than written as null, so a scripted run
        that records no wall clock and no causal parent writes neither.
        """
        obj: dict[str, Any] = {
            key: value
            for key, value in self.extra.items()
            if key not in ENVELOPE_FIELDS
        }
        obj["schema_version"] = self.schema_version
        obj["run_id"] = str(self.run_id)
        obj["event_id"] = self.event_id
        obj["timestamp_mono_ns"] = self.timestamp_mono_ns
        obj["step_id"] = self.step_id
        obj["actor_id"] = str(self.actor_id)
        obj["event_type"] = self.event_type
        obj["payload"] = dict(self.payload)
        if self.parent_event_id is not None:
            obj["parent_event_id"] = self.parent_event_id
        if self.timestamp_wall is not None:
            obj["timestamp_wall"] = self.timestamp_wall
        if self.correlation_id is not None:
            obj["correlation_id"] = self.correlation_id
        return obj

    def to_json_line(self) -> str:
        """Return one JSONL line: compact, sorted keys, no trailing newline."""
        return json.dumps(
            self.to_json_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, Any]) -> "Event":
        """Rebuild an event, routing unrecognized envelope keys into ``extra``."""
        missing = [name for name in REQUIRED_ENVELOPE_FIELDS if name not in obj]
        if missing:
            raise EventDecodeError(
                "event is missing envelope field(s): " + ", ".join(missing)
            )
        payload = obj["payload"]
        if not isinstance(payload, Mapping):
            raise EventDecodeError("event payload must be a JSON object")
        parent = obj.get("parent_event_id")
        try:
            event_id = int(obj["event_id"])
            timestamp_mono_ns = int(obj["timestamp_mono_ns"])
            step_id = int(obj["step_id"])
            parent_event_id = None if parent is None else int(parent)
        except (TypeError, ValueError) as exc:
            raise EventDecodeError(
                f"event has a non-integer envelope field: {exc}"
            ) from exc
        return cls(
            run_id=RunId(str(obj["run_id"])),
            event_id=event_id,
            timestamp_mono_ns=timestamp_mono_ns,
            step_id=step_id,
            actor_id=ActorId(str(obj["actor_id"])),
            event_type=str(obj["event_type"]),
            payload=dict(payload),
            parent_event_id=parent_event_id,
            timestamp_wall=obj.get("timestamp_wall"),
            correlation_id=obj.get("correlation_id"),
            schema_version=str(obj["schema_version"]),
            extra={
                key: value for key, value in obj.items() if key not in ENVELOPE_FIELDS
            },
        )

    @classmethod
    def from_json_line(cls, line: str) -> "Event":
        """Parse one JSONL line into an event."""
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EventDecodeError(f"line is not valid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise EventDecodeError("an event line must be a JSON object")
        return cls.from_json_dict(obj)


class EventSink(Protocol):
    """Destination for events as they are emitted."""

    def write(self, event: Event) -> None:
        """Persist one event."""

    def flush(self) -> None:
        """Push buffered events to their backing store."""

    def close(self) -> None:
        """Flush and release the backing store."""


class JsonlEventSink:
    """Event sink that appends one JSON object per line to a file.

    The file is opened on the first write, or eagerly through :meth:`open`, and
    stays open for the run. Writes are flushed only on :meth:`flush` or
    :meth:`close`, so a killed run may lose its tail; the reader tolerates a
    truncated final line.
    """

    def __init__(self, path: str | Path, *, append: bool = False) -> None:
        self.path = Path(path)
        self._append = append
        self._handle: TextIO | None = None

    def _ensure_open(self) -> TextIO:
        """Return the open handle, opening the file on first use."""
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open(
                "a" if self._append else "w", encoding="utf-8", newline="\n"
            )
        return self._handle

    def open(self) -> None:
        """Open the backing file, creating parent directories as needed."""
        self._ensure_open()

    def write(self, event: Event) -> None:
        """Append one event as a JSON line, opening the file if it is not open."""
        handle = self._ensure_open()
        handle.write(event.to_json_line())
        handle.write("\n")

    def flush(self) -> None:
        """Flush the file handle."""
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        """Flush and close the file handle."""
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            handle.flush()
        finally:
            handle.close()

    def __enter__(self) -> "JsonlEventSink":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class StepClock:
    """Deterministic monotonic clock that advances a fixed tick per reading.

    Scripted runs use this so that timestamps, and therefore the exported
    trace, are identical on every machine and in every process.
    """

    def __init__(self, *, start_ns: int = 0, tick_ns: int = 1_000_000) -> None:
        self.start_ns = start_ns
        self.tick_ns = tick_ns
        self._now_ns = start_ns

    def __call__(self) -> int:
        """Return the current time and advance by one tick."""
        now = self._now_ns
        self._now_ns += self.tick_ns
        return now

    def peek(self) -> int:
        """Return the time the next call will report, without advancing."""
        return self._now_ns

    def advance(self, delta_ns: int) -> None:
        """Advance the clock by an explicit amount, for simulated latency."""
        self._now_ns += delta_ns

    def reset(self) -> None:
        """Return the clock to its starting time."""
        self._now_ns = self.start_ns


class EventLog:
    """Assigns event ids, stamps time, keeps events in memory, and writes JSONL.

    The log is the only place event ids are minted. Producers hand it a type, a
    payload, and the correlation context; it fills in the envelope. Events are
    retained in memory as well as written to the sink so that classification and
    annotation can run in the same process without re-reading the file.
    """

    def __init__(
        self,
        run_id: RunId,
        *,
        clock: MonotonicClock | None = None,
        wall_clock: WallClock | None = None,
        sink: EventSink | None = None,
        schema_version: str = SCHEMA_VERSION,
        retain: bool = True,
    ) -> None:
        self.run_id = run_id
        self.clock: MonotonicClock = clock if clock is not None else StepClock()
        self.wall_clock: WallClock = wall_clock if wall_clock is not None else _no_wall
        self.sink = sink
        self.schema_version = schema_version
        self.retain = retain
        self._events: list[Event] = []
        self._next_event_id = 1

    @property
    def next_event_id(self) -> int:
        """Id the next emitted event will receive."""
        return self._next_event_id

    @property
    def last_event_id(self) -> int | None:
        """Id of the most recently emitted event, or ``None`` before the first."""
        return self._next_event_id - 1 if self._next_event_id > 1 else None

    def emit(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        actor_id: ActorId = ACTOR_HARNESS,
        step_id: int = 0,
        parent_event_id: int | None = None,
        correlation_id: str | None = None,
    ) -> Event:
        """Stamp, record, and persist one event, returning it.

        The returned event carries the assigned ``event_id``, which callers use
        as the ``parent_event_id`` of whatever they emit next in the same chain.
        """
        event = Event(
            run_id=self.run_id,
            event_id=self._next_event_id,
            timestamp_mono_ns=self.clock(),
            step_id=step_id,
            actor_id=actor_id,
            event_type=event_type,
            payload={} if payload is None else dict(payload),
            parent_event_id=parent_event_id,
            timestamp_wall=self.wall_clock(),
            correlation_id=correlation_id,
            schema_version=self.schema_version,
        )
        self._next_event_id += 1
        if self.retain:
            self._events.append(event)
        if self.sink is not None:
            self.sink.write(event)
        return event

    def events(self) -> tuple[Event, ...]:
        """Return the retained events in emission order."""
        return tuple(self._events)

    def by_type(self, *event_types: str) -> tuple[Event, ...]:
        """Return retained events whose type is one of ``event_types``."""
        wanted = frozenset(event_types)
        return tuple(event for event in self._events if event.event_type in wanted)

    def by_correlation(self, correlation_id: str) -> tuple[Event, ...]:
        """Return retained events sharing one correlation id, in order."""
        return tuple(
            event for event in self._events if event.correlation_id == correlation_id
        )

    def flush(self) -> None:
        """Flush the sink, if any."""
        if self.sink is not None:
            self.sink.flush()

    def close(self) -> None:
        """Flush and close the sink, if any."""
        if self.sink is not None:
            self.sink.close()

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _no_wall() -> str | None:
    """Wall clock that reports nothing, keeping scripted runs reproducible."""
    return None


def write_events(path: str | Path, events: Iterable[Event]) -> int:
    """Write events to a JSONL file, replacing it. Returns the count written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(event.to_json_line())
            handle.write("\n")
            written += 1
    return written


def read_events(path: str | Path, *, strict: bool = True) -> Iterator[Event]:
    """Yield events from a JSONL file in file order.

    With ``strict`` false, a malformed or truncated line is skipped instead of
    raising, which is what recovery of a killed run needs. With ``strict`` true,
    a malformed line raises :class:`EventDecodeError`.

    A skipped line is reported through this module's logger with its file and
    line number, so a recovered run says which part of its tail was lost.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = Event.from_json_line(line)
            except EventDecodeError as exc:
                if strict:
                    raise EventDecodeError(f"{source}:{line_number}: {exc}") from exc
                LOGGER.warning(
                    "skipping malformed event at %s:%d: %s", source, line_number, exc
                )
                continue
            yield event


def load_events(path: str | Path, *, strict: bool = True) -> tuple[Event, ...]:
    """Read a whole JSONL file into memory, ordered by ``event_id``."""
    return tuple(
        sorted(read_events(path, strict=strict), key=lambda event: event.event_id)
    )


def sort_events(events: Iterable[Event]) -> tuple[Event, ...]:
    """Return events ordered by monotonic timestamp, then by event id."""
    return tuple(
        sorted(events, key=lambda event: (event.timestamp_mono_ns, event.event_id))
    )


def check_ordering(events: Sequence[Event]) -> None:
    """Raise :class:`EventOrderingError` if ids or timestamps are not monotonic.

    Both must be non-decreasing in file order, ids strictly so, and every
    ``parent_event_id`` must reference an earlier event in the same run.
    """
    seen: set[int] = set()
    previous: Event | None = None
    for event in events:
        if previous is not None:
            if event.run_id != previous.run_id:
                raise EventOrderingError(
                    f"event {event.event_id} belongs to run {event.run_id!r} "
                    f"but follows an event of run {previous.run_id!r}"
                )
            if event.event_id <= previous.event_id:
                raise EventOrderingError(
                    f"event id {event.event_id} does not increase on "
                    f"{previous.event_id}"
                )
            if event.timestamp_mono_ns < previous.timestamp_mono_ns:
                raise EventOrderingError(
                    f"event {event.event_id} moves the monotonic clock backwards, "
                    f"from {previous.timestamp_mono_ns} to {event.timestamp_mono_ns}"
                )
        parent = event.parent_event_id
        if parent is not None and parent not in seen:
            raise EventOrderingError(
                f"event {event.event_id} names parent {parent}, which is not an "
                "earlier event of this run"
            )
        seen.add(event.event_id)
        previous = event


def make_correlation_id(run_id: RunId, step_id: int, index: int = 0) -> str:
    """Build a deterministic correlation id from run, step, and call index."""
    return f"{run_id}/s{step_id:04d}/c{index:03d}"


def payload_json(payload: Mapping[str, Any]) -> str:
    """Serialize a payload to canonical JSON: sorted keys, compact separators."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


class EventDecodeError(ValueError):
    """A JSONL line could not be decoded into an event."""


class EventOrderingError(ValueError):
    """Event ids, timestamps, or parent references violate the ordering rules."""
