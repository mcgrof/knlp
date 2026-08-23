"""Reader compatibility, malformed-tail recovery, and ordering of the JSONL stream.

The event file is the canonical record, so the reader has to survive two things
the rest of the package cannot control: a producer that writes envelope keys
this version has never heard of, and a run that was killed mid-write. An
unknown key must come back out unchanged, and a truncated final line must be
recoverable rather than fatal.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from conftest import publish_through_review, read_notes
from scopetrace.events import (
    ENVELOPE_FIELDS,
    REQUIRED_ENVELOPE_FIELDS,
    Event,
    EventDecodeError,
    EventLog,
    EventOrderingError,
    JsonlEventSink,
    StepClock,
    check_ordering,
    load_events,
    make_correlation_id,
    read_events,
    sort_events,
    write_events,
)
from scopetrace.ids import ACTOR_HARNESS, RunId, Variant


def sample_event(**overrides) -> Event:
    """Build one event with sensible defaults for a reader test."""
    fields = {
        "run_id": RunId("run-1"),
        "event_id": 1,
        "timestamp_mono_ns": 1_000_000,
        "step_id": 2,
        "actor_id": ACTOR_HARNESS,
        "event_type": "run.started",
        "payload": {"note": "hello"},
    }
    fields.update(overrides)
    return Event(**fields)


def test_event_round_trips_through_json() -> None:
    """A serialized event decodes back into an equal event."""
    event = sample_event(
        parent_event_id=None,
        correlation_id="run-1/s0002/c000",
    )
    decoded = Event.from_json_line(event.to_json_line())
    assert decoded == event


def test_optional_envelope_keys_are_omitted_when_unset() -> None:
    """A scripted run writes no wall clock, no parent, and no correlation key."""
    obj = sample_event().to_json_dict()
    assert "timestamp_wall" not in obj
    assert "parent_event_id" not in obj
    assert "correlation_id" not in obj
    assert set(REQUIRED_ENVELOPE_FIELDS) <= set(obj)


def test_unknown_envelope_field_survives_a_round_trip() -> None:
    """An envelope key from a newer producer is preserved through read and write."""
    obj = sample_event().to_json_dict()
    obj["future_envelope_field"] = {"kind": "reserved", "count": 3}
    obj["another_future_key"] = "kept"
    decoded = Event.from_json_dict(obj)
    assert decoded.extra["future_envelope_field"] == {"kind": "reserved", "count": 3}
    assert decoded.extra["another_future_key"] == "kept"
    assert json.loads(decoded.to_json_line()) == obj


def test_unknown_payload_field_survives_a_round_trip() -> None:
    """Payload keys are opaque to the reader and come back unchanged."""
    event = sample_event(payload={"known": 1, "invented_later": [1, 2, {"a": "b"}]})
    decoded = Event.from_json_line(event.to_json_line())
    assert decoded.payload == event.payload


def test_unknown_event_type_is_readable() -> None:
    """An event type this version does not define still decodes."""
    decoded = Event.from_json_line(
        sample_event(event_type="future.thing").to_json_line()
    )
    assert decoded.event_type == "future.thing"


def test_known_envelope_field_wins_over_a_stale_copy_in_extra() -> None:
    """A stale duplicate in ``extra`` never overwrites the real envelope value."""
    event = sample_event(extra={"event_id": 999, "kept": True})
    obj = event.to_json_dict()
    assert obj["event_id"] == 1
    assert obj["kept"] is True


def test_file_round_trip_preserves_unknown_fields(tmp_path: Path) -> None:
    """A whole file of events with future keys reads and writes back unchanged."""
    path = tmp_path / "events.jsonl"
    originals = []
    for index in range(1, 4):
        obj = sample_event(
            event_id=index, timestamp_mono_ns=index * 1000
        ).to_json_dict()
        obj["future_envelope_field"] = index
        originals.append(obj)
    path.write_text(
        "".join(json.dumps(obj, sort_keys=True) + "\n" for obj in originals),
        encoding="utf-8",
    )
    events = load_events(path)
    rewritten = tmp_path / "rewritten.jsonl"
    assert write_events(rewritten, events) == 3
    reread = [json.loads(line) for line in rewritten.read_text("utf-8").splitlines()]
    assert reread == originals


def test_missing_required_field_raises() -> None:
    """A line without a required envelope key cannot be decoded."""
    obj = sample_event().to_json_dict()
    del obj["actor_id"]
    with pytest.raises(EventDecodeError) as caught:
        Event.from_json_dict(obj)
    assert "actor_id" in str(caught.value)


def test_non_object_payload_raises() -> None:
    """A payload that is not a JSON object is a decode error."""
    obj = sample_event().to_json_dict()
    obj["payload"] = ["not", "an", "object"]
    with pytest.raises(EventDecodeError):
        Event.from_json_dict(obj)


def test_non_object_line_raises() -> None:
    """A JSON array on an event line is a decode error."""
    with pytest.raises(EventDecodeError):
        Event.from_json_line("[1, 2, 3]")


def truncated_file(tmp_path: Path) -> tuple[Path, int]:
    """Write three good lines followed by a truncated fourth. Returns path, count."""
    path = tmp_path / "events.jsonl"
    lines = [
        sample_event(event_id=index, timestamp_mono_ns=index * 1000).to_json_line()
        for index in range(1, 4)
    ]
    good = "\n".join(lines) + "\n"
    partial = sample_event(event_id=4, timestamp_mono_ns=4000).to_json_line()
    path.write_text(good + partial[: len(partial) // 2], encoding="utf-8")
    return path, 3


def test_truncated_trailing_line_is_recovered(tmp_path: Path) -> None:
    """A killed run's half-written last line is skipped, not fatal."""
    path, expected = truncated_file(tmp_path)
    recovered = tuple(read_events(path, strict=False))
    assert len(recovered) == expected
    assert [event.event_id for event in recovered] == [1, 2, 3]


def test_truncated_trailing_line_is_reported(tmp_path: Path, caplog) -> None:
    """Recovery says which line was lost, through logging rather than a warning."""
    path, _ = truncated_file(tmp_path)
    with caplog.at_level(logging.WARNING, logger="scopetrace.events"):
        tuple(read_events(path, strict=False))
    assert any(
        "skipping malformed event" in record.getMessage() for record in caplog.records
    )
    assert any(":4:" in record.getMessage() for record in caplog.records)


def test_truncated_trailing_line_raises_in_strict_mode(tmp_path: Path) -> None:
    """Strict reading refuses a malformed line and names its position."""
    path, _ = truncated_file(tmp_path)
    with pytest.raises(EventDecodeError) as caught:
        tuple(read_events(path, strict=True))
    assert ":4:" in str(caught.value)


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    """Empty lines between records are not events and are not errors."""
    path = tmp_path / "events.jsonl"
    line = sample_event().to_json_line()
    path.write_text(f"\n{line}\n\n", encoding="utf-8")
    assert len(tuple(read_events(path, strict=True))) == 1


def test_malformed_middle_line_is_recovered(tmp_path: Path) -> None:
    """A corrupt line in the middle of a file does not stop the reader."""
    path = tmp_path / "events.jsonl"
    lines = [
        sample_event(event_id=1, timestamp_mono_ns=1000).to_json_line(),
        "{not json at all",
        sample_event(event_id=3, timestamp_mono_ns=3000).to_json_line(),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    recovered = tuple(read_events(path, strict=False))
    assert [event.event_id for event in recovered] == [1, 3]


def test_sink_appends_one_line_per_event(tmp_path: Path) -> None:
    """The sink writes exactly one JSON object per line."""
    path = tmp_path / "events.jsonl"
    log = EventLog(RunId("run-1"), clock=StepClock(), sink=JsonlEventSink(path))
    for index in range(5):
        log.emit("run.started", {"index": index})
    log.close()
    lines = path.read_text("utf-8").splitlines()
    assert len(lines) == 5
    assert [json.loads(line)["event_id"] for line in lines] == [1, 2, 3, 4, 5]


def test_event_ids_and_timestamps_are_monotonic(challenge, harness_factory) -> None:
    """A real run's stream passes the ordering check."""
    harness = harness_factory(
        challenge, Variant.CONTROL, [read_notes(), publish_through_review()]
    )
    harness.run()
    check_ordering(harness.events())


def test_check_ordering_rejects_a_repeated_id() -> None:
    """An event id that does not increase is an ordering violation."""
    events = [
        sample_event(event_id=1),
        sample_event(event_id=1, timestamp_mono_ns=2000),
    ]
    with pytest.raises(EventOrderingError):
        check_ordering(events)


def test_check_ordering_rejects_a_backwards_clock() -> None:
    """A monotonic timestamp that moves backwards is an ordering violation."""
    events = [
        sample_event(event_id=1, timestamp_mono_ns=5000),
        sample_event(event_id=2, timestamp_mono_ns=1000),
    ]
    with pytest.raises(EventOrderingError):
        check_ordering(events)


def test_check_ordering_rejects_a_dangling_parent() -> None:
    """A parent reference to an event that never appeared is a violation."""
    events = [sample_event(event_id=1), sample_event(event_id=2, parent_event_id=99)]
    with pytest.raises(EventOrderingError):
        check_ordering(events)


def test_check_ordering_rejects_a_mixed_run() -> None:
    """Two runs' events in one file are not one ordered stream."""
    events = [
        sample_event(event_id=1),
        sample_event(event_id=2, run_id=RunId("other"), timestamp_mono_ns=2000),
    ]
    with pytest.raises(EventOrderingError):
        check_ordering(events)


def test_sort_events_orders_by_time_then_id() -> None:
    """Sorting breaks a timestamp tie with the event id."""
    events = [
        sample_event(event_id=3, timestamp_mono_ns=1000),
        sample_event(event_id=1, timestamp_mono_ns=1000),
        sample_event(event_id=2, timestamp_mono_ns=500),
    ]
    assert [event.event_id for event in sort_events(events)] == [2, 1, 3]


def test_correlation_id_is_deterministic() -> None:
    """The correlation id is a pure function of run, step, and call index."""
    first = make_correlation_id(RunId("run-1"), 3, 2)
    assert first == make_correlation_id(RunId("run-1"), 3, 2)
    assert first != make_correlation_id(RunId("run-1"), 3, 3)
    assert first.startswith("run-1/")


def test_envelope_field_list_covers_the_required_ones() -> None:
    """Every required key is one this version writes."""
    assert set(REQUIRED_ENVELOPE_FIELDS) <= set(ENVELOPE_FIELDS)
