"""Raw model text and private reasoning stay out of the semantic stream.

Redaction in this package is structural rather than a filtering pass. Assistant
text goes into a content-addressed store and the event payload carries only the
``sha256:<hex>`` digest, so promoting a run means copying the events and the
trace and not copying one directory. These tests assert that property against
the bytes: distinctive prose the model emitted must not appear in the event
file, and must not appear in the trace built from it.

Private reasoning is treated more strictly still. A backend that returns one is
recorded as a boolean saying a channel was present, and the text itself never
enters the stream or the store, so a run works identically when it is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import build_harness, complete, read_notes, tool_response
from scopetrace.events import EventType, load_events
from scopetrace.ids import PolicyMode, RunId, Variant
from scopetrace.manifest import digest_text
from scopetrace.trace_export import export_trace

SECRET_PROSE = "PLAN: pivot-to-the-ledger-and-copy-it"
SECRET_SUMMARY = "NOTE: dogs-bark-at-the-moon-marker"
SECRET_REASONING = "hidden-chain-of-thought-marker"
MARKERS = (
    "pivot-to-the-ledger-and-copy-it",
    "dogs-bark-at-the-moon-marker",
    "hidden-chain-of-thought-marker",
)


@pytest.fixture
def talkative_run(challenge, tmp_path: Path):
    """Run a trajectory whose model text carries distinctive, searchable markers."""
    events_path = tmp_path / "events.jsonl"
    harness = build_harness(
        challenge,
        Variant.TREATMENT,
        [
            read_notes(content=SECRET_PROSE),
            tool_response(
                "service.call",
                {
                    "resource_id": "service/direct-publish",
                    "operation": "publish",
                    "purpose_id": "complete-assigned-task",
                },
                content=SECRET_SUMMARY,
                reasoning=SECRET_REASONING,
            ),
            complete(f"{SECRET_SUMMARY} and done"),
        ],
        mode=PolicyMode.OBSERVE,
        run_id=RunId("redaction/run"),
        events_path=events_path,
        stop_on_goal=False,
    )
    harness.run()
    harness.close()
    yield harness, events_path


def test_the_run_actually_produced_the_prose(talkative_run) -> None:
    """The markers are really in the run's text store, so the check is not vacuous."""
    harness, _path = talkative_run
    stored = [harness.text_store.get(digest) for digest in harness.text_store.digests()]
    joined = "\n".join(stored)
    assert MARKERS[0] in joined
    assert MARKERS[1] in joined


@pytest.mark.parametrize("marker", MARKERS)
def test_no_raw_model_text_reaches_the_event_file(talkative_run, marker: str) -> None:
    """Distinctive prose is absent from the bytes of the canonical stream."""
    _harness, path = talkative_run
    assert marker not in path.read_text("utf-8")


@pytest.mark.parametrize("marker", MARKERS)
def test_no_raw_model_text_reaches_the_trace(talkative_run, marker: str) -> None:
    """A trace promoted without the text directory carries no prose either."""
    _harness, path = talkative_run
    events = load_events(path)
    trace = export_trace(events, run_id=events[0].run_id)
    assert marker not in json.dumps(trace)


def test_assistant_text_is_referenced_by_digest(talkative_run) -> None:
    """The event names the text by digest, and the digest resolves in the store."""
    harness, path = talkative_run
    completions = [
        event
        for event in load_events(path)
        if event.event_type == EventType.MODEL_RESPONSE_COMPLETED
    ]
    assert completions
    for event in completions:
        digest = event.payload["response_hash"]
        assert digest.startswith("sha256:")
        assert len(digest) == len("sha256:") + 64
        assert harness.text_store.get(digest) is not None
    stored = harness.text_store.get(completions[0].payload["response_hash"])
    assert stored == SECRET_PROSE
    assert digest_text(stored) == completions[0].payload["response_hash"]


def test_statement_events_carry_only_a_digest(talkative_run) -> None:
    """A statement records where its text is, never the text."""
    _harness, path = talkative_run
    statements = [
        event for event in load_events(path) if event.event_type in EventType.STATEMENT
    ]
    assert statements
    for event in statements:
        assert event.payload["text_digest"].startswith("sha256:")
        assert all(
            not isinstance(value, str) or not value.startswith("PLAN:")
            for value in event.payload.values()
        )


def test_reasoning_is_recorded_as_a_flag_and_nothing_else(talkative_run) -> None:
    """A private reasoning channel is noted as present and is never stored."""
    harness, path = talkative_run
    events = load_events(path)
    flags = [
        event.payload["reasoning_present"]
        for event in events
        if event.event_type == EventType.MODEL_RESPONSE_COMPLETED
    ]
    assert flags == [False, True, False]
    body = path.read_text("utf-8")
    assert "reasoning_content" not in body
    assert SECRET_REASONING not in body
    stored = "\n".join(
        harness.text_store.get(digest) for digest in harness.text_store.digests()
    )
    assert SECRET_REASONING not in stored


def test_a_run_without_reasoning_behaves_identically(challenge, tmp_path: Path) -> None:
    """Dropping the reasoning channel changes no event the harness writes."""

    def run(reasoning: str | None, name: str) -> bytes:
        path = tmp_path / name
        harness = build_harness(
            challenge,
            Variant.TREATMENT,
            [
                tool_response(
                    "service.call",
                    {
                        "resource_id": "service/review-publish",
                        "operation": "publish",
                        "purpose_id": "complete-assigned-task",
                    },
                    content="NOTE: publishing.",
                    reasoning=reasoning,
                ),
                complete(),
            ],
            run_id=RunId("redaction/reasoning"),
            events_path=path,
            stop_on_goal=False,
        )
        harness.run()
        harness.close()
        return path.read_bytes()

    with_reasoning = run(SECRET_REASONING, "with.jsonl")
    without_reasoning = run(None, "without.jsonl")
    differing = [
        (left, right)
        for left, right in zip(
            with_reasoning.decode().splitlines(),
            without_reasoning.decode().splitlines(),
        )
        if left != right
    ]
    assert len(differing) == 1
    assert '"reasoning_present":true' in differing[0][0]
    assert '"reasoning_present":false' in differing[0][1]


def test_tool_arguments_are_recorded_by_digest(talkative_run) -> None:
    """A tool request records the digest of its arguments, not their text."""
    _harness, path = talkative_run
    requests = [
        event
        for event in load_events(path)
        if event.event_type == EventType.TOOL_REQUESTED
    ]
    assert requests
    for event in requests:
        assert event.payload["arguments_hash"].startswith("sha256:")


def test_promoting_events_and_trace_leaves_the_text_behind(
    talkative_run, tmp_path: Path
) -> None:
    """The promotable artifacts are self-sufficient and carry no transcript."""
    harness, path = talkative_run
    events = load_events(path)
    promoted = tmp_path / "promoted"
    promoted.mkdir()
    (promoted / "events.jsonl").write_bytes(path.read_bytes())
    (promoted / "trace.json").write_text(
        json.dumps(export_trace(events, run_id=events[0].run_id)), encoding="utf-8"
    )
    body = "\n".join(entry.read_text("utf-8") for entry in sorted(promoted.iterdir()))
    for marker in MARKERS:
        assert marker not in body
    assert not list(promoted.glob("model_text"))
    assert harness.text_store.digests(), "the text still exists, it was just not copied"
