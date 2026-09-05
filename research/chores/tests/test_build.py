# SPDX-License-Identifier: GPL-2.0-only
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import build
import pytest
from chores_perfetto import render_trace

OKR_PROFILE = build.ROOT / "examples" / "personal-okr-profile.json"
OKR_EVENTS = build.ROOT / "examples" / "personal-okr-events.jsonl"
MAINTAINER_PROFILE = build.ROOT / "examples" / "open-source-maintainer-profile.json"
MAINTAINER_EVENTS = build.ROOT / "examples" / "open-source-maintainer-events.jsonl"
KERNEL_PROFILE = build.ROOT / "examples" / "linux-kernel-review-profile.json"
KERNEL_EVENTS = build.ROOT / "examples" / "linux-kernel-review-events.jsonl"
PUBLIC_COPY = (
    build.ROOT / "README.md",
    build.ROOT / "EXTENSIONS.md",
    build.DEFAULT_PROFILE,
    build.DEFAULT_EVENTS,
    OKR_PROFILE,
    OKR_EVENTS,
    MAINTAINER_PROFILE,
    MAINTAINER_EVENTS,
    KERNEL_PROFILE,
    KERNEL_EVENTS,
    build.DEFAULT_OUTPUT / "index.html",
    build.DEFAULT_OUTPUT / "examples" / OKR_PROFILE.name,
    build.DEFAULT_OUTPUT / "examples" / OKR_EVENTS.name,
    build.DEFAULT_OUTPUT / "examples" / MAINTAINER_PROFILE.name,
    build.DEFAULT_OUTPUT / "examples" / MAINTAINER_EVENTS.name,
    build.DEFAULT_OUTPUT / "examples" / KERNEL_PROFILE.name,
    build.DEFAULT_OUTPUT / "examples" / KERNEL_EVENTS.name,
    build.ROOT.parents[1] / "docs" / "index.html",
)


def test_public_projection_builds_and_validates(tmp_path: Path) -> None:
    profile = build.read_json(build.DEFAULT_PROFILE)
    status = build.build(output_dir=tmp_path)

    build.validator("status.schema.json").validate(status)
    assert status["project"] == profile["project"]
    assert status["surface"] == "public"
    assert "security hygiene" in status["summary"].lower()
    assert [item["id"] for item in status["workstreams"]] == [
        item["id"] for item in profile["workstreams"]
    ]
    assert [item["id"] for item in status["workstreams"][:2]] == [
        "security-hygiene",
        "incident-reporting",
    ]
    project_event = next(
        event for event in status["events"] if event["kind"] == "project"
    )
    assert "immediately report" in project_event["summary"]
    assert status["trace"]["event_count"] == len(status["events"])
    assert (
        status["trace"]["sha256"]
        == hashlib.sha256((tmp_path / "traces/latest.pftrace").read_bytes()).hexdigest()
    )


def test_public_copy_uses_lowercase_knlp_project_name() -> None:
    for path in PUBLIC_COPY:
        assert "KNLP" not in path.read_text(encoding="utf-8"), path


def test_tracked_artifacts_are_current(tmp_path: Path) -> None:
    build.build(output_dir=tmp_path)
    assert (tmp_path / "status.json").read_bytes() == (
        build.DEFAULT_OUTPUT / "status.json"
    ).read_bytes()
    assert (tmp_path / "traces/latest.pftrace").read_bytes() == (
        build.DEFAULT_OUTPUT / "traces/latest.pftrace"
    ).read_bytes()


def test_trace_is_byte_deterministic(tmp_path: Path) -> None:
    profile = build.read_json(build.DEFAULT_PROFILE)
    workstreams = {str(item["id"]) for item in profile["workstreams"]}
    events = build.load_events(
        build.DEFAULT_EVENTS,
        surface="public",
        workstreams=workstreams,
    )
    first = tmp_path / "first.pftrace"
    second = tmp_path / "second.pftrace"
    render_trace(profile, events, first)
    render_trace(profile, events, second)
    assert first.read_bytes() == second.read_bytes()


def test_trace_processor_sees_project_topology(tmp_path: Path) -> None:
    from perfetto.trace_processor import TraceProcessor

    status = build.build(output_dir=tmp_path)
    trace = tmp_path / "traces/latest.pftrace"
    processor = TraceProcessor(trace=str(trace))
    try:
        event_slices = list(
            processor.query(
                "select count(*) n from slice where name in ("
                + ",".join(
                    "'" + event["title"].replace("'", "''") + "'"
                    for event in status["events"]
                )
                + ")"
            )
        )[0].n
        process_slices = list(
            processor.query(
                "select count(*) n from slice s join process_track p "
                "on s.track_id = p.id where p.name is null"
            )
        )[0].n
        tracks = {
            row.name
            for row in processor.query("select name from track where name is not null")
        }
    finally:
        processor.close()

    labels = {item["label"] for item in status["workstreams"]}
    assert event_slices == 2 * len(status["events"])
    assert process_slices == 0
    assert "Project activity" in tracks
    assert labels <= tracks
    assert {f"{label} state" for label in labels} <= tracks


def test_private_projection_requires_explicit_opt_in(tmp_path: Path) -> None:
    profile = build.read_json(build.DEFAULT_PROFILE)
    profile["surface"] = "private"
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    events = [
        json.loads(line)
        for line in build.DEFAULT_EVENTS.read_text(encoding="utf-8").splitlines()
    ]
    for event in events:
        event["surface"] = "private"
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="require --allow-private"):
        build.build(
            profile_path=profile_path,
            events_path=events_path,
            output_dir=tmp_path / "refused",
        )

    with pytest.raises(ValueError, match="separate output directory"):
        build.build(
            profile_path=profile_path,
            events_path=events_path,
            output_dir=build.DEFAULT_OUTPUT,
            allow_private=True,
        )

    status = build.build(
        profile_path=profile_path,
        events_path=events_path,
        output_dir=tmp_path / "private",
        allow_private=True,
    )
    assert status["surface"] == "private"


def test_personal_okr_example_builds_with_review_metadata(tmp_path: Path) -> None:
    from perfetto.trace_processor import TraceProcessor

    status = build.build(
        profile_path=OKR_PROFILE,
        events_path=OKR_EVENTS,
        output_dir=tmp_path,
    )

    build.validator("status.schema.json").validate(status)
    assert status["project"] == "Personal OKR example"
    assert status["surface"] == "public"
    assert len(status["workstreams"]) == 3
    checklist = next(
        item for item in status["workstreams"] if item["id"] == "o1-kr1-checklist"
    )
    assert checklist["progress_percent"] == 40
    assert checklist["performed_by"] == "agent-writer"
    assert checklist["reviewed_by"] == "owner"
    assert checklist["next_review_at"] == "2026-01-19T17:00:00Z"
    assert status["trace"]["event_count"] == len(status["events"])

    processor = TraceProcessor(trace=str(tmp_path / "traces/latest.pftrace"))
    try:
        progress = {
            row.int_value
            for row in processor.query(
                "select distinct a.int_value from slice s join args a "
                "on s.arg_set_id = a.arg_set_id "
                "where s.name = 'Release checklist receives its first review' "
                "and a.key = 'debug.progress_percent'"
            )
        }
        performers = {
            row.string_value
            for row in processor.query(
                "select distinct a.string_value from slice s join args a "
                "on s.arg_set_id = a.arg_set_id "
                "where s.name = 'Release checklist receives its first review' "
                "and a.key = 'debug.performed_by'"
            )
        }
    finally:
        processor.close()

    assert progress == {40}
    assert performers == {"agent-writer"}


def test_private_maintainer_example_preserves_authority(tmp_path: Path) -> None:
    from perfetto.trace_processor import TraceProcessor

    status = build.build(
        profile_path=MAINTAINER_PROFILE,
        events_path=MAINTAINER_EVENTS,
        output_dir=tmp_path,
        allow_private=True,
    )

    build.validator("status.schema.json").validate(status)
    assert status["project"] == "Open-source maintainer assistance example"
    assert status["surface"] == "private"
    assert len(status["workstreams"]) == 4
    pull_requests = next(
        item for item in status["workstreams"] if item["id"] == "pull-request-review"
    )
    assert pull_requests["performed_by"] == "agent-reviewer"
    assert pull_requests["reviewed_by"] == "maintainer"
    assert "do not submit" in pull_requests["authority_scope"]
    assert {event["kind"] for event in status["events"]} >= {
        "triage",
        "review",
        "handoff",
    }

    processor = TraceProcessor(trace=str(tmp_path / "traces/latest.pftrace"))
    try:
        authority = {
            row.string_value
            for row in processor.query(
                "select distinct a.string_value from slice s join args a "
                "on s.arg_set_id = a.arg_set_id "
                "where s.name = "
                "'Assigned pull request receives an evidence-linked review' "
                "and a.key = 'debug.authority_scope'"
            )
        }
    finally:
        processor.close()

    assert authority == {pull_requests["authority_scope"]}


def test_private_kernel_example_preserves_scope_and_priority(tmp_path: Path) -> None:
    from perfetto.trace_processor import TraceProcessor

    status = build.build(
        profile_path=KERNEL_PROFILE,
        events_path=KERNEL_EVENTS,
        output_dir=tmp_path,
        allow_private=True,
    )

    build.validator("status.schema.json").validate(status)
    assert status["project"] == "Linux kernel review assistance example"
    assert status["surface"] == "private"
    assert len(status["workstreams"]) == 5
    assert status["repository_scope"] == {
        "include_paths": ["block/", "drivers/nvme/", "include/linux/blk-mq.h"],
        "exclude_paths": ["drivers/nvme/target/"],
    }
    syzbot = next(
        item for item in status["workstreams"] if item["id"] == "syzbot-reports"
    )
    assert syzbot["priority"] == "critical"
    assert syzbot["matched_paths"] == ["block/blk-mq.c"]

    processor = TraceProcessor(trace=str(tmp_path / "traces/latest.pftrace"))
    try:
        matched_paths = {
            row.string_value
            for row in processor.query(
                "select distinct a.string_value from slice s join args a "
                "on s.arg_set_id = a.arg_set_id "
                "where s.name = "
                "'Reproducer-backed memory-safety report heads the briefing' "
                "and a.key = 'debug.matched_paths'"
            )
        }
        priorities = {
            row.string_value
            for row in processor.query(
                "select distinct a.string_value from slice s join args a "
                "on s.arg_set_id = a.arg_set_id "
                "where s.name = "
                "'Reproducer-backed memory-safety report heads the briefing' "
                "and a.key = 'debug.priority'"
            )
        }
    finally:
        processor.close()

    assert matched_paths == {"block/blk-mq.c"}
    assert priorities == {"critical"}


def test_repository_scope_matches_files_and_recursive_directories() -> None:
    scope = build.read_json(KERNEL_PROFILE)["repository_scope"]

    assert build.path_in_repository_scope("block/blk-mq.c", scope)
    assert build.path_in_repository_scope("drivers/nvme/host/core.c", scope)
    assert build.path_in_repository_scope("include/linux/blk-mq.h", scope)
    assert not build.path_in_repository_scope("include/linux/blkdev.h", scope)
    assert not build.path_in_repository_scope("drivers/nvme/target/core.c", scope)


def test_priority_requires_an_explanation(tmp_path: Path) -> None:
    event = json.loads(KERNEL_EVENTS.read_text(encoding="utf-8").splitlines()[5])
    del event["priority_reason"]

    errors = list(build.validator("event.schema.json").iter_errors(event))
    assert any("priority_reason" in error.message for error in errors)

    status = build.build(
        profile_path=KERNEL_PROFILE,
        events_path=KERNEL_EVENTS,
        output_dir=tmp_path,
        allow_private=True,
    )
    del status["workstreams"][0]["priority_reason"]
    errors = list(build.validator("status.schema.json").iter_errors(status))
    assert any("priority_reason" in error.message for error in errors)


def test_event_stream_rejects_paths_outside_repository_scope(tmp_path: Path) -> None:
    profile = build.read_json(KERNEL_PROFILE)
    event = json.loads(KERNEL_EVENTS.read_text(encoding="utf-8").splitlines()[0])
    event["matched_paths"] = ["net/core/dev.c"]
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside repository scope"):
        build.load_events(
            events_path,
            surface="private",
            workstreams={str(item["id"]) for item in profile["workstreams"]},
            repository_scope=profile["repository_scope"],
        )


def test_event_stream_rejects_duplicate_ids(tmp_path: Path) -> None:
    profile = build.read_json(build.DEFAULT_PROFILE)
    first = build.DEFAULT_EVENTS.read_text(encoding="utf-8").splitlines()[0]
    events_path = tmp_path / "duplicate.jsonl"
    events_path.write_text(first + "\n" + first + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate event id"):
        build.load_events(
            events_path,
            surface="public",
            workstreams={str(item["id"]) for item in profile["workstreams"]},
        )
