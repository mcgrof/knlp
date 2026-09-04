#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Build JSON and Perfetto views from a Chores status projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from chores_perfetto import parse_time, render_trace
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parent
SCHEMAS = ROOT / "schemas"
DEFAULT_PROFILE = ROOT / "data" / "public-profile.json"
DEFAULT_EVENTS = ROOT / "data" / "public-events.jsonl"
DEFAULT_OUTPUT = ROOT / "web" / "public"
DEFAULT_TRACE_URL = "/traces/latest.pftrace"


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from *path*."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def schema_registry() -> tuple[dict[str, dict[str, Any]], Registry]:
    """Load the local schemas and register their canonical identifiers."""
    documents = {path.name: read_json(path) for path in sorted(SCHEMAS.glob("*.json"))}
    registry = Registry()
    for document in documents.values():
        registry = registry.with_resource(
            document["$id"], Resource.from_contents(document)
        )
    return documents, registry


def validator(name: str) -> Draft202012Validator:
    """Return a format-checking validator for one local schema."""
    documents, registry = schema_registry()
    return Draft202012Validator(
        documents[name],
        registry=registry,
        format_checker=FormatChecker(),
    )


def validate_profile(profile: dict[str, Any]) -> None:
    """Validate a profile and reject duplicate workstream identifiers."""
    validator("profile.schema.json").validate(profile)
    identifiers = [str(item["id"]) for item in profile["workstreams"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("profile contains duplicate workstream ids")


def load_events(
    path: Path,
    *,
    surface: str,
    workstreams: set[str],
) -> list[dict[str, Any]]:
    """Load and validate an ordered event stream."""
    check = validator("event.schema.json")
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous_time = -1

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error.msg}") from error

            errors = sorted(check.iter_errors(event), key=lambda item: list(item.path))
            if errors:
                messages = "; ".join(error.message for error in errors)
                raise ValueError(f"{path}:{line_number}: {messages}")

            event_id = str(event["event_id"])
            if event_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate event id {event_id}")
            if event["surface"] != surface:
                raise ValueError(
                    f"{path}:{line_number}: {event_id} belongs to "
                    f"the {event['surface']} surface, not {surface}"
                )
            if event["workstream"] not in workstreams:
                raise ValueError(
                    f"{path}:{line_number}: {event_id} names unknown workstream "
                    f"{event['workstream']}"
                )

            timestamp = parse_time(str(event["occurred_at"]))
            if timestamp < previous_time:
                raise ValueError(f"{path}:{line_number}: timestamps are not ordered")
            previous_time = timestamp
            seen.add(event_id)
            events.append(event)

    if not events:
        raise ValueError(f"{path}: event stream is empty")
    return events


def derive_status(
    profile: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    trace_url: str,
    trace_sha256: str,
) -> dict[str, Any]:
    """Derive the status document from a profile and validated events."""
    latest = {str(item["id"]): None for item in profile["workstreams"]}
    for event in events:
        latest[str(event["workstream"])] = event

    workstreams = []
    for item in profile["workstreams"]:
        workstream_id = str(item["id"])
        event = latest[workstream_id]
        if event is None:
            raise ValueError(f"workstream {workstream_id} has no event")
        workstream = {
            "id": workstream_id,
            "label": item["label"],
            "state": event["state"],
            "coverage": event["coverage"],
            "summary": event["summary"],
            "updated_at": event["occurred_at"],
        }
        for field in (
            "progress_percent",
            "performed_by",
            "reviewed_by",
            "next_review_at",
        ):
            if field in event:
                workstream[field] = event[field]
        workstreams.append(workstream)

    return {
        "schema_version": 1,
        "project": profile["project"],
        "surface": profile["surface"],
        "state": profile["state"],
        "summary": profile["summary"],
        "claim_scope": profile["claim_scope"],
        "updated_at": events[-1]["occurred_at"],
        "refresh_seconds": profile["refresh_seconds"],
        "workstreams": workstreams,
        "events": list(reversed(events)),
        "trace": {
            "format": "perfetto-track-event-protobuf",
            "url": trace_url,
            "event_count": len(events),
            "sha256": trace_sha256,
        },
    }


def build(
    *,
    profile_path: Path = DEFAULT_PROFILE,
    events_path: Path = DEFAULT_EVENTS,
    output_dir: Path = DEFAULT_OUTPUT,
    trace_url: str = DEFAULT_TRACE_URL,
    allow_private: bool = False,
) -> dict[str, Any]:
    """Validate one projection and replace its generated artifacts."""
    profile = read_json(profile_path)
    validate_profile(profile)
    if profile["surface"] != "public":
        if not allow_private:
            raise ValueError("private projections require --allow-private")
        if output_dir.resolve() == DEFAULT_OUTPUT.resolve():
            raise ValueError("private projections require a separate output directory")

    workstreams = {str(item["id"]) for item in profile["workstreams"]}
    events = load_events(
        events_path,
        surface=str(profile["surface"]),
        workstreams=workstreams,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "traces" / "latest.pftrace"
    status_path = output_dir / "status.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        temporary_trace = temporary_dir / "latest.pftrace"
        temporary_status = temporary_dir / "status.json"

        render_trace(profile, events, temporary_trace)
        trace_sha256 = hashlib.sha256(temporary_trace.read_bytes()).hexdigest()
        status = derive_status(
            profile,
            events,
            trace_url=trace_url,
            trace_sha256=trace_sha256,
        )
        validator("status.schema.json").validate(status)
        temporary_status.write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        temporary_trace.replace(trace_path)
        temporary_status.replace(status_path)

    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Chores JSON and Perfetto status views."
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-url", default=DEFAULT_TRACE_URL)
    parser.add_argument(
        "--allow-private",
        action="store_true",
        help="permit a private profile; outputs remain the operator's responsibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = build(
        profile_path=args.profile,
        events_path=args.events,
        output_dir=args.output_dir,
        trace_url=args.trace_url,
        allow_private=args.allow_private,
    )
    print(
        f"built {args.output_dir / 'status.json'} and "
        f"{args.output_dir / 'traces/latest.pftrace'} from "
        f"{len(status['events'])} events"
    )


if __name__ == "__main__":
    main()
