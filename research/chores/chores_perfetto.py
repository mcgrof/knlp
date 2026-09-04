#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Render Chores activity as a deterministic Perfetto TrackEvent trace."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SEQUENCE_ID = 1
PROCESS_ID = 1  # Synthetic grouping key; no host process is represented.

STATE_SCORE = {
    "blocked": 0,
    "held": 0,
    "planned": 1,
    "active": 2,
    "complete": 3,
}


def stable_id(*parts: object) -> int:
    """Return a deterministic positive 63-bit track identifier."""
    digest = hashlib.sha256("/".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def parse_time(value: str) -> int:
    """Parse an RFC 3339 timestamp into nanoseconds since the Unix epoch."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError(f"timestamp has no UTC offset: {value}")
    delta = parsed.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (
        delta.days * 86_400 + delta.seconds
    ) * 1_000_000_000 + delta.microseconds * 1_000


def _set_annotation(annotation: Any, value: Any) -> None:
    if isinstance(value, bool):
        annotation.bool_value = value
    elif isinstance(value, int):
        if value < 0:
            annotation.int_value = value
        else:
            annotation.uint_value = value
    elif isinstance(value, float):
        annotation.double_value = value
    else:
        annotation.string_value = str(value)


def render_trace(
    profile: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
    output: Path,
) -> None:
    """Write one status projection as native TrackEvent protobuf.

    The empty process track groups the projection in Perfetto. A data-bearing
    overview track controls the collapsed summary; workstream event and state
    tracks remain available below it when the group is expanded.
    """
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent
    from perfetto.trace_builder.proto_builder import StreamingTraceProtoBuilder

    rows = sorted(
        (dict(row) for row in events),
        key=lambda row: (parse_time(str(row["occurred_at"])), str(row["event_id"])),
    )
    if not rows:
        raise ValueError("event stream is empty")

    project = str(profile["project"])
    surface = str(profile["surface"])
    workstreams = [dict(item) for item in profile["workstreams"]]
    origin = parse_time(str(rows[0]["occurred_at"]))
    process = stable_id(project, surface, "process")
    overview = stable_id(project, surface, "overview")
    event_tracks = {
        str(item["id"]): stable_id(project, surface, "events", item["id"])
        for item in workstreams
    }
    state_tracks = {
        str(item["id"]): stable_id(project, surface, "state", item["id"])
        for item in workstreams
    }

    output.parent.mkdir(parents=True, exist_ok=True)

    def descriptor(
        builder: Any,
        uuid: int,
        *,
        name: str | None = None,
        parent: int | None = None,
        order: int | None = None,
        process_track: bool = False,
        counter_unit: str | None = None,
        explicit: bool = False,
    ) -> None:
        packet = builder.create_packet()
        track = packet.track_descriptor
        track.uuid = uuid
        if name is not None:
            track.name = name
        if parent is not None:
            track.parent_uuid = parent
        if order is not None:
            track.sibling_order_rank = order
        if process_track:
            track.process.pid = PROCESS_ID
            track.process.process_name = f"{project} / {surface}"
        if counter_unit is not None:
            track.counter.unit_name = counter_unit
        if explicit:
            track.child_ordering = track.EXPLICIT
        builder.write_packet(packet)

    def event_packet(
        builder: Any,
        timestamp: int,
        event_type: int,
        track: int,
        *,
        name: str | None = None,
        annotations: Mapping[str, Any] | None = None,
        counter: int | None = None,
    ) -> None:
        packet = builder.create_packet()
        packet.timestamp = timestamp
        packet.trusted_packet_sequence_id = SEQUENCE_ID
        track_event = packet.track_event
        track_event.type = event_type
        track_event.track_uuid = track
        if name is not None:
            track_event.name = name
        if counter is not None:
            track_event.counter_value = counter
        for key, value in (annotations or {}).items():
            item = track_event.debug_annotations.add()
            item.name = str(key)
            _set_annotation(item, value)
        builder.write_packet(packet)

    with output.open("wb") as target:
        builder = StreamingTraceProtoBuilder(target)
        descriptor(builder, process, process_track=True)
        descriptor(
            builder,
            overview,
            name="Project activity",
            parent=process,
            explicit=True,
        )
        for rank, item in enumerate(workstreams, 1):
            workstream_id = str(item["id"])
            label = str(item["label"])
            descriptor(
                builder,
                event_tracks[workstream_id],
                name=label,
                parent=overview,
                order=rank,
            )
            descriptor(
                builder,
                state_tracks[workstream_id],
                name=f"{label} state",
                parent=overview,
                order=100 + rank,
                counter_unit="state",
            )

        for row in rows:
            timestamp = parse_time(str(row["occurred_at"])) - origin
            workstream_id = str(row["workstream"])
            annotations = {
                "event_id": row["event_id"],
                "occurred_at": row["occurred_at"],
                "kind": row["kind"],
                "workstream": workstream_id,
                "state": row["state"],
                "coverage": row["coverage"],
                "summary": row["summary"],
                "evidence": row["evidence"],
                "surface": row["surface"],
            }
            if "source_url" in row:
                annotations["source_url"] = row["source_url"]
            for track in (overview, event_tracks[workstream_id]):
                event_packet(
                    builder,
                    timestamp,
                    TrackEvent.TYPE_INSTANT,
                    track,
                    name=str(row["title"]),
                    annotations=annotations,
                )
            event_packet(
                builder,
                timestamp,
                TrackEvent.TYPE_COUNTER,
                state_tracks[workstream_id],
                counter=STATE_SCORE[str(row["state"])],
            )
