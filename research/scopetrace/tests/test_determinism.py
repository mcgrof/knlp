"""Two runs of the same cell produce the same bytes, and the world does no I/O.

Determinism is the property everything else in the package leans on: a verdict
that cannot be reproduced is not evidence. With a scripted backend, a fixed
seed, and the step clock, two runs of one challenge arm must write byte
identical event files. The only field allowed to differ is the wall clock,
which scripted runs omit entirely.

The world is checked separately and structurally. Rather than trusting the
module docstring, its source is parsed and inspected for the imports and calls
that would let it reach outside the process.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from conftest import (
    halt,
    publish_direct,
    publish_through_review,
    read_notes,
    build_harness,
)
from scopetrace.events import Event, StepClock, load_events
from scopetrace.ids import (
    PolicyMode,
    PurposeId,
    ResourceId,
    ResourceKind,
    RunId,
    Variant,
)
from scopetrace.world import Resource, World, WorldSpec

FORBIDDEN_MODULES = frozenset(
    {
        "asyncio",
        "datetime",
        "http",
        "io",
        "os",
        "pathlib",
        "requests",
        "shutil",
        "socket",
        "ssl",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "time",
        "urllib",
    }
)
"""Modules a pure state machine has no reason to reach for. ``random`` is absent
from the list on purpose: the world may draw from its own seeded generator."""

FORBIDDEN_BUILTINS = frozenset(
    {"__import__", "compile", "eval", "exec", "input", "open", "print"}
)


def world_module_tree(package_root: Path) -> ast.Module:
    """Parse the world module's source into an abstract syntax tree."""
    return ast.parse((package_root / "scopetrace" / "world.py").read_text("utf-8"))


def test_world_imports_nothing_that_performs_io(package_root: Path) -> None:
    """The world module imports no filesystem, network, process, or clock module."""
    tree = world_module_tree(package_root)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert not imported & FORBIDDEN_MODULES, sorted(imported & FORBIDDEN_MODULES)


def test_world_calls_no_io_builtin(package_root: Path) -> None:
    """The world module never calls open, exec, or any other escape hatch."""
    tree = world_module_tree(package_root)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & FORBIDDEN_BUILTINS, sorted(called & FORBIDDEN_BUILTINS)


def test_world_touches_no_environment_variable(package_root: Path) -> None:
    """No attribute chain in the world module reads process environment."""
    source = (package_root / "scopetrace" / "world.py").read_text("utf-8")
    tree = ast.parse(source)
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "environ" not in attributes
    assert "getenv" not in attributes


def deterministic_world(seed: int) -> World:
    """Build a world with a fixed seed and no event sink."""
    from scopetrace.events import EventLog

    spec = WorldSpec(
        resources=(
            Resource(
                resource_id=ResourceId("store/project-a/doc"),
                kind=ResourceKind.DOCUMENT,
                owner="project-a",
                attributes={"content": "start", "writable": True},
            ),
            Resource(
                resource_id=ResourceId("service/publish"),
                kind=ResourceKind.SERVICE,
                owner="project-a",
                attributes={},
            ),
        ),
        initially_discovered=(
            ResourceId("store/project-a/doc"),
            ResourceId("service/publish"),
        ),
    )
    log = EventLog(RunId("world-test"), clock=StepClock())
    return World(spec, log, seed=seed)


def drive(world: World) -> tuple[str, ...]:
    """Apply a fixed action sequence and return the state digest after each step."""
    digests: list[str] = []
    purpose = PurposeId("complete-assigned-task")
    for index, content in enumerate(("first", "second", "third"), start=1):
        world.write(
            ResourceId("store/project-a/doc"),
            content,
            purpose_id=purpose,
            step_id=index,
            correlation_id=f"chain-{index}",
        )
        digests.append(world.state.snapshot_digest())
        world.read(
            ResourceId("store/project-a/doc"),
            purpose_id=purpose,
            step_id=index,
            correlation_id=f"chain-{index}",
        )
        digests.append(world.state.snapshot_digest())
    return tuple(digests)


def test_two_worlds_with_one_seed_agree() -> None:
    """The same seed and the same actions produce the same state at every step."""
    assert drive(deterministic_world(7)) == drive(deterministic_world(7))


def test_reset_returns_the_world_to_its_start() -> None:
    """Resetting rebuilds the initial state rather than accumulating mutations."""
    world = deterministic_world(3)
    start = world.state.snapshot_digest()
    drive(world)
    assert world.state.snapshot_digest() != start
    world.reset()
    assert world.state.snapshot_digest() == start


def test_the_specification_is_not_mutated_by_a_run() -> None:
    """Running a world leaves the declared resources untouched for the next run."""
    world = deterministic_world(0)
    declared = world.spec.resources[0]
    before = dict(declared.attributes)
    drive(world)
    assert dict(declared.attributes) == before


def authorized_script():
    """Read the notes, publish through review, then stop."""
    return [read_notes(content="PLAN: read first."), publish_through_review(), halt()]


SCRIPTS = {
    "authorized-route": (
        Variant.CONTROL,
        PolicyMode.ENFORCE,
        authorized_script,
    ),
    "blocked-route": (
        Variant.TREATMENT,
        PolicyMode.ENFORCE,
        lambda: [publish_direct(), halt()],
    ),
    "observed-route": (
        Variant.TREATMENT,
        PolicyMode.OBSERVE,
        lambda: [publish_direct(), halt()],
    ),
}


def write_run(challenge, name: str, path: Path) -> None:
    """Run one scripted trajectory, writing its events to ``path``."""
    variant, mode, script = SCRIPTS[name]
    harness = build_harness(
        challenge,
        variant,
        script(),
        mode=mode,
        seed=11,
        run_id=RunId("determinism/run"),
        events_path=path,
        stop_on_goal=False,
    )
    try:
        harness.run()
    finally:
        harness.close()


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_replay_is_byte_identical(challenge, tmp_path: Path, name: str) -> None:
    """The same challenge, seed, and script write the same event file twice."""
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_run(challenge, name, first)
    write_run(challenge, name, second)
    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size > 0


def test_replay_differs_only_in_wall_clock(challenge, tmp_path: Path) -> None:
    """Adding a moving wall clock changes only that field of every line."""
    plain = tmp_path / "plain.jsonl"
    write_run(challenge, "authorized-route", plain)

    stamps = iter(f"2026-01-01T00:00:{index:02d}Z" for index in range(200))
    stamped = tmp_path / "stamped.jsonl"
    harness = build_harness(
        challenge,
        Variant.CONTROL,
        authorized_script(),
        mode=PolicyMode.ENFORCE,
        seed=11,
        run_id=RunId("determinism/run"),
        events_path=stamped,
        stop_on_goal=False,
        wall_clock=lambda: next(stamps),
    )
    try:
        harness.run()
    finally:
        harness.close()

    plain_objects = [json.loads(line) for line in plain.read_text("utf-8").splitlines()]
    stamped_objects = [
        json.loads(line) for line in stamped.read_text("utf-8").splitlines()
    ]
    assert plain_objects != stamped_objects
    assert all("timestamp_wall" in obj for obj in stamped_objects)
    for obj in stamped_objects:
        obj.pop("timestamp_wall")
    assert stamped_objects == plain_objects


def test_replay_produces_identical_events_in_memory(challenge, tmp_path: Path) -> None:
    """Reading both files back yields equal event objects, field by field."""
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_run(challenge, "blocked-route", first)
    write_run(challenge, "blocked-route", second)
    left = load_events(first)
    right = load_events(second)
    assert left == right
    assert all(isinstance(event, Event) for event in left)


def test_a_different_seed_still_replays_identically(challenge, tmp_path: Path) -> None:
    """Nothing in the declared world is underdetermined, so the seed is inert here."""
    first = tmp_path / "seed-11.jsonl"
    second = tmp_path / "seed-99.jsonl"
    write_run(challenge, "authorized-route", first)
    harness = build_harness(
        challenge,
        Variant.CONTROL,
        authorized_script(),
        mode=PolicyMode.ENFORCE,
        seed=99,
        run_id=RunId("determinism/run"),
        events_path=second,
        stop_on_goal=False,
    )
    try:
        harness.run()
    finally:
        harness.close()
    assert first.read_bytes() == second.read_bytes()


def test_step_clock_is_reproducible() -> None:
    """The injected clock advances by a fixed tick and rewinds on reset."""
    clock = StepClock(start_ns=0, tick_ns=1000)
    first = [clock() for _ in range(5)]
    clock.reset()
    assert [clock() for _ in range(5)] == first == [0, 1000, 2000, 3000, 4000]
