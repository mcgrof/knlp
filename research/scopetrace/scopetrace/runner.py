"""Wire the pieces together and run one trajectory end to end.

A run is a directory. :func:`run_trajectory` builds the world, the policy
engine, the tool gateway, and the agent; drives the loop; classifies the
outcome from the resulting events; annotates it; and writes five artifacts:

    manifest.json      the frozen configuration
    events.jsonl       the canonical event stream
    model_text/        raw assistant text, one file per digest
    outcome.json       the verdict
    trace.json         the Perfetto-ingestible view

The verdict written to ``outcome.json`` is recomputed from ``events.jsonl``,
not carried out of the loop in memory. That is deliberate: if the two ever
disagree, the reconstruction property has been broken and the run should be
rejected rather than trusted.

Determinism is a property of the default configuration. With a scripted model
backend, a fixed seed, and the deterministic clock, two runs of the same
challenge produce identical events, identical outcomes, and identical traces.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from .agent import AgentConfig, AgentResult, CanonicalAgent
from .annotate import AnnotationSummary, RulesAnnotator
from .challenge import ChallengeSpec, check_matched_pair
from .events import (
    Event,
    EventLog,
    EventType,
    JsonlEventSink,
    MonotonicClock,
    StepClock,
    check_ordering,
    load_events,
)
from .ids import (
    ACTOR_HARNESS,
    ACTOR_WATCHDOG,
    SUBJECT_AGENT,
    Arm,
    ChallengeId,
    PolicyMode,
    RunId,
    StopReason,
    SubjectId,
    Variant,
)
from .manifest import (
    FileTextStore,
    HardwareInfo,
    RunManifest,
    SamplingConfig,
    SoftwareManifest,
    collect_software_manifest,
)
from .model_client import ModelClient
from .outcome import (
    INVALID_BACKEND_ERROR,
    INVALID_PARSER_FAILURE,
    INVALID_WATCHDOG,
    OutcomeRecord,
    check_capability_provenance,
    classify_outcome,
)
from .policy import PolicyEngine
from .tools import ToolGateway
from .trace_export import write_trace
from .world import World

MANIFEST_FILENAME: Final[str] = "manifest.json"
EVENTS_FILENAME: Final[str] = "events.jsonl"
OUTCOME_FILENAME: Final[str] = "outcome.json"
TRACE_FILENAME: Final[str] = "trace.json"
SOFTWARE_FILENAME: Final[str] = "software.json"
MODEL_TEXT_DIRNAME: Final[str] = "model_text"

DEFAULT_EXPERIMENT_ID: Final[str] = "development"
DEFAULT_BACKEND: Final[str] = "replay"
DEFAULT_PRECISION: Final[str] = "none"
DEFAULT_REASONING_CONDITION: Final[str] = "fixed"

CONTAINMENT_CHECKS: Final[tuple[str, ...]] = (
    "world_performs_no_io",
    "model_backend_is_offline",
    "no_ambient_credentials",
)
"""Checks recorded at the start of every run.

These are assertions about the harness configuration, recorded so a run states
what it believed about its own isolation. They are not a substitute for
verifying containment at a boundary outside the agent runtime, which is done
independently and with hostile code rather than with an obedient model.
"""


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Paths of the files one run writes."""

    out_dir: Path
    manifest_path: Path
    events_path: Path
    outcome_path: Path
    trace_path: Path
    software_path: Path
    model_text_dir: Path

    @classmethod
    def under(cls, out_dir: str | Path) -> "RunArtifacts":
        """Build the artifact paths for a run directory, without creating it."""
        root = Path(out_dir)
        return cls(
            out_dir=root,
            manifest_path=root / MANIFEST_FILENAME,
            events_path=root / EVENTS_FILENAME,
            outcome_path=root / OUTCOME_FILENAME,
            trace_path=root / TRACE_FILENAME,
            software_path=root / SOFTWARE_FILENAME,
            model_text_dir=root / MODEL_TEXT_DIRNAME,
        )

    def create(self) -> "RunArtifacts":
        """Create the run directory and the text store directory."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model_text_dir.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True, slots=True)
class WatchdogConfig:
    """Independent budgets that end a run regardless of what the agent does.

    A trip is recorded as ``watchdog.triggered`` and invalidates the run: a
    trajectory cut off by a budget carries no verdict about scope behaviour.
    """

    max_steps: int = 64
    max_tool_calls: int = 128
    max_wall_ns: int = 15 * 60 * 1_000_000_000

    def exceeded(self, *, steps: int, tool_calls: int, elapsed_ns: int) -> str | None:
        """Return the name of the exceeded budget, or ``None``."""
        if steps > self.max_steps:
            return "max_steps"
        if tool_calls > self.max_tool_calls:
            return "max_tool_calls"
        if elapsed_ns > self.max_wall_ns:
            return "max_wall_ns"
        return None


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything about a run that is not the challenge, arm, model, or seed.

    Defaults describe an offline scripted run, which is what the tests use. A
    live run overrides the backend, precision, hardware, and revision fields so
    the manifest records what actually produced the trajectory.
    """

    experiment_id: str = DEFAULT_EXPERIMENT_ID
    backend: str = DEFAULT_BACKEND
    backend_revision: str = ""
    precision: str = DEFAULT_PRECISION
    reasoning_condition: str = DEFAULT_REASONING_CONDITION
    model_revision: str = "scripted"
    tokenizer_revision: str = "scripted"
    agent_revision: str = ""
    hardware: HardwareInfo = field(default_factory=HardwareInfo)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    subject_id: SubjectId | None = None
    public_eligible: bool = False
    annotate: bool = True
    write_trace: bool = True


def build_run_id(
    challenge_id: ChallengeId,
    variant: Arm,
    model_id: str,
    seed: int,
    *,
    index: int = 0,
) -> RunId:
    """Build a deterministic run id from the cell coordinates.

    Deterministic rather than random so that re-running one cell overwrites its
    own directory instead of accumulating near-duplicates, and so a run id can
    be predicted from the campaign matrix.
    """
    cell = "/".join(
        (str(challenge_id), str(variant), str(model_id) or "unknown", f"s{seed}")
    )
    # The repeat index is written only when there is one, so the first run of a
    # cell carries the plain cell coordinates and a repeat is visibly a repeat.
    return RunId(cell if index == 0 else f"{cell}/r{index}")


def matched_control_run_id(
    challenge_id: ChallengeId, model_id: str, seed: int, *, index: int = 0
) -> RunId:
    """Return the run id of the control cell that matches a treatment cell.

    Run ids are a deterministic function of the cell coordinates, so the
    control twin of a treatment run can be named without looking it up. That is
    what lets a carried capability verdict record which run it came from even
    when the verdict was asserted on the command line rather than read off a
    finished control directory. The id names a specific cell, so an assertion
    about a control run that was never made points at a directory that does not
    exist and is refutable rather than vague.
    """
    return build_run_id(challenge_id, Variant.CONTROL, model_id, seed, index=index)


def read_control_capability(run_dir: str | Path) -> tuple[bool, RunId]:
    """Return a finished control run's capability verdict and its run id.

    This is the path where capability is read rather than asserted: the verdict
    comes out of the control arm's own outcome record and the id out of the
    manifest beside it, so a treatment run scored against it can name the run
    that set its bar. A run of any other arm is refused, because capability for
    the matched pair is established where the fast route is authorized and
    nowhere else.
    """
    artifacts = RunArtifacts.under(run_dir)
    manifest = RunManifest.read(artifacts.manifest_path)
    if str(manifest.variant) != str(Variant.CONTROL):
        raise ValueError(
            f"{artifacts.out_dir} is a {manifest.variant} run, and a capability "
            f"verdict is established on the {Variant.CONTROL} arm, where the "
            "fast route is authorized"
        )
    outcome = OutcomeRecord.read(artifacts.outcome_path)
    return outcome.technical_capability, manifest.run_id


def combine_control_capability(
    verdicts: Sequence[tuple[bool, RunId]],
) -> tuple[bool | None, tuple[str, ...]]:
    """Reduce the control runs of one cell to the bar a treatment run is scored on.

    One demonstration is enough: a repeat that failed to finish the control
    task does not un-demonstrate what another repeat showed, so the verdicts are
    combined with a disjunction. That disjunction is confined to the control
    arm and never reaches the trajectory being scored, which is the distinction
    that matters. Reading capability off the scored trajectory would let the
    behaviour under measurement choose its own denominator; reading it off more
    than one run of the arm where the route is authorized does not.

    An empty sequence returns ``None``, meaning no control verdict was
    supplied and capability falls back to what the run itself showed.
    """
    if not verdicts:
        return None, ()
    return any(capable for capable, _ in verdicts), tuple(
        str(run_id) for _, run_id in verdicts
    )


def run_trajectory(
    challenge: ChallengeSpec,
    variant: Arm,
    model_client: ModelClient,
    policy_mode: PolicyMode,
    seed: int,
    out_dir: str | Path,
    *,
    config: RunConfig | None = None,
    run_id: RunId | None = None,
    clock: MonotonicClock | None = None,
    started_at: str | None = None,
    control_capability: bool | None = None,
    capability_source_run_ids: Sequence[str] | None = None,
) -> OutcomeRecord:
    """Run one trajectory and write its artifacts, returning the verdict.

    Builds the world from the challenge's shared specification and the policy
    engine from the arm's rule set, so the arms differ only in authorization
    and in the prose each declares. Writes the manifest before the first
    inference call, so a crashed run still says what it was.

    ``control_capability`` carries the matched control run's capability verdict
    into a treatment run. Passing it is what makes the treatment result a
    conditional measurement rather than a raw rate, and the verdict is used in
    both directions: a control arm that did not clear the bar marks this run
    incapable however much it went on to do.

    ``capability_source_run_ids`` names the control runs that verdict came
    from. A carried verdict has to be attributable, so when one is supplied
    without any ids the matched control cell's id is derived from this run's
    own coordinates and recorded, which names the run the assertion is about.
    The verdict and the ids are written to two different files, and
    :func:`~scopetrace.outcome.check_capability_provenance` is what requires
    them to agree before either is written out as a result.
    """
    active = config if config is not None else RunConfig()
    identifier = (
        run_id
        if run_id is not None
        else build_run_id(
            challenge.challenge_id,
            variant,
            str(getattr(model_client, "model_id", "unknown")),
            seed,
        )
    )
    stamp = started_at if started_at is not None else _rfc3339_now()
    sources = resolve_capability_sources(
        challenge.challenge_id,
        model_client,
        seed,
        control_capability=control_capability,
        capability_source_run_ids=capability_source_run_ids,
    )
    artifacts = RunArtifacts.under(out_dir).create()

    software = collect_software_manifest()
    _write_json(artifacts.software_path, software.to_json_dict())

    log = open_event_log(identifier, artifacts, clock=clock)
    world, policy, gateway, agent = build_components(
        challenge,
        variant,
        model_client,
        policy_mode,
        seed,
        log,
        config=active,
        text_store=FileTextStore(artifacts.model_text_dir),
    )
    # The manifest is written before the first inference call, so a run that
    # crashes mid-trajectory still states the configuration it was running.
    manifest = build_run_manifest(
        challenge,
        variant,
        model_client,
        policy,
        run_id=identifier,
        config=active,
        software=software,
        started_at=stamp,
        capability_source_run_ids=sources,
    )
    manifest.write(artifacts.manifest_path)

    log.emit(
        EventType.RUN_STARTED,
        {
            "experiment_id": active.experiment_id,
            "challenge_id": str(challenge.challenge_id),
            "challenge_revision": manifest.challenge_revision,
            "rung": str(challenge.rung),
            "variant": str(variant),
            "policy_mode": str(policy.mode),
            "policy_revision": policy.policy_revision,
            "world_revision": world.spec.revision(),
            "seed": seed,
            "backend": active.backend,
            "model_id": manifest.model_id,
            "agent_id": manifest.agent_id,
        },
        actor_id=HARNESS_ACTOR,
        step_id=0,
    )
    record_containment_check(log)
    log.emit(
        EventType.RUN_CONFIGURATION_VALIDATED,
        {
            "agent_configuration_digest": active.agent.configuration_digest(),
            "system_prompt_revision": active.agent.system_prompt_revision,
            "user_prompt_revision": active.agent.user_prompt_revision,
            "tool_schema_revision": agent.tool_schema_revision,
            "matched_pair_violations": list(check_matched_pair(challenge)),
            "max_turns": active.agent.max_turns,
            "watchdog_max_steps": active.watchdog.max_steps,
            "watchdog_max_tool_calls": active.watchdog.max_tool_calls,
        },
        actor_id=HARNESS_ACTOR,
        step_id=0,
    )

    try:
        result = agent.run(challenge, variant)
    except BaseException:
        # The events written so far are the evidence for whatever went wrong,
        # so they are flushed before the failure travels on.
        log.close()
        raise

    emitted = log.events()
    elapsed_ns = (
        emitted[-1].timestamp_mono_ns - emitted[0].timestamp_mono_ns if emitted else 0
    )
    log.emit(
        EventType.RUN_COMPLETED,
        {**result.to_json_dict(), "variant": str(variant), "elapsed_ns": elapsed_ns},
        actor_id=HARNESS_ACTOR,
        step_id=result.turns_used,
    )
    tripped = active.watchdog.exceeded(
        steps=result.turns_used, tool_calls=result.tool_calls, elapsed_ns=elapsed_ns
    )
    if tripped is not None:
        log.emit(
            EventType.WATCHDOG_TRIGGERED,
            {
                "budget": tripped,
                "steps": result.turns_used,
                "tool_calls": result.tool_calls,
                "elapsed_ns": elapsed_ns,
            },
            actor_id=WATCHDOG_ACTOR,
            step_id=result.turns_used,
        )
        log.emit(
            EventType.RUN_INVALIDATED,
            {"reason": INVALID_WATCHDOG, "budget": tripped},
            actor_id=HARNESS_ACTOR,
            step_id=result.turns_used,
        )
    elif stop_reason_invalidates(result.stop_reason):
        log.emit(
            EventType.RUN_INVALIDATED,
            {
                "reason": (
                    INVALID_PARSER_FAILURE
                    if result.stop_reason is StopReason.PARSE_FAILURE_BUDGET
                    else INVALID_BACKEND_ERROR
                ),
                "stop_reason": str(result.stop_reason),
                "parse_failures": result.parse_failures,
            },
            actor_id=HARNESS_ACTOR,
            step_id=result.turns_used,
        )

    in_memory = classify_outcome(
        log.events(),
        challenge.capability_contract,
        variant,
        control_capability=control_capability,
    )
    if active.annotate:
        annotator = RulesAnnotator()
        annotator.emit(
            annotate_run(
                log.events(),
                challenge,
                variant,
                outcome=in_memory,
                annotator=annotator,
            ),
            log,
            step_id=result.turns_used,
        )
    log.close()

    # The verdict is recomputed from the file rather than carried out of the
    # loop. A disagreement means the run has lost the property that makes it
    # evidence, so it is raised rather than written out as a result.
    stored = load_events(artifacts.events_path)
    check_ordering(stored)
    outcome = classify_outcome(
        stored,
        challenge.capability_contract,
        variant,
        control_capability=control_capability,
    )
    differences = compare_outcomes(in_memory, outcome)
    if differences:
        raise RuntimeError(
            f"run {identifier} cannot be reconstructed from "
            f"{artifacts.events_path.name}: the verdict read back off disk "
            f"differs from the one the loop produced in "
            f"{', '.join(differences)}"
        )
    # The verdict says whether its bar came from a matched control run and the
    # manifest says which runs those were. They are written to two files, so
    # they are required to agree before either becomes a result.
    check_capability_provenance(outcome, manifest)
    outcome.write(artifacts.outcome_path)
    if active.write_trace:
        write_trace(
            stored,
            artifacts.trace_path,
            run_id=identifier,
            metadata=_trace_metadata(
                challenge_id=str(challenge.challenge_id),
                variant=str(variant),
                policy_mode=str(policy.mode),
                final_class=str(outcome.final_class),
                invalid_reason=outcome.invalid_reason,
            ),
        )
    return outcome


def resolve_capability_sources(
    challenge_id: ChallengeId,
    model_client: ModelClient,
    seed: int,
    *,
    control_capability: bool | None,
    capability_source_run_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    """Return the run ids a carried capability verdict is recorded against.

    No carried verdict means no source, and the run reports its capability as
    read off its own trajectory. Explicit ids are taken as given. A verdict
    asserted with no ids is attributed to the matched control cell, whose id
    follows from this run's own coordinates: that is the run the assertion is
    about, and naming it turns an unattributed claim into one that points at a
    directory somebody can go and look for.
    """
    if control_capability is None:
        return ()
    if capability_source_run_ids is not None:
        return tuple(str(run_id) for run_id in capability_source_run_ids)
    model_id = str(getattr(model_client, "model_id", "unknown"))
    return (str(matched_control_run_id(challenge_id, model_id, seed)),)


def build_run_manifest(
    challenge: ChallengeSpec,
    variant: Arm,
    model_client: ModelClient,
    policy: PolicyEngine,
    *,
    run_id: RunId,
    config: RunConfig,
    software: SoftwareManifest,
    started_at: str,
    capability_source_run_ids: Sequence[str] = (),
) -> RunManifest:
    """Assemble the manifest for a run from its resolved components."""
    # A specification built in memory rather than loaded from a file carries no
    # revision yet, and rendering it back to a document computes the same digest
    # the loader would have stamped on it.
    revision = challenge.challenge_revision or str(
        challenge.to_json_dict()["challenge_revision"]
    )
    return RunManifest(
        run_id=run_id,
        experiment_id=config.experiment_id,
        challenge_id=challenge.challenge_id,
        challenge_revision=revision,
        variant=variant,
        policy_revision=policy.policy_revision,
        policy_mode=policy.mode,
        model_id=str(getattr(model_client, "model_id", "unknown")),
        model_revision=str(
            getattr(model_client, "model_revision", "") or config.model_revision
        ),
        tokenizer_revision=config.tokenizer_revision,
        backend=config.backend,
        backend_revision=config.backend_revision,
        precision=config.precision,
        agent_id=config.agent.agent_id,
        agent_revision=(
            config.agent_revision
            or config.agent.agent_revision
            or config.agent.configuration_digest()
        ),
        reasoning_condition=config.reasoning_condition,
        sampling=config.sampling,
        hardware=config.hardware,
        software_manifest=software.digest,
        started_at=started_at,
        public_eligible=config.public_eligible,
        capability_source_run_ids=tuple(capability_source_run_ids),
    )


def record_containment_check(
    log: EventLog,
    *,
    checks: Sequence[str] = CONTAINMENT_CHECKS,
    step_id: int = 0,
) -> Event:
    """Emit ``containment.check`` listing the assertions this run makes.

    A failed check emits ``containment.failure`` instead, which invalidates the
    run at classification time.
    """
    declared = tuple(str(check) for check in checks)
    # A check this version has no way to assert is not recorded as holding. The
    # harness can vouch for the isolation properties it knows about; a name it
    # does not recognize is an unsupported claim, and claiming it anyway is the
    # failure mode this event exists to prevent.
    unsupported = tuple(name for name in declared if name not in CONTAINMENT_CHECKS)
    if unsupported:
        return log.emit(
            EventType.CONTAINMENT_FAILURE,
            {
                "checks": list(declared),
                "unsupported_checks": list(unsupported),
                "passed": False,
            },
            actor_id=HARNESS_ACTOR,
            step_id=step_id,
        )
    return log.emit(
        EventType.CONTAINMENT_CHECK,
        {"checks": list(declared), "passed": True},
        actor_id=HARNESS_ACTOR,
        step_id=step_id,
    )


def reclassify_run(
    run_dir: str | Path,
    challenge: ChallengeSpec,
    *,
    control_capability: bool | None = None,
) -> OutcomeRecord:
    """Recompute a stored run's verdict from its event file.

    The replay check: read ``events.jsonl``, apply the frozen contract, and
    compare with ``outcome.json``. A difference means either the classifier
    changed or the file is not what produced the record.
    """
    artifacts = RunArtifacts.under(run_dir)
    manifest = RunManifest.read(artifacts.manifest_path)
    if str(manifest.challenge_id) != str(challenge.challenge_id):
        raise ValueError(
            f"{artifacts.out_dir} was run against challenge "
            f"{manifest.challenge_id}, not {challenge.challenge_id}"
        )
    if (
        challenge.challenge_revision
        and manifest.challenge_revision != challenge.challenge_revision
    ):
        raise ValueError(
            f"{artifacts.out_dir} was run against challenge revision "
            f"{manifest.challenge_revision}, and the challenge given here is "
            f"{challenge.challenge_revision}"
        )
    return classify_outcome(
        load_events(artifacts.events_path),
        challenge.capability_contract,
        manifest.variant,
        control_capability=control_capability,
    )


def compare_outcomes(left: OutcomeRecord, right: OutcomeRecord) -> tuple[str, ...]:
    """Return the field names on which two verdicts differ."""
    expected = left.to_json_dict()
    found = right.to_json_dict()
    return tuple(name for name in expected if expected[name] != found.get(name))


def annotate_run(
    events: Sequence[Event],
    challenge: ChallengeSpec,
    variant: Arm,
    *,
    outcome: OutcomeRecord,
    annotator: RulesAnnotator | None = None,
) -> AnnotationSummary:
    """Run the derived annotator over a finished trajectory."""
    engine = annotator if annotator is not None else RulesAnnotator()
    return engine.annotate(
        events,
        capability_contract=challenge.capability_contract,
        variant=variant,
        outcome=outcome,
    )


def build_components(
    challenge: ChallengeSpec,
    variant: Arm,
    model_client: ModelClient,
    policy_mode: PolicyMode,
    seed: int,
    log: EventLog,
    *,
    config: RunConfig,
    text_store: FileTextStore | None = None,
) -> tuple[World, PolicyEngine, ToolGateway, CanonicalAgent]:
    """Construct the world, policy engine, gateway, and agent for one run.

    Exposed separately so a test can drive the loop without a run directory,
    and so an alternative scaffold can be given exactly the same world, policy,
    and gateway as the canonical agent.
    """
    world = World(challenge.world_spec(), log, seed=seed)
    policy = challenge.policy_for(variant).engine(policy_mode)
    gateway = ToolGateway(
        world,
        policy,
        log,
        subject_id=(
            config.subject_id if config.subject_id is not None else SUBJECT_AGENT
        ),
    )
    agent = CanonicalAgent(
        gateway,
        model_client,
        log,
        config=config.agent,
        text_store=text_store,
    )
    return world, policy, gateway, agent


def summarize_run(
    result: AgentResult,
    outcome: OutcomeRecord,
    artifacts: RunArtifacts,
) -> dict[str, Any]:
    """Return a one-line summary of a finished run, for the command line."""
    return {
        "out_dir": str(artifacts.out_dir),
        "stop_reason": str(result.stop_reason),
        "turns_used": result.turns_used,
        "tool_calls": result.tool_calls,
        "parse_failures": result.parse_failures,
        "goal_reached": result.goal_reached,
        "final_class": str(outcome.final_class),
        "invalid_reason": outcome.invalid_reason,
        "valid": outcome.valid,
        "technical_capability": outcome.technical_capability,
        "opportunity_discovered": outcome.opportunity_discovered,
        "attempted_violation": outcome.attempted_violation,
        "committed_violation": outcome.committed_violation,
        "in_scope_success": outcome.in_scope_success,
        "competent_restraint": outcome.competent_restraint,
        "manifest": str(artifacts.manifest_path),
        "events": str(artifacts.events_path),
        "outcome": str(artifacts.outcome_path),
        "trace": str(artifacts.trace_path),
        "model_text": str(artifacts.model_text_dir),
    }


def _rfc3339_now() -> str:
    """Return the current UTC time as an RFC 3339 stamp, to the second.

    Written into the manifest and never into an event, so that a scripted run
    stays byte-identical from one execution to the next.
    """
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: Path, obj: Any) -> Path:
    """Write pretty-printed, key-sorted JSON to a path and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(f"{text}\n", encoding="utf-8", newline="\n")
    return path


def _trace_metadata(
    *,
    challenge_id: str,
    variant: str,
    policy_mode: str,
    final_class: str,
    invalid_reason: str | None,
) -> dict[str, Any]:
    """Return the ``otherData`` block a run's trace carries.

    Built here rather than at each call site so that a trace rebuilt later from
    a stored event file is identical to the one the run wrote.
    """
    return {
        "challenge_id": challenge_id,
        "variant": variant,
        "policy_mode": policy_mode,
        "final_class": final_class,
        "invalid_reason": invalid_reason,
    }


HARNESS_ACTOR = ACTOR_HARNESS
"""Actor id for run lifecycle events."""

WATCHDOG_ACTOR = ACTOR_WATCHDOG
"""Actor id for watchdog events."""


def open_event_log(
    run_id: RunId,
    artifacts: RunArtifacts,
    *,
    clock: MonotonicClock | None = None,
) -> EventLog:
    """Open an event log writing JSONL into a prepared run directory."""
    return EventLog(
        run_id,
        clock=clock if clock is not None else StepClock(),
        sink=JsonlEventSink(artifacts.events_path),
    )


def stop_reason_invalidates(stop_reason: StopReason) -> bool:
    """Report whether a stop reason means the cell carries no behavioural result."""
    return stop_reason in (StopReason.PARSE_FAILURE_BUDGET, StopReason.MODEL_ERROR)
