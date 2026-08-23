"""Command line: run a trajectory, replay a verdict, export a trace, check files.

Five subcommands, each a thin wrapper over the library so that anything the
command line can do is also reachable from a test without a subprocess:

    run                run one trajectory and write a run directory
    replay             recompute a stored run's verdict from its event file
    export-trace       rebuild the Perfetto view from a stored event file
    validate-challenge validate challenge files against the schema and the
                       matched-pair invariants
    list-challenges    list the challenges in a directory with their revisions

``run`` defaults to the offline scripted backend, which needs a script file and
no network. Pointing it at an OpenAI-compatible endpoint is an explicit choice
made with ``--backend openai --base-url``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from .agent import AgentConfig, AgentResult
from .challenge import (
    ChallengeSpec,
    ChallengeValidationError,
    challenge_schema,
    load_challenge,
    validate_challenge_document,
)
from .events import EventType, load_events
from .ids import (
    ARM_PERMISSIVE,
    Arm,
    PolicyMode,
    RunId,
    StopReason,
    TerminalSignal,
    Variant,
    parse_arm,
)
from .manifest import (
    REASONING_CONDITIONS,
    HardwareInfo,
    RunManifest,
    SamplingConfig,
    short_digest,
)
from .model_client import (
    ModelClient,
    ModelResponse,
    OpenAICompatibleClient,
    ParseFailure,
    ReplayModelClient,
    TokenUsage,
)
from .outcome import OutcomeRecord
from .runner import (
    DEFAULT_EXPERIMENT_ID,
    DEFAULT_PRECISION,
    DEFAULT_REASONING_CONDITION,
    RunArtifacts,
    RunConfig,
    _trace_metadata,
    combine_control_capability,
    compare_outcomes,
    read_control_capability,
    reclassify_run,
    run_trajectory,
    summarize_run,
)
from .trace_export import trace_from_jsonl

PROGRAM_NAME: Final[str] = "scopetrace"

BACKEND_REPLAY: Final[str] = "replay"
BACKEND_OPENAI: Final[str] = "openai"
BACKENDS: Final[tuple[str, ...]] = (BACKEND_REPLAY, BACKEND_OPENAI)

EXIT_OK: Final[int] = 0
EXIT_ERROR: Final[int] = 1
EXIT_INVALID: Final[int] = 2
"""Exit codes: success, a failed operation, and a file that failed validation
or a replay whose verdict disagreed with the stored one."""


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every subcommand."""
    # The demo directory ships beside the package, so the two commands that
    # read challenges work from a checkout without being told where to look.
    demo_dir = Path(__file__).resolve().parents[1] / "challenges" / "demo"
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=(
            "Run a tool-using agent against a deterministic synthetic world "
            "and record whether it requests actions its authorization policy "
            "forbids."
        ),
    )
    subcommands = parser.add_subparsers(dest="subcommand", required=True)

    run = subcommands.add_parser(
        "run",
        help="run one trajectory and write a run directory",
        description=(
            "Run one trajectory. The offline scripted backend is the default; "
            "an OpenAI-compatible endpoint is an explicit choice."
        ),
    )
    run.add_argument(
        "--challenge", required=True, help="path to the challenge JSON file"
    )
    run.add_argument(
        "--variant",
        type=parse_variant,
        default=Variant.CONTROL,
        help=(
            "arm to run: control or treatment for the matched pair, or "
            f"{ARM_PERMISSIVE} for the capability ceiling (default control)"
        ),
    )
    run.add_argument(
        "--policy-mode",
        type=parse_policy_mode,
        default=PolicyMode.ENFORCE,
        help=(
            "enforce blocks a forbidden action, observe records it and lets "
            "the world execute it (default enforce)"
        ),
    )
    run.add_argument("--seed", type=int, default=0, help="run seed (default 0)")
    run.add_argument(
        "--out-dir", required=True, help="directory to write the run artifacts into"
    )
    run.add_argument(
        "--backend",
        choices=BACKENDS,
        default=BACKEND_REPLAY,
        help=f"inference backend (default {BACKEND_REPLAY})",
    )
    run.add_argument(
        "--script",
        help="scripted response file, required by the replay backend",
    )
    run.add_argument("--base-url", help="base URL of an OpenAI-compatible endpoint")
    run.add_argument(
        "--api-key-env",
        default="SCOPETRACE_API_KEY",
        help=(
            "environment variable holding the endpoint API key, so that no key "
            "is passed on the command line (default SCOPETRACE_API_KEY)"
        ),
    )
    run.add_argument("--model", help="model id to record and to request")
    run.add_argument("--model-revision", help="opaque revision of the model")
    run.add_argument(
        "--experiment-id",
        default=DEFAULT_EXPERIMENT_ID,
        help=f"experiment label for the manifest (default {DEFAULT_EXPERIMENT_ID})",
    )
    run.add_argument("--run-id", help="explicit run id, overriding the derived one")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--max-output-tokens", type=int, default=4096)
    run.add_argument("--max-turns", type=int, default=32)
    run.add_argument(
        "--precision",
        default=DEFAULT_PRECISION,
        help=f"numeric precision to record (default {DEFAULT_PRECISION})",
    )
    run.add_argument(
        "--reasoning-condition",
        choices=sorted(REASONING_CONDITIONS),
        default=DEFAULT_REASONING_CONDITION,
        help=(
            "test-time reasoning setting to record; fixed means the model "
            "exposes no reasoning control"
        ),
    )
    run.add_argument(
        "--gpu", default="cpu", help="accelerator to record in the manifest"
    )
    run.add_argument(
        "--control-capability",
        dest="control_capability",
        action="store_true",
        default=None,
        help=(
            "the matched control run demonstrated capability, which is what "
            "makes a treatment result a conditional measurement"
        ),
    )
    run.add_argument(
        "--no-control-capability",
        dest="control_capability",
        action="store_false",
        help="the matched control run did not demonstrate capability",
    )
    run.add_argument(
        "--control-run",
        dest="control_runs",
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "read the capability verdict out of a finished control run "
            "instead of asserting it, and record that run as the source of "
            "this run's bar; repeatable for repeats of the control cell"
        ),
    )
    run.add_argument(
        "--public-eligible",
        action="store_true",
        help="mark the run eligible for public promotion",
    )
    run.add_argument(
        "--no-annotate", action="store_true", help="skip the derived annotator"
    )
    run.add_argument("--no-trace", action="store_true", help="skip writing trace.json")
    run.set_defaults(handler=cmd_run)

    replay = subcommands.add_parser(
        "replay",
        help="recompute a stored run's verdict from its event file",
        description=(
            "Recompute the verdict from events.jsonl and compare it with the "
            "outcome record stored beside it."
        ),
    )
    replay.add_argument("run_dir", help="run directory to replay")
    replay.add_argument("--challenge", help="challenge file this run was made against")
    replay.add_argument(
        "--challenges",
        default=str(demo_dir),
        help=(
            "directory to find the challenge in by id, when --challenge is not "
            "given (default: the shipped demo directory)"
        ),
    )
    replay.add_argument(
        "--control-capability",
        dest="control_capability",
        action="store_true",
        default=None,
        help="capability carried in from the matched control run",
    )
    replay.add_argument(
        "--no-control-capability",
        dest="control_capability",
        action="store_false",
        help="no capability carried in from the matched control run",
    )
    replay.set_defaults(handler=cmd_replay)

    export = subcommands.add_parser(
        "export-trace",
        help="rebuild the trace view from a stored event file",
        description="Rebuild trace.json from events.jsonl.",
    )
    export.add_argument("run_dir", help="run directory to export")
    export.add_argument(
        "--out", help="trace file to write (default: trace.json in the run directory)"
    )
    export.set_defaults(handler=cmd_export_trace)

    validate = subcommands.add_parser(
        "validate-challenge",
        help="validate challenge files against the schema and the pair invariants",
        description=(
            "Validate every challenge file given, reporting all violations "
            "found rather than the first."
        ),
    )
    validate.add_argument(
        "paths",
        nargs="*",
        default=[str(demo_dir)],
        help="challenge files or directories (default: the shipped demo directory)",
    )
    validate.set_defaults(handler=cmd_validate_challenge)

    listing = subcommands.add_parser(
        "list-challenges",
        help="list the challenges in a directory with their rung and revision",
        description="List challenge files with their tier and revision.",
    )
    listing.add_argument(
        "directory",
        nargs="?",
        default=str(demo_dir),
        help="directory to list (default: the shipped demo directory)",
    )
    listing.set_defaults(handler=cmd_list_challenges)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, dispatch to a subcommand, and return an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help(sys.stderr)
        return EXIT_ERROR
    try:
        return int(handler(args))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        # An expected failure is a missing file, an unusable challenge, or a
        # replay that could not be rebuilt. None of those are defects in the
        # harness, so they are reported as a message rather than a traceback.
        print(f"{PROGRAM_NAME}: {error}", file=sys.stderr)
        return EXIT_ERROR


def cmd_run(args: argparse.Namespace) -> int:
    """Run one trajectory and write its run directory.

    Prints the run id, the terminal class, and the artifact paths. A run that
    ends invalid still writes its directory: an invalid cell is a result about
    the backend and needs its evidence kept.
    """
    challenge = load_challenge(Path(args.challenge))
    model_client = load_model_client(args)
    backend = str(getattr(args, "backend", BACKEND_REPLAY))
    max_turns = int(getattr(args, "max_turns", 32))
    config = RunConfig(
        experiment_id=str(getattr(args, "experiment_id", DEFAULT_EXPERIMENT_ID)),
        backend=backend,
        precision=str(getattr(args, "precision", DEFAULT_PRECISION)),
        reasoning_condition=str(
            getattr(args, "reasoning_condition", DEFAULT_REASONING_CONDITION)
        ),
        model_revision=str(getattr(args, "model_revision", None) or "scripted"),
        hardware=HardwareInfo(gpu=str(getattr(args, "gpu", "cpu"))),
        sampling=SamplingConfig(
            temperature=float(getattr(args, "temperature", 0.0)),
            top_p=float(getattr(args, "top_p", 1.0)),
            seed=int(getattr(args, "seed", 0)),
            max_output_tokens=int(getattr(args, "max_output_tokens", 4096)),
            max_turns=max_turns,
        ),
        agent=AgentConfig(max_turns=max_turns),
        public_eligible=bool(getattr(args, "public_eligible", False)),
        annotate=not bool(getattr(args, "no_annotate", False)),
        write_trace=not bool(getattr(args, "no_trace", False)),
    )
    declared_run_id = getattr(args, "run_id", None)
    capability, sources = resolve_control_capability(args)
    outcome = run_trajectory(
        challenge,
        args.variant,
        model_client,
        args.policy_mode,
        int(getattr(args, "seed", 0)),
        Path(args.out_dir),
        config=config,
        run_id=RunId(str(declared_run_id)) if declared_run_id else None,
        # A scripted run takes the deterministic clock so that two runs of one
        # cell are byte-identical; a live run records the time it really took.
        clock=None if backend == BACKEND_REPLAY else time.monotonic_ns,
        control_capability=capability,
        capability_source_run_ids=sources,
    )
    artifacts = RunArtifacts.under(Path(args.out_dir))
    manifest = RunManifest.read(artifacts.manifest_path)
    summary = summarize_run(_agent_result_from_events(artifacts), outcome, artifacts)
    print(f"run_id           {manifest.run_id}")
    print(f"challenge        {manifest.challenge_id} ({manifest.variant})")
    print(f"final_class      {summary['final_class']}")
    if summary["invalid_reason"] is not None:
        print(f"invalid_reason   {summary['invalid_reason']}")
    print(
        "trajectory       "
        f"stop={summary['stop_reason']} turns={summary['turns_used']} "
        f"tool_calls={summary['tool_calls']}"
    )
    print(
        "findings         "
        f"capability={summary['technical_capability']} "
        f"attempted={summary['attempted_violation']} "
        f"committed={summary['committed_violation']}"
    )
    for label in ("manifest", "events", "outcome", "trace", "model_text"):
        path = Path(str(summary[label]))
        if path.exists():
            print(f"{label:<16} {path}")
    return EXIT_OK


def resolve_control_capability(
    args: argparse.Namespace,
) -> tuple[bool | None, tuple[str, ...] | None]:
    """Return the capability verdict to carry in, and the runs it came from.

    ``--control-run`` reads the verdict out of a finished control directory,
    which is the path where capability is measured rather than asserted, and it
    wins over the flag: a directory that says the control arm did not clear the
    bar is a result, and a flag saying otherwise is a claim about the same
    thing with nothing behind it. With no directory given, the flag stands
    alone and the runner attributes it to the matched control cell.
    """
    directories = [str(entry) for entry in getattr(args, "control_runs", []) or ()]
    if not directories:
        return getattr(args, "control_capability", None), None
    verdicts = [read_control_capability(Path(entry)) for entry in directories]
    return combine_control_capability(verdicts)


def cmd_replay(args: argparse.Namespace) -> int:
    """Recompute a stored run's verdict and compare it with the stored one.

    Returns the invalid exit code when the two disagree, listing the fields
    that differ. This is the check that the verdict is a property of the event
    file rather than of the process that wrote it.
    """
    artifacts = RunArtifacts.under(Path(args.run_dir))
    manifest = RunManifest.read(artifacts.manifest_path)
    challenge = _resolve_challenge(args, manifest)
    stored = OutcomeRecord.read(artifacts.outcome_path)
    rebuilt = reclassify_run(
        artifacts.out_dir,
        challenge,
        control_capability=getattr(args, "control_capability", None),
    )
    differences = compare_outcomes(stored, rebuilt)
    print(f"run_id           {manifest.run_id}")
    print(f"challenge        {manifest.challenge_id} ({manifest.variant})")
    print(f"stored           {stored.final_class}")
    print(f"recomputed       {rebuilt.final_class}")
    if not differences:
        print("verdict rebuilt from the event file with no differences")
        return EXIT_OK
    stored_fields = stored.to_json_dict()
    rebuilt_fields = rebuilt.to_json_dict()
    print(
        f"{PROGRAM_NAME}: the verdict recomputed from "
        f"{artifacts.events_path} differs from the stored one",
        file=sys.stderr,
    )
    for name in differences:
        print(
            f"  {name}: stored {stored_fields[name]!r}, "
            f"recomputed {rebuilt_fields[name]!r}",
            file=sys.stderr,
        )
    return EXIT_INVALID


def cmd_export_trace(args: argparse.Namespace) -> int:
    """Rebuild the trace view from a stored event file."""
    artifacts = RunArtifacts.under(Path(args.run_dir))
    declared = getattr(args, "out", None)
    target = Path(declared) if declared else artifacts.trace_path
    # The metadata is rebuilt from the manifest and the outcome record so that
    # a trace exported here matches the one the run wrote.
    metadata: Mapping[str, Any] | None = None
    if artifacts.manifest_path.exists() and artifacts.outcome_path.exists():
        manifest = RunManifest.read(artifacts.manifest_path)
        outcome = OutcomeRecord.read(artifacts.outcome_path)
        metadata = _trace_metadata(
            challenge_id=str(manifest.challenge_id),
            variant=str(manifest.variant),
            policy_mode=str(manifest.policy_mode),
            final_class=str(outcome.final_class),
            invalid_reason=outcome.invalid_reason,
        )
    written = trace_from_jsonl(artifacts.events_path, target, metadata=metadata)
    print(f"wrote {written}")
    return EXIT_OK


def cmd_validate_challenge(args: argparse.Namespace) -> int:
    """Validate challenge files and report every violation found.

    Checks the schema, then the matched-pair invariants: both arms declared,
    the control arm authorizing the fast route, the treatment arm forbidding it
    and authorizing the slow one, and a capability contract whose markers this
    challenge can actually produce.
    """
    files = _challenge_files(getattr(args, "paths", ()) or ())
    if not files:
        print("no challenge files found")
        return EXIT_OK
    schema = challenge_schema()
    rejected = 0
    for path in files:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as error:
            rejected += 1
            print(f"{path}: not valid JSON: {error}", file=sys.stderr)
            continue
        if not isinstance(document, dict):
            rejected += 1
            print(f"{path}: does not hold a challenge object", file=sys.stderr)
            continue
        try:
            validate_challenge_document(document, schema=schema, path=path)
        except ChallengeValidationError as error:
            rejected += 1
            print(f"{path}: {len(error.violations)} violation(s)", file=sys.stderr)
            for violation in error.violations:
                print(f"  {violation}", file=sys.stderr)
            continue
        print(f"{path}: valid")
    checked = len(files)
    if rejected:
        print(
            f"{rejected} of {checked} challenge file(s) failed validation",
            file=sys.stderr,
        )
        return EXIT_INVALID
    print(f"{checked} challenge file(s) validated")
    return EXIT_OK


def cmd_list_challenges(args: argparse.Namespace) -> int:
    """List the challenges in a directory with their rung and revision."""
    directory = Path(str(getattr(args, "directory", "challenges/demo")))
    if not directory.is_dir():
        raise OSError(f"{directory} is not a directory")
    rows: list[tuple[str, str, str, str]] = []
    unreadable = 0
    for path in sorted(directory.glob("*.json")):
        try:
            # Listing is not validation: a file is reported with what it says
            # about itself, and validate-challenge is where it is judged.
            spec = load_challenge(path, validate=False)
        except (OSError, ValueError, KeyError) as error:
            unreadable += 1
            print(f"{path}: could not be read as a challenge: {error}", file=sys.stderr)
            continue
        rows.append(
            (
                str(spec.challenge_id),
                str(spec.rung),
                short_digest(spec.challenge_revision),
                spec.title,
            )
        )
    if not rows:
        print(f"no challenges in {directory}")
        return EXIT_OK
    headers = ("challenge_id", "rung", "revision", "title")
    widths = [
        max(len(header), *(len(row[column]) for row in rows))
        for column, header in enumerate(headers)
    ]
    print(
        "  ".join(header.ljust(widths[column]) for column, header in enumerate(headers))
    )
    for row in rows:
        print(
            "  ".join(value.ljust(widths[column]) for column, value in enumerate(row))
        )
    if unreadable:
        print(
            f"{unreadable} file(s) in {directory} are not challenges", file=sys.stderr
        )
    return EXIT_OK


def load_model_client(args: argparse.Namespace) -> ModelClient:
    """Build the model client a run was asked for.

    The replay backend reads a script file of recorded responses. The
    OpenAI-compatible backend is constructed from the endpoint arguments and
    contacts nothing until the run starts.
    """
    backend = str(getattr(args, "backend", BACKEND_REPLAY))
    if backend == BACKEND_REPLAY:
        script = getattr(args, "script", None)
        if not script:
            raise ValueError(
                f"the {BACKEND_REPLAY} backend needs --script, a file of "
                "recorded model responses"
            )
        return ReplayModelClient(
            load_replay_script(script),
            model_id=str(getattr(args, "model", None) or BACKEND_REPLAY),
            model_revision=str(getattr(args, "model_revision", None) or "scripted"),
        )
    if backend == BACKEND_OPENAI:
        base_url = getattr(args, "base_url", None)
        model_id = getattr(args, "model", None)
        if not base_url:
            raise ValueError(f"the {BACKEND_OPENAI} backend needs --base-url")
        if not model_id:
            raise ValueError(f"the {BACKEND_OPENAI} backend needs --model")
        key_variable = str(getattr(args, "api_key_env", "") or "")
        return OpenAICompatibleClient(
            str(base_url),
            str(model_id),
            api_key=os.environ.get(key_variable) if key_variable else None,
            sampling=SamplingConfig(
                temperature=float(getattr(args, "temperature", 0.0)),
                top_p=float(getattr(args, "top_p", 1.0)),
                seed=int(getattr(args, "seed", 0)),
                max_output_tokens=int(getattr(args, "max_output_tokens", 4096)),
                max_turns=int(getattr(args, "max_turns", 32)),
            ),
            model_revision=str(getattr(args, "model_revision", None) or "unknown"),
        )
    raise ValueError(f"unknown backend: {backend}")


def load_replay_script(path: str | Path) -> tuple[ModelResponse | ParseFailure, ...]:
    """Read a scripted response file for the offline backend.

    Each entry is either a response, with optional content and tool calls, or a
    parse failure with a reason code, so a script can exercise the retry path
    as well as the ordinary one.
    """
    source = Path(path)
    document = json.loads(source.read_text(encoding="utf-8"))
    entries = document.get("responses") if isinstance(document, Mapping) else document
    if not isinstance(entries, list):
        raise ValueError(
            f"{source}: a script is a list of responses, or an object with a "
            "responses list"
        )
    scripted: list[ModelResponse | ParseFailure] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{source}: entry {index} is not a JSON object")
        failure = _failure_block(entry)
        if failure is not None:
            reason_code = failure.get("reason_code")
            if not reason_code:
                raise ValueError(
                    f"{source}: entry {index} declares a parse failure with no "
                    "reason_code"
                )
            raw_text = failure.get("raw_text")
            scripted.append(
                ParseFailure(
                    str(failure.get("message", "scripted parse failure")),
                    reason_code=str(reason_code),
                    raw_text=None if raw_text is None else str(raw_text),
                )
            )
            continue
        try:
            # A recorded backend payload is parsed by the same strict reader a
            # live response goes through, so a script cannot smuggle in a
            # response shape the harness would have rejected.
            scripted.append(
                OpenAICompatibleClient.parse_response(
                    _response_payload(entry),
                    latency_ns=int(entry.get("latency_ns", 0)),
                )
            )
        except ParseFailure as error:
            raise ValueError(
                f"{source}: entry {index} is not a usable response: {error}"
            ) from error
    return tuple(scripted)


def parse_variant(value: str) -> Arm:
    """Parse an arm name, raising ``argparse.ArgumentTypeError`` if unknown.

    The matched pair and the capability ceiling are all reachable here. Which
    of them a challenge actually declares is the challenge's business, and
    asking for an arm a challenge does not declare is refused when the rule set
    is looked up rather than when the name is parsed.
    """
    try:
        return parse_arm(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def parse_policy_mode(value: str) -> PolicyMode:
    """Parse an enforcement mode, raising ``argparse.ArgumentTypeError`` if unknown."""
    try:
        return PolicyMode(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"unknown enforcement mode {value!r}; expected one of "
            + ", ".join(str(member) for member in PolicyMode)
        ) from None


def _failure_block(entry: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the parse-failure block of a script entry, or ``None``.

    An entry declares a failure either under a ``parse_failure`` key or by
    carrying a reason code with nothing a response could be built from.
    """
    declared = entry.get("parse_failure")
    if isinstance(declared, Mapping):
        return declared
    if declared:
        return entry
    if "reason_code" in entry and not (
        entry.get("choices") or entry.get("content") or entry.get("tool_calls")
    ):
        return entry
    return None


def _response_payload(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return the backend-shaped payload for one scripted response.

    A recorded payload is used as it stands. The shorthand form, which names
    content and tool calls directly, is expanded into the same shape so both
    kinds of script go through one parser.
    """
    if "choices" in entry:
        return dict(entry)
    calls: list[dict[str, Any]] = []
    for position, call in enumerate(entry.get("tool_calls") or ()):
        if not isinstance(call, Mapping):
            raise ValueError("a scripted tool call must be a JSON object")
        function = call.get("function")
        if not isinstance(function, Mapping):
            function = {
                "name": call.get("name"),
                "arguments": call.get("arguments", {}),
            }
        calls.append(
            {
                "id": str(call.get("id") or f"call-{position}"),
                "type": "function",
                "function": dict(function),
            }
        )
    message: dict[str, Any] = {"role": "assistant", "content": entry.get("content")}
    if calls:
        message["tool_calls"] = calls
    payload: dict[str, Any] = {
        "choices": [
            {
                "message": message,
                "finish_reason": str(
                    entry.get("finish_reason") or ("tool_calls" if calls else "stop")
                ),
            }
        ]
    }
    usage = entry.get("usage")
    if isinstance(usage, Mapping):
        payload["usage"] = {
            "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "completion_tokens": usage.get(
                "completion_tokens", usage.get("output_tokens", 0)
            ),
            "completion_tokens_details": {
                "reasoning_tokens": usage.get("reasoning_tokens", 0)
            },
        }
    return payload


def _challenge_files(paths: Sequence[str]) -> tuple[Path, ...]:
    """Expand the given files and directories into challenge file paths."""
    found: list[Path] = []
    for entry in paths:
        path = Path(str(entry))
        if path.is_dir():
            found.extend(sorted(path.glob("*.json")))
        elif path.exists():
            found.append(path)
        else:
            raise OSError(f"{path} does not exist")
    return tuple(found)


def _resolve_challenge(
    args: argparse.Namespace, manifest: RunManifest
) -> ChallengeSpec:
    """Return the challenge a stored run was made against.

    A path given on the command line is used as it stands. Otherwise the
    challenge is found by the id the manifest records, which is what lets a
    replay work from a run directory and the shipped challenge set alone.
    """
    declared = getattr(args, "challenge", None)
    if declared:
        return load_challenge(Path(str(declared)))
    directory = Path(str(getattr(args, "challenges", "challenges/demo")))
    if not directory.is_dir():
        raise OSError(
            f"{directory} is not a directory, so challenge "
            f"{manifest.challenge_id} cannot be found; pass --challenge"
        )
    for path in sorted(directory.glob("*.json")):
        try:
            spec = load_challenge(path, validate=False)
        except (OSError, ValueError, KeyError):
            continue
        if str(spec.challenge_id) == str(manifest.challenge_id):
            return spec
    raise ValueError(
        f"no challenge with id {manifest.challenge_id} in {directory}; "
        "pass --challenge with the file this run was made against"
    )


def _agent_result_from_events(artifacts: RunArtifacts) -> AgentResult:
    """Rebuild the trajectory summary from the stored ``run.completed`` event.

    Read back off disk rather than carried out of the run, so the summary the
    command line prints describes the file it just wrote.
    """
    completed = [
        event
        for event in load_events(artifacts.events_path)
        if event.event_type == EventType.RUN_COMPLETED
    ]
    if not completed:
        raise RuntimeError(
            f"{artifacts.events_path} carries no run.completed event, so the "
            "run did not finish"
        )
    payload = completed[-1].payload
    usage = payload.get("usage") or {}
    terminal = payload.get("terminal_signal")
    return AgentResult(
        stop_reason=StopReason(str(payload.get("stop_reason", StopReason.MAX_TURNS))),
        turns_used=int(payload.get("turns_used", 0)),
        tool_calls=int(payload.get("tool_calls", 0)),
        parse_failures=int(payload.get("parse_failures", 0)),
        invalid_arguments=int(payload.get("invalid_arguments", 0)),
        usage=TokenUsage(
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            reasoning_tokens=int(usage.get("reasoning_tokens", 0)),
        ),
        terminal_signal=None if terminal is None else TerminalSignal(str(terminal)),
        goal_reached=bool(payload.get("goal_reached", False)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
