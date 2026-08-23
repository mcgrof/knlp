"""The command line answers the two questions an author asks before running anything.

``list-challenges`` says what is shipped and at which revision, and
``validate-challenge`` says whether a file is a usable matched pair. Both are
thin wrappers over the library, so they are driven in-process rather than
through a subprocess: an exit code and the text on stdout are the whole
contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from conftest import challenge_document
from scopetrace.challenge import load_challenge_dir
from scopetrace.cli import (
    EXIT_ERROR,
    EXIT_INVALID,
    EXIT_OK,
    build_parser,
    main,
    parse_policy_mode,
    parse_variant,
)
from scopetrace.ids import PolicyMode, Variant
from scopetrace.manifest import short_digest


def write_challenge(directory: Path, name: str, document: dict) -> Path:
    """Write one challenge document into a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_the_parser_declares_every_subcommand() -> None:
    """All five documented subcommands are reachable from the parser."""
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    choices: set[str] = set()
    for action in parser._subparsers._group_actions:  # noqa: SLF001
        choices.update(getattr(action, "choices", {}) or {})
    assert choices == {
        "run",
        "replay",
        "export-trace",
        "validate-challenge",
        "list-challenges",
    }


def test_no_subcommand_is_refused_with_a_message(capsys) -> None:
    """Running the program with no subcommand explains itself and fails.

    argparse may refuse a missing subcommand itself, which exits rather than
    returning; either way the run must fail and say something.
    """
    try:
        status = main([])
    except SystemExit as exit_request:
        status = int(exit_request.code or 0)
    assert status != EXIT_OK
    assert capsys.readouterr().err


def test_list_challenges_succeeds_on_the_demo_directory(
    demo_challenge_dir: Path, capsys
) -> None:
    """Listing the shipped challenges succeeds and names every one of them."""
    if not sorted(demo_challenge_dir.glob("*.json")):
        pytest.skip("no demo challenges are shipped yet")
    assert main(["list-challenges", str(demo_challenge_dir)]) == EXIT_OK
    output = capsys.readouterr().out
    for spec in load_challenge_dir(demo_challenge_dir, validate=False):
        assert str(spec.challenge_id) in output
        assert str(spec.rung) in output
        assert short_digest(spec.challenge_revision) in output


def test_list_challenges_reports_a_header(demo_challenge_dir: Path, capsys) -> None:
    """The listing is a table a person can read."""
    if not sorted(demo_challenge_dir.glob("*.json")):
        pytest.skip("no demo challenges are shipped yet")
    main(["list-challenges", str(demo_challenge_dir)])
    first_line = capsys.readouterr().out.splitlines()[0]
    assert "challenge_id" in first_line
    assert "revision" in first_line


def test_list_challenges_on_an_empty_directory(tmp_path: Path, capsys) -> None:
    """An empty directory is reported rather than treated as an error."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["list-challenges", str(empty)]) == EXIT_OK
    assert "no challenges" in capsys.readouterr().out


def test_list_challenges_on_a_missing_directory(tmp_path: Path) -> None:
    """A directory that is not there is a failure, not an empty listing."""
    assert main(["list-challenges", str(tmp_path / "absent")]) == EXIT_ERROR


def test_validate_challenge_succeeds_on_the_demo_directory(
    demo_challenge_dir: Path, capsys
) -> None:
    """Every shipped challenge passes validation from the command line."""
    if not sorted(demo_challenge_dir.glob("*.json")):
        pytest.skip("no demo challenges are shipped yet")
    assert main(["validate-challenge", str(demo_challenge_dir)]) == EXIT_OK
    output = capsys.readouterr().out
    assert output.count("valid") >= len(sorted(demo_challenge_dir.glob("*.json")))


def test_validate_challenge_accepts_a_single_file(
    demo_challenge_dir: Path, capsys
) -> None:
    """One file can be validated on its own."""
    files = sorted(demo_challenge_dir.glob("*.json"))
    if not files:
        pytest.skip("no demo challenges are shipped yet")
    assert main(["validate-challenge", str(files[0])]) == EXIT_OK
    assert files[0].name in capsys.readouterr().out


def test_validate_challenge_accepts_a_written_challenge(tmp_path: Path) -> None:
    """A challenge produced by the fixtures validates from the command line."""
    path = write_challenge(tmp_path / "written", "pair.json", challenge_document())
    assert main(["validate-challenge", str(path)]) == EXIT_OK


def test_validate_challenge_reports_every_violation(tmp_path: Path, capsys) -> None:
    """A broken challenge exits invalid and lists what is wrong with it."""
    document = challenge_document()
    document["fast_route"]["steps"][0]["resource_id"] = "service/not-declared"
    document["capability_contract"]["control_success"] = ["statement.plan"]
    path = write_challenge(tmp_path / "broken", "broken.json", document)
    assert main(["validate-challenge", str(path)]) == EXIT_INVALID
    errors = capsys.readouterr().err
    assert "undeclared resource" in errors
    assert "harness-generated" in errors


def test_validate_challenge_rejects_a_pair_with_nothing_to_measure(
    tmp_path: Path, capsys
) -> None:
    """Two arms with one rule set is not a matched pair."""
    from conftest import CONTROL_RULES

    document = challenge_document(treatment_rules=CONTROL_RULES)
    path = write_challenge(tmp_path / "same", "same.json", document)
    assert main(["validate-challenge", str(path)]) == EXIT_INVALID
    assert "same rule set" in capsys.readouterr().err


def test_validate_challenge_rejects_a_file_that_is_not_json(
    tmp_path: Path, capsys
) -> None:
    """A file that does not parse is reported rather than raised."""
    path = tmp_path / "broken.json"
    path.write_text("{ not json", encoding="utf-8")
    assert main(["validate-challenge", str(path)]) == EXIT_INVALID
    assert "not valid JSON" in capsys.readouterr().err


def test_validate_challenge_on_no_files(tmp_path: Path, capsys) -> None:
    """An empty directory is reported and is not a validation failure."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["validate-challenge", str(empty)]) == EXIT_OK
    assert "no challenge files" in capsys.readouterr().out


def test_variant_and_mode_parse_by_name() -> None:
    """The two enumerated arguments accept their declared spellings."""
    assert parse_variant("control") is Variant.CONTROL
    assert parse_variant("treatment") is Variant.TREATMENT
    assert parse_policy_mode("observe") is PolicyMode.OBSERVE
    assert parse_policy_mode("enforce") is PolicyMode.ENFORCE
    with pytest.raises(argparse.ArgumentTypeError):
        parse_variant("placebo")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_policy_mode("advisory")


def replay_script() -> dict:
    """Return a scripted response file that takes the authorized route."""
    return {
        "responses": [
            {
                "content": "PLAN: read the notes first.",
                "tool_calls": [
                    {
                        "id": "call-read",
                        "name": "resource.read",
                        "arguments": {
                            "resource_id": "store/project-a/notes",
                            "purpose_id": "gather-context",
                        },
                    }
                ],
                "usage": {"input_tokens": 20, "output_tokens": 6},
            },
            {
                "tool_calls": [
                    {
                        "id": "call-review",
                        "name": "service.call",
                        "arguments": {
                            "resource_id": "service/review-publish",
                            "operation": "publish",
                            "purpose_id": "complete-assigned-task",
                        },
                    }
                ]
            },
            {
                "tool_calls": [
                    {
                        "id": "call-complete",
                        "name": "task.complete",
                        "arguments": {
                            "summary": "published",
                            "purpose_id": "complete-assigned-task",
                        },
                    }
                ]
            },
        ]
    }


def perform_cli_run(tmp_path: Path) -> tuple[int, Path, Path]:
    """Run one trajectory through the command line and return status and paths.

    A plain helper rather than a fixture, so that what the subcommand printed
    stays in the capture the calling test reads.
    """
    challenge_path = write_challenge(
        tmp_path / "challenges", "pair.json", challenge_document()
    )
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(replay_script()), encoding="utf-8")
    out_dir = tmp_path / "run"
    status = main(
        [
            "run",
            "--challenge",
            str(challenge_path),
            "--out-dir",
            str(out_dir),
            "--variant",
            "control",
            "--policy-mode",
            "enforce",
            "--script",
            str(script_path),
            "--seed",
            "5",
        ]
    )
    return status, challenge_path, out_dir


def test_run_writes_a_run_directory(tmp_path: Path, capsys) -> None:
    """The run subcommand produces the artifacts and names them on stdout."""
    status, _challenge_path, out_dir = perform_cli_run(tmp_path)
    assert status == EXIT_OK
    for name in ("manifest.json", "events.jsonl", "outcome.json", "trace.json"):
        assert (out_dir / name).is_file(), name
    output = capsys.readouterr().out
    assert "capable_in_scope_success" in output
    assert str(out_dir / "events.jsonl") in output


def test_replay_agrees_with_the_stored_verdict(tmp_path: Path, capsys) -> None:
    """Replaying a run the command line just wrote reproduces its verdict."""
    _status, challenge_path, out_dir = perform_cli_run(tmp_path)
    capsys.readouterr()
    assert main(["replay", str(out_dir), "--challenge", str(challenge_path)]) == EXIT_OK


def test_replay_detects_an_edited_verdict(tmp_path: Path, capsys) -> None:
    """A verdict that no longer matches its events is reported as a disagreement."""
    _status, challenge_path, out_dir = perform_cli_run(tmp_path)
    capsys.readouterr()
    outcome_path = out_dir / "outcome.json"
    record = json.loads(outcome_path.read_text("utf-8"))
    record["in_scope_success"] = False
    record["final_class"] = "capable_forbidden_discovered_avoided"
    outcome_path.write_text(json.dumps(record), encoding="utf-8")
    assert (
        main(["replay", str(out_dir), "--challenge", str(challenge_path)])
        == EXIT_INVALID
    )
    captured = capsys.readouterr()
    assert "final_class" in captured.err + captured.out


def test_export_trace_rebuilds_the_view(tmp_path: Path, capsys) -> None:
    """The trace can be rebuilt from the stored event file alone."""
    _status, _challenge_path, out_dir = perform_cli_run(tmp_path)
    capsys.readouterr()
    rebuilt = tmp_path / "rebuilt.json"
    assert main(["export-trace", str(out_dir), "--out", str(rebuilt)]) == EXIT_OK
    trace = json.loads(rebuilt.read_text("utf-8"))
    assert trace["otherData"]["run_id"] == "publish-summary/control/replay/s5"
    assert (out_dir / "trace.json").read_bytes() == rebuilt.read_bytes()


def test_the_replay_script_loader_accepts_a_parse_failure(tmp_path: Path) -> None:
    """A script can carry a parse failure so the retry path is exercisable."""
    from scopetrace.cli import load_replay_script
    from scopetrace.model_client import ParseFailure

    path = tmp_path / "script.json"
    path.write_text(
        json.dumps(
            {
                "responses": [
                    {"parse_failure": {"reason_code": "no_choices"}},
                    {"content": "NOTE: recovered."},
                ]
            }
        ),
        encoding="utf-8",
    )
    scripted = load_replay_script(path)
    assert isinstance(scripted[0], ParseFailure)
    assert scripted[0].reason_code == "no_choices"
    assert scripted[1].content == "NOTE: recovered."
