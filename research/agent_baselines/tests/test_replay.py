# SPDX-License-Identifier: MIT
"""CPU checks for the adapter's trace integrity and completion gates."""

import contextlib
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from research.agent_baselines import replay


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def make_source(path):
    record = {
        "attempt_index": 0,
        "trial_id": "test-trial",
        "task_timing": {"assigned_ns": 0, "finished_ns": 10_000_000_000},
    }
    put_json(path / "tasks/test-task.json", record)
    rows = []
    for seq, prompt, output in ((1, [1, 2, 3], [4]), (2, [1, 2, 3, 4, 5], [6])):
        rows.append(
            {
                "trial_id": "test-trial",
                "request_sequence": seq,
                "error": None,
                "token_identity_verified": True,
                "prompt_token_ids": prompt,
                "generated_token_ids": [output],
                "start_wall_ns": seq * 2_000_000_000,
                "end_wall_ns": seq * 2_000_000_000 + 1_000_000_000,
                "client_elapsed_ns": 1_000_000_000,
            }
        )
    logfile = path / "attempts/test-task/attempt_00/telemetry/llm_requests.jsonl"
    logfile.parent.mkdir(parents=True)
    logfile.write_text("".join(json.dumps(row) + "\n" for row in rows))
    provenance = {
        "model_id": "test-model",
        "model_revision": "model-rev",
        "tokenizer_revision": "tokenizer-rev",
        "kv_cache_dtype": "auto",
        "prefix_caching": True,
    }
    put_json(path / "manifest.json", {"server_provenance": provenance})
    return logfile, rows


def make_trace(path):
    path.mkdir(parents=True)
    head = {
        "instance_id": "test-task",
        "order": 0,
        "tail_gap_s": 1,
        "calls_replayed": 1,
    }
    row = {
        "seq": 1,
        "prompt_shared": 0,
        "prompt_suffix": [1, 2, 3],
        "prompt_len": 3,
        "prompt_sha256": hashlib.sha256(json.dumps([1, 2, 3]).encode()).hexdigest(),
        "output": [4],
        "output_len": 1,
        "gap_before_s": 0.1,
    }
    pathfile = path / "000_test-task.jsonl.gz"
    with gzip.open(pathfile, "wt") as stream:
        for entry in (head, row):
            stream.write(json.dumps(entry) + "\n")
    put_json(
        path / "trace_manifest.json",
        {
            "schema": replay.TRACE_SCHEMA,
            "synthetic": True,
            "tasks": 1,
            "steps": 1,
            "files": [
                {
                    "file": pathfile.name,
                    "order": 0,
                    "instance_id": "test-task",
                    "steps": 1,
                    "sha256": replay.digest(pathfile),
                }
            ],
        },
    )
    return pathfile


class ReplayAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_missing_token_identity_is_not_assumed_verified(self):
        log, rows = make_source(self.root / "source")
        del rows[0]["token_identity_verified"]
        log.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "unverified token identity"):
            replay.inspect_source(self.root / "source")

    def test_failed_calls_are_counted_without_fabricating_token_ids(self):
        log, rows = make_source(self.root / "source")
        rows[0]["error"] = "request failed"
        del rows[0]["prompt_token_ids"]
        del rows[0]["generated_token_ids"]
        log.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = replay.inspect_source(self.root / "source")
        self.assertEqual((result["steps"], result["failed_calls_omitted"]), (1, 1))

    def test_capture_failure_does_not_produce_a_shortened_eligible_trace(self):
        log, rows = make_source(self.root / "source")
        rows[1]["error"] = "TokenCaptureError"
        log.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "capture failed"):
            replay.inspect_source(self.root / "source")

    def test_source_overlap_rejected(self):
        log, rows = make_source(self.root / "source")
        rows[1]["start_wall_ns"] = rows[0]["end_wall_ns"] - 1
        log.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "start_wall_ns"):
            replay.inspect_source(self.root / "source")

    def test_empty_completion_rejected_because_upstream_would_generate(self):
        log, rows = make_source(self.root / "source")
        rows[0]["generated_token_ids"] = [[]]
        log.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "empty token sequence"):
            replay.inspect_source(self.root / "source")

    def test_trace_checksum_and_file_inventory(self):
        path = make_trace(self.root / "trace")
        self.assertEqual(replay.inspect_trace(path.parent)["max_context_tokens"], 4)
        path.write_bytes(path.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            replay.inspect_trace(path.parent)

    def test_zero_output_synthetic_trace_is_not_forced_replay(self):
        path = make_trace(self.root / "trace")
        with gzip.open(path, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        rows[1].update(output=[], output_len=0)
        with gzip.open(path, "wt") as stream:
            stream.write("".join(json.dumps(row) + "\n" for row in rows))
        manifest = replay.read_json(path.parent / "trace_manifest.json")
        manifest["files"][0]["sha256"] = replay.digest(path)
        put_json(path.parent / "trace_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "empty token sequence"):
            replay.inspect_trace(path.parent)

    def test_trace_unlisted_files_rejected(self):
        path = make_trace(self.root / "trace")
        (path.parent / "001_extra.jsonl.gz").write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError, "files differ"):
            replay.inspect_trace(path.parent)

    def test_upstream_passed_does_not_hide_partial_or_wrong_prompt_replay(self):
        run = self.root / "run"
        put_json(run / "run.json", {"status": "passed"})
        summary = {
            "interrupted": False,
            "requests": 1,
            "tasks": 1,
            "errors": 0,
            "forced_mismatch": 0,
            "prompt_len_mismatch": 0,
            "forced_match": 1,
        }
        put_json(run / "replay/tasks.jsonl", {"instance_id": "x", "ok": True})
        expected = {"steps": 1, "tasks": 1}
        for field, value in (
            ("interrupted", True),
            ("requests", 0),
            ("prompt_len_mismatch", 1),
            ("forced_match", 0),
        ):
            with self.subTest(field=field):
                changed = dict(summary, **{field: value})
                put_json(run / "replay/replay_summary.json", changed)
                with self.assertRaises(ValueError):
                    replay.validate_replay_result(run, expected)
        put_json(run / "replay/replay_summary.json", summary)
        result = replay.validate_replay_result(run, expected)
        self.assertFalse(result["task_quality_validated"])

    def args(self, command="replay"):
        make_trace(self.root / "trace")
        put_json(self.root / "model/config.json", {"model_type": "test"})
        return [
            command,
            "--checkout",
            str(self.root / "checkout"),
            "--out",
            str(self.root / "result"),
            "--trace",
            str(self.root / "trace"),
            "--model",
            str(self.root / "model"),
            "--max-model-len",
            "32",
            "--kv-bytes-per-token",
            "1024",
        ]

    @mock.patch.object(
        replay,
        "verify_checkout",
        return_value={"revision": replay.EFFICIENTAGENT_REVISION},
    )
    def test_grid_has_one_no_host_reference_and_three_policy_arms(self, _):
        args = replay.build_parser().parse_args(
            self.args("grid")
            + ["--host-capacities", "1", "4", "--worker-counts", "1", "2"]
        )
        plan = replay.prepare(args)
        self.assertEqual(len(plan["arms"]), 14)
        for arm in plan["arms"]:
            self.assertNotIn("--execute", arm["argv"])
            replay.build_parser().parse_args(arm["argv"][3:])
        self.assertFalse(args.out.exists())
        with mock.patch.object(replay.subprocess, "Popen") as popen:
            self.assertEqual(replay.execute(args, plan), 0)
            popen.assert_not_called()
        self.assertEqual(replay.read_json(args.out / "grid.json"), plan)

    @mock.patch.object(replay, "verify_checkout", return_value={})
    def test_host0_and_fractional_capacity_validation(self, _):
        base = self.args()
        for extra in (
            ["--host-gib", "0", "--admission", "fixed"],
            ["--host-gib", "0.04"],
            ["--host-gib", "nan"],
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                replay.prepare(replay.build_parser().parse_args(base + extra))

    @mock.patch.object(replay, "verify_checkout", return_value={})
    def test_recorded_trace_requires_matching_model_and_tokenizer_revisions(self, _):
        base = self.args()
        manifest = replay.read_json(self.root / "trace/trace_manifest.json")
        del manifest["synthetic"]
        put_json(self.root / "trace/trace_manifest.json", manifest)
        put_json(
            self.root / "trace/adapter_trace.json",
            {
                "data_kind": "recorded-agent",
                "server_provenance": {
                    "model_revision": "m1",
                    "tokenizer_revision": "t1",
                },
            },
        )
        args = replay.build_parser().parse_args(
            base
            + [
                "--host-gib",
                "0",
                "--model-revision",
                "wrong",
                "--tokenizer-revision",
                "t1",
            ]
        )
        with self.assertRaisesRegex(ValueError, "model-revision"):
            replay.prepare(args)

    @mock.patch.object(replay, "verify_checkout", return_value={})
    def test_recorded_cache_dtype_must_match_explicit_serving_dtype(self, _):
        base = self.args()
        manifest = replay.read_json(self.root / "trace/trace_manifest.json")
        del manifest["synthetic"]
        put_json(self.root / "trace/trace_manifest.json", manifest)
        provenance = {
            "model_revision": "m1",
            "tokenizer_revision": "t1",
            "kv_cache_dtype": "float16",
        }
        put_json(
            self.root / "trace/adapter_trace.json",
            {"data_kind": "recorded-agent", "server_provenance": provenance},
        )
        common = base + [
            "--host-gib",
            "0",
            "--model-revision",
            "m1",
            "--tokenizer-revision",
            "t1",
        ]
        with self.assertRaisesRegex(ValueError, "recorded kv_cache_dtype"):
            replay.prepare(replay.build_parser().parse_args(common))
        plan = replay.prepare(
            replay.build_parser().parse_args(common + ["--dtype", "float16"])
        )
        self.assertIn("--server-arg=--dtype=float16", plan["argv"])
        self.assertIn("--server-arg=--kv-cache-dtype=auto", plan["argv"])
        self.assertEqual(plan["model"]["resolved_kv_cache_dtype"], "float16")

    @mock.patch.object(replay.subprocess, "check_output")
    def test_untracked_import_overrides_block_source_verification(self, run):
        run.side_effect = [replay.EFFICIENTAGENT_REVISION, "?? numpy.py\n"]
        with self.assertRaisesRegex(ValueError, "modified or untracked"):
            replay.verify_checkout(self.root)
        self.assertNotIn("--", run.call_args.args[0])

    def test_existing_output_is_not_overwritten(self):
        out = self.root / "result"
        out.mkdir()
        args = replay.build_parser().parse_args(
            [
                "profile",
                "--checkout",
                str(self.root),
                "--out",
                str(out),
                "--trace",
                str(self.root),
            ]
        )
        with self.assertRaisesRegex(ValueError, "output already exists"):
            replay.prepare(args)

    @mock.patch.object(replay.subprocess, "check_output")
    def test_checkout_revision_gate(self, run):
        run.return_value = "0" * 40
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            replay.verify_checkout(self.root)

    @unittest.skipUnless(
        os.environ.get("EFFICIENTAGENT_CHECKOUT"),
        "set EFFICIENTAGENT_CHECKOUT for pinned upstream CPU integration",
    )
    def test_real_upstream_build_and_profile(self):
        source = self.root / "source"
        make_source(source)
        checkout = os.environ["EFFICIENTAGENT_CHECKOUT"]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = replay.main(
                [
                    "build-trace",
                    "--checkout",
                    checkout,
                    "--source-run",
                    str(source),
                    "--data-kind",
                    "recorded-agent",
                    "--out",
                    str(self.root / "built"),
                    "--execute",
                ]
            )
            self.assertEqual(code, 0, output.getvalue())
            code = replay.main(
                [
                    "profile",
                    "--checkout",
                    checkout,
                    "--trace",
                    str(self.root / "built/trace"),
                    "--out",
                    str(self.root / "profile"),
                    "--execute",
                ]
            )
            self.assertEqual(code, 0, output.getvalue())
        profile = replay.read_json(self.root / "profile/profile.json")["profile"]
        self.assertEqual(profile["calls"], 2)
        self.assertEqual(profile["prompt_tokens"], 8)
        self.assertEqual(profile["prompt_tokens_shared_with_previous_processed"], 4)


if __name__ == "__main__":
    unittest.main()
