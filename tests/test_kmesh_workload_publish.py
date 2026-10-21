# SPDX-License-Identifier: MIT
"""Check aggregate export and refuse contradictory private capture evidence."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from test_kmesh_headroom import invocation
from tools.reproduce.kmesh_residency.cpu import VLLM_COMMIT
from tools.reproduce.kmesh_residency.tot_workload import (
    MODEL,
    MODEL_REVISION,
    TOT_COMMIT,
    WORKLOAD_SECONDS,
)
from tools.reproduce.kmesh_residency.workload_publish import export


class WorkloadExportTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rows = []
        self.search = []
        puzzles = ([1, 1, 4, 6], [1, 1, 11, 11], [1, 1, 3, 8])
        answers = ("(1-1+4)*6", "(11+1)+(11+1)", "1+1+3+8")

        def add_request(task, purpose, parent):
            index = len(self.rows)
            row = invocation(
                f"PRIVATE-REQUEST-{index}",
                list(range(32)),
                [777],
                32,
                16 if index else 0,
                index * 20,
            )
            row["family_id"], row["purpose"], row["parent_request_id"] = (
                f"game24-{task:04}",
                purpose,
                parent,
            )
            row["finish_reason"] = "stop"
            for event in row["cache_observations"]:
                event.update(
                    request_id=row["request_id"],
                    time_ns=row["start_ns"] + 1,
                    free_blocks=0,
                )
            row["cache_observations"].append(
                {
                    "action": "after-free",
                    "request_id": row["request_id"],
                    "time_ns": row["start_ns"] + 2,
                    "free_blocks": 2,
                }
            )
            self.rows.append(row)
            return row["request_id"]

        for task, puzzle in enumerate(puzzles):
            selected, steps, parent = [""], [], None
            for step in range(4):
                parents = [add_request(task, "propose", parent) for _ in selected]
                new = [
                    selected[i % len(selected)]
                    + f"PRIVATE-STATE-{step}-{i}\n"
                    + (answers[task] if step == 3 else "")
                    for i in range(3)
                ]
                for _ in new:
                    add_request(task, "value", parents[0])
                steps.append(
                    {
                        "step": step,
                        "x": " ".join(map(str, puzzle)),
                        "ys": selected,
                        "new_ys": new,
                        "values": [20, 1, 0.001],
                        "select_new_ys": new[:2],
                    }
                )
                selected, parent = new[:2], parents[0]
            self.search.append(
                {
                    "task_index": task,
                    "puzzle": puzzle,
                    "answers": selected,
                    "correct": [task < 2] * 2,
                    "search": {"steps": steps},
                }
            )
        expected = json.loads(
            (
                Path(__file__).parents[1]
                / "tools/reproduce/kmesh_residency/tot-model-files.json"
            ).read_text()
        )
        self.manifest = {
            "schema": "knlp.kmesh.workload-capture.v1",
            "successful_capture": True,
            "workload_source_commit": TOT_COMMIT,
            "model_files_sha256": expected,
            "workload_files_sha256": json.loads(
                (
                    Path(__file__).parents[1]
                    / "tools/reproduce/kmesh_residency/tot-workload-files.json"
                ).read_text()
            ),
            "identity": {
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "model_config_sha256": expected["config.json"],
                "vllm_commit": VLLM_COMMIT,
                "dtype": "bfloat16",
                "block_tokens": 16,
                "adapter": None,
            },
            "runtime": {
                **{
                    key: "1.0"
                    for key in (
                        "python",
                        "torch",
                        "vllm",
                        "numpy",
                        "pandas",
                        "sympy",
                        "transformers",
                        "cuda",
                    )
                },
                "cpu_threads": 2,
                "private_path": "PRIVATE-RUNTIME",
            },
            "configuration": {
                "cache_bytes": 512 << 20,
                "max_model_len": 2048,
                "active_requests": 1,
                "generate": "propose",
                "evaluate": "value",
                "select": "greedy",
                "selected_states": 2,
                "evaluation_samples": 1,
                "max_output_tokens": 192,
                "temperature": 0.7,
                "seed_base": 123,
                "request_limit": 384,
                "workload_deadline_seconds": WORKLOAD_SECONDS,
            },
            "task_indices": [0, 1, 2],
            "task_results": [
                {
                    "task_index": i,
                    "solved": i < 2,
                    "selected_answers": 2,
                    "proposed_states": 12,
                    "selected_states": 8,
                }
                for i in range(3)
            ],
            "engine_cache": {
                "engine_id": "engine",
                "clock_id": "clock",
                "physical_pool_blocks": 3,
                "block_tokens": 16,
                "initial_free_blocks": 2,
                "final_free_blocks": 2,
                "final_nonnull_references": 0,
                "completed_requests": len(self.rows),
                "reserved_null_block_id": 0,
                "evictions": 0,
            },
            "application_events": [],
            "private_path": "PRIVATE-MANIFEST",
        }

    def write(self):
        for name, data in (
            ("workload.json", self.manifest),
            ("search.json", self.search),
            (
                "events.json",
                [event for row in self.rows for event in row["cache_observations"]],
            ),
        ):
            (self.root / name).write_text(json.dumps(data))
        (self.root / "invocations.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows)
        )

    def test_export_recomputes_bound_and_omits_private_payload(self):
        self.write()
        result = export(self.root)
        self.assertEqual(result["solved_tasks"], 2)
        self.assertEqual(result["headroom"]["observed_prefill_tokens"], 928)
        self.assertEqual(result["headroom"]["maximum_prefill_tokens_saved"], 0)
        self.assertEqual(result["lineage"]["non_exact_parent_prefix_edges"], 54)
        self.assertEqual(result["application_value_cache_hits"], 0)
        self.assertEqual(
            result["search_progress"],
            {
                "selected_extensions_after_first_step": 18,
                "selected_whitespace_only_extensions": 0,
            },
        )
        self.assertEqual(
            result["model_finish_counts"],
            {"propose": {"stop": 21}, "value": {"stop": 36}},
        )
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertNotIn("input_token_ids", json.dumps(result))
        self.assertFalse(result["latency_benchmark"])

    def test_unknown_model_finish_reason_is_rejected(self):
        self.rows[0]["finish_reason"] = "unknown"
        self.write()
        with self.assertRaisesRegex(ValueError, "finish reason"):
            export(self.root)

    def test_whitespace_selection_is_reported_without_hiding_task_failure(self):
        steps = self.search[2]["search"]["steps"]
        steps[1]["new_ys"][0] = steps[1]["ys"][0] + "\n"
        steps[1]["select_new_ys"][0] = steps[1]["new_ys"][0]
        steps[2]["ys"] = steps[1]["select_new_ys"][:]
        self.write()
        result = export(self.root)
        self.assertEqual(
            result["search_progress"]["selected_whitespace_only_extensions"], 1
        )
        self.assertFalse(result["tasks"][2]["solved"])

    def test_partial_capture_and_leaked_references_are_rejected(self):
        for field, value in (
            ("successful_capture", False),
            ("final_nonnull_references", 1),
            ("final_free_blocks", 1),
        ):
            with self.subTest(field=field):
                original = copy.deepcopy(self.manifest)
                target = (
                    self.manifest
                    if field == "successful_capture"
                    else self.manifest["engine_cache"]
                )
                target[field] = value
                self.write()
                with self.assertRaises(ValueError):
                    export(self.root)
                self.manifest = original

    def test_wrong_quality_summary_and_model_are_rejected(self):
        for mutate in (
            lambda: self.manifest["task_results"][0].update(solved=False),
            lambda: self.search[0].update(correct=[False, False]),
            lambda: self.manifest["identity"].update(model="other-model"),
        ):
            with self.subTest(mutate=mutate):
                manifest, search = copy.deepcopy(self.manifest), copy.deepcopy(
                    self.search
                )
                mutate()
                self.write()
                with self.assertRaises(ValueError):
                    export(self.root)
                self.manifest, self.search = manifest, search

    def test_referenced_eviction_and_altered_event_stream_are_rejected(self):
        self.rows[-1]["cache_observations"].append(
            {"action": "evicted", "prior_ref_count": 1}
        )
        self.manifest["engine_cache"]["evictions"] = 1
        self.write()
        with self.assertRaisesRegex(ValueError, "allocator summary"):
            export(self.root)
        (self.root / "events.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "allocator events"):
            export(self.root)

    def test_wrong_puzzle_source_and_incomplete_search_are_rejected(self):
        mutations = (
            lambda: self.search[0].update(puzzle=[4, 4, 4, 4]),
            lambda: self.manifest["workload_files_sha256"].update(
                {"src/tot/methods/bfs.py": "0" * 64}
            ),
            lambda: self.search[0]["search"].update(
                steps=self.search[0]["search"]["steps"][:1]
            ),
            lambda: self.search[0]["search"]["steps"][1].update(ys=["wrong-state"]),
            lambda: self.search[0]["search"]["steps"][0].update(select_new_ys=[]),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                manifest, search = copy.deepcopy(self.manifest), copy.deepcopy(
                    self.search
                )
                mutate()
                self.write()
                with self.assertRaises(ValueError):
                    export(self.root)
                self.manifest, self.search = manifest, search

    def test_terminal_release_must_match_request_and_free_pool(self):
        for change in ({"request_id": "wrong-request"}, {"free_blocks": 0}):
            with self.subTest(change=change):
                rows = copy.deepcopy(self.rows)
                self.rows[-1]["cache_observations"][-1].update(change)
                self.write()
                with self.assertRaisesRegex(ValueError, "allocator release"):
                    export(self.root)
                self.rows = rows

    def test_output_beyond_frozen_limit_is_rejected(self):
        row = self.rows[0]
        row["output_token_ids"] = [777] * 193
        row["ready"]["all_token_ids"] = row["input_token_ids"] + row["output_token_ids"]
        self.write()
        with self.assertRaisesRegex(ValueError, "invocation budget"):
            export(self.root)

    def test_application_hit_replaces_one_value_call_and_requires_known_family(self):
        self.rows.pop(1)
        self.manifest["engine_cache"]["completed_requests"] = len(self.rows)
        state = self.search[0]["search"]["steps"][0]["new_ys"][0]
        event = {
            "event": "application-value-cache-hit",
            "family_id": "game24-0000",
            "state_sha256": hashlib.sha256(state.encode()).hexdigest(),
        }
        self.manifest["application_events"] = [event]
        self.write()
        self.assertEqual(export(self.root)["application_value_cache_hits"], 1)
        self.manifest["application_events"].append(
            {**event, "family_id": "unknown-family"}
        )
        self.write()
        with self.assertRaisesRegex(ValueError, "application event"):
            export(self.root)


if __name__ == "__main__":
    unittest.main()
