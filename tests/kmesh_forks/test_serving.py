# SPDX-License-Identifier: MIT
"""Reject serving claims contradicted by the physical block observations."""

import copy
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2] / "tools/reproduce/kmesh_forks"
spec = importlib.util.spec_from_file_location(
    "serving_verify", ROOT / "serving_verify.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def warm_capture():
    rows = []
    for index in range(6):
        rows.append(
            {
                "label": "d0" if index == 0 else f"d0.b{index - 1}",
                "prompt_sha256": str(index),
                "prompt_tokens": 4096 if index == 0 else 4608,
                "num_cached_tokens": 0 if index == 0 else 4096,
                "output_tokens": 1,
                "output_token_ids": [99],
            }
        )
    prefix = list(range(1, 257))
    tables, digests = {}, {}
    events = [
        {
            "action": "allocate",
            "used_blocks": 257,
            "tables": {"parent": [prefix]},
            "prompt_sha256": {"parent": "0"},
        }
    ]
    for index in range(1, 6):
        tables[str(index)] = [
            prefix + list(range(257 + (index - 1) * 32, 257 + index * 32))
        ]
        digests[str(index)] = str(index)
    events.append(
        {
            "action": "allocate",
            "used_blocks": 417,
            "tables": tables,
            "prompt_sha256": digests,
        }
    )
    events.append(
        {"action": "free", "used_blocks": 1, "tables": {}, "prompt_sha256": {}}
    )
    value = {
        "rows": rows,
        "total_blocks": 2000,
        "null_block_id": 0,
        "allocation_events": events,
        "allocation_boundary_peak_used_blocks": 417,
        "sampled_peak_used_blocks": 400,
        "attention_backends": ["FLASH_ATTN"],
        "resolved_scheduler": dict(
            max_num_batched_tokens=16384,
            max_num_seqs=256,
            enable_chunked_prefill=True,
            async_scheduling=True,
        ),
    }
    return {
        "measurement": "serving-fork-replay",
        "seed": 20260805,
        "block_size": 16,
        "tree": "a4096-b5-t512",
        "children_after_parent_cached": value,
    }


class ServingEvidenceTest(unittest.TestCase):
    def test_live_prefix_aliases_pass_even_if_sampling_misses_peak(self):
        result = verifier.verify_case("children_after_parent_cached", warm_capture())
        self.assertEqual(result["observed_shared_sibling_pairs"], 10)
        self.assertEqual(result["allocation_boundary_peak_used_blocks"], 417)

    def test_copied_prefix_cannot_claim_aliasing(self):
        capture = warm_capture()
        capture["children_after_parent_cached"]["allocation_events"][1]["tables"]["2"][
            0
        ][0] = 1900
        with self.assertRaisesRegex(ValueError, "copied instead"):
            verifier.verify_case("children_after_parent_cached", capture)

    def test_private_sibling_tail_cannot_alias(self):
        capture = warm_capture()
        tables = capture["children_after_parent_cached"]["allocation_events"][1][
            "tables"
        ]
        tables["2"][0][-1] = tables["1"][0][-1]
        with self.assertRaisesRegex(ValueError, "private sibling tails"):
            verifier.verify_case("children_after_parent_cached", capture)

    def test_private_tail_cannot_alias_another_prefix(self):
        capture = warm_capture()
        value = capture["cold_sibling_burst"] = capture.pop(
            "children_after_parent_cached"
        )
        value["rows"] = value["rows"][1:]
        for row in value["rows"]:
            row["num_cached_tokens"] = 0
        tables = value["allocation_events"][1]["tables"]
        tables["2"][0][0] = 1900
        tables["2"][0][-1] = tables["1"][0][0]
        with self.assertRaisesRegex(ValueError, "private sibling tails"):
            verifier.verify_case("cold_sibling_burst", capture)

    def test_peak_must_be_derived_from_events(self):
        capture = warm_capture()
        capture["children_after_parent_cached"][
            "allocation_boundary_peak_used_blocks"
        ] = 400
        with self.assertRaisesRegex(ValueError, "peak differs"):
            verifier.verify_case("children_after_parent_cached", capture)

    def test_completed_requests_must_release_all_tables(self):
        capture = warm_capture()
        capture["children_after_parent_cached"]["allocation_events"].pop()
        with self.assertRaisesRegex(ValueError, "ownership remains"):
            verifier.verify_case("children_after_parent_cached", capture)

    def test_sampling_cannot_substitute_for_boundary_observations(self):
        capture = copy.deepcopy(warm_capture())
        capture["children_after_parent_cached"]["allocation_events"] = []
        with self.assertRaisesRegex(ValueError, "ownership remains"):
            verifier.verify_case("children_after_parent_cached", capture)


source_spec = importlib.util.spec_from_file_location("serving", ROOT / "serving.py")
serving = importlib.util.module_from_spec(source_spec)
source_spec.loader.exec_module(serving)


class RecordingTest(unittest.TestCase):
    def test_free_keeps_prior_allocation_snapshots_immutable(self):
        pool = SimpleNamespace(
            blocks=list(range(9)),
            free_block_queue=SimpleNamespace(num_free_blocks=8),
            null_block=SimpleNamespace(block_id=0),
        )
        tables = {}

        def allocate(request):
            tables[request.request_id] = ([1, 2],)
            pool.free_block_queue.num_free_blocks -= 2
            return object()

        def free(request):
            del tables[request.request_id]
            pool.free_block_queue.num_free_blocks += 2

        manager = SimpleNamespace(
            allocate_slots=allocate,
            free=free,
            block_pool=pool,
            get_block_ids=tables.__getitem__,
        )
        runner = SimpleNamespace(
            attn_groups=[
                [SimpleNamespace(backend=SimpleNamespace(get_name=lambda: "TEST"))]
            ]
        )
        core = SimpleNamespace(
            scheduler=SimpleNamespace(kv_cache_manager=manager),
            model_executor=SimpleNamespace(
                driver_worker=SimpleNamespace(
                    worker=SimpleNamespace(model_runner=runner)
                )
            ),
        )
        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(
                engine_core=SimpleNamespace(engine_core=core),
                vllm_config=SimpleNamespace(
                    scheduler_config=SimpleNamespace(
                        max_num_batched_tokens=16,
                        max_num_seqs=2,
                        enable_chunked_prefill=True,
                        async_scheduling=True,
                    )
                ),
            )
        )
        request = SimpleNamespace(request_id="request", prompt_token_ids=[1, 2])
        with serving.PoolSampler(llm) as sampler:
            manager.allocate_slots(request)
            first = copy.deepcopy(sampler.events[0])
            manager.free(request)
            self.assertEqual(sampler.events[0], first)
            for event in sampler.events:
                self.assertEqual(set(event["tables"]), set(event["prompt_sha256"]))
            self.assertEqual(sampler.events[-1]["tables"], {})
        self.assertIs(manager.allocate_slots, allocate)
        self.assertIs(manager.free, free)
