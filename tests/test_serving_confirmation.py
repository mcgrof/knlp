import copy
import json

import pytest

from tools.kv.native.serving_confirmation import (
    backlog,
    memory_observations,
    run,
    stream_gaps,
)
from tools.kv.native.serving_storage_audit import storage


def test_backlog_counts_terminal_failures_and_final_drain():
    second = 1_000_000_000
    result = {
        "start_ns": second,
        "end_ns": 6 * second,
        "plan": {"arrival_duration_ns": 2 * second},
        "requests": [
            {"intended_ns": second, "end_ns": 5 * second, "status": "completed"},
            {"intended_ns": 3 * second, "end_ns": 4 * second, "status": "error"},
        ],
    }
    row = backlog(result)
    assert row["offered_minus_terminal_peak"] == 2
    assert row["request_seconds_of_outstanding_work"] == 5
    assert row["outstanding_at_arrival_end"] == 2
    assert row["drain_seconds_after_last_intended_arrival"] == 2
    assert row["finite_observation_seconds"] == 5
    assert not row["steady_state_or_sustainable_rate_established"]


def test_stream_gaps_preserve_empty_multitoken_and_invisible_events():
    def event(ms, ids, text, finish=None):
        return {
            "at_ns": ms * 1_000_000,
            "data": json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": text,
                            "token_ids": ids,
                            "finish_reason": finish,
                        }
                    ]
                }
            ),
        }

    rows = [
        {
            "status": "completed",
            "events": [
                event(1, [], ""),
                event(2, [1, 2], ""),
                event(5, [3], "a", "length"),
                {"at_ns": 7_000_000, "data": "[DONE]"},
            ],
        },
        {"status": "timeout", "events": []},
        {
            "status": "completed",
            "events": [event(8, [4], "b"), {"at_ns": 9_000_000, "data": "[DONE]"}],
        },
    ]
    result = stream_gaps(rows)
    assert result["counts"]["completed_requests"] == 1
    assert result["counts"]["multi_token_events"] == 1
    assert result["counts"]["events_without_tokens"] == 1
    assert result["counts"]["token_events_without_visible_text"] == 1
    assert result["event_token_count_histogram"] == {2: 1, 1: 1}
    assert result["per_request_gap_counts"] == [1]
    assert result["token_bearing_event_gaps_ms"]["p95"] == 3


def test_physical_storage_rejects_overlap_and_false_element_size():
    k = {
        "shape": [2, 16, 1, 2],
        "dtype": "torch.float16",
        "element_size": 2,
        "logical_bytes": 128,
        "storage_bytes": 192,
        "storage_pointer": 1000,
        "tensor_pointer": 1000,
        "contiguous": True,
    }
    v = dict(
        k,
        dtype="torch.float8_e4m3fn",
        element_size=1,
        logical_bytes=64,
        tensor_pointer=1128,
    )
    row = {
        "layers": [{"layer": "layer0", "K": k, "V": v, "physical_blocks": 2}],
        "physical_blocks_per_layer": 2,
        "logical_KV_tokens_per_layer": 32,
        "unique_KV_allocations": 1,
        "unique_KV_backing_bytes": 192,
        "K_payload_bytes": 128,
        "V_payload_bytes": 64,
        "serving_workspace_payload_bytes": 4096 * 16 * 1 * 2 * 4,
        "unique_scale_backing_bytes": 4,
        "unique_metadata_backing_bytes": 0,
        "allocator_peak_bytes": 600000,
        "sampled_driver_used_bytes": 800000,
    }
    geometry = {"layers": 1, "kv_heads": 1, "head_dim": 2}
    assert storage(row, True, geometry)["unique_KV_backing_bytes"] == 192
    for field, value in [("tensor_pointer", 1064), ("element_size", 2)]:
        changed = copy.deepcopy(row)
        changed["layers"][0]["V"][field] = value
        with pytest.raises(AssertionError):
            storage(changed, True, geometry)


def test_startup_driver_observation_cannot_be_hidden_by_later_smaller_values(tmp_path):
    (tmp_path / "server").mkdir()
    (tmp_path / "server/allocation.jsonl").write_text(
        json.dumps(
            {
                "phase": "device_initialized_before_model_load",
                "driver_device_used_bytes": 100,
            }
        )
        + "\n"
    )
    result = memory_observations(
        tmp_path,
        [{"sampled_driver_used_bytes": 10, "allocator_peak_bytes": 5}] * 2,
        {"open-near": {"sampled_driver_bytes": 20, "peak_reserved_bytes": 10}},
        {"total_driver_envelope_bytes": 90, "torch_allocator_reserved_cap_bytes": 9},
    )
    assert result["maximum_observed_driver_used_bytes"] == 100
    assert result["maximum_recorded_torch_reserved_peak_bytes"] == 10
    assert not result["driver_envelope_passed_at_observation_points"]
    assert not result["torch_envelope_passed_at_observation_points"]


def test_failed_http_launches_remain_closed_without_reduced_family(tmp_path):
    root, plan = tmp_path / "captures", tmp_path / "plan"
    plan.mkdir()
    (root / "jobs").mkdir(parents=True)
    arms = ["candidate", "native_fi", "native_fa3"]
    (plan / "DESIGN.json").write_text(
        json.dumps(
            {
                "format": "fp16",
                "length": 8192,
                "output_tokens": 128,
                "blocks": 10,
                "arms": arms,
                "orders": [arms] * 10,
                "source_sha": "synthetic",
                "primary_family": [{"id": "must-not-be-analyzed"}],
            }
        )
    )
    missing = run(root, plan)
    assert not missing["complete"] and len(missing["pending"]) == 33
    assert missing["primary_family"] == []
    for block in list(range(10)) + ["descriptive"]:
        for arm in arms:
            (root / "jobs" / f"000-fp16-8k-{block}-{arm}-result.json").write_text(
                json.dumps({"passed": False, "error": "synthetic startup failure"})
            )
    failed = run(root, plan)
    assert failed["complete"] and not failed["pending"]
    assert len(failed["failed_launches"]) == 33
    assert not failed["all_blocks_valid_and_complete"]
    assert failed["primary_family"] == []
