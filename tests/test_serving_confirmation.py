import copy
import json

import pytest

from tools.kv.native.serving_confirmation import backlog, stream_gaps
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
    def event(ms, ids, text):
        return {
            "at_ns": ms * 1_000_000,
            "data": json.dumps(
                {"choices": [{"index": 0, "text": text, "token_ids": ids}]}
            ),
        }

    rows = [
        {
            "status": "completed",
            "events": [
                event(1, [], ""),
                event(2, [1, 2], ""),
                event(5, [3], "a"),
                {"at_ns": 7_000_000, "data": "[DONE]"},
            ],
        },
        {"status": "timeout", "events": []},
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
