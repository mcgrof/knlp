# SPDX-License-Identifier: MIT
import copy

import pytest

from tools.kv.native.promotion import decision


def fixture():
    layers = [str(i) for i in range(36)]
    orders = [["candidate", "accepted", "native"]] * 10
    expected = {
        "identity": {"source": "fixed", "scales": "fixed", "protocol": "fixed"},
        "layers": layers,
        "sample_ids": {
            s: [str(i) for i in range(n)]
            for s, n in [("nll", 32), ("retrieval", 100), ("gsm8k", 1319)]
        },
        "pilot_orders": orders[:3],
        "confirmation_orders": orders,
    }
    calls = [
        {
            "layer": layer,
            "routing_equivalence_passed": True,
            "forward_placement_verified": True,
            "finite": True,
            "lengths": [8192] * fresh,
            "decode_tokens_checked": decode,
            "output_sha256": "a",
            "trusted_sha256": "a",
            "lse_sha256": "b",
            "trusted_lse_sha256": "b",
        }
        for fresh, decode in zip([8, 7, 7, 7, 3], [0, 8, 15, 22, 29])
        for layer in layers
    ]
    quality = {
        s: {k: (1.0 if s == "nll" else True) for k in ids}
        for s, ids in expected["sample_ids"].items()
    }
    block = {
        "order": orders[0],
        "synchronized_complete_request_timer_verified": True,
        "ratios": {"accepted": 1.02, "native": 1.03},
    }
    e = {
        "identity": expected["identity"],
        "complete": True,
        "fatal": None,
        "infrastructure_errors": [],
        "production": {p: calls for p in ["production-check-0", "production-check-1"]},
        "clean_memory": {
            "oracle_buffers_present": False,
            "pool_bytes": 48 * 2**30,
            "payload_bytes": 256 * 2**20,
            "payload_allocations": 1,
            "persistent_v_shadow_bytes": 0,
            "allocator_peak_bytes": 64 * 2**30,
            "driver_observed_max_bytes": 70 * 2**30,
            "driver_samples": 10,
            "driver_cadence_seconds": 1,
        },
        "fixture_faults_detected": sorted(
            [
                "omit_source_offset",
                "decode_page_substitution",
                "shift_output_suffix",
                "stale_mapping",
            ]
        ),
        "quality_samples": {
            r: copy.deepcopy(quality) for r in ["candidate", "native", "accepted"]
        },
        "pilot": [block] * 3,
        "confirmation": [block] * 10,
    }
    return e, expected


def test_complete_receipts_promote():
    e, p = fixture()
    assert decision(e, p)["unlocked_speed_claim"]


@pytest.mark.parametrize("pool_gib", [40, 48])
def test_explicit_protocol_pool_promotes_only_matching_receipt(pool_gib):
    e, p = fixture()
    p["pool_bytes"] = pool_gib * 2**30
    e["clean_memory"]["pool_bytes"] = p["pool_bytes"]
    assert decision(e, p)["unlocked_speed_claim"]
    e["clean_memory"]["pool_bytes"] = (48 if pool_gib == 40 else 40) * 2**30
    assert not decision(e, p)["unlocked_speed_claim"]


@pytest.mark.parametrize("pool_bytes", [None, True, 0, -1, "42949672960"])
def test_invalid_expected_pool_cannot_promote(pool_bytes):
    e, p = fixture()
    p["pool_bytes"] = e["clean_memory"]["pool_bytes"] = pool_bytes
    assert not decision(e, p)["unlocked_speed_claim"]


@pytest.mark.parametrize(
    "field",
    [
        "identity",
        "complete",
        "fatal",
        "infrastructure_errors",
        "production",
        "clean_memory",
        "fixture_faults_detected",
        "quality_samples",
        "pilot",
        "confirmation",
    ],
)
def test_missing_each_prerequisite_cannot_promote(field):
    e, p = fixture()
    del e[field]
    assert not decision(e, p)["unlocked_speed_claim"]


@pytest.mark.parametrize(
    "fault",
    [
        "gsm_missing",
        "quality_loss",
        "baseline_floor",
        "wrong_source",
        "incomplete_blocks",
        "infra",
        "wrong_timer",
        "production_failed",
        "nonfinite",
    ],
)
def test_failed_evidence_cannot_promote(fault):
    e, p = fixture()
    e = copy.deepcopy(e)
    if fault == "gsm_missing":
        del e["quality_samples"]["candidate"]["gsm8k"]
    if fault == "quality_loss":
        e["quality_samples"]["candidate"]["retrieval"] = {
            k: False for k in p["sample_ids"]["retrieval"]
        }
    if fault == "baseline_floor":
        e["quality_samples"]["native"]["gsm8k"] = {
            k: False for k in p["sample_ids"]["gsm8k"]
        }
    if fault == "wrong_source":
        e["identity"] = {"source": "wrong"}
    if fault == "incomplete_blocks":
        e["confirmation"] = e["confirmation"][:-1]
    if fault == "infra":
        e["infrastructure_errors"] = ["JIT missing"]
    if fault == "wrong_timer":
        e["confirmation"][0]["synchronized_complete_request_timer_verified"] = False
    if fault == "production_failed":
        e["production"]["production-check-0"][0]["routing_equivalence_passed"] = False
    if fault == "nonfinite":
        e["quality_samples"]["candidate"]["nll"]["0"] = float("nan")
    assert not decision(e, p)["unlocked_speed_claim"]
