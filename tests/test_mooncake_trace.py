# SPDX-License-Identifier: MIT
"""CPU-only unit tests for the Mooncake trace replay library.

Pure Python + numpy; no GPU, vLLM, or LMCache required.  These pin the
faithfulness contract: stable hash->block identity, shared-prefix
preservation, request-unique (never shared) padding, exact length, and
open-loop arrival scheduling.

Run:  python -m pytest tests/test_mooncake_trace.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.reproduce.kvio.mooncake_trace import (  # noqa: E402
    DEFAULT_BLOCK_SIZE,
    MooncakeRecord,
    ScheduledRequest,
    TokenSynthesizer,
    Wait,
    build_config_manifest,
    parse_trace,
    schedule_arrivals,
    synth_requests,
)

BLOCK = DEFAULT_BLOCK_SIZE


def _write_jsonl(tmp_path, rows) -> Path:
    p = tmp_path / "trace.jsonl"
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


# ── Parser ─────────────────────────────────────────────────────────────────


def test_parse_valid_record(tmp_path):
    path = _write_jsonl(
        tmp_path,
        [
            {
                "timestamp": 0,
                "input_length": 1024,
                "output_length": 32,
                "hash_ids": [10, 11],
            },
            {
                "timestamp": 5,
                "input_length": 600,
                "output_length": 8,
                "hash_ids": [10, 12],
            },
        ],
    )
    recs = parse_trace(path)
    assert len(recs) == 2
    assert not any(r.malformed for r in recs)
    assert recs[0].input_length == 1024
    assert recs[0].hash_ids == (10, 11)
    assert recs[1].timestamp_ms == 5.0


def test_schema_validation_missing_field(tmp_path):
    path = _write_jsonl(
        tmp_path,
        [{"timestamp": 0, "input_length": 512, "hash_ids": [1]}],  # no output_length
    )
    recs = parse_trace(path)
    assert len(recs) == 1
    assert recs[0].malformed
    assert "output_length" in recs[0].malformed_reason


def test_schema_validation_bad_lengths(tmp_path):
    path = _write_jsonl(
        tmp_path,
        [{"timestamp": 0, "input_length": 0, "output_length": 4, "hash_ids": []}],
    )
    recs = parse_trace(path)
    assert recs[0].malformed


def test_block_count_sanity_flags_wildly_off(tmp_path):
    # input_length=1024 -> expected ceil(1024/512)=2 blocks; 50 is corrupt.
    path = _write_jsonl(
        tmp_path,
        [
            {
                "timestamp": 0,
                "input_length": 1024,
                "output_length": 4,
                "hash_ids": list(range(50)),
            }
        ],
    )
    recs = parse_trace(path)
    assert recs[0].malformed
    assert "hash count" in recs[0].malformed_reason


def test_block_count_short_record_is_valid(tmp_path):
    # Fewer hashes than blocks is legitimate (tail is noise-filled).
    path = _write_jsonl(
        tmp_path,
        [{"timestamp": 0, "input_length": 1500, "output_length": 4, "hash_ids": [7]}],
    )
    recs = parse_trace(path)
    assert not recs[0].malformed


def test_skip_malformed(tmp_path):
    path = _write_jsonl(
        tmp_path,
        [
            {"timestamp": 0, "input_length": 512, "output_length": 4, "hash_ids": [1]},
            {"timestamp": 1, "input_length": 0, "output_length": 4, "hash_ids": []},
        ],
    )
    recs = parse_trace(path, skip_malformed=True)
    assert len(recs) == 1


def test_max_records(tmp_path):
    rows = [
        {"timestamp": i, "input_length": 512, "output_length": 4, "hash_ids": [i]}
        for i in range(10)
    ]
    path = _write_jsonl(tmp_path, rows)
    recs = parse_trace(path, max_records=3)
    assert len(recs) == 3


# ── Token-ID synthesis determinism ─────────────────────────────────────────


def test_block_determinism_same_object():
    synth = TokenSynthesizer(seed=42)
    a = synth.block_for_hash(1234)
    b = synth.block_for_hash(1234)
    assert a == b
    assert len(a) == BLOCK


def test_block_determinism_across_instances():
    # Two independent synthesizers (mimicking separate processes) must
    # agree -- this is the stable-hash guarantee that Python hash() breaks.
    s1 = TokenSynthesizer(seed=7)
    s2 = TokenSynthesizer(seed=7)
    assert s1.block_for_hash(999) == s2.block_for_hash(999)


def test_distinct_hashes_distinct_blocks():
    synth = TokenSynthesizer(seed=1)
    a = synth.block_for_hash(1)
    b = synth.block_for_hash(2)
    assert a != b


def test_tokens_avoid_reserved_ids():
    synth = TokenSynthesizer(seed=3, vocab_size=32000, reserved_ids=1000)
    block = synth.block_for_hash(555)
    assert min(block) >= 1000
    assert max(block) < 32000


def test_shared_leading_hashes_share_prefix():
    synth = TokenSynthesizer(seed=5)
    r1 = MooncakeRecord(0, 0.0, 2 * BLOCK, 4, (100, 200))
    r2 = MooncakeRecord(1, 0.0, 2 * BLOCK, 4, (100, 300))
    t1 = synth.prompt_token_ids(r1)
    t2 = synth.prompt_token_ids(r2)
    # First shared block identical; second (differing hash) differs.
    assert t1[:BLOCK] == t2[:BLOCK]
    assert t1[BLOCK:] != t2[BLOCK:]


# ── No shared padding (no fake hits) ───────────────────────────────────────


def test_no_shared_pad_tail_is_request_unique():
    # Two requests share the same short leading hash but need noise tails.
    synth = TokenSynthesizer(seed=9)
    inp = BLOCK + 300  # one real block + 300 noise tokens
    r1 = MooncakeRecord(0, 0.0, inp, 4, (500,))
    r2 = MooncakeRecord(1, 0.0, inp, 4, (500,))
    t1 = synth.prompt_token_ids(r1)
    t2 = synth.prompt_token_ids(r2)
    # Shared real prefix identical...
    assert t1[:BLOCK] == t2[:BLOCK]
    # ...but the noise tails must differ (no shared pad token).
    assert t1[BLOCK:] != t2[BLOCK:]
    # And a tail is not a constant repeated pad value.
    assert len(set(t1[BLOCK:])) > 1


# ── Length correctness ─────────────────────────────────────────────────────


def test_length_trim_when_too_long():
    synth = TokenSynthesizer(seed=2)
    # Two blocks worth of hashes but input_length only 700 -> trim.
    rec = MooncakeRecord(0, 0.0, 700, 4, (1, 2))
    ids = synth.prompt_token_ids(rec)
    assert len(ids) == 700


def test_length_pad_when_too_short():
    synth = TokenSynthesizer(seed=2)
    rec = MooncakeRecord(0, 0.0, BLOCK + 123, 4, (1,))
    ids = synth.prompt_token_ids(rec)
    assert len(ids) == BLOCK + 123


def test_length_exact_multiple():
    synth = TokenSynthesizer(seed=2)
    rec = MooncakeRecord(0, 0.0, 2 * BLOCK, 4, (1, 2))
    ids = synth.prompt_token_ids(rec)
    assert len(ids) == 2 * BLOCK


def test_synth_requests_skips_malformed():
    synth = TokenSynthesizer(seed=2)
    recs = [
        MooncakeRecord(0, 0.0, 512, 4, (1,)),
        MooncakeRecord(1, 1.0, 0, 4, (), malformed=True, malformed_reason="bad"),
    ]
    reqs = synth_requests(recs, synth)
    assert len(reqs) == 1
    assert reqs[0].index == 0


# ── Arrival scheduler ──────────────────────────────────────────────────────


def _reqs(synth, timestamps):
    out = []
    for i, ts in enumerate(timestamps):
        rec = MooncakeRecord(i, float(ts), 512, 4, (i,))
        out.append(synth.synth_request(rec))
    return out


def test_schedule_orders_by_timestamp():
    synth = TokenSynthesizer(seed=1)
    reqs = _reqs(synth, [30, 10, 20])
    events = list(schedule_arrivals(reqs))
    ts = [e.request.timestamp_ms for e in events]
    assert ts == [10, 20, 30]


def test_schedule_ms_to_s_and_speedup():
    synth = TokenSynthesizer(seed=1)
    reqs = _reqs(synth, [1000])  # 1000 ms
    (e,) = list(schedule_arrivals(reqs, ms_to_s=0.001, speedup=1.0))
    assert e.release_s == pytest.approx(1.0)
    (e2,) = list(schedule_arrivals(reqs, ms_to_s=0.001, speedup=2.0))
    assert e2.release_s == pytest.approx(0.5)


def test_schedule_speedup_must_be_positive():
    synth = TokenSynthesizer(seed=1)
    reqs = _reqs(synth, [1000])
    with pytest.raises(ValueError):
        list(schedule_arrivals(reqs, speedup=0))


def test_schedule_max_in_flight_cap():
    synth = TokenSynthesizer(seed=1)
    reqs = _reqs(synth, [0, 1, 2])
    gen = schedule_arrivals(reqs, max_in_flight=1)
    e0 = next(gen)
    assert isinstance(e0, ScheduledRequest)
    # Cap=1 reached (1 in-flight, 0 completed) -> Wait.
    ev = gen.send(0)
    assert isinstance(ev, Wait)
    assert ev.in_flight == 1
    # Report one completion -> next request released.
    e1 = gen.send(1)
    assert isinstance(e1, ScheduledRequest)
    assert e1.request.index == 1


def test_schedule_max_in_flight_allows_concurrency():
    synth = TokenSynthesizer(seed=1)
    reqs = _reqs(synth, [0, 1, 2, 3])
    gen = schedule_arrivals(reqs, max_in_flight=2)
    e0 = next(gen)
    e1 = gen.send(0)  # in-flight=1 < 2 -> release second immediately
    assert isinstance(e0, ScheduledRequest)
    assert isinstance(e1, ScheduledRequest)
    # Now in-flight=2 -> Wait
    ev = gen.send(0)
    assert isinstance(ev, Wait)


# ── Config manifest ────────────────────────────────────────────────────────


def test_config_manifest_fields(tmp_path):
    path = _write_jsonl(
        tmp_path,
        [{"timestamp": 0, "input_length": 512, "output_length": 4, "hash_ids": [1]}],
    )
    m = build_config_manifest(
        model="Qwen/Qwen2.5-7B-Instruct",
        trace_path=path,
        seed=123,
        speedup=4.0,
        max_in_flight=8,
    )
    assert m["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert len(m["trace_sha256"]) == 64
    assert m["seed"] == 123
    assert m["arrival"]["speedup"] == 4.0
    assert m["arrival"]["max_in_flight"] == 8
    # Storage-faithful defaults.
    assert m["vllm"]["automatic_prefix_caching"] is False
    assert m["lmcache"]["save_decode_cache"] is False
    assert m["lmcache"]["save_unfull_chunk"] is False
    assert m["lmcache"]["compression"] == "off"
