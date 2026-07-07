# SPDX-License-Identifier: MIT
"""CPU-only unit tests for the content-bearing trace replay library.

Pure Python + numpy; no GPU, vLLM, LMCache, or (for most tests)
transformers required.  Tiny in-repo synthetic fixtures only -- NO dataset
download (LMSYS-Chat-1M is HF-gated).  These pin the faithfulness contract:
dataset normalization, the repeated (growing / shared) prefix property that
yields real cache hits, deterministic tokenization, and arrival synthesis fed
to the reused open-loop scheduler.

Run:  python -m pytest tests/test_content_trace.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.reproduce.kvio.content_trace import (  # noqa: E402
    HashTokenizer,
    assign_arrivals,
    build_content_manifest,
    build_lmsys_requests,
    build_longbench_requests,
    q_probe_hook,
    synthesize_arrival_times,
)
from tools.reproduce.kvio.content_datasets import (  # noqa: E402
    normalize_lmsys,
    normalize_longbench,
)
from tools.reproduce.kvio.content_datasets.longbench import (  # noqa: E402
    group_longbench,
)
from tools.reproduce.kvio.serving_replay import (  # noqa: E402
    ScheduledRequest,
    SynthRequest,
    Wait,
    schedule_arrivals,
)


# ── Dataset normalization ──────────────────────────────────────────────────


def test_normalize_lmsys_multiturn():
    raw = {
        "conversation_id": "abc123",
        "conversation": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ],
    }
    rec = normalize_lmsys(raw)
    assert rec["conversation_id"] == "abc123"
    assert len(rec["turns"]) == 3
    assert rec["turns"][0] == {"role": "user", "content": "hi"}
    assert rec["turns"][2]["content"] == "bye"


def test_normalize_lmsys_missing_fields_degrade():
    rec = normalize_lmsys({})
    assert rec["conversation_id"] == ""
    assert rec["turns"] == []


def test_normalize_longbench_doc_question():
    raw = {"context": "a long document about foxes", "input": "what about foxes?"}
    rec = normalize_longbench(raw)
    assert rec["document"] == "a long document about foxes"
    assert rec["questions"] == ["what about foxes?"]
    assert rec["doc_id"]  # derived hash id


def test_group_longbench_shares_document():
    # Two questions about the SAME document -> one grouped record.
    raw = [
        normalize_longbench({"context": "doc one", "input": "q1"}),
        normalize_longbench({"context": "doc one", "input": "q2"}),
        normalize_longbench({"context": "doc two", "input": "q3"}),
    ]
    grouped = group_longbench(raw)
    assert len(grouped) == 2
    assert grouped[0]["questions"] == ["q1", "q2"]
    assert grouped[1]["questions"] == ["q3"]


# ── Repeated-prefix builder (the cache-hit property) ───────────────────────


def test_lmsys_growing_prefix_strict():
    tok = HashTokenizer(seed=1)
    conv = {
        "conversation_id": "c0",
        "turns": [
            {"role": "user", "content": "alpha beta"},
            {"role": "assistant", "content": "gamma delta"},
            {"role": "user", "content": "epsilon"},
        ],
    }
    reqs = build_lmsys_requests([conv], tok)
    assert len(reqs) == 3
    # Each request's prompt is a STRICT prefix of the next (growing chain).
    for a, b in zip(reqs, reqs[1:]):
        assert len(b.prompt_token_ids) > len(a.prompt_token_ids)
        assert b.prompt_token_ids[: len(a.prompt_token_ids)] == a.prompt_token_ids


def test_distinct_conversations_distinct_prefixes():
    tok = HashTokenizer(seed=1)
    c0 = {
        "conversation_id": "c0",
        "turns": [{"role": "user", "content": "alpha beta"}],
    }
    c1 = {
        "conversation_id": "c1",
        "turns": [{"role": "user", "content": "zeta eta"}],
    }
    reqs = build_lmsys_requests([c0, c1], tok)
    assert len(reqs) == 2
    # Different conversations -> different (non-prefix-sharing) first tokens.
    assert reqs[0].prompt_token_ids[0] != reqs[1].prompt_token_ids[0]


def test_longbench_shared_document_prefix():
    tok = HashTokenizer(seed=2)
    doc = {
        "doc_id": "d0",
        "document": "the quick brown fox",
        "questions": ["q one", "q two words"],
    }
    reqs = build_longbench_requests([doc], tok)
    assert len(reqs) == 2
    doc_ids = tok.encode("the quick brown fox" + "\n\n")
    # Both questions' prompts start with the shared document token prefix.
    for r in reqs:
        assert r.prompt_token_ids[: len(doc_ids)] == doc_ids
    # ...and diverge after the document (distinct questions).
    assert reqs[0].prompt_token_ids[len(doc_ids):] != reqs[1].prompt_token_ids[
        len(doc_ids):
    ]


def test_builder_indices_are_unique_and_sequential():
    tok = HashTokenizer(seed=1)
    convs = [
        {"conversation_id": "c0", "turns": [{"role": "u", "content": "a b"}]},
        {"conversation_id": "c1", "turns": [{"role": "u", "content": "c d"}]},
    ]
    reqs = build_lmsys_requests(convs, tok)
    idxs = [r.index for r in reqs]
    assert idxs == list(range(len(reqs)))


# ── Tokenization determinism ───────────────────────────────────────────────


def test_hash_tokenizer_deterministic():
    t1 = HashTokenizer(seed=3)
    t2 = HashTokenizer(seed=3)
    assert t1.encode("hello world foo") == t2.encode("hello world foo")
    # Identical text prefix -> identical token prefix.
    a = t1.encode("hello world")
    b = t1.encode("hello world foo")
    assert b[: len(a)] == a


def test_hash_tokenizer_respects_reserved_ids():
    t = HashTokenizer(seed=4, vocab_size=32000, reserved_ids=1000)
    ids = t.encode("some words here to encode")
    assert all(1000 <= i < 32000 for i in ids)


def test_real_tokenizer_roundtrip_deterministic():
    transformers = pytest.importorskip("transformers")  # noqa: F841
    from tools.reproduce.kvio.content_trace import load_hf_tokenizer

    try:
        tok = load_hf_tokenizer("gpt2")
    except Exception as e:
        pytest.skip(f"gpt2 tokenizer unavailable offline: {e}")
    a = tok.encode("The quick brown fox")
    b = tok.encode("The quick brown fox")
    assert a == b
    assert isinstance(a, list) and all(isinstance(i, int) for i in a)


# ── Arrival synthesis ──────────────────────────────────────────────────────


def test_arrivals_fixed_rate_ordering():
    times = synthesize_arrival_times(5, mode="fixed", rate_hz=10.0)
    assert times == [0.0, 100.0, 200.0, 300.0, 400.0]  # 1000/10 = 100 ms
    assert times == sorted(times)


def test_arrivals_poisson_deterministic_under_seed():
    a = synthesize_arrival_times(20, mode="poisson", rate_hz=5.0, seed=42)
    b = synthesize_arrival_times(20, mode="poisson", rate_hz=5.0, seed=42)
    c = synthesize_arrival_times(20, mode="poisson", rate_hz=5.0, seed=43)
    assert a == b
    assert a != c
    assert a == sorted(a)  # non-decreasing cumulative arrivals
    assert a[0] == pytest.approx(0.0)


def test_arrivals_rejects_bad_rate():
    with pytest.raises(ValueError):
        synthesize_arrival_times(3, rate_hz=0)


def test_assign_arrivals_sets_timestamps():
    tok = HashTokenizer(seed=1)
    conv = {
        "conversation_id": "c0",
        "turns": [
            {"role": "u", "content": "a b"},
            {"role": "a", "content": "c d"},
        ],
    }
    reqs = build_lmsys_requests([conv], tok)
    stamped = assign_arrivals(reqs, mode="fixed", rate_hz=10.0)
    assert [r.timestamp_ms for r in stamped] == [0.0, 100.0]
    # prompt_token_ids preserved through arrival assignment.
    assert stamped[0].prompt_token_ids == reqs[0].prompt_token_ids


# ── Reuse of the shared open-loop scheduler on content requests ────────────


def test_schedule_arrivals_on_content_requests():
    tok = HashTokenizer(seed=1)
    convs = [
        {"conversation_id": f"c{i}", "turns": [{"role": "u", "content": f"w{i} x{i}"}]}
        for i in range(3)
    ]
    reqs = build_lmsys_requests(convs, tok)
    stamped = assign_arrivals(reqs, mode="fixed", rate_hz=10.0)
    events = list(schedule_arrivals(stamped))
    assert all(isinstance(e, ScheduledRequest) for e in events)
    ts = [e.request.timestamp_ms for e in events]
    assert ts == sorted(ts)
    # release_s = timestamp_ms * 0.001 / speedup
    assert events[1].release_s == pytest.approx(0.1)


def test_schedule_arrivals_max_in_flight_on_content():
    reqs = [
        SynthRequest(index=i, timestamp_ms=float(i), prompt_token_ids=[1, 2, 3],
                     output_length=4)
        for i in range(3)
    ]
    gen = schedule_arrivals(reqs, max_in_flight=1)
    e0 = next(gen)
    assert isinstance(e0, ScheduledRequest)
    assert isinstance(gen.send(0), Wait)
    e1 = gen.send(1)
    assert isinstance(e1, ScheduledRequest)
    assert e1.request.index == 1


# ── Manifest + Q-probe hook ────────────────────────────────────────────────


def test_content_manifest_capture_discipline():
    m = build_content_manifest(
        model="Qwen/Qwen2.5-7B-Instruct",
        dataset="lmsys",
        kvio_trace_path="/tmp/kvio.jsonl",
        l2_path="/tmp/l2.bin",
    )
    assert m["dataset"]["name"] == "lmsys"
    assert m["dataset"]["synthesised_arrivals"] is True
    # Storage-faithful capture discipline.
    assert m["vllm"]["automatic_prefix_caching"] is False
    assert m["capture"]["kv_cache_dtype"] == "bf16"
    assert m["capture"]["serde"] == "off"
    assert m["capture"]["l2_backend"] == "raw_block"
    assert m["capture"]["kvio_trace_path"] == "/tmp/kvio.jsonl"
    assert m["capture"]["eviction_policy"] == "off"
    assert m["capture"]["tensor_parallel_size"] == 1


def test_q_probe_hook_is_stub():
    assert q_probe_hook(enabled=False) is None
    with pytest.raises(NotImplementedError):
        q_probe_hook(enabled=True)


# ── Serving/capture recipe (content_serve.py) ──────────────────────────────

from tools.reproduce.kvio.content_serve import (  # noqa: E402
    build_lmcache_config,
    build_vllm_argv,
    completion_payload,
    is_nvme_char_device,
    resolve_l2_device,
    server_env,
)


def _lmcache_cfg(device_path="/tmp/l2.bin", use_uring_cmd=False):
    return build_lmcache_config(
        device_path=device_path,
        capacity_bytes=32 * (1 << 30),
        kvio_trace_path="/tmp/kvio.jsonl",
        use_uring_cmd=use_uring_cmd,
        max_local_cpu_bytes=4 * (1 << 30),
        save_decode_cache=False,
    )


def test_lmcache_config_plugin_recipe():
    """The raw_block plugin recipe -- the silent-capture-nothing bug-trap."""
    cfg = _lmcache_cfg()
    # raw_block PLUGIN selected + resolvable to its class.
    assert cfg["storage_plugins"] == ["rust_raw_block"]
    extra = cfg["extra_config"]
    assert (
        extra["storage_plugin.rust_raw_block.module_path"]
        == "lmcache.v1.storage_backend.plugins.rust_raw_block_backend"
    )
    assert extra["storage_plugin.rust_raw_block.class_name"] == "RustRawBlockBackend"
    assert extra["rust_raw_block.device_path"] == "/tmp/l2.bin"
    assert extra["rust_raw_block.io_engine"] == "io_uring"
    assert extra["rust_raw_block.capacity_bytes"] == 32 * (1 << 30)
    # CPU tier on and bounded (> 0), else the backend init raises / skips.
    assert cfg["local_cpu"] is True
    assert cfg["max_local_cpu_size"] > 0
    # NO serde / compression / blending keys anywhere (would change KV bytes).
    flat = json.dumps(cfg).lower()
    for banned in ("serde", "compression", "compress", "blend", "cachegen"):
        assert banned not in flat, f"unexpected {banned!r} key in LMCache config"


def test_lmcache_config_use_uring_cmd_true_only_for_char_device():
    # Regular file -> use_uring_cmd False.
    cfg_file = _lmcache_cfg(device_path="/tmp/l2.bin", use_uring_cmd=False)
    assert cfg_file["extra_config"]["rust_raw_block.use_uring_cmd"] is False
    # NVMe char device -> use_uring_cmd True.
    cfg_dev = _lmcache_cfg(device_path="/dev/ng0n1", use_uring_cmd=True)
    assert cfg_dev["extra_config"]["rust_raw_block.use_uring_cmd"] is True
    # use_uring_cmd True on a NON-char device path is rejected (guards the
    # "uring_cmd on a regular file" mistake).
    with pytest.raises(ValueError):
        _lmcache_cfg(device_path="/tmp/l2.bin", use_uring_cmd=True)


def test_lmcache_config_rejects_zero_capacity():
    with pytest.raises(ValueError):
        build_lmcache_config(
            device_path="/tmp/l2.bin",
            capacity_bytes=0,
            kvio_trace_path="/tmp/kvio.jsonl",
            use_uring_cmd=False,
            max_local_cpu_bytes=1 << 30,
        )


def test_resolve_l2_device_knob_behaviour():
    # Empty knob -> local pre-sized file, use_uring_cmd False.
    dev, uring = resolve_l2_device("", "/run/stage/l2.bin")
    assert dev == "/run/stage/l2.bin"
    assert uring is False
    # /dev/ng* -> that device, use_uring_cmd True.
    dev, uring = resolve_l2_device("/dev/ng0n1", "/run/stage/l2.bin")
    assert dev == "/dev/ng0n1"
    assert uring is True
    # A set non-char path -> that path, use_uring_cmd False.
    dev, uring = resolve_l2_device("/mnt/scratch/l2.bin", "/run/stage/l2.bin")
    assert dev == "/mnt/scratch/l2.bin"
    assert uring is False


def test_is_nvme_char_device_pattern():
    assert is_nvme_char_device("/dev/ng0n1") is True
    assert is_nvme_char_device("/dev/ng12n3") is True
    assert is_nvme_char_device("/dev/nvme0n1") is False  # block, not generic char
    assert is_nvme_char_device("/tmp/l2.bin") is False
    assert is_nvme_char_device("") is False


def test_build_vllm_argv_capture_discipline():
    argv = build_vllm_argv(model="Qwen/Qwen2.5-7B-Instruct", port=8000, tp=1)
    # vLLM automatic prefix caching OFF (force LMCache L2 loads).
    assert "--no-enable-prefix-caching" in argv
    # TP=1.
    i = argv.index("--tensor-parallel-size")
    assert argv[i + 1] == "1"
    # Non-MP LMCacheConnectorV1 (kv_both); NOT the MP/lmcache_mp_connector.
    j = argv.index("--kv-transfer-config")
    kv = json.loads(argv[j + 1])
    assert kv["kv_connector"] == "LMCacheConnectorV1"
    assert kv["kv_role"] == "kv_both"
    assert "mp" not in kv["kv_connector"].lower()
    # No fp8 KV dtype anywhere.
    assert not any("fp8" in a.lower() for a in argv)


def test_build_vllm_argv_rejects_fp8():
    with pytest.raises(ValueError):
        build_vllm_argv(model="m", port=8000, kv_cache_dtype="fp8_e4m3")


def test_completion_payload_token_ids_and_ignore_eos():
    req = SynthRequest(
        index=0, timestamp_ms=0.0, prompt_token_ids=[10, 11, 12], output_length=7
    )
    body = completion_payload(req, model="m")
    # Token-ID prompt (a list of ints), NOT a string.
    assert body["prompt"] == [10, 11, 12]
    assert not isinstance(body["prompt"], str)
    assert body["max_tokens"] == 7
    assert body["ignore_eos"] is True
    assert body["temperature"] == 0.0


def test_server_env_sets_lmcache_capture_vars():
    env = server_env(
        {"PATH": "/usr/bin"},
        lmcache_config_path="/run/lmcache.yaml",
        kvio_trace_path="/run/kvio.jsonl",
    )
    assert env["LMCACHE_CONFIG_FILE"] == "/run/lmcache.yaml"
    assert env["LMCACHE_KVIO_TRACE"] == "/run/kvio.jsonl"
    assert env["PATH"] == "/usr/bin"  # base env preserved
