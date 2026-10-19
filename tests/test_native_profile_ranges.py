# SPDX-License-Identifier: MIT
from tools.kv.native.profile_ranges import join_cuda_events


def test_correlate_async_gpu_time_without_using_host_duration():
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "name": "knlp|prefill|7|layer0",
                "pid": 1,
                "tid": 2,
                "ts": 0,
                "dur": 1000,
            },
            {
                "cat": "cuda_driver",
                "name": "cuLaunchKernel",
                "pid": 1,
                "tid": 2,
                "ts": 100,
                "dur": 10,
                "args": {"correlation": 42},
            },
            {
                "cat": "kernel",
                "name": "attention",
                "ts": 2000,
                "dur": 3,
                "args": {"correlation": 42},
            },
            {
                "cat": "kernel",
                "name": "unmatched",
                "ts": 3000,
                "dur": 99,
                "args": {"correlation": 99},
            },
        ]
    }
    result = join_cuda_events(trace)
    group = result["groups"][0]
    assert result["matched_cuda_kernels"] == 1 and result["unmatched_cuda_kernels"] == 1
    assert group["device_kernel_sum_us"] == 3 and group["device_kernel_union_us"] == 3
    assert group["host_range_us"] == 1000 and group["host_cuda_api_us"] == 10


def test_native_trace_without_compact_module(monkeypatch, tmp_path):
    import json
    import sys
    from types import ModuleType, SimpleNamespace

    from tools.kv.native.profile_ranges import trace_attention_routes

    class Wrapper:
        def run(self, query, cache):
            return "native result"

    class Attention:
        def forward(self, layer, query, cache):
            return Wrapper().run(query, cache)

    class Runner:
        def prepare_inputs(self, output):
            return SimpleNamespace(req_ids=["fresh"])

    class Query:
        dtype = "float16"

        def __len__(self):
            return 3

    def execute(runner, output):
        runner.prepare_inputs(output)
        return Attention().forward(
            SimpleNamespace(layer_name="layer0"),
            Query(),
            SimpleNamespace(dtype="float16"),
        )

    Runner.execute_model = execute
    modules = {
        "flashinfer": {
            "BatchPrefillWithPagedKVCacheWrapper": Wrapper,
            "BatchDecodeWithPagedKVCacheWrapper": type(
                "Decode", (), {"run": Wrapper.run}
            ),
        },
        "vllm.v1.attention.backends.flashinfer": {"FlashInferImpl": Attention},
        "vllm.v1.worker.gpu.model_runner": {"GPUModelRunner": Runner},
    }
    # Keep the native package importable while its optional compact module is absent.
    for name in [
        "vllm",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu",
    ]:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(
        sys.modules, "vllm.v1.attention.backends.compact_prefill", raising=False
    )
    for name, attributes in modules.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    originals = (
        Runner.execute_model,
        Runner.prepare_inputs,
        Attention.forward,
        Wrapper.run,
    )
    path = tmp_path / "operations.json"
    with trace_attention_routes(path):
        assert (
            Runner().execute_model(
                SimpleNamespace(_knlp_batch_id=7, num_scheduled_tokens={"fresh": 3})
            )
            == "native result"
        )
    assert originals == (
        Runner.execute_model,
        Runner.prepare_inputs,
        Attention.forward,
        Wrapper.run,
    )
    rows = json.loads(path.read_text())
    assert len(rows) == 1 and rows[0]["tag"] == "knlp|prefill|7|layer0"
    assert rows[0]["query_tokens"] == 3 and rows[0]["requests"] == ["fresh"]


def test_compact_import_dependency_error_is_not_hidden(monkeypatch, tmp_path):
    import builtins
    import sys
    from types import ModuleType

    import pytest

    from tools.kv.native.profile_ranges import trace_attention_routes

    monkeypatch.setitem(sys.modules, "flashinfer", ModuleType("flashinfer"))
    original = builtins.__import__

    def load(name, *args, **kwargs):
        if name == "vllm.v1.attention.backends.compact_prefill":
            raise ModuleNotFoundError(
                "missing internal dependency", name="compact_dependency"
            )
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", load)
    with pytest.raises(ModuleNotFoundError, match="internal dependency"):
        with trace_attention_routes(tmp_path / "operations.json"):
            raise AssertionError("broken compact import was hidden")


def test_sweep_matches_direct_interval_search_with_shuffled_threads():
    import random

    rng = random.Random(20261019)
    ranges = [
        {
            "ph": "X",
            "name": f"knlp|batch|{i}",
            "pid": i % 2,
            "tid": i % 3,
            "ts": rng.randrange(100),
            "dur": rng.randrange(40),
        }
        for i in range(200)
    ]
    events = ranges + [
        {
            "cat": "cuda_runtime",
            "name": "launch",
            "pid": i % 2,
            "tid": i % 3,
            "ts": i // 6,
            "dur": 1,
            "args": {"correlation": i},
        }
        for i in range(900)
    ]
    rng.shuffle(events)
    original_ranges = [e for e in events if e.get("ph") == "X"]
    expected = {}
    for e in events:
        if e.get("cat") != "cuda_runtime":
            continue
        containing = [
            r
            for r in original_ranges
            if r["pid"] == e["pid"]
            and r["tid"] == e["tid"]
            and r["ts"] <= e["ts"] < r["ts"] + r["dur"]
        ]
        if containing:
            expected[e["args"]["correlation"]] = min(
                containing, key=lambda r: r["dur"]
            )["name"]
    events += [
        {
            "cat": "kernel",
            "name": "attention",
            "ts": 1000 + i,
            "dur": 2,
            "args": {"correlation": i},
        }
        for i in range(900)
    ]
    result = join_cuda_events({"traceEvents": events})
    actual = {
        k["correlation"]: g["tag"] for g in result["groups"] for k in g["kernels"]
    }
    assert actual == expected
    assert result["unmatched_cuda_kernels"] == 900 - len(expected)
