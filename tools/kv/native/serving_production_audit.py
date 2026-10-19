# SPDX-License-Identifier: MIT
"""Read retained HTTP production receipts and raw dispatch traces independently."""
import argparse
import collections
import gzip
import hashlib
import json
import struct
from pathlib import Path

from .serving_storage_audit import storage

if not __debug__:
    raise RuntimeError("Evidence audits require Python assertions enabled")


def load(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fp32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def joined(events, operators):
    calls = [
        e for e in events if e.get("cat") == "cpu_op" and e.get("name") in operators
    ]
    assert calls
    ids = {e["args"]["External id"] for e in calls}
    assert len(ids) == len(calls)
    kernels = [
        e
        for e in events
        if e.get("cat") == "kernel"
        and e.get("dur", 0) > 0
        and e.get("args", {}).get("External id") in ids
    ]
    assert {e["args"]["External id"] for e in kernels} == ids
    return {
        "operators": sorted(operators),
        "calls": len(calls),
        "cuda_kernels": len(kernels),
        "positive_cuda_duration_us": sum(e["dur"] for e in kernels),
        "devices": sorted({e["args"]["device"] for e in kernels}),
        "kernel_names": sorted({e["name"] for e in kernels}),
    }


def audit(root, fmt, arm, tag, prefix=None):
    r = root / f"{prefix or fmt}-{arm}-{tag}"
    assert (
        load(r / "DONE.json")["complete"]
        and load(r / "DONE.json")["excluded_from_confirmation"]
    )
    s = r / "server"
    p = load(s / "inputs/protocol.json")
    shape = p["geometry"]
    mixed = arm == "candidate"
    identity = load(s / "worker-identity.json")
    assert (
        identity["sources"] == p["backend_sources"]
        and identity["arm"] == ("k16_v8" if mixed else "k16_v16")
        and identity["torch_threads"] == 22
        and identity["calibration_before_worker_warmup_and_capture"]
    )
    for name, h in p["inputs_sha256"].items():
        assert sha(s / "inputs" / name) == h
    installed = load(s / "before-capture/installed.json")
    scales = load(s / "inputs/scales.json")["scales"]
    capture = {x["layer"]: x for x in load(s / "capture-time-scales.json")}
    assert len(installed["layers"]) == shape["layers"] == len(capture)
    for layer in installed["layers"]:
        i = int(layer["name"].split("layers.")[1].split(".")[0])
        v = scales[f"{i}.v"] if mixed else 1.0
        assert (
            layer["k_scale"] == 1.0
            and layer["q_scale"] == 1.0
            and layer["v_scale"] == v
        )
        assert capture[layer["name"]]["host"] == v and capture[layer["name"]][
            "gpu"
        ] == fp32(v)
        assert layer["impl"] == (
            "FlashAttentionImpl" if arm == "native_fa3" else "FlashInferImpl"
        )
    if fmt == "awq":
        assert len(installed["packed"]) == 144 and installed["weights"] is None
        assert all(
            x["method"] == "AutoAWQMarlinLinearMethod"
            and x["kernel"] == "MacheteLinearKernel"
            and x["weight_bits"] == 4
            for x in installed["packed"]
        )
    else:
        assert not installed["packed"]
        w = installed["weights"]
        assert (
            w["format"] == "fp16"
            and len(w["linear_modules"]) == 144
            and all(x["dtype"] == "torch.float16" for x in w["parameters"])
        )
    q = []
    for order in range(2):
        d = s / f"qualification-{order}"
        rows = load(d / "steps.json")
        graphs = load(d / "graphs.json")
        receipt = load(d / "receipt.json")
        assert (
            len(rows) == receipt["steps"]
            and receipt["preemption_events"]
            == sum(len(x["preempted_request_ids"]) for x in rows)
            == 0
        )
        assert len(graphs) >= 32 and len({x["sha256"] for x in graphs}) >= 3
        for g in graphs:
            assert (
                hashlib.sha256(
                    json.dumps(g["values"], sort_keys=True).encode()
                ).hexdigest()
                == g["sha256"]
            )
        cohorts = [len(x["new_request_ids"]) for x in rows if x["new_request_ids"]]
        assert sum(cohorts) == 32
        checks = []
        if mixed:
            checks = [load(f) for f in sorted((d / "layers").glob("call-*.json"))]
            assert (
                len(checks) == receipt["oracle_checks"]
                and len({x["layer"] for x in checks}) == shape["layers"]
            )
            for x in checks:
                assert (
                    x["routing_equivalence_passed"]
                    and x["forward_placement_verified"]
                    and x["finite"]
                )
                assert (
                    x["output_sha256"] == x["trusted_sha256"]
                    and x["lse_sha256"] == x["trusted_lse_sha256"]
                )
            assert any(x["decode_tokens_checked"] > 0 for x in checks) and any(
                x["decode_tokens_checked"] == 0 for x in checks
            )
        q.append(
            {
                "steps": len(rows),
                "graph_replays": len(graphs),
                "distinct_graph_inputs": len({x["sha256"] for x in graphs}),
                "admission_cohorts": cohorts,
                "oracle_checks": len(checks),
                "storage": storage(load(d / "storage.json"), mixed, shape),
            }
        )
    events = json.loads(gzip.decompress((s / "profile/trace.json.gz").read_bytes()))[
        "traceEvents"
    ]
    weight = joined(events, {"_C::machete_mm"} if fmt == "awq" else {"aten::mm"})
    assert weight["calls"] >= 144
    attn = joined(events, {"_vllm_fa3_C::fwd"}) if arm == "native_fa3" else None
    if arm == "native_fa3":
        assert (
            "Using FlashAttention version 3" in (r / "server.log").read_text()
            and p["engine"]["attention_config"]["flash_attn_version"] == 3
        )
    cells = {}
    for label in [
        "idle",
        "closed-8",
        "closed-32",
        "open-probe",
        "observer-timing",
        "observer-overhead",
    ]:
        a = load(r / f"{label}-audit.json")
        assert a["passed"]
        client = load(r / f"{label}-client.json")
        su = client["summary"]
        rec = load(s / label / "receipt.json")
        assert su["counts"]["completed"] == su["offered"] and all(
            v == 0 for k, v in su["counts"].items() if k != "completed"
        )
        mem = rec["memory"]
        assert (
            mem["peak_reserved_since_reset"] <= p["torch_allocator_reserved_cap_bytes"]
            and mem["cuda_device_used_bytes"] <= p["total_driver_envelope_bytes"]
        )
        metrics = load(r / f"{label}-server-metrics.json")
        cells[label] = {
            "offered": su["offered"],
            "throughput": su["request_throughput_per_s"],
            "latency": su["metrics_conditional_on_valid_completion"],
            "worker_steps": rec["steps"],
            "preemption_events": rec["preemption_events"],
            "peak_reserved_bytes": mem["peak_reserved_since_reset"],
            "sampled_driver_bytes": mem["cuda_device_used_bytes"],
            "metrics_capture_sha256": sha(r / f"{label}-server-metrics.json"),
        }
    return {
        "format": fmt,
        "arm": arm,
        "source_sha": p["source_sha"],
        "backend_sources": p["backend_sources"],
        "cache_bytes": p["engine"]["kv_cache_memory_bytes"],
        "production_passed": True,
        "quality_passed": False,
        "controlled_parity_passed": False,
        "worker_identity_sha256": sha(s / "worker-identity.json"),
        "protocol_sha256": sha(s / "inputs/protocol.json"),
        "qualifications": q,
        "weight_dispatch": weight,
        "fa3_dispatch": attn,
        "pilot_cells_excluded_from_confirmation": cells,
        "observer_throughput_relative_difference": cells["observer-timing"][
            "throughput"
        ]
        / cells["observer-overhead"]["throughput"]
        - 1,
        "reader_sha256": sha(Path(__file__)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--format", required=True)
    p.add_argument("--prefix")
    p.add_argument("--tag", default="qualification-refresh")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = {
        arm: audit(a.root, a.format, arm, a.tag, a.prefix)
        for arm in ["candidate", "native_fi", "native_fa3"]
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        "Production reader passed for all three arms; quality/parity remain separate."
    )


if __name__ == "__main__":
    main()
