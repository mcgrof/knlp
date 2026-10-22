# SPDX-License-Identifier: MIT
"""Validate a separately frozen Qwen3 worker protocol and launch its HTTP arm.

This launcher provides no qualification on its own. Reference selection,
worker numerical checks, quality certificates and serving traces have
separate receipts. The caller must finish those gates before timing.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

from tools.kv.native.graph import ExecutionVariant
from .production_resources import POLICY as RESOURCE_POLICY
from .production_plan import PILOT_RULES

ARMS = ("native_fi", "AF", "native_fa3", "SD")
MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SOURCES = {
    "vllm": "e5c8d87fd6bbf9cdedf266d160890599d938abb0",
    "flashinfer": "800a550f8b736e49f8cfa136b82bfaf6d9d32711",
}
GEOMETRY = {
    "layers": 36,
    "query_heads": 32,
    "kv_heads": 8,
    "head_dim": 128,
    "block_size": 16,
}
NUMERICAL_BOUNDS = {
    "attention": {
        "relative_l2": 0.02,
        "normalized_maximum_error": 0.05,
        "maximum_absolute_error": 0.001,
    },
    "graph": {
        "relative_l2": 0.001,
        "normalized_maximum_error": 0.003,
        "maximum_absolute_error": 0.001,
    },
}
SUPPORT_POLICY = "SD may remain unqualified after a documented startup/numerical failure before any quality item; keep all four certificate slots and their original alpha allocation"
SCALE_POLICIES = {"AF": "fixed", "SD": "per_token_head"}
DYNAMIC_WRITER = {
    "format": "e4m3fn",
    "absmax_floor_fp32": 1e-12,
    "scale_reciprocal_fp32": 0.0022321429569274187,
    "normalization": "float32_div_rn",
    "rounding": "nearest_even",
    "environment": "VLLM_FP8_PER_TOKEN_HEAD_REFERENCE",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(inputs):
    """Verify the locally downloaded official revision before model execution."""
    from huggingface_hub import snapshot_download

    model = Path(
        snapshot_download(repo_id=MODEL, revision=REVISION, local_files_only=True)
    )
    official = json.loads((inputs / "official-weights.json").read_text())
    if official["model"] != MODEL or official["revision"] != REVISION:
        raise ValueError("Official checkpoint identity differs")
    actual = {str(p.relative_to(model)): sha(p) for p in model.rglob("*.safetensors")}
    if not actual or actual != {k: v["sha256"] for k, v in official["files"].items()}:
        raise ValueError("Checkpoint weight bytes differ from the official revision")
    metadata = json.loads((inputs / "official-model-files.json").read_text())
    if metadata["model"] != MODEL or metadata["revision"] != REVISION:
        raise ValueError("Official model-metadata identity differs")
    if not {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    }.issubset(metadata["files"]):
        raise ValueError("Official model configuration/index files were not frozen")
    for name, record in metadata["files"].items():
        if Path(name).name != name:
            raise ValueError("Unexpected model metadata filename")
        raw = (model / name).read_bytes()
        actual_digest = (
            hashlib.sha256(raw).hexdigest()
            if record["algorithm"] == "sha256"
            else (
                hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest()
                if record["algorithm"] == "git_blob_sha1"
                else None
            )
        )
        if (
            actual_digest is None
            or len(raw) != record["bytes"]
            or actual_digest != record["digest"]
        ):
            raise ValueError(
                "Model metadata bytes differ from official pinned objects: " + name
            )
    config = json.loads((model / "config.json").read_text())
    if any(
        config[k] != v
        for k, v in {
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "model_type": "qwen3",
        }.items()
    ):
        raise ValueError("Pinned model geometry differs")
    files = {str(p.relative_to(model)): sha(p) for p in model.rglob("*.json")}
    return model, {
        "model": MODEL,
        "revision": REVISION,
        "weights_sha256": actual,
        "configuration_and_tokenizer_json_sha256": files,
    }


def fp32(value):
    rounded = struct.unpack("f", struct.pack("f", value))[0]
    if not math.isfinite(rounded) or rounded <= 0:
        raise ValueError("Scale must be finite and positive in FP32")
    return rounded


def transferred_scales(record):
    """Preserve the reference calibration; explicitly canonicalize its wire type."""
    if (
        record["model"] != MODEL
        or record["revision"] != REVISION
        or len(record["scales"]) != 36
    ):
        raise ValueError("Unqualified calibration checkpoint or geometry")
    return {
        f"{layer}.{component.lower()}": fp32(row[component])
        for layer, row in enumerate(record["scales"])
        for component in ("K", "V")
    }


def variant(arm):
    if arm not in ARMS:
        raise ValueError("Unknown worker arm")
    return ExecutionVariant(SOURCES, arm == "AF", True)


def environment(arm):
    return {
        **variant(arm).environment({}),
        "VLLM_COMPACT_PREFILL": str(int(arm == "AF")),
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_KV_CACHE_LAYOUT": "NHD",
        "VLLM_FP8_PER_TOKEN_HEAD_REFERENCE": str(int(arm == "SD")),
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "360",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "0",
        "KNLP_NATIVE_OBSERVER": "off",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_ALLOC_CONF": "backend:native",
        "OMP_NUM_THREADS": "1",
    }


def validate(protocol, data, calibration):
    if protocol["schema"] != 2:
        raise ValueError("Worker protocol requires the mixed-scale schema")
    if protocol["symmetric_support_policy"] != SUPPORT_POLICY:
        raise ValueError("Prospective symmetric support-gap policy differs")
    if protocol["serving_pilot_rules"] != PILOT_RULES:
        raise ValueError("Prospective native-only serving pilot rules differ")
    if protocol["resource_policy"] != RESOURCE_POLICY:
        raise ValueError("Common driver-memory and CPU resource policy differs")
    if protocol["numerical_bounds"] != NUMERICAL_BOUNDS:
        raise ValueError("Frozen numerical bounds differ")
    if protocol["backend_sources"] != SOURCES or protocol["geometry"] != GEOMETRY:
        raise ValueError("Unqualified backend sources or model geometry")
    if (
        data["model"] != MODEL
        or data["revision"] != REVISION
        or data["dtype"] != "float16"
    ):
        raise ValueError("Only the pinned dense FP16 Qwen3 checkpoint is admitted")
    if protocol["scales"] != transferred_scales(calibration):
        raise ValueError("Worker scales differ from the frozen FP32 transfer")
    if (
        protocol["arms"] != list(ARMS)
        or protocol["query_dtype"] != "float16"
        or protocol["scale_policy"] != SCALE_POLICIES
        or protocol["dynamic_writer"] != DYNAMIC_WRITER
    ):
        raise ValueError("Worker requires native Q, fixed AF and per-token/head SD")
    if protocol["serving_qualification_route"] != "aggregate_math":
        raise ValueError("This study preselected the aggregate-math serving route")
    if protocol["torch"] != "2.11.0+cu129" or protocol["transformers"] != "5.5.3":
        raise ValueError("The worker package recipe differs from the current installer")
    if protocol["persistent_budget_bytes"] != 40 * 1024**3:
        raise ValueError("Require the declared common 40-GiB persistent-cache budget")
    if protocol["equal_slot_blocks"] != 17000:
        raise ValueError(
            "Equal-slot geometry probe must allocate 17000 blocks per layer"
        )
    engine = protocol["engine"]
    required = {
        "dtype": "float16",
        "tensor_parallel_size": 1,
        "max_model_len": 10240,
        "max_num_seqs": 96,
        "max_num_batched_tokens": 65536,
        "block_size": 16,
        "enforce_eager": False,
        "disable_log_stats": False,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "calculate_kv_scales": False,
        "distributed_executor_backend": "uni",
        "seed": 20261020,
        "stream_interval": 1,
        "generation_config": "vllm",
    }
    if any(engine.get(key) != value for key, value in required.items()):
        raise ValueError("Worker engine settings differ from the declared recipe")
    # Native/default optimized dispatch remains enabled. Mixed-dtype K/V has
    # its own backend support restriction; do not impose it on the controls.
    if any(
        k in engine
        for k in ("attention_config", "worker_cls", "num_gpu_blocks_override")
    ):
        raise ValueError("Arm-specific backend and worker settings are derived here")
    capture = engine["compilation_config"]
    if capture != {
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 8, 32, 44, 45, 59, 60, 96],
    }:
        raise ValueError("Graph policy differs")
    allowance = protocol["scale_allocator_allowance_bytes"]
    kv_allowance = protocol["kv_allocator_allowance_bytes"]
    if (
        allowance != 36 * 3 * 512
        or kv_allowance != 36 * 1024**2
        or engine["kv_cache_memory_bytes"]
        != protocol["persistent_budget_bytes"] - allowance - kv_allowance
    ):
        raise ValueError(
            "KV payload budget must leave room for live KV and scale allocation rounding"
        )
    if protocol["torch_allocator_reserved_cap_bytes"] != 68 * 1024**3:
        raise ValueError("The common Torch allocator ceiling differs")
    counts = {
        k: sum(r["kind"] == k for r in data["samples"]["confirmation"])
        for k in ("nll", "retrieval", "gsm8k")
    }
    if counts != {"nll": 128, "retrieval": 400, "gsm8k": 1319}:
        raise ValueError("Worker panel sizes differ from the reference protocol")
    if (
        protocol["quality_concurrency"] != 32
        or data["gsm_parser"] != "last_numeric_answer_label_v1"
        or protocol["eos_token_ids"] != [151645, 151643]
    ):
        raise ValueError("Quality cohort size or frozen parser differs")
    certificates = protocol["certificates"]
    if certificates != {
        "family_alpha": 0.025,
        "certificates_per_objective": 4,
        "native_accuracy_floor": 0.5,
        "gsm_gross_loss_margin": 0.02,
        "retrieval_gross_loss_margin": 0.05,
        "nll_relative_mean_margin": 0.01,
        "gsm_net_margin": -0.01,
        "bootstrap_seed": 20261020,
        "bootstrap_resamples": 100000,
    }:
        raise ValueError("Prospective certificate family or endpoint rules differ")


def read_inputs(inputs, digest):
    protocol_path = inputs / "protocol.json"
    if sha(protocol_path) != digest:
        raise ValueError("Frozen production protocol hash differs")
    protocol = json.loads(protocol_path.read_text())
    for name, expected in protocol["inputs_sha256"].items():
        if Path(name).name != name or sha(inputs / name) != expected:
            raise ValueError("Frozen worker input file differs: " + name)
    if not {
        "inputs.json.gz",
        "reference-scales.json",
        "official-weights.json",
        "official-model-files.json",
    }.issubset(protocol["inputs_sha256"]):
        raise ValueError("Required worker inputs were not frozen")
    tokenizer_files = protocol["tokenizer_files_sha256"]
    if not tokenizer_files:
        raise ValueError("The original tokenizer files must be frozen")
    for name, expected in tokenizer_files.items():
        if Path(name).name != name or sha(inputs / "tokenizer" / name) != expected:
            raise ValueError("Worker tokenizer bytes differ: " + name)
    data = json.loads(gzip.decompress((inputs / "inputs.json.gz").read_bytes()))
    calibration = json.loads((inputs / "reference-scales.json").read_text())
    if calibration["inputs_sha256"] != sha(inputs / "inputs.json.gz"):
        raise ValueError("Transferred calibration belongs to different inputs")
    validate(protocol, data, calibration)
    return protocol, data


def verify_runner(protocol):
    source = Path(__file__).resolve().parents[3]
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--", "tools/kv"], text=True
    ).strip()
    if actual != protocol["source_sha"] or dirty:
        raise ValueError("Worker requires the committed, clean frozen knlp source")
    for relative, digest in protocol["runner_files_sha256"].items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source / "tools/kv") or sha(path) != digest:
            raise ValueError("Runtime source bytes differ: " + relative)
    for name in ("production.py", "production_worker.py", "paged_memory.py"):
        if "tools/kv/decision/" + name not in protocol["runner_files_sha256"]:
            raise ValueError("Production runner file was not frozen")


def prepare(install, inputs, output, arm, digest, allocation="capacity"):
    from tools.kv.native.october_setup import BASE, WHEEL_SHA

    if allocation not in ("equal_slots", "capacity"):
        raise ValueError("Unknown allocation phase")
    protocol, data = read_inputs(inputs, digest)
    verify_runner(protocol)
    definition = variant(arm)
    setup = json.loads((install / "setup.json").read_text())
    if (
        setup["recipe"] != "decision-http"
        or setup["transformers"] != protocol["transformers"]
        or {name: pin[1] for name, pin in setup["sources"].items()} != SOURCES
        or setup["compiled_base"] != BASE
        or setup["wheel_sha256"] != WHEEL_SHA
    ):
        raise ValueError(
            "Installer recipe, dependency sources or compiled wheel differs"
        )
    definition.verify_sources({name: install / name for name in SOURCES})
    output.mkdir(parents=True, exist_ok=False)
    # An empty observer log must remain distinguishable from a missing capture
    # when every offered request disconnects before ASGI entry.
    (output / "asgi-events.jsonl").touch(exist_ok=False)
    model, model_identity = checkpoint(inputs)
    (output / "checkpoint-identity.json").write_text(
        json.dumps(model_identity, indent=2) + "\n"
    )
    engine = dict(protocol["engine"])
    if allocation == "equal_slots":
        engine["num_gpu_blocks_override"] = protocol["equal_slot_blocks"]
    engine["worker_cls"] = "tools.kv.decision.production_worker.ProductionWorker"
    engine["attention_config"] = (
        {"backend": "FLASH_ATTN", "flash_attn_version": 3}
        if arm == "native_fa3"
        else (
            {"backend": "TRITON_ATTN"}
            if arm == "SD"
            else {"backend": "FLASHINFER", "disable_flashinfer_q_quantization": True}
        )
    )
    settings = environment(arm)
    env = {**os.environ, **settings}
    env.update(
        {
            "KNLP_DECISION_INPUTS": str(inputs),
            "KNLP_DECISION_OUTPUT": str(output),
            "KNLP_DECISION_INSTALL": str(install),
            "KNLP_DECISION_ARM": arm,
            "KNLP_DECISION_ALLOCATION": allocation,
            "KNLP_DECISION_PROTOCOL_SHA256": digest,
            "KNLP_DECISION_HTTP_EVENTS": str(output / "asgi-events.jsonl"),
            "KNLP_NATIVE_ALLOC_TRACE": str(output / "allocation.jsonl"),
            "KNLP_NATIVE_TORCH_LIMIT_BYTES": str(
                protocol["torch_allocator_reserved_cap_bytes"]
            ),
        }
    )
    cache = (
        "auto,fp8_e4m3"
        if arm == "AF"
        else "fp8_per_token_head" if arm == "SD" else "auto"
    )
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--served-model-name",
        "study",
        "--model",
        str(model),
        "--tokenizer",
        str(inputs / "tokenizer"),
        "--revision",
        REVISION,
        "--tokenizer-revision",
        REVISION,
        "--kv-cache-dtype",
        cache,
        "--no-enable-log-requests",
        "--middleware",
        "tools.kv.decision.serving.AcceptedRequests",
    ]
    for key, value in engine.items():
        flag = key.replace("_", "-")
        if key == "disable_log_stats" and value is False:
            continue
        if isinstance(value, bool):
            argv.append("--" + ("" if value else "no-") + flag)
        elif isinstance(value, dict):
            argv.extend(["--" + flag, json.dumps(value)])
        elif value is not None:
            argv.extend(["--" + flag, str(value)])
    (output / "launch.json").write_text(
        json.dumps(
            {
                "argv": argv,
                "arm": arm,
                "allocation": allocation,
                "protocol_sha256": digest,
                "environment": settings,
                "source_sha": protocol["source_sha"],
            },
            indent=2,
        )
        + "\n"
    )
    return argv, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument(
        "--allocation", choices=("equal_slots", "capacity"), default="capacity"
    )
    args = parser.parse_args()
    argv, env = prepare(
        args.install.resolve(),
        args.inputs.resolve(),
        args.output.resolve(),
        args.arm,
        args.protocol_sha256,
        args.allocation,
    )
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
