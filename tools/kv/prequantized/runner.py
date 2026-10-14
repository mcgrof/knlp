"""Six-stage packed-weight cached-decode quality screen.

Run from the package directory (see README) to avoid the repository's
unrelated `kernels` package shadowing Hugging Face's kernel loader.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Load the external kernel loader before exposing the repository package.
try:
    import kernels
except ImportError:
    kernels = None
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from tools.kv.prequantized.cache import (
    AttentionReadProof,
    Observer,
    make_cache,
    qdq,
    reference_e4m3,
)
from tools.kv.prequantized.contract import (
    ARMS,
    GATES,
    append_result,
    completed_rows,
    digest,
    paired_nll,
    paired_retrieval,
    pinned,
    read,
    validate_manifest,
    validate_matrix,
    write_once,
)
from tools.kv.prequantized.fixtures import parse_final, prepare_samples
from tools.kv.prequantized.recipes import manifest_recipe, recipe
from tools.kv.prequantized.validation import (
    expected_samples,
    row_provenance,
    validate_execution_manifest,
    validate_gates,
    validate_results,
    validate_row,
    validate_scales,
)


def source_sha():
    root = Path(__file__).resolve().parents[3]
    receipt = root / ".source-receipt.json"
    if receipt.exists():
        archived = read(receipt)
        for name, expected in archived["files"].items():
            path = (root / name).resolve()
            path.relative_to(root)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"archived source was modified: {name}")
        return pinned(archived["source_sha"])
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--error-unmatch",
            *(
                str(p.relative_to(root))
                for p in sorted((root / "tools/kv/prequantized").glob("*.py"))
            ),
        ],
        capture_output=True,
        check=False,
    )
    if tracked.returncode:
        raise ValueError("commit all runner modules before recording source provenance")
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "tools/kv/prequantized",
            "tools/kv/fp8_failure/statistics.py",
            "tests/test_prequantized_kv.py",
            "tests/test_deferred_kv.py",
            "tests/test_kv_replay_drift.py",
        ],
        check=False,
    )
    if dirty.returncode:
        raise ValueError("commit the runner before recording source provenance")
    return subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def version_receipt():
    versions = {}
    for name in (
        "torch",
        "transformers",
        "triton",
        "autoawq",
        "kernels",
        "datasets",
        "accelerate",
        "numpy",
        "pyarrow",
        "scipy",
        "safetensors",
        "tokenizers",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def threads():
    maximum = os.cpu_count() or 1
    path = Path("/sys/fs/cgroup/cpu.max")
    if path.exists():
        quota, period = path.read_text().split()
        if quota != "max":
            maximum = min(maximum, max(1, int(quota) // int(period)))
    torch.set_num_threads(min(16, maximum))
    return torch.get_num_threads()


def prepare(args):
    from transformers import AutoTokenizer

    generator_source = source_sha()
    matrix = read(args.matrix)
    validate_matrix(matrix)
    revision = pinned(args.revision)
    dataset_revision = pinned(args.dataset_revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    samples, inventory, cal_ids, eval_ids = prepare_samples(
        tokenizer, matrix, args.model, dataset_revision, args.dataset
    )
    if source_sha() != generator_source:
        raise ValueError(
            "source changed during fixture generation; prepare a new manifest"
        )
    config = next(x for x in matrix["models"] if x["id"] == args.model)
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "dataset": args.dataset,
        "dataset_revision": dataset_revision,
        "source_sha": generator_source,
        "matrix_hash": digest(matrix),
        "matrix": matrix,
        "activation_dtype": config["activation_dtype"],
        "max_output_tokens": config["max_output_tokens"],
        "chat_template_hash": digest(tokenizer.chat_template),
        "samples": samples,
        "sample_hash": digest(samples),
        "inventory": inventory,
        "calibration_article_ids": cal_ids,
        "evaluation_article_ids": eval_ids,
        "kernel_revision": (
            pinned(args.kernel_revision) if args.kernel_revision else None
        ),
        "current_token_semantics": "fresh unrounded K/V; previously stored K/V rounded once; prefill reads fresh K/V",
        "cache_recipe": recipe(),
    }
    validate_manifest(manifest)
    write_once(args.manifest, manifest)
    print(
        json.dumps(
            {
                "stage": "prepare",
                "manifest": str(args.manifest),
                "hash": digest(manifest),
                "inventory": inventory,
            }
        ),
        flush=True,
    )


def load(manifest):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required; CPU or expanded-weight fallback is forbidden"
        )
    kwargs = {
        "revision": manifest["model_revision"],
        "dtype": getattr(torch, manifest["activation_dtype"]),
        "device_map": {"": 0},
        "attn_implementation": (
            "sdpa" if manifest["model"].startswith("Qwen/") else "eager"
        ),
    }
    if manifest["model"].startswith("Qwen/"):
        import transformers.activations as activations

        # AutoAWQ imports this removed alias while loading its calibration
        # modules. Qwen's shipped SiLU activation never calls this class.
        if not hasattr(activations, "PytorchGELUTanh"):

            class LegacyGELUTanh(torch.nn.GELU):
                def __init__(self):
                    super().__init__(approximate="tanh")

            activations.PytorchGELUTanh = LegacyGELUTanh
        import awq.modules.linear.gemm as awq_gemm

        if awq_gemm.awq_ext is not None or not awq_gemm.TRITON_AVAILABLE:
            raise RuntimeError(
                "this AWQ receipt requires the pinned Triton packed GEMM"
            )
        original_gemm = awq_gemm.awq_gemm_triton
        if not getattr(original_gemm, "_single_partition", False):

            def single_partition(
                input, qweight, scales, qzeros, split_k_iters, *args, **kw
            ):
                return original_gemm(input, qweight, scales, qzeros, 1, *args, **kw)

            single_partition._single_partition = True
            awq_gemm.awq_gemm_triton = single_partition
    if manifest["model"].startswith("openai/"):
        if kernels is None or not hasattr(kernels, "get_kernel"):
            raise RuntimeError(
                "external Hugging Face kernels loader is unavailable or shadowed"
            )
        original_kernel = kernels.get_kernel

        def pinned_kernel(repo_id, *args, **kw):
            if repo_id == "kernels-community/triton_kernels":
                kw["revision"] = pinned(manifest["kernel_revision"])
            else:
                raise RuntimeError(f"unrecorded kernel repository: {repo_id}")
            return original_kernel(repo_id, *args, **kw)

        kernels.get_kernel = pinned_kernel
        import transformers.integrations.mxfp4 as integration

        def forbidden_expansion(*args, **kw):
            raise RuntimeError("persistent MXFP4 expansion was requested")

        integration.convert_moe_packed_tensors = forbidden_expansion
        kwargs["quantization_config"] = Mxfp4Config(dequantize=False)
    model = AutoModelForCausalLM.from_pretrained(manifest["model"], **kwargs).eval()
    if manifest["model"].startswith("Qwen/") and model.config.hidden_act != "silu":
        raise RuntimeError("AWQ import compatibility is qualified only for SiLU Qwen")
    if any(type(module).__name__ == "LegacyGELUTanh" for module in model.modules()):
        raise RuntimeError("import compatibility alias must not enter the model")
    if getattr(model.config.quantization_config, "dequantize", False):
        raise RuntimeError("loader enabled persistent dequantization")
    model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"], revision=manifest["tokenizer_revision"]
    )
    if digest(tokenizer.chat_template) != manifest["chat_template_hash"]:
        raise ValueError("chat template changed")
    return model, tokenizer


def forward(model, ids, cache=None):
    tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
    output = model(
        input_ids=tokens, past_key_values=cache, use_cache=True, logits_to_keep=1
    )
    if not torch.isfinite(output.logits).all():
        raise RuntimeError("nonfinite model logits")
    return output


def tensors(value, depth=0):
    if isinstance(value, torch.Tensor):
        return [value]
    if depth >= 5 or value is None or isinstance(value, (int, str, float, bool)):
        return []
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (tuple, list)):
        children = value
    else:
        children = vars(value).values() if hasattr(value, "__dict__") else []
    return [tensor for child in children for tensor in tensors(child, depth + 1)]


def packed_receipt(model, manifest):
    modules = []
    storages = {}
    for name, module in model.named_modules():
        packed = []
        if hasattr(module, "qweight"):
            if module.qweight.dtype != torch.int32:
                raise RuntimeError("AWQ qweight is not packed int32")
            packed = [module.qweight, module.qzeros, module.scales]
        elif type(module).__name__ == "Mxfp4GptOssExperts":
            packed = (
                tensors(module.gate_up_proj)
                + tensors(module.down_proj)
                + tensors(module.gate_up_proj_precision_config)
                + tensors(module.down_proj_precision_config)
            )
            if not any(x.dtype == torch.uint8 for x in packed):
                raise RuntimeError(
                    "MXFP4 expert payload does not expose uint8 packed storage"
                )
        if packed:
            rows = []
            for tensor in packed:
                if tensor.device.type != "cuda":
                    raise RuntimeError(
                        "packed weight offload changes the selected workload"
                    )
                storage = tensor.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
                rows.append(
                    {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "bytes": tensor.numel() * tensor.element_size(),
                    }
                )
            modules.append(
                {
                    "name": name,
                    "class": type(module).__module__ + "." + type(module).__name__,
                    "packed_buffers": rows,
                }
            )
    expected = model.config.num_hidden_layers * (
        7 if manifest["model"].startswith("Qwen/") else 1
    )
    if len(modules) != expected:
        raise RuntimeError(f"packed module count {len(modules)} != expected {expected}")
    attention = model.model.layers[0].self_attn
    if (
        attention.q_proj.weight.dtype != getattr(torch, manifest["activation_dtype"])
        if hasattr(attention.q_proj, "weight")
        else attention.q_proj.scales.dtype
        != getattr(torch, manifest["activation_dtype"])
    ):
        raise RuntimeError("attention activation storage dtype mismatch")
    quant = model.config.quantization_config
    return {
        "modules": modules,
        "packed_resident_bytes": sum(storages.values()),
        "quantization_config": quant.to_dict() if hasattr(quant, "to_dict") else quant,
        "activation_dtype": manifest["activation_dtype"],
        "attention_backend": model.config._attn_implementation,
        "weights_runtime": "packed",
        "activation_function": model.config.hidden_act,
        "import_compatibility_alias_in_model": False,
        "weight_kernel_settings": (
            {"awq_triton_split_k": 1}
            if manifest["model"].startswith("Qwen/")
            else {"mxfp4": "pinned packed Triton kernel"}
        ),
    }


def error(a, b):
    difference = (a.float() - b.float()).abs()
    return {
        "max_logit_error": float(difference.max()),
        "mean_logit_error": float(difference.mean()),
        "mean_logsoftmax_error": float(
            (a.float().log_softmax(-1) - b.float().log_softmax(-1)).abs().mean()
        ),
    }


def require_packed_dispatch(model, events):
    if model == "Qwen/Qwen3-8B-AWQ":
        matched = any("awq_gemm_kernel" in name for name in events)
    elif model == "openai/gpt-oss-20b":
        matched = any("_matmul_ogs_" in name and "mxfp4" in name for name in events)
    else:
        raise ValueError("no packed-dispatch contract for this model")
    if not matched:
        raise RuntimeError("model-specific packed-weight decode kernel is absent")


def gate_path(args, gate):
    return Path(args.output) / "gates" / f"{gate}.json"


def gate(args, name, data, provenance):
    write_once(
        gate_path(args, name),
        {"gate": name, "status": "passed", "provenance": provenance, **data},
    )
    print(json.dumps({"gate": name, "status": "passed"}), flush=True)


def provenance(manifest):
    return {
        "manifest_hash": digest(manifest),
        "source_sha": source_sha(),
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "versions": version_receipt(),
        "kernel_revision": manifest["kernel_revision"],
        "weight_kernel_settings": (
            {"awq_triton_split_k": 1}
            if manifest["model"].startswith("Qwen/")
            else {"mxfp4": "pinned packed Triton kernel"}
        ),
        "backend_release_commits": {
            "torch": torch.version.git_version,
            "transformers_4.57.6": "753d61104116eefc8ffc977327b441ee0c8d599f",
            "autoawq_0.2.9": "88e4c76b20755db275574e6a03c83c84ba3bece5",
            "triton_3.4.0": "c817b9b63d40ead1ed023b7663f5ea14f676f4bc",
        },
    }


def require_gates(args, prov):
    validate_gates({name: read(gate_path(args, name)) for name in GATES}, prov)


def require_sensitivity(args, manifest, prov):
    if manifest["matrix"].get("retrieval_family") != "retrieval_v3_deferred_query":
        return
    receipt = read(Path(args.output) / "sensitivity.json")
    if (
        receipt.get("status") != "passed"
        or receipt.get("provenance") != prov
        or receipt.get("criterion") != manifest["sensitivity_criterion"]
    ):
        raise ValueError("deferred-query workload sensitivity is unqualified")
    rows = [
        json.loads(line)
        for line in (Path(args.output) / "sensitivity-rows.jsonl")
        .read_text()
        .splitlines()
    ]
    if digest(rows) != receipt["rows_hash"] or any(
        row["provenance"] != prov for row in rows
    ):
        raise ValueError("sensitivity evidence hash or provenance mismatch")
    from tools.kv.prequantized.sensitivity import sensitivity_verdict

    recomputed = sensitivity_verdict(rows, manifest["sensitivity_criterion"])
    if recomputed["status"] != "passed" or recomputed["groups"] != receipt["groups"]:
        raise ValueError("sensitivity verdict differs from its raw controls")


@torch.inference_mode()
def generate(model, tokenizer, sample, manifest, observer, catalogue_cache=None):
    cache = (
        catalogue_cache
        if catalogue_cache is not None
        else make_cache(model.config, observer)
    )
    split = sample.get("catalogue_tokens")
    if split:
        if catalogue_cache is None:
            forward(model, sample["token_ids"][:split], cache)
        # The query is absent from prefill. Its logits must read stored facts.
        output = forward(model, sample["token_ids"][split:], cache)
    else:
        output = forward(model, sample["token_ids"], cache)
    generated = []
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    stopped = False
    for _ in range(manifest["max_output_tokens"]):
        token = int(output.logits[0, -1].argmax())
        generated.append(token)
        if token in eos:
            stopped = True
            break
        output = forward(model, [token], cache)
    final, malformed = parse_final(tokenizer, generated, manifest["model"])
    return {
        "generated_ids": generated,
        "output": tokenizer.decode(generated, skip_special_tokens=False),
        "final": final,
        "malformed": malformed,
        "truncated": not stopped,
        "correct": bool(stopped and not malformed and final == sample.get("answer")),
        "output_tokens": len(generated),
    }


@torch.inference_mode()
def doctor(args, manifest):
    model, tokenizer = load(manifest)
    prov = provenance(manifest)
    sample = next(x for x in manifest["samples"] if x["kind"] == "nll")
    ids = sample["token_ids"][:300]
    packed = packed_receipt(model, manifest)
    warm = forward(model, ids[:32])
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        forward(model, [ids[32]], warm.past_key_values)
    events = [item.key for item in profiler.key_averages()]
    kernels_seen = [
        item
        for item in events
        if any(
            word in item.lower()
            for word in ("awq", "mxfp", "matmul", "gemm", "ogs", "triton")
        )
    ]
    require_packed_dispatch(manifest["model"], kernels_seen)
    gate(
        args,
        "T00",
        {
            **packed,
            "kernel_dispatch": kernels_seen,
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "peak_vram": torch.cuda.max_memory_allocated(),
            "threads": torch.get_num_threads(),
        },
        prov,
    )
    baseline = forward(model, ids[:127])
    native = forward(model, [ids[127]], baseline.past_key_values).logits
    repeated = forward(model, ids[:127])
    repeated_logits = forward(model, [ids[127]], repeated.past_key_values).logits
    # Predeclare tolerances before any quantized arm; bypass should be exact.
    tolerance = 0.0
    observer = Observer()
    cache = make_cache(model.config, observer)
    with AttentionReadProof(model, observer):
        forward(model, ids[:127], cache)
        hooked = forward(model, [ids[127]], cache).logits
    bypass_error = error(hooked, native)
    repeat_error = error(repeated_logits, native)
    if bypass_error["max_logit_error"] > tolerance:
        raise RuntimeError("bypass mismatch")
    if repeat_error["max_logit_error"] > tolerance:
        raise RuntimeError("native same-path repeat noise exceeds frozen limit")
    gate(
        args,
        "T01",
        {
            "tolerance_max_logit": tolerance,
            "bypass": bypass_error,
            "repeat_baseline": repeat_error,
            "tolerance_repeat_max_logit": tolerance,
            "finite_logits_required": True,
        },
        prov,
    )
    values = [-500, -448, -0.003, 0, 2**-10, 3 * 2**-10, 1.0625, 1.1875, 447, 500]
    quantized = qdq(torch.tensor(values, device="cuda"), 1).cpu().tolist()
    expected = [reference_e4m3(x) for x in values]
    if quantized != expected:
        raise RuntimeError("FP8 reference mismatch")
    untouched = torch.randn(
        1, 2, 3, 4, device="cuda", dtype=getattr(torch, manifest["activation_dtype"])
    )
    for arm, kind in (("k16_v8", "k"), ("k8_v16", "v")):
        if not torch.equal(
            Observer(arm, {"0.k": 1, "0.v": 1}).transform(0, kind, untouched), untouched
        ):
            raise RuntimeError("unselected cache component changed")
    gate(
        args,
        "T02",
        {
            "values": values,
            "expected": expected,
            "observed": quantized,
            "scale_arithmetic_dtype": "float32",
            "rounding": "nearest_ties_even",
            "unselected_component_unchanged": True,
        },
        prov,
    )
    # A zeroed-past control uses the same fresh-token semantics as bypass.
    zero_observer = Observer()
    zero_cache = make_cache(model.config, zero_observer)
    forward(model, ids[:127], zero_cache)
    for layer in zero_cache.layers:
        layer.keys.zero_()
        layer.values.zero_()
    zero_logits = forward(model, [ids[127]], zero_cache).logits
    control_error = error(zero_logits, native)
    if control_error["max_logit_error"] <= max(
        1e-4, bypass_error["max_logit_error"] * 10
    ):
        raise RuntimeError("zeroed past does not affect logits")
    if len(observer.attention_reads) != model.config.num_hidden_layers or any(
        x < 127 for x in observer.attention_reads.values()
    ):
        raise RuntimeError("attention reads did not cover old cache in every layer")
    gate(
        args,
        "T03",
        {
            "zeroed_past": control_error,
            "bypass_cache": observer.receipt(),
            "write_once": True,
            "current_token_semantics": manifest["current_token_semantics"],
        },
        prov,
    )
    lengths = (
        [127, 128, 129, 255, 256, 257]
        if manifest["model"].startswith("openai/")
        else [127, 257]
    )
    architecture = []
    for length in lengths:
        obs = Observer(cache_recipe=manifest_recipe(manifest)["id"])
        cached = make_cache(model.config, obs)
        forward(model, ids[:length], cached)
        unmodified = forward(model, ids[:length]).past_key_values
        for appended in range(3):
            end = length + appended + 1
            output = forward(model, [ids[end - 1]], cached).logits
            reference = forward(model, [ids[end - 1]], unmodified).logits
            full_history = forward(model, ids[:end]).logits
            measured = error(output, reference)
            states_equal = all(
                torch.equal(layer.keys, original.keys)
                and torch.equal(layer.values, original.values)
                for layer, original in zip(cached.layers, unmodified.layers)
            )
            architecture.append(
                {
                    "prefix_tokens": length,
                    "append": appended + 1,
                    **measured,
                    "unmodified_cached_vs_full_history": error(reference, full_history),
                    "cache_states_equal_to_unmodified": states_equal,
                    "cache_lengths": [layer.keys.shape[-2] for layer in cached.layers],
                }
            )
            if measured["max_logit_error"] > tolerance or not states_equal:
                raise RuntimeError(
                    f"cached/reference architecture mismatch at {length}+{appended+1}"
                )
    gate(
        args,
        "T04",
        {
            "boundary_tests": architecture,
            "layer_types": getattr(model.config, "layer_types", None),
            "attention_sinks": (
                "original backend eager softmax denominator"
                if manifest["model"].startswith("openai/")
                else "not applicable"
            ),
            "tolerance_max_logit": tolerance,
            "reference": "unmodified backend cached decode with identical appends",
            "full_history_replay": "diagnostic; native cached/full discrepancy is reported separately; root cause not established",
        },
        prov,
    )
    formats = []
    for sample in (x for x in manifest["samples"] if x["kind"] == "dev"):
        obs = Observer()
        with AttentionReadProof(model, obs):
            result = generate(model, tokenizer, sample, manifest, obs)
        formats.append({"sample_id": sample["id"], **result})
        if result["malformed"] or result["truncated"]:
            write_once(
                Path(args.output) / "gates" / "T05-failed-fixtures.json", formats
            )
            raise RuntimeError(
                "native development fixture malformed or truncated; revise and freeze cap/template before comparison"
            )
    gate(
        args,
        "T05",
        {
            "fixtures": formats,
            "chat_template_hash": manifest["chat_template_hash"],
            "output_cap": manifest["max_output_tokens"],
        },
        prov,
    )


@torch.inference_mode()
def calibrate(args, manifest):
    prov = provenance(manifest)
    require_gates(args, prov)
    model, _ = load(manifest)
    observer = Observer(calibrate=True, cache_recipe=manifest_recipe(manifest)["id"])
    ids_done = []
    for sample in manifest["samples"]:
        if sample["kind"] != "calibration":
            continue
        length = sample["length"]
        cache = make_cache(model.config, observer)
        with AttentionReadProof(model, observer):
            forward(model, sample["token_ids"][:length], cache)
            for token in sample["token_ids"][length : length + 128]:
                forward(model, [token], cache)
        ids_done.append(sample["id"])
        print(
            json.dumps({"calibrated": sample["id"], "completed": len(ids_done)}),
            flush=True,
        )
    scales = observer.frozen_scales()
    if len(scales) != 2 * model.config.num_hidden_layers:
        raise RuntimeError("calibration did not cover all layers")
    data = {
        "provenance": prov,
        "sample_ids": ids_done,
        "scales": scales,
        "scale_hash": digest(scales),
        "calibration_receipt": observer.receipt(),
        "scale_storage_dtype": "float32",
    }
    # Freeze to FP32 values, not Python double, to match the actual computation.
    data["scales"] = {
        key: float(torch.tensor(value, dtype=torch.float32))
        for key, value in scales.items()
    }
    data["scale_hash"] = digest(data["scales"])
    write_once(args.scales, data)


@torch.inference_mode()
def quality(args, manifest):
    prov = provenance(manifest)
    require_gates(args, prov)
    require_sensitivity(args, manifest, prov)
    scale_file = read(args.scales)
    validate_scales(scale_file, prov, manifest)
    base = prov
    prov = row_provenance(base, manifest, args.arm, scale_file["scale_hash"])
    path = Path(args.output) / "runs" / args.run_id / f"{args.arm}.jsonl"
    completed = completed_rows(path, prov)
    samples = expected_samples(manifest)
    for key, row in completed.items():
        if key not in samples:
            raise ValueError("resume contains an unknown sample")
        validate_row(row, samples[key], prov)
    model, tokenizer = load(manifest)
    for sample in manifest["samples"]:
        if sample["kind"] not in ("nll", "retrieval", "generation") or sample.get(
            "reserve"
        ):
            continue
        if (
            args.length
            and sample["kind"] != "generation"
            and sample["length"] != args.length
        ):
            continue
        if sample["id"] in completed:
            continue
        observer = Observer(
            args.arm, scale_file["scales"], cache_recipe=manifest_recipe(manifest)["id"]
        )
        started = time.monotonic()
        row = {
            "sample_id": sample["id"],
            "provenance": prov,
            "kind": sample["kind"],
            "context_tokens": sample["length"],
            "prompt_hash": sample["prompt_hash"],
            "status": "completed",
            "hardware": torch.cuda.get_device_name(),
            "run_id": args.run_id,
            "seed": 20261013,
        }
        with AttentionReadProof(model, observer):
            if sample["kind"] == "nll":
                length = sample["length"]
                cache = make_cache(model.config, observer)
                forward(model, sample["token_ids"][:length], cache)
                losses = []
                top1_ids = []
                for index in range(128):
                    output = forward(
                        model, [sample["token_ids"][length + index]], cache
                    )
                    target = sample["token_ids"][length + index + 1]
                    losses.append(-output.logits[0, -1].float().log_softmax(-1)[target])
                    top1_ids.append(output.logits[0, -1].argmax())
                stacked = torch.stack(losses)
                if not torch.isfinite(stacked).all():
                    raise RuntimeError("nonfinite cached-decode NLL")
                row.update(
                    {
                        "mean_nll": float(stacked.mean()),
                        "scored_tokens": 128,
                        "token_nll": stacked.cpu().tolist(),
                        "top1_ids": torch.stack(top1_ids).cpu().tolist(),
                        "article_id": sample["article_id"],
                    }
                )
            else:
                row.update(generate(model, tokenizer, sample, manifest, observer))
                if "depth" in sample:
                    row["depth"] = sample["depth"]
        receipt = observer.receipt()
        if len(receipt["attention_reads"]) != model.config.num_hidden_layers:
            raise RuntimeError("missing read instrumentation")
        if any(item["nonfinite"] for item in receipt["quantization"].values()):
            raise RuntimeError("nonfinite quantized cache")
        row.update({"cache_receipt": receipt, "seconds": time.monotonic() - started})
        validate_row(row, sample, prov)
        append_result(path, row, completed)
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "sample_id": sample["id"],
                    "completed": len(completed),
                    "seconds": row["seconds"],
                    "mean_nll": row.get("mean_nll"),
                    "correct": row.get("correct"),
                }
            ),
            flush=True,
        )


def native(args, manifest):
    require_gates(args, provenance(manifest))
    summary = read(Path(args.output) / "summary.json")
    if summary["model_verdict"] != "screen_supported":
        reason = (
            "native performance gate closed: quality preservation was not established"
        )
    else:
        reason = "the qualified Transformers DynamicCache has 16-bit tensors; no maintained packed mixed-cache implementation with identical write-time static scales and fresh-token semantics has been qualified in this runner"
    write_once(
        Path(args.output) / "native.json",
        {
            "status": "unsupported",
            "measured": False,
            "reason": reason,
            "integration_gpu_hours": 0,
            "provenance": provenance(manifest),
        },
    )


def summarize(args, manifest):
    base = read(gate_path(args, "T00"))["provenance"]
    if (
        base["manifest_hash"] != digest(manifest)
        or base["source_sha"] != manifest["source_sha"]
        or base["model"] != manifest["model"]
        or base["model_revision"] != manifest["model_revision"]
    ):
        raise ValueError("gate provenance differs from current manifest")
    require_gates(args, base)
    require_sensitivity(args, manifest, base)
    if not args.scales:
        raise ValueError("summarize requires frozen --scales")
    scale_file = read(args.scales)
    validate_scales(scale_file, base, manifest)
    rows = []
    for path in (Path(args.output) / "runs").glob("*/*.jsonl"):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    by_arm = validate_results(rows, manifest, base, scale_file)
    comparisons = []
    results = []
    full_matrix = True
    for length in (2048, 8192):
        for kind, metric in (("nll", "mean_nll"), ("retrieval", "correct")):
            arms = {
                arm: {
                    key: row[metric]
                    for key, row in entries.items()
                    if row["kind"] == kind and row["context_tokens"] == length
                }
                for arm, entries in by_arm.items()
            }
            expected = 32 if kind == "nll" else 100
            complete = all(len(entries) == expected for entries in arms.values())
            full_matrix &= complete
            for arm in ARMS:
                result = {
                    "model": manifest["model"],
                    "arm": arm,
                    "length": length,
                    "measurement": kind,
                    "measurement_level": "fake_quant_cached_decode",
                    "status": "incomplete",
                    "samples": len(arms[arm]),
                }
                if complete:
                    stats = (
                        paired_nll(arms[arm], arms["k16_v16"])
                        if kind == "nll"
                        else paired_retrieval(arms[arm], arms["k16_v16"])
                    )
                    result.update(stats, status="completed")
                results.append(result)
            if complete:
                for comparator in ("k8_v16", "k8_v8"):
                    stats = (
                        paired_nll(arms["k16_v8"], arms[comparator])
                        if kind == "nll"
                        else paired_retrieval(arms["k16_v8"], arms[comparator])
                    )
                    comparisons.append(
                        {
                            "length": length,
                            "measurement": kind,
                            "arm": "k16_v8",
                            "comparator": comparator,
                            **stats,
                        }
                    )
    supported = full_matrix and all(
        row.get("supported_margin", False) for row in results if row["arm"] == "k16_v8"
    )
    generation_diagnostics = []
    for arm, entries in by_arm.items():
        for key, row in entries.items():
            if row["kind"] != "generation" or key not in by_arm["k16_v16"]:
                continue
            baseline_ids = by_arm["k16_v16"][key]["generated_ids"]
            arm_ids = row["generated_ids"]
            common = 0
            for baseline_token, arm_token in zip(baseline_ids, arm_ids):
                if baseline_token != arm_token:
                    break
                common += 1
            equal = baseline_ids == arm_ids
            grams = [
                tuple(arm_ids[index : index + 5])
                for index in range(max(0, len(arm_ids) - 4))
            ]
            generation_diagnostics.append(
                {
                    "sample_id": key,
                    "arm": arm,
                    "first_divergence": None if equal else common,
                    "prefix_agreement": (
                        common / len(baseline_ids) if baseline_ids else None
                    ),
                    "repeated_fivegrams": len(grams) - len(set(grams)),
                    "truncated": row["truncated"],
                    "malformed": row["malformed"],
                }
            )
    summary = {
        "model": manifest["model"],
        "model_verdict": (
            "screen_supported"
            if supported
            else "inconclusive" if full_matrix else "incomplete"
        ),
        "exploratory": True,
        "rows": results,
        "asymmetry_comparisons": comparisons,
        "analysis_source_sha": source_sha(),
        "input_provenance_hashes": sorted({digest(row["provenance"]) for row in rows}),
        "generation_diagnostics": generation_diagnostics,
        "generation": {
            arm: [
                {
                    "sample_id": key,
                    "generated_ids": row["generated_ids"],
                    "truncated": row["truncated"],
                    "malformed": row["malformed"],
                }
                for key, row in entries.items()
                if row["kind"] == "generation"
            ]
            for arm, entries in by_arm.items()
        },
    }
    write_once(Path(args.output) / "summary.json", summary)
    path = Path(args.output) / "summary.csv"
    fields = sorted(set().union(*(row.keys() for row in results)))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(
        json.dumps({"model": manifest["model"], "verdict": summary["model_verdict"]}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("doctor", "prepare", "calibrate", "quality", "native", "summarize"),
    )
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument(
        "--dataset",
        default="Salesforce/wikitext",
        choices=("Salesforce/wikitext", "wikimedia/wikipedia"),
    )
    parser.add_argument("--dataset-revision")
    parser.add_argument("--kernel-revision")
    parser.add_argument("--matrix")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scales")
    parser.add_argument("--arm", choices=ARMS, default="k16_v16")
    parser.add_argument("--length", type=int, choices=(2048, 8192))
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", default="quality-01")
    args = parser.parse_args()
    threads()
    if args.stage == "prepare":
        prepare(args)
        return
    manifest = read(args.manifest)
    validate_execution_manifest(manifest, source_sha())
    if args.stage in ("calibrate", "quality") and not args.scales:
        parser.error("calibrate and quality require --scales")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    try:
        globals()[args.stage](args, manifest)
    except Exception as exception:
        failure = {
            "stage": args.stage,
            "type": type(exception).__name__,
            "message": str(exception),
            "source_sha": source_sha(),
            "provenance": provenance(manifest),
        }
        path = Path(args.output) / f"failure-{args.stage}-{time.time_ns()}.json"
        write_once(path, failure)
        raise


if __name__ == "__main__":
    main()
