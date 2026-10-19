# SPDX-License-Identifier: MIT
"""Audit a pinned newer model and calibrate native writes before transfer.

This eager, single-request capability probe is excluded from performance and
does not qualify the asymmetric route. Hybrid state and FP32 parameters are
retained explicitly rather than treating every layer as conventional KV.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-bytes", type=int, default=16 * 2**30)
    p.add_argument(
        "--backend", choices=("FLASHINFER", "TRITON_ATTN"), default="FLASHINFER"
    )
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    p.add_argument("--diagnostic-one-case", action="store_true")
    args = p.parse_args()
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise ValueError("native calibration requires the explicit in-process mode")
    if os.environ.get("VLLM_COMPACT_PREFILL") != "0":
        raise ValueError("native probe requires compact prefill disabled")
    import torch
    import vllm
    import flashinfer
    import transformers
    from vllm import LLM, SamplingParams
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    from .october import VARIANTS
    from .instrument import native_memory_state, save, tensor_inventory
    from .current import save as update

    definition = VARIANTS["native"]
    definition.verify_environment()
    definition.verify_sources(
        {
            "vllm": Path(vllm.__file__).resolve().parents[1],
            "flashinfer": Path(flashinfer.__file__).resolve().parents[1],
        },
        {"vllm": vllm.__file__, "flashinfer": flashinfer.__file__},
    )
    if torch.__version__ != "2.11.0+cu129":
        raise RuntimeError("probe requires the pinned Torch/CUDA build")
    if transformers.__version__ != "5.10.4":
        raise RuntimeError("recent probe requires the separately pinned Transformers")
    args.output.mkdir(parents=True, exist_ok=False)
    raw = (args.inputs / "manifest.json.gz").read_bytes()
    manifest = json.loads(gzip.decompress(raw))
    if set(manifest["calibration_article_ids"]) & set(
        manifest["evaluation_article_ids"]
    ):
        raise ValueError("calibration and evaluation documents overlap")
    policy = {
        "model": manifest["model"],
        "revision": manifest["model_revision"],
        "tokenizer_revision": manifest["tokenizer_revision"],
        "dtype": args.dtype,
        "kv_cache_dtype": "auto",
        "tensor_parallel_size": 1,
        "max_model_len": 10240,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 16384,
        "block_size": 16,
        "kv_cache_memory_bytes": args.cache_bytes,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "calculate_kv_scales": False,
        "attention_config": {"backend": args.backend, "use_trtllm_attention": False},
        "distributed_executor_backend": "uni",
        "seed": 20261019,
        "language_model_only": True,
    }
    save(
        args.output / "provenance.json",
        {
            "policy": policy,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "sources": dict(definition.sources),
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "flashinfer": flashinfer.__version__,
            },
            "purpose": "native capability/calibration only; no performance claim",
        },
    )
    llm = LLM(**policy)
    runner = llm.collective_rpc(lambda worker: worker.model_runner)[0]
    model = runner.get_model()
    original_logits = model.compute_logits

    def finite_logits(*a, **kw):
        logits = original_logits(*a, **kw)
        if not bool(torch.isfinite(logits).all()):
            save(
                args.output / "nonfinite-logits.json",
                {
                    "sample_id": sample_id,
                    "shape": list(logits.shape),
                    "dtype": str(logits.dtype),
                    "nan_count": int(torch.isnan(logits).sum()),
                    "positive_infinity_count": int(torch.isposinf(logits).sum()),
                    "negative_infinity_count": int(torch.isneginf(logits).sum()),
                    "finite_count": int(torch.isfinite(logits).sum()),
                },
            )
            raise RuntimeError("nonfinite native generation logits")
        return logits

    model.compute_logits = finite_logits
    attention = {}
    modules = []
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if method is not None:
            modules.append(
                {
                    "name": name,
                    "class": type(module).__name__,
                    "quant_method": type(method).__name__,
                    "weight": tensor_inventory(getattr(module, "weight", None)),
                }
            )
        if hasattr(module, "_v_scale") and hasattr(module, "kv_cache"):
            attention[id(module)] = {
                "name": name,
                "layer": int(name.split("layers.")[1].split(".")[0]),
                "query_heads": module.num_heads,
                "kv_heads": module.num_kv_heads,
                "head_dim": module.head_size,
                "impl": type(module.impl).__name__,
                "cache": tensor_inventory(module.kv_cache),
            }
    params = [
        {"name": name, "tensors": tensor_inventory(value)}
        for name, value in model.named_parameters()
    ]
    caches = tensor_inventory(runner.kv_caches)
    storages = {(x["device"], x["storage_pointer"]): x["storage_bytes"] for x in caches}
    save(
        args.output / "model-inventory.json",
        {
            "attention": list(attention.values()),
            "weight_modules": modules,
            "parameters": params,
            "all_runner_cache_tensors_including_hybrid_state": caches,
            "unique_runner_cache_backing_bytes": sum(storages.values()),
            "cache_configuration": repr(runner.kv_cache_config),
            "memory": llm.collective_rpc(native_memory_state),
        },
    )
    if not attention:
        raise RuntimeError("native probe found no full/sliding attention layers")
    maxima, writes = {}, []
    implementation = {
        "FLASHINFER": FlashInferImpl,
        "TRITON_ATTN": TritonAttentionImpl,
    }[args.backend]
    if any(row["impl"] != implementation.__name__ for row in attention.values()):
        raise RuntimeError(
            "actual attention implementation differs from the requested backend"
        )
    original = implementation.do_kv_cache_update
    sample_id = None

    def observe(impl, layer, key, value, cache, slots):
        if sample_id is not None:
            record = attention[id(layer)]
            for component, tensor in (("k", key), ("v", value)):
                live = tensor[: len(slots)][slots >= 0]
                if live.dtype != getattr(torch, args.dtype) or not bool(
                    torch.isfinite(live).all()
                ):
                    raise RuntimeError(
                        "nonfinite or unexpected-dtype native cache write"
                    )
                maximum = float(live.float().abs().max())
                field = f"{record['layer']}.{component}"
                maxima[field] = max(maxima.get(field, 0.0), maximum)
                writes.append(
                    {
                        "sample_id": sample_id,
                        "field": field,
                        "maximum": maximum,
                        "elements": live.numel(),
                    }
                )
        return original(impl, layer, key, value, cache, slots)

    implementation.do_kv_cache_update = observe
    cases = [s for s in manifest["samples"] if s["kind"] == "calibration"]
    if args.diagnostic_one_case:
        cases = cases[:1]
    try:
        for case in cases:
            sample_id = case["id"]
            values = llm.generate(
                [{"prompt_token_ids": case["token_ids"][: case["length"]]}],
                SamplingParams(temperature=0, max_tokens=128, ignore_eos=True),
                use_tqdm=False,
            )
            save(
                args.output / (sample_id + ".json"),
                {
                    "token_ids": list(values[0].outputs[0].token_ids),
                    "text": values[0].outputs[0].text,
                },
            )
            update(
                args.output / "calibration-progress.json",
                {"sample_id": sample_id, "scales_seen": len(maxima)},
            )
    finally:
        implementation.do_kv_cache_update = original
        save(args.output / "write-maxima.json", writes)
    if len(maxima) != 2 * len(attention):
        raise RuntimeError("native writer calibration coverage incomplete")
    if args.diagnostic_one_case:
        save(
            args.output / "DONE.json",
            {
                "native_single_case_diagnostic_passed": True,
                "sample_id": cases[0]["id"],
                "native_calibration_complete": False,
                "asymmetric_qualified": False,
                "scored_quality_complete": False,
            },
        )
        return
    save(
        args.output / "scales.json",
        {
            "method": "max_abs_div_448_float32_once",
            "model": manifest["model"],
            "model_revision": manifest["model_revision"],
            "backend_sources": dict(definition.sources),
            "attention_layer_indices": sorted(r["layer"] for r in attention.values()),
            "scales": {
                key: float(
                    torch.tensor(value / 448 if value else 1, dtype=torch.float32)
                )
                for key, value in sorted(maxima.items())
            },
            "maxima": maxima,
            "calibration_ids": [s["id"] for s in cases],
        },
    )
    save(
        args.output / "DONE.json",
        {
            "native_calibration_complete": True,
            "asymmetric_qualified": False,
            "scored_quality_complete": False,
            "memory": llm.collective_rpc(native_memory_state),
        },
    )


if __name__ == "__main__":
    main()
