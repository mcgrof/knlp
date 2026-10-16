# SPDX-License-Identifier: MIT
"""Freeze FP32 max-abs/448 KV scales from native disjoint calibration writes."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def main():
    import os
    import torch
    import vllm
    import flashinfer
    from vllm import LLM, SamplingParams
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl
    from .instrument import save

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-bytes", type=int, required=True)
    args = parser.parse_args()
    from .october import VARIANTS

    VARIANTS["native"].verify_environment()
    if os.environ.get("VLLM_COMPACT_PREFILL") != "0":
        raise ValueError("calibration requires the native route")
    VARIANTS["native"].verify_sources(
        {
            "vllm": Path(vllm.__file__).resolve().parents[1],
            "flashinfer": Path(flashinfer.__file__).resolve().parents[1],
        },
        {"vllm": vllm.__file__, "flashinfer": flashinfer.__file__},
    )
    manifest = json.loads(gzip.decompress((args.inputs / "qwen.json.gz").read_bytes()))
    if set(manifest["calibration_article_ids"]) & set(
        manifest["evaluation_article_ids"]
    ):
        raise ValueError("calibration documents overlap evaluation")
    args.output.mkdir(parents=True, exist_ok=False)
    llm = LLM(
        model=manifest["model"],
        revision=manifest["model_revision"],
        tokenizer_revision=manifest["tokenizer_revision"],
        dtype="float16",
        kv_cache_dtype="auto",
        max_model_len=10240,
        max_num_seqs=8,
        max_num_batched_tokens=65536,
        kv_cache_memory_bytes=args.cache_bytes,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        calculate_kv_scales=False,
        attention_config={"backend": "FLASHINFER", "use_trtllm_attention": False},
        distributed_executor_backend="uni",
        seed=20261018,
    )
    runner = llm.collective_rpc(lambda w: w.model_runner)[0]
    modules = {
        id(m): int(n.split("layers.")[1].split(".")[0])
        for n, m in runner.get_model().named_modules()
        if hasattr(m, "_v_scale") and hasattr(m, "kv_cache")
    }
    if len(modules) != 36:
        raise ValueError("this calibration entrypoint supports the Qwen3 recipe only")
    maxima = defaultdict(float)
    old = FlashInferImpl.do_kv_cache_update
    case_id = None

    def observed(impl, layer, key, value, cache, slots):
        for component, tensor in (("k", key), ("v", value)):
            live = tensor[: len(slots)][slots >= 0]
            if live.dtype != torch.float16 or not bool(torch.isfinite(live).all()):
                raise RuntimeError("invalid native calibration activation")
            maximum = float(live.float().abs().max())
            name = f"{modules[id(layer)]}.{component}"
            maxima[name] = max(maxima[name], maximum)
            with (args.output / "write-maxima.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"sample_id": case_id, "name": name, "maximum": maximum})
                    + "\n"
                )
        return old(impl, layer, key, value, cache, slots)

    FlashInferImpl.do_kv_cache_update = observed
    cases = [s for s in manifest["samples"] if s["kind"] == "calibration"]
    try:
        for case in cases:
            case_id = case["id"]
            values = llm.generate(
                [{"prompt_token_ids": case["token_ids"][: case["length"]]}],
                SamplingParams(temperature=0, max_tokens=128, ignore_eos=True),
                use_tqdm=False,
            )
            save(
                args.output / (case_id + ".json"), list(values[0].outputs[0].token_ids)
            )
    finally:
        FlashInferImpl.do_kv_cache_update = old
    if len(maxima) != 72:
        raise RuntimeError("calibration layer coverage incomplete")
    save(
        args.output / "scales.json",
        {
            "scales": {
                k: float(torch.tensor(v / 448 if v else 1, dtype=torch.float32))
                for k, v in sorted(maxima.items())
            },
            "method": "max_abs_div_448_float32_once",
            "maxima": dict(maxima),
            "calibration_ids": [c["id"] for c in cases],
            "model": manifest["model"],
            "model_revision": manifest["model_revision"],
            "backend_sources": dict(VARIANTS["native"].sources),
        },
    )


if __name__ == "__main__":
    main()
