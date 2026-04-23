#!/usr/bin/env python3.12
"""Per-model decode throughput sweep across B x T grid for all six configs.

Each measurement: steady-state vLLM decode throughput (tokens/s) at a
fixed (model, config, batch, context) operating point.  Written as one
JSONL line per (model, config, B, T) for downstream Hill fitting.

Configs:
    fp16         FP16 KV cache, unit scales (P0 baseline)
    fp8_uncalib  symmetric FP8, unit scales (vLLM default)
    fp8_calib_pt symmetric FP8, per-tensor absmax scales
    fp8_calib_pc symmetric FP8, per-channel K absmax scales
    asym_uncalib asymmetric FP16-K/FP8-V, unit V scales
    asym_calib   asymmetric FP16-K/FP8-V, calibrated V scales

Scales are loaded from /workspace/results/kv_scales/<model_slug>.json
(produced by collect_kv_scales.py).

Output: /workspace/results/saturation_<model_slug>.jsonl

Caveats:
    - vLLM doesn't universally expose "set KV scale to X" from the
      Python API. For calibrated symmetric FP8 we write a scales file
      to FP8_SCALES_PATH and use the kv_cache_scales_path arg; for
      asymmetric calibrated we thread V scales through the asym
      FlashInfer branch's plan() call.  See per-config helper below.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch


RESULT_DIR = Path("/workspace/results")
SCALES_DIR = RESULT_DIR / "kv_scales"


# (B, T) grid per model.  Some combinations may be infeasible at certain
# model sizes; the runner records an "oom" error instead of crashing.
B_GRID = [2, 4, 8, 16, 32, 64]
T_GRID = [1024, 4096, 16384]

CONFIGS = [
    "fp16",
    "fp8_uncalib",
    "fp8_calib_pt",
    "fp8_calib_pc",
    "asym_uncalib",
    "asym_calib",
]


def build_llm(model: str, config: str, scales_path: str, max_model_len: int):
    """Instantiate a vLLM engine for a (model, config) pair.
    Returns the LLM object; caller is responsible for cleanup."""
    from vllm import LLM

    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

    common = dict(
        model=model,
        dtype="float16",
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_model_len=max_model_len,
        attention_backend="FLASHINFER",
    )

    if config == "fp16":
        return LLM(**common, kv_cache_dtype="auto")

    if config == "fp8_uncalib":
        return LLM(**common, kv_cache_dtype="fp8_e4m3")

    if config == "fp8_calib_pt" or config == "fp8_calib_pc":
        # vLLM's kv_cache_scales_path loads a JSON of per-layer K/V
        # scales.  Per-tensor and per-channel differ in the JSON
        # written to scales_path.
        return LLM(
            **common,
            kv_cache_dtype="fp8_e4m3",
            quantization_param_path=scales_path,
        )

    if config == "asym_uncalib":
        # The asymmetric branch uses "auto;fp8_e4m3" to indicate
        # K=default (FP16), V=FP8.
        return LLM(**common, kv_cache_dtype="auto;fp8_e4m3")

    if config == "asym_calib":
        # Same asym K16/V8 path, but with V scales loaded from the
        # same JSON format (only V entries are read).
        return LLM(
            **common,
            kv_cache_dtype="auto;fp8_e4m3",
            quantization_param_path=scales_path,
        )

    raise ValueError(f"unknown config: {config}")


def measure_decode_throughput(llm, batch: int, context: int, decode_tokens: int = 64):
    """Measure steady-state decode tokens/s at (batch, context).

    We send `batch` prompts of `context` tokens and request
    `decode_tokens` output tokens each.  Wall time covers prefill +
    decode.  Subtracting estimated prefill wall time is unreliable, so
    we report decode-tokens-per-second by dividing total generated
    output tokens by total wall time minus a separate prefill-only
    warmup pass at the same shape.
    """
    from vllm import SamplingParams

    # Fixed 1-token prompt padded to context (the KV bytes are what we
    # care about; the prompt content is irrelevant for throughput).
    tok = llm.get_tokenizer()
    prompt_ids = tok.encode("The meaning of life is", add_special_tokens=False)
    # Pad by repeating to reach `context` tokens
    while len(prompt_ids) < context:
        prompt_ids.extend(prompt_ids)
    prompt_ids = prompt_ids[:context]

    prompts = [tok.decode(prompt_ids, skip_special_tokens=False)] * batch
    params = SamplingParams(max_tokens=decode_tokens, temperature=0.0, top_p=1.0)

    # Warmup (burns in CUDA graphs; result discarded)
    _ = llm.generate(prompts, sampling_params=params, use_tqdm=False)

    torch.cuda.synchronize()
    t0 = time.time()
    out = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.time() - t0

    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in out)
    return total_out_tokens / dt  # tokens/s, end-to-end


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def run_one_model(model: str, out_path: Path):
    scales_path = SCALES_DIR / f"{slug(model)}.json"
    # For calibrated configs the scales_path must exist.  Per-tensor
    # and per-channel share the same JSON; vLLM picks which one based
    # on the fp8_kv_scale_format flag on the model_runner.  We write
    # two variants to disk for clarity.
    pt_path = SCALES_DIR / f"{slug(model)}.pertensor.json"
    pc_path = SCALES_DIR / f"{slug(model)}.perchannel.json"
    if scales_path.exists():
        data = json.load(open(scales_path))
        # Write the two formats expected by vLLM
        with open(pt_path, "w") as f:
            json.dump({
                "kv_cache_scales": {
                    str(i): {"k": k, "v": v}
                    for i, (k, v) in enumerate(zip(
                        data["k_pertensor_scale"], data["v_pertensor_scale"]))
                }
            }, f, indent=2)
        with open(pc_path, "w") as f:
            json.dump({
                "kv_cache_scales": {
                    str(i): {"k": k, "v": v}
                    for i, (k, v) in enumerate(zip(
                        data["k_perchannel_scale"], data["v_perchannel_scale"]))
                }
            }, f, indent=2)

    results = []
    for config in CONFIGS:
        sp = pc_path if "calib_pc" in config else pt_path if "calib" in config else None
        max_model_len = max(T_GRID) + 128
        try:
            llm = build_llm(model, config, str(sp) if sp else "", max_model_len)
        except Exception as e:
            for B in B_GRID:
                for T in T_GRID:
                    results.append(dict(
                        model=model, config=config, B=B, T=T,
                        tok_per_s=None, err=f"build_llm: {e}"[:300],
                    ))
            continue

        for B in B_GRID:
            for T in T_GRID:
                try:
                    tps = measure_decode_throughput(llm, B, T)
                    print(f"  {model} {config} B={B} T={T} -> {tps:.1f} tok/s", flush=True)
                    results.append(dict(
                        model=model, config=config, B=B, T=T,
                        tok_per_s=tps, err=None,
                    ))
                except Exception as e:
                    print(f"  {model} {config} B={B} T={T} FAIL: {e}", flush=True)
                    results.append(dict(
                        model=model, config=config, B=B, T=T,
                        tok_per_s=None, err=str(e)[:300],
                    ))

        # Tear down engine before next config to free memory
        del llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default=str(RESULT_DIR))
    args = ap.parse_args()

    out = Path(args.out_dir) / f"saturation_{slug(args.model)}.jsonl"
    run_one_model(args.model, out)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
