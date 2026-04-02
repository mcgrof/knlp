#!/usr/bin/env python3
"""Marin 8B-class RA head-count screen on H100.

Sweep: baseline, RA-20, RA-24, RA-28, RA-32
Each variant: 20-minute wall-clock cap
Method: cache-transform PPL evaluation (reciprocal attention on KV cache)
"""

from __future__ import annotations
import json, math, time, os, sys
from pathlib import Path
from statistics import mean
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

DEVICE = "cuda"
MODEL_NAME = "marin-community/marin-8b-base"
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
ROOT = Path("/data/knlp-key-results/marin-ra-headscale") / f"7b-{TIMESTAMP}"
ROOT.mkdir(parents=True, exist_ok=True)

WALL_CAP_SECONDS = 1200  # 20 minutes per variant
FIM_JSON = Path("/data/knlp/fim_traces_marin8b_wikitext.json")

LOG = ROOT / "screen.log"


def log(msg):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {msg}'
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def reciprocal_mix_cache(past, selected_layers, beta=0.2, mode="mixed"):
    new_cache = DynamicCache()
    # transformers 5.x: DynamicCache.layers[i].keys / .values
    if hasattr(past, "layers"):
        num_layers = len(past.layers)
        for layer_idx in range(num_layers):
            layer = past.layers[layer_idx]
            k = layer.keys.clone()
            v = layer.values.clone()
            if layer_idx in selected_layers:
                kf = F.normalize(k.float(), dim=-1)
                scores = torch.matmul(kf, kf.transpose(-1, -2)) / math.sqrt(kf.size(-1))
                scores = torch.softmax(scores, dim=-1)
                reciprocal_v = torch.matmul(scores, v.float()).to(v.dtype)
                if mode == "replace":
                    v = reciprocal_v
                else:
                    v = (1.0 - beta) * v + beta * reciprocal_v
            new_cache.update(k, v, layer_idx)
    elif hasattr(past, "key_cache") and past.key_cache is not None:
        # transformers 4.x fallback
        for layer_idx, (k_src, v_src) in enumerate(
            zip(past.key_cache, past.value_cache)
        ):
            k = k_src.clone()
            v = v_src.clone()
            if layer_idx in selected_layers:
                kf = F.normalize(k.float(), dim=-1)
                scores = torch.matmul(kf, kf.transpose(-1, -2)) / math.sqrt(kf.size(-1))
                scores = torch.softmax(scores, dim=-1)
                reciprocal_v = torch.matmul(scores, v.float()).to(v.dtype)
                if mode == "replace":
                    v = reciprocal_v
                else:
                    v = (1.0 - beta) * v + beta * reciprocal_v
            new_cache.update(k, v, layer_idx)
    else:
        raise RuntimeError(f"Unsupported cache type: {type(past)}")
    return new_cache


def get_eval_tokens(tokenizer, max_tokens=200000):
    ds = load_dataset("wikitext", "wikitext-103-v1", split="test")
    text = "\n".join(x["text"] for x in ds if x["text"].strip())
    toks = tokenizer.encode(text)
    return toks[:max_tokens]


def eval_variant(
    model,
    tokens,
    selected_layers,
    beta,
    mode,
    label,
    prompt_len=2048,
    wall_cap=WALL_CAP_SECONDS,
):
    """Run cache-transform eval under a wall-clock cap."""
    logit_errs, agrees, ppls, tok_times = [], [], [], []
    wall_start = time.time()
    seed = 0
    n_done = 0

    while True:
        elapsed = time.time() - wall_start
        if elapsed >= wall_cap:
            break

        rng = np.random.RandomState(seed + 42)
        start_idx = rng.randint(0, max(1, len(tokens) - prompt_len - 100))
        input_ids = (
            torch.tensor(tokens[start_idx : start_idx + prompt_len], dtype=torch.long)
            .unsqueeze(0)
            .to(DEVICE)
        )
        next_tok = torch.tensor(
            [[tokens[start_idx + prompt_len]]], dtype=torch.long
        ).to(DEVICE)

        with torch.no_grad():
            # Baseline forward
            out_fp = model(input_ids, use_cache=True)

            t0 = time.perf_counter()
            # IMPORTANT: compare independent cache copies.
            # DynamicCache is mutable and advances in-place during decode.
            # If we reuse the same cache object for both baseline and variant
            # calls, baseline token agreement is artificially < 1.0 because the
            # second call sees an already-advanced cache state.
            baseline_cache = reciprocal_mix_cache(
                out_fp.past_key_values, set(), beta=0.0, mode="mixed"
            )
            if selected_layers is not None and len(selected_layers) > 0:
                transformed_cache = reciprocal_mix_cache(
                    out_fp.past_key_values, selected_layers, beta=beta, mode=mode
                )
            else:
                transformed_cache = reciprocal_mix_cache(
                    out_fp.past_key_values, set(), beta=0.0, mode="mixed"
                )

            o_base = model(next_tok, past_key_values=baseline_cache, use_cache=True)
            o_var = model(next_tok, past_key_values=transformed_cache, use_cache=True)
            dt = time.perf_counter() - t0

        # Logit error
        err = (
            (o_base.logits[0, 0].float() - o_var.logits[0, 0].float())
            .abs()
            .max()
            .item()
        )
        agree = (
            1 if o_base.logits[0, -1].argmax() == o_var.logits[0, -1].argmax() else 0
        )

        # PPL from variant logits over the prompt
        with torch.no_grad():
            var_out = model(input_ids, use_cache=False)
            shift_logits = var_out.logits[0, :-1, :]
            shift_labels = input_ids[0, 1:]
            loss = F.cross_entropy(shift_logits, shift_labels)
            ppl = math.exp(min(loss.item(), 20.0))

        logit_errs.append(err)
        agrees.append(agree)
        ppls.append(ppl)
        tok_times.append(dt)
        seed += 1
        n_done += 1

        del out_fp, transformed_cache, o_base, o_var, var_out
        torch.cuda.empty_cache()

        if n_done % 5 == 0:
            log(
                f"  [{label}] {n_done} prompts done, elapsed {elapsed:.0f}s, "
                f"avg_ppl={mean(ppls):.3f} avg_agree={mean(agrees):.4f}"
            )

    result = {
        "label": label,
        "n_prompts": n_done,
        "prompt_len": prompt_len,
        "wall_seconds": round(time.time() - wall_start, 1),
        "avg_ppl": round(float(mean(ppls)), 4) if ppls else None,
        "avg_logit_error": round(float(mean(logit_errs)), 6) if logit_errs else None,
        "avg_token_agreement": round(float(mean(agrees)), 6) if agrees else None,
        "p95_logit_error": (
            round(float(np.percentile(logit_errs, 95)), 6) if logit_errs else None
        ),
        "avg_eval_seconds": round(float(mean(tok_times)), 6) if tok_times else None,
        "selected_layers": sorted(selected_layers) if selected_layers else [],
        "beta": beta,
        "mode": mode,
        "per_prompt_ppls": [round(float(x), 4) for x in ppls],
    }
    return result


def main():
    log(f"[start] Marin 7B-class RA head-count screen")
    log(f"[model] MODEL_NAME={MODEL_NAME}")
    log(f"[result_root] {ROOT}")

    # Resolve model — check if it's actually 8B
    log("[model_resolution] Loading model info...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0"
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    num_params = sum(p.numel() for p in model.parameters())
    model_class = f"{num_params/1e9:.1f}B"

    log(
        f"[model_resolved] {MODEL_NAME} -> {model_class} params, "
        f"{num_layers} layers, {num_heads} attn heads, hidden={hidden_size}"
    )
    log(
        f'[note] Task says "7B-class" but actual model is {model_class}. '
        f"This is marin-community/marin-8b-base (Llama-style 8B)."
    )

    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log(f"[gpu] {gpu_name}, {gpu_mem:.1f} GB")

    # Load FIM ranking
    fim = json.loads(FIM_JSON.read_text())
    layer_ranking = fim["layer_ranking"]
    log(f"[fim] Loaded layer ranking: {layer_ranking}")

    # Load eval tokens
    log("[data] Loading wikitext eval tokens...")
    tokens = get_eval_tokens(tokenizer)
    log(f"[data] {len(tokens)} eval tokens ready")

    # Define variants
    variants = [
        {"label": "baseline", "n_heads": 0, "layers": [], "beta": 0.0, "mode": "mixed"},
        {
            "label": "RA-20",
            "n_heads": 20,
            "layers": sorted(layer_ranking[:20]),
            "beta": 0.15,
            "mode": "mixed",
        },
        {
            "label": "RA-24",
            "n_heads": 24,
            "layers": sorted(layer_ranking[:24]),
            "beta": 0.15,
            "mode": "mixed",
        },
        {
            "label": "RA-28",
            "n_heads": 28,
            "layers": sorted(layer_ranking[:28]),
            "beta": 0.15,
            "mode": "mixed",
        },
        {
            "label": "RA-32",
            "n_heads": 32,
            "layers": sorted(layer_ranking[:32]),
            "beta": 0.15,
            "mode": "mixed",
        },
    ]

    # Save config
    config = {
        "model": MODEL_NAME,
        "model_class": model_class,
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        "hidden_size": hidden_size,
        "gpu": gpu_name,
        "gpu_memory_gb": round(gpu_mem, 1),
        "wall_cap_seconds": WALL_CAP_SECONDS,
        "fim_source": str(FIM_JSON),
        "layer_ranking": layer_ranking,
        "variants": [
            {
                "label": v["label"],
                "n_heads": v["n_heads"],
                "layers": v["layers"],
                "beta": v["beta"],
                "mode": v["mode"],
            }
            for v in variants
        ],
        "timestamp": TIMESTAMP,
    }
    (ROOT / "config.json").write_text(json.dumps(config, indent=2))

    # Run sweep
    all_results = []
    for vi, var in enumerate(variants):
        log(
            f'[variant_start] {var["label"]} ({vi+1}/{len(variants)}) '
            f'heads={var["n_heads"]} layers={var["layers"]}'
        )

        result = eval_variant(
            model,
            tokens,
            selected_layers=set(var["layers"]),
            beta=var["beta"],
            mode=var["mode"],
            label=var["label"],
        )
        all_results.append(result)

        log(
            f'[variant_finish] {var["label"]} '
            f'n_prompts={result["n_prompts"]} '
            f'wall={result["wall_seconds"]}s '
            f'avg_ppl={result["avg_ppl"]} '
            f'avg_agree={result["avg_token_agreement"]} '
            f'avg_logit_err={result["avg_logit_error"]}'
        )

        # Write partial results after every completed variant
        partial = {
            "model": MODEL_NAME,
            "model_class": model_class,
            "hardware": gpu_name,
            "timestamp": TIMESTAMP,
            "completed_variants": len(all_results),
            "total_variants": len(variants),
            "results": all_results,
        }
        (ROOT / "results_partial.json").write_text(json.dumps(partial, indent=2))
        log(
            f"[partial_save] Wrote partial results ({len(all_results)}/{len(variants)} done)"
        )

    # Final summary
    baseline_ppl = all_results[0]["avg_ppl"] if all_results else None
    summary_lines = []
    for r in all_results:
        delta = (r["avg_ppl"] - baseline_ppl) if baseline_ppl and r["avg_ppl"] else None
        delta_pct = (
            (delta / baseline_ppl * 100) if delta is not None and baseline_ppl else None
        )
        summary_lines.append(
            {
                "label": r["label"],
                "avg_ppl": r["avg_ppl"],
                "delta_ppl": round(delta, 4) if delta is not None else None,
                "delta_pct": round(delta_pct, 3) if delta_pct is not None else None,
                "avg_token_agreement": r["avg_token_agreement"],
                "n_prompts": r["n_prompts"],
                "wall_seconds": r["wall_seconds"],
            }
        )

    final = {
        "model": MODEL_NAME,
        "model_class": model_class,
        "hardware": gpu_name,
        "method": "cache_transform_ra_headcount_screen",
        "wall_cap_seconds": WALL_CAP_SECONDS,
        "timestamp": TIMESTAMP,
        "result_root": str(ROOT),
        "summary": summary_lines,
        "full_results": all_results,
        "config": config,
    }
    (ROOT / "results_final.json").write_text(json.dumps(final, indent=2))
    log(f'[final_save] {ROOT / "results_final.json"}')

    # Print summary table
    log("[summary_table]")
    log(
        f'  {"Variant":<12} {"PPL":>10} {"delta":>10} {"delta%":>10} {"agree":>10} {"prompts":>8} {"wall_s":>8}'
    )
    for s in summary_lines:
        log(
            f'  {s["label"]:<12} {s["avg_ppl"]:>10} '
            f'{s["delta_ppl"] if s["delta_ppl"] is not None else "---":>10} '
            f'{s["delta_pct"] if s["delta_pct"] is not None else "---":>10} '
            f'{s["avg_token_agreement"]:>10} {s["n_prompts"]:>8} {s["wall_seconds"]:>8}'
        )

    log("[done] All variants complete.")


if __name__ == "__main__":
    main()
