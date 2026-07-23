#!/usr/bin/env python3
"""STILL honest baselines - separate the confounded systems effects.

an external review flagged that the earlier "32K full-prefill OOMs at 64 GB"
headline is an eager T^2 base-attention artifact (32 heads * 32768^2 * 2B =
64 GiB = the materialized score tensor), NOT a KV-cache result. The faithful
comparison needs a chunked full-KV baseline that avoids the T^2 scores but keeps
all KV, so we can separate three effects:
  1. chunked prefill eliminating the base-attention T^2 workspace;
  2. STILL eliminating the RETAINED KV;
  3. (the fused kernel eliminating compactor workspace - shown in the fused kernel).

Three arms at each T: eager full prefill (may OOM on the T^2 tensor), chunked
full-KV (no compaction, keeps the ordinary KV), chunked STILL (compact+free).
Peak measured as live max_memory_allocated in isolated context. Quality N/A.
"""
import os
import argparse, gc, sys
import torch
from transformers import AutoModelForCausalLM, DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from still_compactor import (STILLCompactorLayer, apply_rope,
                             cache_to_legacy, legacy_to_cache)  # noqa: E402

GiB = 2 ** 30


def peak_since_reset(di):
    return torch.cuda.max_memory_allocated(di) / GiB


def eager_full(model, ids, di):
    torch.cuda.reset_peak_memory_stats(di)
    with torch.no_grad():
        model(ids, use_cache=True)
    return peak_since_reset(di)


def chunked_fullkv(model, ids, chunk, di):
    """Chunked prefill, keep ALL KV (no compaction). Avoids the T^2 scores."""
    torch.cuda.reset_peak_memory_stats(di)
    cache, dev, T = None, ids.device, ids.shape[1]
    with torch.no_grad():
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            Lc = 0 if cache is None else cache.get_seq_length()
            out = model(ids[:, s:e], past_key_values=cache, use_cache=True,
                        position_ids=torch.arange(s, e, device=dev).unsqueeze(0),
                        cache_position=torch.arange(Lc, Lc + (e - s), device=dev))
            cache = out.past_key_values                # KEEP everything
    return peak_since_reset(di), cache.get_seq_length()


def chunked_still(model, comp, ids, chunk, tchunk, theta, di):
    torch.cuda.reset_peak_memory_stats(di)
    cache, prior = None, [(None, None)] * model.config.num_hidden_layers
    dev, T = ids.device, ids.shape[1]
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        Lc = 0 if cache is None else cache.get_seq_length()
        pos = torch.arange(s, e, device=dev)
        with torch.no_grad():
            out = model(ids[:, s:e], past_key_values=cache, use_cache=True,
                        position_ids=pos.unsqueeze(0),
                        cache_position=torch.arange(Lc, Lc + (e - s), device=dev))
        full = cache_to_legacy(out.past_key_values)
        new = []
        with torch.no_grad():
            for li, (k, v) in enumerate(full):
                k_un = apply_rope(k[0, :, Lc:, :], pos, theta, inverse=True)
                c_k, c_v, _ = comp(k_un, v[0, :, Lc:, :], pos)
                pk, pv = prior[li]
                if pk is not None:
                    c_k, c_v = torch.cat([pk, c_k], 1), torch.cat([pv, c_v], 1)
                new.append((c_k.unsqueeze(0), c_v.unsqueeze(0)))
                prior[li] = (c_k, c_v)
        del out, full; gc.collect(); torch.cuda.empty_cache()
        cache = legacy_to_cache(tuple(new))
    return peak_since_reset(di), cache.get_seq_length()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--t-chunk", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    di = int(args.device.split(":")[1]); torch.cuda.set_device(di); dev = args.device
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    cfg = model.config
    H, d, theta = cfg.num_key_value_heads, cfg.head_dim, float(cfg.rope_theta)
    comp = STILLCompactorLayer(H, d, t=args.t_chunk, base_theta=theta).to(dev, torch.bfloat16)
    W = torch.cuda.memory_allocated(di) / GiB
    print(f"{args.model}: weights={W:.2f} GiB, chunk={args.chunk}, "
          f"t_chunk={args.t_chunk} (c_eff={args.chunk//args.t_chunk}x)\n")
    print(f"{'T':>7} {'eager-full':>12} {'chunked-full-KV':>16} "
          f"{'chunked-STILL':>14} {'STILL vs chunkedKV':>19}")
    for T in (8192, 16384, 32768, 65536):
        ids = torch.randint(0, cfg.vocab_size, (1, T), device=dev)
        try:
            ef = f"{eager_full(model, ids, di):.2f}"
        except RuntimeError:
            ef = "OOM"
        gc.collect(); torch.cuda.empty_cache()
        try:
            ck, cklen = chunked_fullkv(model, ids, args.chunk, di)
            cks = f"{ck:.2f}"
        except RuntimeError:
            ck, cks = None, "OOM"
        gc.collect(); torch.cuda.empty_cache()
        cs, cslen = chunked_still(model, comp, ids, args.chunk, args.t_chunk, theta, di)
        gc.collect(); torch.cuda.empty_cache()
        save = f"{ck-cs:.2f} GiB less" if ck else "-"
        print(f"{T:>7} {ef:>12} {cks:>16} {cs:>13.2f} {save:>19}")
        del ids
    print("\nRead: eager-full's blowup is the T^2 score tensor (chunking removes "
          "it); the FAITHFUL STILL win is chunked-STILL vs chunked-full-KV = the "
          "RETAINED-KV savings, which GROWS with T (flat STILL vs linearly "
          "growing full KV). That is the real, non-confounded frontier result.")


if __name__ == "__main__":
    main()
