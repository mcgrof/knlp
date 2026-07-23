#!/usr/bin/env python3
"""STILL - contended-memory concurrency projection (real chunked path).

An earlier ledger projected concurrency from synthetic KV allocation; this grounds the
projection in the real Qwen3-4B chunked-streaming path by measuring the two
atomic terms and validating a real 2-concurrent point:

  peak_hbm(T, C, policy) = W + C * compact(T) + transient_term(policy)

  W               : frozen base weights (+ shared compactor), once per replica
  compact(T)      : per-request accumulated compact KV residency (grows in T)
  transient       : the per-chunk forward working set (activations + raw chunk
                    KV). SERIALIZED compaction shares ONE such buffer across all
                    in-flight requests; PARALLEL compaction pays C of them.

The serialized-vs-parallel split is the "shared reusable workspace" question and
sets the concurrency cliff. Quality not evaluated (untrained compactor).
"""
import os
import argparse, gc, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from still_compactor import (STILLCompactorLayer, apply_rope,
                             cache_to_legacy, legacy_to_cache)  # noqa: E402

GiB = 2 ** 30
BYTES_PER_TOKEN = 36 * 8 * 128 * 2 * 2            # Qwen3-4B, K+V bf16, all layers
USABLE = 46.0                                     # 48 GB card, ~2 GiB headroom


def live(di):
    return torch.cuda.memory_allocated(di) / GiB


def build_compact(model, comp, ids, T, chunk, t_chunk, theta, di,
                  measure_transient=False):
    """Run the chunked construction; return (compact_cache, max_transient)."""
    cfg = model.config
    dev = ids.device
    compact_cache = None
    prior = [(None, None)] * cfg.num_hidden_layers
    max_transient = 0.0
    n = (T + chunk - 1) // chunk
    for ci in range(n):
        s, e = ci * chunk, min(ci * chunk + chunk, T)
        Lc = 0 if compact_cache is None else compact_cache.get_seq_length()
        pos = torch.arange(s, e, device=dev)
        resident_before = live(di)
        torch.cuda.reset_peak_memory_stats(di)
        with torch.no_grad():
            out = model(ids[:, s:e], past_key_values=compact_cache,
                        use_cache=True, position_ids=pos.unsqueeze(0),
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
        transient = torch.cuda.max_memory_allocated(di) / GiB - resident_before
        max_transient = max(max_transient, transient)
        del out, full
        gc.collect(); torch.cuda.empty_cache()
        compact_cache = legacy_to_cache(tuple(new))
    return compact_cache, max_transient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    di = int(args.device.split(":")[1]); torch.cuda.set_device(di)
    dev = args.device
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    cfg = model.config
    H, d, theta = cfg.num_key_value_heads, cfg.head_dim, float(cfg.rope_theta)
    W = live(di)
    print(f"{args.model}: weights W={W:.2f} GiB")

    # --- measure the transient working set vs chunk size (real forward) ---
    print("\n[measure] per-chunk transient working set (T=8192):")
    trans = {}
    for chunk in (2048, 4096):
        comp = STILLCompactorLayer(H, d, t=max(1, chunk // 32),
                                   base_theta=theta).to(dev, torch.bfloat16)
        ids = torch.randint(0, cfg.vocab_size, (1, 8192), device=dev)
        cc, tr = build_compact(model, comp, ids, 8192, chunk, chunk // 32,
                               theta, di)
        trans[chunk] = tr
        del cc; gc.collect(); torch.cuda.empty_cache()
        print(f"  chunk={chunk}: transient={tr:.2f} GiB")
    chunk = 2048
    Tr = trans[chunk]

    # --- validate: real C=2 (two compact caches resident + one active) ---
    comp = STILLCompactorLayer(H, d, t=chunk // 32,
                               base_theta=theta).to(dev, torch.bfloat16)
    ids = torch.randint(0, cfg.vocab_size, (1, 8192), device=dev)
    bg, _ = build_compact(model, comp, ids, 8192, chunk, chunk // 32, theta, di)
    torch.cuda.reset_peak_memory_stats(di)
    active, _ = build_compact(model, comp, ids, 8192, chunk, chunk // 32, theta, di)
    real_c2_peak = torch.cuda.max_memory_allocated(di) / GiB
    compact_8k = bg.get_seq_length() * BYTES_PER_TOKEN / GiB
    proj_c2 = W + 2 * compact_8k + Tr
    print(f"\n[validate C=2 @8K serialized] real peak={real_c2_peak:.2f} GiB | "
          f"projector={proj_c2:.2f} GiB (compact/req={compact_8k*1024:.0f} MiB, "
          f"transient={Tr:.2f}); err={abs(real_c2_peak-proj_c2):.2f} GiB")
    del bg, active; gc.collect(); torch.cuda.empty_cache()

    # --- project the concurrency ceiling (validated projector) ---
    def compact_gib(T, c_eff=32):
        return (T // c_eff) * BYTES_PER_TOKEN / GiB

    def max_C(T, policy):
        cq = compact_gib(T)
        for C in range(1, 200):
            per = cq + (Tr if policy == "parallel" else 0)
            fixed = W + (0 if policy == "parallel" else Tr)
            if fixed + C * per > USABLE:
                return C - 1
        return 199
    print(f"\n[project] max concurrent chunked compactions on 48 GB "
          f"(usable {USABLE} GiB), c_eff=32, chunk={chunk}:")
    print(f"{'T':>7} {'compact/req':>12} {'serialized':>11} {'parallel':>9}")
    for T in (8192, 16384, 32768, 65536):
        print(f"{T:>7} {compact_gib(T)*1024:>10.0f}MiB "
              f"{max_C(T,'serialized'):>11} {max_C(T,'parallel'):>9}")
    print(f"\nShared-workspace verdict: serializing compaction shares ONE "
          f"{Tr:.1f} GiB transient buffer, so the per-request cost is just the "
          f"compact KV; parallel compaction pays {Tr:.1f} GiB per in-flight "
          f"request and hits the HBM cliff far sooner. Serialize compaction.")


if __name__ == "__main__":
    main()
