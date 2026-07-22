#!/usr/bin/env python3
"""STILL - chunked-streaming construction (the make-or-break), HF prototype.

The the gate proved the frontier win needs a schedule that never holds the full
source KV. The the spike compacted AFTER a full prefill, so it did not bound the
construction peak. This prototype builds the paper's iterative chunked schedule
on a real Qwen3-4B and measures whether peak HBM stays bounded:

  compact_cache = empty
  for each chunk of the context:
      forward the chunk attending to the accumulated compact_cache (queries at
        LOGICAL positions, keys written at PHYSICAL slots);
      take the chunk's fresh raw KV, inverse-RoPE the keys, compact to t_chunk
        with the STILL reference compactor, re-rotate output keys;
      APPEND the compacted chunk to compact_cache and FREE the raw chunk KV.
  decode from the final compact_cache at the logical end position.

Never resident: the full source KV or a full-length prefill activation. We
compare the chunked peak against a full-prefill baseline. Quality is not
evaluated (untrained identity-init compactor); this measures the SYSTEMS
envelope and the correctness of the position bookkeeping.
"""
import os
import argparse, gc, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from still_compactor import STILLCompactorLayer, apply_rope  # noqa: E402

GiB = 2 ** 30


def live(di):
    return torch.cuda.memory_allocated(di) / GiB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--T", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--t-chunk", type=int, default=64)   # compact per chunk
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    di = int(args.device.split(":")[1]); torch.cuda.set_device(di)
    dev, dt = args.device, torch.bfloat16

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dt, attn_implementation="eager").to(dev).eval()
    cfg = model.config
    H, d, theta = cfg.num_key_value_heads, cfg.head_dim, float(cfg.rope_theta)
    comp = STILLCompactorLayer(H, d, t=args.t_chunk, base_theta=theta).to(dev, dt)
    base_live = live(di)
    print(f"{args.model}: layers={cfg.num_hidden_layers} kv_heads={H} "
          f"head_dim={d}; weights live={base_live:.2f} GiB")
    ids = torch.randint(0, cfg.vocab_size, (1, args.T), device=dev)

    # ---- baseline: full prefill (what a full prefill does). OOM is a valid result. ----
    torch.cuda.reset_peak_memory_stats(di)
    try:
        with torch.no_grad():
            _ = model(ids, use_cache=True)
        full_peak = torch.cuda.max_memory_allocated(di) / GiB
        del _
        print(f"\n[baseline] full prefill T={args.T}: peak live={full_peak:.2f} GiB")
    except RuntimeError as ex:
        full_peak = float("inf")
        print(f"\n[baseline] full prefill T={args.T}: OOM/FAIL "
              f"({str(ex)[:50]}) -- full prefill does not fit")
    gc.collect(); torch.cuda.empty_cache()

    # ---- chunked-streaming construction ----
    torch.cuda.reset_peak_memory_stats(di)
    compact_cache = None
    prior = [(None, None)] * cfg.num_hidden_layers
    n_chunks = (args.T + args.chunk - 1) // args.chunk
    chunk_peaks = []
    for ci in range(n_chunks):
        s = ci * args.chunk
        e = min(s + args.chunk, args.T)
        chunk_ids = ids[:, s:e]
        clen = e - s
        Lc = 0 if compact_cache is None else compact_cache.get_seq_length()
        pos_ids = torch.arange(s, e, device=dev).unsqueeze(0)        # LOGICAL
        cache_pos = torch.arange(Lc, Lc + clen, device=dev)          # PHYSICAL
        with torch.no_grad():
            out = model(chunk_ids, past_key_values=compact_cache, use_cache=True,
                        position_ids=pos_ids, cache_position=cache_pos)
        full = out.past_key_values.to_legacy_cache()
        src_pos = torch.arange(s, e, device=dev)
        new_compact = []
        with torch.no_grad():
            for li, (k, v) in enumerate(full):
                k_ch = k[0, :, Lc:, :]                    # [H, clen, d] raw
                v_ch = v[0, :, Lc:, :]
                k_un = apply_rope(k_ch, src_pos, theta, inverse=True)
                c_k, c_v, _ = comp(k_un, v_ch, src_pos)   # [H, t_chunk, d]
                pk, pv = prior[li]
                if pk is not None:                        # append to accumulated
                    c_k = torch.cat([pk, c_k], dim=1)
                    c_v = torch.cat([pv, c_v], dim=1)
                new_compact.append((c_k.unsqueeze(0), c_v.unsqueeze(0)))
                prior[li] = (c_k, c_v)
        del out, full                                     # FREE raw chunk KV
        gc.collect(); torch.cuda.empty_cache()
        compact_cache = DynamicCache.from_legacy_cache(tuple(new_compact))
        pk_live = torch.cuda.max_memory_allocated(di) / GiB
        chunk_peaks.append(pk_live)
        print(f"  chunk {ci+1}/{n_chunks} [{s}:{e}] compact_len="
              f"{compact_cache.get_seq_length()} peak_live={pk_live:.2f} GiB "
              f"resident={live(di):.2f} GiB")
    chunked_peak = max(chunk_peaks)

    # ---- decode from final compact cache at the logical end ----
    phys = compact_cache.get_seq_length()
    nxt = torch.randint(0, cfg.vocab_size, (1, 1), device=dev)
    with torch.no_grad():
        dout = model(nxt, past_key_values=compact_cache, use_cache=True,
                     position_ids=torch.tensor([[args.T]], device=dev),
                     cache_position=torch.tensor([phys], device=dev))
    finite = torch.isfinite(dout.logits).all().item()

    fp = "OOM/inf" if full_peak == float("inf") else f"{full_peak:.2f} GiB"
    ratio = ("inf (full prefill OOMs)" if full_peak == float("inf")
             else f"{full_peak/chunked_peak:.2f}x lower")
    print(f"\n[result] full-prefill peak={fp} | "
          f"chunked-streaming peak={chunked_peak:.2f} GiB ({ratio}); "
          f"final compact_len={phys} (logical {args.T}); "
          f"decode@logical={args.T} finite={finite}")
    ok = finite and chunked_peak < full_peak and phys == n_chunks * args.t_chunk
    print(f"CHUNKED-STREAMING {'PASS' if ok else 'CHECK'}: chunked prefill "
          f"bounds the construction peak below full prefill, source freed per "
          f"chunk, decode finite at the logical position. Quality not evaluated "
          f"(untrained compactor).")


if __name__ == "__main__":
    main()
