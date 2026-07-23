#!/usr/bin/env python3
"""STILL - I/O accounting, query-reuse break-even, SSD-offload projection.

Grounds the storage decision in measured bytes + measured construction/decode
time on the real Qwen3-4B chunked path, plus realistic device bandwidths. It
answers the user's fourth focus area: does tiering STILL's source KV or its
destination compact KV to NVMe make systems sense? Quality not evaluated.
"""
import os
import argparse, gc, sys, statistics
import torch
from transformers import AutoModelForCausalLM, DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from still_compactor import (STILLCompactorLayer, apply_rope,
                             cache_to_legacy, legacy_to_cache)  # noqa: E402

GiB = 2 ** 30
MiB = 2 ** 20
BPT = 36 * 8 * 128 * 2 * 2                         # Qwen3-4B KV bytes/token
# realistic sustained bandwidths (prune): HBM ~ W7900, NVMe Gen5/Gen4, PCIe4 x16
BW_HBM = 800 * GiB                                # ~0.8 TB/s effective
BW_NVME_G5, BW_NVME_G4 = 14 * GiB, 7 * GiB
BW_PCIE4 = 26 * GiB


def timed(fn, di, warmup=1, iters=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(di)
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize(di)
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


def build_compact(model, comp, ids, T, chunk, tchunk, theta, di):
    cc, prior = None, [(None, None)] * model.config.num_hidden_layers
    dev = ids.device
    for ci in range((T + chunk - 1) // chunk):
        s, e = ci * chunk, min(ci * chunk + chunk, T)
        Lc = 0 if cc is None else cc.get_seq_length()
        pos = torch.arange(s, e, device=dev)
        with torch.no_grad():
            out = model(ids[:, s:e], past_key_values=cc, use_cache=True,
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
        cc = legacy_to_cache(tuple(new))
    return cc


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
    comp = STILLCompactorLayer(H, d, t=64, base_theta=theta).to(dev, torch.bfloat16)

    # --- measured construction + decode times ---
    print("[measure] chunked construction + decode time (real Qwen3-4B):")
    tbuild, tprefill, tdecode = {}, {}, {}
    for T in (8192, 32768):
        ids = torch.randint(0, cfg.vocab_size, (1, T), device=dev)
        tb = timed(lambda: build_compact(model, comp, ids, T, 2048, 64, theta, di),
                   di, warmup=0, iters=1)
        cc = build_compact(model, comp, ids, T, 2048, 64, theta, di)
        nxt = torch.randint(0, cfg.vocab_size, (1, 1), device=dev)
        phys = cc.get_seq_length()

        def dec():
            with torch.no_grad():
                model(nxt, past_key_values=legacy_to_cache(
                    cache_to_legacy(cc)), use_cache=True,
                    position_ids=torch.tensor([[T]], device=dev),
                    cache_position=torch.tensor([phys], device=dev))
        td = timed(dec, di)
        tbuild[T], tdecode[T] = tb, td
        print(f"  T={T:>6}: build(prefill+compact)={tb/1000:.2f}s  "
              f"decode/step(compact)={td:.2f}ms  compact_len={phys}")
        del cc, ids; gc.collect(); torch.cuda.empty_cache()

    # --- I/O byte accounting per new context (chunked STILL) ---
    print("\n[I/O bytes per new context, chunked STILL] "
          "(source KV read once per chunk from HBM, compact written to HBM):")
    for T in (8192, 32768, 65536):
        src = T * BPT / GiB
        cpt = (T // 32) * BPT / MiB
        print(f"  T={T:>6}: HBM source-read ~{src:.2f} GiB, "
              f"HBM compact-write ~{cpt:.0f} MiB, NVMe 0, PCIe 0 (untiered)")

    # --- query-reuse break-even: STILL vs full-context repeated prefill ---
    print("\n[break-even] cost after Q queries; full-context re-prefills each "
          "query, STILL builds once then decodes cheap:")
    for T in (8192, 32768):
        P = tbuild[T]                                 # build ~= a prefill+compact
        # full-context prefill alone (no compaction) approximated by prefill part
        # use build time as the STILL one-time cost; full re-prefill per query:
        for Q in (1, 2, 4, 8, 16, 64):
            still = P + Q * tdecode[T]
            full = Q * P                              # re-prefill each query
            winner = "STILL" if still < full else "full-ctx"
            if Q in (1, 2, 8, 64):
                print(f"  T={T:>6} Q={Q:>3}: STILL={still/1000:6.2f}s  "
                      f"full-ctx={full/1000:6.2f}s  -> {winner}")

    # --- SSD-offload projection ---
    print("\n[SSD-offload projection]")
    for T in (32768, 65536):
        cpt_b = (T // 32) * BPT
        src_b = T * BPT
        load_g5 = cpt_b / BW_NVME_G5 * 1000
        recompute = tbuild.get(T, tbuild[32768]) * (T / 32768)
        print(f"  T={T}: compact KV={cpt_b/MiB:.0f} MiB, full source KV="
              f"{src_b/GiB:.2f} GiB")
        print(f"    - source-KV offload: reads once per chunk then freed, NO "
              f"reuse -> NVMe adds latency on the critical path for zero gain: "
              f"NO.")
        print(f"    - compact-KV offload (inactive contexts): reload "
              f"{cpt_b/MiB:.0f} MiB from Gen5 NVMe ~{load_g5:.0f} ms vs "
              f"recompute ~{recompute/1000:.1f} s -> reload ~"
              f"{recompute/load_g5/1000*1000:.0f}x cheaper: viable, but this is "
              f"prefix caching; STILL's compact KV is {32}x smaller than full "
              f"KV so ~{32}x more contexts fit per byte.")
    print("\nVerdict: do NOT tier STILL's source KV (read-once, no reuse). "
          "Tiering the small compact KV of INACTIVE reused contexts to NVMe is "
          "viable and converges with CAS/LMCache prefix offload; STILL's win "
          "there is that its compact artifact is ~32x smaller, so the same "
          "HBM/NVMe budget holds ~32x more contexts. On the 48 GB W7900 the "
          "compact KV is small enough (36-288 MiB) that HBM already holds 100+ "
          "contexts, so NVMe tiering only matters at high context cardinality.")


if __name__ == "__main__":
    main()
