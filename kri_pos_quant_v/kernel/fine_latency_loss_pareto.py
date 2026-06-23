#!/usr/bin/env python3
"""Fine latency/loss Pareto for low-rank V: measure the FULL loss impact (delta-NLL,
KL, argmax-flip) at every rank on the real model, held-out, paired with the decode-
attention kernel latency -- no gating at "lossless", every operating point measured.

Uniform low-rank-V on all layers (static per-(layer,kv-head) SVD basis fit on a fit
split), held-out eval. Fine rank grid through the near-lossless regime so the small-
loss frontier is resolved.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# v_lowrank_calibrate lives in the calibration dir; allow an explicit path
for p in (
    "/data/knlp-key-results/kri-pos-quant-v-20260609/calibration",
):
    if Path(p).exists():
        sys.path.insert(0, p)

from v_lowrank_calibrate import VCompressHook, get_layers, run_model, capture_logits  # noqa: E402
from vlowrank_kernel import dense_decode_triton, lowrank_decode  # noqa: E402


def cuda_p50(fn, warmup=12, iters=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def windows(tok, n, seq_len, skip):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    buf, w, sk = [], [], 0
    for t in ds["text"]:
        if len(t.strip()) < 40:
            continue
        buf.extend(tok(t).input_ids)
        while len(buf) >= seq_len:
            if sk < skip:
                sk += 1
            else:
                w.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(w) >= n:
                return torch.tensor(w)
    return torch.tensor(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fit", type=int, default=12)
    ap.add_argument("--hold", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--ranks", default="24,32,40,48,56,64,72,80,88,96,104,112,120,128")
    ap.add_argument("--lat-heads", type=int, default=32)
    ap.add_argument("--lat-ctx", type=int, default=32768)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev)
    model.eval()
    torch.set_grad_enabled(False)
    cfg = model.config
    Hkv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    dh = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    layers = get_layers(model)
    nL = len(layers)
    ranks = [int(x) for x in args.ranks.split(",")]
    print(f"[{args.model}] L={nL} Hkv={Hkv} dh={dh}", flush=True)

    fit = windows(tok, args.fit, args.seq_len, 0)
    hold = windows(tok, args.hold, args.seq_len, args.fit + 5)

    # fit static per-(layer,kv-head) SVD bases on fit split
    caps = {i: [] for i in range(nL)}
    hs = [layers[i].self_attn.v_proj.register_forward_hook(
            lambda m, inp, out, i=i: caps[i].append(
                out.detach().reshape(-1, Hkv, dh).float().cpu()))
          for i in range(nL)]
    _ = capture_logits(model, fit, dev)
    for h in hs:
        h.remove()
    basis = {}
    for i in range(nL):
        Vi = torch.cat(caps[i]).to(dev)
        Bi = torch.empty(Hkv, dh, dh, device=dev)
        for hh in range(Hkv):
            _, _, Wt = torch.linalg.svd(Vi[:, hh, :], full_matrices=False)
            Bi[hh] = Wt.t()
        basis[i] = Bi
        caps[i] = None

    base_logits = capture_logits(model, hold, dev)
    base = run_model(model, hold, dev)
    base_nll = base["nll"]
    base_ppl = math.exp(base_nll)
    print(f"held-out base ppl={base_ppl:.3f} nll={base_nll:.4f}", flush=True)

    # decode-attn latency baseline (synthetic shape)
    H, T = args.lat_heads, args.lat_ctx
    Q = torch.randn(H, 1, dh, device=dev, dtype=torch.bfloat16)
    K = torch.randn(H, T, dh, device=dev, dtype=torch.bfloat16)
    Vt = torch.randn(H, T, dh, device=dev, dtype=torch.bfloat16)
    t_dense = cuda_p50(lambda: dense_decode_triton(Q, K, Vt))

    rows = []
    for r in ranks:
        hk = [layers[i].self_attn.v_proj.register_forward_hook(
                VCompressHook(basis[i], Hkv, dh, r))
              for i in range(nL) if r < dh]
        m = run_model(model, hold, dev, base_logits=base_logits)
        for h in hk:
            h.remove()
        dnll = m["nll"] - base_nll
        kl = m["kl"]
        flip = 1.0 - m["argmax_agree"]
        # latency at this rank
        proj = torch.randn(H, T, r, device=dev, dtype=torch.bfloat16)
        Bb = torch.randn(H, dh, r, device=dev, dtype=torch.float32)
        t_low = cuda_p50(lambda: lowrank_decode(Q, K, proj, Bb))
        rows.append({
            "rank": r, "v_read_ratio": (dh + r) / (2.0 * dh),
            "attn_speedup": t_dense / t_low,
            "dnll": dnll, "kl": kl, "flip": flip,
            "ppl": math.exp(base_nll + dnll), "ppl_pct": 100 * (math.exp(dnll) - 1),
        })
        print(f"  r={r:>3} spdup={t_dense/t_low:>4.2f}x  dNLL={dnll:+.4f} "
              f"kl={kl:.2e} flip={flip*100:.2f}% ppl={math.exp(base_nll+dnll):.3f} "
              f"(+{100*(math.exp(dnll)-1):.2f}%)", flush=True)

    Path(args.out).write_text(json.dumps({
        "model": args.model, "base_ppl": base_ppl, "base_nll": base_nll,
        "lat_heads": H, "lat_ctx": T, "dense_ms": t_dense, "rows": rows}, indent=2))
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
