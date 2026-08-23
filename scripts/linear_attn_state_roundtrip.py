#!/usr/bin/env python3
"""Check that a linear-attention state can be saved, restored, and continued.

This is the test that decides whether a bounded recurrent state is genuinely
cacheable rather than secretly recomputed: run a prefix, keep the state it
ends in, restore that state into a fresh call, continue with the suffix, and
require the result to match an uninterrupted run.

It deliberately uses only the chunked kernel, with `output_final_state` to
save and `initial_state` to restore. The dedicated recurrent kernel is not
required for any of this, which matters on hardware where that kernel
misbehaves — see the linear_attn_kernel_probe script for how to check.

Two modes are covered:

    split      one prefix and one suffix, at several split points
    stepwise   one token at a time, carrying state, the decode pattern

    python3 scripts/linear_attn_state_roundtrip.py
"""

import argparse
import json

import torch
import torch.nn.functional as F


def make(B, T, H, D, device, dtype, seed=0):
    torch.manual_seed(seed)
    mk = lambda: torch.randn(B, T, H, D, device=device, dtype=dtype)
    q, k, v = mk(), mk(), mk()
    beta = torch.sigmoid(torch.randn(B, T, H, device=device)).to(dtype)
    # the decay is kept in 32-bit, which is what the kernels expect
    g = F.logsigmoid(torch.randn(B, T, H, device=device).float())
    return q, k, v, beta, g


def compare(a, b):
    return dict(
        cosine=float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
        rel_l2=float((a - b).norm() / b.norm()),
        finite=bool(torch.isfinite(a).all()),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--stepwise-len", type=int, default=64)
    ap.add_argument("--json-out", default="linear_attn_state_roundtrip.json")
    args = ap.parse_args()

    from fla.ops import chunk_gated_delta_rule as C

    dev, dt = "cuda", torch.bfloat16
    kw = dict(use_qk_l2norm_in_kernel=True)
    B, T, H, D = args.batch, args.seqlen, args.heads, args.head_dim
    report = {"config": vars(args), "split": [], "stepwise": None}

    q, k, v, beta, g = make(B, T, H, D, dev, dt)
    full = C(q, k, v, g, beta, **kw)
    full = (full[0] if isinstance(full, tuple) else full).float()

    for split in (T // 4, T // 2, 3 * T // 4):
        head = C(
            q[:, :split],
            k[:, :split],
            v[:, :split],
            g[:, :split],
            beta[:, :split],
            output_final_state=True,
            **kw,
        )
        o1, state = head[0].float(), head[1]
        tail = C(
            q[:, split:],
            k[:, split:],
            v[:, split:],
            g[:, split:],
            beta[:, split:],
            initial_state=state,
            **kw,
        )
        o2 = (tail[0] if isinstance(tail, tuple) else tail).float()
        r = compare(torch.cat([o1, o2], dim=1), full)
        r["split"] = split
        report["split"].append(r)
        print(
            f"  split at {split:5d}: cosine={r['cosine']:.6f} rel={r['rel_l2']:.5f}",
            flush=True,
        )

    # one token at a time, the pattern decoding would use
    Ts = args.stepwise_len
    q, k, v, beta, g = make(B, Ts, H, D, dev, dt, seed=1)
    ref = C(q, k, v, g, beta, **kw)
    ref = (ref[0] if isinstance(ref, tuple) else ref).float()
    state, outs = None, []
    for t in range(Ts):
        r = C(
            q[:, t : t + 1],
            k[:, t : t + 1],
            v[:, t : t + 1],
            g[:, t : t + 1],
            beta[:, t : t + 1],
            initial_state=state,
            output_final_state=True,
            **kw,
        )
        outs.append(r[0].float())
        state = r[1]
    r = compare(torch.cat(outs, dim=1), ref)
    report["stepwise"] = r
    print(
        f"  stepwise ({Ts} tokens): cosine={r['cosine']:.6f} rel={r['rel_l2']:.5f}",
        flush=True,
    )

    json.dump(report, open(args.json_out, "w"), indent=1)
    ok = all(x["cosine"] > 0.9999 for x in report["split"]) and (
        report["stepwise"]["cosine"] > 0.9999
    )
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
