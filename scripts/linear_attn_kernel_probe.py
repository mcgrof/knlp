#!/usr/bin/env python3
"""Probe which linear-attention kernels actually work on this GPU.

Importing a kernel library is not the same as being able to use it, and a
kernel that returns finite numbers for one input is not the same as a kernel
that is correct. This walks the delta-rule family in flash-linear-attention
and reports, per entry point, whether it runs, whether the output is finite,
and whether the chunked (training) path agrees with the recurrent (decode)
path on the same inputs.

The distinction matters when planning work: the chunked kernels carry matched
training runs, while the recurrent kernels carry streaming generation and any
state save/restore test. A machine can be fine for one and useless for the
other, and that should be discovered before a training plan depends on it.

Inputs follow the contract the library's own layers use: keys and queries
normalized inside the kernel, a sigmoid gate, and a log-domain decay. Getting
that contract wrong produces overflow that looks exactly like a broken kernel,
so the contract is applied here in one place rather than guessed per call.

    python3 scripts/linear_attn_kernel_probe.py --json-out probe.json
"""

import argparse
import json
import platform

import torch
import torch.nn.functional as F


def environment():
    env = {
        "host": platform.node(),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env["gpu"] = p.name
        env["arch"] = getattr(p, "gcnArchName", None) or f"sm_{p.major}{p.minor}"
    for mod in ("triton", "fla", "titans_pytorch"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "present")
        except Exception as exc:
            env[mod] = f"MISSING ({type(exc).__name__})"
    return env


def make_inputs(B, T, H, D, dtype, device, seed):
    """The input contract the library's own layers use."""
    torch.manual_seed(seed)
    mk = lambda: torch.randn(B, T, H, D, device=device, dtype=dtype)
    q, k, v = mk(), mk(), mk()
    beta = torch.sigmoid(torch.randn(B, T, H, device=device)).to(dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device).float()).to(dtype)
    return q, k, v, beta, g


def call(fn, gated, q, k, v, beta, g):
    kw = dict(use_qk_l2norm_in_kernel=True)
    out = fn(q, k, v, g, beta, **kw) if gated else fn(q, k, v, beta, **kw)
    out = out[0] if isinstance(out, tuple) else out
    return out.float()


def probe(B, T, H, D, dtypes, device="cuda"):
    from fla.ops import (
        chunk_delta_rule,
        chunk_gated_delta_rule,
        fused_recurrent_delta_rule,
        fused_recurrent_gated_delta_rule,
    )

    families = {
        "delta_rule": dict(
            gated=False, chunk=chunk_delta_rule, recurrent=fused_recurrent_delta_rule
        ),
        "gated_delta_rule": dict(
            gated=True,
            chunk=chunk_gated_delta_rule,
            recurrent=fused_recurrent_gated_delta_rule,
        ),
    }

    rows = []
    for dtype in dtypes:
        for fam, spec in families.items():
            args = make_inputs(B, T, H, D, dtype, device, seed=0)
            outs = {}
            for path in ("chunk", "recurrent"):
                row = {
                    "family": fam,
                    "path": path,
                    "dtype": str(dtype).split(".")[-1],
                    "T": T,
                }
                try:
                    o = call(spec[path], spec["gated"], *args)
                    torch.cuda.synchronize()
                    finite = torch.isfinite(o)
                    row["runs"] = True
                    row["finite"] = bool(finite.all())
                    row["nonfinite_fraction"] = float((~finite).float().mean())
                    outs[path] = o
                except Exception as exc:
                    row["runs"] = False
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                rows.append(row)
            # do the two paths agree where both produced numbers?
            agree = {
                "family": fam,
                "dtype": str(dtype).split(".")[-1],
                "check": "chunk_vs_recurrent",
            }
            if len(outs) == 2 and all(torch.isfinite(o).all() for o in outs.values()):
                a, b = outs["chunk"], outs["recurrent"]
                agree["cosine"] = float(
                    F.cosine_similarity(a.flatten(), b.flatten(), dim=0)
                )
                agree["rel_l2"] = float((a - b).norm() / b.norm())
                agree["comparable"] = True
            else:
                agree["comparable"] = False
                agree["reason"] = "a path did not produce finite output"
            rows.append(agree)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seqlen", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--json-out", default="linear_attn_kernel_probe.json")
    args = ap.parse_args()

    report = {"environment": environment(), "results": []}
    print(json.dumps(report["environment"], indent=1), flush=True)
    if not torch.cuda.is_available():
        print("no GPU visible; nothing to probe", flush=True)
        json.dump(report, open(args.json_out, "w"), indent=1)
        return 1

    for T in args.seqlen:
        report["results"].extend(
            probe(
                args.batch,
                T,
                args.heads,
                args.head_dim,
                [torch.bfloat16, torch.float16],
            )
        )

    for r in report["results"]:
        if r.get("check") == "chunk_vs_recurrent":
            if r["comparable"]:
                print(
                    f"  {r['family']:18s} {r['dtype']:9s} chunk vs recurrent: "
                    f"cos={r['cosine']:.6f} rel={r['rel_l2']:.5f}",
                    flush=True,
                )
            else:
                print(
                    f"  {r['family']:18s} {r['dtype']:9s} chunk vs recurrent: "
                    f"NOT COMPARABLE ({r['reason']})",
                    flush=True,
                )
        else:
            state = (
                "ok"
                if r.get("finite")
                else (
                    "NONFINITE %.3f" % r["nonfinite_fraction"]
                    if r.get("runs")
                    else "ERROR"
                )
            )
            print(
                f"  T={r['T']:5d} {r['family']:18s} {r['path']:10s} "
                f"{r['dtype']:9s} {state}",
                flush=True,
            )

    json.dump(report, open(args.json_out, "w"), indent=1)
    print(f"wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
