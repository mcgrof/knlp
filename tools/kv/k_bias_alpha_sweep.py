"""Deliverable 4: causal alpha sweep on the K-projection bias.

Patches ONLY the K bias by a factor alpha (Q/V untouched), and at each alpha measures the
QUANTIZATION-SPECIFIC gap = error(FP8 vs BF16) at that alpha. Scaling the bias may damage the
model even in BF16, so the BF16-at-alpha run is the per-alpha baseline; we only claim support if
the FP8-vs-BF16 gap GROWS with alpha (more bias magnitude -> more FP8-specific damage) and pre-bias
FP8 FLATTENS that curve. K16/V8 should stay stable unless alpha breaks BF16 itself. Biasless
controls have nothing to scale -> flat (printed as a sanity check).
"""

import argparse
import os

import torch
import torch.nn.functional as F

import k_bias_common as kbc

ALPHAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]


@torch.no_grad()
def run_cell(model, infos, ids_list, device, cell, fp8_layout):
    h = kbc.KQuantHarness(model, infos, cell=cell, fp8_layout=fp8_layout)
    h.install()
    outs = []
    try:
        for ids in ids_list:
            t = torch.tensor(ids).unsqueeze(0).to(device)
            outs.append(model(t).logits[0].float().cpu())
    finally:
        h.remove()
    return outs


def err_vs(base, other):
    import statistics as S

    me = [(b - c).abs().mean().item() for b, c in zip(base, other)]
    nl = []
    for b, c in zip(base, other):
        tgt = b.argmax(-1)
        nl.append(
            (
                -F.log_softmax(c, -1).gather(-1, tgt.unsqueeze(-1)).mean()
                + F.log_softmax(b, -1).gather(-1, tgt.unsqueeze(-1)).mean()
            ).item()
        )
    return S.mean(me), S.mean(nl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--short-name", default=None)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--num-prompts", type=int, default=6)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp8-layout", default="per_tensor")
    ap.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS))
    ap.add_argument("--output-dir", default="artifacts/k_bias_audit/alpha")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()
    sn = args.short_name or args.model.split("/")[-1]
    trc = args.trust_remote_code or any(
        x in args.model.lower() for x in ("deepseek", "phi")
    )
    alphas = [float(a) for a in args.alphas.split(",")]

    model, tok = kbc.load_model(args.model, args.dtype, args.device, trc)
    infos = kbc.discover_attention(model)
    has_k_bias = any(i["has_k_bias"] for i in infos)
    ids_list = kbc.calib_prompts(tok, n=args.num_prompts, seq_len=args.seq_len)
    print(
        f"[alpha] {sn}: has_k_bias={has_k_bias}, {len(ids_list)}x{args.seq_len}, alphas={alphas}"
    )

    rows = []
    for alpha in alphas:
        with kbc.AlphaKBiasPatch(infos, alpha):
            base = run_cell(
                model, infos, ids_list, args.device, "bf16", args.fp8_layout
            )
            fp8 = run_cell(model, infos, ids_list, args.device, "fp8", args.fp8_layout)
            pre = run_cell(
                model, infos, ids_list, args.device, "prebias_fp8", args.fp8_layout
            )
            k16 = run_cell(
                model, infos, ids_list, args.device, "k16v8", args.fp8_layout
            )
        fp8_gap, fp8_nll = err_vs(base, fp8)
        pre_gap, pre_nll = err_vs(base, pre)
        k16_gap, _ = err_vs(base, k16)
        # absolute BF16 drift vs alpha=1.0 baseline not needed; report the GAP (quant-specific)
        row = dict(
            model=sn,
            alpha=alpha,
            has_k_bias=has_k_bias,
            fp8_gap=fp8_gap,
            fp8_nll_gap=fp8_nll,
            prebias_gap=pre_gap,
            prebias_nll_gap=pre_nll,
            k16v8_gap=k16_gap,
        )
        rows.append(row)
        print(
            f"  alpha={alpha:<4} fp8_gap={fp8_gap:.4f} prebias_gap={pre_gap:.4f} "
            f"k16v8_gap={k16_gap:.4f}"
        )

    # K-FP8-SPECIFIC gap = fp8_gap - k16v8_gap (removes V-quant + shared model degradation).
    # alpha=0 zeroes a load-bearing bias -> breaks BF16 itself, so it is degenerate; the causal
    # read uses alpha>0 (bias present). Support = K-specific FP8 gap is large with bias present
    # AND pre-bias removes most of it (prebias residual small).
    import statistics as S

    pos = [r for r in rows if r["alpha"] > 0]
    for r in rows:
        r["k_specific_fp8_gap"] = r["fp8_gap"] - r["k16v8_gap"]
        r["prebias_residual"] = r["prebias_gap"] - r["k16v8_gap"]
    k_spec = S.mean([r["k_specific_fp8_gap"] for r in pos]) if pos else 0.0
    pre_res = S.mean([r["prebias_residual"] for r in pos]) if pos else 0.0
    a_lo = min(pos, key=lambda r: r["alpha"]) if pos else rows[0]
    a_hi = max(pos, key=lambda r: r["alpha"]) if pos else rows[-1]
    k_spec_slope = (a_hi["k_specific_fp8_gap"] - a_lo["k_specific_fp8_gap"]) / max(
        a_hi["alpha"] - a_lo["alpha"], 1e-9
    )
    # causal: K-FP8 gap appears with bias and pre-bias kills it (prebias residual << k-specific)
    causal = bool(has_k_bias and k_spec > 0.1 and pre_res < 0.34 * max(k_spec, 1e-9))
    summary = dict(
        model=sn,
        model_id=args.model,
        has_k_bias=has_k_bias,
        mean_k_specific_fp8_gap=k_spec,
        mean_prebias_residual=pre_res,
        k_specific_slope_over_alpha=k_spec_slope,
        fp8_gap_at_min_pos_alpha=a_lo["fp8_gap"],
        fp8_gap_at_max_alpha=a_hi["fp8_gap"],
        prebias_kills_fp8_gap=bool(pre_res < 0.34 * max(k_spec, 1e-9)),
        alpha_causal=causal,
    )
    print(
        f"  mean K-specific FP8 gap (alpha>0)={k_spec:.4f}  prebias residual={pre_res:.4f}  "
        f"K-spec slope={k_spec_slope:.4f}/alpha  causal={causal}"
    )

    base_out = os.path.join(args.output_dir, sn)
    kbc.write_csv(
        os.path.join(base_out, "alpha_sweep.csv"),
        rows,
        [
            "model",
            "alpha",
            "has_k_bias",
            "fp8_gap",
            "fp8_nll_gap",
            "prebias_gap",
            "prebias_nll_gap",
            "k16v8_gap",
            "k_specific_fp8_gap",
            "prebias_residual",
        ],
    )
    kbc.write_json(os.path.join(base_out, "alpha_summary.json"), summary)
    with open(os.path.join(base_out, "alpha_sweep.jsonl"), "w") as f:
        import json

        for r in rows:
            f.write(json.dumps(r) + "\n")
    _append(os.path.join(args.output_dir, "alpha_sweep_summary.csv"), summary)
    _plot(base_out, rows, sn)


def _append(path, summary):
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not exists:
            w.writeheader()
        w.writerow(summary)


def _plot(base, rows, sn):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    a = [r["alpha"] for r in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(a, [r["fp8_gap"] for r in rows], "o-", label="FP8-BF16 gap")
    plt.plot(a, [r["prebias_gap"] for r in rows], "s-", label="pre-bias FP8 gap")
    plt.plot(a, [r["k16v8_gap"] for r in rows], "^-", label="K16/V8 gap")
    plt.xlabel("K-bias alpha")
    plt.ylabel("quant-specific logit gap")
    plt.title(f"{sn}: causal alpha sweep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base, "alpha_gap_curve.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
