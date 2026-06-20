"""Deliverable 3: fake-FP8 K-cache quant comparison (normal vs pre-bias vs K16/V8).

Answers Q4/Q5: does symmetric FP8 K collapse on the sensitive models, and does PRE-BIAS FP8 K
recover it while a biasless control sees pre-bias as a no-op? Runs each cell against the BF16
baseline on a small calibration set and measures logit error / NLL delta / top-1 agreement, plus
the paper's INT8-vs-INT6 K-only error ratio so this connects to the existing classifier.

Cells: bf16 (baseline), fp8 (normal post-RoPE symmetric FP8 K + FP8 V), prebias_fp8 (faithful
pre-RoPE residual fix), k16v8 (K native, V FP8), fp8_pc (per-channel K), int8 (INT8 K + FP8 V).
"""

import argparse
import os

import torch
import torch.nn.functional as F

import k_bias_common as kbc

CELLS = ["fp8", "prebias_fp8", "k16v8", "fp8_pc", "int8"]


@torch.no_grad()
def logits_for(model, infos, ids_list, device, cell, fp8_layout):
    outs = []
    h = kbc.KQuantHarness(model, infos, cell=cell, fp8_layout=fp8_layout)
    h.install()
    try:
        for ids in ids_list:
            t = torch.tensor(ids).unsqueeze(0).to(device)
            lg = model(t).logits[0].float().cpu()  # [T, vocab]
            outs.append(lg)
    finally:
        h.remove()
    return outs


def compare(base, cellL):
    # per-prompt logit error + top-1 agreement + NLL delta, averaged
    me, mx, t1, nll = [], [], [], []
    for b, c in zip(base, cellL):
        diff = (b - c).abs()
        me.append(diff.mean().item())
        mx.append(diff.max().item())
        t1.append((b.argmax(-1) == c.argmax(-1)).float().mean().item())
        lpb = F.log_softmax(b, -1)
        lpc = F.log_softmax(c, -1)
        # NLL delta on argmax(base) tokens (teacher = base's own prediction)
        tgt = b.argmax(-1)
        nll_b = -lpb.gather(-1, tgt.unsqueeze(-1)).mean().item()
        nll_c = -lpc.gather(-1, tgt.unsqueeze(-1)).mean().item()
        nll.append(nll_c - nll_b)
    import statistics as S

    return dict(
        mean_logit_err=S.mean(me),
        max_logit_err=max(mx),
        top1_agree=S.mean(t1),
        nll_delta=S.mean(nll),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--short-name", default=None)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp8-layout", default="per_tensor")
    ap.add_argument("--output-dir", default="artifacts/k_bias_audit/fp8")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()
    sn = args.short_name or args.model.split("/")[-1]
    trc = args.trust_remote_code or any(
        x in args.model.lower() for x in ("deepseek", "phi")
    )

    model, tok = kbc.load_model(args.model, args.dtype, args.device, trc)
    infos = kbc.discover_attention(model)
    has_k_bias = any(i["has_k_bias"] for i in infos)
    ids_list = kbc.calib_prompts(tok, n=args.num_prompts, seq_len=args.seq_len)
    print(
        f"[fp8] {sn}: has_k_bias={has_k_bias}, {len(ids_list)} prompts x {args.seq_len}"
    )

    base = logits_for(model, infos, ids_list, args.device, "bf16", args.fp8_layout)
    rows = []
    summary = dict(model=sn, model_id=args.model, has_k_bias=has_k_bias)
    for cell in CELLS:
        cl = logits_for(model, infos, ids_list, args.device, cell, args.fp8_layout)
        m = compare(base, cl)
        rows.append(dict(model=sn, cell=cell, **m))
        summary[f"{cell}_mean_logit_err"] = m["mean_logit_err"]
        summary[f"{cell}_top1"] = m["top1_agree"]
        print(
            f"  {cell:<12} mean_logit_err={m['mean_logit_err']:.4f} "
            f"top1={m['top1_agree']:.3f} nll_delta={m['nll_delta']:.4f}"
        )

    # paper INT8-vs-INT6 K-only ratio classifier
    k8 = compare(
        base, logits_for(model, infos, ids_list, args.device, "kint8", args.fp8_layout)
    )
    k6 = compare(
        base, logits_for(model, infos, ids_list, args.device, "kint6", args.fp8_layout)
    )
    ratio = k6["mean_logit_err"] / max(k8["mean_logit_err"], 1e-9)
    summary["int8_int6_err_ratio"] = ratio
    summary["kint8_err"] = k8["mean_logit_err"]
    summary["kint6_err"] = k6["mean_logit_err"]
    # recovery signals
    fp8_err = summary["fp8_mean_logit_err"]
    pre_err = summary["prebias_fp8_mean_logit_err"]
    summary["prebias_recovery_ratio"] = pre_err / max(fp8_err, 1e-9)  # <1 = recovers
    print(f"  INT8/INT6 K err ratio = {ratio:.2f}  (paper classifier; >~3 => fragile)")
    print(
        f"  prebias recovery: fp8_err={fp8_err:.4f} -> prebias_err={pre_err:.4f} "
        f"(ratio {summary['prebias_recovery_ratio']:.3f})"
    )

    base_out = os.path.join(args.output_dir, sn)
    kbc.write_csv(
        os.path.join(base_out, "fp8_bias_probe_cells.csv"),
        rows,
        ["model", "cell", "mean_logit_err", "max_logit_err", "top1_agree", "nll_delta"],
    )
    kbc.write_json(os.path.join(base_out, "fp8_summary.json"), summary)
    _append(os.path.join(args.output_dir, "fp8_bias_probe_summary.csv"), summary)
    with open(os.path.join(base_out, "fp8_bias_probe.jsonl"), "w") as f:
        import json

        for r in rows:
            f.write(json.dumps(r) + "\n")


def _append(path, summary):
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not exists:
            w.writeheader()
        w.writerow(summary)


if __name__ == "__main__":
    main()
