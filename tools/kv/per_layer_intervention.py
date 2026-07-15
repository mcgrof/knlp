"""Bidirectional per-layer K-format intervention matrix (KVTuner mechanism test, rungs T0-T2).

Question (from KVTUNER_ANALYSIS_PLAN.md): is there a small, causally identifiable subset of
Qwen2.5-7B layers whose K FORMAT controls the symmetric-FP8-K failure, and can a static checkpoint
feature (K-bias magnitude) find it? V is held at FP8 throughout; K states are {BF16, FP8-e4m3,
affine-INT8}. Two directions, because nonlinear error accumulation makes them differ:

  * safe-side demotion  -- from all-K16/V8 (the incumbent asym), drop ONE layer's K to FP8/INT8;
                           measure downstream damage. A cheap-to-demote layer is a byte-saving
                           candidate.
  * failure-side repair -- from all-FP8/V8 (the Qwen failure), restore ONE layer's K to BF16/INT8;
                           measure downstream rescue. A layer that alone rescues is the culprit.

Primary metric: teacher-forced mean NLL over calibration chunks (no baseline needed). Secondary:
top-1 next-token disagreement rate vs the BF16/BF16 reference (a cheap sensitivity proxy, stores
only argmax ids). Per-layer static features (K-bias magnitude stats + depth) are dumped so the
sensitivity curve can be correlated against bias vs a depth-only control.

Free W7900 fake-quant -- no serving kernel. Incremental JSON so a drop mid-run resumes.
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k_bias_common as kbc


@torch.no_grad()
def eval_config(
    model, chunks, device, k_specs, v_specs, k_default, v_default, ref_argmax
):
    """Run every chunk under a PerLayerKVHarness; return mean NLL/PPL + disagreement-vs-ref."""
    h = kbc.PerLayerKVHarness(
        model,
        INFOS,
        k_specs=k_specs,
        v_specs=v_specs,
        k_default=k_default,
        v_default=v_default,
    )
    h.install()
    tot_nll, tot_tok, tot_dis = 0.0, 0, 0
    try:
        for ci, ids in enumerate(chunks):
            t = torch.tensor(ids).unsqueeze(0).to(device)
            logits = model(t).logits[0, :-1].float()
            tgt = t[0, 1:]
            lp = torch.log_softmax(logits, dim=-1)
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            tot_nll += nll.sum().item()
            tot_tok += tgt.numel()
            if ref_argmax is not None:
                am = logits.argmax(-1)
                tot_dis += (am != ref_argmax[ci].to(device)).sum().item()
    finally:
        h.remove()
    mean_nll = tot_nll / max(tot_tok, 1)
    return dict(
        mean_nll=mean_nll,
        ppl=math.exp(min(mean_nll, 20.0)),
        disagree=(tot_dis / max(tot_tok, 1)) if ref_argmax is not None else None,
        tokens=tot_tok,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--num-prompts", type=int, default=12)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument(
        "--k-fail",
        default="fp8:per_tensor",
        help="the K failure format (naive post-RoPE FP8)",
    )
    ap.add_argument(
        "--k-int", default="int8:per_token", help="the affine-INT8 middle K format"
    )
    ap.add_argument(
        "--v", default="fp8:per_tensor", help="V format, held fixed everywhere"
    )
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    BF16 = kbc.parse_spec("bf16")
    KFAIL = kbc.parse_spec(a.k_fail)
    KINT = kbc.parse_spec(a.k_int)
    VSPEC = kbc.parse_spec(a.v)

    print(f"[load] {a.model}", flush=True)
    model, tok = kbc.load_model(a.model, a.dtype, a.device)
    global INFOS
    INFOS = kbc.discover_attention(model)
    nL = len(INFOS)
    chunks = kbc.calib_prompts(tok, n=a.num_prompts, seq_len=a.seq_len)
    print(f"[info] {nL} layers, {len(chunks)} chunks x {a.seq_len} tok", flush=True)

    # per-layer static K-bias features + depth
    feats = []
    for info in INFOS:
        vec = kbc.k_bias_vector(info, device="cpu")
        bs = kbc.bias_stats(vec)
        feats.append(
            dict(layer=info["layer_idx"], depth_frac=info["layer_idx"] / (nL - 1), **bs)
        )
    kbc.write_json(os.path.join(a.output_dir, "layer_features.json"), feats)

    # ---- reference (BF16/BF16, no quant): teacher for disagreement + NLL floor
    print("[ref] BF16/BF16 baseline", flush=True)
    ref_argmax = []
    tot_nll, tot_tok = 0.0, 0
    with torch.no_grad():
        for ids in chunks:
            t = torch.tensor(ids).unsqueeze(0).to(a.device)
            logits = model(t).logits[0, :-1].float()
            tgt = t[0, 1:]
            lp = torch.log_softmax(logits, dim=-1)
            tot_nll += (-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).sum().item()
            tot_tok += tgt.numel()
            ref_argmax.append(logits.argmax(-1).cpu())
    ref_nll = tot_nll / tot_tok
    print(f"[ref] nll={ref_nll:.4f} ppl={math.exp(ref_nll):.3f}", flush=True)

    results = {
        "meta": dict(
            model=a.model,
            nL=nL,
            num_prompts=len(chunks),
            seq_len=a.seq_len,
            k_fail=a.k_fail,
            k_int=a.k_int,
            v=a.v,
            ref_nll=ref_nll,
            ref_ppl=math.exp(ref_nll),
        ),
        "configs": [],
    }
    out_path = os.path.join(a.output_dir, "intervention_results.json")

    def record(name, direction, layer, kfmt, r):
        r = dict(
            name=name,
            direction=direction,
            layer=layer,
            kfmt=kfmt,
            d_nll=r["mean_nll"] - ref_nll,
            **r,
        )
        results["configs"].append(r)
        kbc.write_json(out_path, results)
        dz = f" dis={r['disagree']:.4f}" if r["disagree"] is not None else ""
        print(
            f"[{direction}] {name} L{layer} {kfmt}: nll={r['mean_nll']:.4f} "
            f"dNLL={r['d_nll']:+.4f} ppl={r['ppl']:.2f}{dz}",
            flush=True,
        )

    # ---- anchors (uniform policies)
    record(
        "asym_k16v8",
        "anchor",
        -1,
        "bf16",
        eval_config(model, chunks, a.device, {}, {}, BF16, VSPEC, ref_argmax),
    )
    record(
        "fail_fp8v8",
        "anchor",
        -1,
        a.k_fail,
        eval_config(model, chunks, a.device, {}, {}, KFAIL, VSPEC, ref_argmax),
    )
    record(
        "int8_all_v8",
        "anchor",
        -1,
        a.k_int,
        eval_config(model, chunks, a.device, {}, {}, KINT, VSPEC, ref_argmax),
    )

    # ---- safe-side demotion: base = all-K16/V8, drop layer l's K to FP8 then INT8
    for kfmt, kspec in (("fp8", KFAIL), ("int8", KINT)):
        for l in range(nL):
            record(
                f"demote_{kfmt}",
                "demote",
                l,
                kfmt,
                eval_config(
                    model, chunks, a.device, {l: kspec}, {}, BF16, VSPEC, ref_argmax
                ),
            )

    # ---- failure-side repair: base = all-FP8/V8, restore layer l's K to BF16 then INT8
    for kfmt, kspec in (("bf16", BF16), ("int8", KINT)):
        for l in range(nL):
            record(
                f"repair_{kfmt}",
                "repair",
                l,
                kfmt,
                eval_config(
                    model, chunks, a.device, {l: kspec}, {}, KFAIL, VSPEC, ref_argmax
                ),
            )

    print(f"[done] {len(results['configs'])} configs -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
