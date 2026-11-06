#!/usr/bin/env python3
"""Check and time the fused MoE path against EMO's reference loop (GPU).

The same checks run for the full model and for the fixed expert pool. For
each, the model is loaded with EMO's reference code, which runs one small
matrix multiply per expert in a Python loop. The script times prefill and
decode with that code, records the input and output of three MoE blocks on
one screening prompt, then replaces every MoE block with the fused kernel
(fused_moe.py) and repeats the measurements.

Numerical checks:
- Per block: the fused output and the reference output are each compared
  with an FP32 computation that uses the same routing and BF16 weights.
  The fused error must not exceed 1.5 times the reference error plus 1e-3.
- Whole model (full model only): last-token logits and 32 greedy tokens for
  eight fixed prompts, fused against reference.
- Screen: every screening question is scored with the fused path and its
  chosen answer compared with the reference choice from the earlier screen.

Timing: per-layer and per-MoE-block prefill time (GPU events) at several
prompt lengths, and the mean time of 32 single-token decode steps.

Usage:
    fused_check.py MODEL_DIR POOL.json SCREEN.json REF_PER_EXAMPLE.jsonl OUT_DIR
        [--configs TUNED.json] [--lengths 128 512 1024 2048]
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import apply_pool, load_model, mem_snapshot, runtime_identity  # noqa
from common import write_json  # noqa
from fused_moe import VLLM_SOURCE, ConfigTable, install_fused  # noqa
from pilot import PILOT_PROMPTS, greedy, time_decode, timing_ids  # noqa
from score_screen import score_all  # noqa

CAPTURE_LAYERS = (0, 7, 15)


def param_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


@torch.no_grad()
def capture(model, ids):
    """Inputs and reference outputs of the MoE blocks in CAPTURE_LAYERS."""
    got, hs = {}, []
    for li in CAPTURE_LAYERS:

        def hook(mod, inp, out, li=li):
            y = out[0] if isinstance(out, tuple) else out
            got[li] = (inp[0].detach().clone(), y.detach().clone())

        hs.append(model.model.layers[li].mlp.register_forward_hook(hook))
    model(input_ids=ids)
    for h in hs:
        h.remove()
    return got


@torch.no_grad()
def truth_fp32(m, x):
    """FP32 MoE output with the fused block's routing and BF16 weights."""
    xf = x.reshape(-1, x.shape[-1])
    _, w, ids = m.route(xf.contiguous())
    xf = xf.float()
    out = torch.zeros_like(xf)
    inter = m.w13.size(1) // 2
    for e in range(m.num_routed):
        tok, slot = torch.where(ids == e)
        if tok.numel() == 0:
            continue
        g = xf[tok] @ m.w13[e].float().T
        y = (F.silu(g[:, :inter]) * g[:, inter:]) @ m.w2[e].float().T
        out.index_add_(0, tok, y * w[tok, slot, None])
    for j in range(m.num_shared):
        g = xf @ m.s13[j].float().T
        out += (F.silu(g[:, :inter]) * g[:, inter:]) @ m.s2[j].float().T
    return out.reshape(x.shape)


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@torch.no_grad()
def block_errors(model, cap):
    rows = []
    for li, (x, y_ref) in cap.items():
        m = model.model.layers[li].mlp
        y_f = m(x)[0]
        t = truth_fp32(m, x)
        r_ref, r_f = rel(y_ref, t), rel(y_f, t)
        rows.append(
            {
                "layer": li,
                "tokens": int(x.shape[1]),
                "ref_vs_fp32": r_ref,
                "fused_vs_fp32": r_f,
                "fused_vs_ref": rel(y_f, y_ref),
                "max_abs_fused_vs_ref": float(
                    (y_f.float() - y_ref.float()).abs().max()
                ),
                "pass": r_f <= 1.5 * r_ref + 1e-3,
            }
        )
    return rows


@torch.no_grad()
def e2e(model, tok, new_tokens=32):
    out = []
    for p in PILOT_PROMPTS:
        enc = tok(p, return_tensors="pt", return_token_type_ids=False).to("cuda")
        last = model(**enc).logits[0, -1].float()
        _, seq, _ = greedy(model, tok, p, new_tokens)
        out.append({"last_logits": last, "greedy": seq})
    return out


def compare_e2e(ref, fus):
    rows = []
    for a, b in zip(ref, fus):
        n = 0
        for x, y in zip(a["greedy"], b["greedy"]):
            if x != y:
                break
            n += 1
        rows.append(
            {
                "top1_agree": int(a["last_logits"].argmax())
                == int(b["last_logits"].argmax()),
                "max_abs_logit": float(
                    (a["last_logits"] - b["last_logits"]).abs().max()
                ),
                "greedy_prefix_match": n,
                "greedy_len": len(a["greedy"]),
            }
        )
    return {
        "prompts": rows,
        "top1_agree": sum(r["top1_agree"] for r in rows),
        "greedy_identical": sum(
            r["greedy_prefix_match"] == r["greedy_len"] for r in rows
        ),
    }


@torch.no_grad()
def prefill_profile(model, tok, n, reps=5):
    """Median per-layer and per-MoE-block prefill times for n tokens."""
    ids = timing_ids(tok, n)
    layers = model.model.layers
    ev = {"layer": {}, "moe": {}}
    hs = []

    def mk(kind, li):
        def pre(mod, inp):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            ev[kind].setdefault(li, []).append([e, None])

        def post(mod, inp, out):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            ev[kind][li][-1][1] = e

        return pre, post

    for li, layer in enumerate(layers):
        for kind, mod in (("layer", layer), ("moe", layer.mlp)):
            pre, post = mk(kind, li)
            hs.append(mod.register_forward_pre_hook(pre))
            hs.append(mod.register_forward_hook(post))
    for _ in range(2):
        model(input_ids=ids)
    torch.cuda.synchronize()
    ev["layer"].clear()
    ev["moe"].clear()
    tot = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(input_ids=ids)
        torch.cuda.synchronize()
        tot.append(time.perf_counter() - t0)
    for h in hs:
        h.remove()
    med = lambda kind, li: statistics.median(  # noqa
        a.elapsed_time(b) / 1e3 for a, b in ev[kind][li]
    )
    per = [med("layer", li) for li in range(len(layers))]
    moe = [med("moe", li) for li in range(len(layers))]
    return {
        "tokens": n,
        "total_wall_median_s": statistics.median(tot),
        "per_layer_median_s": per,
        "per_moe_median_s": moe,
        "before_last_layer_s": sum(per[:-1]),
        "moe_share": sum(moe) / sum(per),
    }


def timing(model, tok, lengths):
    res = {"prefill": {n: prefill_profile(model, tok, n) for n in lengths}}
    ids = timing_ids(tok, 128)
    time_decode(model, ids, 4)
    res["decode_s_per_token_128ctx"] = statistics.median(
        time_decode(model, ids, 32) for _ in range(3)
    )
    return res


def screen_agreement(fused_rows, ref_rows, side):
    agree = sum(
        a["pred_per_char"] == b[side]["pred_per_char"]
        for a, b in zip(fused_rows, ref_rows)
    )
    return {
        "agree_per_char": agree,
        "n": len(fused_rows),
        "fused_correct": sum(a["correct_per_char"] for a in fused_rows),
        "ref_correct": sum(b[side]["correct_per_char"] for b in ref_rows),
        "pass": agree >= len(fused_rows) - 4,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("pool")
    ap.add_argument("screen")
    ap.add_argument("ref_per_example")
    ap.add_argument("out_dir")
    ap.add_argument("--configs", default=None)
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1024, 2048])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.screen) as f:
        items = json.load(f)["items"]
    with open(args.pool) as f:
        pool = json.load(f)
    with open(args.ref_per_example) as f:
        ref_rows = [json.loads(line) for line in f]
    assert [r["id"] for r in ref_rows] == [it["id"] for it in items]
    cfgs = ConfigTable(args.configs)
    res = {
        "runtime": runtime_identity(),
        "kernel_source": VLLM_SOURCE,
        "configs": cfgs.source,
        "capture_layers": CAPTURE_LAYERS,
    }
    out_path = os.path.join(args.out_dir, "fused-check.json")
    rows_by = {}

    for side in ("full", "pool"):
        t0 = time.time()
        model, tok = load_model(args.model_dir)
        if side == "pool":
            apply_pool(model, pool)
            gc.collect()
            torch.cuda.empty_cache()
        enc = tok(items[0]["context"], return_tensors="pt", add_special_tokens=False)
        cap = capture(model, enc["input_ids"].cuda())
        r = {"ref_param_bytes": param_bytes(model)}
        r["ref_timing"] = timing(model, tok, args.lengths)
        ref_e2e = e2e(model, tok) if side == "full" else None
        install_fused(model, cfgs)
        gc.collect()
        torch.cuda.empty_cache()
        r["fused_param_bytes"] = param_bytes(model)
        r["fused_mem"] = mem_snapshot()
        r["block_errors"] = block_errors(model, cap)
        del cap
        if ref_e2e is not None:
            r["e2e"] = compare_e2e(ref_e2e, e2e(model, tok))
        r["fused_timing"] = timing(model, tok, args.lengths)
        rows, secs = score_all(model, tok, items, f"fused-{side}")
        rows_by[side] = rows
        r["screen"] = screen_agreement(rows, ref_rows, side)
        r["screen_seconds"] = secs
        r["seconds"] = time.time() - t0
        res[side] = r
        write_json(out_path, res)
        # Drop every reference before the next load; the full model and
        # the pool do not fit together.
        del model, rows, ref_e2e
        gc.collect()
        torch.cuda.empty_cache()

    fw = sum(
        1
        for a, b in zip(rows_by["full"], rows_by["pool"])
        if a["correct_per_char"] and not b["correct_per_char"]
    )
    pw = sum(
        1
        for a, b in zip(rows_by["full"], rows_by["pool"])
        if b["correct_per_char"] and not a["correct_per_char"]
    )
    res["fused_pool_net_losses"] = fw - pw
    gates = {
        "block_errors": all(
            x["pass"] for s in ("full", "pool") for x in res[s]["block_errors"]
        ),
        "e2e_top1": res["full"]["e2e"]["top1_agree"] >= 7,
        "screen_full": res["full"]["screen"]["pass"],
        "screen_pool": res["pool"]["screen"]["pass"],
        "pool_net_losses_le_4": res["fused_pool_net_losses"] <= 4,
    }
    res["gates"] = gates
    res["all_gates_pass"] = all(gates.values())
    write_json(out_path, res)
    with open(os.path.join(args.out_dir, "fused-per-example.jsonl"), "w") as f:
        for a, b in zip(rows_by["full"], rows_by["pool"]):
            f.write(json.dumps({"id": a["id"], "full": a, "pool": b}) + "\n")
    summary = {
        s: {
            "before_last_layer_ms": {
                n: [
                    round(res[s][k]["prefill"][n]["before_last_layer_s"] * 1e3, 2)
                    for k in ("ref_timing", "fused_timing")
                ]
                for n in args.lengths
            },
            "decode_ms": [
                round(res[s][k]["decode_s_per_token_128ctx"] * 1e3, 2)
                for k in ("ref_timing", "fused_timing")
            ],
        }
        for s in ("full", "pool")
    }
    print(json.dumps({"gates": gates, "timing_ref_vs_fused": summary}, indent=1))


if __name__ == "__main__":
    main()
