# SPDX-License-Identifier: GPL-2.0
"""A4: is the residual error concentrated in a few layers worth recomputing?

If most layers translate well and a small subset carries the damage, the right
method is not a better translator but a hybrid: translate the easy layers and
let the target produce the hard ones itself. A4 measures whether that subset
exists and what recomputing it actually costs.

The cost is the part that is easy to get wrong, so it is stated up front. A
target layer's native key/value block cannot be obtained in isolation. It is a
function of the residual stream at that depth, which is produced by every layer
before it, so "repair layer twenty" means running the target's prefill through
layer twenty. Repair is therefore a prefix, not a subset: natively computing
the first N layers and translating the rest reintroduces N/L of the prefill,
while natively computing a late layer costs nearly all of it. Any scheme that
claims to repair scattered deep layers cheaply has miscounted.

Two measurements follow from that.

A per-layer damage profile, which is diagnostic and ignores cost: give one
layer its native block and translate every other, and see how much of the gap
that single layer closes. This says where the error lives even where fixing it
would be expensive, and it is the honest place to look for concentration.

A prefix sweep, which is the deployable shape: run the target natively for the
first N layers, translate the remainder, and report divergence against the
fraction of prefill reintroduced and the measured wall time. A hybrid that
approaches a full prefill is not a transfer win however good its divergence
looks, so the crossover is reported rather than a quality number alone.

Env: HIP_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.cartridges_cas.scripts.cas_kv_rope import rerot  # noqa: E402
from research.jacobian_kv import artifacts as art  # noqa: E402
from research.kv_translate.fit import (  # noqa: E402
    Accumulator,
    SourceLayout,
    fit_block,
    select_layers,
)
from research.kv_translate.pairs import describe, prefill, to_content_keys  # noqa: E402
from research.kv_translate.run_a0 import (
    continuation_logits,
    score,
    wikitext_chunks,
)  # noqa: E402
from research.kv_translate.run_a1 import (
    Mapper,
    flat_features,
    target_blocks,
)  # noqa: E402
from research.kv_translate.weighted import (  # noqa: E402
    WeightedAccumulator,
    attention_mass,
    fit_weighted,
    normalise,
)


@torch.no_grad()
def prefix_prefill_cost(model, ids, n_layers_native, reps=3):
    """Wall time for a prefill truncated after ``n_layers_native`` layers.

    Two things this is careful about.

    It times the base transformer, not the causal-language-model wrapper. A
    prefill whose purpose is to fill a cache never needs a vocabulary
    projection over every prompt position, and that projection does not shrink
    when layers are removed, so including it would add a large constant to
    every depth and flatten the cost curve into uselessness.

    And it measures rather than assuming cost is proportional to depth:
    attention is quadratic in context while the rest of a layer is linear, and
    the embedding and final norm sit outside the loop.
    """
    if n_layers_native <= 0:
        return 0.0
    base = model.model
    original = base.layers
    base.layers = torch.nn.ModuleList(list(original[:n_layers_native]))
    try:
        from transformers import DynamicCache

        for _ in range(2):
            base(input_ids=ids, past_key_values=DynamicCache(), use_cache=True)
        if ids.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            base(input_ids=ids, past_key_values=DynamicCache(), use_cache=True)
        if ids.device.type == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t0) / reps
    finally:
        base.layers = original


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--cont-len", type=int, default=64)
    ap.add_argument("--calib", type=int, default=96)
    ap.add_argument("--eval", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--ridge", type=float, default=1e-5)
    ap.add_argument("--exponent", type=float, default=1.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default="a4")
    args = ap.parse_args()

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    t_start = time.time()

    tok = AutoTokenizer.from_pretrained(args.target)
    models, geom = {}, {}
    for role, mid in (("source", args.source), ("target", args.target)):
        m = AutoModelForCausalLM.from_pretrained(
            mid, dtype=dtype, attn_implementation="eager"
        ).to(dev)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models[role], geom[role] = m, describe(m, mid)
    sg, tg = geom["source"], geom["target"]

    chunks = wikitext_chunks(
        tok, args.calib + args.eval, args.ctx, args.cont_len, args.seed
    )
    calib, held = chunks[: args.calib], chunks[args.calib :]
    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    n_targets = tg.n_layers * tg.n_kv_heads
    pos = torch.arange(args.ctx, device=dev)

    # ---- the attention-weighted affine map A2 selected -------------------
    acc = {
        k: Accumulator(layout, n_targets, tg.head_dim, device=dev) for k in ("k", "v")
    }
    for _, ids in calib:
        ids = ids[: args.ctx].unsqueeze(0).to(dev)
        sp = prefill(models["source"], ids, sg)
        tp = prefill(models["target"], ids, tg)
        acc["k"].add(
            flat_features(to_content_keys(sp)), target_blocks(to_content_keys(tp))
        )
        acc["v"].add(
            flat_features([v.float() for v in sp.values]),
            target_blocks([v.float() for v in tp.values]),
        )
        del sp, tp
    scale = {k: float(acc[k].xtx.diagonal().mean()) for k in ("k", "v")}
    sel = {
        (kd, li, h): select_layers(
            acc[kd], li * tg.n_kv_heads + h, h, args.ridge * scale[kd], args.k, False
        )
        for kd in ("k", "v")
        for li in range(tg.n_layers)
        for h in range(tg.n_kv_heads)
    }
    wacc = {
        (kd, li, h): WeightedAccumulator(
            layout.columns_for(sel[(kd, li, h)], h, False), tg.head_dim, device=dev
        )
        for kd in ("k", "v")
        for li in range(tg.n_layers)
        for h in range(tg.n_kv_heads)
    }
    for _, ids in calib:
        ids = ids[: args.ctx].unsqueeze(0).to(dev)
        sp = prefill(models["source"], ids, sg)
        cache = DynamicCache()
        out = models["target"](
            input_ids=ids, past_key_values=cache, use_cache=True, output_attentions=True
        )
        attn = [a.detach() for a in out.attentions]
        del out
        Xk = flat_features(to_content_keys(sp))
        Xv = flat_features([v.float() for v in sp.values])
        from research.cartridges_cas.scripts.cas_kv_rope import derot

        Yk = target_blocks(
            [derot(l.keys.detach().float(), pos, tg.rope_theta) for l in cache.layers]
        )
        Yv = target_blocks([l.values.detach().float() for l in cache.layers])
        for li in range(tg.n_layers):
            mass = attention_mass(attn[li], tg.n_kv_heads, args.ctx)
            for h in range(tg.n_kv_heads):
                t = li * tg.n_kv_heads + h
                w = normalise(mass[h].pow(args.exponent).unsqueeze(0))[0]
                wacc[("k", li, h)].add(Xk, Yk[t], w)
                wacc[("v", li, h)].add(Xv, Yv[t], w)
        del sp, cache, attn, Xk, Xv, Yk, Yv
    mapper = {
        kd: Mapper(
            {
                (li, h): fit_weighted(
                    wacc[(kd, li, h)],
                    target_layer=li,
                    head=h,
                    kind=kd,
                    layers=sel[(kd, li, h)],
                    ridge=args.ridge * scale[kd],
                    head_local=False,
                )
                for li in range(tg.n_layers)
                for h in range(tg.n_kv_heads)
            },
            layout,
            tg,
            kd,
        )
        for kd in ("k", "v")
    }
    del wacc, acc
    log("attention-weighted affine map fitted")

    # ---- cost of a truncated prefill, measured ---------------------------
    probe_ids = held[0][1][: args.ctx].unsqueeze(0).to(dev)
    depths = sorted({0, 2, 4, 8, 14, 20, tg.n_layers})
    prefill_cost = {
        n: prefix_prefill_cost(models["target"], probe_ids, n) for n in depths
    }
    full = prefill_cost[tg.n_layers]
    log("prefill cost by native depth:")
    for n in depths:
        log(
            f"  first {n:2d} of {tg.n_layers} layers: {1000 * prefill_cost[n]:6.1f} ms "
            f"({prefill_cost[n] / max(full, 1e-9):.3f} of full)"
        )

    # ---- per-layer damage, and the prefix sweep --------------------------
    rows = []
    for ci, (cid, ids) in enumerate(held):
        ids = ids.to(dev)
        prompt = ids[: args.ctx].unsqueeze(0)
        cont = ids[args.ctx : args.ctx + args.cont_len].unsqueeze(0)
        sp = prefill(models["source"], prompt, sg)
        tp = prefill(models["target"], prompt, tg)
        ref = continuation_logits(
            models["target"], tp.keys_post, tp.values, cont, args.ctx
        ).float()
        Xk = flat_features(to_content_keys(sp))
        Xv = flat_features([v.float() for v in sp.values])
        tk = [
            rerot(k.float(), pos, tg.rope_theta).to(dtype)
            for k in mapper["k"].apply(Xk)
        ]
        tv = [v.to(dtype) for v in mapper["v"].apply(Xv)]

        def run(keys, values, arm, extra=None):
            lg = continuation_logits(
                models["target"], keys, values, cont, args.ctx
            ).float()
            row = {"context": cid, "arm": arm, **score(ref, lg, cont)}
            if extra:
                row.update(extra)
            rows.append(row)
            del lg

        run(tp.keys_post, tp.values, "target_native")
        run(None, None, "no_cache")
        run(tk, tv, "translated_all")

        # one layer native, everything else translated: where does damage live
        for li in range(tg.n_layers):
            k2 = list(tk)
            v2 = list(tv)
            k2[li], v2[li] = tp.keys_post[li], tp.values[li]
            run(k2, v2, "native_one_layer", {"layer": li})

        # the deployable shape: native prefix, translated remainder
        for n in depths:
            k2 = [tp.keys_post[i] if i < n else tk[i] for i in range(tg.n_layers)]
            v2 = [tp.values[i] if i < n else tv[i] for i in range(tg.n_layers)]
            run(k2, v2, "native_prefix", {"n_native": n})

        del sp, tp, ref, Xk, Xv, tk, tv
        if dev == "cuda":
            torch.cuda.empty_cache()
        log(f"  held-out {ci + 1}/{len(held)}")

    # ---- report -----------------------------------------------------------
    def mean(arm, **f):
        sel_rows = [
            r
            for r in rows
            if r["arm"] == arm and all(r.get(k) == v for k, v in f.items())
        ]
        return float(np.mean([r["kl"] for r in sel_rows]))

    native = mean("target_native")
    floor = mean("no_cache")
    allt = mean("translated_all")
    log("")
    log(f"no cache {floor:.4f} | translated {allt:.4f} | native {native:.4f}")

    per_layer = [
        {"layer": li, "kl": mean("native_one_layer", layer=li)}
        for li in range(tg.n_layers)
    ]
    for d in per_layer:
        d["gap_closed"] = (allt - d["kl"]) / max(allt - native, 1e-12)
    top = sorted(per_layer, key=lambda d: d["kl"])[:6]
    log("")
    log("single native layer, rest translated -- where the damage lives:")
    for d in top:
        log(
            f"  layer {d['layer']:2d}: KL {d['kl']:.4f}  closes "
            f"{100 * d['gap_closed']:5.1f}% of the remaining gap"
        )
    spread = float(np.std([d["kl"] for d in per_layer]))
    log(f"  spread across layers (std of KL): {spread:.5f}")

    prefix = []
    for n in depths:
        kl = mean("native_prefix", n_native=n)
        prefix.append(
            {
                "n_native": n,
                "kl": kl,
                "prefill_fraction": prefill_cost[n] / max(full, 1e-9),
                "prefill_ms": 1000 * prefill_cost[n],
                "gap_closed": (allt - kl) / max(allt - native, 1e-12),
            }
        )
    log("")
    log("native prefix, translated remainder -- the deployable shape:")
    log(f"  {'N':>3s} {'KL':>8s} {'gap closed':>11s} {'prefill cost':>13s}")
    for p in prefix:
        log(
            f"  {p['n_native']:3d} {p['kl']:8.4f} {100 * p['gap_closed']:10.1f}% "
            f"{100 * p['prefill_fraction']:11.1f}%"
        )

    # the crossover: is any hybrid better per unit of reintroduced compute?
    useful = [p for p in prefix if 0 < p["n_native"] < tg.n_layers]
    best = (
        max(useful, key=lambda p: p["gap_closed"] / max(p["prefill_fraction"], 1e-9))
        if useful
        else None
    )
    verdict = {
        "translated_all_kl": allt,
        "no_cache_kl": floor,
        "damage_is_concentrated": bool(spread > 0.1 * max(allt - native, 1e-12)),
        "layer_kl_spread": spread,
        "best_single_layer": top[0]["layer"],
        "best_single_layer_gap_closed": top[0]["gap_closed"],
        "best_prefix": best,
    }
    log("")
    log(f"A4: {json.dumps(verdict)}")

    out = art.ensure_run_dir(args.out_dir, args.run_id)
    art.write_json(
        os.path.join(out, "a4.json"),
        {
            "source": sg.to_dict(),
            "target": tg.to_dict(),
            "config": {"k": args.k, "ridge": args.ridge, "exponent": args.exponent},
            "prefill_cost_s": prefill_cost,
            "per_layer": per_layer,
            "prefix": prefix,
            "verdict": verdict,
            "ctx": args.ctx,
            "seed": args.seed,
            "dtype": args.dtype,
            "calib_indices": [c for c, _ in calib],
            "eval_indices": [c for c, _ in held],
            "wall_s": time.time() - t_start,
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "versions": art.versions(),
            "code_commit": art.git_state()[0],
        },
    )
    with art.JsonlWriter(os.path.join(out, "rows.jsonl")) as w:
        for r in rows:
            w.write(r)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
