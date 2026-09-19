# SPDX-License-Identifier: GPL-2.0
"""A2: does weighting calibration by receiver attention improve the map?

A1 established that a plain structured affine map carries most of a cross-model
cache. A2 asks whether telling the fit which cached positions the target
actually reads makes it carry more, at no extra online cost.

The comparison is built to be attributable. Both arms are fitted in the same
run, on the same calibration prompts, with the same source-layer selection, the
same regularisation grid, the same mapper byte count and the same code applying
them. The only difference is a per-token weight in the least-squares problem.
Running the baseline here rather than comparing against a separate A1 run also
makes the two arms paired on the identical held-out prompts, so the difference
between them is not carrying any prompt-sampling noise.

The weighting is the attention mass each cached position receives in the target
model's own prefill, which costs no backward pass and exists when a cache is
written. Both exponents are scored: the square is principled for values, since
an attention output is a sum over positions and uncorrelated errors accumulate
as the square of the weights, while the right exponent for keys is less obvious.
Whichever wins is chosen on a development split, never on the held-out prompts,
and the choice is recorded.

The bar, set before running: a ten percent relative reduction in excess
divergence over the baseline, on both seeds, with no increase in online cost.
Raw reconstruction getting slightly worse while behaviour improves is an
acceptable outcome and the reverse is not, so both are reported.

Env: HIP_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
Run:
    python -m research.kv_translate.run_a2 --out-dir <dir>
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

from research.jacobian_kv import artifacts as art  # noqa: E402
from research.jacobian_kv.evaluate import teacher_forced_kl  # noqa: E402
from research.kv_translate.fit import (  # noqa: E402
    Accumulator,
    SourceLayout,
    fit_block,
    select_layers,
)
from research.cartridges_cas.scripts.cas_kv_rope import derot  # noqa: E402
from research.kv_translate.pairs import (  # noqa: E402
    describe,
    prefill,
    to_content_keys,
    to_stored_keys,
)
from research.kv_translate.run_a0 import (  # noqa: E402
    continuation_logits,
    score,
    wikitext_chunks,
)
from research.kv_translate.run_a1 import (  # noqa: E402
    Mapper,
    flat_features,
    target_blocks,
)
from research.kv_translate.weighted import (  # noqa: E402
    WeightedAccumulator,
    attention_mass,
    fit_weighted,
    normalise,
)


@torch.no_grad()
def prefill_with_attention(model, ids, geom):
    """Prefill keeping the prompt's self-attention, for the weighting."""
    from transformers import DynamicCache

    cache = DynamicCache()
    out = model(
        input_ids=ids, past_key_values=cache, use_cache=True, output_attentions=True
    )
    attn = [a.detach() for a in out.attentions]
    del out
    return cache, attn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--cont-len", type=int, default=64)
    ap.add_argument("--calib", type=int, default=96)
    ap.add_argument("--dev", type=int, default=8)
    ap.add_argument("--eval", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--k", type=int, default=8, help="source layers, fixed from A1")
    ap.add_argument("--head-local", action="store_true")
    ap.add_argument("--ridge-grid", default="1e-5,1e-3,1e-1,1e0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default="a2")
    args = ap.parse_args()

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    total = args.calib + args.dev + args.eval
    chunks = wikitext_chunks(tok, total, args.ctx, args.cont_len, args.seed)
    calib = chunks[: args.calib]
    devset = chunks[args.calib : args.calib + args.dev]
    held = chunks[args.calib + args.dev :]
    log(f"{len(calib)} calibration / {len(devset)} dev / {len(held)} held-out")

    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    n_targets = tg.n_layers * tg.n_kv_heads
    ridges = [float(x) for x in args.ridge_grid.split(",")]
    exponents = [1.0, 2.0]

    # ---- pass one: unweighted statistics, which also fix the support -----
    acc = {
        k: Accumulator(layout, n_targets, tg.head_dim, device=dev) for k in ("k", "v")
    }
    t0 = time.time()
    for i, (_, ids) in enumerate(calib):
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
        if (i + 1) % 32 == 0:
            log(f"  unweighted {i + 1}/{len(calib)}")
    log(f"unweighted statistics in {time.time() - t0:.1f}s")

    scale = {k: float(acc[k].xtx.diagonal().mean()) for k in ("k", "v")}
    sel = {}
    for kind in ("k", "v"):
        for li in range(tg.n_layers):
            for h in range(tg.n_kv_heads):
                t = li * tg.n_kv_heads + h
                sel[(kind, li, h)] = select_layers(
                    acc[kind], t, h, ridges[0] * scale[kind], args.k, args.head_local
                )
    log(
        f"support fixed: {args.k} source layers per block, head_local={args.head_local}"
    )

    # ---- pass two: weighted statistics on exactly that support -----------
    wacc = {
        (kind, e): {
            (li, h): WeightedAccumulator(
                layout.columns_for(sel[(kind, li, h)], h, args.head_local),
                tg.head_dim,
                device=dev,
            )
            for li in range(tg.n_layers)
            for h in range(tg.n_kv_heads)
        }
        for kind in ("k", "v")
        for e in exponents
    }
    t0 = time.time()
    for i, (_, ids) in enumerate(calib):
        ids = ids[: args.ctx].unsqueeze(0).to(dev)
        sp = prefill(models["source"], ids, sg)
        cache, attn = prefill_with_attention(models["target"], ids, tg)
        # target blocks in the frames their maps are fitted in: keys de-rotated
        # into the content frame, values as stored
        tkeys = [layer.keys.detach() for layer in cache.layers]
        tvals = [layer.values.detach() for layer in cache.layers]
        pos = torch.arange(args.ctx, device=dev)
        Yk = target_blocks([derot(k.float(), pos, tg.rope_theta) for k in tkeys])
        Yv = target_blocks([v.float() for v in tvals])
        Xk = flat_features(to_content_keys(sp))
        Xv = flat_features([v.float() for v in sp.values])
        for li in range(tg.n_layers):
            mass = attention_mass(attn[li], tg.n_kv_heads, args.ctx)
            for h in range(tg.n_kv_heads):
                t = li * tg.n_kv_heads + h
                for e in exponents:
                    w = normalise(mass[h].pow(e).unsqueeze(0))[0]
                    wacc[("k", e)][(li, h)].add(Xk, Yk[t], w)
                    wacc[("v", e)][(li, h)].add(Xv, Yv[t], w)
        del sp, cache, attn, Xk, Xv, Yk, Yv, tkeys, tvals
        if (i + 1) % 32 == 0:
            log(f"  weighted {i + 1}/{len(calib)}")
    log(f"weighted statistics in {time.time() - t0:.1f}s")

    # ---- build both arms --------------------------------------------------
    def build_unweighted(kind, ridge):
        maps = {}
        for li in range(tg.n_layers):
            for h in range(tg.n_kv_heads):
                t = li * tg.n_kv_heads + h
                maps[(li, h)] = fit_block(
                    acc[kind],
                    target=t,
                    target_layer=li,
                    head=h,
                    kind=kind,
                    layers=sel[(kind, li, h)],
                    ridge=ridge,
                    head_local=args.head_local,
                )
        return Mapper(maps, layout, tg, kind)

    def build_weighted(kind, ridge, e):
        maps = {}
        for li in range(tg.n_layers):
            for h in range(tg.n_kv_heads):
                maps[(li, h)] = fit_weighted(
                    wacc[(kind, e)][(li, h)],
                    target_layer=li,
                    head=h,
                    kind=kind,
                    layers=sel[(kind, li, h)],
                    ridge=ridge,
                    head_local=args.head_local,
                )
        return Mapper(maps, layout, tg, kind)

    def translate(sp, mk, mv, n):
        Xk = flat_features(to_content_keys(sp))
        Xv = flat_features([v.float() for v in sp.values])
        keys = to_stored_keys(mk.apply(Xk), tg, n)
        return [k.to(dtype) for k in keys], [v.to(dtype) for v in mv.apply(Xv)]

    def eval_split(split, mk, mv):
        out = []
        for _, ids in split:
            ids = ids.to(dev)
            prompt = ids[: args.ctx].unsqueeze(0)
            cont = ids[args.ctx : args.ctx + args.cont_len].unsqueeze(0)
            sp = prefill(models["source"], prompt, sg)
            tp = prefill(models["target"], prompt, tg)
            ref = continuation_logits(
                models["target"], tp.keys_post, tp.values, cont, args.ctx
            ).float()
            k, v = translate(sp, mk, mv, args.ctx)
            lg = continuation_logits(models["target"], k, v, cont, args.ctx).float()
            out.append(float(teacher_forced_kl(ref, lg).mean()))
            del sp, tp, ref, lg, k, v
        return float(np.mean(out))

    trials = []
    best = {"unweighted": None, "weighted": None}
    for r in ridges:
        mk = build_unweighted("k", r * scale["k"])
        mv = build_unweighted("v", r * scale["v"])
        kl = eval_split(devset, mk, mv)
        trials.append({"arm": "unweighted", "ridge": r, "dev_kl": kl})
        log(f"  dev unweighted ridge={r:g}: KL {kl:.4f}")
        if best["unweighted"] is None or kl < best["unweighted"][0]:
            best["unweighted"] = (kl, r, mk, mv)
    for e in exponents:
        for r in ridges:
            mk = build_weighted("k", r * scale["k"], e)
            mv = build_weighted("v", r * scale["v"], e)
            kl = eval_split(devset, mk, mv)
            trials.append({"arm": "weighted", "exponent": e, "ridge": r, "dev_kl": kl})
            log(f"  dev weighted p={e:g} ridge={r:g}: KL {kl:.4f}")
            if best["weighted"] is None or kl < best["weighted"][0]:
                best["weighted"] = (kl, r, mk, mv, e)

    log(f"unweighted picked ridge={best['unweighted'][1]:g}")
    log(
        f"weighted picked ridge={best['weighted'][1]:g} exponent={best['weighted'][4]:g}"
    )

    # ---- paired held-out comparison --------------------------------------
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
        rows.append({"context": cid, "arm": "target_native", **score(ref, ref, cont)})
        lg = continuation_logits(models["target"], None, None, cont, args.ctx).float()
        rows.append({"context": cid, "arm": "no_cache", **score(ref, lg, cont)})
        for name, sel_best in (
            ("unweighted", best["unweighted"]),
            ("weighted", best["weighted"]),
        ):
            k, v = translate(sp, sel_best[2], sel_best[3], args.ctx)
            lg = continuation_logits(models["target"], k, v, cont, args.ctx).float()
            rows.append({"context": cid, "arm": name, **score(ref, lg, cont)})
            del k, v, lg
        del sp, tp, ref
        if dev == "cuda":
            torch.cuda.empty_cache()
        log(f"  held-out {ci + 1}/{len(held)}")

    arms = sorted({r["arm"] for r in rows})
    summary = {
        a: {
            m: float(np.mean([r[m] for r in rows if r["arm"] == a]))
            for m in ("kl", "delta_nll", "argmax_flip", "nll")
        }
        for a in arms
    }
    log("")
    log(f"{'arm':16s} {'KL':>10s} {'dNLL':>9s} {'flip':>7s}")
    for a in sorted(summary, key=lambda x: summary[x]["kl"]):
        s = summary[a]
        log(f"{a:16s} {s['kl']:10.4f} {s['delta_nll']:9.4f} {s['argmax_flip']:7.3f}")

    base = summary["unweighted"]["kl"]
    weig = summary["weighted"]["kl"]
    rel = (base - weig) / max(base, 1e-12)
    ub = np.array([r["kl"] for r in rows if r["arm"] == "unweighted"])
    wb = np.array([r["kl"] for r in rows if r["arm"] == "weighted"])
    verdict = {
        "unweighted_kl": base,
        "weighted_kl": weig,
        "relative_reduction": rel,
        "wins_on_n_of_m_prompts": int((wb < ub).sum()),
        "n_prompts": int(ub.size),
        "no_cache_kl": summary["no_cache"]["kl"],
        "mapper_bytes_equal": True,
        "A2_ADVANCE": bool(rel >= 0.10),
    }
    log("")
    log(f"A2 {'ADVANCE' if verdict['A2_ADVANCE'] else 'STOP'}: {json.dumps(verdict)}")

    out = art.ensure_run_dir(args.out_dir, args.run_id)
    art.write_json(
        os.path.join(out, "a2.json"),
        {
            "source": sg.to_dict(),
            "target": tg.to_dict(),
            "fixed_support": {"k": args.k, "head_local": args.head_local},
            "trials": trials,
            "arms": summary,
            "verdict": verdict,
            "selected": {
                "unweighted_ridge": best["unweighted"][1],
                "weighted_ridge": best["weighted"][1],
                "weighted_exponent": best["weighted"][4],
            },
            "ctx": args.ctx,
            "seed": args.seed,
            "dtype": args.dtype,
            "calib_indices": [c for c, _ in calib],
            "eval_indices": [c for c, _ in held],
            "wall_s": time.time() - t_start,
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "versions": art.versions(),
            "code_commit": art.git_state()[0],
            "code_dirty": art.git_state()[1],
        },
    )
    with art.JsonlWriter(os.path.join(out, "rows.jsonl")) as w:
        for r in rows:
            w.write(r)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
