# SPDX-License-Identifier: GPL-2.0
"""Why the linear weighting won, and whether eight source layers was enough.

A2 left two loose ends and both have to be closed before anything is built on
top of it.

The first is an exponent that contradicts its own theory. An attention output
is a sum over cached positions, so if translation errors were uncorrelated
across positions the expected squared output error would go as the *square* of
the attention weights, making the square the principled choice. The linear
exponent beat it on both seeds. Two explanations fit, and they point in
different directions: either the errors are correlated across positions, in
which case squaring models them wrongly, or squaring concentrates the fit onto
too few positions and the ignored tail costs more than the focus gains.

Both are measurable. Concentration is read off the weights directly as a
participation ratio -- the effective number of positions a weighting actually
fits. Correlation is read off the residuals: the mean cosine between
translation errors at different positions. Uncorrelated errors average to zero
cosine; a systematic bias shows up as a positive one.

The second loose end is that both seeds chose the largest layer budget offered,
so the budget was never shown to saturate and eight may simply have been the
ceiling of the search rather than the right answer.

Sweeping the two together also guards against reading one through the other: an
exponent could look better only because it happens to suit a particular layer
budget.

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

from research.cartridges_cas.scripts.cas_kv_rope import derot  # noqa: E402
from research.jacobian_kv import artifacts as art  # noqa: E402
from research.jacobian_kv.evaluate import teacher_forced_kl  # noqa: E402
from research.kv_translate.fit import (  # noqa: E402
    Accumulator,
    SourceLayout,
    fit_block,
    select_layers,
)
from research.kv_translate.pairs import (  # noqa: E402
    describe,
    prefill,
    to_content_keys,
    to_stored_keys,
)
from research.kv_translate.run_a0 import (
    continuation_logits,
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


def participation_ratio(w: torch.Tensor) -> float:
    """Effective number of positions a weighting actually fits.

    ``(sum w)^2 / sum w^2``. Equal to the number of positions when the weight is
    flat, and small when a few positions dominate. This is what "squaring
    concentrates the fit" means quantitatively.
    """
    w = w.float().flatten()
    return float(w.sum().pow(2) / w.pow(2).sum().clamp(min=1e-30))


def residual_cosine(err: torch.Tensor, max_pairs: int = 4096, seed: int = 0) -> float:
    """Mean cosine between translation errors at different cached positions.

    ``err`` is ``[T, D]`` for one block. Errors that are independent across
    positions average to zero; a shared systematic component shows up positive,
    which would mean the squared-weight model of how they accumulate is wrong.
    """
    T = err.shape[0]
    if T < 2:
        return float("nan")
    g = torch.Generator(device="cpu").manual_seed(seed)
    i = torch.randint(0, T, (max_pairs,), generator=g)
    j = torch.randint(0, T, (max_pairs,), generator=g)
    keep = i != j
    i, j = i[keep], j[keep]
    a = torch.nn.functional.normalize(err[i].float(), dim=-1)
    b = torch.nn.functional.normalize(err[j].float(), dim=-1)
    return float((a * b).sum(-1).mean())


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
    ap.add_argument("--k-grid", default="8,16,24")
    ap.add_argument("--exp-grid", default="0,0.5,1,2")
    ap.add_argument("--ridge-grid", default="1e-5,1e-3")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default="a2probe")
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

    total = args.calib + args.dev + args.eval
    chunks = wikitext_chunks(tok, total, args.ctx, args.cont_len, args.seed)
    calib = chunks[: args.calib]
    devset = chunks[args.calib : args.calib + args.dev]
    held = chunks[args.calib + args.dev :]

    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    n_targets = tg.n_layers * tg.n_kv_heads
    ks = [int(x) for x in args.k_grid.split(",")]
    exps = [float(x) for x in args.exp_grid.split(",")]
    ridges = [float(x) for x in args.ridge_grid.split(",")]
    pos = torch.arange(args.ctx, device=dev)

    def blocks_for(ids):
        sp = prefill(models["source"], ids, sg)
        cache = DynamicCache()
        out = models["target"](
            input_ids=ids, past_key_values=cache, use_cache=True, output_attentions=True
        )
        attn = [a.detach() for a in out.attentions]
        del out
        tk = [layer.keys.detach() for layer in cache.layers]
        tv = [layer.values.detach() for layer in cache.layers]
        Xk = flat_features(to_content_keys(sp))
        Xv = flat_features([v.float() for v in sp.values])
        Yk = target_blocks([derot(k.float(), pos, tg.rope_theta) for k in tk])
        Yv = target_blocks([v.float() for v in tv])
        return sp, attn, Xk, Xv, Yk, Yv, tk, tv

    # ---- concentration, straight off the weights -------------------------
    log("measuring how concentrated each exponent's weighting is")
    pr = {e: [] for e in exps}
    for _, ids in calib[:8]:
        ids = ids[: args.ctx].unsqueeze(0).to(dev)
        cache = DynamicCache()
        out = models["target"](
            input_ids=ids, past_key_values=cache, use_cache=True, output_attentions=True
        )
        for li in range(tg.n_layers):
            mass = attention_mass(out.attentions[li], tg.n_kv_heads, args.ctx)
            for h in range(tg.n_kv_heads):
                for e in exps:
                    pr[e].append(participation_ratio(mass[h].pow(e)))
        del out, cache
    concentration = {
        f"exponent_{e:g}": {
            "effective_positions": float(np.mean(pr[e])),
            "fraction_of_context": float(np.mean(pr[e])) / args.ctx,
        }
        for e in exps
    }
    for e in exps:
        c = concentration[f"exponent_{e:g}"]
        log(
            f"  p={e:g}: {c['effective_positions']:7.1f} of {args.ctx} positions "
            f"({100 * c['fraction_of_context']:.1f}%)"
        )

    # ---- the sweep --------------------------------------------------------
    results = []
    best_overall = None
    for k in ks:
        log(f"=== layer budget k={k}")
        acc = {
            kd: Accumulator(layout, n_targets, tg.head_dim, device=dev)
            for kd in ("k", "v")
        }
        for _, ids in calib:
            ids = ids[: args.ctx].unsqueeze(0).to(dev)
            sp, attn, Xk, Xv, Yk, Yv, tk, tv = blocks_for(ids)
            acc["k"].add(Xk, Yk)
            acc["v"].add(Xv, Yv)
            del sp, attn, Xk, Xv, Yk, Yv, tk, tv
        scale = {kd: float(acc[kd].xtx.diagonal().mean()) for kd in ("k", "v")}
        sel = {
            (kd, li, h): select_layers(
                acc[kd], li * tg.n_kv_heads + h, h, ridges[0] * scale[kd], k, False
            )
            for kd in ("k", "v")
            for li in range(tg.n_layers)
            for h in range(tg.n_kv_heads)
        }

        def translate(sp, mk, mv):
            Xk = flat_features(to_content_keys(sp))
            Xv = flat_features([v.float() for v in sp.values])
            keys = to_stored_keys(mk.apply(Xk), tg, args.ctx)
            return [x.to(dtype) for x in keys], [x.to(dtype) for x in mv.apply(Xv)]

        def evaluate(split, mk, mv):
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
                kk, vv = translate(sp, mk, mv)
                lg = continuation_logits(
                    models["target"], kk, vv, cont, args.ctx
                ).float()
                out.append(float(teacher_forced_kl(ref, lg).mean()))
                del sp, tp, ref, lg, kk, vv
            return float(np.mean(out))

        for e in exps:
            if e == 0.0:
                mk_by_r = {
                    r: Mapper(
                        {
                            (li, h): fit_block(
                                acc["k"],
                                target=li * tg.n_kv_heads + h,
                                target_layer=li,
                                head=h,
                                kind="k",
                                layers=sel[("k", li, h)],
                                ridge=r * scale["k"],
                                head_local=False,
                            )
                            for li in range(tg.n_layers)
                            for h in range(tg.n_kv_heads)
                        },
                        layout,
                        tg,
                        "k",
                    )
                    for r in ridges
                }
                mv_by_r = {
                    r: Mapper(
                        {
                            (li, h): fit_block(
                                acc["v"],
                                target=li * tg.n_kv_heads + h,
                                target_layer=li,
                                head=h,
                                kind="v",
                                layers=sel[("v", li, h)],
                                ridge=r * scale["v"],
                                head_local=False,
                            )
                            for li in range(tg.n_layers)
                            for h in range(tg.n_kv_heads)
                        },
                        layout,
                        tg,
                        "v",
                    )
                    for r in ridges
                }
            else:
                wacc = {
                    (kd, li, h): WeightedAccumulator(
                        layout.columns_for(sel[(kd, li, h)], h, False),
                        tg.head_dim,
                        device=dev,
                    )
                    for kd in ("k", "v")
                    for li in range(tg.n_layers)
                    for h in range(tg.n_kv_heads)
                }
                for _, ids in calib:
                    ids = ids[: args.ctx].unsqueeze(0).to(dev)
                    sp, attn, Xk, Xv, Yk, Yv, tk, tv = blocks_for(ids)
                    for li in range(tg.n_layers):
                        mass = attention_mass(attn[li], tg.n_kv_heads, args.ctx)
                        for h in range(tg.n_kv_heads):
                            t = li * tg.n_kv_heads + h
                            w = normalise(mass[h].pow(e).unsqueeze(0))[0]
                            wacc[("k", li, h)].add(Xk, Yk[t], w)
                            wacc[("v", li, h)].add(Xv, Yv[t], w)
                    del sp, attn, Xk, Xv, Yk, Yv, tk, tv
                mk_by_r = {
                    r: Mapper(
                        {
                            (li, h): fit_weighted(
                                wacc[("k", li, h)],
                                target_layer=li,
                                head=h,
                                kind="k",
                                layers=sel[("k", li, h)],
                                ridge=r * scale["k"],
                                head_local=False,
                            )
                            for li in range(tg.n_layers)
                            for h in range(tg.n_kv_heads)
                        },
                        layout,
                        tg,
                        "k",
                    )
                    for r in ridges
                }
                mv_by_r = {
                    r: Mapper(
                        {
                            (li, h): fit_weighted(
                                wacc[("v", li, h)],
                                target_layer=li,
                                head=h,
                                kind="v",
                                layers=sel[("v", li, h)],
                                ridge=r * scale["v"],
                                head_local=False,
                            )
                            for li in range(tg.n_layers)
                            for h in range(tg.n_kv_heads)
                        },
                        layout,
                        tg,
                        "v",
                    )
                    for r in ridges
                }
                del wacc

            best_r, best_kl = None, None
            for r in ridges:
                kl = evaluate(devset, mk_by_r[r], mv_by_r[r])
                if best_kl is None or kl < best_kl:
                    best_r, best_kl = r, kl
            held_kl = evaluate(held, mk_by_r[best_r], mv_by_r[best_r])
            bytes_ = mk_by_r[best_r].n_bytes + mv_by_r[best_r].n_bytes
            results.append(
                {
                    "k": k,
                    "exponent": e,
                    "ridge": best_r,
                    "dev_kl": best_kl,
                    "held_kl": held_kl,
                    "mapper_bytes": bytes_,
                }
            )
            log(
                f"  k={k} p={e:g}: dev {best_kl:.4f} held-out {held_kl:.4f} "
                f"({bytes_ / 2**20:.0f} MiB)"
            )
            if best_overall is None or held_kl < best_overall[0]:
                best_overall = (held_kl, k, e, best_r, mk_by_r[best_r], mv_by_r[best_r])
            del mk_by_r, mv_by_r
            if dev == "cuda":
                torch.cuda.empty_cache()
        del acc
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ---- are translation errors correlated across positions? -------------
    log("measuring residual correlation across cached positions")
    _, bk, be, br, mk, mv = best_overall
    cos_k, cos_v = [], []
    for _, ids in held[:6]:
        ids = ids[: args.ctx].unsqueeze(0).to(dev)
        sp = prefill(models["source"], ids, sg)
        tp = prefill(models["target"], ids, tg)
        Xk = flat_features(to_content_keys(sp))
        Xv = flat_features([v.float() for v in sp.values])
        pk, pv = mk.apply(Xk), mv.apply(Xv)
        ck = to_content_keys(tp)
        for li in range(0, tg.n_layers, 4):
            for h in range(tg.n_kv_heads):
                cos_k.append(residual_cosine((pk[li][0, h] - ck[li][0, h]).cpu()))
                cos_v.append(
                    residual_cosine((pv[li][0, h] - tp.values[li][0, h].float()).cpu())
                )
        del sp, tp, Xk, Xv, pk, pv, ck
    correlation = {
        "key_residual_mean_cosine": float(np.mean(cos_k)),
        "value_residual_mean_cosine": float(np.mean(cos_v)),
    }
    log(f"  key residual cosine {correlation['key_residual_mean_cosine']:+.4f}")
    log(f"  value residual cosine {correlation['value_residual_mean_cosine']:+.4f}")

    # ---- report -----------------------------------------------------------
    log("")
    log(f"{'k':>4s} {'p':>5s} {'dev KL':>9s} {'held KL':>9s} {'MiB':>6s}")
    for r in sorted(results, key=lambda x: x["held_kl"]):
        log(
            f"{r['k']:4d} {r['exponent']:5g} {r['dev_kl']:9.4f} {r['held_kl']:9.4f} "
            f"{r['mapper_bytes'] / 2**20:6.0f}"
        )
    log("")
    log(f"best: k={bk} exponent={be:g} ridge={br:g}")

    out = art.ensure_run_dir(args.out_dir, args.run_id)
    art.write_json(
        os.path.join(out, "a2probe.json"),
        {
            "source": sg.to_dict(),
            "target": tg.to_dict(),
            "results": results,
            "concentration": concentration,
            "residual_correlation": correlation,
            "best": {"k": bk, "exponent": be, "ridge": br},
            "ctx": args.ctx,
            "seed": args.seed,
            "dtype": args.dtype,
            "wall_s": time.time() - t_start,
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "versions": art.versions(),
            "code_commit": art.git_state()[0],
        },
    )
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
