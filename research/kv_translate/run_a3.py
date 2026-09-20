# SPDX-License-Identifier: GPL-2.0
"""A3: does a nonlinear correction help, and does its objective matter more?

Four arms, all built on the same affine map, all with the same correction
architecture and the same step budget, so the comparisons isolate one thing
each.

  ``affine``            the structured ridge baseline, unweighted
  ``mlp_kv``            plus a small residual network trained on target cache error
  ``mlp_behaviour``     the same network, identically initialised, trained on the
                        target's own continuation divergence instead
  ``weighted_behaviour`` the same behavioural training, started from the
                        attention-weighted affine map rather than the plain one

The pair that matters is the middle two. They differ only in what the
correction is asked to minimise: reproduce the target's tensors, or reproduce
the target's behaviour. The preceding lane in this program established that
cache reconstruction error ranks real damage poorly, which predicts the second
should win even though the first is the obvious objective.

Behavioural training is also where receiver sensitivity enters, and it enters
exactly rather than as an average. Gradients run from the frozen target's
logits back through attention and the re-rotation into the correction, so each
example contributes its own true sensitivity, nonlinearly. Nothing here
estimates a Jacobian; the backward pass is the Jacobian, applied where it is
needed and never materialised.

The correction's last layer starts at zero, so every arm begins exactly at its
affine baseline. An arm that learns nothing degrades to the baseline rather
than to noise, and any gain is attributable to training.

Cost is reported beside quality, because a correction that needs more target
compute than it saves is not a refinement. Parameters, added bytes, training
time and peak memory are recorded for each arm.

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

from research.cartridges_cas.scripts.cas_kv_rope import derot, rerot  # noqa: E402
from research.jacobian_kv import artifacts as art  # noqa: E402
from research.jacobian_kv.evaluate import teacher_forced_kl  # noqa: E402
from research.kv_translate.fit import (  # noqa: E402
    Accumulator,
    SourceLayout,
    fit_block,
    select_layers,
)
from research.kv_translate.pairs import describe, prefill, to_content_keys  # noqa: E402
from research.kv_translate.refine import (  # noqa: E402
    ResidualMapper,
    TrainReport,
    behaviour_loss,
    kv_error_loss,
)
from research.kv_translate.run_a0 import (
    continuation_logits,
    continuation_logits_grad,
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--cont-len", type=int, default=64)
    ap.add_argument("--calib", type=int, default=96)
    ap.add_argument("--train", type=int, default=24, help="prompts for the correction")
    ap.add_argument(
        "--train-pool",
        type=int,
        default=0,
        help="reserve this many correction slots regardless of --train, so the "
        "held-out split is the same chunks at every training size; 0 means "
        "reserve exactly --train",
    )
    ap.add_argument("--eval", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr-grid", default="3e-4,1e-3,3e-3")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--ridge", type=float, default=1e-5)
    ap.add_argument("--exponent", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default="a3")
    args = ap.parse_args()

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    t_start = time.time()
    torch.manual_seed(args.seed)

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

    # The correction slots are reserved from a pool of fixed size, so varying
    # --train changes only how many of them are used and never which chunks
    # land in the held-out split. Without this the evaluation set moves with
    # the training size and divergences are not comparable across a sweep --
    # an earlier sweep had its affine baseline wander between 0.264 and 0.390
    # for that reason alone.
    pool = max(args.train_pool, args.train)
    total = args.calib + pool + args.eval
    chunks = wikitext_chunks(tok, total, args.ctx, args.cont_len, args.seed)
    calib = chunks[: args.calib]
    trainset = chunks[args.calib : args.calib + args.train]
    n_dev = max(2, len(trainset) // 5)
    held = chunks[args.calib + pool :]
    assert not ({c for c, _ in calib} & {c for c, _ in held})
    assert not ({c for c, _ in trainset} & {c for c, _ in held})
    log(
        f"{len(calib)} calibration / {len(trainset) - n_dev} correction "
        f"({n_dev} held back to pick the learning rate) / {len(held)} held-out"
    )

    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    n_targets = tg.n_layers * tg.n_kv_heads
    pos = torch.arange(args.ctx, device=dev)

    def source_features(sp):
        return (
            flat_features(to_content_keys(sp)),
            flat_features([v.float() for v in sp.values]),
        )

    # ---- the two affine starting points ----------------------------------
    acc = {
        k: Accumulator(layout, n_targets, tg.head_dim, device=dev) for k in ("k", "v")
    }
    for _, ids in calib:
        ids = ids[: args.ctx].unsqueeze(0).to(dev)
        sp = prefill(models["source"], ids, sg)
        tp = prefill(models["target"], ids, tg)
        Xk, Xv = source_features(sp)
        acc["k"].add(Xk, target_blocks(to_content_keys(tp)))
        acc["v"].add(Xv, target_blocks([v.float() for v in tp.values]))
        del sp, tp, Xk, Xv
    scale = {k: float(acc[k].xtx.diagonal().mean()) for k in ("k", "v")}
    sel = {
        (kd, li, h): select_layers(
            acc[kd], li * tg.n_kv_heads + h, h, args.ridge * scale[kd], args.k, False
        )
        for kd in ("k", "v")
        for li in range(tg.n_layers)
        for h in range(tg.n_kv_heads)
    }
    plain = {
        kd: Mapper(
            {
                (li, h): fit_block(
                    acc[kd],
                    target=li * tg.n_kv_heads + h,
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
    log("plain affine fitted")

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
        Xk, Xv = source_features(sp)
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
    weighted = {
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
    log("attention-weighted affine fitted")

    # ---- cache what the correction trains against ------------------------
    train_cache = []
    for _, ids in trainset:
        ids = ids.to(dev)
        prompt = ids[: args.ctx].unsqueeze(0)
        cont = ids[args.ctx : args.ctx + args.cont_len].unsqueeze(0)
        sp = prefill(models["source"], prompt, sg)
        tp = prefill(models["target"], prompt, tg)
        Xk, Xv = source_features(sp)
        with torch.no_grad():
            ref = continuation_logits(
                models["target"], tp.keys_post, tp.values, cont, args.ctx
            ).float()
            ref_lp = torch.log_softmax(ref, -1).half().cpu()
        # Held in half precision. These are activations, the correction they
        # train is small, and the storage is what caps how many prompts the
        # correction can see -- which the overfitting measured at twenty
        # prompts says is the binding constraint.
        train_cache.append(
            {
                "Xk": Xk.detach().half().cpu(),
                "Xv": Xv.detach().half().cpu(),
                "Yk": [k.detach().half().cpu() for k in to_content_keys(tp)],
                "Yv": [v.detach().half().cpu() for v in tp.values],
                "cont": cont.cpu(),
                "ref_lp": ref_lp,
            }
        )
        del sp, tp, Xk, Xv, ref, ref_lp
    fit_cache, dev_cache = train_cache[:-n_dev], train_cache[-n_dev:]
    log(f"cached {len(fit_cache)} correction + {len(dev_cache)} rate-selection prompts")

    def dev_loss(rk, rv, objective, items):
        """The objective's own value on prompts the correction never trained on.

        Each arm picks its own learning rate here rather than sharing one. A
        rate that suits the behavioural objective need not suit the cache-error
        one -- an early pilot had the cache-error loss rising during training
        at a rate the behavioural arm was happy with -- and letting a shared
        rate decide would turn this into a comparison of learning rates.
        """
        tot = []
        with torch.no_grad():
            for item in items:
                Xk, Xv = item["Xk"].to(dev).float(), item["Xv"].to(dev).float()
                pk, pv = rk(Xk), rv(Xv)
                if objective == "kv":
                    val = kv_error_loss(
                        pk, [y.to(dev).float() for y in item["Yk"]]
                    ) + kv_error_loss(pv, [y.to(dev).float() for y in item["Yv"]])
                else:
                    keys = [rerot(k.float(), pos, tg.rope_theta).to(dtype) for k in pk]
                    lg = continuation_logits(
                        models["target"],
                        keys,
                        [v.to(dtype) for v in pv],
                        item["cont"].to(dev),
                        args.ctx,
                    )
                    val = behaviour_loss(item["ref_lp"], lg)
                tot.append(float(val))
                del Xk, Xv, pk, pv
        return float(np.mean(tot))

    def train(base, objective, steps, tag, lr):
        rk = ResidualMapper(base["k"], layout, tg, "k", args.hidden).to(dev)
        rv = ResidualMapper(base["v"], layout, tg, "v", args.hidden).to(dev)
        opt = torch.optim.Adam(
            list(rk.blocks.parameters()) + list(rv.blocks.parameters()), lr=lr
        )
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        first, last = None, None
        g = torch.Generator().manual_seed(args.seed)
        for step in range(steps):
            item = fit_cache[int(torch.randint(len(fit_cache), (1,), generator=g))]
            Xk = item["Xk"].to(dev).float()
            Xv = item["Xv"].to(dev).float()
            pk, pv = rk(Xk), rv(Xv)
            if objective == "kv":
                loss = kv_error_loss(
                    pk, [y.to(dev).float() for y in item["Yk"]]
                ) + kv_error_loss(pv, [y.to(dev).float() for y in item["Yv"]])
            else:
                keys = [rerot(k.float(), pos, tg.rope_theta).to(dtype) for k in pk]
                vals = [v.to(dtype) for v in pv]
                lg = continuation_logits_grad(
                    models["target"], keys, vals, item["cont"].to(dev), args.ctx
                )
                loss = behaviour_loss(item["ref_lp"], lg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            v = float(loss.detach())
            first = v if first is None else first
            last = v
            if (step + 1) % 100 == 0:
                log(f"    {tag} step {step + 1}/{steps} loss {v:.5f}")
            del Xk, Xv, pk, pv, loss
        rep = TrainReport(
            objective=objective,
            steps=steps,
            lr=lr,
            residual_params=rk.n_residual_params + rv.n_residual_params,
            residual_bytes=(rk.n_residual_params + rv.n_residual_params) * 2.0,
            train_seconds=time.time() - t0,
            peak_mem_bytes=(
                int(torch.cuda.max_memory_allocated()) if dev == "cuda" else 0
            ),
            first_loss=first,
            last_loss=last,
        )
        rep.dev_loss = dev_loss(rk, rv, objective, dev_cache)
        log(
            f"  {tag} lr={lr:g}: train {rep.first_loss:.5f} -> {rep.last_loss:.5f}, "
            f"dev {rep.dev_loss:.5f}, {rep.train_seconds:.0f}s, "
            f"{rep.residual_params / 1e6:.2f}M params"
        )
        return rk, rv, rep

    def train_best(base, objective, tag):
        """Train at each rate and keep the one with the lowest held-back loss."""
        best = None
        for lr in lrs:
            rk, rv, r = train(base, objective, args.steps, tag, lr)
            if best is None or r.dev_loss < best[2].dev_loss:
                best = (rk, rv, r)
        log(f"  {tag}: chose lr={best[2].lr:g}")
        return best

    lrs = [float(x) for x in args.lr_grid.split(",")]
    reports = {}
    arms = {"affine": (plain["k"], plain["v"], None)}
    log("training correction on target cache error")
    rk2, rv2, reports["mlp_kv"] = train_best(plain, "kv", "mlp_kv")
    arms["mlp_kv"] = (rk2, rv2, None)
    log("training correction on target behaviour")
    rk3, rv3, reports["mlp_behaviour"] = train_best(plain, "behaviour", "mlp_behaviour")
    arms["mlp_behaviour"] = (rk3, rv3, None)
    log("training correction on target behaviour, from the weighted start")
    rk4, rv4, reports["weighted_behaviour"] = train_best(
        weighted, "behaviour", "weighted_behaviour"
    )
    arms["weighted_behaviour"] = (rk4, rv4, None)
    arms["weighted_affine"] = (weighted["k"], weighted["v"], None)

    # ---- held-out ---------------------------------------------------------
    rows = []
    for ci, (cid, ids) in enumerate(held):
        ids = ids.to(dev)
        prompt = ids[: args.ctx].unsqueeze(0)
        cont = ids[args.ctx : args.ctx + args.cont_len].unsqueeze(0)
        sp = prefill(models["source"], prompt, sg)
        tp = prefill(models["target"], prompt, tg)
        with torch.no_grad():
            ref = continuation_logits(
                models["target"], tp.keys_post, tp.values, cont, args.ctx
            ).float()
            Xk, Xv = source_features(sp)
            rows.append(
                {"context": cid, "arm": "target_native", **score(ref, ref, cont)}
            )
            lg = continuation_logits(
                models["target"], None, None, cont, args.ctx
            ).float()
            rows.append({"context": cid, "arm": "no_cache", **score(ref, lg, cont)})
            for name, (mk, mv, _) in arms.items():
                pk = mk(Xk) if isinstance(mk, ResidualMapper) else mk.apply(Xk)
                pv = mv(Xv) if isinstance(mv, ResidualMapper) else mv.apply(Xv)
                keys = [rerot(k.float(), pos, tg.rope_theta).to(dtype) for k in pk]
                vals = [v.to(dtype) for v in pv]
                lg = continuation_logits(
                    models["target"], keys, vals, cont, args.ctx
                ).float()
                rows.append({"context": cid, "arm": name, **score(ref, lg, cont)})
                del pk, pv, keys, vals, lg
        del sp, tp, ref, Xk, Xv
        if dev == "cuda":
            torch.cuda.empty_cache()
        log(f"  held-out {ci + 1}/{len(held)}")

    names = sorted({r["arm"] for r in rows})
    summary = {
        a: {
            m: float(np.mean([r[m] for r in rows if r["arm"] == a]))
            for m in ("kl", "delta_nll", "argmax_flip", "nll")
        }
        for a in names
    }
    log("")
    log(f"{'arm':22s} {'KL':>10s} {'dNLL':>9s} {'flip':>7s}")
    for a in sorted(summary, key=lambda x: summary[x]["kl"]):
        s = summary[a]
        log(f"{a:22s} {s['kl']:10.4f} {s['delta_nll']:9.4f} {s['argmax_flip']:7.3f}")

    aff = summary["affine"]["kl"]
    verdict = {
        "affine_kl": aff,
        "weighted_affine_kl": summary["weighted_affine"]["kl"],
        "mlp_kv_kl": summary["mlp_kv"]["kl"],
        "mlp_behaviour_kl": summary["mlp_behaviour"]["kl"],
        "weighted_behaviour_kl": summary["weighted_behaviour"]["kl"],
        "behaviour_over_kv_objective": (
            summary["mlp_kv"]["kl"] - summary["mlp_behaviour"]["kl"]
        ),
        "best_arm": min(
            summary,
            key=lambda a: summary[a]["kl"] if a not in ("target_native",) else 1e9,
        ),
    }
    log("")
    log(f"A3: {json.dumps(verdict)}")

    out = art.ensure_run_dir(args.out_dir, args.run_id)
    art.write_json(
        os.path.join(out, "a3.json"),
        {
            "source": sg.to_dict(),
            "target": tg.to_dict(),
            "config": {
                "k": args.k,
                "ridge": args.ridge,
                "exponent": args.exponent,
                "hidden": args.hidden,
                "steps": args.steps,
                "lr_grid": lrs,
            },
            "training": {k: v.to_dict() for k, v in reports.items()},
            "arms": summary,
            "verdict": verdict,
            "affine_mapper_bytes": plain["k"].n_bytes + plain["v"].n_bytes,
            "calib_indices": [c for c, _ in calib],
            "train_indices": [c for c, _ in trainset],
            "eval_indices": [c for c, _ in held],
            "train_pool": pool,
            "ctx": args.ctx,
            "seed": args.seed,
            "dtype": args.dtype,
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
