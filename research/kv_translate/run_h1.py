# SPDX-License-Identifier: GPL-2.0
"""H1: is a corrected translator worth deploying on the model that matters?

The uncompressed map is already fast enough, so the question is no longer how
cheaply it can be stored but whether it is useful. This runs the comparison on
the 1.5B-to-7B pair at the empirical knee of ninety-six correction prompts,
and it is arranged so that each comparison isolates one thing.

Two frozen affine starting points are carried forward from the earlier lane:
the best-quality unweighted map at sixteen source layers per block, and the
byte-efficient attention-weighted map at eight. Because those differ in two
ways at once, an unweighted map at eight layers is included as well, so
weighting can be read off at equal width instead of inferred across unequal
ones.

On each of the two promoted starting points, two corrections are trained. One
is a low-rank linear update, which matters because a composition of affine
maps is an affine map: a trained linear correction folds back into the map it
corrects and then costs exactly nothing online. The other is the nonlinear
residual network from the earlier lane, which cannot be folded and must carry
its weights, its operator and its latency into every later accounting. The two
have identical parameter counts, identical shapes and identical zeroed
initialisation, and differ only by one activation function, so the comparison
measures the nonlinearity rather than capacity.

Everything else is held fixed across the trained arms: the same step budget,
the same learning-rate grid with each arm free to pick its own rate on the
same held-back split, and the same sequence of training examples drawn from
the same seed. The uncorrected maps are scored as controls, so an arm that
learns nothing is visibly no worse than where it started rather than lost in
noise.

The objective is the target's own continuation behaviour. The earlier lane
established that cache reconstruction error ranks real damage poorly, and its
negative control is archived; it is not repeated here.

Cost is reported beside quality, because a correction that costs more target
compute than it saves is not a refinement: target GPU-seconds, peak memory,
serialized and resident bytes, and the online latency of applying the arm.

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
from research.kv_translate.fit import (  # noqa: E402
    Accumulator,
    AffineMap,
    SourceLayout,
    fit_block,
    select_layers,
)
from research.kv_translate.pairs import (  # noqa: E402
    check_pair,
    describe,
    prefill,
    to_content_keys,
    verify_rope,
)
from research.kv_translate import freeze  # noqa: E402
from research.kv_translate.refine import (  # noqa: E402
    ResidualMapper,
    TrainReport,
    behaviour_loss,
)
from research.kv_translate.run_a0 import (  # noqa: E402
    continuation_logits,
    continuation_logits_grad,
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

GIB = float(2**30)


class MassCollector:
    """Per-layer attention mass, reduced inside the layer that produced it.

    The obvious way to get this is ``output_attentions=True`` and then read
    ``out.attentions``, which is what the small pair did. On a 7B target that
    list is twenty-eight layers of ``[28, T, T]`` held simultaneously -- about
    eleven gigabytes at a context of 512 -- for a statistic that is two
    numbers per head and position. Reducing in a hook and dropping the weights
    on the way out keeps one layer resident instead of all of them.

    The hook returns ``None`` in place of the weights it consumed, so the
    decoder assembles its usual output tuple and nothing downstream holds the
    large tensor.
    """

    def __init__(self, model, n_kv_heads: int, n_positions: int):
        self.n_kv_heads = n_kv_heads
        self.n_positions = n_positions
        self.mass: dict = {}
        self._handles = []
        for li, layer in enumerate(model.model.layers):
            self._handles.append(
                layer.self_attn.register_forward_hook(self._make(li), with_kwargs=False)
            )

    def _make(self, li):
        def hook(_mod, _inp, out):
            if not isinstance(out, tuple) or len(out) < 2 or out[1] is None:
                raise RuntimeError(
                    f"layer {li} returned no attention weights; the target must be "
                    "loaded with attn_implementation='eager' and called with "
                    "output_attentions=True for the weighted calibration"
                )
            self.mass[li] = attention_mass(
                out[1].detach(), self.n_kv_heads, self.n_positions
            ).cpu()
            return (out[0], None) + tuple(out[2:])

        return hook

    def take(self) -> list:
        m = [self.mass[li] for li in sorted(self.mass)]
        self.mass = {}
        return m

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def fit_affine(acc, sel, layout, tg, ridge_scaled, kinds=("k", "v")):
    return {
        kd: Mapper(
            {
                (li, h): fit_block(
                    acc[kd],
                    target=li * tg.n_kv_heads + h,
                    target_layer=li,
                    head=h,
                    kind=kd,
                    layers=sel[(kd, li, h)],
                    ridge=ridge_scaled[kd],
                    head_local=False,
                )
                for li in range(tg.n_layers)
                for h in range(tg.n_kv_heads)
            },
            layout,
            tg,
            kd,
        )
        for kd in kinds
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--cont-len", type=int, default=64)
    ap.add_argument("--calib", type=int, default=96)
    ap.add_argument("--train", type=int, default=96, help="the empirical knee")
    ap.add_argument("--eval", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr-grid", default="3e-4,1e-3")
    ap.add_argument("--k-best", type=int, default=16, help="unweighted initializer")
    ap.add_argument("--k-bytes", type=int, default=8, help="weighted initializer")
    ap.add_argument("--ridge", type=float, default=1e-5)
    ap.add_argument("--exponent", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--solve-dtype", default="float32")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    sdtype = getattr(torch, args.solve_dtype)
    t_start = time.time()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.target)
    stok = AutoTokenizer.from_pretrained(args.source)

    # The weighted calibration needs the target's own attention weights, which
    # only the eager path produces. Everything after it runs under SDPA, where
    # the behavioural backward is affordable. The target is therefore loaded
    # eager, used once, and re-loaded; the alternative is paying eager
    # attention's memory for every one of the thousands of training steps.
    models, geom = {}, {}
    for role, mid in (("source", args.source), ("target", args.target)):
        m = AutoModelForCausalLM.from_pretrained(
            mid, dtype=dtype, attn_implementation="sdpa"
        ).to(dev)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models[role], geom[role] = m, describe(m, mid)
    sg, tg = geom["source"], geom["target"]

    pair = check_pair(sg, tg, tokenizers_match=(stok.get_vocab() == tok.get_vocab()))
    assert not pair.get("blockers"), pair
    log(f"pair: {pair}")

    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    n_targets = tg.n_layers * tg.n_kv_heads
    pos = torch.arange(args.ctx, device=dev)

    total = args.calib + args.train + args.eval
    chunks = wikitext_chunks(tok, total, args.ctx, args.cont_len, args.seed)
    calib = chunks[: args.calib]
    trainset = chunks[args.calib : args.calib + args.train]
    n_dev = max(4, len(trainset) // 5)
    held = chunks[args.calib + args.train :]
    for a, b in ((calib, held), (trainset, held), (calib, trainset)):
        assert not ({c for c, _ in a} & {c for c, _ in b})
    log(
        f"{len(calib)} calibration / {len(trainset) - n_dev} correction "
        f"({n_dev} held back to pick the rate) / {len(held)} held-out"
    )

    def source_features(sp):
        return (
            flat_features(to_content_keys(sp)).to(sdtype),
            flat_features([v.float() for v in sp.values]).to(sdtype),
        )

    # ---- unweighted accumulation, shared by both widths ------------------
    acc = {
        k: Accumulator(layout, n_targets, tg.head_dim, device=dev) for k in ("k", "v")
    }
    rope_checks = {}
    for ci, (_, ids) in enumerate(calib):
        ids = ids[: args.ctx].unsqueeze(0).to(dev)
        sp = prefill(models["source"], ids, sg)
        tp = prefill(models["target"], ids, tg)
        if ci == 0:
            # The whole method fits keys in a de-rotated frame and re-rotates
            # them on the way back in. If that frame is wrong every number
            # downstream is meaningless, so it is checked against both models
            # on real activations before any of them are produced.
            rope_checks = {
                role: {
                    kk: vv for kk, vv in verify_rope(pp).items() if kk != "per_layer"
                }
                for role, pp in (("source", sp), ("target", tp))
            }
            for role, chk in rope_checks.items():
                assert chk["passed"], f"{role} rope frame failed: {chk}"
            log(f"rope frame verified: {json.dumps(rope_checks)}")
        Xk, Xv = source_features(sp)
        acc["k"].add(Xk, target_blocks(to_content_keys(tp)))
        acc["v"].add(Xv, target_blocks([v.float() for v in tp.values]))
        del sp, tp, Xk, Xv
        if (ci + 1) % 32 == 0:
            log(f"  calibrated on {ci + 1}/{len(calib)}")
    scale = {k: float(acc[k].xtx.diagonal().mean()) for k in ("k", "v")}
    ridge_scaled = {k: args.ridge * scale[k] for k in ("k", "v")}

    sel = {}
    for k in (args.k_best, args.k_bytes):
        for kd in ("k", "v"):
            for li in range(tg.n_layers):
                for h in range(tg.n_kv_heads):
                    sel[(k, kd, li, h)] = select_layers(
                        acc[kd],
                        li * tg.n_kv_heads + h,
                        h,
                        ridge_scaled[kd],
                        k,
                        False,
                    )
    log(f"layer selection done for k={args.k_best} and k={args.k_bytes}")

    inits = {}
    for k in (args.k_best, args.k_bytes):
        s = {(kd, li, h): sel[(k, kd, li, h)] for (kk, kd, li, h) in sel if kk == k}
        inits[f"unw_k{k}"] = fit_affine(acc, s, layout, tg, ridge_scaled)
        log(f"unweighted affine fitted at k={k}")

    # ---- attention-weighted accumulation at the byte-efficient width -----
    kb = args.k_bytes
    wacc = {
        (kd, li, h): WeightedAccumulator(
            layout.columns_for(sel[(kb, kd, li, h)], h, False), tg.head_dim, device=dev
        )
        for kd in ("k", "v")
        for li in range(tg.n_layers)
        for h in range(tg.n_kv_heads)
    }
    del models["target"]
    torch.cuda.empty_cache()
    eager = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dtype, attn_implementation="eager"
    ).to(dev)
    eager.eval()
    for p in eager.parameters():
        p.requires_grad_(False)
    coll = MassCollector(eager, tg.n_kv_heads, args.ctx)
    for ci, (_, ids) in enumerate(calib):
        ids = ids[: args.ctx].unsqueeze(0).to(dev)
        sp = prefill(models["source"], ids, sg)
        cache = DynamicCache()
        with torch.no_grad():
            eager(
                input_ids=ids,
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
            )
        masses = coll.take()
        Xk, Xv = source_features(sp)
        Yk = target_blocks(
            [derot(l.keys.detach().float(), pos, tg.rope_theta) for l in cache.layers]
        )
        Yv = target_blocks([l.values.detach().float() for l in cache.layers])
        for li in range(tg.n_layers):
            mass = masses[li].to(dev)
            for h in range(tg.n_kv_heads):
                t = li * tg.n_kv_heads + h
                w = normalise(mass[h].pow(args.exponent).unsqueeze(0))[0]
                wacc[("k", li, h)].add(Xk, Yk[t], w)
                wacc[("v", li, h)].add(Xv, Yv[t], w)
        del sp, cache, masses, Xk, Xv, Yk, Yv
        if (ci + 1) % 32 == 0:
            log(f"  weighted calibration on {ci + 1}/{len(calib)}")
    coll.close()
    del eager, coll
    torch.cuda.empty_cache()

    inits[f"wtd_k{kb}"] = {
        kd: Mapper(
            {
                (li, h): fit_weighted(
                    wacc[(kd, li, h)],
                    target_layer=li,
                    head=h,
                    kind=kd,
                    layers=sel[(kb, kd, li, h)],
                    ridge=ridge_scaled[kd],
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
    torch.cuda.empty_cache()
    log(f"attention-weighted affine fitted at k={kb}")

    models["target"] = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dtype, attn_implementation="sdpa"
    ).to(dev)
    models["target"].eval()
    for p in models["target"].parameters():
        p.requires_grad_(False)

    # ---- what the corrections train against ------------------------------
    train_cache = []
    for ci, (_, ids) in enumerate(trainset):
        ids = ids.to(dev)
        prompt = ids[: args.ctx].unsqueeze(0)
        cont = ids[args.ctx : args.ctx + args.cont_len].unsqueeze(0)
        sp = prefill(models["source"], prompt, sg)
        tp = prefill(models["target"], prompt, tg)
        Xk, Xv = source_features(sp)
        with torch.no_grad():
            ref_lp = (
                torch.log_softmax(
                    continuation_logits(
                        models["target"], tp.keys_post, tp.values, cont, args.ctx
                    ).float(),
                    -1,
                )
                .half()
                .cpu()
            )
        train_cache.append(
            {
                "Xk": Xk.detach().half().cpu(),
                "Xv": Xv.detach().half().cpu(),
                "cont": cont.cpu(),
                "ref_lp": ref_lp,
            }
        )
        del sp, tp, Xk, Xv, ref_lp
        if (ci + 1) % 32 == 0:
            log(f"  cached {ci + 1}/{len(trainset)} correction prompts")
    fit_cache, dev_cache = train_cache[:-n_dev], train_cache[-n_dev:]
    log(f"cached {len(fit_cache)} correction + {len(dev_cache)} rate-selection")

    # ---- training --------------------------------------------------------
    lrs = [float(x) for x in args.lr_grid.split(",")]

    def clone(m):
        """A private copy of a fitted map.

        Merging a trained linear correction rewrites the map it corrects in
        place. Every arm that starts from the same fit therefore needs its own
        copy, or the first merge would quietly change the uncorrected control
        and the nonlinear arm that were supposed to be compared against it.
        """
        maps = {}
        for key, a in m.maps.items():
            maps[key] = AffineMap(
                M=a.M.clone(),
                b=a.b.clone(),
                layers=a.layers,
                head=a.head,
                kind=a.kind,
                target_layer=a.target_layer,
                ridge=a.ridge,
                head_local=a.head_local,
                n_calib_tokens=a.n_calib_tokens,
                info=dict(a.info),
            )
        return Mapper(maps, m.layout, m.geom, m.kind)

    def build(base, linear):
        rk = ResidualMapper(
            clone(base["k"]), layout, tg, "k", args.hidden, linear=linear
        )
        rv = ResidualMapper(
            clone(base["v"]), layout, tg, "v", args.hidden, linear=linear
        )
        return rk.to(dev), rv.to(dev)

    def behaviour(rk, rv, item, grad):
        Xk = item["Xk"].to(dev).to(sdtype)
        Xv = item["Xv"].to(dev).to(sdtype)
        pk, pv = rk(Xk), rv(Xv)
        keys = [rerot(kk.float(), pos, tg.rope_theta).to(dtype) for kk in pk]
        vals = [v.to(dtype) for v in pv]
        fn = continuation_logits_grad if grad else continuation_logits
        lg = fn(models["target"], keys, vals, item["cont"].to(dev), args.ctx)
        return behaviour_loss(item["ref_lp"], lg)

    def dev_loss(rk, rv, items):
        with torch.no_grad():
            return float(np.mean([float(behaviour(rk, rv, it, False)) for it in items]))

    def train(base, linear, tag, lr):
        rk, rv = build(base, linear)
        rk.freeze_affine()
        rv.freeze_affine()
        opt = torch.optim.Adam(
            list(rk.blocks.parameters()) + list(rv.blocks.parameters()), lr=lr
        )
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        first, last = None, None
        # Seeded here rather than once per run, so every arm sees the same
        # sequence of examples. Without that the comparison would include a
        # difference in which prompts each arm happened to be shown.
        g = torch.Generator().manual_seed(args.seed)
        for step in range(args.steps):
            item = fit_cache[int(torch.randint(len(fit_cache), (1,), generator=g))]
            loss = behaviour(rk, rv, item, True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            v = float(loss.detach())
            first = v if first is None else first
            last = v
            if (step + 1) % 150 == 0:
                log(f"    {tag} lr={lr:g} step {step + 1}/{args.steps} loss {v:.5f}")
            del loss
        rep = TrainReport(
            objective="behaviour",
            steps=args.steps,
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
        rep.dev_loss = dev_loss(rk, rv, dev_cache)
        log(
            f"  {tag} lr={lr:g}: train {rep.first_loss:.5f} -> {rep.last_loss:.5f}, "
            f"dev {rep.dev_loss:.5f}, {rep.train_seconds:.0f}s, "
            f"{rep.residual_params / 1e6:.2f}M params"
        )
        return rk, rv, rep

    def train_best(base, linear, tag):
        best = None
        for lr in lrs:
            rk, rv, r = train(base, linear, tag, lr)
            if best is None or r.dev_loss < best[2].dev_loss:
                best = (rk, rv, r)
            else:
                del rk, rv
            torch.cuda.empty_cache()
        log(f"  {tag}: chose lr={best[2].lr:g}")
        return best

    arms, reports = {}, {}
    for name, base in inits.items():
        # Stored at serving precision, like every other arm. A map left at the
        # double precision it was solved in is converted on each apply --
        # AffineMap.apply casts to the input's dtype every call -- so an
        # uncorrected arm would pay a conversion of its whole weight set that
        # a corrected arm, whose weights were cast once when it was built,
        # does not. That asymmetry is the mirror image of the one that made
        # the corrected arms look slow, and it flatters them just as much.
        arms[name] = {
            "k": base["k"].cast(sdtype),
            "v": base["v"].cast(sdtype),
            "corrected": False,
        }
    promoted = [f"unw_k{args.k_best}", f"wtd_k{kb}"]
    for base_name in promoted:
        for linear in (True, False):
            tag = f"{base_name}+{'lin' if linear else 'mlp'}"
            log(f"training {tag}")
            rk, rv, rep = train_best(inits[base_name], linear, tag)
            reports[tag] = rep.to_dict()
            reports[tag]["linear"] = linear
            reports[tag]["init"] = base_name
            # The frozen map underneath is stored at the double precision the
            # ridge was solved in. A plain map is cast down when it is applied
            # (AffineMap.apply casts to the input's dtype); a corrected one is
            # not, because its forward casts the input up to the map instead.
            # Left alone that asymmetry times the corrected arms in double and
            # the arms they are compared against in float32, which reads as the
            # correction being expensive when what is expensive is the
            # precision.
            rk.cast(sdtype)
            rv.cast(sdtype)
            arms[tag] = {"k": rk, "v": rv, "corrected": True, "linear": linear}

    # A trained linear correction is only interesting if it really does fold
    # away, so it is folded here and the folded map is what gets scored. If
    # the merge were wrong the arm would score differently from its unfolded
    # self, which the check below would catch.
    for tag, a in list(arms.items()):
        if a.get("corrected") and a.get("linear"):
            with torch.no_grad():
                probe = torch.randn(8, a["k"].cols[(0, 0)].numel(), device=dev).to(
                    sdtype
                )
                cols = a["k"].cols[(0, 0)].to(dev)
                full = torch.zeros(8, layout.n_features, device=dev, dtype=sdtype)
                full[:, cols] = probe
                before = a["k"](full)[0][0, 0].clone()
                merged_k = a["k"].merge()
                merged_v = a["v"].merge()
                after = merged_k.apply(full)[0][0, 0]
                gap = float((before - after).abs().max())
            log(f"  {tag}: merged, max |unfolded - folded| = {gap:.3e}")
            reports[tag]["merge_max_abs_gap"] = gap
            arms[tag] = {
                "k": merged_k,
                "v": merged_v,
                "corrected": True,
                "folded": True,
            }

    # ---- held-out --------------------------------------------------------
    def blocks_of(arm, Xk, Xv):
        if isinstance(arm["k"], Mapper):
            pk, pv = arm["k"].apply(Xk), arm["v"].apply(Xv)
        else:
            with torch.no_grad():
                pk, pv = arm["k"](Xk), arm["v"](Xv)
        keys = [rerot(b[0].float(), pos, tg.rope_theta).unsqueeze(0) for b in pk]
        return [k.to(dtype) for k in keys], [v.to(dtype) for v in pv]

    latency = {}
    for name, arm in arms.items():
        ids0 = held[0][1].to(dev)[: args.ctx].unsqueeze(0)
        sp0 = prefill(models["source"], ids0, sg)
        Xk0, Xv0 = source_features(sp0)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        for _ in range(3):
            blocks_of(arm, Xk0, Xv0)
        torch.cuda.synchronize()
        ts = []
        for _ in range(10):
            ev[0].record()
            blocks_of(arm, Xk0, Xv0)
            ev[1].record()
            torch.cuda.synchronize()
            ts.append(ev[0].elapsed_time(ev[1]))
        latency[name] = {
            "p50_ms": float(np.percentile(ts, 50)),
            "p95_ms": float(np.percentile(ts, 95)),
        }
        del sp0, Xk0, Xv0
    log(
        "online apply latency: "
        + ", ".join(f"{n} {v['p50_ms']:.1f} ms" for n, v in latency.items())
    )

    rows = []
    for ci, (cid, ids) in enumerate(held):
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
        rows.append(dict(score(ref, ref, cont), arm="target_native", context=cid))
        with torch.no_grad():
            lg = continuation_logits(models["target"], None, None, cont, args.ctx)
        rows.append(dict(score(ref, lg, cont), arm="no_cache", context=cid))
        for name, arm in arms.items():
            keys, vals = blocks_of(arm, Xk, Xv)
            with torch.no_grad():
                lg = continuation_logits(models["target"], keys, vals, cont, args.ctx)
            rows.append(dict(score(ref, lg, cont), arm=name, context=cid))
            del keys, vals, lg
        del sp, tp, Xk, Xv, ref
        if (ci + 1) % 8 == 0:
            log(f"  scored {ci + 1}/{len(held)} held-out")

    arm_bytes = {}
    for name, arm in arms.items():
        if isinstance(arm["k"], Mapper):
            arm_bytes[name] = {
                "map_bytes": arm["k"].n_bytes + arm["v"].n_bytes,
                "residual_bytes": 0.0,
                "folded": bool(arm.get("folded")),
            }
        else:
            arm_bytes[name] = {
                "map_bytes": arm["k"].affine.n_bytes + arm["v"].affine.n_bytes,
                "residual_bytes": (
                    arm["k"].n_residual_params + arm["v"].n_residual_params
                )
                * 2.0,
                "folded": False,
            }
        arm_bytes[name]["total_bytes"] = (
            arm_bytes[name]["map_bytes"] + arm_bytes[name]["residual_bytes"]
        )

    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    summary = {
        a: {
            m: float(np.mean([x[m] for x in rs]))
            for m in ("kl", "delta_nll", "argmax_flip", "nll")
        }
        for a, rs in by_arm.items()
    }
    floor = summary["no_cache"]["kl"]
    for a in summary:
        summary[a]["gap_closed"] = 1.0 - summary[a]["kl"] / floor

    order = sorted(summary, key=lambda a: summary[a]["kl"])
    log("held-out, best first:")
    for a in order:
        s = summary[a]
        log(
            f"  {a:<20s} kl {s['kl']:.4f}  dnll {s['delta_nll']:+.4f}  "
            f"flip {s['argmax_flip']:.4f}  closes {100 * s['gap_closed']:.1f}%"
        )

    out = {
        "source": args.source,
        "target": args.target,
        "pair": pair,
        "rope_checks": rope_checks,
        "config": vars(args),
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "calib_indices": [c for c, _ in calib],
        "train_indices": [c for c, _ in trainset],
        "eval_indices": [c for c, _ in held],
        "affine_bytes": {
            n: inits[n]["k"].n_bytes + inits[n]["v"].n_bytes for n in inits
        },
        "arm_bytes": arm_bytes,
        "latency": latency,
        "reports": reports,
        "summary": summary,
        "wall_s": time.time() - t_start,
        "peak_alloc_gib": (
            torch.cuda.max_memory_allocated() / GIB if dev == "cuda" else 0.0
        ),
    }
    # Persist every arm, so the task gate scores exactly the weights this run
    # produced rather than a re-fit that would differ from them. A folded arm
    # is an ordinary affine map and needs nothing else; an unfolded one has to
    # carry its residual network, which is precisely the cost that folding
    # removes.
    arm_dir = os.path.join(args.out_dir, "arms")
    os.makedirs(arm_dir, exist_ok=True)
    saved = {}
    src_id = freeze.model_identity(args.source, models["source"], stok)
    tgt_id = freeze.model_identity(args.target, models["target"], tok)
    for name, arm in arms.items():
        base = os.path.join(arm_dir, name.replace("+", "_"))
        mk = arm["k"] if isinstance(arm["k"], Mapper) else arm["k"].affine
        mv = arm["v"] if isinstance(arm["v"], Mapper) else arm["v"].affine
        man = freeze.save(
            base + ".pt",
            mk,
            mv,
            source=src_id,
            target=tgt_id,
            fit_context=args.ctx,
            calib_doc_ids=[c for c, _ in calib],
            dev_doc_ids=[c for c, _ in trainset[-n_dev:]],
            eval_doc_ids=[c for c, _ in held],
            config={**vars(args), "arm": name},
        )
        rec = {"affine": base + ".pt", "joint_hash": man["joint_weight_sha256"]}
        if not isinstance(arm["k"], Mapper):
            torch.save(
                {
                    "k": arm["k"].blocks.state_dict(),
                    "v": arm["v"].blocks.state_dict(),
                    "hidden": args.hidden,
                    "linear": bool(arm.get("linear")),
                },
                base + ".residual.pt",
            )
            rec["residual"] = base + ".residual.pt"
            rec["residual_bytes_on_disk"] = os.path.getsize(base + ".residual.pt")
        saved[name] = rec
    log(f"saved {len(saved)} arms to {arm_dir}")
    out["saved_arms"] = saved

    with open(os.path.join(args.out_dir, "h1.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    log(f"wrote {args.out_dir}/h1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
