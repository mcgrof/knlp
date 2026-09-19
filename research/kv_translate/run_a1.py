# SPDX-License-Identifier: GPL-2.0
"""A1: can a structured affine map hand one model's cache to another?

The primary gate. A source model prefills a prompt; a per-target-head ridge map
turns that cache into something the target model can decode from. The question
is whether the target then behaves like it prefilled the prompt itself, and
whether doing so is cheaper than just letting it.

Deliberately no receiver metric is attached. The screen that preceded this lane
established that a global Jacobian second moment does not rank cache damage
better than cheap alternatives, and it never tested a mapper. So this is the
strongest *structural* baseline the literature points at -- paired native
caches, de-rotated content keys, architecture-aligned head support, separate key
and value maps, and selected source layers -- and nothing else.

Arms are chosen so that a failure is attributable. Mapping keys and values
together says whether the idea works; mapping only one and taking the other
native says which half is responsible. That split matters here because keys and
values fail differently in every prior result in this program.

Selection happens on a development split, never on the held-out prompts the
verdict is read from, and the controls from A0 are carried through so a mapper
that emits plausible-looking target state without carrying the source content
cannot pass.

Env: HIP_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
Run:
    python -m research.kv_translate.run_a1 --out-dir <dir>
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
from research.jacobian_kv.accounting import KVGeometry, cache_bytes  # noqa: E402
from research.jacobian_kv.evaluate import (  # noqa: E402
    argmax_flip_rate,
    delta_nll,
    teacher_forced_kl,
)
from research.kv_translate.fit import (  # noqa: E402
    Accumulator,
    SourceLayout,
    fit_block,
    select_layers,
)
from research.kv_translate.pairs import (  # noqa: E402
    check_pair,
    describe,
    prefill,
    to_content_keys,
    to_stored_keys,
)
from research.kv_translate.run_a0 import (  # noqa: E402
    continuation_logits,
    make_cache,
    score,
    wikitext_chunks,
)


def flat_features(blocks) -> torch.Tensor:
    """``[L][1, H, T, D]`` -> ``[T, L*H*D]`` in the layout the fitter expects."""
    parts = [b[0].permute(1, 0, 2).reshape(b.shape[2], -1) for b in blocks]
    return torch.cat(parts, dim=1)


def target_blocks(blocks) -> torch.Tensor:
    """``[L][1, H, T, D]`` -> ``[L*H, T, D]``, one row per target block."""
    return torch.cat([b[0] for b in blocks], dim=0)


class Mapper:
    """A fitted set of block maps plus the cost of owning it."""

    def __init__(self, maps, layout, tgt_geom, kind):
        self.maps = maps  # {(layer, head): AffineMap}
        self.layout = layout
        self.geom = tgt_geom
        self.kind = kind

    @property
    def n_bytes(self) -> float:
        """Stored at half precision, which is what a deployed mapper would use."""
        return sum(m.n_params for m in self.maps.values()) * 2.0

    @torch.no_grad()
    def apply(self, X: torch.Tensor) -> list:
        """``[T, d_full]`` source features -> ``[L][1, H, T, D]`` target blocks."""
        out = []
        for li in range(self.geom.n_layers):
            heads = []
            for h in range(self.geom.n_kv_heads):
                m = self.maps[(li, h)]
                cols = self.layout.columns_for(m.layers, h, m.head_local).to(X.device)
                heads.append(m.apply(X[:, cols]))
            out.append(torch.stack(heads, 0).unsqueeze(0))
        return out


def build(acc, layout, tgt_geom, kind, ridge, k, head_local):
    maps = {}
    for li in range(tgt_geom.n_layers):
        for h in range(tgt_geom.n_kv_heads):
            t = li * tgt_geom.n_kv_heads + h
            layers = select_layers(acc, t, h, ridge, k, head_local)
            maps[(li, h)] = fit_block(
                acc,
                target=t,
                target_layer=li,
                head=h,
                kind=kind,
                layers=layers,
                ridge=ridge,
                head_local=head_local,
            )
    return Mapper(maps, layout, tgt_geom, kind)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--cont-len", type=int, default=64)
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--dev", type=int, default=8)
    ap.add_argument("--eval", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--ridge-grid", default="1e-4,1e-2,1e0,1e2")
    ap.add_argument("--k-grid", default="2,4,8")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default="a1")
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
    log(
        f"source {sg.n_layers}L D={sg.head_dim} -> target {tg.n_layers}L D={tg.head_dim}"
    )
    compat = check_pair(sg, tg)
    assert not compat["blockers"], compat["blockers"]

    total = args.calib + args.dev + args.eval
    chunks = wikitext_chunks(tok, total, args.ctx, args.cont_len, args.seed)
    calib = chunks[: args.calib]
    devset = chunks[args.calib : args.calib + args.dev]
    held = chunks[args.calib + args.dev :]
    assert not ({c for c, _ in calib} & {c for c, _ in held}), "calib/eval overlap"
    log(f"{len(calib)} calibration / {len(devset)} dev / {len(held)} held-out")

    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    n_targets = tg.n_layers * tg.n_kv_heads
    acc = {
        kind: Accumulator(layout, n_targets, tg.head_dim, device=dev)
        for kind in ("k", "v")
    }

    def features(ids):
        """Source features and target blocks for one prompt, both key frames."""
        sp = prefill(models["source"], ids, sg)
        tp = prefill(models["target"], ids, tg)
        Xk = flat_features(to_content_keys(sp))
        Xv = flat_features([v.float() for v in sp.values])
        Yk = target_blocks(to_content_keys(tp))
        Yv = target_blocks([v.float() for v in tp.values])
        return (Xk, Xv, Yk, Yv), sp, tp

    # ---- calibration pass ------------------------------------------------
    t_fit = time.time()
    calib_tokens = 0
    for i, (_, ids) in enumerate(calib):
        ids = ids[: args.ctx].unsqueeze(0).to(dev)
        (Xk, Xv, Yk, Yv), sp, tp = features(ids)
        acc["k"].add(Xk, Yk)
        acc["v"].add(Xv, Yv)
        calib_tokens += args.ctx
        del Xk, Xv, Yk, Yv, sp, tp
        if (i + 1) % 16 == 0:
            log(f"  accumulated {i + 1}/{len(calib)} calibration prompts")
    log(f"statistics done in {time.time() - t_fit:.1f}s over {calib_tokens} tokens")

    # ---- model selection on the dev split --------------------------------
    def translate(sp, mapper_k, mapper_v, n_tokens):
        Xk = flat_features(to_content_keys(sp))
        Xv = flat_features([v.float() for v in sp.values])
        content = mapper_k.apply(Xk)
        keys = to_stored_keys(content, tg, n_tokens)
        values = mapper_v.apply(Xv)
        return [k.to(dtype) for k in keys], [v.to(dtype) for v in values]

    def dev_kl(mk, mv):
        tot = []
        for _, ids in devset:
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
            tot.append(float(teacher_forced_kl(ref, lg).mean()))
            del sp, tp, ref, lg, k, v
        return float(np.mean(tot))

    ridges = [float(x) for x in args.ridge_grid.split(",")]
    ks = [int(x) for x in args.k_grid.split(",")]
    trials = []
    best = None
    for head_local in (True, False):
        for k in ks:
            for r in ridges:
                scale = float(acc["k"].xtx.diagonal().mean())
                mk = build(acc["k"], layout, tg, "k", r * scale, k, head_local)
                mv = build(
                    acc["v"],
                    layout,
                    tg,
                    "v",
                    r * float(acc["v"].xtx.diagonal().mean()),
                    k,
                    head_local,
                )
                kl = dev_kl(mk, mv)
                bytes_ = mk.n_bytes + mv.n_bytes
                trials.append(
                    {
                        "head_local": head_local,
                        "k": k,
                        "ridge": r,
                        "dev_kl": kl,
                        "mapper_bytes": bytes_,
                    }
                )
                log(
                    f"  dev head_local={head_local} k={k} ridge={r:g}: KL {kl:.4f} "
                    f"({bytes_ / 2**20:.1f} MiB)"
                )
                if best is None or kl < best[0]:
                    best = (kl, head_local, k, r, mk, mv)
                else:
                    del mk, mv
    _, bl_head_local, bl_k, bl_ridge, mk, mv = best
    log(f"selected: head_local={bl_head_local} k={bl_k} ridge={bl_ridge:g}")

    # ---- held-out evaluation --------------------------------------------
    rows = []
    timings = {"target_prefill_s": [], "source_prefill_s": [], "apply_s": []}
    for ci, (cid, ids) in enumerate(held):
        ids = ids.to(dev)
        prompt = ids[: args.ctx].unsqueeze(0)
        cont = ids[args.ctx : args.ctx + args.cont_len].unsqueeze(0)

        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        tp = prefill(models["target"], prompt, tg)
        if dev == "cuda":
            torch.cuda.synchronize()
        timings["target_prefill_s"].append(time.time() - t0)

        t0 = time.time()
        sp = prefill(models["source"], prompt, sg)
        if dev == "cuda":
            torch.cuda.synchronize()
        timings["source_prefill_s"].append(time.time() - t0)

        ref = continuation_logits(
            models["target"], tp.keys_post, tp.values, cont, args.ctx
        ).float()

        t0 = time.time()
        tk, tv = translate(sp, mk, mv, args.ctx)
        if dev == "cuda":
            torch.cuda.synchronize()
        timings["apply_s"].append(time.time() - t0)

        other = held[(ci + 1) % len(held)][1][: args.ctx].unsqueeze(0).to(dev)
        wp = prefill(models["target"], other, tg)

        arms = {
            "target_native": (tp.keys_post, tp.values),
            "no_cache": (None, None),
            "wrong_prompt": (wp.keys_post, wp.values),
            "translated": (tk, tv),
            "translated_v_only": (tp.keys_post, tv),
            "translated_k_only": (tk, tp.values),
        }
        for name, (k, v) in arms.items():
            lg = continuation_logits(models["target"], k, v, cont, args.ctx).float()
            rows.append({"context": cid, "arm": name, **score(ref, lg, cont)})
            del lg
        del tp, sp, wp, ref, tk, tv, arms
        if dev == "cuda":
            torch.cuda.empty_cache()
        log(f"  held-out {ci + 1}/{len(held)}")

    # ---- report ----------------------------------------------------------
    arms = sorted({r["arm"] for r in rows})
    summary = {
        a: {
            m: float(np.mean([r[m] for r in rows if r["arm"] == a]))
            for m in ("kl", "delta_nll", "argmax_flip", "nll")
        }
        for a in arms
    }
    log("")
    log(f"{'arm':22s} {'KL':>10s} {'dNLL':>9s} {'flip':>7s} {'NLL':>8s}")
    for a in sorted(summary, key=lambda x: summary[x]["kl"]):
        s = summary[a]
        log(
            f"{a:22s} {s['kl']:10.4f} {s['delta_nll']:9.4f} {s['argmax_flip']:7.3f} {s['nll']:8.4f}"
        )

    tgt_cache = cache_bytes(
        KVGeometry(tg.n_layers, tg.n_kv_heads, tg.head_dim, args.ctx)
    ).total
    mapper_bytes = mk.n_bytes + mv.n_bytes
    med = lambda xs: float(np.median(xs))  # noqa: E731
    floor = summary["no_cache"]["kl"]
    trans = summary["translated"]["kl"]
    cost = {
        "mapper_bytes": mapper_bytes,
        "target_cache_bytes_per_prefix": tgt_cache,
        "mapper_costs_n_target_caches": mapper_bytes / tgt_cache,
        "target_prefill_s": med(timings["target_prefill_s"]),
        "source_prefill_s": med(timings["source_prefill_s"]),
        "apply_s": med(timings["apply_s"]),
        "apply_vs_target_prefill": med(timings["apply_s"])
        / med(timings["target_prefill_s"]),
        "source_plus_apply_vs_target_prefill": (
            (med(timings["source_prefill_s"]) + med(timings["apply_s"]))
            / med(timings["target_prefill_s"])
        ),
    }
    verdict = {
        "carries_source_content": trans < 0.5 * floor,
        "fraction_of_gap_closed": 1.0 - trans / max(floor, 1e-12),
        "beats_wrong_prompt": trans < summary["wrong_prompt"]["kl"],
        "cheaper_than_target_prefill": cost["apply_vs_target_prefill"] < 1.0,
        "translated_kl": trans,
        "no_cache_kl": floor,
        "key_only_kl": summary["translated_k_only"]["kl"],
        "value_only_kl": summary["translated_v_only"]["kl"],
    }
    verdict["A1_ADVANCE"] = bool(
        verdict["carries_source_content"]
        and verdict["beats_wrong_prompt"]
        and verdict["cheaper_than_target_prefill"]
    )
    log("")
    log(f"cost: {json.dumps(cost)}")
    log(f"A1 {'ADVANCE' if verdict['A1_ADVANCE'] else 'STOP'}: {json.dumps(verdict)}")

    out = art.ensure_run_dir(args.out_dir, args.run_id)
    art.write_json(
        os.path.join(out, "a1.json"),
        {
            "source": sg.to_dict(),
            "target": tg.to_dict(),
            "pair": compat,
            "selected": {"head_local": bl_head_local, "k": bl_k, "ridge": bl_ridge},
            "trials": trials,
            "arms": summary,
            "cost": cost,
            "verdict": verdict,
            "calib_tokens": calib_tokens,
            "calib_indices": [c for c, _ in calib],
            "eval_indices": [c for c, _ in held],
            "ctx": args.ctx,
            "cont_len": args.cont_len,
            "seed": args.seed,
            "dtype": args.dtype,
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
