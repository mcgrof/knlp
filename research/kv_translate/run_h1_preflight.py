# SPDX-License-Identifier: GPL-2.0
"""Does a behavioural backward through a frozen 7B target actually fit?

Training a translator against what the target says, rather than against the
tensors the target would have produced, means gradients run from the target's
logits back through its attention into the correction. The target's weights
are frozen and need no gradients of their own, but the graph still has to
exist, and the last measured inference peak on this pair was already near 35
GiB on a 40 GB card. Whether the backward fits is therefore a real question
and not a formality, and it decides whether the next stage runs locally on a
48 GB W7900 or needs a rented 80 GB card.

Memory is a function of shapes, not of values, so this fits nothing. It builds
a mapper with the true per-block geometry -- the real layer selection width,
the real source and target head counts, the real head dimension -- fills it
with noise, and runs the identical forward, loss and backward the training
stage will run. That keeps the preflight to a couple of minutes and costs no
calibration.

Reported per configuration: peak allocated and reserved memory, the split
between frozen weights and everything else, and whether the backward
completed. A configuration that runs out of memory is recorded as such rather
than crashing the sweep, because knowing the largest context that fits is the
point.

Env: HIP_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.kv_translate.fit import AffineMap, SourceLayout  # noqa: E402
from research.kv_translate.pairs import (  # noqa: E402
    describe,
    prefill,
    to_content_keys,
    to_stored_keys,
)
from research.kv_translate.refine import ResidualMapper, behaviour_loss  # noqa: E402
from research.kv_translate.run_a0 import (  # noqa: E402
    continuation_logits,
    continuation_logits_grad,
    wikitext_chunks,
)
from research.kv_translate.run_a1 import Mapper, flat_features  # noqa: E402

GIB = float(2**30)


def synthetic_mapper(layout, sgeom, tgeom, k, head_local, dtype, dev, seed=0):
    """A mapper with the true shapes and arbitrary contents.

    The selection is a contiguous span of ``k`` source layers ending at the
    proportional depth, which is what the fitted selections in this lane
    actually look like; a scattered selection of the same width would occupy
    identical memory in any case.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    maps = {}
    for li in range(tgeom.n_layers):
        end = min(
            sgeom.n_layers, max(k, round((li + 1) * sgeom.n_layers / tgeom.n_layers))
        )
        layers = tuple(range(end - k, end))
        for h in range(tgeom.n_kv_heads):
            src_head = h % sgeom.n_kv_heads if head_local else 0
            cols = layout.columns_for(layers, src_head, head_local)
            d_in = int(cols.numel())
            maps[(li, h)] = AffineMap(
                M=(torch.randn(d_in, tgeom.head_dim, generator=g) * 0.02).to(
                    dev, dtype
                ),
                b=torch.zeros(tgeom.head_dim, device=dev, dtype=dtype),
                layers=layers,
                head=src_head,
                kind="k",
                target_layer=li,
                ridge=0.0,
                head_local=head_local,
                n_calib_tokens=0,
            )
    return Mapper(maps, layout, tgeom, "k")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--contexts", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--ks", type=int, nargs="+", default=[16])
    ap.add_argument("--cont-len", type=int, default=64)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16", help="weights and mapper dtype")
    ap.add_argument("--head-local", action="store_true", default=False)
    ap.add_argument("--no-head-local", dest="head_local", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.target)
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
    torch.cuda.synchronize()
    weights_gib = torch.cuda.memory_allocated() / GIB
    log(
        f"source {sg.n_layers}L x {sg.n_kv_heads}kv x {sg.head_dim} | "
        f"target {tg.n_layers}L x {tg.n_kv_heads}kv x {tg.head_dim} | "
        f"frozen weights {weights_gib:.2f} GiB in {args.dtype}"
    )

    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    rows = []

    for ctx in args.contexts:
        chunks = wikitext_chunks(tok, args.prompts, ctx, args.cont_len, args.seed)
        for k in args.ks:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            tag = f"ctx{ctx}_k{k}"
            row = {
                "ctx": ctx,
                "k": k,
                "hidden": args.hidden,
                "dtype": args.dtype,
                "head_local": args.head_local,
                "cont_len": args.cont_len,
                "weights_gib": weights_gib,
                "ok": False,
                "error": None,
            }
            try:
                mk = synthetic_mapper(
                    layout, sg, tg, k, args.head_local, dtype, dev, seed=args.seed
                )
                mv = synthetic_mapper(
                    layout, sg, tg, k, args.head_local, dtype, dev, seed=args.seed + 1
                )
                rk = ResidualMapper(mk, layout, tg, "k", args.hidden).to(dev)
                rv = ResidualMapper(mv, layout, tg, "v", args.hidden).to(dev)
                rk.freeze_affine()
                rv.freeze_affine()
                row["residual_params"] = rk.n_residual_params + rv.n_residual_params
                row["affine_params"] = sum(
                    m.n_params for m in list(mk.maps.values()) + list(mv.maps.values())
                )
                params = list(rk.blocks.parameters()) + list(rv.blocks.parameters())
                opt = torch.optim.Adam(params, lr=1e-4)

                t0 = time.time()
                for ci, (_, ids) in enumerate(chunks):
                    ids = ids.to(dev)
                    p_ids = ids[:ctx].unsqueeze(0)
                    c_ids = ids[ctx : ctx + args.cont_len].unsqueeze(0)
                    with torch.no_grad():
                        sp = prefill(models["source"], p_ids, sg)
                        Xk = flat_features(to_content_keys(sp)).to(dtype)
                        Xv = flat_features([v.float() for v in sp.values]).to(dtype)
                        tp = prefill(models["target"], p_ids, tg)
                        ref = torch.log_softmax(
                            continuation_logits(
                                models["target"], tp.keys_post, tp.values, c_ids, ctx
                            ).float(),
                            dim=-1,
                        ).to(torch.float16)
                        del sp, tp
                    opt.zero_grad(set_to_none=True)
                    keys = to_stored_keys([b[0] for b in rk(Xk)], tg, ctx)
                    keys = [x.unsqueeze(0).to(dtype) for x in keys]
                    vals = [x.to(dtype) for x in rv(Xv)]
                    lg = continuation_logits_grad(
                        models["target"], keys, vals, c_ids, ctx
                    )
                    loss = behaviour_loss(ref, lg)
                    loss.backward()
                    opt.step()
                    del keys, vals, lg, loss, Xk, Xv, ref
                row["seconds_per_step"] = (time.time() - t0) / max(len(chunks), 1)
                row["ok"] = True
            except torch.cuda.OutOfMemoryError as e:
                row["error"] = f"OOM: {str(e)[:160]}"
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                row["error"] = f"OOM: {str(e)[:160]}"

            torch.cuda.synchronize()
            row["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / GIB
            row["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / GIB
            row["peak_over_weights_gib"] = row["peak_alloc_gib"] - weights_gib
            rows.append(row)
            log(
                f"{tag}: {'ok' if row['ok'] else row['error']} | "
                f"peak alloc {row['peak_alloc_gib']:.2f} GiB "
                f"(+{row['peak_over_weights_gib']:.2f} over weights), "
                f"reserved {row['peak_reserved_gib']:.2f} GiB"
                + (f", {row['seconds_per_step']:.2f} s/step" if row["ok"] else "")
            )
            try:
                del rk, rv, mk, mv, opt, params
            except NameError:
                pass
            torch.cuda.empty_cache()

    total = (
        torch.cuda.get_device_properties(0).total_memory / GIB if dev == "cuda" else 0.0
    )
    largest = max((r["ctx"] for r in rows if r["ok"]), default=None)
    verdict = {
        "device_total_gib": total,
        "largest_context_that_fits": largest,
        "all_requested_contexts_fit": all(r["ok"] for r in rows),
        "headroom_gib_at_largest": (
            total - max((r["peak_reserved_gib"] for r in rows if r["ok"]), default=0.0)
            if largest is not None
            else None
        ),
        "needs_larger_card": not all(r["ok"] for r in rows),
    }
    out = {
        "source": args.source,
        "target": args.target,
        "config": vars(args),
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "rows": rows,
        "verdict": verdict,
    }
    with open(os.path.join(args.out, "preflight.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    log(f"verdict: {json.dumps(verdict)}")
    log(f"wrote {args.out}/preflight.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
