# SPDX-License-Identifier: GPL-2.0
"""G1: measure what applying a cache map actually costs, and make it cheap.

The earlier cost number was not wrong about the code. It measured a path that
applied the map one block at a time, in double precision, rebuilding its column
indices on every call, which reports the cost of dispatching work rather than
of doing it. The fix is to repack the map and measure again -- but only if the
new measurement is of the *same quantity*, against a fair baseline, and only if
a faster arm is also a correct one. An adversarial review of the first version
of this file found it failing all three, so each is now handled explicitly.

**The timed region starts at the stored source cache.** A cache holds keys
after the rotary embedding, so a deployed map must de-rotate and flatten the
source cache on every prefix. An earlier draft computed those outside the
timer, which silently measured less work than the number it was revising and
would have credited the difference to batching. Timing now begins at
``keys_post``/``values``, exactly what a resident source cache is.

**The baseline is a prefill anyone would actually serve.** Eager attention is
needed to hook pre-rotary keys during fitting, and is materially slower than
the scaled-dot-product path a deployment uses, especially as context grows.
Dividing by an eager prefill inflates every ratio in the method's favour, so
the baseline is timed on a separate model loaded with the fast backend, and
that backend is recorded next to the number.

**A faster arm must also be a correct one.** The guidance permits a
lower-precision serving arm only if it costs at most one percent of excess
divergence. Ranking arms by latency alone would let bfloat16 win a
deployability verdict on speed while nobody had checked what it computes, so
every arm is scored for fidelity against a float64 block-at-a-time reference
and is ineligible above a stated bound.

Gates are reported per context rather than as one boolean over the best row
anywhere, the hard gate divides the map's 95th percentile by the baseline's
median (the unit the gate names), results are written after every context so an
out-of-memory rung costs one row rather than the run, and the component profile
is measured in disjoint stages that are asserted to sum to the whole.

Env: TOKENIZERS_PARALLELISM=false
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.cartridges_cas.scripts.cas_kv_rope import rerot  # noqa: E402
from research.jacobian_kv import artifacts as art  # noqa: E402
from research.kv_translate.apply import (  # noqa: E402
    BatchedMapper,
    selection_groups,
    time_callable,
)
from research.kv_translate.fit import (  # noqa: E402
    Accumulator,
    SourceLayout,
    fit_block,
    select_layers,
)
from research.kv_translate.pairs import describe, prefill, to_content_keys  # noqa: E402
from research.kv_translate.run_a0 import wikitext_chunks  # noqa: E402
from research.kv_translate.run_a1 import (
    Mapper,
    flat_features,
    target_blocks,
)  # noqa: E402

FIDELITY_TOL = 1e-2  # relative error against a float64 reference apply


def provenance(model_id, model, tok) -> dict:
    """Identity a later reader can check rather than infer from a name."""
    try:
        from huggingface_hub import snapshot_download

        rev = os.path.basename(
            snapshot_download(model_id, local_files_only=True).rstrip("/")
        )
    except Exception:  # noqa: BLE001
        rev = "unresolved"
    vocab = tok.get_vocab()
    tmpl = getattr(tok, "chat_template", None) or ""
    return {
        "model_id": model_id,
        "revision": rev,
        "tokenizer_sha256": hashlib.sha256(
            json.dumps(sorted(vocab.items()), separators=(",", ":")).encode()
        ).hexdigest()[:32],
        "tokenizer_len": len(vocab),
        "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest()[:32],
        "model_dtype": str(next(model.parameters()).dtype),
        "attn_implementation": getattr(model.config, "_attn_implementation", "unknown"),
    }


def prefill_fn(model, ids):
    """A prefill that fills a cache and nothing else: no vocabulary projection."""
    from transformers import DynamicCache

    base = model.model

    @torch.no_grad()
    def run():
        base(input_ids=ids, past_key_values=DynamicCache(), use_cache=True)

    return run


@torch.no_grad()
def reference_apply(mapper, layout, X64):
    """The block-at-a-time path in float64: what the packing must reproduce."""
    out = {}
    for (li, h), m in mapper.maps.items():
        cols = layout.columns_for(list(m.layers), h, m.head_local).to(X64.device)
        out[(li, h)] = X64[:, cols] @ m.M + m.b
    return out


def fidelity(packed_blocks, ref, geom) -> float:
    """Relative error of a packed apply against the float64 reference."""
    num = 0.0
    den = 0.0
    for (li, h), r in ref.items():
        got = packed_blocks[li][0, h].double()
        num += float((got - r).pow(2).sum())
        den += float(r.pow(2).sum())
    return (num / max(den, 1e-30)) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--contexts", default="512,1024,2048")
    ap.add_argument("--calib", type=int, default=48)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--ridge", type=float, default=1e-5)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--serving-dtypes", default="float32,bfloat16")
    ap.add_argument("--strategies", default="grouped,dense")
    ap.add_argument("--tf32", default="off,on")
    ap.add_argument("--baseline-attn", default="sdpa")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default="g1")
    args = ap.parse_args()

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.target)
    models, geom, prov = {}, {}, {}
    for role, mid in (("source", args.source), ("target", args.target)):
        m = AutoModelForCausalLM.from_pretrained(
            mid, dtype=dtype, attn_implementation="eager"
        ).to(dev)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models[role], geom[role] = m, describe(m, mid)
        prov[role] = provenance(mid, m, AutoTokenizer.from_pretrained(mid))
    sg, tg = geom["source"], geom["target"]

    # A separate target for the baseline, on the backend a deployment serves
    # with. Eager exists only so fitting can hook pre-rotary keys; timing
    # against it would inflate every ratio in this method's favour.
    base_model = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dtype, attn_implementation=args.baseline_attn
    ).to(dev)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    prov["baseline"] = provenance(args.target, base_model, tok)
    log(
        f"{args.source} -> {args.target} in {args.dtype}; "
        f"baseline prefill on {args.baseline_attn}, fit path on eager"
    )

    out_dir = art.ensure_run_dir(args.out_dir, args.run_id)
    results, skipped = [], []

    def persist(verdict=None):
        art.write_json(
            os.path.join(out_dir, "g1.json"),
            {
                "provenance": prov,
                "source": sg.to_dict(),
                "target": tg.to_dict(),
                "numerics": {
                    "fit_dtype": "float64",
                    "model_dtype": args.dtype,
                    "baseline_attn": args.baseline_attn,
                    "fidelity_tolerance": FIDELITY_TOL,
                },
                "config": vars(args),
                "results": results,
                "skipped": skipped,
                "verdict": verdict or {},
                "wall_s": time.time() - t0,
                "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                "versions": art.versions(),
                "code_commit": art.git_state()[0],
            },
        )

    for ctx in [int(c) for c in args.contexts.split(",")]:
        log(f"=== context {ctx}")
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            chunks = wikitext_chunks(tok, args.calib + 4, ctx, 8, args.seed)
            calib, probe = chunks[: args.calib], chunks[args.calib :]
            layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
            n_targets = tg.n_layers * tg.n_kv_heads

            acc = {
                k: Accumulator(layout, n_targets, tg.head_dim, device=dev)
                for k in ("k", "v")
            }
            for _, ids in calib:
                ids = ids[:ctx].unsqueeze(0).to(dev)
                sp = prefill(models["source"], ids, sg)
                tp = prefill(models["target"], ids, tg)
                acc["k"].add(
                    flat_features(to_content_keys(sp)),
                    target_blocks(to_content_keys(tp)),
                )
                acc["v"].add(
                    flat_features([v.float() for v in sp.values]),
                    target_blocks([v.float() for v in tp.values]),
                )
                del sp, tp
            scale = {k: float(acc[k].xtx.diagonal().mean()) for k in ("k", "v")}
            mapper = {
                kd: Mapper(
                    {
                        (li, h): fit_block(
                            acc[kd],
                            target=li * tg.n_kv_heads + h,
                            target_layer=li,
                            head=h,
                            kind=kd,
                            layers=select_layers(
                                acc[kd],
                                li * tg.n_kv_heads + h,
                                h,
                                args.ridge * scale[kd],
                                args.k,
                                False,
                            ),
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
            del acc
            if dev == "cuda":
                torch.cuda.empty_cache()
            ngroups = len(selection_groups(mapper["k"], layout))
            log(f"  fitted; {n_targets} blocks in {ngroups} distinct selections")

            ids = probe[0][1][:ctx].unsqueeze(0).to(dev)
            sp = prefill(models["source"], ids, sg)
            pos = torch.arange(ctx, device=dev)
            # the float64 reference the packings must reproduce
            Xk64 = flat_features(to_content_keys(sp)).double()
            ref_k = reference_apply(mapper["k"], layout, Xk64)
            del Xk64

            base_t = time_callable(
                prefill_fn(base_model, ids), reps=args.reps, device=dev
            )
            src_t = time_callable(
                prefill_fn(models["source"], ids), reps=args.reps, device=dev
            )
            log(
                f"  baseline target prefill p50 {base_t['p50_ms']:.2f} ms "
                f"(p95 {base_t['p95_ms']:.2f}) | source prefill p50 {src_t['p50_ms']:.2f} ms"
            )

            src_bytes = sum(
                t.numel() * t.element_size()
                for t in list(sp.keys_post) + list(sp.values)
            )
            host = torch.empty(src_bytes, dtype=torch.uint8, device="cpu").pin_memory()
            xfer = time_callable(
                lambda: host.to(dev, non_blocking=False), reps=args.reps, device=dev
            )

            for tf32 in args.tf32.split(","):
                torch.backends.cuda.matmul.allow_tf32 = tf32 == "on"
                torch.backends.cudnn.allow_tf32 = tf32 == "on"
                for strategy in args.strategies.split(","):
                    for sd in args.serving_dtypes.split(","):
                        sdt = getattr(torch, sd)
                        bk = BatchedMapper(
                            mapper["k"], layout, tg, dtype=sdt, strategy=strategy
                        )
                        bv = BatchedMapper(
                            mapper["v"], layout, tg, dtype=sdt, strategy=strategy
                        )

                        # correctness before speed: a broken packing must not
                        # be timed and reported as passing
                        Xk_chk = flat_features(to_content_keys(sp)).to(sdt)
                        rel = fidelity(bk.apply(Xk_chk), ref_k, tg)
                        del Xk_chk

                        # the whole per-prefix path, starting from the cache
                        def full():
                            xk = flat_features(to_content_keys(sp)).to(sdt)
                            xv = flat_features([v.float() for v in sp.values]).to(sdt)
                            keys = [
                                rerot(x.float(), pos, tg.rope_theta).to(dtype)
                                for x in bk.apply(xk)
                            ]
                            vals = [x.to(dtype) for x in bv.apply(xv)]
                            return keys, vals

                        t_full = time_callable(full, reps=args.reps, device=dev)

                        # disjoint stages
                        t_prep = time_callable(
                            lambda: (
                                flat_features(to_content_keys(sp)).to(sdt),
                                flat_features([v.float() for v in sp.values]).to(sdt),
                            ),
                            reps=args.reps,
                            device=dev,
                        )
                        xk_p = flat_features(to_content_keys(sp)).to(sdt)
                        xv_p = flat_features([v.float() for v in sp.values]).to(sdt)
                        t_affine = time_callable(
                            lambda: (bk.apply(xk_p), bv.apply(xv_p)),
                            reps=args.reps,
                            device=dev,
                        )
                        ck = bk.apply(xk_p)
                        t_rerope = time_callable(
                            lambda: [
                                rerot(x.float(), pos, tg.rope_theta).to(dtype)
                                for x in ck
                            ],
                            reps=args.reps,
                            device=dev,
                        )

                        # cold load of the packed artifact
                        blob = (
                            [g["W"] for g in bk.groups]
                            if strategy == "grouped"
                            else [bk.W]
                        )
                        blob += (
                            [g["W"] for g in bv.groups]
                            if strategy == "grouped"
                            else [bv.W]
                        )
                        host_w = [t.detach().to("cpu").pin_memory() for t in blob]
                        t_cold = time_callable(
                            lambda: [t.to(dev, non_blocking=False) for t in host_w],
                            reps=max(10, args.reps // 5),
                            device=dev,
                        )
                        del host_w, blob

                        bytes_ = bk.stats.weight_bytes + bv.stats.weight_bytes
                        tgt_cache = (
                            tg.n_layers * tg.n_kv_heads * tg.head_dim * ctx * 2 * 2
                        )
                        saved_ms = base_t["p50_ms"] - t_full["p50_ms"]
                        row = {
                            "ctx": ctx,
                            "strategy": strategy,
                            "serving_dtype": sd,
                            "tf32": tf32,
                            "fidelity_rel_err": rel,
                            "fidelity_ok": rel <= FIDELITY_TOL,
                            "n_blocks": bk.stats.n_blocks,
                            "n_groups": bk.stats.n_groups,
                            "n_matmuls_per_kind": bk.stats.n_matmuls,
                            "d_selected_max": bk.stats.d_selected_max,
                            "mapper_bytes": bytes_,
                            "mapper_in_target_caches": bytes_ / tgt_cache,
                            "baseline_attn": args.baseline_attn,
                            "target_prefill_p50_ms": base_t["p50_ms"],
                            "target_prefill_p95_ms": base_t["p95_ms"],
                            "source_prefill_p50_ms": src_t["p50_ms"],
                            "transfer_p50_ms": xfer["p50_ms"],
                            "map_p50_ms": t_full["p50_ms"],
                            "map_p95_ms": t_full["p95_ms"],
                            "component_prep_p50_ms": t_prep["p50_ms"],
                            "component_affine_p50_ms": t_affine["p50_ms"],
                            "component_rerope_p50_ms": t_rerope["p50_ms"],
                            "component_sum_p50_ms": t_prep["p50_ms"]
                            + t_affine["p50_ms"]
                            + t_rerope["p50_ms"],
                            "cold_load_p50_ms": t_cold["p50_ms"],
                            "cold_load_break_even_prefixes": (
                                t_cold["p50_ms"] / saved_ms
                                if saved_ms > 0
                                else float("inf")
                            ),
                            # the gate names one target prefill, so the
                            # denominator is the baseline's median
                            "regime_resident_p50": t_full["p50_ms"] / base_t["p50_ms"],
                            "regime_resident_p95": t_full["p95_ms"] / base_t["p50_ms"],
                            "regime_transferred_p50": (
                                t_full["p50_ms"] + xfer["p50_ms"]
                            )
                            / base_t["p50_ms"],
                            "regime_recreated_p50": (t_full["p50_ms"] + src_t["p50_ms"])
                            / base_t["p50_ms"],
                            "peak_mem_gib": (
                                torch.cuda.max_memory_allocated() / 2**30
                                if dev == "cuda"
                                else 0.0
                            ),
                        }
                        results.append(row)
                        log(
                            f"  tf32={tf32:3s} {strategy:8s}/{sd:9s} "
                            f"mm={row['n_matmuls_per_kind']:3d} map p50 {row['map_p50_ms']:7.2f} "
                            f"resident p50 {row['regime_resident_p50']:.3f} "
                            f"p95 {row['regime_resident_p95']:.3f} "
                            f"relerr {rel:.2e}{'' if row['fidelity_ok'] else ' UNFAITHFUL'} "
                            f"{bytes_/2**20:6.1f} MiB"
                        )
                        del bk, bv, xk_p, xv_p, ck
                        if dev == "cuda":
                            torch.cuda.empty_cache()
            del sp, mapper, ref_k, host
            if dev == "cuda":
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:  # noqa: PERF203
            skipped.append(
                {"ctx": ctx, "reason": "cuda out of memory", "detail": str(e)[:200]}
            )
            log(f"  ctx {ctx} skipped: out of memory")
            if dev == "cuda":
                torch.cuda.empty_cache()
        persist()

    # ---- gates, per context, over faithful arms only --------------------
    by_ctx = {}
    for ctx in sorted({r["ctx"] for r in results}):
        rows = [r for r in results if r["ctx"] == ctx and r["fidelity_ok"]]
        if not rows:
            by_ctx[str(ctx)] = {"eligible_arms": 0}
            continue
        best_p95 = min(rows, key=lambda r: r["regime_resident_p95"])
        best_p50 = min(rows, key=lambda r: r["regime_resident_p50"])
        by_ctx[str(ctx)] = {
            "eligible_arms": len(rows),
            "hard_gate_map_p95_under_one_prefill": best_p95["regime_resident_p95"]
            < 1.0,
            "promotion_p50_at_or_under_0.75": best_p50["regime_resident_p50"] <= 0.75,
            "best_p95_arm": best_p95,
            "best_p50_arm": best_p50,
        }
    passing = [
        c for c, v in by_ctx.items() if v.get("hard_gate_map_p95_under_one_prefill")
    ]
    verdict = {
        "by_ctx": by_ctx,
        "contexts_measured": sorted(by_ctx),
        "contexts_passing_hard_gate": passing,
        "hard_gate_at_every_measured_context": len(passing) == len(by_ctx)
        and bool(by_ctx),
        "unfaithful_arms": [
            {
                k: r[k]
                for k in (
                    "ctx",
                    "strategy",
                    "serving_dtype",
                    "tf32",
                    "fidelity_rel_err",
                )
            }
            for r in results
            if not r["fidelity_ok"]
        ],
    }
    log("")
    for c, v in by_ctx.items():
        if v.get("eligible_arms"):
            b = v["best_p95_arm"]
            log(
                f"ctx {c}: hard gate {v['hard_gate_map_p95_under_one_prefill']}  "
                f"promotion {v['promotion_p50_at_or_under_0.75']}  "
                f"best {b['strategy']}/{b['serving_dtype']}/tf32={b['tf32']} "
                f"p95 {b['regime_resident_p95']:.3f} p50 {b['regime_resident_p50']:.3f}"
            )
    log(
        f"hard gate at EVERY measured context: {verdict['hard_gate_at_every_measured_context']}"
    )
    if verdict["unfaithful_arms"]:
        log(f"arms excluded for fidelity: {len(verdict['unfaithful_arms'])}")
    persist(verdict)
    log(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
