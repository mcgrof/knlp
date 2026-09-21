# SPDX-License-Identifier: GPL-2.0
"""H0: does one frozen mapper behave the same however it is packed and stored?

G1 established that a batched apply is fast enough. It did not establish that
the thing being applied is a deployable artifact, for two reasons this stage
exists to close.

Every earlier run refitted the map at the context it was about to measure, so
the context curve belongs to a family of mappers rather than to any one of
them. Here a single artifact is fitted once, at one length, on pinned
documents, frozen with its hashes, and then applied unchanged at every length.
If its behaviour falls away from a context-refitted reference, the "context
generalisation" reported earlier was a property of refitting.

And fidelity was only ever measured as tensor agreement. Two packings agreeing
to a part in ten million says nothing about whether the frozen target *behaves*
the same after the perturbation, which is the only thing that matters. Every
arm here is scored on the target's own continuation: divergence, negative
log-likelihood, and how often the greedy token changes.

A lower-precision arm advances only on the pre-registered rule -- at most one
percent more excess divergence than strict float32 -- and not on latency.

The last measurement is the one a serving path would actually make. Translation
does not produce the target's final hidden state or its next-token logits, so
comparing it against a bare prefill quietly credits it for work it never did.
The output-equivalent arm translates the first L-1 positions, installs them,
and runs the final prompt token through the whole target including the
vocabulary projection, against a native prefill of all L positions producing
the same next-token logits. The cast into the cache's real dtype and layout is
charged to the translated side.

Env: HIP_VISIBLE_DEVICES=0 or CUDA_VISIBLE_DEVICES, TOKENIZERS_PARALLELISM=false
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
from research.jacobian_kv.evaluate import (  # noqa: E402
    argmax_flip_rate,
    delta_nll,
    teacher_forced_kl,
)
from research.kv_translate import freeze  # noqa: E402
from research.kv_translate.apply import BatchedMapper, time_callable  # noqa: E402
from research.kv_translate.fit import (  # noqa: E402
    Accumulator,
    SourceLayout,
    fit_block,
    select_layers,
)
from research.kv_translate.pairs import describe, prefill, to_content_keys  # noqa: E402
from research.kv_translate.run_a0 import (  # noqa: E402
    continuation_logits,
    make_cache,
    wikitext_chunks,
)
from research.kv_translate.run_a1 import (
    Mapper,
    flat_features,
    target_blocks,
)  # noqa: E402

KL_TOLERANCE = 0.01  # a lower-precision arm may add at most 1% excess divergence
RETENTION = 0.95  # frozen must keep this much of a refitted map's gain


def fit_mapper(models, geom, layout, calib, ctx, k, ridge, dev):
    """The ordinary fit, used both for the frozen artifact and the reference."""
    sg, tg = geom["source"], geom["target"]
    n_targets = tg.n_layers * tg.n_kv_heads
    acc = {
        kd: Accumulator(layout, n_targets, tg.head_dim, device=dev) for kd in ("k", "v")
    }
    for _, ids in calib:
        ids = ids[:ctx].unsqueeze(0).to(dev)
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
    scale = {kd: float(acc[kd].xtx.diagonal().mean()) for kd in ("k", "v")}
    out = {
        kd: Mapper(
            {
                (li, h): fit_block(
                    acc[kd],
                    target=li * tg.n_kv_heads + h,
                    target_layer=li,
                    head=h,
                    kind=kd,
                    layers=select_layers(
                        acc[kd], li * tg.n_kv_heads + h, h, ridge * scale[kd], k, False
                    ),
                    ridge=ridge * scale[kd],
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
    return out


@torch.no_grad()
def translate_cache(bk, bv, sp, tg, pos, cache_dtype):
    """Source cache -> target cache, finishing in the cache's real dtype."""
    xk = flat_features(to_content_keys(sp)).to(bk.dtype)
    xv = flat_features([v.float() for v in sp.values]).to(bv.dtype)
    keys = [rerot(x.float(), pos, tg.rope_theta).to(cache_dtype) for x in bk.apply(xk)]
    vals = [x.to(cache_dtype) for x in bv.apply(xv)]
    return keys, vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--fit-context", type=int, default=2048)
    ap.add_argument("--contexts", default="512,2048,4096")
    ap.add_argument("--calib", type=int, default=48)
    ap.add_argument("--eval", type=int, default=12)
    ap.add_argument("--cont-len", type=int, default=64)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--ridge", type=float, default=1e-5)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--serving-dtypes", default="float32,bfloat16")
    ap.add_argument("--strategies", default="dense,grouped")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default="h0")
    args = ap.parse_args()

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dtype = getattr(torch, args.dtype)
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False

    tok = AutoTokenizer.from_pretrained(args.target)
    models, geom, ident = {}, {}, {}
    for role, mid in (("source", args.source), ("target", args.target)):
        m = AutoModelForCausalLM.from_pretrained(
            mid, dtype=cache_dtype, attn_implementation="eager"
        ).to(dev)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models[role], geom[role] = m, describe(m, mid)
        ident[role] = freeze.model_identity(mid, m, AutoTokenizer.from_pretrained(mid))
    sg, tg = geom["source"], geom["target"]
    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)

    contexts = [int(c) for c in args.contexts.split(",")]
    longest = max(contexts + [args.fit_context])
    pool = wikitext_chunks(
        tok, args.calib + args.eval, longest, args.cont_len, args.seed
    )
    calib, held = pool[: args.calib], pool[args.calib :]
    log(
        f"{args.source} -> {args.target}; fit at {args.fit_context}, "
        f"apply at {contexts}; {len(calib)} calibration / {len(held)} held-out documents"
    )

    # ---- fit once, freeze, and do not look at a held-out result first ----
    frozen = fit_mapper(
        models, geom, layout, calib, args.fit_context, args.k, args.ridge, dev
    )
    out_dir = art.ensure_run_dir(args.out_dir, args.run_id)
    art_path = os.path.join(out_dir, "mapper.pt")
    manifest = freeze.save(
        art_path,
        frozen["k"],
        frozen["v"],
        source=ident["source"],
        target=ident["target"],
        fit_context=args.fit_context,
        calib_doc_ids=[c for c, _ in calib],
        dev_doc_ids=[],
        eval_doc_ids=[c for c, _ in held],
        config={
            "k": args.k,
            "ridge": args.ridge,
            "seed": args.seed,
            "calib": args.calib,
            "support": "full fan-in",
        },
    )
    log(
        f"froze {manifest['n_blocks']} blocks, joint hash "
        f"{manifest['joint_weight_sha256'][:16]}, {manifest['serialized_bytes']/2**20:.1f} MiB"
    )
    # reload through the hash check so everything below uses the artifact
    fk, fv, manifest = freeze.load(art_path, layout, tg, device=dev)

    rows, first_token = [], []
    for ctx in contexts:
        log(f"=== context {ctx}")
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        pos = torch.arange(ctx, device=dev)
        # the reference this artifact must not fall behind: a map refitted here
        refit = fit_mapper(models, geom, layout, calib, ctx, args.k, args.ridge, dev)

        packs = {}
        for strategy in args.strategies.split(","):
            for sd in args.serving_dtypes.split(","):
                sdt = getattr(torch, sd)
                packs[(strategy, sd)] = (
                    BatchedMapper(fk, layout, tg, dtype=sdt, strategy=strategy),
                    BatchedMapper(fv, layout, tg, dtype=sdt, strategy=strategy),
                )
        refit_pack = (
            BatchedMapper(
                refit["k"], layout, tg, dtype=torch.float32, strategy="dense"
            ),
            BatchedMapper(
                refit["v"], layout, tg, dtype=torch.float32, strategy="dense"
            ),
        )

        per_arm = {
            name: []
            for name in list(packs)
            + [("refit", "float32"), ("no_cache", "-"), ("fp64_ref", "-")]
        }
        for cid, ids in held:
            ids = ids.to(dev)
            prompt = ids[:ctx].unsqueeze(0)
            cont = ids[ctx : ctx + args.cont_len].unsqueeze(0)
            sp = prefill(models["source"], prompt, sg)
            tp = prefill(models["target"], prompt, tg)
            ref = continuation_logits(
                models["target"], tp.keys_post, tp.values, cont, ctx
            ).float()

            def score(keys, vals):
                lg = continuation_logits(
                    models["target"], keys, vals, cont, ctx
                ).float()
                r = {
                    "kl": float(teacher_forced_kl(ref, lg).mean()),
                    "dnll": float(delta_nll(ref, lg, cont).mean()),
                    "flip": argmax_flip_rate(ref, lg),
                }
                del lg
                return r

            lg0 = continuation_logits(models["target"], None, None, cont, ctx).float()
            per_arm[("no_cache", "-")].append(
                {
                    "kl": float(teacher_forced_kl(ref, lg0).mean()),
                    "dnll": float(delta_nll(ref, lg0, cont).mean()),
                    "flip": argmax_flip_rate(ref, lg0),
                }
            )
            del lg0

            # the float64 block-at-a-time reference the packings must match
            xk64 = flat_features(to_content_keys(sp)).double()
            xv64 = flat_features([v.float() for v in sp.values]).double()
            k64 = [
                rerot(fk.apply(xk64)[i].float(), pos, tg.rope_theta).to(cache_dtype)
                for i in range(tg.n_layers)
            ]
            v64 = [x.to(cache_dtype) for x in fv.apply(xv64)]
            per_arm[("fp64_ref", "-")].append(score(k64, v64))
            del xk64, xv64, k64, v64

            for name, (bk, bv) in packs.items():
                keys, vals = translate_cache(bk, bv, sp, tg, pos, cache_dtype)
                per_arm[name].append(score(keys, vals))
                del keys, vals
            keys, vals = translate_cache(
                refit_pack[0], refit_pack[1], sp, tg, pos, cache_dtype
            )
            per_arm[("refit", "float32")].append(score(keys, vals))
            del keys, vals, sp, tp, ref
            if dev == "cuda":
                torch.cuda.empty_cache()

        base = {k: float(np.mean([r["kl"] for r in v])) for k, v in per_arm.items()}
        fp32_key = (
            ("dense", "float32") if ("dense", "float32") in base else list(packs)[0]
        )
        floor = base[("no_cache", "-")]
        refit_gain = floor - base[("refit", "float32")]
        log(
            f"  no cache {floor:.4f} | fp64 ref {base[('fp64_ref','-')]:.4f} "
            f"| refit {base[('refit','float32')]:.4f}"
        )

        # timing on the frozen artifact, in the cache's real dtype
        probe = held[0][1][:ctx].unsqueeze(0).to(dev)
        sp = prefill(models["source"], probe, sg)
        for name, (bk, bv) in packs.items():
            strategy, sd = name
            t = time_callable(
                lambda: translate_cache(bk, bv, sp, tg, pos, cache_dtype),
                reps=args.reps,
                device=dev,
            )
            arm_kl = base[name]
            excess = arm_kl - base[fp32_key]
            rel_excess = excess / max(base[fp32_key], 1e-12)
            gain = floor - arm_kl
            rows.append(
                {
                    "ctx": ctx,
                    "strategy": strategy,
                    "serving_dtype": sd,
                    "kl": arm_kl,
                    "dnll": float(np.mean([r["dnll"] for r in per_arm[name]])),
                    "flip": float(np.mean([r["flip"] for r in per_arm[name]])),
                    "kl_fp64_ref": base[("fp64_ref", "-")],
                    "kl_refit_fp32": base[("refit", "float32")],
                    "kl_no_cache": floor,
                    "excess_vs_fp32": excess,
                    "rel_excess_vs_fp32": rel_excess,
                    "dtype_ok": rel_excess <= KL_TOLERANCE,
                    "gain_over_no_cache": gain,
                    "retention_vs_refit": gain / max(refit_gain, 1e-12),
                    "retention_ok": (gain / max(refit_gain, 1e-12)) >= RETENTION,
                    "map_p50_ms": t["p50_ms"],
                    "map_p95_ms": t["p95_ms"],
                    "resident_bytes": bk.stats.weight_bytes + bv.stats.weight_bytes,
                    "serialized_bytes": manifest["serialized_bytes"],
                    "n_matmuls_per_kind": bk.stats.n_matmuls,
                    "peak_mem_gib": (
                        torch.cuda.max_memory_allocated() / 2**30
                        if dev == "cuda"
                        else 0.0
                    ),
                }
            )
            r = rows[-1]
            log(
                f"  {strategy:8s}/{sd:9s} KL {arm_kl:.4f} flip {r['flip']:.3f} "
                f"excess {rel_excess:+.2%}{'' if r['dtype_ok'] else ' DTYPE-FAIL'} "
                f"retention {r['retention_vs_refit']:.3f}"
                f"{'' if r['retention_ok'] else ' RETENTION-FAIL'} "
                f"map {t['p50_ms']:.1f} ms"
            )

        # ---- the output-equivalent first-token path --------------------
        bk, bv = (
            packs[("dense", "bfloat16")]
            if ("dense", "bfloat16") in packs
            else packs[fp32_key]
        )
        full_ids = probe
        head_ids = full_ids[:, : ctx - 1]
        last_tok = full_ids[:, ctx - 1 :]
        sp_head = prefill(models["source"], head_ids, sg)
        pos_head = torch.arange(ctx - 1, device=dev)

        @torch.no_grad()
        def native_first_token():
            # Only the last position's logits are wanted, and only the last
            # position's are computed. Letting the head run over all L
            # positions charges the native arm work no serving stack does on
            # a prefill, which inflates the denominator of every ratio here.
            return models["target"](input_ids=full_ids, logits_to_keep=1).logits[:, -1]

        @torch.no_grad()
        def translated_first_token():
            xk = flat_features(to_content_keys(sp_head)).to(bk.dtype)
            xv = flat_features([v.float() for v in sp_head.values]).to(bv.dtype)
            keys = [
                rerot(x.float(), pos_head, tg.rope_theta).to(cache_dtype)
                for x in bk.apply(xk)
            ]
            vals = [x.to(cache_dtype) for x in bv.apply(xv)]
            attn = torch.ones(1, ctx, dtype=torch.long, device=dev)
            p = torch.arange(ctx - 1, ctx, device=dev).unsqueeze(0)
            return models["target"](
                input_ids=last_tok,
                attention_mask=attn,
                position_ids=p,
                past_key_values=make_cache(keys, vals),
                use_cache=True,
            ).logits[:, -1]

        nat = native_first_token().float()
        tra = translated_first_token().float()
        lp, lq = torch.log_softmax(nat, -1), torch.log_softmax(tra, -1)
        t_nat = time_callable(native_first_token, reps=args.reps, device=dev)
        t_tra = time_callable(translated_first_token, reps=args.reps, device=dev)
        first_token.append(
            {
                "ctx": ctx,
                "next_token_kl": float((lp.exp() * (lp - lq)).sum()),
                "argmax_agrees": bool(nat.argmax(-1).item() == tra.argmax(-1).item()),
                "native_p50_ms": t_nat["p50_ms"],
                "native_p95_ms": t_nat["p95_ms"],
                "translated_p50_ms": t_tra["p50_ms"],
                "translated_p95_ms": t_tra["p95_ms"],
                "ratio_p50": t_tra["p50_ms"] / t_nat["p50_ms"],
                "ratio_p95": t_tra["p95_ms"] / t_nat["p50_ms"],
                "gate_p95_under_native": t_tra["p95_ms"] / t_nat["p50_ms"] < 1.0,
                "gate_p50_at_or_under_075": t_tra["p50_ms"] / t_nat["p50_ms"] <= 0.75,
            }
        )
        f = first_token[-1]
        log(
            f"  first-token path: native {f['native_p50_ms']:.1f} ms, translated "
            f"{f['translated_p50_ms']:.1f} ms, p50 {f['ratio_p50']:.3f} p95 {f['ratio_p95']:.3f}, "
            f"next-token KL {f['next_token_kl']:.4f}, argmax agrees {f['argmax_agrees']}"
        )

        del packs, refit_pack, refit, sp, sp_head
        if dev == "cuda":
            torch.cuda.empty_cache()

    verdict = {
        "all_arms_retain_refit": all(r["retention_ok"] for r in rows),
        "low_precision_arms_within_kl_tolerance": all(
            r["dtype_ok"] for r in rows if r["serving_dtype"] != "float32"
        ),
        "first_token_gate_every_ctx": all(
            f["gate_p95_under_native"] for f in first_token
        ),
        "first_token_promotion_every_ctx": all(
            f["gate_p50_at_or_under_075"] for f in first_token
        ),
        "failures": [
            {
                k: r[k]
                for k in (
                    "ctx",
                    "strategy",
                    "serving_dtype",
                    "rel_excess_vs_fp32",
                    "retention_vs_refit",
                )
            }
            for r in rows
            if not (r["dtype_ok"] and r["retention_ok"])
        ],
    }
    log("")
    log(f"H0: {json.dumps(verdict, default=str)[:500]}")

    art.write_json(
        os.path.join(out_dir, "h0.json"),
        {
            "manifest": manifest,
            "rows": rows,
            "first_token": first_token,
            "verdict": verdict,
            "thresholds": {"kl_tolerance": KL_TOLERANCE, "retention": RETENTION},
            "config": vars(args),
            "wall_s": time.time() - t0,
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "versions": art.versions(),
            "code_commit": art.git_state()[0],
        },
    )
    log(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
