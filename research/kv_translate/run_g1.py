# SPDX-License-Identifier: GPL-2.0
"""G1: measure what applying a cache map actually costs, and make it cheap.

The earlier cost numbers were not wrong about the code; they were measured on
a path that applied the map one block at a time, in double precision, with the
column indices rebuilt on every call. That reports the cost of dispatching work
rather than of doing it, so the honest move is to fix the path and measure
again rather than to argue about the number.

What this measures, and how, matters as much as what it finds.

Timing uses device events around a warmed-up loop of a hundred repetitions and
reports the median and the 95th percentile, because a mean over a handful of
wall-clock samples around an asynchronous launch is not a latency.

The baseline is the target model's own prefill on the same GPU, in the same
dtype, through the same attention backend, and *without* the vocabulary
projection -- a prefill whose job is to fill a cache never computes logits over
every prompt position, and leaving that in adds a constant that flatters the
map by hiding behind it.

Three deployment regimes are reported separately and never averaged, because
they are different products:

  resident      the source cache is already on the GPU, so the map alone
                competes with a target prefill. This is the intended case.
  transferred   the source cache has to arrive from host memory first, so the
                copy counts against the map.
  recreated     the source model is run only to produce the cache, so a source
                prefill counts too. This one has never had a plausible win and
                is measured to keep it honest rather than because it is hoped
                for.

The component profile splits the map into gather, affine multiply, key
re-rotation and cache materialisation, so a cost that fails a gate can be
attributed to a stage rather than to the method.

Env: HF_HOME may point anywhere; set TOKENIZERS_PARALLELISM=false.
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


def provenance(model_id, model, tok) -> dict:
    """Identity a later reader can check rather than infer from a name."""
    rev = ""
    try:
        from huggingface_hub import snapshot_download

        rev = os.path.basename(
            snapshot_download(model_id, local_files_only=True).rstrip("/")
        )
    except Exception:  # noqa: BLE001
        rev = "unresolved"
    vocab = tok.get_vocab()
    tok_hash = hashlib.sha256(
        json.dumps(sorted(vocab.items()), separators=(",", ":")).encode()
    ).hexdigest()[:32]
    tmpl = getattr(tok, "chat_template", None) or ""
    return {
        "model_id": model_id,
        "revision": rev,
        "tokenizer_sha256": tok_hash,
        "tokenizer_len": len(vocab),
        "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest()[:32],
        "model_dtype": str(next(model.parameters()).dtype),
        "attn_implementation": getattr(model.config, "_attn_implementation", "unknown"),
    }


@torch.no_grad()
def target_prefill_fn(model, ids):
    """A prefill that fills a cache and nothing else: no vocabulary projection."""
    from transformers import DynamicCache

    base = model.model

    def run():
        base(input_ids=ids, past_key_values=DynamicCache(), use_cache=True)

    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--contexts", default="512,1024,2048")
    ap.add_argument("--calib", type=int, default=48)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--ridge", type=float, default=1e-5)
    ap.add_argument(
        "--dtype", default="bfloat16", help="model dtype, the deployment one"
    )
    ap.add_argument("--serving-dtypes", default="float32,bfloat16")
    ap.add_argument("--strategies", default="grouped,dense")
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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

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
    log(f"{args.source} -> {args.target}, models in {args.dtype}")

    results = []
    for ctx in [int(c) for c in args.contexts.split(",")]:
        log(f"=== context {ctx}")
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
                flat_features(to_content_keys(sp)), target_blocks(to_content_keys(tp))
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
        torch.cuda.empty_cache() if dev == "cuda" else None
        ngroups = len(selection_groups(mapper["k"], layout))
        log(f"  fitted; {n_targets} blocks fall into {ngroups} distinct selections")

        ids = probe[0][1][:ctx].unsqueeze(0).to(dev)
        sp = prefill(models["source"], ids, sg)
        Xk = flat_features(to_content_keys(sp))
        Xv = flat_features([v.float() for v in sp.values])
        pos = torch.arange(ctx, device=dev)

        tgt_fn = target_prefill_fn(models["target"], ids)
        base_t = time_callable(tgt_fn, reps=args.reps, device=dev)
        src_fn = target_prefill_fn(models["source"], ids)
        src_t = time_callable(src_fn, reps=args.reps, device=dev)
        log(
            f"  target prefill p50 {base_t['p50_ms']:.2f} ms (p95 {base_t['p95_ms']:.2f}) "
            f"| source prefill p50 {src_t['p50_ms']:.2f} ms"
        )

        # host-to-device copy of a source cache, for the transferred regime
        src_bytes = sum(v.numel() * v.element_size() for v in sp.values) * 2
        host = torch.empty(
            src_bytes // 2, dtype=torch.float16, device="cpu"
        ).pin_memory()
        xfer = time_callable(
            lambda: host.to(dev, non_blocking=False), reps=args.reps, device=dev
        )

        for strategy in args.strategies.split(","):
            for sd in args.serving_dtypes.split(","):
                sdt = getattr(torch, sd)
                bk = BatchedMapper(
                    mapper["k"], layout, tg, dtype=sdt, strategy=strategy
                )
                bv = BatchedMapper(
                    mapper["v"], layout, tg, dtype=sdt, strategy=strategy
                )
                Xk_s, Xv_s = Xk.to(sdt), Xv.to(sdt)

                def full():
                    ck = bk.apply(Xk_s)
                    keys = [rerot(x.float(), pos, tg.rope_theta).to(dtype) for x in ck]
                    vals = [x.to(dtype) for x in bv.apply(Xv_s)]
                    return keys, vals

                t_full = time_callable(full, reps=args.reps, device=dev)
                t_affine = time_callable(
                    lambda: (bk.apply(Xk_s), bv.apply(Xv_s)), reps=args.reps, device=dev
                )
                ck = bk.apply(Xk_s)
                t_rerope = time_callable(
                    lambda: [
                        rerot(x.float(), pos, tg.rope_theta).to(dtype) for x in ck
                    ],
                    reps=args.reps,
                    device=dev,
                )
                t_gather = time_callable(
                    lambda: (
                        [Xk_s[:, g["cols"]] for g in bk.groups]
                        if strategy == "grouped"
                        else (lambda: Xk_s)()
                    ),
                    reps=args.reps,
                    device=dev,
                )
                bytes_ = bk.stats.weight_bytes + bv.stats.weight_bytes
                tgt_cache = tg.n_layers * tg.n_kv_heads * tg.head_dim * ctx * 2 * 2
                row = {
                    "ctx": ctx,
                    "strategy": strategy,
                    "serving_dtype": sd,
                    "n_blocks": bk.stats.n_blocks,
                    "n_groups": bk.stats.n_groups,
                    "n_matmuls_per_kind": bk.stats.n_matmuls,
                    "mapper_bytes": bytes_,
                    "mapper_in_target_caches": bytes_ / tgt_cache,
                    "target_prefill_p50_ms": base_t["p50_ms"],
                    "target_prefill_p95_ms": base_t["p95_ms"],
                    "source_prefill_p50_ms": src_t["p50_ms"],
                    "transfer_p50_ms": xfer["p50_ms"],
                    "map_p50_ms": t_full["p50_ms"],
                    "map_p95_ms": t_full["p95_ms"],
                    "component_affine_p50_ms": t_affine["p50_ms"],
                    "component_rerope_p50_ms": t_rerope["p50_ms"],
                    "component_gather_p50_ms": t_gather["p50_ms"],
                    "regime_resident_p50": t_full["p50_ms"] / base_t["p50_ms"],
                    "regime_resident_p95": t_full["p95_ms"] / base_t["p95_ms"],
                    "regime_transferred_p50": (t_full["p50_ms"] + xfer["p50_ms"])
                    / base_t["p50_ms"],
                    "regime_recreated_p50": (t_full["p50_ms"] + src_t["p50_ms"])
                    / base_t["p50_ms"],
                }
                results.append(row)
                log(
                    f"  {strategy:8s}/{sd:9s} matmuls={row['n_matmuls_per_kind']:3d} "
                    f"map p50 {row['map_p50_ms']:7.2f} ms  resident {row['regime_resident_p50']:.3f} "
                    f"p95 {row['regime_resident_p95']:.3f}  bytes {bytes_/2**20:6.1f} MiB "
                    f"({row['mapper_in_target_caches']:.1f} caches)"
                )
                del bk, bv
                torch.cuda.empty_cache() if dev == "cuda" else None
        del sp, Xk, Xv, mapper
        torch.cuda.empty_cache() if dev == "cuda" else None

    best = min(results, key=lambda r: r["regime_resident_p95"])
    verdict = {
        "hard_gate_map_only_p95_below_one_prefill": best["regime_resident_p95"] < 1.0,
        "promotion_target_p50_at_or_below_0.75": best["regime_resident_p50"] <= 0.75,
        "best": best,
    }
    log("")
    log(f"G1: {json.dumps(verdict['best'], default=str)[:400]}")
    log(
        f"hard gate (p95 map-only < 1 prefill): {verdict['hard_gate_map_only_p95_below_one_prefill']}"
    )
    log(
        f"promotion target (p50 <= 0.75): {verdict['promotion_target_p50_at_or_below_0.75']}"
    )

    out = art.ensure_run_dir(args.out_dir, args.run_id)
    art.write_json(
        os.path.join(out, "g1.json"),
        {
            "provenance": prov,
            "source": sg.to_dict(),
            "target": tg.to_dict(),
            "numerics": {
                "fit_dtype": "float64",
                "model_dtype": args.dtype,
                "serving_dtypes": args.serving_dtypes.split(","),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "accumulation": "torch default for the storage dtype",
            },
            "config": {
                "k": args.k,
                "ridge": args.ridge,
                "calib": args.calib,
                "reps": args.reps,
                "seed": args.seed,
            },
            "results": results,
            "verdict": verdict,
            "wall_s": time.time() - t0,
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "versions": art.versions(),
            "code_commit": art.git_state()[0],
        },
    )
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
