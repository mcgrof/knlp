#!/usr/bin/env python3
"""Time the paths a translated cache can be served on, and keep the samples.

The previous timing supported its verdict and a review still could not check
it, for four reasons this fixes.

It reported percentiles and discarded the samples, so a reader could not see
whether a median sat in a tight cluster or between two modes. Every sample is
archived here.

It timed one availability condition -- the cold path, where nothing exists and
the source model must read the document -- and the plan asked for two. The one
it skipped is the condition an ecosystem actually deploys: a source cache that
already exists, because something else produced it. Both are measured, and so
is the receiver-resident case where only the question's repair remains.

It measured on the device and called the result a latency. A device timer omits
the host work on either side, and the gap between the two is what a request
sees. Both are recorded.

And it compared a mapped path answering after installation against a native
path that prefilled document and question together, which is a difference in
what is computed and not only in how. The comparison here holds the required
output fixed: every arm produces the first answer token and nothing else.

None of this is a production latency distribution. It is six documents on one
card with warm caches, and the number of repetitions is small enough that the
tails are indicative.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from research.cartridges_cas.scripts.cas_kv_rope import rerot  # noqa: E402
from research.kv_translate import freeze  # noqa: E402
from research.kv_translate.fit import SourceLayout  # noqa: E402
from research.kv_translate.pairs import describe, prefill, to_content_keys  # noqa: E402
from research.kv_translate.recovery import BLOCK, mask_from_blocks  # noqa: E402
from research.kv_translate.recovery_runner import make_cache, select  # noqa: E402
from research.kv_translate.replay import replay_selected  # noqa: E402
from research.kv_translate.run_a1 import flat_features  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class Timer:
    """Device and host time for the same call, with every sample kept."""

    def __init__(self, device):
        self.cuda = device == "cuda"
        self.ev = (
            [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            if self.cuda
            else None
        )

    def run(self, fn, reps, warmup=2):
        for _ in range(warmup):
            fn()
        if self.cuda:
            torch.cuda.synchronize()
        dev_ms, host_ms = [], []
        for _ in range(reps):
            h0 = time.perf_counter()
            if self.cuda:
                self.ev[0].record()
            fn()
            if self.cuda:
                self.ev[1].record()
                torch.cuda.synchronize()
                dev_ms.append(self.ev[0].elapsed_time(self.ev[1]))
            host_ms.append((time.perf_counter() - h0) * 1e3)
        return dev_ms, host_ms


def summarise(dev_ms, host_ms):
    def pct(xs, p):
        return float(np.percentile(xs, p)) if xs else None

    out = {
        "n": len(host_ms),
        "host_p50_ms": pct(host_ms, 50),
        "host_p95_ms": pct(host_ms, 95),
        "host_samples_ms": [round(x, 4) for x in host_ms],
    }
    if dev_ms:
        out.update(
            device_p50_ms=pct(dev_ms, 50),
            device_p95_ms=pct(dev_ms, 95),
            device_samples_ms=[round(x, 4) for x in dev_ms],
            host_minus_device_p50_ms=round(pct(host_ms, 50) - pct(dev_ms, 50), 4),
        )
    if len(host_ms) > 1:
        out["host_stdev_ms"] = round(statistics.stdev(host_ms), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--target-revision", required=True)
    ap.add_argument("--arm", required=True, help="name=path.pt")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--docs", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument(
        "--budgets",
        default="0,1,8,25,64,128,256",
        help="replay budgets in blocks, for the component profile",
    )
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--solve-dtype", default="float32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype, sdtype = getattr(torch, args.dtype), getattr(torch, args.solve_dtype)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    name, path = args.arm.split("=", 1)
    tok = AutoTokenizer.from_pretrained(args.target, revision=args.target_revision)
    models, geom = {}, {}
    for role, mid, rev in (
        ("source", args.source, args.source_revision),
        ("target", args.target, args.target_revision),
    ):
        m = AutoModelForCausalLM.from_pretrained(
            mid, revision=rev, dtype=dtype, attn_implementation="sdpa"
        ).to(dev)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models[role], geom[role] = m, describe(m, mid)
    sg, tg = geom["source"], geom["target"]
    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    mk, mv, man = freeze.load(path, layout, tg, device=dev)
    mk, mv = mk.cast(sdtype), mv.cast(sdtype)
    pos = torch.arange(args.ctx, device=dev)
    budgets = [int(x) for x in args.budgets.split(",") if x != ""]

    fixture = json.load(open(args.manifest))
    docs = [d for d in fixture["documents"] if d["in_screen"]][: args.docs]
    T = Timer(dev)
    paths, components = {}, {}

    def add(store, key, dev_ms, host_ms):
        store.setdefault(key, {"device": [], "host": []})
        store[key]["device"].extend(dev_ms)
        store[key]["host"].extend(host_ms)

    for d in docs:
        ids = torch.tensor(d["prompt_ids"], dtype=torch.long, device=dev).unsqueeze(0)
        qids = tok(d["retrieval_query"], return_tensors="pt").input_ids.to(dev)
        n = qids.shape[1]
        log(f"{d['doc'][:44]}")

        def translate(sp):
            xk = flat_features(to_content_keys(sp)).to(sdtype)
            xv = flat_features([v.float() for v in sp.values]).to(sdtype)
            keys = [
                rerot(b[0].float(), pos, tg.rope_theta).unsqueeze(0).to(dtype)
                for b in mk.apply(xk)
            ]
            return keys, [v.to(dtype) for v in mv.apply(xv)]

        # ---- the whole path, from raw tokens, nothing precomputed ---------
        @torch.no_grad()
        def cold_native():
            return models["target"](
                input_ids=torch.cat([ids, qids], dim=1), logits_to_keep=1
            ).logits

        @torch.no_grad()
        def cold_mapped(blocks):
            keys, values = translate(prefill(models["source"], ids, sg))
            if blocks:
                sel, _ = select(
                    "mapped_attn",
                    blocks,
                    model=models["target"],
                    mapped=(keys, values),
                    native=None,
                    query_ids=qids,
                    n_positions=args.ctx,
                )
                m = mask_from_blocks(args.ctx, sel).to(dev)
                keys, values, _ = replay_selected(
                    models["target"], ids, m, keys, values, in_place=True
                )
            return models["target"](
                input_ids=qids,
                attention_mask=torch.ones(
                    1, args.ctx + n, dtype=torch.long, device=dev
                ),
                position_ids=torch.arange(args.ctx, args.ctx + n, device=dev).unsqueeze(
                    0
                ),
                past_key_values=make_cache(keys, values),
                use_cache=True,
                logits_to_keep=1,
            ).logits

        add(paths, "cold_native", *T.run(cold_native, args.reps))
        add(paths, "cold_mapped_unrepaired", *T.run(lambda: cold_mapped(0), args.reps))
        add(paths, "cold_mapped_repair64", *T.run(lambda: cold_mapped(64), args.reps))
        add(paths, "cold_mapped_repair25", *T.run(lambda: cold_mapped(25), args.reps))

        # ---- the source cache already exists, because something made it ---
        # This is the condition the plan asked for and the last run skipped.
        # It is the ecosystem case: a small model read the document earlier,
        # for its own reasons, and its cache is resident.
        sp_resident = prefill(models["source"], ids, sg)

        @torch.no_grad()
        def resident_mapped(blocks):
            keys, values = translate(sp_resident)
            if blocks:
                sel, _ = select(
                    "mapped_attn",
                    blocks,
                    model=models["target"],
                    mapped=(keys, values),
                    native=None,
                    query_ids=qids,
                    n_positions=args.ctx,
                )
                m = mask_from_blocks(args.ctx, sel).to(dev)
                keys, values, _ = replay_selected(
                    models["target"], ids, m, keys, values, in_place=True
                )
            return models["target"](
                input_ids=qids,
                attention_mask=torch.ones(
                    1, args.ctx + n, dtype=torch.long, device=dev
                ),
                position_ids=torch.arange(args.ctx, args.ctx + n, device=dev).unsqueeze(
                    0
                ),
                past_key_values=make_cache(keys, values),
                use_cache=True,
                logits_to_keep=1,
            ).logits

        add(
            paths,
            "resident_source_unrepaired",
            *T.run(lambda: resident_mapped(0), args.reps),
        )
        add(
            paths,
            "resident_source_repair64",
            *T.run(lambda: resident_mapped(64), args.reps),
        )
        add(
            paths,
            "resident_source_repair25",
            *T.run(lambda: resident_mapped(25), args.reps),
        )

        # ---- the receiver's own cache is resident: what native reuse costs -
        # The baseline for a document asked about more than once. Charging a
        # native prefill every time while treating a translated prefix as free
        # is the comparison a serving system would never make.
        tp = prefill(models["target"], ids, tg)
        nk = [k.to(dtype) for k in tp.keys_post]
        nv = [v.to(dtype) for v in tp.values]

        @torch.no_grad()
        def resident_native():
            return models["target"](
                input_ids=qids,
                attention_mask=torch.ones(
                    1, args.ctx + n, dtype=torch.long, device=dev
                ),
                position_ids=torch.arange(args.ctx, args.ctx + n, device=dev).unsqueeze(
                    0
                ),
                past_key_values=make_cache(nk, nv),
                use_cache=True,
                logits_to_keep=1,
            ).logits

        add(paths, "resident_native_prefix", *T.run(resident_native, args.reps))

        # ---- components, measured alone ----------------------------------
        add(
            components,
            "source_prefill",
            *T.run(lambda: prefill(models["source"], ids, sg), args.reps),
        )
        add(
            components,
            "map_apply_and_rerotate",
            *T.run(lambda: translate(sp_resident), args.reps),
        )
        keys0, values0 = translate(sp_resident)
        add(
            components,
            "selector_rank_on_mapped",
            *T.run(
                lambda: select(
                    "mapped_attn",
                    64,
                    model=models["target"],
                    mapped=(keys0, values0),
                    native=None,
                    query_ids=qids,
                    n_positions=args.ctx,
                ),
                args.reps,
            ),
        )
        add(
            components,
            "target_prefill_document_only",
            *T.run(lambda: prefill(models["target"], ids, tg), args.reps),
        )
        add(
            components,
            "question_over_resident_cache",
            *T.run(resident_native, args.reps),
        )
        for b in budgets:
            if b == 0:
                continue
            mb = mask_from_blocks(
                args.ctx, list(range(0, 256, max(256 // b, 1)))[:b]
            ).to(dev)
            add(
                components,
                f"replay_{b:03d}_blocks",
                *T.run(
                    lambda mm=mb: replay_selected(
                        models["target"], ids, mm, keys0, values0, in_place=False
                    ),
                    args.reps,
                ),
            )

    out = {
        "contract": "path_profile_v1",
        "boundary": (
            "raw document tokens and the question in hand, to the first answer "
            "token, with every arm producing exactly that and nothing more"
        ),
        "availability_conditions": {
            "cold": "nothing precomputed; the source model reads the document",
            "resident_source": (
                "the source cache already exists because something else made "
                "it; the map, the selection and the repair remain"
            ),
            "resident_native_prefix": (
                "the receiver's own cache is resident: what ordinary prefix "
                "reuse costs, and the baseline for a repeated document"
            ),
        },
        "caveat": (
            "warm caches, few repetitions, one card, six documents. Medians "
            "are indicative and tails are not a production distribution."
        ),
        "documents": [d["doc"] for d in docs],
        "reps_per_document": args.reps,
        "ctx": args.ctx,
        "block_tokens": BLOCK,
        "arm": {"name": name, "joint_weight_sha256": man["joint_weight_sha256"]},
        "models": {
            "source": {"id": args.source, "revision": args.source_revision},
            "target": {"id": args.target, "revision": args.target_revision},
        },
        "numeric_mode": {
            "model_dtype": args.dtype,
            "map_dtype": args.solve_dtype,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        },
        "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "torch": torch.__version__,
        "paths": {k: summarise(v["device"], v["host"]) for k, v in paths.items()},
        "components": {
            k: summarise(v["device"], v["host"]) for k, v in components.items()
        },
    }
    base = out["paths"].get("cold_native", {}).get("device_p50_ms")
    if base:
        for k, v in out["paths"].items():
            if v.get("device_p50_ms"):
                v["ratio_device_p50_over_cold_native"] = round(
                    v["device_p50_ms"] / base, 4
                )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, sort_keys=True, default=str)

    log("")
    log(f"{'path':32s} {'dev p50':>9s} {'host p50':>9s} {'/cold native':>13s}")
    for k in sorted(out["paths"]):
        v = out["paths"][k]
        r = v.get("ratio_device_p50_over_cold_native")
        log(
            f"{k:32s} {v.get('device_p50_ms', 0):9.1f} {v['host_p50_ms']:9.1f} "
            f"{(f'{r:.3f}' if r else '-'):>13s}"
        )
    log("")
    for k in sorted(out["components"]):
        v = out["components"][k]
        log(f"  {k:34s} {v.get('device_p50_ms', 0):8.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
