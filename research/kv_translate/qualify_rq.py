#!/usr/bin/env python3
"""The replay qualification the previous one should have been.

Runs the checks in ``replay_checks`` against the real pair on the card, under
an envelope declared here and fixed before the run. The envelope's provenance
is part of the receipt: these limits were revised once, after an earlier check
failed a working implementation on a statistic that described its tail, and
that revision is recorded rather than smoothed over. They are not revised
again here, and a failure under them is a failure to localise, not a limit to
widen.

The result is agreement inside an envelope on a small number of documents. It
is evidence that the operation performed is the one intended. It is not proof.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from research.cartridges_cas.scripts.cas_kv_rope import rerot  # noqa: E402
from research.kv_translate import freeze  # noqa: E402
from research.kv_translate.fit import SourceLayout  # noqa: E402
from research.kv_translate.pairs import describe, prefill, to_content_keys  # noqa: E402
from research.kv_translate.recovery import BLOCK, mask_from_blocks  # noqa: E402
from research.kv_translate.recovery_runner import make_cache, select  # noqa: E402
from research.kv_translate.replay import replay_selected  # noqa: E402
from research.kv_translate.replay_checks import (  # noqa: E402
    check_initial_cache_independence,
    check_layer0_exact,
    check_native_repeat,
    check_ownership_modes,
    check_partial_mask_per_layer,
    error_quantiles,
    one_minus_cos,
    per_layer_profile,
    rel_frobenius,
    rel_max,
)

# Frozen before the run. The first four are the previous stage's revised
# numbers, carried unchanged so this stage is not a fresh choice of bar; the
# rest are exactness requirements that admit no tolerance at all.
ENVELOPE = {
    "relative_frobenius_max": 0.1,
    "one_minus_cosine_max": 1e-3,
    "continuation_KL_max": 1e-3,
    "next_token_disagreements_max": 0,
    "improvement_over_initial_cache_min": 20.0,
    "initial_cache_independence_max": 1e-6,
}
ENVELOPE_PROVENANCE = (
    "The first four limits are the previous stage's, revised there after an "
    "elementwise-maximum rule failed a working implementation, and carried "
    "here unchanged so that this stage does not re-choose them. The "
    "improvement ratio is a declared threshold and not a tolerance-free "
    "criterion: its numerator is the error of the cache being repaired, so its "
    "size reflects how poor the starting map is as much as how correct the "
    "replay is. Independence is exactness up to floating-point equality."
)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@torch.no_grad()
def continuation_logits(model, keys, values, qids, prompt_len, device):
    n = qids.shape[1]
    return (
        model(
            input_ids=qids,
            attention_mask=torch.ones(
                1, prompt_len + n, dtype=torch.long, device=device
            ),
            position_ids=torch.arange(
                prompt_len, prompt_len + n, device=device
            ).unsqueeze(0),
            past_key_values=make_cache(keys, values),
            use_cache=True,
            logits_to_keep=1,
        )
        .logits[0, -1]
        .float()
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--target-revision", required=True)
    ap.add_argument("--arm", required=True, help="name=path.pt")
    ap.add_argument("--expect-joint-sha256", default="")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--docs", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--queries", default="retrieval,cloze")
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
    joint = man["joint_weight_sha256"]
    if args.expect_joint_sha256 and joint != args.expect_joint_sha256:
        raise SystemExit(f"INVALID: arm hashes to {joint}")
    log(f"arm {name}: joint {joint[:16]}")

    fixture = json.load(open(args.manifest))
    docs = [d for d in fixture["documents"] if d["in_screen"]][: args.docs]
    pos = torch.arange(args.ctx, device=dev)
    queries = [q for q in args.queries.split(",") if q]
    per_doc, failures = [], []

    for d in docs:
        ids = torch.tensor(d["prompt_ids"], dtype=torch.long, device=dev).unsqueeze(0)
        log(d["doc"][:48])
        rec = {"doc": d["doc"]}

        # Is a native prefill on this card reproducible at all?
        def _prefill():
            p = prefill(models["target"], ids, tg)
            return ([k.to(dtype) for k in p.keys_post], [v.to(dtype) for v in p.values])

        rec["native_repeat"] = check_native_repeat(_prefill)
        nk, nv = _prefill()

        # the translated cache this repair starts from
        sp = prefill(models["source"], ids, sg)
        xk = flat = None
        from research.kv_translate.run_a1 import flat_features

        xk = flat_features(to_content_keys(sp)).to(sdtype)
        xv = flat_features([v.float() for v in sp.values]).to(sdtype)
        ak = [
            rerot(b[0].float(), pos, tg.rope_theta).unsqueeze(0).to(dtype)
            for b in mk.apply(xk)
        ]
        av = [v.to(dtype) for v in mv.apply(xv)]

        full = torch.ones(args.ctx, dtype=torch.bool, device=dev)

        def replay_from(initial):
            k, v = initial
            rk, rv, _ = replay_selected(
                models["target"],
                ids,
                full,
                [t.clone() for t in k],
                [t.clone() for t in v],
            )
            return rk, rv

        rk, rv = replay_from((ak, av))

        # agreement with native, recorded rather than summarised
        rec["agreement"] = {
            "keys_rel_frobenius": max(rel_frobenius(a, b) for a, b in zip(rk, nk)),
            "values_rel_frobenius": max(rel_frobenius(a, b) for a, b in zip(rv, nv)),
            "keys_one_minus_cos": max(one_minus_cos(a, b) for a, b in zip(rk, nk)),
            "values_one_minus_cos": max(one_minus_cos(a, b) for a, b in zip(rv, nv)),
            "keys_rel_max": max(rel_max(a, b) for a, b in zip(rk, nk)),
            "values_rel_max": max(rel_max(a, b) for a, b in zip(rv, nv)),
            "rel_max_note": (
                "both tensors are reported: an earlier report gave the key "
                "figure alone, and the value figure is the larger of the two"
            ),
        }
        rec["per_layer"] = per_layer_profile(rk, rv, nk, nv)
        mid_layer = len(rk) // 2
        rec["error_quantiles_mid_layer"] = {
            "layer": mid_layer,
            "keys": error_quantiles(rk[mid_layer], nk[mid_layer]),
            "values": error_quantiles(rv[mid_layer], nv[mid_layer]),
        }
        rec["element_counts"] = {
            "one_layer_one_tensor": int(nk[0].numel()),
            "whole_cache_keys_and_values": int(
                sum(t.numel() for t in nk) + sum(t.numel() for t in nv)
            ),
            "note": (
                "an earlier report attributed the whole-cache count to a "
                "single layer, overstating it by the layer count times two"
            ),
        }

        # layer zero, keys and values
        rec["layer0"] = check_layer0_exact(rk, rv, nk, nv)

        # independence from whatever it started with
        zero = ([torch.zeros_like(t) for t in ak], [torch.zeros_like(t) for t in av])
        g = torch.Generator(device="cpu").manual_seed(1234)
        wrong = (
            [torch.randn(t.shape, generator=g).to(t.device, t.dtype) for t in ak],
            [torch.randn(t.shape, generator=g).to(t.device, t.dtype) for t in av],
        )
        rec["initial_cache_independence"] = check_initial_cache_independence(
            replay_from,
            (ak, av),
            zero,
            wrong,
            ENVELOPE["initial_cache_independence_max"],
        )

        # improvement over the cache it repaired, named as the ratio it is
        unrep = max(rel_frobenius(a, b) for a, b in zip(ak, nk))
        imp = unrep / max(rec["agreement"]["keys_rel_frobenius"], 1e-30)
        rec["improvement_over_initial_cache"] = {
            "factor": imp,
            "initial_cache_error": unrep,
            "limit": ENVELOPE["improvement_over_initial_cache_min"],
            "passed": bool(imp >= ENVELOPE["improvement_over_initial_cache_min"]),
            "scope": (
                "a declared threshold, not a tolerance-free criterion; "
                "the numerator is the starting map's error"
            ),
        }

        # the question the task actually asks, for every query kind
        rec["continuations"] = {}
        for qk in queries:
            key = f"{qk}_query" if qk != "free" else "free_query"
            qids = tok(d[key], return_tensors="pt").input_ids.to(dev)
            ln = continuation_logits(models["target"], nk, nv, qids, args.ctx, dev)
            lr = continuation_logits(models["target"], rk, rv, qids, args.ctx, dev)
            kl = float(
                torch.nn.functional.kl_div(
                    torch.log_softmax(lr, -1),
                    torch.log_softmax(ln, -1),
                    log_target=True,
                    reduction="sum",
                )
            )
            rec["continuations"][qk] = {
                "kl_nats": kl,
                "argmax_disagreement": float(int(ln.argmax()) != int(lr.argmax())),
            }

        # a partial repair, per layer, keys and values
        b1, _ = select(
            "mapped_attn",
            args.blocks,
            model=models["target"],
            mapped=(ak, av),
            native=None,
            query_ids=tok(d["retrieval_query"], return_tensors="pt").input_ids.to(dev),
            n_positions=args.ctx,
        )
        b2, _ = select(
            "mapped_attn",
            args.blocks,
            model=models["target"],
            mapped=(ak, av),
            native=None,
            query_ids=tok(d["retrieval_query"], return_tensors="pt").input_ids.to(dev),
            n_positions=args.ctx,
        )
        rec["selector_deterministic"] = bool(list(b1) == list(b2))
        m = mask_from_blocks(args.ctx, b1).to(dev)
        pk, pv, _ = replay_selected(models["target"], ids, m, ak, av)
        rec["partial_mask"] = check_partial_mask_per_layer(ak, av, pk, pv, m.cpu())

        # zero mask
        zk, zv, zinfo = replay_selected(
            models["target"],
            ids,
            torch.zeros(args.ctx, dtype=torch.bool, device=dev),
            ak,
            av,
        )
        rec["zero_mask_no_op"] = {
            "returns_same_objects": bool(zk is ak and zv is av),
            "replayed": int(zinfo["replayed"]),
            "passed": bool(zinfo["replayed"] == 0 and zk is ak and zv is av),
        }

        # the mode that was timed against the mode that was checked
        rec["ownership"] = check_ownership_modes(
            lambda: replay_selected(models["target"], ids, m, ak, av)[:2],
            lambda: replay_selected(
                models["target"],
                ids,
                m,
                [t.clone() for t in ak],
                [t.clone() for t in av],
                in_place=True,
            )[:2],
            ak,
            av,
        )

        per_doc.append(rec)
        a = rec["agreement"]
        log(
            f"  fro k {a['keys_rel_frobenius']:.3e} v {a['values_rel_frobenius']:.3e} | "
            f"layer0 K {rec['layer0']['keys_exact']} V {rec['layer0']['values_exact']} | "
            f"independent {rec['initial_cache_independence']['passed']} | "
            f"ownership {rec['ownership']['passed']} | "
            f"native repeat {rec['native_repeat']['bitwise_identical']}"
        )

    # ---- the verdict -----------------------------------------------------
    def worst(path_fn):
        return max(path_fn(r) for r in per_doc)

    checks = {
        "relative_frobenius": {
            "measured": max(
                worst(lambda r: r["agreement"]["keys_rel_frobenius"]),
                worst(lambda r: r["agreement"]["values_rel_frobenius"]),
            ),
            "limit": ENVELOPE["relative_frobenius_max"],
        },
        "one_minus_cosine": {
            "measured": max(
                worst(lambda r: r["agreement"]["keys_one_minus_cos"]),
                worst(lambda r: r["agreement"]["values_one_minus_cos"]),
            ),
            "limit": ENVELOPE["one_minus_cosine_max"],
        },
        "continuation_KL": {
            "measured": max(
                max(c["kl_nats"] for c in r["continuations"].values()) for r in per_doc
            ),
            "limit": ENVELOPE["continuation_KL_max"],
        },
        "next_token_disagreements": {
            "measured": sum(
                c["argmax_disagreement"]
                for r in per_doc
                for c in r["continuations"].values()
            ),
            "limit": ENVELOPE["next_token_disagreements_max"],
        },
    }
    for k, c in checks.items():
        c["passed"] = bool(c["measured"] <= c["limit"])
        if not c["passed"]:
            failures.append(f"{k}: {c['measured']:.4e} exceeds {c['limit']:.4e}")

    exact = {
        "layer0_keys": all(r["layer0"]["keys_exact"] for r in per_doc),
        "layer0_values": all(r["layer0"]["values_exact"] for r in per_doc),
        "initial_cache_independence": all(
            r["initial_cache_independence"]["passed"] for r in per_doc
        ),
        "zero_mask_no_op": all(r["zero_mask_no_op"]["passed"] for r in per_doc),
        "partial_unselected_preserved": all(
            r["partial_mask"]["all_unselected_preserved"] for r in per_doc
        ),
        "partial_selected_written": all(
            r["partial_mask"]["all_selected_written"] for r in per_doc
        ),
        "ownership_modes_agree": all(r["ownership"]["passed"] for r in per_doc),
        "selector_deterministic": all(r["selector_deterministic"] for r in per_doc),
        "improvement_over_initial": all(
            r["improvement_over_initial_cache"]["passed"] for r in per_doc
        ),
    }
    for k, ok in exact.items():
        if not ok:
            failures.append(f"{k}: failed")

    native_repeat_identical = all(
        r["native_repeat"]["bitwise_identical"] for r in per_doc
    )

    receipt = {
        "contract": "replay_qualification_rq_v1",
        "scope": (
            "measured agreement inside a declared envelope on a small number "
            "of documents, plus negative controls a wrong implementation "
            "fails. Evidence that the operation is the one intended; not a "
            "proof that it always is."
        ),
        "envelope": ENVELOPE,
        "envelope_provenance": ENVELOPE_PROVENANCE,
        "arm": {
            "name": name,
            "joint_weight_sha256": joint,
            "path": os.path.abspath(path),
        },
        "models": {
            "source": {"id": args.source, "revision": args.source_revision},
            "target": {"id": args.target, "revision": args.target_revision},
        },
        "numeric_mode": {
            "model_dtype": args.dtype,
            "map_dtype": args.solve_dtype,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
        "ctx": args.ctx,
        "blocks": args.blocks,
        "block_tokens": BLOCK,
        "queries": queries,
        "documents": [d["doc"] for d in docs],
        "envelope_checks": checks,
        "exact_checks": exact,
        "native_prefill_repeats_identically": native_repeat_identical,
        "native_repeat_note": (
            "measured, not assumed. If it is false, that difference is the "
            "floor under every agreement figure above."
        ),
        "per_document": per_doc,
        "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "torch": torch.__version__,
        "failures": failures,
        "passed": not failures,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(receipt, open(args.out, "w"), indent=2, sort_keys=True, default=str)

    log("")
    for k, c in checks.items():
        log(
            f"  {k:32s} {c['measured']:.4e} <= {c['limit']:.1e}  "
            f"{'pass' if c['passed'] else 'FAIL'}"
        )
    for k, ok in exact.items():
        log(f"  {k:32s} {'pass' if ok else 'FAIL'}")
    log(f"  {'native_prefill_repeats':32s} {native_repeat_identical}")
    log(f"replay_qualification_rq {'PASSED' if not failures else 'FAILED'}")
    for f in failures:
        log(f"  {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
