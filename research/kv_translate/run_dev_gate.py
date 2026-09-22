# SPDX-License-Identifier: GPL-2.0
"""Score candidate translators on the fixed development documents.

The gate is the one inherited from the parent plan and is not restated here as
a new standard: prompt specificity against each arm's own wrong-prompt
control, at least 75 percent recovery of the target's incremental task gain
with a lower bound above 50 percent, no more than five absolute points behind
the target, no generation-health regression, and a first-token path under the
target's own at p95 and at or under three quarters of it at p50.

Two things differ from the run that produced the historical verdict. Retrieval
is scored by whole-token occurrence of the gold code rather than by substring,
so an answer with the code embedded in something longer is a miss. And the
gold answers come from a manifest written before generation, with each
prompt's token ids verified on load, rather than being implicit in the
harness that generated them.

Every condition an arm is judged against is measured for that arm: its own
wrong-prompt cache, its own health, its own first-token path. A criterion that
was not measured is reported as absent, never as satisfied.

Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
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
from research.kv_translate import freeze  # noqa: E402
from research.kv_translate.fit import SourceLayout  # noqa: E402
from research.kv_translate.pairs import (  # noqa: E402
    describe,
    prefill,
    to_content_keys,
)
from research.kv_translate.run_a0 import make_cache  # noqa: E402
from research.kv_translate.run_a1 import flat_features  # noqa: E402
from research.kv_translate.tasks import (  # noqa: E402
    grade_cloze,
    grade_free,
    grade_retrieval,
)

WRONG = "|wrong"
RECOVERY_POINT = 0.75
RECOVERY_INTERVAL_FLOOR = 0.50
DETERMINISTIC_POINTS = 5.0
MIN_HEADROOM = 0.05


def log_now(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@torch.no_grad()
def greedy(model, keys, values, query_ids, prompt_len, max_new, eos_ids):
    dev = query_ids.device
    cache, start = (None, 0) if keys is None else (make_cache(keys, values), prompt_len)
    n = query_ids.shape[1]
    out = model(
        input_ids=query_ids,
        attention_mask=torch.ones(1, start + n, dtype=torch.long, device=dev),
        position_ids=torch.arange(start, start + n, device=dev).unsqueeze(0),
        past_key_values=cache,
        use_cache=True,
    )
    cache = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    got, cur = [int(nxt)], start + n
    for _ in range(max_new - 1):
        if got[-1] in eos_ids:
            break
        out = model(
            input_ids=nxt,
            attention_mask=torch.ones(1, cur + 1, dtype=torch.long, device=dev),
            position_ids=torch.tensor([[cur]], device=dev),
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        got.append(int(nxt))
        cur += 1
    return got


def by_doc(rows, kind, cond, field="primary"):
    d = {}
    for r in rows:
        if r["kind"] == kind and r["condition"] == cond:
            d.setdefault(r["doc"], []).append(r[field])
    return d


def cluster_bootstrap(vals_by_doc, n_boot, seed):
    rng = np.random.default_rng(seed)
    docs = list(vals_by_doc)
    if not docs:
        return (float("nan"),) * 3
    point = float(np.mean([v for d in docs for v in vals_by_doc[d]]))
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(docs), len(docs))
        boots.append(float(np.mean([v for i in pick for v in vals_by_doc[docs[i]]])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_delta(rows, kind, a, b, n_boot, seed):
    da, db = by_doc(rows, kind, a), by_doc(rows, kind, b)
    docs = sorted(set(da) & set(db))
    if not docs:
        return None

    def m(sel):
        return float(np.mean([v for d in sel for v in da[d]])) - float(
            np.mean([v for d in sel for v in db[d]])
        )

    rng = np.random.default_rng(seed)
    boots = [
        m([docs[i] for i in rng.integers(0, len(docs), len(docs))])
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": m(docs), "lo": float(lo), "hi": float(hi), "n_docs": len(docs)}


def paired_recovery(rows, kind, cond, n_boot, seed):
    arm = by_doc(rows, kind, cond)
    nat = by_doc(rows, kind, "target_native")
    emp = by_doc(rows, kind, "empty_cache")
    docs = sorted(set(arm) & set(nat) & set(emp))
    if not docs:
        return None

    def ratio(sel):
        a = float(np.mean([v for d in sel for v in arm[d]]))
        n = float(np.mean([v for d in sel for v in nat[d]]))
        e = float(np.mean([v for d in sel for v in emp[d]]))
        return None if abs(n - e) < 1e-9 else ((a - e) / (n - e), n - a)

    base = ratio(docs)
    if base is None:
        return None
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        r = ratio([docs[i] for i in rng.integers(0, len(docs), len(docs))])
        if r is not None:
            boots.append(r[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "point": base[0],
        "lo": float(lo),
        "hi": float(hi),
        "abs_points_behind_native": 100.0 * base[1],
        "headroom": float(
            np.mean([v for d in docs for v in nat[d]])
            - np.mean([v for d in docs for v in emp[d]])
        ),
        "n_docs": len(docs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument(
        "--source-revision",
        default="main",
        help="immutable snapshot id for the source weights. A config hash "
        "identifies a configuration, not the weight files, and 'main' moves, "
        "so both arms and every control pin this explicitly.",
    )
    ap.add_argument("--target-revision", default="main")
    ap.add_argument(
        "--verdict-scope",
        default="deployment",
        choices=("deployment", "quality_health"),
        help="which conjunction decides 'passes'. 'deployment' is the parent "
        "plan's, including the first-token criteria and the comparison against "
        "a latency-matched native suffix. 'quality_health' drops those two, "
        "for a stage told to test quality and health at this length and not "
        "deployment at it. Both scopes measure the same things and report the "
        "same numbers; they differ only in what is allowed to decide.",
    )
    ap.add_argument("--gold", required=True, help="dev manifest with prompt ids")
    ap.add_argument(
        "--arm",
        action="append",
        default=[],
        required=True,
        help="name=path.pt, repeatable; the first is the candidate",
    )
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--max-new-free", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--solve-dtype", default="float32")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    sdtype = getattr(torch, args.solve_dtype)
    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    tok = AutoTokenizer.from_pretrained(args.target, revision=args.target_revision)
    models, geom = {}, {}
    revs = {"source": args.source_revision, "target": args.target_revision}
    for role, mid in (("source", args.source), ("target", args.target)):
        m = AutoModelForCausalLM.from_pretrained(
            mid, revision=revs[role], dtype=dtype, attn_implementation="sdpa"
        ).to(dev)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models[role], geom[role] = m, describe(m, mid)
    sg, tg = geom["source"], geom["target"]
    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    pos = torch.arange(args.ctx, device=dev)
    eos_ids = {
        e
        for e in (
            {tok.eos_token_id}
            | {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
        )
        if e is not None and e >= 0
    }

    # Everything scored comes from the manifest, which was written before the
    # first generation: the prompt token ids, the queries and the gold. The
    # items are not rebuilt. Reconstruction is deterministic but its random
    # draws depend on how many candidate documents were walked to reach these
    # sixty-four, so rebuilding from the surviving subset alone would produce
    # different prompts -- and a record that can only be read by re-running
    # the machinery that wrote it is not a record.
    gold = json.load(open(args.gold))
    gmap = {d["doc"]: d for d in gold["documents"]}
    need = (
        "prompt_ids",
        "retrieval_gold",
        "retrieval_query",
        "cloze_gold",
        "cloze_query",
        "free_query",
    )
    incomplete = [d for d, v in gmap.items() if any(k not in v for k in need)]
    if incomplete:
        raise SystemExit(
            f"INVALID: {len(incomplete)} manifest entries lack a required "
            f"field, first: {sorted(incomplete)[:3]}"
        )
    bad = []
    for d, v in gmap.items():
        h = hashlib.sha256(json.dumps(v["prompt_ids"]).encode()).hexdigest()
        if h != v.get("prompt_ids_sha256") or len(v["prompt_ids"]) != args.ctx:
            bad.append(d)
    if bad:
        raise SystemExit(
            f"INVALID: {len(bad)} prompts do not match their recorded hash or "
            f"length, first: {sorted(bad)[:3]}"
        )

    class Item:
        __slots__ = ("doc_id", "kind", "query")

        def __init__(self, doc_id, kind, query):
            self.doc_id, self.kind, self.query = doc_id, kind, query

    items = [
        Item(d, kind, gmap[d][f"{kind}_query" if kind != "free" else "free_query"])
        for d in sorted(gmap)
        for kind in ("retrieval", "cloze", "free")
    ]
    n_doc = len(gmap)
    log_now(f"{len(items)} items over {n_doc} documents, all prompts verified")

    arms = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        mk, mv, man = freeze.load(path, layout, tg, device=dev)
        arms[name] = (mk.cast(sdtype), mv.cast(sdtype), man)
        log_now(f"arm {name}: {man['joint_weight_sha256'][:16]} from {path}")
    candidate = args.arm[0].split("=", 1)[0]

    def source_features(sp):
        return (
            flat_features(to_content_keys(sp)).to(sdtype),
            flat_features([v.float() for v in sp.values]).to(sdtype),
        )

    def translate(name, xk, xv):
        mk, mv, _ = arms[name]
        pk, pv = mk.apply(xk), mv.apply(xv)
        keys = [
            rerot(b[0].float(), pos, tg.rope_theta).unsqueeze(0).to(dtype) for b in pk
        ]
        return keys, [v.to(dtype) for v in pv]

    ids_of = {
        d: torch.tensor(v["prompt_ids"], dtype=torch.long, device=dev).unsqueeze(0)
        for d, v in gmap.items()
    }

    # ---- first-token path, and the window the candidate's cost buys --------
    probe = ids_of[items[0].doc_id]
    head, last = probe[:, : args.ctx - 1], probe[:, args.ctx - 1 :]
    sp_head = prefill(models["source"], head, sg)
    pos_head = torch.arange(args.ctx - 1, device=dev)
    ev = (
        [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        if dev == "cuda"
        else None
    )

    @torch.no_grad()
    def native_first():
        return models["target"](input_ids=probe, logits_to_keep=1).logits[:, -1]

    @torch.no_grad()
    def translated_first(name):
        mk, mv, _ = arms[name]
        xk = flat_features(to_content_keys(sp_head)).to(sdtype)
        xv = flat_features([v.float() for v in sp_head.values]).to(sdtype)
        keys = [
            rerot(b[0].float(), pos_head, tg.rope_theta).unsqueeze(0).to(dtype)
            for b in mk.apply(xk)
        ]
        vals = [v.to(dtype) for v in mv.apply(xv)]
        return models["target"](
            input_ids=last,
            attention_mask=torch.ones(1, args.ctx, dtype=torch.long, device=dev),
            position_ids=torch.arange(args.ctx - 1, args.ctx, device=dev).unsqueeze(0),
            past_key_values=make_cache(keys, vals),
            use_cache=True,
        ).logits[:, -1]

    def timed(fn, reps=10):
        for _ in range(3):
            fn()
        if dev != "cuda":
            return float("nan")
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            ev[0].record()
            fn()
            ev[1].record()
            torch.cuda.synchronize()
            ts.append(ev[0].elapsed_time(ev[1]))
        return ts

    nat_ts = timed(native_first)
    nat_p50 = float(np.percentile(nat_ts, 50))
    first_token = {}
    for name in arms:
        ts = timed(lambda n=name: translated_first(n))
        p50, p95 = float(np.percentile(ts, 50)), float(np.percentile(ts, 95))
        nl, tl = native_first().float(), translated_first(name).float()
        first_token[name] = {
            "native_p50_ms": nat_p50,
            "translated_p50_ms": p50,
            "translated_p95_ms": p95,
            "ratio_p50": p50 / nat_p50,
            "ratio_p95": p95 / nat_p50,
            "argmax_agrees": bool(int(nl.argmax()) == int(tl.argmax())),
            "gate_p95_under_native": bool(p95 / nat_p50 < 1.0),
            "gate_p50_at_or_under_075": bool(p50 / nat_p50 <= 0.75),
        }
        log_now(
            f"  first token {name}: {p50:.1f} ms, p50 {p50 / nat_p50:.3f}, "
            f"{'pass' if first_token[name]['gate_p95_under_native'] and first_token[name]['gate_p50_at_or_under_075'] else 'FAIL'}"
        )
    budget = float(np.percentile(timed(lambda: translated_first(candidate)), 50))
    # Every declared size is measured. The previous version stopped at the
    # first size over budget, which assumes native prefill time is monotone in
    # tokens; kernel selection thresholds and timing noise both break that,
    # and the assumption silently shrinks the baseline the method must beat.
    window, win_ms, ladder = 0, 0.0, []
    for w in (16, 32, 48, 64, 96, 128, 192, 256, 384, 512):
        if w > args.ctx:
            continue

        @torch.no_grad()
        def nat_w(w=w):
            models["target"](input_ids=probe[:, -w:], use_cache=True, logits_to_keep=1)

        t = float(np.percentile(timed(nat_w, reps=6), 50))
        ladder.append({"tokens": w, "ms": t, "within_budget": bool(t <= budget)})
        if t <= budget and w > window:
            window, win_ms = w, t
    log_now(
        f"candidate {candidate} costs {budget:.1f} ms end to end, which buys a "
        f"native window of {window} tokens ({win_ms:.1f} ms)"
    )
    del sp_head

    # ---- score every condition -------------------------------------------
    refs = ["target_native", "empty_cache"] + ([f"recent_{window}"] if window else [])
    conditions = refs + sorted(arms) + [a + WRONG for a in sorted(arms)]
    free_conditions = set(refs) | set(arms)
    doc_ids = sorted(ids_of)
    shift = {d: doc_ids[(i + 1) % len(doc_ids)] for i, d in enumerate(doc_ids)}

    feats = {}
    for i, d in enumerate(doc_ids):
        sp = prefill(models["source"], ids_of[d], sg)
        xk, xv = source_features(sp)
        # Host staging at the solver precision, not half. Scoring used to
        # cache these as float16 and upcast on use while the timed path built
        # them in float32 and never rounded, so the numbers being scored and
        # the numbers being timed came from different features. If reduced
        # precision staging is ever the product contract it must apply to both
        # paths and its cost must appear in the timing; until then both are
        # float32.
        feats[d] = (xk.to(sdtype).cpu(), xv.to(sdtype).cpu())
        del sp, xk, xv
        if (i + 1) % 16 == 0:
            log_now(f"  source side {i + 1}/{len(doc_ids)}")

    rows = []
    for ii, it in enumerate(items):
        q = tok(it.query, return_tensors="pt", add_special_tokens=False).input_ids.to(
            dev
        )
        cap = args.max_new_free if it.kind == "free" else args.max_new
        ids = ids_of[it.doc_id]
        g = gmap[it.doc_id]
        for cond in conditions:
            if it.kind == "free" and cond not in free_conditions:
                continue
            if cond == "target_native":
                tp = prefill(models["target"], ids, tg)
                keys, vals = tp.keys_post, tp.values
            elif cond == "empty_cache":
                keys, vals = None, None
            elif cond.startswith("recent_"):
                tp = prefill(models["target"], ids[:, -int(cond.split("_")[1]) :], tg)
                keys, vals = tp.keys_post, tp.values
            else:
                wrong = cond.endswith(WRONG)
                name = cond[: -len(WRONG)] if wrong else cond
                src = shift[it.doc_id] if wrong else it.doc_id
                xk = feats[src][0].to(dev)
                xv = feats[src][1].to(dev)
                keys, vals = translate(name, xk, xv)
                del xk, xv
            plen = keys[0].shape[2] if keys is not None else 0
            got = greedy(models["target"], keys, vals, q, plen, cap, eos_ids)
            text = tok.decode(got, skip_special_tokens=True)
            r = {
                "doc": it.doc_id,
                "kind": it.kind,
                "condition": cond,
                "produced": text,
                "n_tokens": len(got),
            }
            if it.kind == "retrieval":
                r.update(grade_retrieval(g["retrieval_gold"], text))
                r["gold"] = g["retrieval_gold"]
                r["primary"] = r["exact_any_run"]
            elif it.kind == "cloze":
                r.update(grade_cloze(g["cloze_gold"], text))
                r["gold"] = g["cloze_gold"]
                r["primary"] = r["overlap"]
            else:
                r.update(grade_free(text, got, eos_ids, cap))
                r["primary"] = r["healthy"]
            rows.append(r)
            del keys, vals
        if (ii + 1) % 16 == 0:
            log_now(f"  graded {ii + 1}/{len(items)}")

    with open(os.path.join(args.out_dir, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    task_kinds = ["retrieval", "cloze"]
    summary = {}
    for kind in task_kinds + ["free"]:
        present = sorted({r["condition"] for r in rows if r["kind"] == kind})
        summary[kind] = {}
        for c in present:
            p, lo, hi = cluster_bootstrap(by_doc(rows, kind, c), args.n_boot, args.seed)
            summary[kind][c] = {"point": p, "lo": lo, "hi": hi}
    recovery = {
        c: {k: paired_recovery(rows, k, c, args.n_boot, args.seed) for k in task_kinds}
        for c in conditions
    }

    gates = {}
    for cond in sorted(arms):
        g = {"retrieval_point": summary["retrieval"][cond]["point"]}
        for nm, ctrl in (
            ("empty_cache", "empty_cache"),
            ("wrong_prompt", cond + WRONG),
        ):
            d = paired_delta(rows, "retrieval", cond, ctrl, args.n_boot, args.seed)
            g["vs_" + nm] = d
            g["beats_" + nm] = bool(d is not None and d["lo"] > 0.0)
        g["prompt_specific"] = g["beats_empty_cache"] and g["beats_wrong_prompt"]
        ok_rec, ok_abs, measured = True, True, []
        for kind in task_kinds:
            rc = recovery[cond].get(kind)
            if rc is None or abs(rc.get("headroom", 0.0)) < MIN_HEADROOM:
                continue
            measured.append(kind)
            ok_rec = ok_rec and rc["point"] >= RECOVERY_POINT
            ok_rec = ok_rec and rc["lo"] > RECOVERY_INTERVAL_FLOOR
            ok_abs = ok_abs and rc["abs_points_behind_native"] <= DETERMINISTIC_POINTS
        g["kinds_with_headroom"] = measured
        g["recovers_incremental_gain"] = bool(measured) and ok_rec
        g["within_absolute_points"] = bool(measured) and ok_abs
        fr = summary["free"].get(cond, {}).get("point")
        if fr is None or fr != fr:
            g["no_pathology_regression"] = None
        else:
            d = paired_delta(
                rows, "free", cond, "target_native", args.n_boot, args.seed
            )
            g["vs_native_pathology"] = d
            g["no_pathology_regression"] = bool(d is not None and d["lo"] > -0.10)
        if window:
            d = paired_delta(
                rows, "retrieval", cond, f"recent_{window}", args.n_boot, args.seed
            )
            g["vs_recent_window"] = d
            g["beats_latency_matched_native"] = bool(d is not None and d["lo"] > 0.0)
        else:
            g["beats_latency_matched_native"] = None
        ft = first_token.get(cond, {})
        g["retains_h0_latency_gates"] = bool(
            ft.get("gate_p95_under_native") and ft.get("gate_p50_at_or_under_075")
        )
        behavioural = bool(
            g["prompt_specific"]
            and g["recovers_incremental_gain"]
            and g["within_absolute_points"]
            and g["no_pathology_regression"] is True
        )
        deployment = bool(
            g["beats_latency_matched_native"] is True and g["retains_h0_latency_gates"]
        )
        g["behavioural_requirements"] = behavioural
        g["deployment_requirements"] = deployment
        g["verdict_scope"] = args.verdict_scope
        # The cost criteria are always measured and always reported. The scope
        # decides only whether they are allowed to settle the verdict, because
        # a stage asked about quality at this length should not be answered by
        # a cost failure already known at it.
        g["passes"] = (
            behavioural
            if args.verdict_scope == "quality_health"
            else (behavioural and deployment)
        )
        gates[cond] = g

    log_now("development gate:")
    for c, g in sorted(gates.items()):
        rc = recovery[c]["retrieval"]
        log_now(
            f"  {c:<16s} retrieval {summary['retrieval'][c]['point'] * 64:.0f}/64 "
            f"recovers {100 * rc['point']:.1f}% behind {rc['abs_points_behind_native']:.1f} "
            f"specific {g['prompt_specific']} health {g['no_pathology_regression']} "
            f"latency {g['retains_h0_latency_gates']} "
            f"=> {'PASS' if g['passes'] else 'FAIL'}"
        )

    out = {
        "config": vars(args),
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "tf32_allowed": (
            bool(torch.backends.cuda.matmul.allow_tf32) if dev == "cuda" else None
        ),
        "arm_hashes": {k: v[2]["joint_weight_sha256"] for k, v in arms.items()},
        "candidate": candidate,
        "documents": doc_ids,
        "conditions": conditions,
        "first_token": first_token,
        "latency_matched_window": {
            "tokens": window,
            "ms": win_ms,
            "budget_ms": budget,
            "matched_to": candidate,
            "ladder": ladder,
            "note": "every declared size measured; largest within budget wins",
        },
        "source_feature_dtype": args.solve_dtype,
        "summary": summary,
        "recovery": recovery,
        "gates": gates,
        "passing_arms": [c for c, g in gates.items() if g["passes"]],
        "thresholds": {
            "recovery_point": RECOVERY_POINT,
            "recovery_interval_floor": RECOVERY_INTERVAL_FLOOR,
            "deterministic_points": DETERMINISTIC_POINTS,
            "min_headroom": MIN_HEADROOM,
        },
        "wall_seconds": time.time() - t_start,
    }
    with open(os.path.join(args.out_dir, "dev_gate.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    log_now(f"passing arms: {out['passing_arms'] or 'none'}")
    log_now(f"wrote {args.out_dir}/dev_gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
