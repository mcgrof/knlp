# SPDX-License-Identifier: GPL-2.0
"""The task gate: does a translated cache still do the job, on fresh documents?

Divergence measurements have carried this lane so far, and they are the right
instrument for ranking arms against each other. They are the wrong instrument
for deciding whether to keep going, because a distribution can stay close
while the one fact the prompt was about has been destroyed. This asks the
other question, with answers that are graded by string comparison rather than
by a number nobody can act on.

Every item comes from a document the mapper never saw. Separation is checked
rather than assumed: each candidate document is rejected if a distinctive span
of it appears anywhere in the corpus the maps were fitted on, so adjacency in
a token stream cannot substitute for having learned the document.

The conditions are arranged so that passing is hard for the right reasons.

Target-native is the ceiling: the target prefills the prompt itself. Empty
cache is the floor: no prefix at all. Between them sits the incremental gain a
cache is worth, and every arm is reported as the fraction of that gain it
recovers, which is the only scale on which arms and tasks can be compared.

Two negative controls establish that a score means what it appears to. A
translated cache built from a *different* document tests whether an arm's
score comes from the document or from the model's language ability; on the
planted-code task it must be zero. Empty cache tests the same thing from the
other side.

The last condition is the one that decides whether any of this is worth
deploying. Given the time the translation costs, a serving stack could simply
let the target natively prefill a short suffix of the prompt. That arm is
constructed here with its window chosen to cost what the translation costs, so
the comparison is against what the latency could otherwise have bought rather
than against doing nothing.

Intervals are bootstrapped over documents, not over items, because three items
share each document and treating them as independent would understate the
spread by about the square root of three.

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

from research.cartridges_cas.scripts.cas_kv_rope import rerot  # noqa: E402
from research.kv_translate import freeze  # noqa: E402
from research.kv_translate.fit import SourceLayout  # noqa: E402
from research.kv_translate.pairs import describe, prefill, to_content_keys  # noqa: E402
from research.kv_translate.refine import ResidualMapper  # noqa: E402
from research.kv_translate.run_a0 import make_cache  # noqa: E402
from research.kv_translate.run_a1 import Mapper, flat_features  # noqa: E402
from research.kv_translate.tasks import (  # noqa: E402
    build_items,
    grade_cloze,
    grade_free,
    grade_retrieval,
    wikitext_documents,
)

GIB = float(2**30)
RECOVERY_POINT = 0.75  # of target-native incremental gain
RECOVERY_INTERVAL_FLOOR = 0.50  # the interval must rule out losing more
DETERMINISTIC_POINTS = 5.0  # absolute points behind target-native, at most
MIN_HEADROOM = 0.05  # below this a task cannot distinguish arms and is not scored


@torch.no_grad()
def greedy(model, keys, values, query_ids, prompt_len, max_new, eos_ids):
    """Greedy continuation on top of a supplied cache.

    ``keys``/``values`` may be ``None``, in which case the query is answered
    with no prefix and its positions start at zero -- the model is not told a
    context exists that it cannot see.
    """
    dev = query_ids.device
    if keys is None:
        cache, start = None, 0
    else:
        cache, start = make_cache(keys, values), prompt_len
    n = query_ids.shape[1]
    pos = torch.arange(start, start + n, device=dev).unsqueeze(0)
    attn = torch.ones(1, start + n, dtype=torch.long, device=dev)
    out = model(
        input_ids=query_ids,
        attention_mask=attn,
        position_ids=pos,
        past_key_values=cache,
        use_cache=True,
    )
    cache = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    got = [int(nxt)]
    cur = start + n
    for _ in range(max_new - 1):
        if got[-1] in eos_ids:
            break
        attn = torch.ones(1, cur + 1, dtype=torch.long, device=dev)
        out = model(
            input_ids=nxt,
            attention_mask=attn,
            position_ids=torch.tensor([[cur]], device=dev),
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        got.append(int(nxt))
        cur += 1
    return got


def cluster_bootstrap(values_by_doc, n_boot, seed):
    """Resample documents, not items. Three items share each document."""
    rng = np.random.default_rng(seed)
    docs = list(values_by_doc)
    if not docs:
        return (float("nan"),) * 3
    point = float(np.mean([v for d in docs for v in values_by_doc[d]]))
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(docs), len(docs))
        vals = [v for i in pick for v in values_by_doc[docs[i]]]
        if vals:
            boots.append(float(np.mean(vals)))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return point, float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--arms-dir", required=True)
    ap.add_argument("--h1-json", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--docs", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--max-new-free", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--solve-dtype", default="float32")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    sdtype = getattr(torch, args.solve_dtype)
    t_start = time.time()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

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
    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    pos = torch.arange(args.ctx, device=dev)
    eos_ids = {tok.eos_token_id} | {
        tok.convert_tokens_to_ids(t)
        for t in ("<|im_end|>", "<|endoftext|>")
        if tok.convert_tokens_to_ids(t) is not None
    }
    eos_ids = {e for e in eos_ids if e is not None and e >= 0}

    # ---- documents the maps were not fitted on ---------------------------
    # Both sides are whitespace-normalised before comparison. Normalising only
    # the probe means a span that happens to cross a line break can never
    # match, which would let a document that really was in the fitting corpus
    # through and quietly turn this check into decoration.
    fit_corpus = " ".join(
        " ".join(
            t
            for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
            if t.strip()
        ).split()
    )
    all_docs = wikitext_documents()
    docs, dropped, too_short = [], 0, 0
    for title, body in all_docs:
        flat = " ".join(body.split())
        probes = [
            flat[i : i + 160] for i in (400, len(flat) // 2, max(0, len(flat) - 600))
        ]
        probes = [q for q in probes if len(q) >= 120]
        if not probes:
            too_short += 1
            continue
        if (
            any(q in fit_corpus for q in probes)
            or " ".join(title.split()) in fit_corpus
        ):
            dropped += 1
            continue
        docs.append((title, body))
        if len(docs) >= args.docs * 3:
            break
    log(
        f"{len(docs)} candidate documents; {dropped} rejected as seen during "
        f"fitting, {too_short} skipped as too short to probe"
    )

    items = build_items(docs, tok, args.ctx, seed=args.seed)
    keep_docs = []
    for it in items:
        if it.doc_id not in keep_docs:
            keep_docs.append(it.doc_id)
    keep_docs = set(keep_docs[: args.docs])
    items = [it for it in items if it.doc_id in keep_docs]
    n_doc = len({it.doc_id for it in items})
    kinds = {}
    for it in items:
        kinds[it.kind] = kinds.get(it.kind, 0) + 1
    log(f"{len(items)} items over {n_doc} documents: {kinds}")

    # ---- the arms --------------------------------------------------------
    h1 = json.load(open(args.h1_json))
    arms = {}
    for name, rec in h1["saved_arms"].items():
        path = os.path.join(args.arms_dir, os.path.basename(rec["affine"]))
        mk, mv, man = freeze.load(path, layout, tg, device=dev)
        if "residual" in rec:
            blob = torch.load(
                os.path.join(args.arms_dir, os.path.basename(rec["residual"])),
                map_location=dev,
            )
            rk = ResidualMapper(mk, layout, tg, "k", blob["hidden"], blob["linear"])
            rv = ResidualMapper(mv, layout, tg, "v", blob["hidden"], blob["linear"])
            rk.blocks.load_state_dict(blob["k"])
            rv.blocks.load_state_dict(blob["v"])
            # The frozen map is stored at the double precision it was solved
            # in. Casting it here is not a detail: without it a corrected arm
            # runs its dominant matmul in double while the arm it is compared
            # against runs it in float32, and the measurement becomes a
            # comparison of precisions rather than of methods.
            arms[name] = (
                rk.to(dev).eval().cast(sdtype),
                rv.to(dev).eval().cast(sdtype),
            )
        else:
            arms[name] = (mk.cast(sdtype), mv.cast(sdtype))
        assert man["joint_weight_sha256"] == rec["joint_hash"], name
    log(
        f"loaded {len(arms)} arms; every affine block matched the hash H1 "
        f"recorded, and all are applied at {args.solve_dtype}. Residual "
        f"weights carry no hash of their own and are loaded unverified."
    )

    def translate(arm, Xk, Xv):
        mk, mv = arm
        if isinstance(mk, Mapper):
            pk, pv = mk.apply(Xk), mv.apply(Xv)
        else:
            with torch.no_grad():
                pk, pv = mk(Xk), mv(Xv)
        keys = [
            rerot(b[0].float(), pos, tg.rope_theta).unsqueeze(0).to(dtype) for b in pk
        ]
        return keys, [v.to(dtype) for v in pv]

    def source_features(sp):
        return (
            flat_features(to_content_keys(sp)).to(sdtype),
            flat_features([v.float() for v in sp.values]).to(sdtype),
        )

    # ---- the latency-matched native window -------------------------------
    probe_ids = torch.tensor(
        items[0].meta["prompt_ids"], dtype=torch.long, device=dev
    ).unsqueeze(0)
    sp0 = prefill(models["source"], probe_ids, sg)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def timed(fn, reps=8):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            ev[0].record()
            fn()
            ev[1].record()
            torch.cuda.synchronize()
            ts.append(ev[0].elapsed_time(ev[1]))
        return float(np.percentile(ts, 50))

    # ---- the H0 latency gates, re-measured with the correction in place --
    # A correction that improves quality and loses the latency argument has
    # not advanced anything, so the same output-equivalent first-token path
    # H0 gated is rerun here for every arm: translate the first L-1 positions,
    # install them, and run the last prompt token through the full decoder and
    # the language-model head. The comparison is the target prefilling all L
    # positions itself.
    ft_ids = probe_ids
    head_ids, last_tok = ft_ids[:, : args.ctx - 1], ft_ids[:, args.ctx - 1 :]
    # Only the source prefill is hoisted, because the regime being measured is
    # one where the small model's cache is already resident. Everything after
    # it -- de-rotating the stored keys into the content frame and flattening
    # both sides into the feature block -- is work the translation itself has
    # to do, and H0 charges it inside its timed region. Hoisting it here would
    # compare a cheaper path against H0's threshold.
    sp_head = prefill(models["source"], head_ids, sg)
    pos_head = torch.arange(args.ctx - 1, device=dev)

    @torch.no_grad()
    def native_first_token():
        # Only the last position's logits are wanted, and only the last
        # position's are computed. Letting the head run over all L positions
        # charges the native arm work no serving stack does on a prefill,
        # which inflates the denominator of every ratio here.
        return models["target"](input_ids=ft_ids, logits_to_keep=1).logits[:, -1]

    @torch.no_grad()
    def translated_first_token(arm):
        mk, mv = arm
        xk = flat_features(to_content_keys(sp_head)).to(sdtype)
        xv = flat_features([v.float() for v in sp_head.values]).to(sdtype)
        if isinstance(mk, Mapper):
            pk, pv = mk.apply(xk), mv.apply(xv)
        else:
            pk, pv = mk(xk), mv(xv)
        keys = [
            rerot(b[0].float(), pos_head, tg.rope_theta).unsqueeze(0).to(dtype)
            for b in pk
        ]
        vals = [v.to(dtype) for v in pv]
        attn = torch.ones(1, args.ctx, dtype=torch.long, device=dev)
        pp = torch.arange(args.ctx - 1, args.ctx, device=dev).unsqueeze(0)
        return models["target"](
            input_ids=last_tok,
            attention_mask=attn,
            position_ids=pp,
            past_key_values=make_cache(keys, vals),
            use_cache=True,
        ).logits[:, -1]

    nat_p50 = timed(native_first_token, reps=10)
    first_token = {}
    for name, arm in arms.items():
        ts = []
        for _ in range(3):
            translated_first_token(arm)
        torch.cuda.synchronize()
        for _ in range(10):
            ev[0].record()
            translated_first_token(arm)
            ev[1].record()
            torch.cuda.synchronize()
            ts.append(ev[0].elapsed_time(ev[1]))
        p50, p95 = float(np.percentile(ts, 50)), float(np.percentile(ts, 95))
        nlg = native_first_token().float()
        tlg = translated_first_token(arm).float()
        first_token[name] = {
            "native_p50_ms": nat_p50,
            "translated_p50_ms": p50,
            "translated_p95_ms": p95,
            "ratio_p50": p50 / nat_p50,
            "ratio_p95": p95 / nat_p50,
            "next_token_kl": float(
                torch.nn.functional.kl_div(
                    torch.log_softmax(tlg, -1),
                    torch.log_softmax(nlg, -1),
                    log_target=True,
                    reduction="sum",
                )
            ),
            "argmax_agrees": bool(int(nlg.argmax()) == int(tlg.argmax())),
            "gate_p95_under_native": bool(p95 / nat_p50 < 1.0),
            "gate_p50_at_or_under_075": bool(p50 / nat_p50 <= 0.75),
        }
        del nlg, tlg
    log(f"first-token path, native {nat_p50:.1f} ms:")
    for name, f in first_token.items():
        log(
            f"  {name:<18s} {f['translated_p50_ms']:6.1f} ms  p50 {f['ratio_p50']:.3f} "
            f"p95 {f['ratio_p95']:.3f}  nextKL {f['next_token_kl']:.4f}  "
            f"argmax {'agrees' if f['argmax_agrees'] else 'differs'}  "
            f"{'PASS' if f['gate_p95_under_native'] and f['gate_p50_at_or_under_075'] else 'FAIL'}"
        )
    del sp_head

    # The wrong-prompt control and the 96-token generation are the expensive
    # conditions, so they are run for the arm this gate is actually about.
    # Every arm is still run on both deterministic tasks, so the comparison
    # between arms is complete; what is economised is only the controls.
    #
    # Promotion is by divergence among the arms that keep the latency gates,
    # not by divergence alone. Both orderings use the same two criteria, but
    # taking latency second promotes an arm that is already disqualified and
    # then matches the native-window baseline to that arm's cost -- which, for
    # a correction slower than a prefill, buys the whole prefill and turns the
    # baseline into a copy of target-native. The latency gates are measured
    # above, before any task has been scored.
    ok_latency = [
        a
        for a, f in first_token.items()
        if f["gate_p95_under_native"] and f["gate_p50_at_or_under_075"]
    ]
    corrected = [a for a in arms if a.endswith(("+lin", "+mlp", "_lin", "_mlp"))]
    assert corrected, f"no corrected arm found among {sorted(arms)}"
    pool = [a for a in corrected if a in ok_latency]
    if not pool:
        # Falling back to a disqualified arm would size the native-window
        # baseline against a correction slower than a prefill, which buys the
        # whole prefill and turns that baseline into a copy of target-native.
        raise SystemExit(
            "no corrected arm keeps the latency gates; there is nothing to "
            f"promote. Measured: {json.dumps({a: first_token[a]['ratio_p50'] for a in corrected})}"
        )
    ranked = sorted(pool, key=lambda a: h1["summary"].get(a, {}).get("kl", 9e9))
    promoted_name = ranked[0]
    log(
        f"promoted arm: {promoted_name} (best held-out divergence among the "
        f"{len(pool)} corrected arms that keep the latency gates; "
        f"{sorted(set(corrected) - set(ok_latency))} excluded on latency)"
    )

    def full_apply(arm):
        """Everything between a resident source cache and an installable one."""
        xk, xv = source_features(sp0)
        return translate(arm, xk, xv)

    apply_ms = {n: timed(lambda a=a: full_apply(a)) for n, a in arms.items()}
    # Matched to the promoted arm: the question is what the latency that arm
    # costs could have bought natively instead, so a different arm's cost is
    # not the budget to spend.
    budget = float(apply_ms[promoted_name])

    @torch.no_grad()
    def native_prefill(w):
        # Same correction as the native first-token path: a prefill computes
        # the head on its last position only. Charging it the head over every
        # window position makes the native window look slower than it is, and
        # a slower baseline buys a shorter window for the translator to beat.
        models["target"](input_ids=probe_ids[:, -w:], use_cache=True, logits_to_keep=1)

    window, win_ms = 0, 0.0
    for w in (16, 32, 48, 64, 96, 128, 192, 256, 384, 512):
        if w > args.ctx:
            break
        t = timed(lambda w=w: native_prefill(w), reps=6)
        if t <= budget:
            window, win_ms = w, t
        else:
            break
    log(
        f"apply p50 by arm: "
        + ", ".join(f"{n} {v:.1f} ms" for n, v in apply_ms.items())
        + f" | budget {budget:.1f} ms buys a native window of {window} tokens "
        f"({win_ms:.1f} ms)"
    )
    del sp0

    # ---- run every condition on every item -------------------------------
    # Every arm gets its own wrong-prompt control rather than borrowing the
    # promoted arm's. A control built from different weights answers a
    # different question, and "this arm carries the document" is a claim about
    # this arm. The controls cost one extra generation per arm per item, which
    # is the price of the gate meaning what it says.
    WRONG = "|wrong"
    refs = ["target_native", "empty_cache"] + ([f"recent_{window}"] if window else [])
    conditions = refs + sorted(arms) + [a + WRONG for a in sorted(arms)]
    # The wrong-prompt controls exist to interpret the deterministic tasks; a
    # mismatched cache has no meaningful pathology score of its own.
    free_conditions = set(refs) | set(arms)

    rows = []
    doc_ids = sorted({it.doc_id for it in items})
    shift = {d: doc_ids[(i + 1) % len(doc_ids)] for i, d in enumerate(doc_ids)}

    cache_by_doc = {}
    for di, d in enumerate(doc_ids):
        pid = next(it.meta["prompt_ids"] for it in items if it.doc_id == d)
        ids = torch.tensor(pid, dtype=torch.long, device=dev).unsqueeze(0)
        sp = prefill(models["source"], ids, sg)
        Xk, Xv = source_features(sp)
        cache_by_doc[d] = {
            "prompt_ids": ids.cpu(),
            "Xk": Xk.half().cpu(),
            "Xv": Xv.half().cpu(),
        }
        del sp, Xk, Xv
        if (di + 1) % 16 == 0:
            log(f"  source-side prepared {di + 1}/{len(doc_ids)}")

    for ii, it in enumerate(items):
        q = tok(it.query, return_tensors="pt", add_special_tokens=False)
        q = q.input_ids.to(dev)
        cap = args.max_new_free if it.kind == "free" else args.max_new
        ids = cache_by_doc[it.doc_id]["prompt_ids"].to(dev)
        for cond in conditions:
            if it.kind == "free" and cond not in free_conditions:
                continue
            if cond == "target_native":
                tp = prefill(models["target"], ids, tg)
                keys, vals = tp.keys_post, tp.values
            elif cond == "empty_cache":
                keys, vals = None, None
            elif cond.startswith("recent_"):
                w = int(cond.split("_")[1])
                tp = prefill(models["target"], ids[:, -w:], tg)
                keys, vals = tp.keys_post, tp.values
            else:
                wrong = cond.endswith(WRONG)
                use = cond[: -len(WRONG)] if wrong else cond
                src = shift[it.doc_id] if wrong else it.doc_id
                base = cache_by_doc[src]
                Xk = base["Xk"].to(dev).to(sdtype)
                Xv = base["Xv"].to(dev).to(sdtype)
                keys, vals = translate(arms[use], Xk, Xv)
                del Xk, Xv
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
                r.update(grade_retrieval(it.answer, text))
                r["primary"] = r["correct"]
            elif it.kind == "cloze":
                r.update(grade_cloze(it.answer, text))
                r["primary"] = r["overlap"]
            else:
                r.update(grade_free(text, got, eos_ids, cap))
                r["primary"] = r["healthy"]
            rows.append(r)
            del keys, vals
        if (ii + 1) % 16 == 0:
            log(f"  graded {ii + 1}/{len(items)} items")

    with open(os.path.join(args.out_dir, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    # ---- gates -----------------------------------------------------------
    def by_doc(kind, cond, field="primary"):
        d = {}
        for r in rows:
            if r["kind"] == kind and r["condition"] == cond:
                d.setdefault(r["doc"], []).append(r[field])
        return d

    task_kinds = ["retrieval", "cloze"]
    summary = {}
    for kind in task_kinds + ["free"]:
        summary[kind] = {}
        for cond in conditions if kind != "free" else sorted(free_conditions):
            p, lo, hi = cluster_bootstrap(by_doc(kind, cond), args.n_boot, args.seed)
            summary[kind][cond] = {"point": p, "lo": lo, "hi": hi}

    def paired_recovery(kind, cond, n_boot, seed):
        """Interval on (arm - empty) / (native - empty), resampling the ratio.

        The ceiling and the floor are estimates from the same documents as the
        arm, not constants. Holding them fixed and resampling only the
        numerator produces an interval on the wrong quantity and makes it
        narrower than the ratio's own, which matters because the gate tests
        this interval's lower bound. Documents are resampled once per replicate
        and all three conditions are recomputed on that resample, so the
        pairing between them is preserved.
        """
        arm = by_doc(kind, cond)
        nat = by_doc(kind, "target_native")
        emp = by_doc(kind, "empty_cache")
        docs = sorted(set(arm) & set(nat) & set(emp))
        if not docs:
            return None

        def ratio(sel):
            a = float(np.mean([v for d in sel for v in arm[d]]))
            n = float(np.mean([v for d in sel for v in nat[d]]))
            e = float(np.mean([v for d in sel for v in emp[d]]))
            g = n - e
            return None if abs(g) < 1e-9 else ((a - e) / g, n - a)

        base = ratio(docs)
        if base is None:
            return None
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            sel = [docs[i] for i in rng.integers(0, len(docs), len(docs))]
            r = ratio(sel)
            if r is not None:
                boots.append(r[0])
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"),) * 2
        return {
            "point": base[0],
            "lo": float(lo),
            "hi": float(hi),
            "abs_points_behind_native": 100.0 * base[1],
            "n_docs": len(docs),
            "headroom": float(
                np.mean([v for d in docs for v in nat[d]])
                - np.mean([v for d in docs for v in emp[d]])
            ),
        }

    recovery = {}
    for cond in conditions:
        recovery[cond] = {
            kind: paired_recovery(kind, cond, args.n_boot, args.seed)
            for kind in task_kinds
        }

    log("task scores (point [95% CI over documents]):")
    for kind in task_kinds + ["free"]:
        log(f"  {kind}:")
        for cond in sorted(summary[kind]):
            s = summary[kind][cond]
            extra = ""
            if kind in task_kinds and recovery[cond].get(kind):
                extra = f"  recovers {100 * recovery[cond][kind]['point']:.1f}%"
            log(
                f"    {cond:<18s} {s['point']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]{extra}"
            )

    def paired_delta(kind, a, b, n_boot, seed):
        """Interval on the per-document difference a - b.

        Both conditions are scored on the same documents, so the difference is
        paired. Comparing two independently-resampled intervals for overlap
        instead would be testing a weaker claim than the one the gate states,
        and would be needlessly conservative on exactly the comparison the
        negative controls exist to make.
        """
        da, db = by_doc(kind, a), by_doc(kind, b)
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
        return {"point": m(docs), "lo": float(lo), "hi": float(hi)}

    def gate(cond):
        g = {}
        # Carries prompt-specific information: it must beat both controls on
        # the task that cannot be answered without the document. Tested as a
        # paired difference, because every condition is scored on the same
        # documents.
        g["retrieval_point"] = summary["retrieval"][cond]["point"]
        for name, ctrl in (
            ("empty_cache", "empty_cache"),
            ("wrong_prompt", cond + WRONG),
        ):
            d = paired_delta("retrieval", cond, ctrl, args.n_boot, args.seed)
            g["vs_" + name] = d
            g["beats_" + name] = bool(d is not None and d["lo"] > 0.0)
        g["prompt_specific"] = g["beats_empty_cache"] and g["beats_wrong_prompt"]

        # Recovery of the target-native incremental gain. A task with no
        # headroom -- where an empty cache already scores what the native
        # cache does -- is recorded as unmeasurable rather than skipped, so it
        # cannot silently satisfy the criterion it was dropped from.
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

        # Generation pathology. An arm this was never run for is None, not
        # True: a criterion that was not measured must not be reported as one
        # the arm satisfied.
        free_nat = summary["free"].get("target_native", {}).get("point")
        fr = summary["free"].get(cond, {}).get("point")
        if fr is None or free_nat is None or fr != fr:
            g["no_pathology_regression"] = None
        else:
            d = paired_delta("free", cond, "target_native", args.n_boot, args.seed)
            g["no_pathology_regression"] = bool(d is not None and d["lo"] > -0.10)
            g["vs_native_pathology"] = d
        g["pathology_measured"] = g["no_pathology_regression"] is not None

        # What the same latency buys natively. The module docstring calls this
        # the condition that decides deployment, so it is a criterion and not
        # a printed aside.
        if window:
            d = paired_delta(
                "retrieval", cond, f"recent_{window}", args.n_boot, args.seed
            )
            g["vs_recent_window"] = d
            g["beats_latency_matched_native"] = bool(d is not None and d["lo"] > 0.0)
        else:
            g["vs_recent_window"] = None
            g["beats_latency_matched_native"] = None

        ft = first_token.get(cond, {})
        g["retains_h0_latency_gates"] = bool(
            ft.get("gate_p95_under_native") and ft.get("gate_p50_at_or_under_075")
        )
        g["passes"] = bool(
            g["prompt_specific"]
            and g["recovers_incremental_gain"]
            and g["within_absolute_points"]
            and g["no_pathology_regression"] is True
            and g["beats_latency_matched_native"] is True
            and g["retains_h0_latency_gates"]
        )
        return g

    gates = {c: gate(c) for c in sorted(arms)}
    passing = [c for c, g in gates.items() if g["passes"]]
    log("gate:")
    for c, g in gates.items():
        log(
            f"  {c:<18s} prompt-specific {g['prompt_specific']} "
            f"recovers {g['recovers_incremental_gain']} "
            f"within-{DETERMINISTIC_POINTS:.0f}pts {g['within_absolute_points']} "
            f"no-pathology {g['no_pathology_regression']} "
            f"beats-recent-window {g['beats_latency_matched_native']} "
            f"latency {g['retains_h0_latency_gates']} "
            f"=> {'PASS' if g['passes'] else 'FAIL'}"
        )

    out = {
        "config": vars(args),
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "documents": sorted(keep_docs),
        "n_items": len(items),
        "conditions": conditions,
        "kinds": kinds,
        "apply_ms": apply_ms,
        "first_token": first_token,
        "latency_matched_window": {"tokens": window, "ms": win_ms, "budget_ms": budget},
        "summary": summary,
        "recovery": recovery,
        "gates": gates,
        "passing_arms": passing,
        "thresholds": {
            "recovery_point": RECOVERY_POINT,
            "recovery_interval_floor": RECOVERY_INTERVAL_FLOOR,
            "deterministic_points": DETERMINISTIC_POINTS,
        },
        "wall_s": time.time() - t_start,
    }
    with open(os.path.join(args.out_dir, "smoke.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    log(f"passing arms: {passing or 'none'}")
    log(f"wrote {args.out_dir}/smoke.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
