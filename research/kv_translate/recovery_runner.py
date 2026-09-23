# SPDX-License-Identifier: GPL-2.0
"""Run the recovery conditions and score them.

One runner covers the whole stage. Earlier stages had a separate script per
condition, and the conditions drifted: two of them ended up scoring the same
arm under different first-token boundaries, and the difference was invisible
until someone compared the sources. Here every condition is a row in one
matrix, produced by one loop over one document, so a difference between two
conditions can only come from the repair.

Three repairs are available and they are not interchangeable. ``oracle``
substitutes the receiver's own cache at the selected positions: it is what a
perfect recomputation would produce, it is unavailable at serving time, and it
exists to say whether any sparse set of positions carries the loss.
``replay`` recomputes those positions with the receiver, reading the mixed
cache, which is the deployable operation and the only one a cost may be quoted
for. ``none`` leaves the translated cache alone.

The selector runs before the repair and must not contaminate what it selects.
It reads the question, so its pass over the cache appends question positions;
those are discarded and the cache handed to the repair is the one that existed
before the selector ran. Getting this wrong would leave the question resident
in the prefix, and every condition would then be answering with the question
already in context.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from research.kv_translate import freeze
from research.kv_translate.pairs import describe, prefill, to_content_keys
from research.kv_translate.recovery import (
    BLOCK,
    CacheContract,
    mask_from_blocks,
    rank_blocks_on_device,
    repair_cache,
    uniform_blocks,
)
from research.kv_translate.replay import replay_selected
from research.kv_translate.tasks import grade_cloze, grade_free, grade_retrieval
from research.cartridges_cas.scripts.cas_kv_rope import rerot
from research.kv_translate.fit import SourceLayout
from research.kv_translate.run_a0 import make_cache
from research.kv_translate.run_a1 import flat_features

TASK_KINDS = ("retrieval", "cloze", "free")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fingerprint(keys, values):
    """A cheap witness that a cache was not modified behind our back."""
    h = hashlib.sha256()
    for t in list(keys) + list(values):
        h.update(str(tuple(t.shape)).encode())
        h.update(t[..., :1, :4].float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


class eager_attention:
    """Report attention weights for one pass, without paying for them elsewhere.

    The fused implementation never materialises the attention matrix, so it
    cannot return one; asking it to yields ``None`` and a warning, and a
    selector that ranked on ``None`` would have to fall back to something that
    is not attention. Loading the whole model eagerly instead would make every
    document prefill materialise a four-thousand-square matrix per layer, for
    the sake of a pass over a question of a few dozen tokens.

    So the implementation is switched for the selector pass only. The
    attention modules read it from the config on each forward, which is what
    makes this work; the restore is unconditional.
    """

    def __init__(self, model):
        self.cfg = model.config
        self.prev = None

    def __enter__(self):
        self.prev = self.cfg._attn_implementation
        self.cfg._attn_implementation = "eager"
        return self

    def __exit__(self, *exc):
        self.cfg._attn_implementation = self.prev
        return False


@torch.no_grad()
def question_attention(model, keys, values, query_ids, n_positions):
    """Mean attention the question pays to each cache position.

    The pass that produces this appends the question to a copy of the cache and
    the copy is dropped on return. Only the columns belonging to the document
    are read; the question's attention to itself says nothing about which
    document positions matter.
    """
    before = fingerprint(keys, values)
    dev = keys[0].device
    n = query_ids.shape[1]
    with eager_attention(model):
        out = model(
            input_ids=query_ids,
            attention_mask=torch.ones(1, n_positions + n, dtype=torch.long, device=dev),
            position_ids=torch.arange(
                n_positions, n_positions + n, device=dev
            ).unsqueeze(0),
            past_key_values=make_cache(keys, values),
            use_cache=True,
            output_attentions=True,
        )
    if out.attentions is None or out.attentions[0] is None:
        raise SystemExit(
            "INVALID: the model returned no attention weights, so an "
            "attention selector cannot be computed. Load with an "
            "implementation that reports them rather than ranking on "
            "something else and calling it attention."
        )
    acc = torch.zeros(n_positions, dtype=torch.float32, device=dev)
    for a in out.attentions:
        acc += a[0, :, :, :n_positions].float().mean(dim=(0, 1))
    acc /= len(out.attentions)
    if fingerprint(keys, values) != before:
        raise SystemExit("INVALID: the selector pass modified the cache it read")
    return acc


def select(kind, k, *, model, mapped, native, query_ids, n_positions):
    """Return the chosen block indices for one selector at one budget."""
    if kind == "uniform":
        return uniform_blocks(n_positions, k), None
    if kind == "native_attn":
        if native is None:
            raise ValueError(
                "the native-attention selector needs the receiver's own cache; "
                "it is an oracle and has none of its own"
            )
        src = native
    elif kind == "mapped_attn":
        src = mapped
    else:
        raise ValueError(f"unknown selector {kind!r}")
    a = question_attention(model, src[0], src[1], query_ids, n_positions)
    return rank_blocks_on_device(a, n_positions, k), a


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy(model, keys, values, query_ids, prompt_len, max_new, eos_ids):
    """Greedy continuation on top of a supplied cache.

    ``keys`` may be None for the no-cache control, in which case the query
    starts at position zero: telling the model a prefix exists that it cannot
    read would make the floor artificially low.
    """
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


def score(kind, tok, doc, produced_ids, eos_ids, max_new):
    text = tok.decode(produced_ids, skip_special_tokens=True)
    if kind == "retrieval":
        g = grade_retrieval(doc["retrieval_gold"], text)
        return {"primary": g["correct"], **g}
    if kind == "cloze":
        g = grade_cloze(doc["cloze_gold"], text)
        return {"primary": g["overlap"], **g}
    g = grade_free(text, produced_ids, eos_ids, max_new)
    return {"primary": g["healthy"], **g}


# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------

REPAIRS = ("none", "oracle", "replay", "native", "empty", "span")
SELECTORS = ("uniform", "native_attn", "mapped_attn", "lexical")


def check_conditions(defs, matrix):
    """Reject a condition file before a card is rented for it.

    Every name used by the matrix must be defined, every definition must name
    a repair that exists, and a repair that consumes a budget must state one.
    A condition named ``uniform_25`` that silently repaired nothing would look
    exactly like a negative result.
    """
    named = {c for cs in matrix.values() for c in cs}
    undefined = sorted(named - set(defs))
    if undefined:
        raise SystemExit(f"INVALID: matrix names undefined conditions {undefined}")
    unused = sorted(set(defs) - named)
    for name, spec in defs.items():
        r = spec.get("repair")
        if r not in REPAIRS:
            raise SystemExit(f"INVALID: {name}: repair {r!r} not in {REPAIRS}")
        if r in ("oracle", "replay", "span"):
            if not isinstance(spec.get("blocks"), int) or spec["blocks"] <= 0:
                raise SystemExit(f"INVALID: {name}: repair {r!r} needs a block count")
            sel = spec.get("selector")
            if sel not in SELECTORS:
                raise SystemExit(
                    f"INVALID: {name}: selector {sel!r} not in {SELECTORS}"
                )
            if r == "span" and sel != "lexical":
                raise SystemExit(
                    f"INVALID: {name}: the raw-text span baseline selects on the "
                    "text, not on a cache it does not have"
                )
        if spec.get("source", "own") not in ("own", "wrong"):
            raise SystemExit(f"INVALID: {name}: source must be 'own' or 'wrong'")
        if spec.get("oracle") and r == "replay":
            raise SystemExit(f"INVALID: {name}: a replay repair is not an oracle")
    return unused


def is_oracle(spec):
    """Does this condition's *score* borrow something unavailable at serving?

    A full native prefill is in here because it is the quality ceiling, not
    because it is unobtainable: a serving system can always just prefill the
    document. That distinction is the whole point of :func:`is_cost_oracle`.
    """
    return (
        spec["repair"] in ("oracle", "native")
        or spec.get("selector") == "native_attn"
        or bool(spec.get("oracle"))
    )


def is_cost_oracle(spec):
    """Is this condition's *cost* unquotable?

    Narrower than :func:`is_oracle`, and deliberately. Donor substitution
    installs a cache the receiver would have had to prefill anyway, so its
    cost is the native prefill plus the repair and quoting it as a saving
    would be quoting an operation nobody can perform. Ranking on the
    receiver's own cache has the same problem. But a plain native prefill is
    exactly what a serving system does today: it is the baseline every other
    condition is measured against, so it must be timed, not skipped. Treating
    it as a cost oracle left the comparison with no denominator at all.
    """
    return (
        spec["repair"] == "oracle"
        or spec.get("selector") == "native_attn"
        or bool(spec.get("oracle"))
    )


# ---------------------------------------------------------------------------
# the raw-text baseline
# ---------------------------------------------------------------------------


def lexical_span(tok, prompt_ids, query, k, block=BLOCK):
    """The contiguous window of ``k`` blocks whose text best matches the query.

    This is the baseline a translated cache has to beat on its own economics:
    no source model, no map, just read the part of the document the question
    looks like and prefill that natively. It is charged for its own selection,
    which is why the scoring is done on text rather than on an oracle span.
    """
    import re

    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    n = len(prompt_ids) // block
    per = np.zeros(n)
    for b in range(n):
        seg = tok.decode(prompt_ids[b * block : (b + 1) * block]).lower()
        per[b] = sum(1 for w in re.findall(r"[a-z0-9]+", seg) if w in words)
    if k >= n:
        return 0, n
    cs = np.concatenate([[0.0], np.cumsum(per)])
    tot = cs[k:] - cs[:-k] if k < len(cs) else cs[-1:] - cs[:1]
    start = int(np.argmax(tot))
    return start, start + k


# ---------------------------------------------------------------------------
# building the cache one condition reads
# ---------------------------------------------------------------------------


class _Hashable(dict):
    def __hash__(self):
        return hash(json.dumps(self, sort_keys=True))


def spec_of(defs, name):
    return _Hashable(defs[name])


def build_cache(
    spec,
    *,
    arm,
    doc,
    ids,
    qids,
    mapped,
    wrong_mapped,
    native,
    models,
    tg,
    dtype,
    contract,
    tok,
    query,
    ctx,
):
    """Produce the cache a condition answers from, and say how it was made.

    The returned ``prompt_len`` is the number of positions the question is
    told precede it. It is the full context for every condition that holds a
    full-length cache, and it is the span length for the raw-text baseline,
    which really does have a shorter prefix. Reporting the full length there
    would place the question after positions the model cannot see.
    """
    repair = spec["repair"]
    info = {"prompt_len": ctx}

    if repair == "empty":
        return None, None, {"prompt_len": 0, "replayed_positions": 0}

    if repair == "native":
        contract.check_pair(native[0], native[1], "native condition")
        return native[0], native[1], {**info, "replayed_positions": ctx}

    if repair == "span":
        k = spec["blocks"]
        a, b = lexical_span(tok, doc["prompt_ids"], query, k)
        seg = ids[:, a * BLOCK : b * BLOCK]
        sp = prefill(models["target"], seg, tg)
        return (
            [x.to(dtype) for x in sp.keys_post],
            [v.to(dtype) for v in sp.values],
            {
                "prompt_len": int(seg.shape[1]),
                "span_blocks": [a, b],
                "replayed_positions": int(seg.shape[1]),
            },
        )

    base = (wrong_mapped if spec.get("source", "own") == "wrong" else mapped)[arm]
    if repair == "none":
        return base[0], base[1], {**info, "replayed_positions": 0}

    k = spec["blocks"]
    contract.check_fraction(k * BLOCK / ctx)
    blocks, scores = select(
        spec["selector"],
        k,
        model=models["target"],
        mapped=base,
        native=native,
        query_ids=qids,
        n_positions=ctx,
    )
    # The selection is checked before it is used, not after the repair has
    # already read it: a repeated or out-of-range block silently repairs the
    # same positions twice and reports a budget it never spent.
    if scores is not None:
        contract.check_scores(scores)
    contract.check_blocks(blocks)
    mask = mask_from_blocks(ctx, blocks).to(base[0][0].device)
    contract.check_mask(mask)
    info["selected_blocks"] = sorted(int(b) for b in blocks)
    info["replayed_positions"] = int(mask.sum())
    if info["replayed_positions"] != min(k, ctx // BLOCK) * BLOCK:
        raise RuntimeError(
            f"budget was {k} blocks but the mask covers "
            f"{info['replayed_positions']} positions"
        )

    if repair == "oracle":
        keys = repair_cache(base[0], native[0], mask)
        values = repair_cache(base[1], native[1], mask)
    else:
        keys, values, rinfo = replay_selected(
            models["target"], ids, mask, base[0], base[1]
        )
        info["layers_replayed"] = len(rinfo["layers_written"])
        if rinfo["replayed"] and info["layers_replayed"] != tg.n_layers:
            raise RuntimeError(
                f"replay wrote {info['layers_replayed']} of {tg.n_layers} layers; "
                "a partially repaired cache is not the condition being scored"
            )
    contract.check_pair(keys, values, f"repaired {spec['repair']}")
    return keys, values, info


# ---------------------------------------------------------------------------
# the complete path, timed
# ---------------------------------------------------------------------------


def time_conditions(
    docs,
    *,
    defs,
    matrix,
    models,
    tok,
    sg,
    tg,
    dtype,
    dev,
    ctx,
    translate,
    default_arm,
    contract,
    by_doc,
    reps,
):
    """Question-to-first-answer wall time for every deployable condition.

    The boundary is the one the parent plan asked for and the one the earlier
    round got wrong. It does not begin at a resident cache: a translated cache
    does not exist until the source model has read the document and the map
    has been applied, and a system that must answer a question about a
    document it has just been handed pays for that. So the timer opens on the
    raw tokens and closes on the first answer token, for every arm.

    Oracle conditions are not timed. Their cost is the native prefill they
    borrow from plus the repair, which is strictly worse than native, and
    quoting it as a saving would be quoting the cost of an operation that
    cannot be performed.
    """
    names = sorted({c for cs in matrix.values() for c in cs})
    out = {
        "boundary": (
            "raw document token ids and the question in hand, to the first "
            "answer token; source prefill, feature extraction, map "
            "application, re-rotation, selection and repair are all inside"
        ),
        "query_kind": "retrieval",
        "documents": [d["doc"] for d in docs],
        "reps": reps,
        "conditions": {},
    }
    if dev != "cuda":
        out["skipped"] = "no device timer available on this host"
        return out
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def measure(fn):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            ev[0].record()
            fn()
            ev[1].record()
            torch.cuda.synchronize()
            ts.append(ev[0].elapsed_time(ev[1]))
        return ts

    for name in names:
        spec = defs[name]
        if is_cost_oracle(spec):
            out["conditions"][name] = {
                "timed": False,
                "reason": (
                    "the cost of this condition is not available at serving "
                    "time: it installs or ranks on a cache the receiver would "
                    "have had to prefill anyway"
                ),
            }
            continue
        samples = []
        for d in docs:
            ids = torch.tensor(d["prompt_ids"], dtype=torch.long, device=dev).unsqueeze(
                0
            )
            qids = tok(d["retrieval_query"], return_tensors="pt").input_ids.to(dev)
            fn = _path_for(
                spec,
                ids=ids,
                qids=qids,
                query=d["retrieval_query"],
                doc=d,
                arm=spec.get("arm", default_arm),
                models=models,
                tok=tok,
                sg=sg,
                tg=tg,
                dtype=dtype,
                dev=dev,
                ctx=ctx,
                translate=translate,
                by_doc=by_doc,
            )
            samples.extend(measure(fn))
        out["conditions"][name] = {
            "timed": True,
            "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)),
            "n": len(samples),
        }
    nat = out["conditions"].get("target_native")
    if nat and nat.get("timed"):
        for name, c in out["conditions"].items():
            if c.get("timed"):
                c["ratio_p50_over_native_p50"] = c["p50_ms"] / nat["p50_ms"]
                c["ratio_p95_over_native_p50"] = c["p95_ms"] / nat["p50_ms"]
    return out


def _path_for(
    spec,
    *,
    ids,
    qids,
    query,
    doc,
    arm,
    models,
    tok,
    sg,
    tg,
    dtype,
    dev,
    ctx,
    translate,
    by_doc,
):
    """One closure that performs a condition end to end, for the timer."""
    repair = spec["repair"]
    tgt = models["target"]

    @torch.no_grad()
    def native():
        return tgt(input_ids=torch.cat([ids, qids], dim=1), logits_to_keep=1).logits

    @torch.no_grad()
    def empty():
        return tgt(input_ids=qids, logits_to_keep=1).logits

    @torch.no_grad()
    def span():
        k = spec["blocks"]
        a, b = lexical_span(tok, doc["prompt_ids"], query, k)
        seg = ids[:, a * BLOCK : b * BLOCK]
        return tgt(input_ids=torch.cat([seg, qids], dim=1), logits_to_keep=1).logits

    @torch.no_grad()
    def mapped_path():
        src = by_doc[doc["wrong_source_doc"]] if spec.get("source") == "wrong" else doc
        sids = (
            ids
            if src is doc
            else torch.tensor(
                src["prompt_ids"], dtype=torch.long, device=dev
            ).unsqueeze(0)
        )
        keys, values = translate(arm, prefill(models["source"], sids, sg))
        if repair == "replay":
            blocks, _ = select(
                spec["selector"],
                spec["blocks"],
                model=tgt,
                mapped=(keys, values),
                native=None,
                query_ids=qids,
                n_positions=ctx,
            )
            mask = mask_from_blocks(ctx, blocks).to(keys[0].device)
            keys, values, _ = replay_selected(tgt, ids, mask, keys, values)
        n = qids.shape[1]
        return tgt(
            input_ids=qids,
            attention_mask=torch.ones(1, ctx + n, dtype=torch.long, device=dev),
            position_ids=torch.arange(ctx, ctx + n, device=dev).unsqueeze(0),
            past_key_values=make_cache(keys, values),
            use_cache=True,
        ).logits

    return {"native": native, "empty": empty, "span": span}.get(repair, mapped_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--source-revision", default="main")
    ap.add_argument("--target-revision", default="main")
    ap.add_argument("--manifest", required=True, help="the v2 fixture")
    ap.add_argument(
        "--conditions",
        required=True,
        help="JSON with {'definitions': {...}, 'matrix': {kind: [names]}}",
    )
    ap.add_argument(
        "--arm",
        action="append",
        default=[],
        required=True,
        help="name=path.pt, repeatable; conditions name an arm or use the first",
    )
    ap.add_argument("--docs", default="all", choices=("all", "screen"))
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--max-new-free", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--solve-dtype", default="float32")
    ap.add_argument(
        "--time-docs",
        type=int,
        default=0,
        help="measure the complete question-to-first-answer path on this many "
        "documents; 0 skips timing. Scores are unaffected either way.",
    )
    ap.add_argument("--time-reps", type=int, default=5)
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="recompute cells a previous run recorded as failed. Without this "
        "a resumed run preserves them, so an interruption cannot quietly "
        "turn an error into a score.",
    )
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    sdtype = getattr(torch, args.solve_dtype)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    cfg = json.load(open(args.conditions))
    defs, matrix = cfg["definitions"], cfg["matrix"]
    unused = check_conditions(defs, matrix)
    if unused:
        log(f"note: {len(unused)} defined conditions are not in the matrix: {unused}")

    man = json.load(open(args.manifest))
    if man.get("ctx") != args.ctx:
        raise SystemExit(
            f"INVALID: the fixture is {man.get('ctx')} tokens and the run asked "
            f"for {args.ctx}. A repair budget in blocks means a different "
            "fraction of a different length."
        )
    docs = [d for d in man["documents"] if args.docs == "all" or d["in_screen"]]
    docs.sort(key=lambda d: d["dev_index"])
    by_doc = {d["doc"]: d for d in man["documents"]}
    log(f"{len(docs)} documents, fixture {man['version']}, ctx {man['ctx']}")

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

    arms = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        mk, mv, amn = freeze.load(path, layout, tg, device=dev)
        arms[name] = (mk.cast(sdtype), mv.cast(sdtype), amn)
        log(f"arm {name}: joint {amn['joint_weight_sha256'][:16]} from {path}")
    default_arm = args.arm[0].split("=", 1)[0]
    # A condition naming an arm nobody passed fails deep inside the document
    # loop, after a card has already prefilled a document for it.
    want_arms = {defs[c].get("arm", default_arm) for cs in matrix.values() for c in cs}
    absent = sorted(want_arms - set(arms))
    if absent:
        raise SystemExit(
            f"INVALID: the matrix names arms that were not supplied: {absent}"
        )

    contract = CacheContract(
        n_layers=tg.n_layers,
        n_kv_heads=tg.n_kv_heads,
        n_positions=args.ctx,
        head_dim=tg.head_dim,
        dtype=dtype,
        device=torch.device(dev),
    )

    def translate(arm, sp):
        mk, mv, _ = arms[arm]
        xk = flat_features(to_content_keys(sp)).to(sdtype)
        xv = flat_features([v.float() for v in sp.values]).to(sdtype)
        keys = [
            rerot(b[0].float(), pos, tg.rope_theta).unsqueeze(0).to(dtype)
            for b in mk.apply(xk)
        ]
        return keys, [v.to(dtype) for v in mv.apply(xv)]

    # ---- resume ---------------------------------------------------------
    rows_path = os.path.join(args.out_dir, "rows.jsonl")
    done, failed = set(), set()
    if os.path.exists(rows_path):
        keep = []
        raw = [ln.strip() for ln in open(rows_path)]
        raw = [ln for ln in raw if ln]
        for i, line in enumerate(raw):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # A process killed mid-append leaves a partial line, and only
                # ever the last one. Dropping it is recovery; dropping one in
                # the middle would be hiding corruption, so that is an error.
                if i == len(raw) - 1:
                    log("dropping a torn final row from an interrupted run")
                    continue
                raise SystemExit(
                    f"INVALID: {rows_path} line {i + 1} is not JSON and is not "
                    "the last line, so it was not a torn append"
                )
            key = (r["kind"], r["condition"], r["doc"])
            if key in done or key in failed:
                continue  # a duplicate from an interrupted append; drop one
            if r.get("error"):
                if args.retry_failed:
                    continue
                failed.add(key)
            else:
                done.add(key)
            keep.append(line)
        with open(rows_path, "w") as fh:
            fh.write("\n".join(keep) + ("\n" if keep else ""))
        log(
            f"resuming: {len(done)} cells scored, {len(failed)} preserved failures"
            + (", retrying failures" if args.retry_failed else "")
        )
    out = open(rows_path, "a")
    # The scorer is told which cells must exist, and it is told by a file.
    # Writing it here rather than extracting it by hand later means the set
    # the scorer demands is the set this run was configured to produce.
    json.dump(matrix, open(os.path.join(args.out_dir, "matrix.json"), "w"), indent=1)
    # And which documents. The scorer demands a cell for every document in the
    # manifest it is handed, which is right, so handing it the whole fixture
    # for a sixteen-document screen makes a complete stage look incomplete.
    # The scope is decided here, from the frozen fixture and the declared
    # selection, before a single score exists -- it is a declaration of what
    # this stage will cover, not a record of what it managed to finish.
    json.dump(
        {
            "contract": "recovery_stage_scope_v1",
            "derived_from": os.path.abspath(args.manifest),
            "version": man["version"],
            "ctx": man["ctx"],
            "docs_scope": args.docs,
            "task_joint_sha256": man.get("task_joint_sha256"),
            "documents": docs,
        },
        open(os.path.join(args.out_dir, "scope.json"), "w"),
    )

    def emit(row):
        out.write(json.dumps(row) + "\n")
        out.flush()

    # ---- the per-document loop ------------------------------------------
    t0 = time.time()
    timing = {}
    for di, d in enumerate(docs):
        doc = d["doc"]
        wanted = [
            (kind, c)
            for kind, cs in matrix.items()
            for c in cs
            if (kind, c, doc) not in done and (kind, c, doc) not in failed
        ]
        if not wanted:
            continue
        ids = torch.tensor(d["prompt_ids"], dtype=torch.long, device=dev).unsqueeze(0)
        h = hashlib.sha256(
            np.asarray(d["prompt_ids"], dtype="<i8").tobytes()
        ).hexdigest()
        if h != d["prompt_ids_sha256_le_i64"] or ids.shape[1] != args.ctx:
            raise SystemExit(f"INVALID: {doc} does not match its recorded prompt")

        need = {spec_of(defs, c) for _, c in wanted}
        need_own = any(s.get("source", "own") == "own" for s in need)
        need_wrong = any(s.get("source", "own") == "wrong" for s in need)
        need_native = any(
            s["repair"] in ("oracle", "native") or s.get("selector") == "native_attn"
            for s in need
        )

        mapped, wrong_mapped, native = {}, {}, None
        if need_own or need_wrong:
            sp = prefill(models["source"], ids, sg)
        for _, c in wanted:
            s = defs[c]
            arm = s.get("arm", default_arm)
            if s["repair"] in ("native", "empty", "span"):
                continue
            if s.get("source", "own") == "own":
                if arm not in mapped:
                    mapped[arm] = translate(arm, sp)
            elif arm not in wrong_mapped:
                w = by_doc[d["wrong_source_doc"]]
                wids = torch.tensor(
                    w["prompt_ids"], dtype=torch.long, device=dev
                ).unsqueeze(0)
                wrong_mapped[arm] = translate(arm, prefill(models["source"], wids, sg))
        if need_native:
            tp = prefill(models["target"], ids, tg)
            native = (
                [k.to(dtype) for k in tp.keys_post],
                [v.to(dtype) for v in tp.values],
            )
            contract.check_pair(native[0], native[1], f"{doc} native donor")
        for arm, kv in list(mapped.items()) + list(wrong_mapped.items()):
            contract.check_pair(kv[0], kv[1], f"{doc} mapped {arm}")

        for kind, cname in wanted:
            s = defs[cname]
            arm = s.get("arm", default_arm)
            query = d[f"{kind}_query"] if kind != "free" else d["free_query"]
            qids = tok(query, return_tensors="pt").input_ids.to(dev)
            mx = args.max_new_free if kind == "free" else args.max_new
            row = {
                "kind": kind,
                "condition": cname,
                "doc": doc,
                "manifest_version": man["version"],
                "arm": arm,
                "repair": s["repair"],
                "selector": s.get("selector"),
                "blocks": s.get("blocks"),
                "source": s.get("source", "own"),
                "oracle": is_oracle(s),
                "dev_index": d["dev_index"],
                "quartile": d["quartile"],
            }
            try:
                keys, values, info = build_cache(
                    s,
                    arm=arm,
                    doc=d,
                    ids=ids,
                    qids=qids,
                    mapped=mapped,
                    wrong_mapped=wrong_mapped,
                    native=native,
                    models=models,
                    tg=tg,
                    dtype=dtype,
                    contract=contract,
                    tok=tok,
                    query=query,
                    ctx=args.ctx,
                )
                row.update(info)
                got = greedy(
                    models["target"],
                    keys,
                    values,
                    qids,
                    info.get("prompt_len", args.ctx),
                    mx,
                    eos_ids,
                )
                row.update(score(kind, tok, d, got, eos_ids, mx))
                row["produced"] = tok.decode(got, skip_special_tokens=True)[:200]
            except Exception as e:  # noqa: BLE001 -- recorded, never swallowed
                row["error"] = f"{type(e).__name__}: {e}"
                log(f"  FAILED {kind}/{cname}/{doc}: {row['error']}")
            emit(row)
        if (di + 1) % 4 == 0 or di + 1 == len(docs):
            log(f"{di + 1}/{len(docs)} documents, {time.time() - t0:.0f}s")
    out.close()

    # ---- the complete question-to-first-answer path ----------------------
    # Not first-next-token on a cache that is already resident. Every cost a
    # serving system would pay to answer this question from raw document
    # tokens is inside the timer, source prefill and selection included, for
    # every condition including the ones that pay nothing.
    if args.time_docs > 0:
        timing = time_conditions(
            docs[: args.time_docs],
            defs=defs,
            matrix=matrix,
            models=models,
            tok=tok,
            sg=sg,
            tg=tg,
            dtype=dtype,
            dev=dev,
            ctx=args.ctx,
            translate=translate,
            default_arm=default_arm,
            contract=contract,
            by_doc=by_doc,
            reps=args.time_reps,
        )
        json.dump(
            timing, open(os.path.join(args.out_dir, "timing.json"), "w"), indent=1
        )

    receipt = {
        "contract": "kv_recovery_runner_v1",
        "manifest": os.path.abspath(args.manifest),
        "manifest_version": man["version"],
        "manifest_task_joint_sha256": man.get("task_joint_sha256"),
        "conditions": os.path.abspath(args.conditions),
        "documents": len(docs),
        "docs_scope": args.docs,
        "ctx": args.ctx,
        "matrix": matrix,
        "definitions": defs,
        "arms": {
            n: {"joint_weight_sha256": a[2]["joint_weight_sha256"]}
            for n, a in arms.items()
        },
        "models": {
            "source": {"id": args.source, "revision": args.source_revision},
            "target": {"id": args.target, "revision": args.target_revision},
        },
        "dtype": args.dtype,
        "solve_dtype": args.solve_dtype,
        "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "torch": torch.__version__,
        "block_tokens": BLOCK,
        "elapsed_s": time.time() - t0,
        "timing_documents": args.time_docs,
    }
    json.dump(receipt, open(os.path.join(args.out_dir, "RECEIPT.json"), "w"), indent=1)
    n_err = sum(1 for line in open(rows_path) if '"error"' in line)
    log(f"done in {time.time() - t0:.0f}s, {n_err} failed cells, rows at {rows_path}")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
