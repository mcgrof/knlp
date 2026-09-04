#!/usr/bin/env python3
"""Separate the completion cap's effect on accuracy from resampling noise.

The evaluator samples a completion under a token cap and scores the answer
tag that follows the model's thinking block. When the cap cuts the thinking
block short the answer tag is never written, so a truncated cell scores near
chance regardless of what the cartridge knows. Comparing two caps therefore
mixes two effects: cells that escaped truncation, and cells that were resampled
into different text because the run is not reproducible across machines.

This tool separates them. It stratifies each patient's cells on a pre-treatment
variable -- whether the cell hit the cap in the LOWER-cap run -- and reports the
two strata separately. The never-capped stratum is the control: those cells
could not have been helped by a longer budget, so whatever they do is the noise
floor, and any honest reading of a cap effect has to clear it.

Two measurement traps are handled explicitly. The evaluator's ``truncated``
field records whether the model closed ``</think>``, which is not the same as
running out of budget; on the stored data it disagrees with the real thing in
both directions. And ``n_chars`` is not a token count. So cap hits are measured
by re-tokenising the stored completion: the generation loop breaks on a stop
token before appending, so a cell that hit the cap has exactly ``cap`` content
tokens, an equality rather than a threshold. When no tokenizer is available the
tool falls back to ``tag_found``, which on every row measured so far is an exact
indicator of a cap hit, and it says in its output that it did so.

Usage:
    cas_cap_decomposition.py OUT.json DIR_OR_FILE [DIR_OR_FILE ...]

Env:
    TOKENIZER      model id for the token count (default Qwen/Qwen3-8B)
    TOKCACHE       token-count cache, keyed by file and row (default beside OUT)
    NO_TOKENIZE    set to 1 to skip tokenisation and use the tag_found proxy
"""

import json
import os
import re
import sys
from collections import defaultdict

TOKENIZER = os.environ.get("TOKENIZER", "Qwen/Qwen3-8B")
NO_TOKENIZE = os.environ.get("NO_TOKENIZE", "0") == "1"

CAP_RE = re.compile(r"max (\d+)")
# t15_<arm>_<patient>[_cap<n>]_runs<n>.json -- the arm names the cartridge, so
# two cartridges scored on the same patient at the same cap stay separate
ARM_RE = re.compile(r"^t15_(.+?)_(patient_\d+)(?:_cap\d+)?_runs\d+$")


def _load_tokenizer():
    if NO_TOKENIZE:
        return None
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(TOKENIZER)
    except Exception as exc:
        print(
            f"note: no tokenizer ({type(exc).__name__}); using the tag proxy",
            file=sys.stderr,
        )
        return None


def _discover(paths):
    """Summary files, ignoring the raws that sit beside them."""
    found = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if (
                    name.startswith("t15_")
                    and name.endswith(".json")
                    and not name.endswith(".raw.json")
                ):
                    found.append(os.path.join(p, name))
        elif p.endswith(".json") and not p.endswith(".raw.json"):
            found.append(p)
    return found


def _cap_of(summary):
    """The cap the run used, from the field if present, else the protocol."""
    if isinstance(summary.get("max_completion"), int):
        return summary["max_completion"], "field"
    m = CAP_RE.search(summary.get("protocol", "") or "")
    if m:
        return int(m.group(1)), "protocol"
    return None, "missing"


def _sampler_of(summary):
    if "top_k" in summary or "top_p" in summary:
        return f"top-k {summary.get('top_k')}, top-p {summary.get('top_p')}"
    m = re.search(r"top-k (\S+), top-p (\S+)", summary.get("protocol", "") or "")
    if m:
        return f"top-k {m.group(1)}, top-p {m.group(2)}"
    return "full vocabulary (no top-k/top-p)"


def _read(path, tok, cache):
    """One evaluation file as rows plus its metadata, or None if unusable."""
    summary = json.load(open(path))
    # Pair by filename stem. The summary's raw_path is the path on the machine
    # that produced it, which is a pod that no longer exists.
    raw_path = os.path.splitext(path)[0] + ".raw.json"
    if not os.path.exists(raw_path):
        return None
    raw = json.load(open(raw_path))
    rows = raw["rows"] if isinstance(raw, dict) else raw
    if not rows:
        return None
    m = ARM_RE.match(os.path.splitext(os.path.basename(path))[0])
    arm = m.group(1) if m else os.path.splitext(os.path.basename(path))[0]
    cap, cap_src = _cap_of(summary)
    if cap is None:
        print(
            f"note: {os.path.basename(path)} records no cap; skipped", file=sys.stderr
        )
        return None
    key = os.path.basename(path)
    counts = cache.setdefault(key, {})
    for i, r in enumerate(rows):
        rk = str(i)
        if tok is not None:
            if rk not in counts:
                counts[rk] = len(tok(r["text"], add_special_tokens=False)["input_ids"])
            r["_ntok"] = counts[rk]
            r["_cap_hit"] = r["_ntok"] >= cap
        else:
            r["_ntok"] = None
            r["_cap_hit"] = not r.get("tag_found", True)
        r["_tag_hit"] = not r.get("tag_found", True)
    return {
        "path": path,
        "file": key,
        "arm": arm,
        "cap": cap,
        "cap_source": cap_src,
        "sampler": _sampler_of(summary),
        "patients": summary.get("patients") or sorted({r["patient"] for r in rows}),
        "rows": rows,
        "acc": sum(bool(r["correct"]) for r in rows) / len(rows),
    }


def _field_disagreement(rows):
    """Where the stored truncated flag differs from a measured cap hit."""
    fp = sum(1 for r in rows if r.get("truncated") and not r["_cap_hit"])
    fn = sum(1 for r in rows if not r.get("truncated") and r["_cap_hit"])
    tag = sum(1 for r in rows if r["_tag_hit"] != r["_cap_hit"])
    return {
        "field_says_yes_but_no": fp,
        "field_says_no_but_yes": fn,
        "tag_proxy_mismatch": tag,
    }


def _acc(rows):
    return (sum(bool(r["correct"]) for r in rows) / len(rows)) if rows else None


def _decompose(lower, upper):
    """One cap transition for one patient, stratified on the lower-cap state.

    The stratifying variable is measured BEFORE the change, so the strata are
    fixed sets of cells and the two rows add back to the total. Cells still at
    the cap after the raise are reported separately: they say whether the upper
    cap is itself still binding, which decides whether the number can be read
    as cap-free.
    """
    key = lambda r: (r["patient"], r["run"], r["qid"])
    lo = {key(r): r for r in lower["rows"]}
    hi = {key(r): r for r in upper["rows"]}
    both = sorted(set(lo) & set(hi))
    if not both:
        return None
    hit = [k for k in both if lo[k]["_cap_hit"]]
    miss = [k for k in both if not lo[k]["_cap_hit"]]
    lo_rows, hi_rows = [lo[k] for k in both], [hi[k] for k in both]
    resampled = sum(1 for k in miss if lo[k]["text"] != hi[k]["text"])
    flipped = sum(1 for k in miss if bool(lo[k]["correct"]) != bool(hi[k]["correct"]))
    d_hit = (_acc([hi[k] for k in hit]) - _acc([lo[k] for k in hit])) if hit else None
    d_miss = (
        (_acc([hi[k] for k in miss]) - _acc([lo[k] for k in miss])) if miss else None
    )
    total = _acc(hi_rows) - _acc(lo_rows)
    share = (len(hit) / len(both) * d_hit) if hit else 0.0
    return {
        "n": len(both),
        "lower_cap": lower["cap"],
        "upper_cap": upper["cap"],
        "lower_only": len(set(lo) - set(hi)),
        "upper_only": len(set(hi) - set(lo)),
        "acc_lower": _acc(lo_rows),
        "acc_upper": _acc(hi_rows),
        "delta": total,
        "hit_cap": {
            "n": len(hit),
            "acc_lower": _acc([lo[k] for k in hit]) if hit else None,
            "acc_upper": _acc([hi[k] for k in hit]) if hit else None,
            "delta": d_hit,
        },
        "never_capped": {
            "n": len(miss),
            "acc_lower": _acc([lo[k] for k in miss]) if miss else None,
            "acc_upper": _acc([hi[k] for k in miss]) if miss else None,
            "delta": d_miss,
            "resampled_text": resampled,
            "flipped_correct": flipped,
        },
        "delta_from_escaping_cap": share,
        "still_at_upper_cap": sum(1 for k in both if hi[k]["_cap_hit"]),
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    out_path, inputs = sys.argv[1], sys.argv[2:]
    cache_path = os.environ.get(
        "TOKCACHE", os.path.splitext(out_path)[0] + ".tokcache.json"
    )
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    tok = _load_tokenizer()

    files = []
    for path in _discover(inputs):
        rec = _read(path, tok, cache)
        if rec:
            files.append(rec)
    if not files:
        print("no usable evaluation files found", file=sys.stderr)
        return 1
    json.dump(cache, open(cache_path, "w"))

    per_file = []
    by_patient_cap = defaultdict(dict)
    for f in files:
        for pt in f["patients"]:
            rows = [r for r in f["rows"] if r["patient"] == pt]
            if not rows:
                continue
            entry = {
                "file": f["file"],
                "arm": f["arm"],
                "patient": pt,
                "cap": f["cap"],
                "cap_source": f["cap_source"],
                "sampler": f["sampler"],
                "n": len(rows),
                "acc": _acc(rows),
                "cap_hits": sum(1 for r in rows if r["_cap_hit"]),
                "max_tokens": max(
                    (r["_ntok"] for r in rows if r["_ntok"] is not None), default=None
                ),
                "truncated_field_disagreement": _field_disagreement(rows),
            }
            per_file.append(entry)
            # a patient may be scored more than once at one cap; the sampler
            # distinguishes those runs, so key on both
            by_patient_cap[pt][(f["arm"], f["cap"], f["sampler"])] = {**f, "rows": rows}

    transitions = []
    for pt, runs in sorted(by_patient_cap.items()):
        # a transition is only meaningful within one cartridge and one sampler
        for arm, samp in sorted({(a, s) for a, _, s in runs}):
            caps = sorted(c for a, c, s in runs if a == arm and s == samp)
            for lo_cap, hi_cap in zip(caps, caps[1:]):
                d = _decompose(runs[(arm, lo_cap, samp)], runs[(arm, hi_cap, samp)])
                if d:
                    transitions.append(
                        {"patient": pt, "arm": arm, "sampler": samp, **d}
                    )

    unpaired = sorted(
        {e["patient"] for e in per_file} - {t["patient"] for t in transitions}
    )
    result = {
        "provenance": {
            "inputs": inputs,
            "tokenizer": None if tok is None else TOKENIZER,
            "cap_hit_rule": (
                "re-tokenised completion length equals the cap"
                if tok is not None
                else "answer tag missing (tokenizer unavailable)"
            ),
            "files_read": [f["path"] for f in files],
        },
        "per_run": per_file,
        "transitions": transitions,
        "not_decomposable": unpaired,
    }
    json.dump(result, open(out_path, "w"), indent=1)

    w = sys.stdout.write
    w(f"cap hits measured by {result['provenance']['cap_hit_rule']}\n\n")
    w(f"{'arm':<14}{'patient':<12}{'cap':>6}{'n':>5}{'acc':>8}{'at cap':>8}  sampler\n")
    for e in sorted(
        per_file, key=lambda e: (e["arm"], e["patient"], e["sampler"], e["cap"])
    ):
        w(
            f"{e['arm']:<14}{e['patient']:<12}{e['cap']:>6}{e['n']:>5}{e['acc']:>8.4f}"
            f"{e['cap_hits']:>5}/{e['n']:<3}  {e['sampler']}\n"
        )
    dis = [
        e
        for e in per_file
        if any(v for v in e["truncated_field_disagreement"].values())
    ]
    if dis:
        w("\nstored truncated field vs a measured cap hit:\n")
        for e in dis:
            d = e["truncated_field_disagreement"]
            w(
                f"  {e['patient']} cap {e['cap']}: {d['field_says_yes_but_no']} flagged but not at the cap, "
                f"{d['field_says_no_but_yes']} at the cap but not flagged, "
                f"{d['tag_proxy_mismatch']} tag-proxy mismatches\n"
            )
    if transitions:
        w("\ncap transitions, stratified on the lower cap:\n")
        for t in transitions:
            h, m = t["hit_cap"], t["never_capped"]
            w(
                f"\n  {t['patient']}  {t['lower_cap']} -> {t['upper_cap']}  "
                f"({t['sampler']})\n"
            )
            w(
                f"    overall      n={t['n']:<4} {t['acc_lower']:.4f} -> {t['acc_upper']:.4f}  "
                f"delta {t['delta']:+.4f}\n"
            )
            if h["n"]:
                w(
                    f"    hit the cap  n={h['n']:<4} {h['acc_lower']:.4f} -> {h['acc_upper']:.4f}  "
                    f"delta {h['delta']:+.4f}\n"
                )
            if m["n"]:
                w(
                    f"    never capped n={m['n']:<4} {m['acc_lower']:.4f} -> {m['acc_upper']:.4f}  "
                    f"delta {m['delta']:+.4f}   <- noise floor\n"
                )
                w(
                    f"                 {m['resampled_text']} of {m['n']} resampled into different "
                    f"text, {m['flipped_correct']} changed score\n"
                )
            w(f"    still at the upper cap: {t['still_at_upper_cap']} of {t['n']}\n")
    if unpaired:
        w(
            "\nno cap transition available for: "
            + ", ".join(f"{a}/{p}" for a, p in unpaired)
            + "\n"
        )
    w(f"\nwrote {out_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
