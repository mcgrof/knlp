#!/usr/bin/env python3
"""Score each expert pool on its own domain and on the other one (GPU).

A prompt-driven selector that picks the wrong pool runs a request on
experts calibrated for another domain. This script measures what that
costs, on two domains: ARC-Challenge (grade-school science, multiple
choice) and GSM8K (grade-school math word problems). Each request set is
scored with the full model, with the ARC pool and with the GSM8K pool, so
both the right-pool and the wrong-pool cells are measured against the
same full-model baseline.

ARC items are scored as in score_screen.py: for each choice, the summed
log probability of the choice given the context; the primary metric picks
the highest log probability per choice character, the secondary the raw
sum. GSM8K items are answered by greedy decoding of up to --max-new
tokens, batch one. The completion is cut at the first blank line that
follows a "####" marker, or at the first "\\n\\nQuestion:", whichever is
first; the primary (strict) answer is the number after the first "####"
in the cut text, as in the dataset's reference format, and the secondary
(flexible) answer is the last number in the cut text. Numbers are
compared after removing commas, a leading "$" and a trailing ".".

All three models run the fused MoE kernel (fused_moe.py) with the tuned
tile table. The model is reloaded before each pool is applied, since
applying a pool frees the other experts.

Usage:
    pool_quality.py MODEL_DIR ARC_SCREEN.json GSM8K_REQUESTS.json \\
        POOL_ARC.json POOL_GSM8K.json OUT_DIR --configs TUNED.json \\
        [--max-new 256]
"""

import argparse
import gc
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import apply_pool, load_model, runtime_identity, write_json  # noqa
from score_screen import score_item  # noqa

NUM = re.compile(r"-?[\d,]*\.?\d+")


def norm(s):
    s = s.strip().replace(",", "").lstrip("$").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return s
    return str(int(v)) if v == int(v) else str(v)


def cut(text):
    """The answer part of a completion, as described in the docstring."""
    ends = []
    q = text.find("\n\nQuestion:")
    if q >= 0:
        ends.append(q)
    h = text.find("####")
    if h >= 0:
        b = text.find("\n\n", h)
        if b >= 0:
            ends.append(b)
    return text[: min(ends)] if ends else text


def extract(text):
    c = cut(text)
    strict = None
    h = c.find("####")
    if h >= 0:
        m = NUM.search(c[h + 4 :])
        if m:
            strict = norm(m.group(0))
    nums = NUM.findall(c)
    flexible = norm(nums[-1]) if nums else None
    return c, strict, flexible


def reference(answer):
    return norm(answer.split("####")[-1])


@torch.no_grad()
def generate(model, tok, prompt, max_new, eos):
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids], device="cuda")
    out = model(input_ids=x, use_cache=True)
    past = out.past_key_values
    nxt = int(out.logits[0, -1].argmax())
    toks = [nxt]
    for step in range(max_new - 1):
        if nxt == eos:
            break
        if step % 8 == 7:
            text = tok.decode(toks)
            if cut(text) != text:
                break
        o = model(
            input_ids=torch.tensor([[nxt]], device="cuda"),
            past_key_values=past,
            use_cache=True,
        )
        past = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
        toks.append(nxt)
    return toks, len(ids)


def score_gsm8k(model, tok, items, max_new, eos, tag):
    t0 = time.time()
    rows = []
    for it in items:
        toks, n_prompt = generate(model, tok, it["context"], max_new, eos)
        text = tok.decode(toks)
        c, strict, flexible = extract(text)
        ref = reference(it["answer"])
        rows.append(
            {
                "id": it["id"],
                "prompt_tokens": n_prompt,
                "new_tokens": len(toks),
                "completion": c,
                "ref": ref,
                "strict": strict,
                "flexible": flexible,
                "correct_strict": strict == ref,
                "correct_flexible": flexible == ref,
            }
        )
    el = time.time() - t0
    acc = sum(r["correct_strict"] for r in rows)
    print(f"{tag} gsm8k: {acc}/{len(rows)} strict, {el:.0f} s", flush=True)
    return rows, el


def score_arc(model, tok, items, tag):
    t0 = time.time()
    rows = [score_item(model, tok, it) for it in items]
    el = time.time() - t0
    acc = sum(r["correct_per_char"] for r in rows)
    print(f"{tag} arc: {acc}/{len(rows)} per-char, {el:.0f} s", flush=True)
    return rows, el


def paired(base, other, key):
    assert [r["id"] for r in base] == [r["id"] for r in other]
    lost = sum(1 for a, b in zip(base, other) if a[key] and not b[key])
    won = sum(1 for a, b in zip(base, other) if b[key] and not a[key])
    return {
        "base_correct": sum(r[key] for r in base),
        "correct": sum(r[key] for r in other),
        "base_only": lost,
        "other_only": won,
        "net_losses": lost - won,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("arc_screen")
    ap.add_argument("gsm8k_requests")
    ap.add_argument("pool_arc")
    ap.add_argument("pool_gsm8k")
    ap.add_argument("out_dir")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from fused_moe import ConfigTable, install_fused

    with open(args.arc_screen) as f:
        arc = json.load(f)["items"]
    with open(args.gsm8k_requests) as f:
        gsm = json.load(f)["items"]
    pools = {}
    for name, p in (("arc-pool", args.pool_arc), ("gsm8k-pool", args.pool_gsm8k)):
        with open(p) as f:
            pools[name] = json.load(f)
    res = {"runtime": runtime_identity(), "args": vars(args), "models": {}}
    out_path = os.path.join(args.out_dir, "pool-quality.json")

    for name in ("full", "arc-pool", "gsm8k-pool"):
        model, tok = load_model(args.model_dir)
        if name != "full":
            apply_pool(model, pools[name])
        install_fused(model, ConfigTable(args.configs))
        gc.collect()
        torch.cuda.empty_cache()
        a_rows, a_s = score_arc(model, tok, arc, name)
        g_rows, g_s = score_gsm8k(model, tok, gsm, args.max_new, tok.eos_token_id, name)
        res["models"][name] = {
            "arc": a_rows,
            "arc_s": a_s,
            "gsm8k": g_rows,
            "gsm8k_s": g_s,
        }
        write_json(out_path, res)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    full = res["models"]["full"]
    summ = {}
    for name in ("arc-pool", "gsm8k-pool"):
        m = res["models"][name]
        summ[name] = {
            "arc_per_char": paired(full["arc"], m["arc"], "correct_per_char"),
            "arc_raw": paired(full["arc"], m["arc"], "correct_raw"),
            "gsm8k_strict": paired(full["gsm8k"], m["gsm8k"], "correct_strict"),
            "gsm8k_flexible": paired(full["gsm8k"], m["gsm8k"], "correct_flexible"),
        }
    summ["full"] = {
        "arc_per_char": sum(r["correct_per_char"] for r in full["arc"]),
        "gsm8k_strict": sum(r["correct_strict"] for r in full["gsm8k"]),
        "gsm8k_flexible": sum(r["correct_flexible"] for r in full["gsm8k"]),
        "gsm8k_hit_max_new": sum(
            r["new_tokens"] >= args.max_new for r in full["gsm8k"]
        ),
    }
    res["summary"] = summ
    write_json(out_path, res)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
