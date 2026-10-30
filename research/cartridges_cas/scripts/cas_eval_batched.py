#!/usr/bin/env python3
"""Score a cartridge on the 20-question panel with batched generation.

``cas_eval_table15.py`` MODE=cart generates one answer at a time
through the library's FlexAttention decode loop, about 0.1 s per token
on a single GPU whatever the card, so a panel of 60 long answers takes
two to three hours and evaluation, not training, sets the pace of a
cartridge study. This script keeps the panel's protocol (its system
prompt, user prompt, chat template with thinking on, temperature 0.6
with top-k 20 then nucleus 0.95, the model's end-of-sequence ids, the
completion cap and the scoring by fuzzy option match) and changes only
how the answers are produced: the reader's standard attention runs a
batch of answers at once, with the cartridge preloaded as a key-value
prefix of every row.

The cartridge holds keys already rotated at positions 0 .. S-1 (the
frozen sink first). Each prompt is left-padded after the prefix, and
the attention mask covers the prefix and the real prompt tokens, so
each prompt token sits at position S + j as in the library, where the
conversation follows the cartridge. The random draws differ from the
one-at-a-time loop, so scores from the two evaluators are not paired
sample by sample; compare a candidate with controls scored by the same
evaluator.

``check`` mode computes the next-token logits after the first N panel
prompts and compares them with a ``cas_logit_parity.py`` dump of the
library path, so the prefix and position handling is verified before
any panel is trusted.

Env:
    CART           cartridge file (frozen sink + trainable layout)
    PATIENT        patient id (default patient_01)
    MAX_Q          questions (default 20)
    RUNS           repeats of the question set (default 3)
    MAX_COMPLETION completion cap in tokens (default 4096)
    BATCH          answers generated together (default 20)
    SEED           base seed (default 0); batch b uses SEED + b
    OUT_JSON       results; SAVE_RAW=1 also writes <OUT_JSON>.raw.json
    MODEL, DEVICE  reader and device (defaults Qwen/Qwen3-8B, cuda)
    PARITY_DUMP    (check mode) a cas_logit_parity.py dump to compare with
    N              (check mode) prompts (default 5)

Usage:
    python cas_eval_batched.py panel
    python cas_eval_batched.py check
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
DEVICE = os.environ.get("DEVICE", "cuda")
PATIENT = os.environ.get("PATIENT", "patient_01")


def load(tok_only=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok_only:
        return tok, None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(DEVICE)
    model.eval()
    return tok, model


def prompt_ids(tok, q):
    import cas_eval_table15 as ev

    return tok.apply_chat_template(
        [
            {"role": "system", "content": ev.SYSTEM_PROMPT},
            {"role": "user", "content": ev.user_prompt(q)},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def cart_kv(path):
    from cas_cart_init import load_cart

    keys, values, _ = load_cart(path, device=DEVICE)
    return keys, values


def prefix_cache(keys, values, B):
    from transformers import DynamicCache

    cache = DynamicCache()
    for l, (k, v) in enumerate(zip(keys, values)):
        cache.update(
            k.expand(B, -1, -1, -1).contiguous(),
            v.expand(B, -1, -1, -1).contiguous(),
            l,
        )
    return cache


def batch_inputs(tok, prompts, S):
    """Full-length ids and mask: S placeholder tokens for the cached
    prefix, then left padding, then each prompt."""
    T = max(len(p) for p in prompts)
    pad = tok.pad_token_id
    ids, mask = [], []
    for p in prompts:
        n = T - len(p)
        ids.append([pad] * (S + n) + list(p))
        mask.append([1] * S + [0] * n + [1] * len(p))
    return (
        torch.tensor(ids, device=DEVICE),
        torch.tensor(mask, device=DEVICE),
    )


def check():
    from cartridges.data.longhealth.utils import load_longhealth_dataset

    n = int(os.environ.get("N", "5"))
    ref = torch.load(os.environ["PARITY_DUMP"], weights_only=False)
    tok, model = load()
    keys, values = cart_kv(os.environ["CART"])
    S = keys[0].shape[2]
    pat = load_longhealth_dataset([PATIENT])[0]
    qs = pat.questions[:n]
    prompts = [prompt_ids(tok, q) for q in qs]
    ids, mask = batch_inputs(tok, prompts, S)
    cache = prefix_cache(keys, values, len(prompts))
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        out = model(
            input_ids=ids[:, S:],
            attention_mask=mask,
            position_ids=pos[:, S:],
            past_key_values=cache,
            use_cache=True,
        )
    for i, (q, r) in enumerate(zip(qs, ref["rows"])):
        assert r["qid"] == q.question_id and r["prompt_len"] == len(prompts[i])
        la = out.logits[i, -1].float().cpu()
        lb = r["logits"]
        pa, pb = torch.log_softmax(la, -1), torch.log_softmax(lb, -1)
        kl = float((pb.exp() * (pb - pa)).sum())
        print(
            f"BATCHED_CHECK,{q.question_id},maxabs={float((la - lb).abs().max()):.4f},"
            f"kl={kl:.2e},top1={int(la.argmax() == lb.argmax())}",
            flush=True,
        )


def panel():
    import cas_eval_table15 as ev
    from cartridges.data.longhealth.utils import load_longhealth_dataset

    cart = os.environ["CART"]
    max_q = int(os.environ.get("MAX_Q", "20"))
    runs = int(os.environ.get("RUNS", "3"))
    cap = int(os.environ.get("MAX_COMPLETION", "4096"))
    B = int(os.environ.get("BATCH", "20"))
    seed = int(os.environ.get("SEED", "0"))
    out_json = os.environ["OUT_JSON"]
    save_raw = os.environ.get("SAVE_RAW", "1") == "1"
    t0 = time.time()
    tok, model = load()
    keys, values = cart_kv(cart)
    S = keys[0].shape[2]
    eos = model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    pat = load_longhealth_dataset([PATIENT])[0]
    qs = pat.questions[:max_q]
    items = [(r, q) for r in range(runs) for q in qs]
    rows, n_gen = [], 0
    for b in range(0, len(items), B):
        chunk = items[b : b + B]
        prompts = [prompt_ids(tok, q) for _, q in chunk]
        ids, mask = batch_inputs(tok, prompts, S)
        cache = prefix_cache(keys, values, len(chunk))
        torch.manual_seed(seed + b // B)
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids,
                attention_mask=mask,
                past_key_values=cache,
                do_sample=True,
                temperature=0.6,
                top_k=ev.TOP_K,
                top_p=ev.TOP_P,
                max_new_tokens=cap,
                eos_token_id=eos,
                pad_token_id=tok.pad_token_id,
            )
        new = gen[:, ids.shape[1] :]
        for (r, q), g in zip(chunk, new.tolist()):
            g = [t for t in g if t != tok.pad_token_id]
            cut = next((i for i, t in enumerate(g) if t in eos), len(g))
            n_gen += cut
            text = tok.decode(g[:cut], skip_special_tokens=True)
            corr, tag = ev.score_response(text, q)
            rows.append(
                dict(
                    run=r,
                    patient=PATIENT,
                    qid=q.question_id,
                    correct=int(corr),
                    tag_found=bool(tag),
                    n_chars=len(text),
                    n_tokens=cut,
                    truncated="</think>" not in text,
                    text=text,
                )
            )
        del cache, gen
        torch.cuda.empty_cache()
        print(
            f"[batched] {min(b + B, len(items))}/{len(items)} answers "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    per_run = []
    for r in range(runs):
        rr = [x for x in rows if x["run"] == r]
        per_run.append(
            (
                sum(x["correct"] for x in rr),
                len(rr),
                sum(1 for x in rr if not x["tag_found"]),
            )
        )
    tagged = [x for x in rows if x["tag_found"]]
    untagged = [x for x in rows if not x["tag_found"]]
    res = dict(
        protocol="Table 15 panel, batched HF generation with the cartridge "
        f"as a prefix; temp 0.6, top-k {ev.TOP_K}, top-p {ev.TOP_P}, "
        f"max {cap}",
        evaluator="cas_eval_batched",
        model=MODEL,
        cart=os.path.abspath(cart),
        patient=PATIENT,
        max_q=max_q,
        batch=B,
        seed=seed,
        summary=ev.aggregate(per_run),
        tag_breakdown=dict(
            tagged_n=len(tagged),
            tagged_acc=(
                round(sum(x["correct"] for x in tagged) / len(tagged), 4)
                if tagged
                else None
            ),
            untagged_n=len(untagged),
            untagged_truncated=sum(1 for x in untagged if x["truncated"]),
        ),
        generated_tokens=n_gen,
        seconds=time.time() - t0,
        device=torch.cuda.get_device_name(),
    )
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    if save_raw:
        raw = out_json.replace(".json", ".raw.json")
        json.dump(rows, open(raw, "w"), indent=1)
        res["raw_path"] = raw
    json.dump(res, open(out_json, "w"), indent=2)
    print(
        f"BATCHED_PANEL_DONE acc={res['summary']['acc_mean']} "
        f"+/-{res['summary']['acc_std']} tokens={n_gen} "
        f"({res['seconds']:.0f}s) -> {out_json}",
        flush=True,
    )


if __name__ == "__main__":
    {"panel": panel, "check": check}[sys.argv[1]]()
