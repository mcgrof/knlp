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

Each answer records separately whether thinking closed (the
``</think>`` token was generated), whether an answer tag was found,
how many tokens were generated against the budget, and why generation
ended: ``stop`` at an end-of-sequence token (which one is kept) or
``length`` at the budget with none. The old ``truncated`` field is
kept for older readers and means only that no ``</think>`` appears in
the text; it is not a length flag.

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
                   with one row per answer (SAVE_TOKENS=1, the default,
                   keeps its generated token ids there)
    MODEL, DEVICE  reader and device (defaults Qwen/Qwen3-8B, cuda)
    TASKS          (tasks mode) task JSON; RUNS repeats it (default 1)
    PARITY_DUMP    (check mode) a cas_logit_parity.py dump to compare with
    N              (check mode) prompts (default 5)

``tasks`` mode answers short source-grounded questions (TASKS, a JSON
list with question, reference, required and forbidden fields) with the
same prefix and sampling, asking for a brief answer in <answer> tags.
The scored text is the last tagged answer after the thinking, or the
text after ``</think>`` when no tag is found (recorded). An answer is
correct when, lowercased with whitespace collapsed, it contains one
alternative from every ``required`` group and no ``forbidden`` string,
each matched as a word (see ``rubric_hit``).
Answers that never close their thinking score as wrong.

Usage:
    python cas_eval_batched.py panel
    python cas_eval_batched.py tasks
    python cas_eval_batched.py check
"""

import hashlib
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


def generate(tok, model, keys, values, prompts, cap, B, seed):
    """Sample one answer per prompt with the cartridge as prefix, B at
    a time; batch b uses seed SEED + b. Returns per-answer text and the
    separate completion facts."""
    import cas_eval_table15 as ev

    S = keys[0].shape[2]
    eos = model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    think_end = tok.convert_tokens_to_ids("</think>")
    t0, out = time.time(), []
    for b in range(0, len(prompts), B):
        chunk = prompts[b : b + B]
        ids, mask = batch_inputs(tok, chunk, S)
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
        for g in gen[:, ids.shape[1] :].tolist():
            # a finished row is padded after its end-of-sequence token,
            # and the pad id is itself one of the end ids, so the first
            # end id is where generation stopped
            cut = next((i for i, t in enumerate(g) if t in eos), len(g))
            stop_id = g[cut] if cut < len(g) else None
            g = g[:cut]
            text = tok.decode(g, skip_special_tokens=True)
            out.append(
                dict(
                    text=text,
                    n_tokens=cut,
                    budget=cap,
                    finish_reason="stop" if stop_id is not None else "length",
                    stop_token=stop_id,
                    think_closed=think_end in g,
                    token_ids=g,
                )
            )
        del cache, gen
        torch.cuda.empty_cache()
        print(
            f"[batched] {min(b + B, len(prompts))}/{len(prompts)} answers "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    return out


def completion_fields(g, save_tokens):
    rec = {k: g[k] for k in ("n_tokens", "budget", "finish_reason", "stop_token")}
    rec.update(
        think_closed=g["think_closed"],
        truncated="</think>" not in g["text"],
        text=g["text"],
    )
    if save_tokens:
        rec["token_ids"] = g["token_ids"]
    return rec


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
    save_tokens = os.environ.get("SAVE_TOKENS", "1") == "1"
    t0 = time.time()
    tok, model = load()
    keys, values = cart_kv(cart)
    pat = load_longhealth_dataset([PATIENT])[0]
    qs = pat.questions[:max_q]
    items = [(r, q) for r in range(runs) for q in qs]
    gens = generate(
        tok, model, keys, values, [prompt_ids(tok, q) for _, q in items], cap, B, seed
    )
    rows, n_gen = [], 0
    for (r, q), g in zip(items, gens):
        n_gen += g["n_tokens"]
        corr, tag = ev.score_response(g["text"], q)
        rows.append(
            dict(
                run=r,
                patient=PATIENT,
                qid=q.question_id,
                correct=int(corr),
                tag_found=bool(tag),
                n_chars=len(g["text"]),
                **completion_fields(g, save_tokens),
            )
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
            untagged_at_budget=sum(
                1 for x in untagged if x["finish_reason"] == "length"
            ),
        ),
        completion=dict(
            at_budget=sum(1 for x in rows if x["finish_reason"] == "length"),
            think_unclosed=sum(1 for x in rows if not x["think_closed"]),
            no_answer_tag=len(untagged),
            think_closed_at_budget=sum(
                1 for x in rows if x["finish_reason"] == "length" and x["think_closed"]
            ),
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


TASK_SYSTEM_PROMPT = (
    "Please reference the patient medical records to answer the user's "
    "questions. Answer briefly and precisely with the specific fact asked "
    "for.\n\nWrap your answer in: <answer> your short answer here </answer>"
)


def task_prompt_ids(tok, question):
    return tok.apply_chat_template(
        [
            {"role": "system", "content": TASK_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def score_task(text, task):
    """(correct, tagged, scored text) for one short answer."""
    import re

    if "</think>" not in text:
        return False, False, ""
    after = text.split("</think>", 1)[1]
    tags = re.findall(r"<answer>(.*?)</answer>", after, re.DOTALL | re.IGNORECASE)
    ans = tags[-1] if tags else after
    a = " ".join(ans.lower().split())
    ok = all(any(rubric_hit(a, alt) for alt in grp) for grp in task["required"])
    ok = ok and not any(rubric_hit(a, f) for f in task.get("forbidden", []))
    return ok, bool(tags), ans.strip()


def rubric_hit(a, alt):
    """alt occurs in the normalized answer a as a word, not inside one: no
    letter or digit (or, before a number, a decimal point) may precede it,
    and a number or a short alternative (under five characters) may not
    be followed by one. Longer alternatives may match as word prefixes
    ("hyperintens"), so "2" does not match "2021" and "mr" not "mrs"."""
    import re

    alt = " ".join(alt.lower().split())
    if not alt:
        return False
    pre = ""
    if alt[0].isalnum():
        pre = r"(?<![a-z0-9.])" if alt[0].isdigit() else r"(?<![a-z0-9])"
    post = ""
    if alt[-1].isalnum() and (alt[-1].isdigit() or len(alt) < 5):
        post = r"(?![a-z0-9])"
    return re.search(pre + re.escape(alt) + post, a) is not None


def tasks():
    cart = os.environ["CART"]
    task_list = json.load(open(os.environ["TASKS"]))
    runs = int(os.environ.get("RUNS", "1"))
    cap = int(os.environ.get("MAX_COMPLETION", "4096"))
    B = int(os.environ.get("BATCH", "16"))
    seed = int(os.environ.get("SEED", "0"))
    out_json = os.environ["OUT_JSON"]
    save_tokens = os.environ.get("SAVE_TOKENS", "1") == "1"
    t0 = time.time()
    tok, model = load()
    keys, values = cart_kv(cart)
    items = [(r, t) for r in range(runs) for t in task_list]
    gens = generate(
        tok,
        model,
        keys,
        values,
        [task_prompt_ids(tok, t["question"]) for _, t in items],
        cap,
        B,
        seed,
    )
    rows = []
    for (r, t), g in zip(items, gens):
        ok, tagged, ans = score_task(g["text"], t)
        rows.append(
            dict(
                run=r,
                row=t["row"],
                correct=int(ok),
                tag_found=tagged,
                answer=ans,
                **completion_fields(g, save_tokens),
            )
        )
    n = len(rows)
    res = dict(
        protocol="short source-grounded answers, batched HF generation with "
        f"the cartridge as a prefix; temp 0.6, max {cap}, rubric match",
        evaluator="cas_eval_batched tasks",
        model=MODEL,
        cart=os.path.abspath(cart),
        tasks=os.path.abspath(os.environ["TASKS"]),
        tasks_sha256=hashlib.sha256(open(os.environ["TASKS"], "rb").read()).hexdigest(),
        runs=runs,
        seed=seed,
        accuracy=sum(x["correct"] for x in rows) / n,
        correct=sum(x["correct"] for x in rows),
        n=n,
        completion=dict(
            at_budget=sum(1 for x in rows if x["finish_reason"] == "length"),
            think_unclosed=sum(1 for x in rows if not x["think_closed"]),
            no_answer_tag=sum(1 for x in rows if not x["tag_found"]),
        ),
        seconds=time.time() - t0,
        device=torch.cuda.get_device_name(),
    )
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    raw = out_json.replace(".json", ".raw.json")
    json.dump(rows, open(raw, "w"), indent=1)
    res["raw_path"] = raw
    json.dump(res, open(out_json, "w"), indent=2)
    print(
        f"TASKS_DONE acc={res['accuracy']:.4f} ({res['correct']}/{n}) "
        f"({res['seconds']:.0f}s) -> {out_json}",
        flush=True,
    )


if __name__ == "__main__":
    {"panel": panel, "tasks": tasks, "check": check}[sys.argv[1]]()
