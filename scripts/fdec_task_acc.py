#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Task accuracy of the fdec bricks.

The composition matrix (fdec_compose.py) scores perplexity and top-1
agreement on held-out text.  This script asks the task question: does a
model with int8 decoder weights, an asymmetric K16/V8 cache, the certified
output head, or a combination of them, answer as many questions correctly
as the dense model?

A cell is the dense model ("base") or a set of bricks joined by "+":

  w8    int8 weights in every linear layer of the decoder blocks;
  kv    asymmetric KV cache: keys stay 16-bit, values go to FP8 with a
        direct cast (--kv-scale selects a per-tensor scale);
  cert  the token the certified output head returns.

Two tasks, both scored on the same items for every cell so the comparison
is paired:

  gsm8k  grade-school math word problems, test split.  The prompt holds
         the first --shots training problems with their worked answers.
         Decode is greedy.  An answer is correct when the first number
         after the first "####" in the completion equals the reference.
  mmlu   four-choice questions, test split of every subject.  The prompt
         holds the five development examples of the subject.  The answer
         is the choice letter with the largest next-token logit.  The log
         also gives the accuracy averaged over the subjects.

The certified head returns the token with the largest float32 logit; the
composition matrix checks that at every position of every cell.  A task
run needs only that token, so a "cert" cell computes the logits of the
last position in float32 and takes the largest.  The other cells use the
dense head of the model, which rounds its logits to the model dtype.  Two
bf16 logits within one rounding step tie, so the two heads can select
different tokens there.

For each cell the report gives accuracy, the paired difference against the
dense model with a 95% interval, the count of items that flipped in each
direction, and the exact McNemar p-value.  For a cell with two or more
bricks it also gives the departure from additivity: the accuracy change of
the cell minus the sum of the changes of its bricks alone.

The result file keeps four GSM8K completions per cell.  --completions
writes all of them to a second file, to read the items that change between
two cells.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from decimal import Decimal, InvalidOperation

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fdec_compose import quantize_weights_int8  # noqa: E402
from fdec_matrix import (  # noqa: E402
    install_asym_kv,
    install_fp32_attention,
    restore,
)

# Run order: every cell without int8 weights comes before the in-place
# weight quantization.
ARMS = ("base", "kv", "cert", "kv+cert", "w8", "w8+kv", "w8+cert", "w8+kv+cert")
DEFAULT_ARMS = "base,kv,cert,w8,w8+kv,w8+kv+cert"
LETTERS = "ABCD"
# batch * length^2 limit for the float32 attention matrices (16 x 1024^2)
ATTN_BUDGET = 16 * 1024 * 1024
NUM = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


def norm_number(text):
    m = NUM.search(text)
    if not m:
        return None
    try:
        return str(Decimal(m.group(0).replace(",", "").replace("$", "")).normalize())
    except InvalidOperation:
        return None


def bricks(arm):
    return set() if arm == "base" else set(arm.split("+"))


class Float32Head(torch.nn.Module):
    """Stand-in for the certified output head: float32 logits of the last
    position.  Their argmax is the token the certificate returns.

    The head weights are cast a block of rows at a time, so the model does
    not need a second, float32 copy of its head in memory."""

    def __init__(self, head, rows=16384):
        super().__init__()
        assert getattr(head, "bias", None) is None, "expects a bias-free head"
        self.dense = head
        self.rows = rows

    def forward(self, hidden):
        h = hidden[:, -1:].float()
        w = self.dense.weight
        return torch.cat(
            [
                F.linear(h, w[i : i + self.rows].float())
                for i in range(0, w.shape[0], self.rows)
            ],
            dim=-1,
        )


def swap_head(model, new):
    """Replace the output head of the model; returns the module it replaced."""
    old = model.get_output_embeddings()
    for name, child in model.named_children():
        if child is old:
            setattr(model, name, new)
            return old
    raise AssertionError("output head is not a direct child of the model")


def gsm_answer(completion):
    """First number after the first '####'; None when the marker is absent."""
    if "####" not in completion:
        return None
    return norm_number(completion.split("####", 1)[1])


def keep_completions(path, arm, texts):
    """Add the completions of one cell to a JSON file: {cell: [text, ...]}."""
    kept = {}
    if os.path.exists(path):
        with open(path) as f:
            kept = json.load(f)
    kept[arm] = texts
    with open(path, "w") as f:
        json.dump(kept, f)


def gsm_clean(solution):
    return re.sub(r"<<[^>]*>>", "", solution).strip()


def gsm_items(shots, limit):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")
    head = "".join(
        f"Question: {q}\nAnswer: {gsm_clean(a)}\n\n"
        for q, a in zip(ds["train"]["question"][:shots], ds["train"]["answer"][:shots])
    )
    test = ds["test"]
    n = len(test) if limit <= 0 else min(limit, len(test))
    return [
        dict(
            prompt=head + f"Question: {test[i]['question']}\nAnswer:",
            gold=norm_number(test[i]["answer"].rsplit("####", 1)[1]),
        )
        for i in range(n)
    ]


def mmlu_fmt(row, with_answer):
    s = row["question"].strip() + "\n"
    for letter, choice in zip(LETTERS, row["choices"]):
        s += f"{letter}. {choice}\n"
    s += "Answer:"
    if with_answer:
        s += f" {LETTERS[row['answer']]}\n\n"
    return s


def mmlu_items(per_subject):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all")
    dev = {}
    for row in ds["dev"]:
        dev.setdefault(row["subject"], []).append(row)
    items, seen = [], {}
    for row in ds["test"]:
        sub = row["subject"]
        seen[sub] = seen.get(sub, 0) + 1
        if per_subject > 0 and seen[sub] > per_subject:
            continue
        head = (
            "The following are multiple choice questions (with answers) about "
            f"{sub.replace('_', ' ')}.\n\n"
        )
        head += "".join(mmlu_fmt(r, True) for r in dev[sub][:5])
        items.append(
            dict(prompt=head + mmlu_fmt(row, False), gold=row["answer"], subject=sub)
        )
    return items


def subject_mean(ok, items):
    """Accuracy averaged over the subjects, each subject with equal weight.
    Some model cards report MMLU this way, so it is the number to compare
    with them.  The paired statistics use the items."""
    per = {}
    for x, it in zip(ok, items):
        a = per.setdefault(it["subject"], [0, 0])
        a[0] += x
        a[1] += 1
    return sum(a / n for a, n in per.values()) / len(per)


def batches(order, size, lens=None, budget=0):
    """Batches over `order` (longest first).  With a budget, the batch shrinks
    so that batch * length^2 stays under it: the attention function holds
    one float32 length-by-length matrix per head and sequence."""
    i = 0
    while i < len(order):
        n = size
        if budget:
            n = max(1, min(size, budget // (lens[order[i]] ** 2)))
        yield order[i : i + n]
        i += n


@torch.no_grad()
def run_gsm(model, tok, items, device, batch, max_new):
    out = [None] * len(items)
    lens = [len(tok(it["prompt"]).input_ids) for it in items]
    order = sorted(range(len(items)), key=lambda i: -lens[i])
    for idx in batches(order, batch):
        enc = tok([items[i]["prompt"] for i in idx], return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        gen = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new,
            stop_strings=["Question:"],
            tokenizer=tok,
            pad_token_id=tok.pad_token_id,
        )
        texts = tok.batch_decode(
            gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for i, text in zip(idx, texts):
            out[i] = text.split("Question:", 1)[0]
    return out


@torch.no_grad()
def run_mmlu(model, tok, items, device, batch, max_len, letter_ids):
    pred = [None] * len(items)
    ids = [tok(it["prompt"]).input_ids[-max_len:] for it in items]
    order = sorted(range(len(items)), key=lambda i: -len(ids[i]))
    lens = [len(x) for x in ids]
    for idx in batches(order, batch, lens, ATTN_BUDGET):
        width = max(len(ids[i]) for i in idx)
        x = torch.full((len(idx), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for row, i in enumerate(idx):
            x[row, width - len(ids[i]) :] = torch.tensor(ids[i])
            mask[row, width - len(ids[i]) :] = 1
        logits = model(input_ids=x.to(device), attention_mask=mask.to(device)).logits[
            :, -1
        ]
        choice = logits[:, letter_ids].float().argmax(-1).cpu().tolist()
        for i, c in zip(idx, choice):
            pred[i] = c
    return pred


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def nonadditivity(per, arm):
    """Accuracy change of a multi-brick cell minus the sum of the changes of
    its bricks alone, in accuracy, with a 95% interval over the items.
    None when the cell has one brick or a single-brick cell is missing."""
    parts = sorted(bricks(arm))
    if len(parts) < 2 or any(b not in per for b in parts + ["base"]):
        return None
    base, cell = per["base"]["ok"], per[arm]["ok"]
    solo = [per[b]["ok"] for b in parts]
    n = len(base)
    v = [
        cell[i] - sum(x[i] for x in solo) + (len(parts) - 1) * base[i] for i in range(n)
    ]
    mean = sum(v) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1) / n)
    return dict(value=mean, ci95=[mean - 1.96 * se, mean + 1.96 * se])


def paired(base_ok, arm_ok):
    n = len(base_ok)
    b = sum(1 for x, y in zip(base_ok, arm_ok) if x and not y)
    c = sum(1 for x, y in zip(base_ok, arm_ok) if y and not x)
    diff = (c - b) / n
    se = math.sqrt(max((b + c) / n - diff * diff, 0.0) / n)
    return dict(
        n=n,
        accuracy=sum(arm_ok) / n,
        diff_vs_base=diff,
        ci95=[diff - 1.96 * se, diff + 1.96 * se],
        lost=b,
        gained=c,
        mcnemar_p=mcnemar_exact(b, c),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tasks", default="gsm8k,mmlu")
    ap.add_argument(
        "--arms",
        default=DEFAULT_ARMS,
        help="cells to score, from: " + ", ".join(ARMS),
    )
    ap.add_argument("--w8-group", type=int, default=128)
    ap.add_argument(
        "--kv-scale",
        choices=("one", "tensor"),
        default="one",
        help="how the values reach FP8: a direct cast, the vLLM default, or "
        "one scale per tensor (largest magnitude over 448)",
    )
    ap.add_argument("--shots", type=int, default=8, help="GSM8K prompt examples")
    ap.add_argument("--gsm-limit", type=int, default=0, help="0 means all")
    ap.add_argument("--gsm-batch", type=int, default=16)
    ap.add_argument("--gsm-max-new", type=int, default=256)
    ap.add_argument(
        "--mmlu-per-subject", type=int, default=0, help="0 means every test item"
    )
    ap.add_argument("--mmlu-batch", type=int, default=16)
    ap.add_argument("--mmlu-max-len", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--completions",
        default="",
        help="also write every GSM8K completion of every cell to this JSON file",
    )
    args = ap.parse_args()
    unknown = set(args.arms.split(",")) - set(ARMS)
    assert not unknown, f"unknown cells: {sorted(unknown)}"
    arms = [a for a in ARMS if a in args.arms.split(",")]
    assert arms and arms[0] == "base", "the dense arm is the paired reference"
    tasks = args.tasks.split(",")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    torch.manual_seed(0)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager"
    ).to(device)
    model.eval()
    # Every arm, the dense model included, computes attention the way a fused
    # serving kernel does.  The stock eager function rounds the attention
    # scores to bf16 before the softmax.  In a 32-problem GSM8K check that
    # alone took Qwen2.5-7B from 24 solved problems to 7 (see fp32_attention
    # in fdec_matrix.py).
    install_fp32_attention(model)

    data = {}
    if "gsm8k" in tasks:
        data["gsm8k"] = gsm_items(args.shots, args.gsm_limit)
    if "mmlu" in tasks:
        data["mmlu"] = mmlu_items(args.mmlu_per_subject)
    letter_ids = [tok.encode(" " + c, add_special_tokens=False)[-1] for c in LETTERS]

    res = {}
    if os.path.exists(args.out):
        # Resume only a run of the same model, settings and items.  The
        # arms are paired item by item, so a result from other settings or
        # from an item limit must not be mixed in.
        with open(args.out) as f:
            old = json.load(f)
        want = settings(args)
        diff = {
            k: (old.get(k, LEGACY.get(k)), v)
            for k, v in want.items()
            if old.get(k, LEGACY.get(k)) != v and not (k in UNRECORDED and k not in old)
        }
        assert not diff, f"{args.out} holds a run with other settings: {diff}"
        res = old.get("raw", {})
        for task, per in res.items():
            for arm, r in per.items():
                assert task not in data or len(r["ok"]) == len(data[task]), (
                    f"{args.out}: {task}/{arm} has {len(r['ok'])} items, "
                    f"this run has {len(data[task])}"
                )
    quantized = False
    for arm in arms:
        todo = [t for t in data if arm not in res.get(t, {})]
        if not todo:
            continue
        on = bricks(arm)
        if "w8" in on and not quantized:
            quantize_weights_int8(model, args.w8_group)
            quantized = True
        assert quantized == ("w8" in on), "dense arms must run before int8 weights"
        saved = install_asym_kv(model, args.kv_scale) if "kv" in on else {}
        if "kv" in on:
            assert saved, "kv brick did not patch any attention module"
        dense_head = None
        if "cert" in on:
            dense_head = swap_head(model, Float32Head(model.get_output_embeddings()))
        try:
            for task in todo:
                t1 = time.time()
                items = data[task]
                if task == "gsm8k":
                    texts = run_gsm(
                        model, tok, items, device, args.gsm_batch, args.gsm_max_new
                    )
                    ok = [gsm_answer(t) == it["gold"] for t, it in zip(texts, items)]
                    if args.completions:
                        keep_completions(args.completions, arm, texts)
                    extra = dict(
                        no_marker=sum("####" not in t for t in texts),
                        samples=texts[:4],
                    )
                else:
                    pred = run_mmlu(
                        model,
                        tok,
                        items,
                        device,
                        args.mmlu_batch,
                        args.mmlu_max_len,
                        letter_ids,
                    )
                    ok = [p == it["gold"] for p, it in zip(pred, items)]
                    extra = dict(subject_mean=subject_mean(ok, items))
                res.setdefault(task, {})[arm] = dict(
                    ok=[int(x) for x in ok], seconds=time.time() - t1, **extra
                )
                more = (
                    f" subject_mean={extra['subject_mean']:.4f}"
                    if "subject_mean" in extra
                    else ""
                )
                print(
                    f"[fdec-task] {arm:10s} {task:6s} acc={sum(ok) / len(ok):.4f}"
                    f"{more} n={len(ok)} {time.time() - t1:.0f}s",
                    flush=True,
                )
                write(args, res, t0)
        finally:
            restore(saved)
            if dense_head is not None:
                swap_head(model, dense_head)
    summary, extra = write(args, res, t0)
    for task, per in summary.items():
        for arm, s in per.items():
            print(
                f"{task:6s} {arm:10s} acc={s['accuracy']:.4f} "
                f"diff={s['diff_vs_base']:+.4f} "
                f"ci=[{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}] "
                f"lost={s['lost']} gained={s['gained']} p={s['mcnemar_p']:.3f}"
            )
        for arm, x in extra.get(task, {}).items():
            print(
                f"{task:6s} {arm:10s} minus the sum of its bricks alone: "
                f"{x['value']:+.4f} ci=[{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}]"
            )


def settings(args):
    """What must match before two arms can be paired."""
    return dict(
        model=args.model,
        dtype=args.dtype,
        w8_group=args.w8_group,
        shots=args.shots,
        gsm_max_new=args.gsm_max_new,
        mmlu_max_len=args.mmlu_max_len,
        mmlu_per_subject=args.mmlu_per_subject,
        attention="float32 scores, softmax and value sum",
        kv_scale=args.kv_scale,
        gsm_batch=args.gsm_batch,
        mmlu_batch=args.mmlu_batch,
    )


# Settings of a result file written before the setting existed.  The batch
# sizes were not recorded then, so they are not checked for such a file.
LEGACY = dict(kv_scale="tensor")
UNRECORDED = ("gsm_batch", "mmlu_batch")


def write(args, res, t0):
    summary, extra = {}, {}
    for task, per in res.items():
        if "base" not in per:
            continue
        summary[task] = {
            arm: paired(per["base"]["ok"], r["ok"]) for arm, r in per.items()
        }
        found = {arm: nonadditivity(per, arm) for arm in per}
        extra[task] = {arm: x for arm, x in found.items() if x}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            dict(
                **settings(args),
                torch=torch.__version__,
                seconds=time.time() - t0,
                summary=summary,
                nonadditivity=extra,
                raw=res,
            ),
            f,
        )
    return summary, extra


if __name__ == "__main__":
    main()
