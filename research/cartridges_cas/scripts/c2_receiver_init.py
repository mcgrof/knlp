#!/usr/bin/env python3
"""Evaluate receiver-initializer arms against the frozen question set.

This is the entry point the target guard has to be reached through. A guard
that only exists in a helper is documentation: it enforces nothing until the
program that scores a candidate actually calls it, so that call lives here, at
the top of the run, before a model is even loaded.

Four differences from the source-utility screen, each of which is a defect in
that script for this purpose rather than a preference.

It takes the questions from a frozen file instead of re-selecting them, and it
scores against the option texts in that file rather than re-reading them from
the benchmark. The frozen set is the authority; anything that re-derives the
options can drift from what the target was frozen against.

It writes every row to disk the moment the row exists. The earlier screen
accumulated an arm's rows in memory and raised on the first token past its
budget, which threw away every row it had already produced — the most expensive
possible way to hit a limit.

It writes the five option texts into each row, so a later reviewer can re-score
the raw generations without the benchmark in hand.

And it takes many arms in one invocation, because a floor plus three
initializers at five checkpoints is sixteen evaluations and the receiver should
be loaded once, not sixteen times.

The receiver's weights are never touched. Only the cache differs between arms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from research.cartridges_cas.scripts import c2_target  # noqa: E402
from research.cartridges_cas.scripts.c2_questions import (  # noqa: E402
    load_patients_plain,
)
from research.cartridges_cas.scripts.score_longhealth import score  # noqa: E402
from research.cartridges_cas.scripts.score_longhealth import summarise  # noqa: E402

CONTRACT = "c2_receiver_init_v1"


def parse_arm_spec(spec):
    """ "arm=path" or "arm.stepN=path" into (arm, checkpoint, path).

    The checkpoint is a label on the arm, not a separate arm: the registry that
    decides whether something is a candidate keys on the arm alone, so a
    checkpoint suffix can never turn a mapped arm into an unrecognised one.
    """
    if "=" not in spec:
        raise SystemExit(f"INVALID: --cartridge-list entry {spec!r} has no '='")
    label, path = spec.split("=", 1)
    arm, _, ckpt = label.partition(".")
    c2_target.classify_arm(arm)  # refuses anything undeclared
    return arm, (ckpt or "step0"), (path or "")


def load_frozen(questions_path, benchmark, benchmark_sha256=""):
    with open(questions_path, encoding="utf-8") as f:
        frozen = json.load(f)
    _, patients = load_patients_plain(
        benchmark, [frozen["patient"]], want_sha256=benchmark_sha256
    )
    return frozen, patients[0]


def question_namespaces(frozen):
    """Question objects the library's prompt wrapper can consume.

    Built from the frozen options rather than the benchmark, so the text the
    model sees is the text the target was frozen against.
    """
    from types import SimpleNamespace

    out = []
    for q in frozen["questions"]:
        fields = {f"answer_{c}": q["options"][i] for i, c in enumerate("abcde")}
        out.append(
            SimpleNamespace(
                question_id=q["question_id"],
                question=q["question"],
                correct=q["gold_text"],
                gold_index=q["gold_index"],
                options=q["options"],
                **fields,
            )
        )
    return out


def evaluate_arm(
    arm,
    checkpoint,
    cart_path,
    questions,
    patient,
    *,
    model,
    tok,
    wrap,
    device,
    max_new_tokens,
    rows_path,
    load_cart_fn,
):
    """One arm at one checkpoint, every question, rows flushed as they land."""
    import torch

    from cartridges.generation import flex_generate

    stop_ids = sorted(
        {
            i
            for i in (
                tok.eos_token_id,
                tok.convert_tokens_to_ids("<|im_end|>"),
                tok.convert_tokens_to_ids("<|endoftext|>"),
            )
            if i is not None and i >= 0
        }
    )

    cache = load_cart_fn(cart_path) if cart_path else None
    rows = []
    with open(rows_path, "w", encoding="utf-8") as sink:
        for q in questions:
            prompt = wrap(q, patient)
            ids = (
                tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                    enable_thinking=True,
                )
                .to(device)
                .flatten()
            )
            if cache is not None:
                cache.clear()
            n_prompt = int(ids.shape[0])
            t0 = time.time()
            with torch.no_grad():
                out = flex_generate(
                    model,
                    tok,
                    ids,
                    seq_ids=torch.zeros(n_prompt, dtype=torch.long, device=device),
                    position_ids=torch.arange(n_prompt, device=device),
                    cache=cache,
                    stop_token_ids=stop_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                )
            gen = out.get(0, [])
            text = tok.decode(gen, skip_special_tokens=True)
            sc = score(
                text,
                q.options,
                q.gold_index,
                n_new_tokens=len(gen),
                max_new_tokens=max_new_tokens,
            )
            row = {
                "arm": arm,
                "checkpoint": checkpoint,
                "question_id": q.question_id,
                "options": q.options,
                "gold_index": q.gold_index,
                "generated": text,
                "n_new_tokens": len(gen),
                "prompt_tokens": n_prompt,
                "seconds": round(time.time() - t0, 3),
                **sc,
            }
            rows.append(row)
            # Flushed per question: a limit reached halfway must not destroy
            # the answers already produced.
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--benchmark-sha256", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--expected-model-name",
        default="",
        help="refuse to run unless the loaded tokenizer reports this name; the "
        "two models here are geometrically identical so nothing else can tell "
        "them apart",
    )
    ap.add_argument(
        "--cartridge-list",
        required=True,
        help="comma-separated arm[.checkpoint]=path; an empty path is the "
        "no-cartridge floor",
    )
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target-file", default="")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    specs = [parse_arm_spec(s) for s in args.cartridge_list.split(",") if s.strip()]
    arms = sorted({a for a, _, _ in specs})

    # The guard, reached through the real entry point and before anything
    # expensive happens. It refuses a mapped arm unless a target frozen against
    # this same question set already exists and its references have not moved.
    target = c2_target.require_target_for_mapped(args.run_dir, args.questions, arms)
    if (
        any(c2_target.classify_arm(a) == "mapped" for a in arms)
        and not args.target_file
    ):
        raise SystemExit(
            "BLOCKED: a mapped arm was requested without --target-file. The bar "
            "must be on disk, and frozen, before a candidate is scored."
        )

    import torch
    from transformers import AutoTokenizer

    from cartridges.initialization.tokenization_utils import MODELS_WITH_THINKING
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.expected_model_name and tok.name_or_path != args.expected_model_name:
        raise SystemExit(
            f"INVALID: tokenizer reports {tok.name_or_path!r}, expected "
            f"{args.expected_model_name!r}"
        )
    if tok.name_or_path not in MODELS_WITH_THINKING:
        raise SystemExit(
            f"INVALID: {tok.name_or_path!r} is not in the library's thinking-model "
            "set, so the prompt wrapper would emit a different instruction than "
            "the one this protocol was defined with. Spelling matters here."
        )

    from research.cartridges_cas.scripts.c1_source_utility import make_wrapper
    from research.cartridges_cas.scripts.cas_cart_init import load_cart

    frozen, patient = load_frozen(args.questions, args.benchmark, args.benchmark_sha256)
    questions = question_namespaces(frozen)
    wrap = make_wrapper(tok)

    device = "cuda"
    # Same construction as the screen that produced the reference numbers:
    # load, then move, then cast. Changing the order changes the numerics.
    model = (
        FlexQwen3ForCausalLM.from_pretrained(args.model).to(device).to(torch.bfloat16)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    from cartridges.cache import AttnConfig, TrainableCache

    attn = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=getattr(
            model.config,
            "head_dim",
            model.config.hidden_size // model.config.num_attention_heads,
        ),
    )

    def load_cart_fn(path):
        # Never the library's from_pretrained: it reads the frozen-token count
        # off the head axis and returns eight where this checkpoint holds one.
        keys, values, nfrozen = load_cart(path, device=device)
        cache = TrainableCache(
            config=attn, init_keys=keys, init_values=values, num_frozen_tokens=nfrozen
        ).to(device)
        return cache

    receipt = {
        "contract": CONTRACT,
        "model": args.model,
        "tokenizer_name_or_path": tok.name_or_path,
        "question_set_sha256": frozen["question_set_sha256"],
        "benchmark_sha256": frozen["benchmark_sha256"],
        "patient": frozen["patient"],
        "questions": len(questions),
        "max_new_tokens": args.max_new_tokens,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "target_present": bool(target),
        "arms": {},
    }

    for arm, ckpt, path in specs:
        label = f"{arm}.{ckpt}" if ckpt != "step0" or path else arm
        rows_path = os.path.join(args.run_dir, f"rows.{arm}.{ckpt}.jsonl")
        if not path:
            rows_path = os.path.join(args.run_dir, f"rows.{arm}.jsonl")
        print(f"[{time.strftime('%H:%M:%S')}] {label} -> {rows_path}", flush=True)
        rows = evaluate_arm(
            arm,
            ckpt,
            path,
            questions,
            patient,
            model=model,
            tok=tok,
            wrap=wrap,
            device=device,
            max_new_tokens=args.max_new_tokens,
            rows_path=rows_path,
            load_cart_fn=load_cart_fn,
        )
        s = summarise(rows)
        receipt["arms"][label] = {
            "rows": rows_path,
            "n_correct": s["n_correct"],
            "unresolved": s["unresolved"],
            "cap_hit": s["cap_hit"],
            "bad": c2_target.count_bad(rows),
        }
        print(
            f"    {s['n_correct']}/{s['n']} correct, {s['unresolved']} unresolved, "
            f"{s['cap_hit']} cap-hit",
            flush=True,
        )

    with open(os.path.join(args.run_dir, "EVAL_RECEIPT.json"), "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
