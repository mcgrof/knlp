#!/usr/bin/env python3
"""Do scoring in one pass and scoring from a cache agree, and in what precision?

The two are the same computation, so in exact arithmetic they return the same
log-likelihoods. In float32 they should agree to several decimal places, and
a disagreement there is a defect in positions, masking or cache installation.
In bf16 they round differently and drift apart; that drift is arithmetic, and
its size is the smallest accuracy difference the evaluation can resolve.

This runs both paths over the same harness requests for one receiver and
reports, per task, how far the log-likelihoods moved and how many examples
changed their answer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

FLOAT32_TOLERANCE_NATS = 1e-3
LEAVES = {"mmlu": 57}


def compare(records_a, records_b):
    """Differences between two {context: (lengths, log-likelihoods)} maps."""
    import numpy as np

    if set(records_a) != set(records_b):
        raise ValueError("the two paths were not given the same contexts")
    diffs, flips, flips_norm = [], 0, 0
    for key, (lens, a) in records_a.items():
        _, b = records_b[key]
        a, b, n = np.asarray(a), np.asarray(b), np.asarray(lens, dtype=float)
        diffs.append(float(np.abs(a - b).max()))
        flips += int(a.argmax() != b.argmax())
        flips_norm += int((a / n).argmax() != (b / n).argmax())
    return {
        "contexts": len(diffs),
        "largest_difference_nats": max(diffs),
        "median_difference_nats": float(np.median(diffs)),
        "answers_changed": flips,
        "answers_changed_length_normalised": flips_norm,
    }


def run(name, revision, tasks, examples, dtypes=("float32",), device="cuda"):
    import lm_eval
    import torch
    from lm_eval.tasks import TaskManager
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .run_pair import FEWSHOT
    from .scoring import PrefixScorer, make_lm

    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    out = {"model": name, "revision": revision, "examples_per_leaf": examples}
    for dname in dtypes:
        t0 = time.time()
        # placed shard by shard: a float32 copy of a large receiver would
        # otherwise have to fit in host memory before it reached the card
        model = AutoModelForCausalLM.from_pretrained(
            name,
            revision=revision,
            torch_dtype=getattr(torch, dname),
            device_map={"": device},
        ).eval()
        load_s = time.time() - t0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        per_task = {}
        for task in tasks:
            seen, acc = {}, {}
            for mode in ("direct", "native"):
                scorer = PrefixScorer(model, mode)
                rec = {}
                inner = scorer.score

                def score(ctx, conts, inner=inner, rec=rec):
                    got = inner(ctx, conts)
                    rec[tuple(ctx)] = ([len(c) for c in conts], [g[0] for g in got])
                    return got

                scorer.score = score
                res = lm_eval.simple_evaluate(
                    model=make_lm(scorer, tok),
                    tasks=[task],
                    # the limit applies to each subject of a grouped task
                    limit=max(1, examples // LEAVES.get(task, 1)),
                    num_fewshot=FEWSHOT.get(task),
                    bootstrap_iters=0,
                    log_samples=False,
                    task_manager=TaskManager(),
                )
                seen[mode] = rec
                row = res["results"].get(task, {})
                acc[mode] = {k: v for k, v in row.items() if k.startswith("acc")}
            per_task[task] = {
                **compare(seen["direct"], seen["native"]),
                "accuracy": acc,
            }
            print(f"PARITY {dname} {task} {json.dumps(per_task[task])}", flush=True)
        worst = max(v["largest_difference_nats"] for v in per_task.values())
        changed = sum(v["answers_changed"] for v in per_task.values())
        out[dname] = {
            "tasks": per_task,
            "largest_difference_nats": worst,
            "answers_changed": changed,
            "seconds_loading": round(load_s, 1),
            "peak_gpu_gib": (
                round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if device == "cuda"
                else None
            ),
        }
        if dname == "float32":
            out[dname]["passes"] = bool(
                worst <= FLOAT32_TOLERANCE_NATS and changed == 0
            )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--tasks", default="hellaswag,arc_challenge,winogrande,mmlu")
    ap.add_argument("--examples", type=int, default=25)
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = run(
        args.model,
        args.revision,
        args.tasks.split(","),
        args.examples,
        tuple(args.dtypes.split(",")),
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
    print("PARITY_RESULT " + json.dumps(out, sort_keys=True))
    f32 = out.get("float32")
    return 0 if f32 is None or f32["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
