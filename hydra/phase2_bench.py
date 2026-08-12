# SPDX-License-Identifier: MIT
"""hydra: capability-router benchmark trace builder (phase 2).

Builds the golden per-query x per-model correctness trace a capability
router is trained and evaluated on, from public benchmarks with
deterministic scoring (no LLM judge):

  gsm8k       math reasoning   exact-match on the final number
  arc / mmlu  knowledge        option-letter logprob argmax
  mbpp / humaneval  code       unit-test execution

Sub-commands:
  gen     Download datasets, emit prompts.jsonl (query, dimension,
          benchmark, gold, options/tests, provenance group).
  infer   Batch one model over prompts.jsonl with vLLM; writes
          raw_<model>.jsonl with generations / choice logprobs and
          per-benchmark wall time (the GPU-seconds cost source).
  score   Deterministically score raw_<model>.jsonl ->
          scored_<model>.jsonl (correct 0/1 per query).
  merge   Join all scored files into trace.parquet (one row per
          query: text, dim, benchmark, group, per-model correct,
          per-model gpu_seconds share).

Dimensions (K=3, declared subset of HyDRA's four -- debugging and
tool-use lack public deterministic ground truth):
  math_reasoning, knowledge, code
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

DIMS = {
    "gsm8k": "math_reasoning",
    "arc": "knowledge",
    "mmlu": "knowledge",
    "mbpp": "code",
    "humaneval": "code",
}

LETTERS = ["A", "B", "C", "D", "E"]


def _jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def cmd_gen(args):
    from datasets import load_dataset  # noqa: PLC0415

    out = []

    ds = load_dataset("openai/gsm8k", "main", split="test")
    n = min(args.gsm8k, len(ds))
    for i, r in enumerate(ds.select(range(n))):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append(
            {
                "qid": f"gsm8k-{i}",
                "benchmark": "gsm8k",
                "dim": DIMS["gsm8k"],
                "kind": "gen",
                "prompt": (
                    r["question"] + "\n\nThink step by step, then give the final "
                    "numeric answer on the last line as: #### <answer>"
                ),
                "gold": gold,
                "group": f"gsm8k-g{i % 10}",
            }
        )

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    n = min(args.arc, len(ds))
    for i, r in enumerate(ds.select(range(n))):
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        norm = dict(zip(labels, texts))
        letters = LETTERS[: len(labels)]
        remap = {let: norm[lab] for let, lab in zip(letters, labels)}
        gold_idx = labels.index(r["answerKey"])
        opts = "\n".join(f"{let}. {remap[let]}" for let in letters)
        out.append(
            {
                "qid": f"arc-{i}",
                "benchmark": "arc",
                "dim": DIMS["arc"],
                "kind": "choice",
                "prompt": (
                    r["question"]
                    + "\n"
                    + opts
                    + "\nAnswer with a single letter.\nAnswer:"
                ),
                "gold": LETTERS[gold_idx],
                "n_options": len(labels),
                "group": f"arc-g{i % 10}",
            }
        )

    subjects = [
        "college_computer_science",
        "high_school_mathematics",
        "professional_medicine",
        "world_religions",
        "econometrics",
        "logical_fallacies",
    ]
    per_subj = max(1, args.mmlu // len(subjects))
    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj, split="test")
        n = min(per_subj, len(ds))
        for i, r in enumerate(ds.select(range(n))):
            opts = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(r["choices"]))
            out.append(
                {
                    "qid": f"mmlu-{subj}-{i}",
                    "benchmark": "mmlu",
                    "dim": DIMS["mmlu"],
                    "kind": "choice",
                    "prompt": (
                        r["question"]
                        + "\n"
                        + opts
                        + "\nAnswer with a single letter.\nAnswer:"
                    ),
                    "gold": LETTERS[r["answer"]],
                    "n_options": 4,
                    "group": f"mmlu-{subj}",
                }
            )

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    for i, r in enumerate(ds):
        out.append(
            {
                "qid": f"mbpp-{i}",
                "benchmark": "mbpp",
                "dim": DIMS["mbpp"],
                "kind": "code",
                "prompt": (
                    r["prompt"] + "\nWrite only the Python function, inside a "
                    "```python code block. It must satisfy:\n"
                    + "\n".join(r["test_list"])
                ),
                "tests": "\n".join(r["test_list"]),
                "setup": r.get("test_setup_code") or "",
                "group": f"mbpp-g{i % 10}",
            }
        )

    ds = load_dataset("openai/openai_humaneval", split="test")
    for i, r in enumerate(ds):
        out.append(
            {
                "qid": f"humaneval-{i}",
                "benchmark": "humaneval",
                "dim": DIMS["humaneval"],
                "kind": "code",
                "prompt": (
                    "Complete this Python function. Reply with the "
                    "complete function (signature included) inside a "
                    "```python code block.\n\n" + r["prompt"]
                ),
                "tests": r["test"],
                "entry_point": r["entry_point"],
                "group": f"humaneval-g{i % 10}",
            }
        )

    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    counts = {}
    for r in out:
        counts[r["benchmark"]] = counts.get(r["benchmark"], 0) + 1
    print(json.dumps({"total": len(out), "per_benchmark": counts}))


def cmd_infer(args):
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    prompts = list(_jsonl(args.prompts))
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.92,
    )
    tok = llm.get_tokenizer()

    def chat(text):
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    recs = {}
    timing = {}
    by_kind = {}
    for p in prompts:
        by_kind.setdefault((p["benchmark"], p["kind"]), []).append(p)

    for (bench, kind), items in by_kind.items():
        t0 = time.perf_counter()
        if kind == "choice":
            sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
        elif kind == "gen":
            sp = SamplingParams(max_tokens=512, temperature=0.0)
        else:
            sp = SamplingParams(max_tokens=768, temperature=0.0)
        outs = llm.generate([chat(p["prompt"]) for p in items], sp)
        dt = time.perf_counter() - t0
        timing[bench] = timing.get(bench, 0.0) + dt
        for p, o in zip(items, outs):
            rec = {"qid": p["qid"], "text": o.outputs[0].text}
            if kind == "choice":
                lps = {}
                if o.outputs[0].logprobs:
                    for t, lp in o.outputs[0].logprobs[0].items():
                        s = lp.decoded_token.strip()
                        if s in LETTERS and (s not in lps or lp.logprob > lps[s]):
                            lps[s] = lp.logprob
                rec["letter_logprobs"] = lps
            recs[p["qid"]] = rec

    n_by_bench = {}
    for p in prompts:
        n_by_bench[p["benchmark"]] = n_by_bench.get(p["benchmark"], 0) + 1
    with open(args.out, "w") as f:
        f.write(
            json.dumps(
                {
                    "meta": {
                        "model": args.model,
                        "timing_seconds": timing,
                        "n": n_by_bench,
                    }
                }
            )
            + "\n"
        )
        for r in recs.values():
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"model": args.model, "timing": timing}))


def _extract_number(text):
    m = re.findall(r"####\s*([-+]?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"([-+]?[\d,]*\.?\d+)", text)
    if not m:
        return None
    return m[-1].replace(",", "").rstrip(".")


def _extract_code(text):
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return m[0] if m else text


def _run_code(code, tests, setup="", entry_point=None, timeout=8):
    if entry_point:
        harness = code + "\n\n" + tests + f"\n\ncheck({entry_point})\n"
    else:
        harness = setup + "\n" + code + "\n\n" + tests + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        r = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            timeout=timeout,
            env={"PATH": os.environ.get("PATH", ""), "HOME": "/tmp"},
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def cmd_score(args):
    prompts = {p["qid"]: p for p in _jsonl(args.prompts)}
    rows = list(_jsonl(args.raw))
    meta = rows[0]["meta"]
    out = []
    for r in rows[1:]:
        p = prompts[r["qid"]]
        if p["kind"] == "gen":
            correct = _extract_number(r["text"]) == p["gold"]
        elif p["kind"] == "choice":
            lps = r.get("letter_logprobs") or {}
            pred = max(lps, key=lps.get) if lps else None
            correct = pred == p["gold"]
        else:
            correct = _run_code(
                _extract_code(r["text"]),
                p["tests"],
                p.get("setup", ""),
                p.get("entry_point"),
            )
        out.append({"qid": r["qid"], "correct": int(correct)})
    per_bench = {}
    for o in out:
        b = prompts[o["qid"]]["benchmark"]
        agg = per_bench.setdefault(b, [0, 0])
        agg[0] += o["correct"]
        agg[1] += 1
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(
        json.dumps(
            {
                "model": meta["model"],
                "accuracy": {b: round(c / n, 4) for b, (c, n) in per_bench.items()},
            }
        )
    )


def cmd_merge(args):
    import pandas as pd  # noqa: PLC0415

    prompts = list(_jsonl(args.prompts))
    base = {
        p["qid"]: {
            "qid": p["qid"],
            "benchmark": p["benchmark"],
            "dim": p["dim"],
            "group": p["group"],
            "prompt": p["prompt"],
        }
        for p in prompts
    }
    models = []
    for path in args.scored:
        rows = list(_jsonl(path))
        meta = rows[0]["meta"]
        model = meta["model"]
        models.append(model)
        gpu_s = {
            b: meta["timing_seconds"][b] / meta["n"][b] for b in meta["timing_seconds"]
        }
        for r in rows[1:]:
            base[r["qid"]][f"correct::{model}"] = r["correct"]
            b = base[r["qid"]]["benchmark"]
            base[r["qid"]][f"gpu_s::{model}"] = gpu_s.get(b)
    df = pd.DataFrame(list(base.values()))
    df.to_parquet(args.out)
    print(json.dumps({"rows": len(df), "models": models, "out": args.out}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--gsm8k", type=int, default=800)
    g.add_argument("--arc", type=int, default=600)
    g.add_argument("--mmlu", type=int, default=600)
    g.add_argument("--out", default="prompts.jsonl")
    g.set_defaults(fn=cmd_gen)

    i = sub.add_parser("infer")
    i.add_argument("--model", required=True)
    i.add_argument("--prompts", default="prompts.jsonl")
    i.add_argument("--max-model-len", type=int, default=4096)
    i.add_argument("--out", required=True)
    i.set_defaults(fn=cmd_infer)

    s = sub.add_parser("score")
    s.add_argument("--raw", required=True)
    s.add_argument("--prompts", default="prompts.jsonl")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_score)

    m = sub.add_parser("merge")
    m.add_argument("--prompts", default="prompts.jsonl")
    m.add_argument("--scored", nargs="+", required=True)
    m.add_argument("--out", default="trace.parquet")
    m.set_defaults(fn=cmd_merge)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
