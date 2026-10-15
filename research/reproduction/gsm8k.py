# SPDX-License-Identifier: MIT
"""Check clean-checkpoint math accuracy separately from story perplexity."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request
from decimal import Decimal

import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from research.reproduction.kv_cache import inputs, load
from research.mo_variants.stage_c_common import write
from research.tinystories_depth.data import TOKEN_REV
from research.tinystories_depth.train import state_hash

REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
DATA_SHA = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
URL = f"https://raw.githubusercontent.com/openai/grade-school-math/{REVISION}/grade_school_math/data/test.jsonl"
PROMPT = "Question: {question}\nAnswer:"


def number(text, strict=False):
    if strict:
        if "####" not in text:
            return None
        text = text.rsplit("####", 1)[1]
    values = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return (
        str(Decimal(values[0 if strict else -1].replace(",", "")).normalize())
        if values
        else None
    )


@torch.inference_mode()
def generate(model, prompts, device, eos=50256, max_new=128, context=256):
    """Right padding is always in the future of each selected causal logit."""
    lengths = [len(x) for x in prompts]
    if not lengths or min(lengths) < 1:
        raise ValueError("Nonempty prompts are required")
    longest = max(lengths)
    budget = min(max_new, context - longest)
    if budget < 64:
        raise ValueError("Less than 64 generation tokens remain; do not truncate")
    ids = torch.full(
        (len(prompts), longest + budget), eos, dtype=torch.long, device=device
    )
    for i, tokens in enumerate(prompts):
        ids[i, : len(tokens)] = torch.tensor(tokens, device=device)
    lengths_gpu = torch.tensor(lengths, device=device)
    rows = torch.arange(len(prompts), device=device)
    outputs, ended = [[] for _ in prompts], [False] * len(prompts)
    for step in range(budget):
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(ids[:, : longest + step])
        selected = logits[rows, lengths_gpu + step - 1]
        if not torch.isfinite(selected).all():
            raise FloatingPointError("Nonfinite generation logits")
        next_ids = selected.argmax(-1)
        ids[rows, lengths_gpu + step] = next_ids
        for i, token in enumerate(next_ids.cpu().tolist()):
            if not ended[i]:
                if token == eos:
                    ended[i] = True
                else:
                    outputs[i].append(token)
        if all(ended):
            break
    return outputs, ended, budget


def run(root, source, cache, device="cuda"):
    root, cache = Path(root), Path(cache)
    repo = Path(__file__).resolve().parents[2]
    if root.resolve().is_relative_to(repo):
        raise ValueError("Generated outputs must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=True)
    dataset = cache / f"gsm8k-{REVISION}-test.jsonl"
    if not dataset.exists():
        dataset.write_bytes(urllib.request.urlopen(URL, timeout=60).read())
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError("GSM8K test input hash mismatch")
    tokenizer = Tokenizer.from_file(
        hf_hub_download(
            "gpt2", "tokenizer.json", revision=TOKEN_REV, cache_dir=cache / "hub"
        )
    )
    examples = [json.loads(x) for x in dataset.read_text().splitlines()][:16]
    prompts = [PROMPT.format(**x) for x in examples]
    encoded = [tokenizer.encode(x).ids for x in prompts]
    jobs, _, _ = inputs(source, "depth")
    jobs = [j for j in jobs if j["seed"] == 75211 and j["depth"] == 32]
    if {j["arm"] for j in jobs} != {"R", "M", "M-O"}:
        raise ValueError("Expected the paired residual/mHC/M-O checkpoints")
    report = dict(
        status="running",
        scientific_confirmation_allowed=False,
        scope="First 16 official GSM8K test problems, zero-shot clean-checkpoint feasibility only",
        dataset_revision=REVISION,
        dataset_sha256=DATA_SHA,
        prompt=PROMPT,
        seed=75211,
        depth=32,
        context=256,
        prompt_lengths=[len(x) for x in encoded],
        models=[],
        torch_version=torch.__version__,
        device=device,
    )
    write(root / "REPORT.json", report)
    began = time.monotonic()
    try:
        for job in jobs:
            model = load(job, device)
            before = state_hash(model.state_dict())
            outputs, ended, budget = generate(model, encoded, device)
            samples = []
            for i, (example, tokens) in enumerate(zip(examples, outputs)):
                completion = tokenizer.decode(tokens)
                expected = number(example["answer"], True)
                samples.append(
                    dict(
                        index=i,
                        question=example["question"],
                        prompt=prompts[i],
                        completion=completion,
                        completion_tokens=tokens,
                        gold_answer=example["answer"],
                        expected=expected,
                        strict_correct=number(completion, True) == expected,
                        lenient_correct=number(completion) == expected,
                        eos_stopped=ended[i],
                    )
                )
            if state_hash(model.state_dict()) != before:
                raise RuntimeError("Model state changed during evaluation")
            report["models"].append(
                dict(
                    arm=job["arm"],
                    checkpoint=job["checkpoint"],
                    n=len(samples),
                    parameters=sum(p.numel() for p in model.parameters()),
                    strict_correct=sum(s["strict_correct"] for s in samples),
                    lenient_correct=sum(s["lenient_correct"] for s in samples),
                    max_new_tokens=budget,
                    weights_unchanged=True,
                    samples=samples,
                )
            )
            write(root / "REPORT.json", report)
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="incomplete", error=str(error))
        raise
    finally:
        report["seconds"] = time.monotonic() - began
        write(root / "REPORT.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(args.root, args.source_run, args.cache, args.device)
