# SPDX-License-Identifier: MIT
"""Frozen GSM8K chat prompts and separate strict/flexible answer extraction."""

import hashlib
import json
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

STRICT = r"The answer is (\-?[0-9\.\,]+)."
FLEXIBLE = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
FINAL_INSTRUCTION = "\nEnd your response with: The answer is <number>."


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def reserve_dispatch(directory, subset):
    """Keep smoke receipts distinct across pilot, full and resumed invocations."""
    if subset not in ("development", "pilot", "full"):
        raise ValueError("unknown GSM8K subset")
    parent = Path(directory) / "dispatch"
    parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        path = parent / f"{subset}-{attempt:04d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            attempt += 1


def number(text):
    """Normalize only the numeric representation, without answer guessing."""
    try:
        value = Decimal(text.replace(",", "").replace("$", "").strip())
        return str(value.normalize()) if value.is_finite() else None
    except (InvalidOperation, AttributeError):
        return None


def extract(text, flexible=False):
    if flexible:
        matches = re.findall(FLEXIBLE, text)
        value = next((a or b for a, b in reversed(matches)), None)
    else:
        match = re.search(STRICT, text)
        value = match.group(1) if match else None
    return number(value) if value is not None else None


def render(tokenizer, question, final_instruction=False):
    text = f"Q: {question}\nA: Let's think step by step."
    if final_instruction:
        text += FINAL_INSTRUCTION
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    return {"prompt": prompt, "token_ids": ids, "prompt_hash": digest(ids)}


def build(tokenizer, train, test, pins, final_instruction=False):
    if len(test) != 1319:
        raise ValueError("GSM8K test split must contain exactly 1319 questions")
    dev_ids = random.Random(20261018).sample(range(len(train)), 16)
    pilot_ids = random.Random(20261019).sample(range(len(test)), 200)

    def rows(data, indices, split):
        result = []
        for index in indices:
            row = data[index]
            answer = row["answer"].split("####")[-1].strip()
            gold = number(answer)
            if gold is None:
                raise ValueError("invalid official numeric answer")
            result.append(
                {
                    "id": f"{split}-{index:04d}",
                    "dataset_index": index,
                    "question": row["question"],
                    "gold": gold,
                    "official_answer": row["answer"],
                    **render(tokenizer, row["question"], final_instruction),
                }
            )
        return result

    samples = rows(test, range(1319), "test")
    return {
        "schema": 1,
        "pins": pins,
        "task": "gsm8k_cot_zeroshot_chat_adaptation",
        "thinking": False,
        "strict_regex": STRICT,
        "flexible_regex": FLEXIBLE,
        "final_instruction": final_instruction,
        "max_output_tokens": 2048,
        "stop": ["Q:", "</s>", "<|im_end|>"],
        "temperature": 0,
        "baseline_accuracy_floor": 0.5,
        "loss_margin": 0.02,
        "pilot_ids": [f"test-{i:04d}" for i in pilot_ids],
        "parser_failure_audit_order": random.Random(20261020).sample(
            [r["id"] for r in samples], len(samples)
        ),
        "development": rows(train, dev_ids, "train"),
        "samples": samples,
        "sample_hash": digest(samples),
    }


def read(path):
    data = json.loads(Path(path).read_text())
    if len(data["samples"]) != 1319 or digest(data["samples"]) != data["sample_hash"]:
        raise ValueError("frozen GSM8K sample identity changed")
    expected = {f"test-{i:04d}" for i in range(1319)}
    if {r["id"] for r in data["samples"]} != expected:
        raise ValueError("GSM8K question IDs are incomplete or duplicated")
    if (
        len(data["pilot_ids"]) != 200
        or len(set(data["pilot_ids"])) != 200
        or not set(data["pilot_ids"]) <= expected
    ):
        raise ValueError(
            "GSM8K pilot is not the frozen independent 200-question subset"
        )
    if any(r["prompt_hash"] != digest(r["token_ids"]) for r in data["samples"]):
        raise ValueError("GSM8K prompt token identity changed")
    return data
