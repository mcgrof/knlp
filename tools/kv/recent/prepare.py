"""Freeze recent-model inputs before any candidate output is observed."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from tools.kv.native import gsm8k
from tools.kv.prequantized.fixtures import prepare_samples

MODELS = {
    "qwen38": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    "gemma4": ("google/gemma-4-26B-A4B-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"),
}
WIKI = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
GSM = "740312add88f781978c0658806c59bc2815b9866"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    model, revision = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    config = AutoConfig.from_pretrained(model, revision=revision)
    samples, selected, calibration, evaluation = prepare_samples(
        tokenizer,
        {"quality": {"input_tokens": [1024, 8192]}},
        model,
        WIKI,
        "wikimedia/wikipedia",
    )
    samples = [r for r in samples if not r.get("reserve", False)]
    train = list(load_dataset("openai/gsm8k", "main", split="train", revision=GSM))
    test = list(load_dataset("openai/gsm8k", "main", split="test", revision=GSM))
    task = gsm8k.build(
        tokenizer,
        train,
        test,
        {"model_revision": revision, "dataset_revision": GSM},
        final_instruction=True,
    )
    # Correct literal terminator and allow harmless Markdown around the label.
    # Freeze before any output. No post-hoc rescoring determines admission.
    task["primary_parser"] = "last_numeric_answer_label_v1"
    task["primary_regex"] = (
        r"(?i)the\s+answer\s+is\s*[:=]?\s*\**\s*\$?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)"
    )
    task["primary_parser_reason"] = (
        "Avoid the prior unescaped terminal-dot and Markdown formatting failure; require an explicit answer label."
    )
    output = {
        "model": model,
        "revision": revision,
        "config": config.to_dict(),
        "samples": samples,
        "selection": selected,
        "calibration_ids": calibration,
        "evaluation_ids": evaluation,
        "wiki_revision": WIKI,
        "gsm8k": task,
    }
    encoded = json.dumps(output, allow_nan=False).encode()
    data = gzip.compress(encoded, mtime=0)
    path = args.out / "inputs.json.gz"
    path.write_bytes(data)
    tokenizer.save_pretrained(args.out / "tokenizer")
    (args.out / "identity.json").write_text(
        json.dumps(
            {
                "model": model,
                "revision": revision,
                "inputs_sha256": hashlib.sha256(data).hexdigest(),
                "samples": len(samples),
                "gsm8k": len(task["samples"]),
                "calibration_used": False,
            },
            indent=2,
        )
        + "\n"
    )
    print((args.out / "identity.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
