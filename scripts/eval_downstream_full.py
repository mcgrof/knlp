#!/usr/bin/env python3
"""
Full Downstream Benchmark Suite for KV Compression.

Evaluates compression impact on multiple benchmarks:
- GSM8K (math reasoning)
- MMLU (multi-task knowledge)
- ARC-Easy/Challenge (science reasoning)
- HellaSwag (commonsense NLI)
- Winogrande (commonsense reasoning)
- PIQA (physical intuition)
- TruthfulQA (truthfulness)

Usage:
    python scripts/eval_downstream_full.py --model Qwen/Qwen2.5-7B \
        --preset kv_preset_qwen-qwen2.5-7b_v9.json \
        --tasks all
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    load_calibrated_compressors,
)


ALL_TASKS = [
    "gsm8k",
    "mmlu",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "piqa",
    "truthfulqa",
]


# Dataset loaders


def load_gsm8k(num_samples: int = 100) -> List[Dict]:
    """Load GSM8K test samples."""
    try:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="test")
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            answer = item["answer"].split("####")[-1].strip()
            samples.append({"question": item["question"], "answer": answer})
        return samples
    except Exception as e:
        print(f"Warning: Could not load GSM8K: {e}")
        return []


def load_mmlu(num_samples: int = 100, subject: str = None) -> List[Dict]:
    """Load MMLU test samples."""
    try:
        from datasets import load_dataset

        # Load all subjects or specific one
        if subject:
            ds = load_dataset("cais/mmlu", subject, split="test")
        else:
            ds = load_dataset("cais/mmlu", "all", split="test")

        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            samples.append(
                {
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],  # Index of correct choice
                }
            )
        return samples
    except Exception as e:
        print(f"Warning: Could not load MMLU: {e}")
        return []


def load_arc(split: str = "easy", num_samples: int = 100) -> List[Dict]:
    """Load ARC samples."""
    try:
        from datasets import load_dataset

        config = "ARC-Easy" if split == "easy" else "ARC-Challenge"
        ds = load_dataset("ai2_arc", config, split="test")
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            choices = item["choices"]
            samples.append(
                {
                    "question": item["question"],
                    "choices": choices["text"],
                    "labels": choices["label"],
                    "answer": item["answerKey"],
                }
            )
        return samples
    except Exception as e:
        print(f"Warning: Could not load ARC: {e}")
        return []


def load_hellaswag(num_samples: int = 100) -> List[Dict]:
    """Load HellaSwag samples."""
    try:
        from datasets import load_dataset

        ds = load_dataset("hellaswag", split="validation")
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            samples.append(
                {
                    "context": item["ctx"],
                    "endings": item["endings"],
                    "answer": int(item["label"]),
                }
            )
        return samples
    except Exception as e:
        print(f"Warning: Could not load HellaSwag: {e}")
        return []


def load_winogrande(num_samples: int = 100) -> List[Dict]:
    """Load Winogrande samples."""
    try:
        from datasets import load_dataset

        ds = load_dataset("winogrande", "winogrande_xl", split="validation")
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            samples.append(
                {
                    "sentence": item["sentence"],
                    "option1": item["option1"],
                    "option2": item["option2"],
                    "answer": int(item["answer"]) - 1,  # Convert 1/2 to 0/1
                }
            )
        return samples
    except Exception as e:
        print(f"Warning: Could not load Winogrande: {e}")
        return []


def load_piqa(num_samples: int = 100) -> List[Dict]:
    """Load PIQA samples."""
    try:
        from datasets import load_dataset

        ds = load_dataset("piqa", split="validation")
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            samples.append(
                {
                    "goal": item["goal"],
                    "sol1": item["sol1"],
                    "sol2": item["sol2"],
                    "answer": item["label"],
                }
            )
        return samples
    except Exception as e:
        print(f"Warning: Could not load PIQA: {e}")
        return []


def load_truthfulqa(num_samples: int = 100) -> List[Dict]:
    """Load TruthfulQA samples."""
    try:
        from datasets import load_dataset

        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            # mc1 format: single correct answer
            samples.append(
                {
                    "question": item["question"],
                    "choices": item["mc1_targets"]["choices"],
                    "labels": item["mc1_targets"][
                        "labels"
                    ],  # 1 for correct, 0 for wrong
                }
            )
        return samples
    except Exception as e:
        print(f"Warning: Could not load TruthfulQA: {e}")
        return []


# Evaluation functions


def evaluate_gsm8k(model, tokenizer, samples, device, cache=None) -> Tuple[float, List]:
    """Evaluate GSM8K accuracy.

    Note: GSM8K uses generation, which is incompatible with compressed KV cache.
    Cache is ignored for this task to ensure correct generation.
    """
    correct = 0
    results = []

    for sample in tqdm(samples, desc="GSM8K"):
        prompt = (
            f"Question: {sample['question']}\nAnswer: Let me solve this step by step.\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Note: Skip compressed cache for generation tasks - cache dimension
        # mismatches occur during autoregressive generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt) :]
        predicted = extract_number(response)
        is_correct = predicted == sample["answer"]
        if is_correct:
            correct += 1
        results.append(
            {
                "expected": sample["answer"],
                "predicted": predicted,
                "correct": is_correct,
            }
        )

    return correct / len(samples) if samples else 0, results


def evaluate_mmlu(model, tokenizer, samples, device, cache=None) -> Tuple[float, List]:
    """Evaluate MMLU accuracy."""
    correct = 0
    results = []

    for sample in tqdm(samples, desc="MMLU"):
        question = sample["question"]
        choices = sample["choices"]
        expected = sample["answer"]

        # Score each choice
        scores = []
        for i, choice in enumerate(choices):
            prompt = f"Question: {question}\nAnswer: {choice}"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            if cache is not None:
                cache.reset()

            with torch.no_grad():
                outputs = model(
                    **inputs, labels=inputs.input_ids, past_key_values=cache
                )
                scores.append(-outputs.loss.item())

        predicted = scores.index(max(scores))
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        results.append(
            {"expected": expected, "predicted": predicted, "correct": is_correct}
        )

    return correct / len(samples) if samples else 0, results


def evaluate_arc(model, tokenizer, samples, device, cache=None) -> Tuple[float, List]:
    """Evaluate ARC accuracy."""
    correct = 0
    results = []

    for sample in tqdm(samples, desc="ARC"):
        question = sample["question"]
        choices = sample["choices"]
        labels = sample["labels"]
        expected = sample["answer"]

        scores = []
        for choice in choices:
            prompt = f"Question: {question}\nAnswer: {choice}"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            if cache is not None:
                cache.reset()

            with torch.no_grad():
                outputs = model(
                    **inputs, labels=inputs.input_ids, past_key_values=cache
                )
                scores.append(-outputs.loss.item())

        predicted_idx = scores.index(max(scores))
        predicted = labels[predicted_idx]
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        results.append(
            {"expected": expected, "predicted": predicted, "correct": is_correct}
        )

    return correct / len(samples) if samples else 0, results


def evaluate_hellaswag(
    model, tokenizer, samples, device, cache=None
) -> Tuple[float, List]:
    """Evaluate HellaSwag accuracy."""
    correct = 0
    results = []

    for sample in tqdm(samples, desc="HellaSwag"):
        context = sample["context"]
        endings = sample["endings"]
        expected = sample["answer"]

        scores = []
        for ending in endings:
            prompt = f"{context} {ending}"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            if cache is not None:
                cache.reset()

            with torch.no_grad():
                outputs = model(
                    **inputs, labels=inputs.input_ids, past_key_values=cache
                )
                scores.append(-outputs.loss.item())

        predicted = scores.index(max(scores))
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        results.append(
            {"expected": expected, "predicted": predicted, "correct": is_correct}
        )

    return correct / len(samples) if samples else 0, results


def evaluate_winogrande(
    model, tokenizer, samples, device, cache=None
) -> Tuple[float, List]:
    """Evaluate Winogrande accuracy."""
    correct = 0
    results = []

    for sample in tqdm(samples, desc="Winogrande"):
        sentence = sample["sentence"]
        options = [sample["option1"], sample["option2"]]
        expected = sample["answer"]

        scores = []
        for option in options:
            # Replace _ with the option
            filled = sentence.replace("_", option)
            inputs = tokenizer(filled, return_tensors="pt").to(device)

            if cache is not None:
                cache.reset()

            with torch.no_grad():
                outputs = model(
                    **inputs, labels=inputs.input_ids, past_key_values=cache
                )
                scores.append(-outputs.loss.item())

        predicted = scores.index(max(scores))
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        results.append(
            {"expected": expected, "predicted": predicted, "correct": is_correct}
        )

    return correct / len(samples) if samples else 0, results


def evaluate_piqa(model, tokenizer, samples, device, cache=None) -> Tuple[float, List]:
    """Evaluate PIQA accuracy."""
    correct = 0
    results = []

    for sample in tqdm(samples, desc="PIQA"):
        goal = sample["goal"]
        solutions = [sample["sol1"], sample["sol2"]]
        expected = sample["answer"]

        scores = []
        for sol in solutions:
            prompt = f"Goal: {goal}\nSolution: {sol}"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            if cache is not None:
                cache.reset()

            with torch.no_grad():
                outputs = model(
                    **inputs, labels=inputs.input_ids, past_key_values=cache
                )
                scores.append(-outputs.loss.item())

        predicted = scores.index(max(scores))
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        results.append(
            {"expected": expected, "predicted": predicted, "correct": is_correct}
        )

    return correct / len(samples) if samples else 0, results


def evaluate_truthfulqa(
    model, tokenizer, samples, device, cache=None
) -> Tuple[float, List]:
    """Evaluate TruthfulQA accuracy."""
    correct = 0
    results = []

    for sample in tqdm(samples, desc="TruthfulQA"):
        question = sample["question"]
        choices = sample["choices"]
        labels = sample["labels"]

        scores = []
        for choice in choices:
            prompt = f"Question: {question}\nAnswer: {choice}"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            if cache is not None:
                cache.reset()

            with torch.no_grad():
                outputs = model(
                    **inputs, labels=inputs.input_ids, past_key_values=cache
                )
                scores.append(-outputs.loss.item())

        predicted_idx = scores.index(max(scores))
        expected_idx = labels.index(1)  # Index where label is 1
        is_correct = predicted_idx == expected_idx
        if is_correct:
            correct += 1
        results.append(
            {
                "expected": expected_idx,
                "predicted": predicted_idx,
                "correct": is_correct,
            }
        )

    return correct / len(samples) if samples else 0, results


def extract_number(text: str) -> str:
    """Extract the last number from text."""
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def load_preset_cache(preset_path, num_layers, device):
    """Load compression cache from preset."""
    try:
        with open(preset_path) as f:
            preset = json.load(f)

        calib_path = preset["calibration_file"]
        bits = preset.get("bits", 16)
        quantize_bits = bits if bits < 16 else None

        k_comp, v_comp, _ = load_calibrated_compressors(
            calib_path,
            device=torch.device(device),
            dtype=torch.float16,
            quantize_bits=quantize_bits,
        )

        target = preset.get("target", "v")
        if target == "k":
            v_comp = [IdentityCompressor() for _ in range(num_layers)]
        elif target == "v":
            k_comp = [IdentityCompressor() for _ in range(num_layers)]

        return CompressedDynamicCache(k_comp, v_comp, num_layers)
    except Exception as e:
        print(f"Error loading preset: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Full downstream benchmark suite")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--preset", type=str, default=None, help="Compression preset")
    parser.add_argument(
        "--tasks",
        type=str,
        default="gsm8k,arc_easy",
        help="Tasks to run (comma-sep or 'all')",
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("Full Downstream Benchmark Suite")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Preset: {args.preset or 'baseline'}")

    # Parse tasks
    if args.tasks.lower() == "all":
        tasks = ALL_TASKS
    else:
        tasks = [t.strip().lower() for t in args.tasks.split(",")]
    print(f"Tasks: {tasks}")

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device
    )
    model.eval()
    num_layers = model.config.num_hidden_layers

    # Load cache
    cache = None
    if args.preset:
        cache = load_preset_cache(args.preset, num_layers, args.device)
        print(f"\nLoaded compressed cache from {args.preset}")
        print(
            "Note: GSM8K skips cache (uses generation, incompatible with compression)"
        )

    # Run evaluations
    all_results = {}
    evaluators = {
        "gsm8k": (load_gsm8k, evaluate_gsm8k),
        "mmlu": (load_mmlu, evaluate_mmlu),
        "arc_easy": (lambda n: load_arc("easy", n), evaluate_arc),
        "arc_challenge": (lambda n: load_arc("challenge", n), evaluate_arc),
        "hellaswag": (load_hellaswag, evaluate_hellaswag),
        "winogrande": (load_winogrande, evaluate_winogrande),
        "piqa": (load_piqa, evaluate_piqa),
        "truthfulqa": (load_truthfulqa, evaluate_truthfulqa),
    }

    for task in tasks:
        if task not in evaluators:
            print(f"\nUnknown task: {task}")
            continue

        print(f"\n--- {task.upper()} ---")
        loader, evaluator = evaluators[task]
        samples = loader(args.num_samples)

        if samples:
            acc, results = evaluator(model, tokenizer, samples, args.device, cache)
            all_results[task] = {"accuracy": acc, "samples": len(samples)}
            print(f"  Accuracy: {acc*100:.1f}%")
        else:
            print(f"  No samples loaded")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Task':<20} {'Accuracy':<15} {'Samples'}")
    print("-" * 50)

    total_acc = 0
    n_tasks = 0
    for task, result in all_results.items():
        print(f"{task:<20} {result['accuracy']*100:.1f}%{'':<10} {result['samples']}")
        total_acc += result["accuracy"]
        n_tasks += 1

    if n_tasks > 0:
        print("-" * 50)
        print(f"{'Average':<20} {total_acc/n_tasks*100:.1f}%")

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {"model": args.model, "preset": args.preset, "results": all_results},
                f,
                indent=2,
            )
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
