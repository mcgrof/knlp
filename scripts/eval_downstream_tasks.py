#!/usr/bin/env python3
"""
Evaluate KV compression on downstream tasks.

Tests compression impact on real-world benchmarks beyond perplexity.

Supported tasks:
- GSM8K: Math reasoning (accuracy on grade school math)
- ARC-Easy/Challenge: Science reasoning
- Winogrande: Commonsense reasoning
- HellaSwag: Commonsense NLI

Usage:
    python scripts/eval_downstream_tasks.py --model Qwen/Qwen2.5-7B \
        --preset kv_preset_qwen-qwen2.5-7b_v9.json --tasks gsm8k,arc_easy
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


def load_gsm8k(num_samples: int = 100) -> List[Dict]:
    """Load GSM8K test samples."""
    try:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="test")
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            # Extract numerical answer
            answer = item["answer"].split("####")[-1].strip()
            samples.append(
                {
                    "question": item["question"],
                    "answer": answer,
                }
            )
        return samples
    except Exception as e:
        print(f"Warning: Could not load GSM8K: {e}")
        return []


def load_arc(split: str = "easy", num_samples: int = 100) -> List[Dict]:
    """Load ARC (AI2 Reasoning Challenge) samples."""
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


def evaluate_gsm8k(
    model,
    tokenizer,
    samples: List[Dict],
    device: str,
    cache=None,
    max_new_tokens: int = 256,
) -> Tuple[float, List[Dict]]:
    """
    Evaluate GSM8K accuracy.

    Uses chain-of-thought prompting and extracts final numerical answer.
    """
    correct = 0
    results = []

    for sample in tqdm(samples, desc="GSM8K"):
        prompt = f"""Solve this math problem step by step.

Question: {sample['question']}

Solution: Let me solve this step by step.
"""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Reset cache for each sample
        if cache is not None:
            cache.reset()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                past_key_values=cache,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt) :]

        # Extract numerical answer from response
        predicted = extract_number(response)
        expected = sample["answer"]

        is_correct = predicted == expected
        if is_correct:
            correct += 1

        results.append(
            {
                "question": sample["question"][:100],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(samples) if samples else 0
    return accuracy, results


def evaluate_arc(
    model,
    tokenizer,
    samples: List[Dict],
    device: str,
    cache=None,
) -> Tuple[float, List[Dict]]:
    """
    Evaluate ARC multiple choice accuracy.

    Uses likelihood scoring for each choice.
    """
    correct = 0
    results = []

    for sample in tqdm(samples, desc="ARC"):
        question = sample["question"]
        choices = sample["choices"]
        labels = sample["labels"]
        expected = sample["answer"]

        # Score each choice by perplexity
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
                loss = outputs.loss.item()
                scores.append(-loss)  # Higher score = better

        predicted_idx = scores.index(max(scores))
        predicted = labels[predicted_idx]

        is_correct = predicted == expected
        if is_correct:
            correct += 1

        results.append(
            {
                "question": question[:100],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(samples) if samples else 0
    return accuracy, results


def evaluate_hellaswag(
    model,
    tokenizer,
    samples: List[Dict],
    device: str,
    cache=None,
) -> Tuple[float, List[Dict]]:
    """
    Evaluate HellaSwag accuracy.

    Uses likelihood scoring for each ending.
    """
    correct = 0
    results = []

    for sample in tqdm(samples, desc="HellaSwag"):
        context = sample["context"]
        endings = sample["endings"]
        expected = sample["answer"]

        # Score each ending
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
                loss = outputs.loss.item()
                scores.append(-loss)

        predicted = scores.index(max(scores))

        is_correct = predicted == expected
        if is_correct:
            correct += 1

        results.append(
            {
                "context": context[:100],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(samples) if samples else 0
    return accuracy, results


def extract_number(text: str) -> str:
    """Extract the last number from text (for GSM8K answers)."""
    # Look for patterns like "= 42" or "is 42" or just standalone numbers
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        # Return last number, removing commas
        return numbers[-1].replace(",", "")
    return ""


def load_preset_cache(
    preset_path: str,
    num_layers: int,
    device: str = "cuda",
) -> Optional[CompressedDynamicCache]:
    """Load compression cache from preset."""
    try:
        with open(preset_path) as f:
            preset = json.load(f)

        calib_path = preset["calibration_file"]
        quantize_bits = preset.get("bits")
        if quantize_bits == 16:
            quantize_bits = None

        k_comp, v_comp, metadata = load_calibrated_compressors(
            calib_path,
            device=torch.device(device),
            dtype=torch.float16,
            quantize_bits=quantize_bits,
        )

        # Apply target filter
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
    parser = argparse.ArgumentParser(description="Evaluate downstream tasks")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Compression preset JSON (omit for baseline)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="gsm8k,arc_easy",
        help="Comma-separated tasks to evaluate",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples per task",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file",
    )
    args = parser.parse_args()

    print(f"Downstream Task Evaluation")
    print(f"  Model: {args.model}")
    print(f"  Preset: {args.preset or 'baseline'}")
    print(f"  Tasks: {args.tasks}")
    print(f"  Samples per task: {args.num_samples}")
    print("=" * 70)

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers

    # Load cache if preset provided
    cache = None
    if args.preset:
        print(f"Loading compression preset...")
        cache = load_preset_cache(args.preset, num_layers, args.device)
        if cache:
            with open(args.preset) as f:
                preset = json.load(f)
            print(f"  Compression: {preset.get('total_compression', 'N/A')}x")

    # Evaluate tasks
    tasks = args.tasks.split(",")
    all_results = {}

    for task in tasks:
        task = task.strip().lower()
        print(f"\n--- Evaluating {task.upper()} ---")

        if task == "gsm8k":
            samples = load_gsm8k(args.num_samples)
            if samples:
                acc, results = evaluate_gsm8k(
                    model, tokenizer, samples, args.device, cache
                )
                all_results["gsm8k"] = {"accuracy": acc, "samples": len(samples)}
                print(f"  Accuracy: {acc*100:.1f}%")

        elif task == "arc_easy":
            samples = load_arc("easy", args.num_samples)
            if samples:
                acc, results = evaluate_arc(
                    model, tokenizer, samples, args.device, cache
                )
                all_results["arc_easy"] = {"accuracy": acc, "samples": len(samples)}
                print(f"  Accuracy: {acc*100:.1f}%")

        elif task == "arc_challenge":
            samples = load_arc("challenge", args.num_samples)
            if samples:
                acc, results = evaluate_arc(
                    model, tokenizer, samples, args.device, cache
                )
                all_results["arc_challenge"] = {
                    "accuracy": acc,
                    "samples": len(samples),
                }
                print(f"  Accuracy: {acc*100:.1f}%")

        elif task == "hellaswag":
            samples = load_hellaswag(args.num_samples)
            if samples:
                acc, results = evaluate_hellaswag(
                    model, tokenizer, samples, args.device, cache
                )
                all_results["hellaswag"] = {"accuracy": acc, "samples": len(samples)}
                print(f"  Accuracy: {acc*100:.1f}%")

        else:
            print(f"  Unknown task: {task}")

    # Summary
    print(f"\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Task':<20} {'Accuracy':<15} {'Samples'}")
    print("-" * 50)
    for task, result in all_results.items():
        print(f"{task:<20} {result['accuracy']*100:.1f}%{'':<10} {result['samples']}")

    # Save results
    if args.output:
        output_data = {
            "model": args.model,
            "preset": args.preset,
            "results": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
