#!/usr/bin/env python3
"""
Task Accuracy Evaluation Script for KV Plugin

Evaluates accuracy on standard benchmarks:
- GSM8K (math reasoning)
- Winogrande (commonsense)
- RTE (natural language inference)
- PIQA (physical commonsense)
- CommonsenseQA (multiple choice)

Follows evaluation methodology from:
- Palu (ICLR 2025)
- MiniCache (NeurIPS 2024)
- PyramidKV (NeurIPS 2024)
- AsymKV (NeurIPS 2025)

Usage:
    python scripts/eval_tasks.py --model gpt2 --tasks gsm8k winogrande
    python scripts/eval_tasks.py --model Qwen/Qwen2.5-7B-Instruct --preset orthogonal_int4
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from datasets import load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from gpt2.compression.kv_plugin import KVPlugin


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TaskEvaluator:
    """Base class for task evaluators."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(self, max_samples: int = 100) -> Dict:
        """Run evaluation and return results."""
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate text from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated = outputs[0, inputs.input_ids.size(1) :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    @torch.no_grad()
    def get_logprobs(self, prompt: str, choices: List[str]) -> List[float]:
        """Get log probabilities for multiple choice answers."""
        logprobs = []

        for choice in choices:
            full_text = prompt + choice
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)

            outputs = self.model(inputs.input_ids)
            logits = outputs.logits[0, -len(self.tokenizer.encode(choice)) - 1 : -1]

            choice_ids = self.tokenizer.encode(choice, add_special_tokens=False)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            total_logprob = 0.0
            for i, token_id in enumerate(choice_ids):
                if i < len(log_probs):
                    total_logprob += log_probs[i, token_id].item()

            logprobs.append(total_logprob)

        return logprobs


class GSM8KEvaluator(TaskEvaluator):
    """GSM8K math reasoning evaluation."""

    def evaluate(self, max_samples: int = 100) -> Dict:
        dataset = load_dataset("gsm8k", "main", split="test")

        correct = 0
        total = 0
        results = []

        for item in tqdm(list(dataset)[:max_samples], desc="GSM8K"):
            question = item["question"]
            answer = item["answer"]

            # Extract final answer
            final_answer = answer.split("####")[-1].strip()

            # Create prompt
            prompt = f"Question: {question}\n\nSolve this step by step and give the final answer.\n\nAnswer:"

            # Generate
            response = self.generate(prompt, max_new_tokens=512)

            # Extract predicted answer
            predicted = self._extract_number(response)
            expected = self._extract_number(final_answer)

            is_correct = predicted == expected
            if is_correct:
                correct += 1
            total += 1

            results.append(
                {
                    "question": question,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": is_correct,
                }
            )

        return {
            "task": "gsm8k",
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "examples": results[:5],
        }

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract final number from text."""
        numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        return None


class WinograndeEvaluator(TaskEvaluator):
    """Winogrande commonsense evaluation."""

    def evaluate(self, max_samples: int = 100) -> Dict:
        dataset = load_dataset("winogrande", "winogrande_xl", split="validation")

        correct = 0
        total = 0

        for item in tqdm(list(dataset)[:max_samples], desc="Winogrande"):
            sentence = item["sentence"]
            option1 = item["option1"]
            option2 = item["option2"]
            answer = int(item["answer"]) - 1  # 1-indexed to 0-indexed

            # Create filled sentences
            sent1 = sentence.replace("_", option1)
            sent2 = sentence.replace("_", option2)

            # Get log probabilities
            logprobs = self.get_logprobs("Complete: ", [sent1, sent2])

            predicted = 0 if logprobs[0] > logprobs[1] else 1

            if predicted == answer:
                correct += 1
            total += 1

        return {
            "task": "winogrande",
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
        }


class PIQAEvaluator(TaskEvaluator):
    """PIQA physical commonsense evaluation."""

    def evaluate(self, max_samples: int = 100) -> Dict:
        dataset = load_dataset("piqa", split="validation")

        correct = 0
        total = 0

        for item in tqdm(list(dataset)[:max_samples], desc="PIQA"):
            goal = item["goal"]
            sol1 = item["sol1"]
            sol2 = item["sol2"]
            answer = item["label"]

            prompt = f"Goal: {goal}\n\nWhich solution is better?\n"

            logprobs = self.get_logprobs(prompt, [sol1, sol2])
            predicted = 0 if logprobs[0] > logprobs[1] else 1

            if predicted == answer:
                correct += 1
            total += 1

        return {
            "task": "piqa",
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
        }


class RTEEvaluator(TaskEvaluator):
    """RTE natural language inference evaluation."""

    def evaluate(self, max_samples: int = 100) -> Dict:
        dataset = load_dataset("glue", "rte", split="validation")

        correct = 0
        total = 0

        for item in tqdm(list(dataset)[:max_samples], desc="RTE"):
            premise = item["sentence1"]
            hypothesis = item["sentence2"]
            label = item["label"]

            prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\n\nDoes the premise entail the hypothesis?"

            logprobs = self.get_logprobs(prompt + " Answer: ", ["Yes", "No"])
            predicted = 0 if logprobs[0] > logprobs[1] else 1

            if predicted == label:
                correct += 1
            total += 1

        return {
            "task": "rte",
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
        }


class CommonsenseQAEvaluator(TaskEvaluator):
    """CommonsenseQA multiple choice evaluation."""

    def evaluate(self, max_samples: int = 100) -> Dict:
        dataset = load_dataset("commonsense_qa", split="validation")

        correct = 0
        total = 0

        for item in tqdm(list(dataset)[:max_samples], desc="CommonsenseQA"):
            question = item["question"]
            choices = item["choices"]["text"]
            answer_key = item["answerKey"]
            answer_idx = ord(answer_key) - ord("A")

            prompt = f"Question: {question}\n\nChoices:\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(ord('A') + i)}) {choice}\n"
            prompt += "\nAnswer: "

            # Get logprobs for each choice letter
            choice_letters = [chr(ord("A") + i) for i in range(len(choices))]
            logprobs = self.get_logprobs(prompt, choice_letters)

            predicted = logprobs.index(max(logprobs))

            if predicted == answer_idx:
                correct += 1
            total += 1

        return {
            "task": "commonsense_qa",
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
        }


EVALUATORS = {
    "gsm8k": GSM8KEvaluator,
    "winogrande": WinograndeEvaluator,
    "piqa": PIQAEvaluator,
    "rte": RTEEvaluator,
    "commonsense_qa": CommonsenseQAEvaluator,
}


def run_evaluation(
    model_name: str,
    preset: str = "none",
    tasks: List[str] = ["gsm8k"],
    max_samples: int = 100,
    output_file: Optional[str] = None,
) -> Dict:
    """
    Run task evaluation.

    Args:
        model_name: HuggingFace model name
        preset: KV plugin preset
        tasks: List of tasks to evaluate
        max_samples: Max samples per task
        output_file: Optional output file

    Returns:
        Dict with results
    """
    device = get_device()
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    print(f"Preset: {preset}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if device != "cuda":
        model = model.to(device)

    model.eval()

    # Create plugin if needed
    plugin = None
    if preset != "none":
        print(f"Creating KV plugin with preset: {preset}")
        plugin = KVPlugin.from_preset(preset, model)

    # Run evaluations
    results = {
        "model": model_name,
        "preset": preset,
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "tasks": {},
    }

    for task_name in tasks:
        if task_name not in EVALUATORS:
            print(f"Unknown task: {task_name}")
            continue

        print(f"\nEvaluating {task_name}...")
        evaluator = EVALUATORS[task_name](model, tokenizer, device)

        try:
            task_results = evaluator.evaluate(max_samples)
            results["tasks"][task_name] = task_results
            print(f"  Accuracy: {task_results['accuracy']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            results["tasks"][task_name] = {"error": str(e)}

    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate task accuracy with KV compression"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model name",
    )
    parser.add_argument("--preset", type=str, default="none", help="KV plugin preset")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["gsm8k"],
        choices=list(EVALUATORS.keys()),
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--max-samples", type=int, default=100, help="Max samples per task"
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    if not HF_AVAILABLE:
        print("Error: transformers library required")
        sys.exit(1)

    if not DATASETS_AVAILABLE:
        print("Error: datasets library required")
        sys.exit(1)

    results = run_evaluation(
        model_name=args.model,
        preset=args.preset,
        tasks=args.tasks,
        max_samples=args.max_samples,
        output_file=args.output,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("TASK ACCURACY RESULTS")
    print("=" * 60)
    print(f"Model: {results['model']}")
    print(f"Preset: {results['preset']}")
    print("-" * 60)

    for name, data in results["tasks"].items():
        if "error" in data:
            print(f"{name}: ERROR - {data['error']}")
        else:
            print(f"{name}: {data['accuracy']:.4f} ({data['correct']}/{data['total']})")


if __name__ == "__main__":
    main()
