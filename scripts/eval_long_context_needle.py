#!/usr/bin/env python3
"""
Needle-in-a-Haystack Long-Context Evaluation.

Tests whether KV compression affects the model's ability to retrieve
information from long contexts.

The test:
1. Generates a "haystack" of filler text
2. Inserts a "needle" (specific fact) at various positions
3. Asks a question about the needle at the end
4. Measures if the model retrieves the correct answer

Usage:
    python scripts/eval_long_context_needle.py --model Qwen/Qwen2.5-0.5B
    python scripts/eval_long_context_needle.py --model Qwen/Qwen2.5-7B \
        --context-lengths 1024,2048,4096 \
        --preset kv_preset_qwen-qwen2.5-7b_v9.json
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    load_calibrated_compressors,
)


# Needle templates - facts that can be verified
NEEDLE_TEMPLATES = [
    {
        "needle": "The secret code for the vault is {value}.",
        "question": "What is the secret code for the vault?",
        "answer": "{value}",
        "values": ["7392", "4851", "2069", "8476", "3215"],
    },
    {
        "needle": "The capital of the fictional country Zephyria is {value}.",
        "question": "What is the capital of Zephyria?",
        "answer": "{value}",
        "values": ["Windholm", "Crystalburg", "Starview", "Moondale", "Sunridge"],
    },
    {
        "needle": "Professor Smith discovered that the optimal temperature is {value} degrees.",
        "question": "What temperature did Professor Smith discover was optimal?",
        "answer": "{value}",
        "values": ["42", "73", "28", "91", "56"],
    },
]

# Filler text for the haystack
FILLER_PARAGRAPHS = [
    "Machine learning models have become increasingly sophisticated in recent years. "
    "Deep neural networks can now perform tasks that were once thought impossible. "
    "From image recognition to natural language processing, AI is transforming industries.",
    "The history of computing spans several decades of rapid innovation. "
    "From room-sized mainframes to pocket-sized smartphones, technology has evolved dramatically. "
    "Moore's Law predicted the exponential growth of transistor density.",
    "Climate scientists continue to study the effects of global warming. "
    "Rising sea levels and extreme weather events are concerning trends. "
    "International cooperation is essential for addressing these challenges.",
    "The field of medicine has seen remarkable advances in recent decades. "
    "Gene therapy and personalized medicine offer new hope for patients. "
    "Vaccines have been instrumental in controlling infectious diseases.",
    "Space exploration has captured the human imagination for generations. "
    "From the Moon landings to Mars rovers, we continue to push boundaries. "
    "Private companies are now entering the space industry alongside government agencies.",
    "The global economy is interconnected in complex ways. "
    "Supply chain disruptions can have far-reaching effects. "
    "Digital currencies and blockchain technology are creating new financial systems.",
    "Education systems around the world are adapting to new technologies. "
    "Online learning has become more prevalent and accessible. "
    "Critical thinking and problem-solving skills are increasingly valued.",
    "Renewable energy sources are becoming more cost-effective. "
    "Solar and wind power are growing rapidly in many countries. "
    "Energy storage technology is key to grid stability.",
]


def generate_haystack(target_tokens: int, tokenizer) -> str:
    """Generate filler text of approximately target_tokens length."""
    text = ""
    while True:
        paragraph = random.choice(FILLER_PARAGRAPHS)
        text += paragraph + "\n\n"
        tokens = len(tokenizer.encode(text))
        if tokens >= target_tokens:
            break
    return text


def create_needle_test(
    context_length: int,
    needle_position: float,  # 0.0 = start, 0.5 = middle, 1.0 = end
    tokenizer,
) -> Tuple[str, str, str]:
    """
    Create a needle-in-haystack test case.

    Returns:
        (full_prompt, needle_sentence, expected_answer)
    """
    # Select random needle template
    template = random.choice(NEEDLE_TEMPLATES)
    value = random.choice(template["values"])

    needle = template["needle"].format(value=value)
    question = template["question"]
    answer = template["answer"].format(value=value)

    # Calculate haystack size (leave room for needle and question)
    needle_tokens = len(tokenizer.encode(needle))
    question_tokens = len(tokenizer.encode(question)) + 50  # Buffer
    haystack_tokens = context_length - needle_tokens - question_tokens

    if haystack_tokens < 100:
        haystack_tokens = 100  # Minimum

    # Generate haystack
    full_haystack = generate_haystack(int(haystack_tokens * 1.2), tokenizer)

    # Split haystack at needle position
    haystack_chars = len(full_haystack)
    split_point = int(haystack_chars * needle_position)

    # Find a good split point (paragraph boundary)
    for i in range(split_point, min(split_point + 200, haystack_chars)):
        if full_haystack[i : i + 2] == "\n\n":
            split_point = i + 2
            break

    before = full_haystack[:split_point]
    after = full_haystack[split_point:]

    # Trim to fit context
    combined = before + needle + "\n\n" + after
    tokens = tokenizer.encode(combined)
    if len(tokens) > context_length - question_tokens:
        # Truncate the after part
        target_combined_tokens = context_length - question_tokens
        combined = tokenizer.decode(tokens[:target_combined_tokens])

    # Add question
    prompt = combined + f"\n\nQuestion: {question}\nAnswer:"

    return prompt, needle, answer


def evaluate_retrieval(
    model,
    tokenizer,
    prompt: str,
    expected_answer: str,
    device: str,
    cache: Optional[CompressedDynamicCache] = None,
    max_new_tokens: int = 20,
) -> Tuple[bool, str, float]:
    """
    Evaluate if model can retrieve the needle.

    Returns:
        (success, generated_text, confidence)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs.input_ids.to(device)

    with torch.no_grad():
        if cache is not None:
            # First pass to fill cache
            outputs = model(input_ids, past_key_values=cache, use_cache=True)
            past = outputs.past_key_values

            # Generate
            generated_ids = input_ids
            for _ in range(max_new_tokens):
                outputs = model(
                    generated_ids[:, -1:], past_key_values=past, use_cache=True
                )
                past = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        else:
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    generated_text = tokenizer.decode(
        generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True
    )
    generated_text = generated_text.strip()

    # Check if answer is in generated text
    success = expected_answer.lower() in generated_text.lower()

    # Calculate confidence (simple heuristic)
    confidence = 1.0 if success else 0.0

    return success, generated_text, confidence


def run_needle_test(
    model_name: str,
    context_lengths: List[int],
    needle_positions: List[float],
    num_trials: int,
    preset_path: Optional[str] = None,
    device: str = "cuda",
) -> Dict:
    """
    Run needle-in-haystack evaluation.

    Args:
        model_name: HuggingFace model name
        context_lengths: List of context lengths to test
        needle_positions: List of needle positions (0.0-1.0)
        num_trials: Number of trials per configuration
        preset_path: Path to compression preset JSON
        device: Device to use

    Returns:
        Results dict
    """
    print(f"Needle-in-Haystack Long-Context Evaluation")
    print(f"  Model: {model_name}")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Needle positions: {needle_positions}")
    print(f"  Trials per config: {num_trials}")
    if preset_path:
        print(f"  Preset: {preset_path}")
    print("=" * 70)

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers

    # Load preset if provided
    compressors = None
    preset_info = None
    if preset_path:
        with open(preset_path) as f:
            preset_info = json.load(f)

        k_comp, v_comp, metadata = load_calibrated_compressors(
            preset_info["calibration_file"],
            device=torch.device(device),
            dtype=torch.float16,
            quantize_bits=preset_info["bits"] if preset_info["bits"] < 16 else None,
        )

        # Apply target filter
        if preset_info["target"] == "v":
            k_comp = [IdentityCompressor() for _ in range(num_layers)]
        elif preset_info["target"] == "k":
            v_comp = [IdentityCompressor() for _ in range(num_layers)]

        compressors = (k_comp, v_comp)
        print(f"  Loaded preset: {preset_info['total_compression']:.2f}x compression")

    results = {
        "model": model_name,
        "preset": preset_path,
        "baseline": {},
        "compressed": {},
    }

    for ctx_len in context_lengths:
        print(f"\n--- Context Length: {ctx_len} tokens ---")

        for pos in needle_positions:
            pos_name = f"pos_{int(pos*100)}"

            baseline_successes = 0
            compressed_successes = 0

            for trial in range(num_trials):
                # Create test case
                random.seed(trial + ctx_len + int(pos * 1000))
                prompt, needle, answer = create_needle_test(ctx_len, pos, tokenizer)

                # Test baseline
                success_b, gen_b, _ = evaluate_retrieval(
                    model, tokenizer, prompt, answer, device, cache=None
                )
                baseline_successes += int(success_b)

                # Test compressed (if preset provided)
                if compressors:
                    cache = CompressedDynamicCache(
                        compressors[0], compressors[1], num_layers
                    )
                    success_c, gen_c, _ = evaluate_retrieval(
                        model, tokenizer, prompt, answer, device, cache=cache
                    )
                    compressed_successes += int(success_c)
                    del cache
                    torch.cuda.empty_cache()

            baseline_rate = baseline_successes / num_trials
            compressed_rate = compressed_successes / num_trials if compressors else None

            key = f"ctx_{ctx_len}"
            if key not in results["baseline"]:
                results["baseline"][key] = {}
                results["compressed"][key] = {}

            results["baseline"][key][pos_name] = baseline_rate
            if compressed_rate is not None:
                results["compressed"][key][pos_name] = compressed_rate

            if compressors:
                print(
                    f"  Position {pos:.1f}: Baseline={baseline_rate:.0%}, Compressed={compressed_rate:.0%}"
                )
            else:
                print(f"  Position {pos:.1f}: Baseline={baseline_rate:.0%}")

    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Context':<12} {'Position':<12} {'Baseline':<12} {'Compressed':<12} {'Diff':<12}"
    )
    print("-" * 60)

    for ctx_len in context_lengths:
        key = f"ctx_{ctx_len}"
        for pos in needle_positions:
            pos_name = f"pos_{int(pos*100)}"
            baseline = results["baseline"][key][pos_name]
            compressed = results["compressed"].get(key, {}).get(pos_name, None)

            if compressed is not None:
                diff = compressed - baseline
                print(
                    f"{ctx_len:<12} {pos:<12.1f} {baseline:<12.0%} {compressed:<12.0%} {diff:+.0%}"
                )
            else:
                print(f"{ctx_len:<12} {pos:<12.1f} {baseline:<12.0%} {'N/A':<12}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Needle-in-haystack long-context evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model to test",
    )
    parser.add_argument(
        "--context-lengths",
        type=str,
        default="512,1024,2048",
        help="Comma-separated context lengths to test",
    )
    parser.add_argument(
        "--needle-positions",
        type=str,
        default="0.1,0.5,0.9",
        help="Comma-separated needle positions (0.0-1.0)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of trials per configuration",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Compression preset JSON file",
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
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    needle_positions = [float(x) for x in args.needle_positions.split(",")]

    results = run_needle_test(
        model_name=args.model,
        context_lengths=context_lengths,
        needle_positions=needle_positions,
        num_trials=args.num_trials,
        preset_path=args.preset,
        device=args.device,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
