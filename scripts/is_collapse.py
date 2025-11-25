#!/usr/bin/env python3
import argparse
import math
import sys
import os
from collections import Counter

import torch
import torch.nn.functional as F

# Add gpt2 directory to path to import model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gpt2"))
from model import GPT, GPTConfig

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken not installed. Install with: pip install tiktoken")
    sys.exit(1)


PROMPTS = [
    "The meaning of life is",
    "In a surprising turn of events,",
    "Once upon a time in a distant galaxy,",
    "The Linux kernel developer woke up and",
    "In deep learning, one of the key challenges is",
]


@torch.no_grad()
def sample_and_measure(
    model, enc, prompt, device, max_new_tokens=64, temperature=0.8, top_k=0
):
    model.eval()
    # Encode prompt
    input_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)

    # Generate tokens
    gen_ids = []
    curr_ids = input_ids

    for _ in range(max_new_tokens):
        # Forward pass
        logits, _ = model(curr_ids)
        logits = logits[:, -1, :] / temperature

        # Apply top-k if specified
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Compute entropy for this step
        entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)

        gen_ids.append((next_token.item(), entropy.item()))
        curr_ids = torch.cat([curr_ids, next_token], dim=1)

    # Extract tokens and entropies
    tokens = [t for t, _ in gen_ids]
    entropies = [e for _, e in gen_ids]

    # Diversity metrics
    unigrams = Counter(tokens)
    bigrams = Counter(zip(tokens, tokens[1:])) if len(tokens) > 1 else Counter()

    distinct1 = len(unigrams) / max(len(tokens), 1)
    distinct2 = len(bigrams) / max(len(tokens) - 1, 1) if len(tokens) > 1 else 0.0

    return {
        "prompt": prompt,
        "text": enc.decode(tokens),
        "avg_entropy": sum(entropies) / max(len(entropies), 1),
        "distinct1": distinct1,
        "distinct2": distinct2,
    }


def main():
    parser = argparse.ArgumentParser(description="Quick GPT-2 collapse sanity check.")
    parser.add_argument(
        "--model",
        default="gpt2",
        help="Model name or path (HF format, or .pt checkpoint)",
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=3,
        help="How many samples per prompt to draw.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--base-model",
        default="gpt2",
        help="Base model for tokenizer (when loading .pt checkpoint)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer (tiktoken for GPT-2)
    enc = tiktoken.get_encoding("gpt2")

    # Load checkpoint
    print(f"Loading checkpoint from {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    # Extract state dict (handle different checkpoint formats)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Create model architecture (GPT-2 small: 12 layers, 768 dim)
    config = GPTConfig.from_name(args.base_model)
    model = GPT2(config)

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Using device: {device}")
    print(f"Testing model: {args.model}")
    print("-" * 80)

    all_outputs = []

    for p in PROMPTS:
        print(f"\n=== PROMPT: {p!r} ===")
        for i in range(args.samples_per_prompt):
            res = sample_and_measure(
                model,
                enc,
                p,
                device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            all_outputs.append(res)

            print(f"\nSample {i+1}:")
            print(res["text"])
            print(
                f"[avg_entropy={res['avg_entropy']:.3f}  "
                f"distinct1={res['distinct1']:.3f}  "
                f"distinct2={res['distinct2']:.3f}]"
            )

    # Very crude global collapse indicators
    texts = [o["text"].strip() for o in all_outputs]
    unique_texts = len(set(texts))
    avg_entropy_global = sum(o["avg_entropy"] for o in all_outputs) / len(all_outputs)
    avg_distinct1 = sum(o["distinct1"] for o in all_outputs) / len(all_outputs)
    avg_distinct2 = sum(o["distinct2"] for o in all_outputs) / len(all_outputs)

    print("\n" + "-" * 80)
    print("GLOBAL STATS")
    print(f"Total samples: {len(all_outputs)}")
    print(f"Unique continuations: {unique_texts}")
    print(f"Avg entropy: {avg_entropy_global:.3f}")
    print(f"Avg distinct-1: {avg_distinct1:.3f}")
    print(f"Avg distinct-2: {avg_distinct2:.3f}")

    print("\nHeuristic interpretation:")
    print(
        " - If unique continuations is ~1–2 and avg_distinct1/2 ~0.0x, "
        "you're probably collapsed."
    )
    print(
        " - If entropy is extremely low (e.g. <1.0) across all steps and "
        "prompts produce nearly the same text, also suspect collapse."
    )
    print(
        " - Healthy models show varied outputs, decent distinct-1/2 "
        "and different shapes per prompt."
    )

    # Detect collapse and return error code
    is_collapsed = False
    reasons = []

    # Repetition collapse: very few unique outputs
    if unique_texts <= 2 and len(all_outputs) >= 5:
        is_collapsed = True
        reasons.append(f"repetition collapse (only {unique_texts} unique outputs)")

    # Low diversity collapse
    if avg_distinct1 < 0.1 and avg_distinct2 < 0.2:
        is_collapsed = True
        reasons.append(
            f"low diversity (distinct1={avg_distinct1:.3f}, distinct2={avg_distinct2:.3f})"
        )

    # Extremely low entropy (deterministic/stuck)
    if avg_entropy_global < 1.0:
        is_collapsed = True
        reasons.append(f"low entropy ({avg_entropy_global:.3f})")

    # Extremely high entropy (random noise)
    if avg_entropy_global > 9.0:
        is_collapsed = True
        reasons.append(f"random output (entropy={avg_entropy_global:.3f})")

    print("\n" + "=" * 80)
    if is_collapsed:
        print("COLLAPSE DETECTED:")
        for reason in reasons:
            print(f"  - {reason}")
        print("=" * 80)
        sys.exit(1)
    else:
        print("Model appears HEALTHY - no collapse detected")
        print("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    main()
