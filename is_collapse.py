#!/usr/bin/env python3
import argparse
import math
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F


PROMPTS = [
    "The meaning of life is",
    "In a surprising turn of events,",
    "Once upon a time in a distant galaxy,",
    "The Linux kernel developer woke up and",
    "In deep learning, one of the key challenges is",
]


@torch.no_grad()
def sample_and_measure(
    model, tokenizer, prompt, device, max_new_tokens=64, temperature=0.8, top_k=0
):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k if top_k > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
    )[0]

    # Only look at newly generated tokens
    gen_ids = output_ids[input_ids.shape[-1] :]

    # Compute per-step entropy for new tokens
    all_entropies = []
    curr_ids = input_ids.clone()
    for tok in gen_ids:
        logits = model(curr_ids).logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        # H = -sum p log p
        entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)
        all_entropies.append(entropy.item())
        curr_ids = torch.cat([curr_ids, tok.view(1, 1)], dim=-1)

    # Diversity inside this one sample (distinct-1 / distinct-2)
    tokens = gen_ids.tolist()
    unigrams = Counter(tokens)
    bigrams = Counter(zip(tokens, tokens[1:])) if len(tokens) > 1 else Counter()

    distinct1 = len(unigrams) / max(len(tokens), 1)
    distinct2 = len(bigrams) / max(len(tokens) - 1, 1) if len(tokens) > 1 else 0.0

    return {
        "prompt": prompt,
        "text": tokenizer.decode(gen_ids, skip_special_tokens=True),
        "avg_entropy": sum(all_entropies) / max(len(all_entropies), 1),
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

    # Check if this is a .pt checkpoint or HF model
    if args.model.endswith(".pt"):
        # Load from custom checkpoint
        print(f"Loading checkpoint from {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.eos_token is None:
            tokenizer.eos_token = tokenizer.pad_token or ""

        # Load base model and override with checkpoint weights
        model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)

        # Extract state dict (handle different checkpoint formats)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
    else:
        # Load from HuggingFace
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.eos_token is None:
            tokenizer.eos_token = tokenizer.pad_token or ""

        model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    print(f"Using device: {device}")
    print(f"Testing model: {args.model}")
    print("-" * 80)

    all_outputs = []

    for p in PROMPTS:
        print(f"\n=== PROMPT: {p!r} ===")
        for i in range(args.samples_per_prompt):
            res = sample_and_measure(
                model,
                tokenizer,
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


if __name__ == "__main__":
    main()
