#!/usr/bin/env python3
"""Generate text samples from a trained model checkpoint."""

import argparse
import torch
import tiktoken
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.model import GPT2, GPTConfig
from gpt2.trainers.ra import RAGPT, RAConfig


def load_model(checkpoint_path, device="cuda"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Try to determine model type from checkpoint
    state_dict = checkpoint.get("model", checkpoint)

    # Check if it's an RA model by looking for router weights
    is_ra = any("router" in k for k in state_dict.keys())

    # Create config
    gpt_config = GPTConfig.from_name("gpt2")
    gpt_config.block_size = 1024

    if is_ra:
        ra_config = RAConfig(
            d_model=gpt_config.n_embd,
            n_heads=gpt_config.n_head,
            block_size=gpt_config.block_size,
        )
        model = RAGPT(gpt_config, ra_config)
        print("Loaded RAGPT model")
    else:
        model = GPT2(gpt_config)
        print("Loaded GPT model")

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model


def generate(model, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40):
    """Generate text from prompt."""
    device = next(model.parameters()).device
    idx = prompt_ids.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = (
                idx
                if idx.size(1) <= model.config.block_size
                else idx[:, -model.config.block_size :]
            )

            # Forward pass
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--prompt", default="The quick brown fox", help="Text prompt")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples")
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Encode prompt
    prompt_ids = torch.tensor([enc.encode(args.prompt)], dtype=torch.long)

    print(f"\nPrompt: {args.prompt}")
    print("=" * 60)

    for i in range(args.num_samples):
        output_ids = generate(model, prompt_ids, args.max_tokens, args.temperature)
        output_text = enc.decode(output_ids[0].tolist())
        print(f"\n--- Sample {i+1} ---")
        print(output_text)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
