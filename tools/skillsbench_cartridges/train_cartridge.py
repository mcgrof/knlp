#!/usr/bin/env python3
"""Train a cartridge (KV cache state) from skill text content.

Usage:
    python train_cartridge.py \
        --model /workspace/models/Qwen2.5-7B-Instruct \
        --skill-text /path/to/SKILL.md \
        --output-dir /workspace/cartridges/citation-check/default-50pct \
        --budget-pct 50

Produces:
    cartridge.pt        — KV cache state (first-k layers, truncated to budget)
    prefix_token_ids.json — token IDs of the skill text
    meta.json           — training metadata
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def train_cartridge(
    model_path: str,
    skill_text_path: str,
    output_dir: str,
    budget_pct: int = 50,
    init_method: str = "default",  # "default" (first-k) or "sci"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer and model
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Tokenize skill text
    skill_text = Path(skill_text_path).read_text()
    inputs = tokenizer(skill_text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"].to(model.device)
    n_tokens = input_ids.shape[1]
    print(f"Skill text: {len(skill_text)} chars, {n_tokens} tokens")

    # Compute budget
    budget_tokens = int(n_tokens * budget_pct / 100)
    print(f"Budget: {budget_pct}% = {budget_tokens} tokens (of {n_tokens})")

    # Generate KV cache via forward pass
    t0 = time.time()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    kv_cache = outputs.past_key_values
    fwd_time = time.time() - t0
    print(f"Forward pass: {fwd_time:.2f}s")

    # Extract and truncate KV cache to budget
    if init_method == "default":
        # First-k: take the first budget_tokens positions from each layer
        cartridge_kv = []
        for layer_kv in kv_cache:
            k, v = layer_kv[0], layer_kv[1]
            # k, v shape: (batch, n_heads, seq_len, head_dim)
            k_trunc = k[:, :, :budget_tokens, :].cpu()
            v_trunc = v[:, :, :budget_tokens, :].cpu()
            cartridge_kv.append((k_trunc, v_trunc))
    elif init_method == "sci":
        # Sliding-window Causal Init: take uniformly spaced positions
        # spanning the full sequence to capture global structure
        indices = torch.linspace(0, n_tokens - 1, budget_tokens).long()
        cartridge_kv = []
        for layer_kv in kv_cache:
            k, v = layer_kv[0], layer_kv[1]
            k_sel = k[:, :, indices, :].cpu()
            v_sel = v[:, :, indices, :].cpu()
            cartridge_kv.append((k_sel, v_sel))
    else:
        raise ValueError(f"Unknown init method: {init_method}")

    # Save cartridge
    cartridge_path = output_dir / "cartridge.pt"
    torch.save(cartridge_kv, cartridge_path)
    cart_size = cartridge_path.stat().st_size
    print(f"Cartridge saved: {cartridge_path} ({cart_size / 1024 / 1024:.1f} MB)")

    # Save prefix token IDs
    prefix_ids = input_ids[0, :budget_tokens].cpu().tolist()
    prefix_path = output_dir / "prefix_token_ids.json"
    with open(prefix_path, "w") as f:
        json.dump(prefix_ids, f)
    print(f"Prefix token IDs saved: {len(prefix_ids)} tokens")

    # Save metadata
    meta = {
        "model": model_path,
        "skill_text": skill_text_path,
        "init_method": init_method,
        "budget_pct": budget_pct,
        "original_tokens": n_tokens,
        "budget_tokens": budget_tokens,
        "cartridge_size_bytes": cart_size,
        "n_layers": len(cartridge_kv),
        "fwd_time_sec": round(fwd_time, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {json.dumps(meta, indent=2)}")

    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--skill-text", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-pct", type=int, default=50)
    parser.add_argument("--init-method", choices=["default", "sci"], default="default")
    args = parser.parse_args()

    train_cartridge(
        model_path=args.model,
        skill_text_path=args.skill_text,
        output_dir=args.output_dir,
        budget_pct=args.budget_pct,
        init_method=args.init_method,
    )
