#!/usr/bin/env python3
"""
Compute FIM (Fisher Information Matrix) importance scores per tensor.

Uses diag-FIM proxy = E[g²] (mean squared gradient) over calibration batches.
This measures how sensitive the loss is to each parameter, indicating which
tensors should be kept at higher precision during quantization.

Reference: llama.cpp discussion #12741 for quantization mechanics.
"""

import os

# Force CPU-only execution (for ROCm systems)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_tensor_group(name: str) -> str | None:
    """
    Map parameter name to tensor group for llama.cpp targeting.

    Returns group name like 'attn_q', 'ffn_down', etc., or None if not matched.
    """
    # Common patterns across model architectures
    patterns = {
        # Attention tensors
        r"\.q_proj\.|\.attn\.c_attn\.|attn_q": "attn_q",
        r"\.k_proj\.|attn_k": "attn_k",
        r"\.v_proj\.|attn_v": "attn_v",
        r"\.o_proj\.|\.attn\.c_proj\.|attn_output": "attn_output",
        # FFN tensors (most sensitive per #12741)
        r"\.gate_proj\.|\.mlp\.c_fc\.|ffn_gate": "ffn_gate",
        r"\.up_proj\.|ffn_up": "ffn_up",
        r"\.down_proj\.|\.mlp\.c_proj\.|ffn_down": "ffn_down",
        # Embeddings (least sensitive per #12741)
        r"embed_tokens|wte|token_emb": "token_embedding",
        r"lm_head|output": "output",
    }

    for pattern, group in patterns.items():
        if re.search(pattern, name, re.IGNORECASE):
            return group

    return None


def extract_layer_index(name: str) -> int | None:
    """Extract layer index from parameter name."""
    # Match patterns like: layers.12., h.12., blocks.12., etc.
    match = re.search(r"(?:layers?|h|blocks?)\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    return None


def load_calibration_data(
    tokenizer, data_path: str | None, num_samples: int, seq_length: int
) -> list[torch.Tensor]:
    """Load calibration sequences from file or generate random ones."""
    sequences = []

    if data_path and Path(data_path).exists():
        with open(data_path) as f:
            text = f.read()

        # Tokenize and split into chunks
        tokens = tokenizer.encode(text, add_special_tokens=False)
        for i in range(0, len(tokens) - seq_length, seq_length):
            if len(sequences) >= num_samples:
                break
            sequences.append(torch.tensor(tokens[i : i + seq_length]))
    else:
        # Generate random sequences for quick testing
        print("No calibration file provided, using random sequences")
        vocab_size = tokenizer.vocab_size
        for _ in range(num_samples):
            sequences.append(torch.randint(0, vocab_size, (seq_length,)))

    return sequences[:num_samples]


def compute_fim_scores(
    model: nn.Module,
    tokenizer,
    calibration_data: list[torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    """
    Compute diagonal FIM approximation via accumulated squared gradients.

    For each parameter: FIM_diag ≈ E[(∂L/∂θ)²]
    """
    model.to(device)
    model.eval()

    # Accumulator for squared gradients
    grad_sq_sum = defaultdict(lambda: 0.0)
    num_batches = 0

    for seq in tqdm(calibration_data, desc="Computing FIM scores"):
        # Prepare input
        input_ids = seq.unsqueeze(0).to(device)

        # Forward pass with gradient computation
        model.zero_grad()
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Sum of squared gradients for this batch
                grad_sq = (param.grad**2).sum().item()
                grad_sq_sum[name] += grad_sq

        num_batches += 1

    # Average over batches
    fim_scores = {name: score / num_batches for name, score in grad_sq_sum.items()}

    return fim_scores


def aggregate_scores(
    fim_scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Aggregate FIM scores by tensor group and by layer.group combination.

    Returns:
        by_group: Average FIM per tensor group (e.g., 'ffn_down': 1.23e-6)
        by_layer_group: FIM per layer.group (e.g., '12.ffn_down': 1.23e-6)
    """
    by_group = defaultdict(list)
    by_layer_group = defaultdict(list)

    for name, score in fim_scores.items():
        group = get_tensor_group(name)
        layer = extract_layer_index(name)

        if group:
            by_group[group].append(score)
            if layer is not None:
                key = f"{layer}.{group}"
                by_layer_group[key].append(score)

    # Average scores within each group
    by_group_avg = {g: sum(scores) / len(scores) for g, scores in by_group.items()}
    by_layer_group_avg = {
        k: sum(scores) / len(scores) for k, scores in by_layer_group.items()
    }

    return by_group_avg, by_layer_group_avg


def compute_normalized_scores(scores: dict[str, float]) -> dict[str, float]:
    """Compute z-scores for robust cross-group comparison."""
    if not scores:
        return {}

    values = list(scores.values())
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    if std < 1e-10:
        return {k: 0.0 for k in scores}

    return {k: (v - mean) / std for k, v in scores.items()}


def compute_percentiles(scores: dict[str, float]) -> dict[str, float]:
    """Compute percentile rank for each score."""
    if not scores:
        return {}

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k])
    n = len(sorted_keys)

    return {k: (i + 1) / n * 100 for i, k in enumerate(sorted_keys)}


def main():
    parser = argparse.ArgumentParser(
        description="Compute FIM importance scores for quantization guidance"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        default=None,
        help="Path to calibration text file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of calibration sequences",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=512,
        help="Sequence length for calibration",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fim_scores.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)

    # Verify CPU-only
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()} (should be False)")

    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # Full precision for accurate gradients
    )

    # Load calibration data
    print(f"Loading calibration data ({args.num_samples} samples)")
    calibration_data = load_calibration_data(
        tokenizer, args.calibration_data, args.num_samples, args.seq_length
    )

    # Compute FIM scores
    print("Computing FIM scores...")
    fim_scores = compute_fim_scores(model, tokenizer, calibration_data, device)

    # Aggregate by group and layer
    by_group, by_layer_group = aggregate_scores(fim_scores)

    # Compute normalized scores
    by_group_zscore = compute_normalized_scores(by_group)
    by_layer_group_zscore = compute_normalized_scores(by_layer_group)

    # Compute percentiles
    by_group_percentile = compute_percentiles(by_group)
    by_layer_group_percentile = compute_percentiles(by_layer_group)

    # Prepare output
    output = {
        "model": args.model,
        "num_samples": args.num_samples,
        "seq_length": args.seq_length,
        "seed": args.seed,
        "by_tensor": fim_scores,
        "by_group": by_group,
        "by_group_zscore": by_group_zscore,
        "by_group_percentile": by_group_percentile,
        "by_layer_group": by_layer_group,
        "by_layer_group_zscore": by_layer_group_zscore,
        "by_layer_group_percentile": by_layer_group_percentile,
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n=== FIM Importance by Tensor Group ===")
    print("(Higher = more sensitive to quantization)")
    for group in sorted(by_group.keys(), key=lambda g: by_group[g], reverse=True):
        score = by_group[group]
        zscore = by_group_zscore[group]
        pct = by_group_percentile[group]
        print(f"  {group:20s}: {score:.6e}  (z={zscore:+.2f}, {pct:.0f}th pct)")

    # Identify top layers for FFN (most sensitive per #12741)
    ffn_layers = {
        k: v for k, v in by_layer_group.items() if any(x in k for x in ["ffn_"])
    }
    if ffn_layers:
        print("\n=== Top 5 FFN Layers by FIM ===")
        sorted_ffn = sorted(ffn_layers.items(), key=lambda x: x[1], reverse=True)[:5]
        for name, score in sorted_ffn:
            print(f"  {name:20s}: {score:.6e}")


if __name__ == "__main__":
    main()
