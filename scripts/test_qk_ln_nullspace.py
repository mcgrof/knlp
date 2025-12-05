#!/usr/bin/env python3
"""
Test Q/K LayerNorm nullspace compression for FLOP reduction.

This tests applying LN nullspace to both K (in cache) and Q (in attention),
which reduces the QK^T matmul from d to d-1 dimensions (~2% FLOP savings).

The key insight:
- Q' = Q @ U, K' = K @ U (both projected to LN nullspace)
- Q'K'^T ≈ QK^T because LN'd vectors already live in this subspace
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    QKLNNullspaceCompressor,
    make_ln_nullspace_basis,
)


def test_qk_matmul_equivalence(
    head_dim: int = 128, seq_len: int = 64, device: str = "cuda"
):
    """Test that Q'K'^T ≈ QK^T when Q,K are LN'd vectors."""
    print(f"\n=== Testing QK Matmul Equivalence (d={head_dim}, T={seq_len}) ===")

    # Create LN nullspace basis
    U = make_ln_nullspace_basis(
        head_dim, device=torch.device(device), dtype=torch.float32
    )

    # Generate test Q, K (simulating LN'd vectors with mean ~0)
    Q = torch.randn(1, 8, seq_len, head_dim, device=device, dtype=torch.float32)
    K = torch.randn(1, 8, seq_len, head_dim, device=device, dtype=torch.float32)

    # Force mean 0 (like LN output)
    Q = Q - Q.mean(dim=-1, keepdim=True)
    K = K - K.mean(dim=-1, keepdim=True)

    # Original attention scores
    scores_orig = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

    # Project to LN nullspace
    Q_proj = torch.matmul(Q, U)  # [B, H, T, d-1]
    K_proj = torch.matmul(K, U)  # [B, H, T, d-1]

    # Attention scores in reduced space
    scores_proj = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / math.sqrt(
        head_dim - 1
    )

    # Compare
    mse = ((scores_orig - scores_proj) ** 2).mean().item()
    max_diff = (scores_orig - scores_proj).abs().max().item()
    corr = torch.corrcoef(torch.stack([scores_orig.flatten(), scores_proj.flatten()]))[
        0, 1
    ].item()

    print(f"Original scores shape: {scores_orig.shape}")
    print(f"Projected scores shape: {scores_proj.shape}")
    print(f"MSE: {mse:.6f}")
    print(f"Max diff: {max_diff:.6f}")
    print(f"Correlation: {corr:.6f}")

    # Check softmax similarity
    attn_orig = F.softmax(scores_orig, dim=-1)
    attn_proj = F.softmax(scores_proj, dim=-1)
    attn_mse = ((attn_orig - attn_proj) ** 2).mean().item()
    print(f"Softmax attention MSE: {attn_mse:.6f}")

    return mse < 0.01 and corr > 0.99


def eval_ppl_with_k_nullspace(
    model_name: str,
    device: str = "cuda",
    max_samples: int = 50,
):
    """Evaluate PPL with K-only LN nullspace (simplest case)."""
    print(f"\n=== PPL Test with K LN Nullspace ===")
    print(f"Model: {model_name}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device
    )
    model.eval()

    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    head_dim = config.hidden_size // config.num_attention_heads

    print(f"Layers: {num_layers}, Head dim: {head_dim}")

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t) > 100][:max_samples]

    def eval_ppl(k_compressors, v_compressors):
        cache = CompressedDynamicCache(k_compressors, v_compressors, num_layers)

        total_loss = 0.0
        total_tokens = 0

        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            input_ids = inputs.input_ids.to(device)

            if input_ids.shape[1] < 2:
                continue

            cache.reset()

            with torch.no_grad():
                outputs = model(input_ids, past_key_values=cache, use_cache=True)

                logits = outputs.logits[:, :-1, :]
                targets = input_ids[:, 1:]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                )

                total_loss += loss.item()
                total_tokens += targets.numel()

        return torch.exp(torch.tensor(total_loss / total_tokens)).item()

    # Baseline (identity)
    print("\nEvaluating baseline...")
    k_id = [IdentityCompressor() for _ in range(num_layers)]
    v_id = [IdentityCompressor() for _ in range(num_layers)]
    ppl_baseline = eval_ppl(k_id, v_id)

    # K-only LN nullspace (V identity)
    # Note: This changes K dim from d to d-1, but V stays at d
    # The model should still work because attention output = attn @ V
    # and V is still in original dim
    print("\nEvaluating K LN nullspace only...")
    k_ln = [
        QKLNNullspaceCompressor(head_dim, device=torch.device(device))
        for _ in range(num_layers)
    ]
    v_id2 = [IdentityCompressor() for _ in range(num_layers)]
    ppl_k_ln = eval_ppl(k_ln, v_id2)

    # V-only LN nullspace (like v15)
    print("\nEvaluating V LN nullspace only...")
    k_id3 = [IdentityCompressor() for _ in range(num_layers)]
    from gpt2.compression.compressed_cache import LayerNormNullspaceCompressor

    v_ln = [
        LayerNormNullspaceCompressor(head_dim, device=torch.device(device))
        for _ in range(num_layers)
    ]
    ppl_v_ln = eval_ppl(k_id3, v_ln)

    # K+V LN nullspace combined
    print("\nEvaluating K+V LN nullspace combined...")
    k_ln2 = [
        QKLNNullspaceCompressor(head_dim, device=torch.device(device))
        for _ in range(num_layers)
    ]
    v_ln2 = [
        LayerNormNullspaceCompressor(head_dim, device=torch.device(device))
        for _ in range(num_layers)
    ]
    ppl_kv_ln = eval_ppl(k_ln2, v_ln2)

    print(f"\n=== Results ===")
    print(f"Baseline PPL: {ppl_baseline:.4f}")
    print(
        f"K LN nullspace PPL: {ppl_k_ln:.4f} ({(ppl_k_ln/ppl_baseline - 1)*100:+.2f}%)"
    )
    print(
        f"V LN nullspace PPL: {ppl_v_ln:.4f} ({(ppl_v_ln/ppl_baseline - 1)*100:+.2f}%)"
    )
    print(
        f"K+V LN nullspace PPL: {ppl_kv_ln:.4f} ({(ppl_kv_ln/ppl_baseline - 1)*100:+.2f}%)"
    )

    # FLOP analysis
    print(f"\n=== FLOP Analysis ===")
    print(f"QK^T matmul: d x d = {head_dim} x {head_dim}")
    print(f"With K nullspace: (d-1) x d = {head_dim-1} x {head_dim}")
    print(f"FLOP reduction: {(1 - (head_dim-1)/head_dim)*100:.2f}%")

    # Memory analysis
    print(f"\n=== Memory Analysis ===")
    print(
        f"K-only compression: {head_dim}/{head_dim-1:.0f} = {head_dim/(head_dim-1):.4f}x"
    )
    print(
        f"V-only compression: {head_dim}/{head_dim-1:.0f} = {head_dim/(head_dim-1):.4f}x"
    )
    print(
        f"K+V compression: 2*{head_dim}/2*{head_dim-1:.0f} = {head_dim/(head_dim-1):.4f}x"
    )

    return ppl_baseline, ppl_k_ln, ppl_v_ln, ppl_kv_ln


def main():
    parser = argparse.ArgumentParser(description="Test Q/K LN nullspace")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B", help="Model to test"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--max-samples", type=int, default=30, help="Max samples for PPL eval"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Q/K LAYERNORM NULLSPACE TEST")
    print("=" * 70)

    # Test 1: QK matmul equivalence
    print("\n[Test 1] QK matmul equivalence...")
    if test_qk_matmul_equivalence(device=args.device):
        print("  PASSED - Q'K'^T ≈ QK^T")
    else:
        print("  WARNING - Higher than expected error")

    # Test 2: PPL evaluation
    print("\n[Test 2] PPL evaluation...")
    eval_ppl_with_k_nullspace(args.model, args.device, args.max_samples)


if __name__ == "__main__":
    main()
