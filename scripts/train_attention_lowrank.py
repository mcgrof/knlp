#!/usr/bin/env python3
"""
Attention-Preserving Low-Rank Projector Trainer.

Trains linear compress/expand matrices that preserve attention behavior
rather than just minimizing reconstruction MSE like PCA.

Loss options:
- logit: ||QK^T - QK_hat^T||^2 (preserve attention logits)
- attn_kl: KL(softmax(QK^T) || softmax(QK_hat^T)) (preserve attention distribution)
- output: ||attn_output - attn_output_hat||^2 (preserve attention output)

Usage:
    python scripts/train_attention_lowrank.py --model Qwen/Qwen2.5-0.5B --layer 0 --rank 56
    python scripts/train_attention_lowrank.py --model Qwen/Qwen2.5-0.5B --layer 0 --rank 48 --loss attn_kl
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer


class AttentionLowRankProjector(nn.Module):
    """
    Learnable low-rank compress/expand matrices for attention.

    compress(x) = x @ W_c  (d_head -> rank)
    expand(z) = z @ W_e    (rank -> d_head)

    Initialized with orthonormal random matrices.
    """

    def __init__(self, d_head: int, rank: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_head = d_head
        self.rank = rank

        # Initialize with orthonormal projection (via QR decomposition)
        W_c_init = torch.randn(d_head, rank, dtype=dtype)
        W_c_init, _ = torch.linalg.qr(W_c_init)
        W_e_init = W_c_init.T.clone()  # Start as transpose for near-identity

        self.W_c = nn.Parameter(W_c_init)
        self.W_e = nn.Parameter(W_e_init)

        # Optional mean centering (can be learned or fixed)
        self.register_buffer("mean", torch.zeros(d_head, dtype=dtype))

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress: [*, d_head] -> [*, rank]"""
        return (x - self.mean) @ self.W_c

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """Expand: [*, rank] -> [*, d_head]"""
        return z @ self.W_e + self.mean

    def round_trip(self, x: torch.Tensor) -> torch.Tensor:
        """Compress then expand."""
        return self.expand(self.compress(x))

    def set_mean(self, mean: torch.Tensor):
        """Set the centering mean from calibration data."""
        self.mean = mean.to(self.mean.device).to(self.mean.dtype)


def compute_logit_loss(
    Q: torch.Tensor,
    K_original: torch.Tensor,
    K_compressed: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Compute attention logit MSE loss.

    Loss = ||QK^T - QK_hat^T||^2

    Args:
        Q: Query tensor [B, H, T, d_head]
        K_original: Original keys [B, H, T, d_head]
        K_compressed: Compressed+expanded keys [B, H, T, d_head]
        scale: Attention scale factor (1/sqrt(d_head))
    """
    # Original attention logits: [B, H, T, T]
    logits_orig = torch.matmul(Q, K_original.transpose(-2, -1)) * scale

    # Compressed attention logits
    logits_comp = torch.matmul(Q, K_compressed.transpose(-2, -1)) * scale

    # MSE loss
    return F.mse_loss(logits_comp, logits_orig)


def compute_attn_kl_loss(
    Q: torch.Tensor,
    K_original: torch.Tensor,
    K_compressed: torch.Tensor,
    scale: float,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute attention distribution KL divergence loss.

    Loss = KL(softmax(QK^T) || softmax(QK_hat^T))

    Uses numerically stable implementation via F.kl_div with log_target=False.

    Args:
        Q: Query tensor [B, H, T, d_head]
        K_original: Original keys [B, H, T, d_head]
        K_compressed: Compressed+expanded keys [B, H, T, d_head]
        scale: Attention scale factor
        mask: Optional causal mask [T, T]
    """
    # Original attention logits
    logits_orig = torch.matmul(Q, K_original.transpose(-2, -1)) * scale

    # Compressed attention logits
    logits_comp = torch.matmul(Q, K_compressed.transpose(-2, -1)) * scale

    # Apply causal mask if provided
    if mask is not None:
        logits_orig = logits_orig.masked_fill(mask == 0, float("-inf"))
        logits_comp = logits_comp.masked_fill(mask == 0, float("-inf"))

    # Compute softmax attention weights (detach target for stability)
    attn_orig = F.softmax(logits_orig, dim=-1).detach()
    log_attn_comp = F.log_softmax(logits_comp, dim=-1)

    # Replace -inf in log_attn_comp with a large negative number to avoid nan
    # Only matters where attn_orig is non-zero
    log_attn_comp = log_attn_comp.clamp(min=-100)

    # Use PyTorch's numerically stable KL divergence
    # kl_div expects log-probabilities as input, probabilities as target
    kl = F.kl_div(log_attn_comp, attn_orig, reduction="batchmean", log_target=False)

    # Clamp result to avoid inf propagation
    kl = kl.clamp(max=100)

    return kl


def compute_output_loss(
    Q: torch.Tensor,
    K_original: torch.Tensor,
    K_compressed: torch.Tensor,
    V: torch.Tensor,
    scale: float,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute attention output MSE loss.

    Loss = ||attn(Q,K,V) - attn(Q,K_hat,V)||^2

    Args:
        Q: Query tensor [B, H, T, d_head]
        K_original: Original keys [B, H, T, d_head]
        K_compressed: Compressed+expanded keys [B, H, T, d_head]
        V: Value tensor [B, H, T, d_head]
        scale: Attention scale factor
        mask: Optional causal mask
    """

    def attn_output(Q, K, V, scale, mask):
        logits = torch.matmul(Q, K.transpose(-2, -1)) * scale
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(logits, dim=-1)
        return torch.matmul(attn, V)

    out_orig = attn_output(Q, K_original, V, scale, mask)
    out_comp = attn_output(Q, K_compressed, V, scale, mask)

    return F.mse_loss(out_comp, out_orig)


def collect_qkv_samples(
    model,
    tokenizer,
    layer_idx: int,
    texts: List[str],
    device: str,
    max_length: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect Q, K, V activations from a specific layer.

    Returns:
        (Q, K, V) each with shape [N, H, T, d_head] where N is number of samples
    """
    Q_samples = []
    K_samples = []
    V_samples = []

    # For Qwen2, we need to capture Q, K, V from within the attention forward
    # This is model-specific; we'll use a simpler approach: recompute from hidden states

    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length",  # Pad to consistent length
            )
            input_ids = inputs.input_ids.to(device)

            # Get hidden states at the target layer
            # We'll use output_hidden_states to get the input to the layer
            outputs = model(
                input_ids,
                output_hidden_states=True,
                output_attentions=False,
            )

            hidden_states = outputs.hidden_states

            # hidden_states[i] is output of layer i-1 (input to layer i)
            # hidden_states[0] is embeddings
            # So for layer_idx, input is hidden_states[layer_idx]
            layer_input = hidden_states[layer_idx]  # [B, T, hidden_size]

            # Access the attention module
            if hasattr(model, "model"):
                # Qwen/Llama style
                attn = model.model.layers[layer_idx].self_attn
            else:
                raise ValueError("Unknown model architecture")

            # Get Q, K, V projections
            B, T, _ = layer_input.shape
            hidden_size = layer_input.shape[-1]
            # Use config for num_heads (Qwen2 style)
            num_heads = attn.config.num_attention_heads
            head_dim = attn.head_dim

            # Compute Q, K, V using model's projection weights
            q_proj = attn.q_proj(layer_input)
            k_proj = attn.k_proj(layer_input)
            v_proj = attn.v_proj(layer_input)

            # Reshape to [B, H, T, d_head]
            # Handle GQA (grouped query attention) if present
            num_kv_heads = attn.config.num_key_value_heads

            Q = q_proj.view(B, T, num_heads, head_dim).transpose(1, 2)
            K = k_proj.view(B, T, num_kv_heads, head_dim).transpose(1, 2)
            V = v_proj.view(B, T, num_kv_heads, head_dim).transpose(1, 2)

            # If GQA, expand K, V to match Q heads
            if num_kv_heads != num_heads:
                repeat_factor = num_heads // num_kv_heads
                K = K.repeat_interleave(repeat_factor, dim=1)
                V = V.repeat_interleave(repeat_factor, dim=1)

            Q_samples.append(Q.cpu())
            K_samples.append(K.cpu())
            V_samples.append(V.cpu())

    # Concatenate along batch dimension
    Q_all = torch.cat(Q_samples, dim=0)  # [N, H, T, d_head]
    K_all = torch.cat(K_samples, dim=0)
    V_all = torch.cat(V_samples, dim=0)

    return Q_all, K_all, V_all


def train_attention_projector(
    model_name: str,
    layer_idx: int,
    rank: int,
    loss_type: str = "logit",
    target: str = "k",  # "k", "v", or "kv"
    lr: float = 1e-3,
    epochs: int = 100,
    device: str = "cuda",
) -> Dict:
    """
    Train attention-preserving low-rank projector for a single layer.

    Args:
        model_name: HuggingFace model name
        layer_idx: Which layer to train projector for
        rank: Target rank for compression
        loss_type: "logit", "attn_kl", or "output"
        target: Which to compress - "k", "v", or "kv"
        lr: Learning rate
        epochs: Training epochs
        device: Device to use

    Returns:
        Dict with trained projector parameters and training info
    """
    print(f"Training attention-preserving projector")
    print(f"  Model: {model_name}")
    print(f"  Layer: {layer_idx}")
    print(f"  Rank: {rank}")
    print(f"  Loss: {loss_type}")
    print(f"  Target: {target}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    # Get head_dim
    if hasattr(model, "model"):
        attn = model.model.layers[layer_idx].self_attn
    else:
        raise ValueError("Unknown model architecture")

    head_dim = attn.head_dim
    scale = head_dim**-0.5
    print(f"  Head dim: {head_dim}, Scale: {scale:.4f}")

    # Calibration texts
    calib_texts = [
        "The quick brown fox jumps over the lazy dog. " * 5,
        "Machine learning has transformed how we approach complex problems. " * 5,
        "Transformer architectures have revolutionized natural language processing. "
        * 5,
        "Large language models can generate coherent text and answer questions. " * 5,
        "Attention mechanisms allow models to focus on relevant parts of input. " * 5,
        "Key-value caching enables efficient autoregressive generation. " * 5,
        "Compression techniques are essential for deploying large models. " * 5,
        "Neural networks learn patterns from data without explicit programming. " * 5,
    ]

    # Collect Q, K, V samples
    print("\nCollecting Q, K, V samples...")
    Q_all, K_all, V_all = collect_qkv_samples(
        model, tokenizer, layer_idx, calib_texts, device, max_length=128
    )
    print(f"  Collected: Q={Q_all.shape}, K={K_all.shape}, V={V_all.shape}")

    # Move to device for training (use float32 for training stability)
    Q_all = Q_all.to(device).float()
    K_all = K_all.to(device).float()
    V_all = V_all.to(device).float()

    # Compute mean for centering
    K_flat = K_all.reshape(-1, head_dim)
    V_flat = V_all.reshape(-1, head_dim)
    K_mean = K_flat.mean(dim=0)
    V_mean = V_flat.mean(dim=0)

    # Create projectors
    k_projector = AttentionLowRankProjector(head_dim, rank, dtype=torch.float32).to(
        device
    )
    v_projector = AttentionLowRankProjector(head_dim, rank, dtype=torch.float32).to(
        device
    )
    k_projector.set_mean(K_mean)
    v_projector.set_mean(V_mean)

    # Setup optimizer
    params = []
    if target in ("k", "kv"):
        params.extend(k_projector.parameters())
    if target in ("v", "kv"):
        params.extend(v_projector.parameters())

    optimizer = Adam(params, lr=lr)

    # Create causal mask
    T = Q_all.shape[2]
    causal_mask = torch.tril(torch.ones(T, T, device=device))

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Apply compression
        if target in ("k", "kv"):
            K_comp = k_projector.round_trip(K_all)
        else:
            K_comp = K_all

        if target in ("v", "kv"):
            V_comp = v_projector.round_trip(V_all)
        else:
            V_comp = V_all

        # Compute loss based on type
        if loss_type == "logit":
            loss = compute_logit_loss(Q_all, K_all, K_comp, scale)
        elif loss_type == "attn_kl":
            loss = compute_attn_kl_loss(Q_all, K_all, K_comp, scale, causal_mask)
        elif loss_type == "output":
            loss = compute_output_loss(Q_all, K_all, K_comp, V_comp, scale, causal_mask)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: Loss = {loss.item():.6f}")

    print(f"\nFinal loss: {losses[-1]:.6f}")

    # Convert projectors to fp16 and extract parameters
    result = {
        "layer": layer_idx,
        "rank": rank,
        "head_dim": head_dim,
        "loss_type": loss_type,
        "target": target,
        "final_loss": losses[-1],
        "losses": losses,
        "K": {
            "U": k_projector.W_c.detach().cpu().half(),  # [d_head, rank]
            "W_e": k_projector.W_e.detach().cpu().half(),  # [rank, d_head]
            "mean": k_projector.mean.cpu().half(),
        },
        "V": {
            "U": v_projector.W_c.detach().cpu().half(),
            "W_e": v_projector.W_e.detach().cpu().half(),
            "mean": v_projector.mean.cpu().half(),
        },
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train attention-preserving low-rank projector"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model to use",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Layer to train projector for",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=56,
        help="Target rank for compression",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="logit",
        choices=["logit", "attn_kl", "output"],
        help="Loss function type",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="k",
        choices=["k", "v", "kv"],
        help="Which to compress",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs",
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
        help="Output file (default: auto-generated)",
    )
    args = parser.parse_args()

    result = train_attention_projector(
        model_name=args.model,
        layer_idx=args.layer,
        rank=args.rank,
        loss_type=args.loss,
        target=args.target,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
    )

    # Save result
    if args.output is None:
        model_short = args.model.replace("/", "-").lower()
        output_path = f"attn_lowrank_{model_short}_l{args.layer}_r{args.rank}_{args.loss}_{args.target}.pt"
    else:
        output_path = args.output

    torch.save(result, output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
