"""
PCA-based Calibration for Latent KV Compression.

Calibrates the projection matrices in CompressionBackend using PCA on actual
Q,K,V statistics from the model. This enables acceptable PPL with compression.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


def collect_qkv_statistics(
    model: nn.Module,
    tokenizer,
    num_samples: int = 100,
    max_length: int = 256,
    device: str = "cuda",
) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
    """
    Collect Q, K, V statistics from a model for PCA calibration.

    Args:
        model: HuggingFace model (unmodified, no compression)
        tokenizer: Tokenizer for the model
        num_samples: Number of calibration samples
        max_length: Maximum sequence length
        device: Device to run on

    Returns:
        Dictionary mapping (layer_idx, head_idx) to {"Q": [...], "K": [...], "V": [...]}
        where each list contains tensors of shape [B*T, d_head]
    """
    # Load calibration data
    print(f"Loading {num_samples} calibration samples...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    samples = [s["text"] for s in dataset.take(num_samples)]

    # Extract model config
    config = model.config
    n_layer = config.n_layer
    n_head = config.n_head
    head_dim = config.n_embd // config.n_head

    # Storage for Q, K, V statistics
    statistics = {}
    for layer_idx in range(n_layer):
        for head_idx in range(n_head):
            statistics[(layer_idx, head_idx)] = {
                "Q": [],
                "K": [],
                "V": [],
            }

    # Hook to capture Q, K, V
    def create_capture_hook(layer_idx: int):
        def hook(module, input, output):
            # GPT2Attention output: (attn_output, (key, value)) or (attn_output, present, attn_weights)
            # We need to capture Q, K, V BEFORE they go into SDPA
            # This is tricky - we need to hook into the internal computation

            # Get hidden states (input to attention)
            hidden_states = input[0]  # [B, T, C]
            B, T, C = hidden_states.shape

            # Compute Q, K, V using the attention's projection
            qkv = module.c_attn(hidden_states)  # [B, T, 3*C]
            q, k, v = qkv.split(module.split_size, dim=2)  # Each [B, T, C]

            # Reshape to heads: [B, T, C] -> [B, T, H, D] -> [B, H, T, D]
            H = n_head
            D = head_dim

            q = q.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
            k = k.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
            v = v.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]

            # Store per-head statistics
            for head_idx in range(H):
                q_h = q[:, head_idx, :, :].detach().cpu().float()  # [B, T, D]
                k_h = k[:, head_idx, :, :].detach().cpu().float()
                v_h = v[:, head_idx, :, :].detach().cpu().float()

                # Flatten batch and sequence dims
                q_flat = q_h.reshape(-1, D)  # [B*T, D]
                k_flat = k_h.reshape(-1, D)
                v_flat = v_h.reshape(-1, D)

                statistics[(layer_idx, head_idx)]["Q"].append(q_flat)
                statistics[(layer_idx, head_idx)]["K"].append(k_flat)
                statistics[(layer_idx, head_idx)]["V"].append(v_flat)

        return hook

    # Register hooks
    handles = []
    for layer_idx, layer in enumerate(model.transformer.h):
        handle = layer.attn.register_forward_hook(create_capture_hook(layer_idx))
        handles.append(handle)

    # Run model on calibration data
    model.eval()
    print("Collecting Q, K, V statistics...")
    with torch.no_grad():
        for sample in tqdm(samples, desc="Calibration"):
            inputs = tokenizer(
                sample, return_tensors="pt", truncation=True, max_length=max_length
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Concatenate statistics
    print("Concatenating statistics...")
    for key in tqdm(statistics.keys(), desc="Concatenating"):
        statistics[key]["Q"] = torch.cat(statistics[key]["Q"], dim=0)  # [N, D]
        statistics[key]["K"] = torch.cat(statistics[key]["K"], dim=0)
        statistics[key]["V"] = torch.cat(statistics[key]["V"], dim=0)

    return statistics


def compute_pca_projections(
    X: torch.Tensor, rank: int, centered: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PCA projection matrices for dimensionality reduction.

    Args:
        X: Data matrix [N, d_full]
        rank: Target latent dimension
        centered: If True, use standard centered PCA. If False, use non-centered
                  PCA which better preserves dot products for attention.

    Returns:
        (W_proj, W_recon, mean):
            W_proj: Projection matrix [d_full, rank] (maps full -> latent)
            W_recon: Reconstruction matrix [rank, d_full] (maps latent -> full)
            mean: Mean vector [d_full] (zero if not centered)
    """
    # Convert to float32 for eigendecomposition (not supported on float16)
    X = X.float()

    if centered:
        # Standard PCA: center the data
        mean = X.mean(dim=0)  # [d_full]
        X_proc = X - mean  # [N, d_full]
        # Covariance matrix
        cov = (X_proc.T @ X_proc) / (X.shape[0] - 1)  # [d_full, d_full]
    else:
        # Non-centered PCA: eigendecomposition of X^T X
        # This preserves dot products better for attention
        mean = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
        # Gram matrix (not covariance)
        cov = (X.T @ X) / X.shape[0]  # [d_full, d_full]

    # Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(cov)  # eigvecs: [d_full, d_full]

    # Sort by descending eigenvalue
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Take top-k components
    W_proj = eigvecs[:, :rank]  # [d_full, rank]

    # For reconstruction, use transpose (orthogonal basis)
    W_recon = W_proj.T  # [rank, d_full]

    return W_proj, W_recon, mean


def calibrate_backend_with_pca(
    statistics: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
    backend,
    device: str = "cuda",
) -> None:
    """
    Calibrate CompressionBackend projection matrices using PCA.

    Updates backend.projections in-place with PCA-calibrated weights.

    Args:
        statistics: Q, K, V statistics from collect_qkv_statistics()
        backend: CompressionBackend to calibrate
        device: Device to place calibrated weights on
    """
    print("Calibrating backend with PCA...")

    for (layer_idx, head_idx), stats in tqdm(
        statistics.items(), desc="Calibrating heads"
    ):
        # Get target rank for this head
        rank = backend.get_rank(layer_idx, head_idx)

        if rank >= backend.head_dim:
            # No compression needed
            continue

        # Check if projection exists
        key = f"L{layer_idx}_H{head_idx}"
        if key not in backend.projections:
            continue

        projection = backend.projections[key]

        # Calibrate Q projection
        Q = stats["Q"]  # [N, d_head]
        W_q_proj, W_q_recon, mean_q = compute_pca_projections(Q, rank, centered=False)

        # Calibrate K projection
        K = stats["K"]
        W_k_proj, W_k_recon, mean_k = compute_pca_projections(K, rank, centered=False)

        # Calibrate V projection
        V = stats["V"]
        W_v_proj, W_v_recon, mean_v = compute_pca_projections(V, rank, centered=False)

        # Update backend weights
        # W_q, W_k, W_v are Linear(d_head, rank), so weight is [rank, d_head]
        # W_q_proj is [d_head, rank], so we transpose to get [rank, d_head]
        projection.W_q.weight.data = W_q_proj.T.to(
            device=device, dtype=projection.W_q.weight.dtype
        )
        projection.W_k.weight.data = W_k_proj.T.to(
            device=device, dtype=projection.W_k.weight.dtype
        )
        projection.W_v.weight.data = W_v_proj.T.to(
            device=device, dtype=projection.W_v.weight.dtype
        )

        # W_out is Linear(rank, d_head), so weight is [d_head, rank]
        # W_v_recon is [rank, d_head], so we transpose to get [d_head, rank]
        projection.W_out.weight.data = W_v_recon.T.to(
            device=device, dtype=projection.W_out.weight.dtype
        )

        # For non-centered PCA, mean vectors are zero, so don't register them
        # This ensures the forward pass doesn't apply any centering
        # (The forward pass checks hasattr() before applying centering)

    print("Calibration complete!")


def calibrate_model_for_compression(
    model,
    tokenizer,
    backend,
    num_samples: int = 100,
    max_length: int = 256,
    device: str = "cuda",
) -> None:
    """
    End-to-end calibration pipeline.

    Args:
        model: HuggingFace model (unmodified)
        tokenizer: Tokenizer
        backend: CompressionBackend to calibrate
        num_samples: Number of calibration samples
        max_length: Maximum sequence length
        device: Device
    """
    # Collect statistics
    statistics = collect_qkv_statistics(
        model, tokenizer, num_samples, max_length, device
    )

    # Calibrate backend
    calibrate_backend_with_pca(statistics, backend, device)

    print("Model calibrated for compression!")
