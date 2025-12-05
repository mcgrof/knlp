"""
KVSplice Compressor: Learned Linear Latent Compression

Learns low-rank linear projections for K and V compression during calibration.
Can be applied post-hoc to any trained model without retraining.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2.compression.base import KVCompressorBase


class KVSpliceCompressor(KVCompressorBase):
    """
    KVSplice-style learned linear compressor.

    For each (layer, head) with compression enabled:
    - Learns projection matrices W_k_in, W_v_in during calibration
    - Optionally learns scale/shift parameters for latent space
    - Compression: Z_k = K @ W_k_in, Z_v = V @ W_v_in
    - Decompression: K_hat = Z_k @ W_k_out, V_hat = Z_v @ W_v_out

    Calibration minimizes reconstruction loss:
        L = ||K - K_hat||^2 + ||V - V_hat||^2
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Per-layer/head compression parameters
        self.compressors = nn.ModuleDict()

        # Initialize compressor modules for each enabled layer/head
        for (layer_idx, head_idx), cfg in self.layer_head_configs.items():
            if not cfg.get("enabled", False):
                continue

            rank = cfg.get("rank", 32)
            d_k = cfg.get("d_k", self.d_head)
            d_v = cfg.get("d_v", self.d_head)

            # Create module for this head
            key = f"L{layer_idx}_H{head_idx}"
            self.compressors[key] = KVSpliceHeadCompressor(
                d_k=d_k,
                d_v=d_v,
                rank=rank,
                learn_scale_shift=cfg.get("learn_scale_shift", True),
            )

    def _get_head_compressor(self, layer_idx: int, head_idx: int):
        """Get compressor module for specific layer/head."""
        key = f"L{layer_idx}_H{head_idx}"
        return self.compressors.get(key, None)

    def start_calibration(self):
        """Initialize calibration buffers."""
        self.calibration_mode = True
        self.calibration_data = {}

        for layer_idx, head_idx in self.layer_head_configs.keys():
            if self.is_enabled(layer_idx, head_idx):
                self.calibration_data[(layer_idx, head_idx)] = []

    def observe_kv(
        self,
        layer_idx: int,
        head_idx: int,
        K: torch.Tensor,
        V: torch.Tensor,
    ):
        """
        Collect KV samples for calibration.

        Downsamples to keep memory bounded (max 10k samples per head).
        """
        if not self.calibration_mode:
            return

        if not self.is_enabled(layer_idx, head_idx):
            return

        key = (layer_idx, head_idx)
        if key not in self.calibration_data:
            self.calibration_data[key] = []

        # Flatten batch and sequence dimensions
        # K, V: [batch, seq, d_k/d_v] -> [batch * seq, d_k/d_v]
        K_flat = K.reshape(-1, K.size(-1))
        V_flat = V.reshape(-1, V.size(-1))

        # Downsample if too many samples
        max_samples = 10000
        if K_flat.size(0) > max_samples:
            idx = torch.randperm(K_flat.size(0))[:max_samples]
            K_flat = K_flat[idx]
            V_flat = V_flat[idx]

        # Store on CPU to save GPU memory
        self.calibration_data[key].append((K_flat.cpu(), V_flat.cpu()))

    def end_calibration(self):
        """
        Fit projection matrices using calibration data.

        Minimizes reconstruction loss via mini-batch gradient descent.
        """
        self.calibration_mode = False

        device = next(self.parameters()).device

        for (layer_idx, head_idx), samples in self.calibration_data.items():
            if not samples:
                continue

            # Concatenate all samples
            K_all = torch.cat([K for K, V in samples], dim=0).to(device)
            V_all = torch.cat([V for K, V in samples], dim=0).to(device)

            # Get compressor for this head
            compressor = self._get_head_compressor(layer_idx, head_idx)
            if compressor is None:
                continue

            # Fit parameters
            self._fit_head_compressor(compressor, K_all, V_all)

        # Clear calibration data to free memory
        self.calibration_data.clear()

    def _fit_head_compressor(
        self,
        compressor: "KVSpliceHeadCompressor",
        K: torch.Tensor,
        V: torch.Tensor,
        num_epochs: int = 10,
        batch_size: int = 1024,
        lr: float = 1e-3,
    ):
        """
        Fit compressor parameters via gradient descent.

        Args:
            compressor: KVSpliceHeadCompressor module
            K, V: Calibration samples [N, d_k/d_v]
            num_epochs: Training epochs
            batch_size: Mini-batch size
            lr: Learning rate
        """
        optimizer = torch.optim.Adam(compressor.parameters(), lr=lr)

        N = K.size(0)
        for epoch in range(num_epochs):
            # Shuffle data
            perm = torch.randperm(N)
            K_shuffle = K[perm]
            V_shuffle = V[perm]

            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, N, batch_size):
                K_batch = K_shuffle[i : i + batch_size]
                V_batch = V_shuffle[i : i + batch_size]

                # Forward: compress + decompress
                Z_k, Z_v = compressor.compress(K_batch, V_batch)
                K_hat, V_hat = compressor.decompress(Z_k, Z_v)

                # Reconstruction loss
                loss = F.mse_loss(K_hat, K_batch) + F.mse_loss(V_hat, V_batch)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}/{num_epochs}: loss = {avg_loss:.6f}")

    def compress(
        self,
        layer_idx: int,
        head_idx: int,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress KV pairs at inference.

        Args:
            K, V: [batch, seq, d_k/d_v]

        Returns:
            Z_k, Z_v: [batch, seq, rank]
        """
        if not self.is_enabled(layer_idx, head_idx):
            # No compression
            return K, V

        compressor = self._get_head_compressor(layer_idx, head_idx)
        if compressor is None:
            return K, V

        # Preserve shape for reshape
        batch, seq, d = K.shape

        # Flatten for compression
        K_flat = K.reshape(batch * seq, d)
        V_flat = V.reshape(batch * seq, V.size(-1))

        # Compress
        Z_k_flat, Z_v_flat = compressor.compress(K_flat, V_flat)

        # Reshape back
        rank_k = Z_k_flat.size(-1)
        rank_v = Z_v_flat.size(-1)
        Z_k = Z_k_flat.reshape(batch, seq, rank_k)
        Z_v = Z_v_flat.reshape(batch, seq, rank_v)

        return Z_k, Z_v

    def decompress(
        self,
        layer_idx: int,
        head_idx: int,
        Z_k: torch.Tensor,
        Z_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress KV pairs at inference.

        Args:
            Z_k, Z_v: [batch, seq, rank]

        Returns:
            K_hat, V_hat: [batch, seq, d_k/d_v]
        """
        if not self.is_enabled(layer_idx, head_idx):
            # No compression was applied
            return Z_k, Z_v

        compressor = self._get_head_compressor(layer_idx, head_idx)
        if compressor is None:
            return Z_k, Z_v

        # Preserve shape for reshape
        batch, seq, rank = Z_k.shape

        # Flatten for decompression
        Z_k_flat = Z_k.reshape(batch * seq, rank)
        Z_v_flat = Z_v.reshape(batch * seq, Z_v.size(-1))

        # Decompress
        K_hat_flat, V_hat_flat = compressor.decompress(Z_k_flat, Z_v_flat)

        # Reshape back
        d_k = K_hat_flat.size(-1)
        d_v = V_hat_flat.size(-1)
        K_hat = K_hat_flat.reshape(batch, seq, d_k)
        V_hat = V_hat_flat.reshape(batch, seq, d_v)

        return K_hat, V_hat


class KVSpliceHeadCompressor(nn.Module):
    """
    Per-head KVSplice compressor module.

    Learns linear projections for K and V compression.
    Optionally includes scale/shift parameters for latent space normalization.
    """

    def __init__(
        self,
        d_k: int,
        d_v: int,
        rank: int,
        learn_scale_shift: bool = True,
    ):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.rank = rank
        self.learn_scale_shift = learn_scale_shift

        # Projection matrices
        self.W_k_in = nn.Linear(d_k, rank, bias=False)
        self.W_v_in = nn.Linear(d_v, rank, bias=False)
        self.W_k_out = nn.Linear(rank, d_k, bias=False)
        self.W_v_out = nn.Linear(rank, d_v, bias=False)

        # Optional scale/shift for latent space
        if learn_scale_shift:
            self.scale_k = nn.Parameter(torch.zeros(rank))
            self.shift_k = nn.Parameter(torch.zeros(rank))
            self.scale_v = nn.Parameter(torch.zeros(rank))
            self.shift_v = nn.Parameter(torch.zeros(rank))
        else:
            self.register_parameter("scale_k", None)
            self.register_parameter("shift_k", None)
            self.register_parameter("scale_v", None)
            self.register_parameter("shift_v", None)

        # Initialize projection matrices (small random values)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values."""
        nn.init.normal_(self.W_k_in.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_v_in.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_k_out.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_v_out.weight, mean=0.0, std=0.02)

    def compress(self, K: torch.Tensor, V: torch.Tensor):
        """
        Compress K, V to latent space.

        Args:
            K: [N, d_k]
            V: [N, d_v]

        Returns:
            Z_k, Z_v: [N, rank]
        """
        Z_k = self.W_k_in(K)
        Z_v = self.W_v_in(V)

        # Apply scale/shift if enabled
        if self.learn_scale_shift:
            # Softplus ensures scale is positive
            scale_k = F.softplus(self.scale_k) + 1e-6
            scale_v = F.softplus(self.scale_v) + 1e-6

            Z_k = Z_k * scale_k + self.shift_k
            Z_v = Z_v * scale_v + self.shift_v

        return Z_k, Z_v

    def decompress(self, Z_k: torch.Tensor, Z_v: torch.Tensor):
        """
        Decompress latent space to K_hat, V_hat.

        Args:
            Z_k, Z_v: [N, rank]

        Returns:
            K_hat, V_hat: [N, d_k/d_v]
        """
        # Invert scale/shift if enabled
        if self.learn_scale_shift:
            scale_k = F.softplus(self.scale_k) + 1e-6
            scale_v = F.softplus(self.scale_v) + 1e-6

            Z_k = (Z_k - self.shift_k) / scale_k
            Z_v = (Z_v - self.shift_v) / scale_v

        K_hat = self.W_k_out(Z_k)
        V_hat = self.W_v_out(Z_v)

        return K_hat, V_hat
