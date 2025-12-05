"""
PCA Compressor: Variance-Based Compression

Fast calibration-only compression using PCA (no training required).
Useful as baseline comparison against learned methods like KVSplice.

Future: Can add spline nonlinearity for heavy-tailed distributions.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from gpt2.compression.base import KVCompressorBase


class PCACompressor(KVCompressorBase):
    """
    PCA-based KV compressor.

    For each (layer, head):
    - Computes PCA basis from calibration data
    - Projects K, V onto top-k principal components
    - No training required, only calibration statistics

    Compression: Z_k = (K - mean_k) @ U_k[:, :rank]
    Decompression: K_hat = Z_k @ U_k[:, :rank].T + mean_k
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Per-layer/head PCA parameters (registered as buffers, not parameters)
        self.pca_modules = nn.ModuleDict()

        # Store eigenvalue spectra for FIM-guided rank selection
        self.eigenvalue_spectra = {}

        # Initialize PCA modules for each enabled layer/head
        for (layer_idx, head_idx), cfg in self.layer_head_configs.items():
            if not cfg.get("enabled", False):
                continue

            rank = cfg.get("rank", 32)
            d_k = cfg.get("d_k", self.d_head)
            d_v = cfg.get("d_v", self.d_head)

            key = f"L{layer_idx}_H{head_idx}"
            self.pca_modules[key] = PCAHeadCompressor(
                d_k=d_k,
                d_v=d_v,
                rank=rank,
            )

    def _get_head_compressor(self, layer_idx: int, head_idx: int):
        """Get PCA module for specific layer/head."""
        key = f"L{layer_idx}_H{head_idx}"
        return self.pca_modules[key] if key in self.pca_modules else None

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
        Collect KV samples for PCA calibration.

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
        Compute PCA bases from calibration data.

        Uses SVD to find principal components.
        """
        self.calibration_mode = False

        # Get device from calibration samples (PCA uses buffers, not parameters)
        device = None
        for samples in self.calibration_data.values():
            if samples:
                device = samples[0][0].device
                break
        if device is None:
            device = torch.device("cpu")

        for (layer_idx, head_idx), samples in self.calibration_data.items():
            if not samples:
                continue

            # Concatenate all samples
            K_all = torch.cat([K for K, V in samples], dim=0).to(device)
            V_all = torch.cat([V for K, V in samples], dim=0).to(device)

            # Get PCA module for this head
            pca_module = self._get_head_compressor(layer_idx, head_idx)
            if pca_module is None:
                continue

            # Fit PCA (now also saves eigenvalue spectra)
            self._fit_pca(pca_module, K_all, V_all, layer_idx, head_idx)

            print(
                f"  Layer {layer_idx}, Head {head_idx}: "
                f"PCA fitted (rank={pca_module.rank})"
            )

        # Clear calibration data to free memory
        self.calibration_data.clear()

    def _fit_pca(
        self,
        pca_module: "PCAHeadCompressor",
        K: torch.Tensor,
        V: torch.Tensor,
        layer_idx: int = None,
        head_idx: int = None,
    ):
        """
        Fit PCA bases for K and V.

        Args:
            pca_module: PCAHeadCompressor module
            K, V: Calibration samples [N, d_k/d_v]
            layer_idx, head_idx: For eigenvalue spectrum storage
        """
        # Compute mean
        mean_k = K.mean(dim=0)
        mean_v = V.mean(dim=0)

        # Center data
        K_centered = K - mean_k
        V_centered = V - mean_v

        # Compute covariance and eigendecomposition for PCA
        # Cov = X^T X / (N-1), eigenvectors of Cov are principal components
        N = K.size(0)

        # Ensure FP32 for eigendecomposition (FP16 not supported on CPU)
        K_centered_fp32 = K_centered.float()
        V_centered_fp32 = V_centered.float()

        cov_k = (K_centered_fp32.T @ K_centered_fp32) / (N - 1)
        cov_v = (V_centered_fp32.T @ V_centered_fp32) / (N - 1)

        # Eigendecomposition: Cov = V @ diag(λ) @ V^T
        eigvals_k, eigvecs_k = torch.linalg.eigh(cov_k)
        eigvals_v, eigvecs_v = torch.linalg.eigh(cov_v)

        # Sort eigenvalues and eigenvectors in descending order
        idx_k = torch.argsort(eigvals_k, descending=True)
        idx_v = torch.argsort(eigvals_v, descending=True)

        S_k = eigvals_k[idx_k]
        S_v = eigvals_v[idx_v]
        U_k = eigvecs_k[:, idx_k]  # [d, d]
        U_v = eigvecs_v[:, idx_v]  # [d, d]

        # Store eigenvalue spectra (eigenvalues of covariance matrix)
        if layer_idx is not None and head_idx is not None:
            eigenvalues_k = S_k.cpu().numpy()
            eigenvalues_v = S_v.cpu().numpy()

            cumvar_k = eigenvalues_k.cumsum() / eigenvalues_k.sum()
            cumvar_v = eigenvalues_v.cumsum() / eigenvalues_v.sum()

            self.eigenvalue_spectra[(layer_idx, head_idx)] = {
                "eigenvalues_k": eigenvalues_k.tolist(),
                "eigenvalues_v": eigenvalues_v.tolist(),
                "cumulative_variance_k": cumvar_k.tolist(),
                "cumulative_variance_v": cumvar_v.tolist(),
            }

        # Store top-k principal components
        rank = pca_module.rank
        pca_module.mean_k.copy_(mean_k)
        pca_module.mean_v.copy_(mean_v)
        pca_module.U_k.copy_(U_k[:, :rank].T)  # [rank, d_k]
        pca_module.U_v.copy_(U_v[:, :rank].T)  # [rank, d_v]

        # Store explained variance (for diagnostics)
        total_var_k = S_k.sum()
        total_var_v = S_v.sum()
        explained_var_k = S_k[:rank].sum() / total_var_k
        explained_var_v = S_v[:rank].sum() / total_var_v

        print(
            f"    Explained variance: K={explained_var_k:.3f}, V={explained_var_v:.3f}"
        )

    def compress(
        self,
        layer_idx: int,
        head_idx: int,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress KV pairs using PCA projection.

        Args:
            K, V: [batch, seq, d_k/d_v]

        Returns:
            Z_k, Z_v: [batch, seq, rank]
        """
        if not self.is_enabled(layer_idx, head_idx):
            return K, V

        pca_module = self._get_head_compressor(layer_idx, head_idx)
        if pca_module is None:
            return K, V

        # Preserve shape for reshape
        batch, seq, d = K.shape

        # Flatten for compression
        K_flat = K.reshape(batch * seq, d)
        V_flat = V.reshape(batch * seq, V.size(-1))

        # Compress
        Z_k_flat, Z_v_flat = pca_module.compress(K_flat, V_flat)

        # Reshape back
        rank = Z_k_flat.size(-1)
        Z_k = Z_k_flat.reshape(batch, seq, rank)
        Z_v = Z_v_flat.reshape(batch, seq, rank)

        return Z_k, Z_v

    def decompress(
        self,
        layer_idx: int,
        head_idx: int,
        Z_k: torch.Tensor,
        Z_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress KV pairs using PCA inverse.

        Args:
            Z_k, Z_v: [batch, seq, rank]

        Returns:
            K_hat, V_hat: [batch, seq, d_k/d_v]
        """
        if not self.is_enabled(layer_idx, head_idx):
            return Z_k, Z_v

        pca_module = self._get_head_compressor(layer_idx, head_idx)
        if pca_module is None:
            return Z_k, Z_v

        # Preserve shape for reshape
        batch, seq, rank = Z_k.shape

        # Flatten for decompression
        Z_k_flat = Z_k.reshape(batch * seq, rank)
        Z_v_flat = Z_v.reshape(batch * seq, rank)

        # Decompress
        K_hat_flat, V_hat_flat = pca_module.decompress(Z_k_flat, Z_v_flat)

        # Reshape back
        d_k = K_hat_flat.size(-1)
        d_v = V_hat_flat.size(-1)
        K_hat = K_hat_flat.reshape(batch, seq, d_k)
        V_hat = V_hat_flat.reshape(batch, seq, d_v)

        return K_hat, V_hat

    def save_eigenvalue_spectra(self, path: str):
        """
        Export eigenvalue spectra to JSON for FIM-guided rank selection.

        Args:
            path: Output JSON file path
        """
        import json

        # Convert eigenvalue spectra to JSON-serializable format
        spectra_dict = {}
        for (layer_idx, head_idx), data in self.eigenvalue_spectra.items():
            key = f"{layer_idx}/{head_idx}"
            spectra_dict[key] = data

        # Write to JSON
        with open(path, "w") as f:
            json.dump(spectra_dict, f, indent=2)

        print(f"Eigenvalue spectra saved to {path}")
        if spectra_dict:
            avg_dim = sum(len(d['eigenvalues_k']) for d in spectra_dict.values()) // len(spectra_dict)
            print(f"  Total heads: {len(spectra_dict)}, Avg eigenvalue dimension: {avg_dim}")
        else:
            print(f"  Warning: No eigenvalue spectra were saved (empty dictionary)")


class PCAHeadCompressor(nn.Module):
    """
    Per-head PCA compressor module.

    Stores PCA bases and means as buffers (not trainable parameters).
    """

    def __init__(self, d_k: int, d_v: int, rank: int):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.rank = rank

        # Register buffers (not trainable, but saved in state_dict)
        self.register_buffer("mean_k", torch.zeros(d_k))
        self.register_buffer("mean_v", torch.zeros(d_v))
        self.register_buffer("U_k", torch.zeros(rank, d_k))
        self.register_buffer("U_v", torch.zeros(rank, d_v))

    def compress(self, K: torch.Tensor, V: torch.Tensor):
        """
        Project K, V onto PCA basis.

        Args:
            K: [N, d_k]
            V: [N, d_v]

        Returns:
            Z_k, Z_v: [N, rank]
        """
        # Center and project
        K_centered = K - self.mean_k
        V_centered = V - self.mean_v

        # Z = X @ U.T  (where U is [rank, d])
        Z_k = K_centered @ self.U_k.T
        Z_v = V_centered @ self.U_v.T

        return Z_k, Z_v

    def decompress(self, Z_k: torch.Tensor, Z_v: torch.Tensor):
        """
        Reconstruct K, V from PCA coefficients.

        Args:
            Z_k, Z_v: [N, rank]

        Returns:
            K_hat, V_hat: [N, d_k/d_v]
        """
        # X_hat = Z @ U + mean
        K_hat = Z_k @ self.U_k + self.mean_k
        V_hat = Z_v @ self.U_v + self.mean_v

        return K_hat, V_hat
