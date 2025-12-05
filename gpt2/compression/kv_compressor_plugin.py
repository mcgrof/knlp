"""
KV Compressor Plugin (FlashBias-Inspired Architecture)

Plugin-based KV compression that runs BEFORE SDPA in the attention
forward pass, following the FlashBias low-rank decomposition pattern.

Three compression modes:
- Learned: MLA-style learned projections with LayerNorm
- PCA: Static SVD/PCA-calibrated projections
- Neural: Nonlinear MLP compress/expand

Key architectural requirement: Compression must happen BEFORE SDPA,
not via hooks that run after.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class KVCompressionConfig:
    """
    Configuration for KV compression modes.

    Examples:
        # V-only residual adapter (Phase 6 winner)
        config = KVCompressionConfig(
            mode="v_residual_adapter",
            rank=32,
            v_only=True,
            init_std=1e-3
        )

        # V-only PCA baseline
        config = KVCompressionConfig(
            mode="v_pca_linear",
            rank=32,
            v_only=True
        )
    """

    mode: str  # "v_residual_adapter", "v_pca_linear", "learned", "neural"
    rank: int  # Latent dimension
    v_only: bool = True  # Only compress V (K stays full rank)
    k_only: bool = False  # Only compress K (V stays full rank)
    init_std: float = 1e-3  # For residual adapter near-zero init


class KVCompressor(nn.Module, ABC):
    """
    Base plugin interface for KV compression.

    Following FlashBias pattern where compression happens at the same
    point in forward pass regardless of decomposition mode (learned,
    PCA, or neural).
    """

    def __init__(self, d_head: int, rank: int):
        super().__init__()
        self.d_head = d_head
        self.rank = rank

    @abstractmethod
    def compress_k(self, k: torch.Tensor) -> torch.Tensor:
        """
        Compress keys to latent dimension.

        Args:
            k: [B, H, T, d_head] or [B, T, d_head]

        Returns:
            k_latent: [B, H, T, rank] or [B, T, rank]
        """
        raise NotImplementedError

    @abstractmethod
    def compress_v(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compress values to latent dimension.

        Args:
            v: [B, H, T, d_head] or [B, T, d_head]

        Returns:
            v_latent: [B, H, T, rank] or [B, T, rank]
        """
        raise NotImplementedError

    def expand_k(self, k_latent: torch.Tensor) -> torch.Tensor:
        """
        Expand keys from latent dimension (optional).

        Args:
            k_latent: [B, H, T, rank] or [B, T, rank]

        Returns:
            k: [B, H, T, d_head] or [B, T, d_head]
        """
        raise NotImplementedError("Expand not implemented for this compressor")

    def expand_v(self, v_latent: torch.Tensor) -> torch.Tensor:
        """
        Expand values from latent dimension (optional).

        Args:
            v_latent: [B, H, T, rank] or [B, T, rank]

        Returns:
            v: [B, H, T, d_head] or [B, T, d_head]
        """
        raise NotImplementedError("Expand not implemented for this compressor")

    def forward(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress K and V before SDPA.

        Args:
            k: [B, H, T, d_head]
            v: [B, H, T, d_head]

        Returns:
            k_compressed: [B, H, T, rank]
            v_compressed: [B, H, T, rank]
        """
        k_compressed = self.compress_k(k)
        v_compressed = self.compress_v(v)
        return k_compressed, v_compressed


class LearnedKVCompressor(KVCompressor):
    """
    Mode A: Exact factorization with learned linear projections.

    MLA-style learned compression with LayerNorm in latent space.
    Trainable via gradient descent on logits MSE (teacher-student).
    """

    def __init__(
        self,
        d_head: int,
        rank: int,
        use_layernorm: bool = True,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(d_head, rank)
        self.use_layernorm = use_layernorm

        # Compression projections
        self.W_k = nn.Linear(d_head, rank, bias=False, dtype=dtype)
        self.W_v = nn.Linear(d_head, rank, bias=False, dtype=dtype)

        # Optional LayerNorm in latent space
        if use_layernorm:
            self.ln_k = nn.LayerNorm(rank, dtype=dtype)
            self.ln_v = nn.LayerNorm(rank, dtype=dtype)

        # Expansion projections (for reconstruction if needed)
        self.W_k_out = nn.Linear(rank, d_head, bias=False, dtype=dtype)
        self.W_v_out = nn.Linear(rank, d_head, bias=False, dtype=dtype)

        # Initialize to approximate identity
        self._init_weights()

    def _init_weights(self):
        """Initialize to near-identity transformation.

        For full-rank case (rank == d_head), uses exact identity
        matrices to guarantee perfect reconstruction. For compressed
        case (rank < d_head), uses orthogonal initialization for
        compression followed by transpose for expansion.
        """
        original_dtype = self.W_k.weight.dtype

        # CRITICAL FIX: Use exact identity for full-rank case
        # This guarantees the compressor path is mathematically a no-op
        if self.rank == self.d_head:
            with torch.no_grad():
                eye_k = torch.eye(
                    self.d_head,
                    self.d_head,
                    dtype=original_dtype,
                    device=self.W_k.weight.device,
                )
                eye_v = torch.eye(
                    self.d_head,
                    self.d_head,
                    dtype=original_dtype,
                    device=self.W_v.weight.device,
                )
                eye_out_k = torch.eye(
                    self.d_head,
                    self.d_head,
                    dtype=original_dtype,
                    device=self.W_k_out.weight.device,
                )
                eye_out_v = torch.eye(
                    self.d_head,
                    self.d_head,
                    dtype=original_dtype,
                    device=self.W_v_out.weight.device,
                )

                self.W_k.weight.copy_(eye_k)
                self.W_v.weight.copy_(eye_v)
                self.W_k_out.weight.copy_(eye_out_k)
                self.W_v_out.weight.copy_(eye_out_v)
            return

        # For rank < d_head: orthogonal init without noise
        # Symmetry breaking comes naturally from:
        # - Non-identical data across heads and layers
        # - Nonlinearity (LN, MLP in the block)
        # - Different K/V distributions
        # No need for artificial noise that just adds reconstruction
        # error
        with torch.no_grad():
            self.W_k.weight.data = self.W_k.weight.float()
            self.W_v.weight.data = self.W_v.weight.float()
            self.W_k_out.weight.data = self.W_k_out.weight.float()
            self.W_v_out.weight.data = self.W_v_out.weight.float()

        nn.init.orthogonal_(self.W_k.weight)
        nn.init.orthogonal_(self.W_v.weight)

        # Initialize expansion as (pseudo)inverse of compression
        # For orthogonal matrix: inverse = transpose
        with torch.no_grad():
            self.W_k_out.weight.copy_(self.W_k.weight.T)
            self.W_v_out.weight.copy_(self.W_v.weight.T)

        # Convert back to original dtype
        with torch.no_grad():
            self.W_k.weight.data = self.W_k.weight.to(dtype=original_dtype)
            self.W_v.weight.data = self.W_v.weight.to(dtype=original_dtype)
            self.W_k_out.weight.data = self.W_k_out.weight.to(dtype=original_dtype)
            self.W_v_out.weight.data = self.W_v_out.weight.to(dtype=original_dtype)

    def compress_k(self, k: torch.Tensor) -> torch.Tensor:
        """K: [B, H, T, d_head] -> [B, H, T, rank]"""
        k_latent = self.W_k(k)
        if self.use_layernorm:
            k_latent = self.ln_k(k_latent)
        return k_latent

    def compress_v(self, v: torch.Tensor) -> torch.Tensor:
        """V: [B, H, T, d_head] -> [B, H, T, rank]"""
        v_latent = self.W_v(v)
        if self.use_layernorm:
            v_latent = self.ln_v(v_latent)
        return v_latent

    def expand_k(self, k_latent: torch.Tensor) -> torch.Tensor:
        """K_latent: [B, H, T, rank] -> [B, H, T, d_head]"""
        return self.W_k_out(k_latent)

    def expand_v(self, v_latent: torch.Tensor) -> torch.Tensor:
        """V_latent: [B, H, T, rank] -> [B, H, T, d_head]"""
        return self.W_v_out(v_latent)


class PCAKVCompressor(KVCompressor):
    """
    Mode B: SVD/PCA decomposition with static projections.

    Initialized via offline PCA calibration on real data.
    Non-trainable after calibration.
    """

    def __init__(
        self,
        d_head: int,
        rank: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        super().__init__(d_head, rank)
        self.dtype = dtype
        self.device = device

        # PCA projection matrices (initialized as identity until calibrated)
        self.register_buffer(
            "W_k_proj", torch.eye(d_head, rank, dtype=dtype, device=device)
        )
        self.register_buffer(
            "W_k_recon", torch.eye(rank, d_head, dtype=dtype, device=device)
        )
        self.register_buffer(
            "W_v_proj", torch.eye(d_head, rank, dtype=dtype, device=device)
        )
        self.register_buffer(
            "W_v_recon", torch.eye(rank, d_head, dtype=dtype, device=device)
        )

    def load_pca_projections(
        self,
        W_k_proj: torch.Tensor,
        W_k_recon: torch.Tensor,
        W_v_proj: torch.Tensor,
        W_v_recon: torch.Tensor,
    ):
        """
        Load PCA projection matrices from calibration.

        Args:
            W_k_proj: [d_head, rank] - K compression
            W_k_recon: [rank, d_head] - K reconstruction
            W_v_proj: [d_head, rank] - V compression
            W_v_recon: [rank, d_head] - V reconstruction
        """
        self.W_k_proj.copy_(W_k_proj.to(self.dtype).to(self.device))
        self.W_k_recon.copy_(W_k_recon.to(self.dtype).to(self.device))
        self.W_v_proj.copy_(W_v_proj.to(self.dtype).to(self.device))
        self.W_v_recon.copy_(W_v_recon.to(self.dtype).to(self.device))

    def compress_k(self, k: torch.Tensor) -> torch.Tensor:
        """K: [B, H, T, d_head] -> [B, H, T, rank]"""
        return k @ self.W_k_proj

    def compress_v(self, v: torch.Tensor) -> torch.Tensor:
        """V: [B, H, T, d_head] -> [B, H, T, rank]"""
        return v @ self.W_v_proj

    def expand_k(self, k_latent: torch.Tensor) -> torch.Tensor:
        """K_latent: [B, H, T, rank] -> [B, H, T, d_head]"""
        return k_latent @ self.W_k_recon

    def expand_v(self, v_latent: torch.Tensor) -> torch.Tensor:
        """V_latent: [B, H, T, rank] -> [B, H, T, d_head]"""
        return v_latent @ self.W_v_recon


class VResidualAdapter(KVCompressor):
    """
    V-only residual adapter with configurable mode and scaling.

    Phase 6 winner - achieved 113.29 PPL on GPT-2 124M (only +5% vs teacher).

    Modes:
        - "residual": V_eff = V + init_scale * ΔV (default, preserves identity at init)
        - "full": V_eff = init_scale * ΔV (full replacement, no residual)
        - "cache": V_eff = W_v_out(W_v(V)) (pure low-rank, cacheable - stores compressed V)

    Architecture:
        V_latent = W_v(V)           # Project to rank-r
        ΔV = W_v_out(V_latent)      # Compute correction
        V_eff = V + init_scale*ΔV   # (residual mode)
        V_eff = init_scale * ΔV     # (full mode)
        V_eff = W_v_out(W_v(V))     # (cache mode - pure low-rank)

    Cache mode enables actual KV cache memory savings by storing only the
    compressed V representation (rank dimensions) instead of full V (d_head).
    This achieves 50% V-cache memory reduction (e.g., 32 vs 64 dims).

    Args:
        d_head: Head dimension (64 for GPT-2)
        rank: Latent dimension (typically d_head/2)
        init_std: Initialization std for weight init (default 1e-3)
        init_scale: Scaling factor for ΔV output (default 1e-3, ignored in cache mode)
        mode: "residual", "full", or "cache" (default "residual")
        dtype: Parameter dtype
        device: Device placement
    """

    def __init__(
        self,
        d_head: int,
        rank: int,
        init_std: float = 1e-3,
        init_scale: float = 1e-3,
        mode: str = "residual",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        super().__init__(d_head, rank)

        if mode not in ("residual", "full", "cache"):
            raise ValueError(f"mode must be 'residual', 'full', or 'cache', got '{mode}'")

        self.init_scale = init_scale
        self.mode = mode

        # Residual projection layers
        self.W_v = nn.Linear(d_head, rank, bias=False, dtype=dtype, device=device)
        self.W_v_out = nn.Linear(rank, d_head, bias=False, dtype=dtype, device=device)

        # Initialize weights (separate from output scaling)
        self.reset_parameters(init_std)

        # Storage for original V during compress/expand cycle
        self._original_v = None

    def reset_parameters(self, init_std: float = 1e-3):
        """Initialize weights with specified std."""
        with torch.no_grad():
            self.W_v.weight.normal_(mean=0.0, std=init_std)
            self.W_v_out.weight.normal_(mean=0.0, std=init_std)

    def compress_k(self, k: torch.Tensor) -> torch.Tensor:
        """K stays uncompressed (identity)."""
        return k

    def expand_k(self, k_latent: torch.Tensor) -> torch.Tensor:
        """K expansion is identity."""
        return k_latent

    def compress_v(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute V in latent space for residual connection.
        Note: Full V_eff is computed in expand_v using original V.
        """
        self._original_v = v
        return self.W_v(v)

    def expand_v(
        self, v_latent: torch.Tensor, v_original: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Expand V with mode-dependent behavior.

        Residual mode: V_eff = V + init_scale * ΔV
        Full mode: V_eff = init_scale * ΔV
        Cache mode: V_eff = W_v_out(v_latent) (pure low-rank reconstruction)

        Args:
            v_latent: Latent representation from compress_v
            v_original: Original V tensor for residual connection (optional, ignored in cache mode)
        """
        if self.mode == "cache":
            # Cache mode: pure low-rank reconstruction, no residual
            # This is the key for actual memory savings - we only store v_latent
            return self.W_v_out(v_latent)

        delta_v = self.W_v_out(v_latent)

        if self.mode == "full":
            # Full replacement - no residual connection
            return self.init_scale * delta_v

        # Residual mode
        v_orig = v_original if v_original is not None else self._original_v

        if v_orig is not None:
            result = v_orig + self.init_scale * delta_v
            # Clear stored value after use
            if v_original is None:
                self._original_v = None
            return result
        else:
            # If no original V available, just return scaled delta
            return self.init_scale * delta_v


class KResidualAdapter(KVCompressor):
    """
    K-only residual adapter - mirror of VResidualAdapter for K compression.

    Used to probe whether K has more low-rank structure than V.
    If M_K = I + s*W_k_out@W_k has more skewed singular values than M_V,
    K compression may be more viable than V compression.

    Modes:
        - "residual": K_eff = K + init_scale * ΔK (default)
        - "full": K_eff = init_scale * ΔK
        - "cache": K_eff = W_k_out(W_k(K)) (pure low-rank, cacheable)

    Args:
        d_head: Head dimension (64 for GPT-2)
        rank: Latent dimension
        init_std: Initialization std for weight init
        init_scale: Scaling factor for ΔK output
        mode: "residual", "full", or "cache"
        dtype: Parameter dtype
        device: Device placement
    """

    def __init__(
        self,
        d_head: int,
        rank: int,
        init_std: float = 1e-3,
        init_scale: float = 1e-3,
        mode: str = "residual",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        super().__init__(d_head, rank)

        if mode not in ("residual", "full", "cache"):
            raise ValueError(f"mode must be 'residual', 'full', or 'cache', got '{mode}'")

        self.init_scale = init_scale
        self.mode = mode

        # K projection layers (mirror of V in VResidualAdapter)
        self.W_k = nn.Linear(d_head, rank, bias=False, dtype=dtype, device=device)
        self.W_k_out = nn.Linear(rank, d_head, bias=False, dtype=dtype, device=device)

        # Initialize weights
        self.reset_parameters(init_std)

        # Storage for original K during compress/expand cycle
        self._original_k = None

    def reset_parameters(self, init_std: float = 1e-3):
        """Initialize weights with specified std."""
        with torch.no_grad():
            self.W_k.weight.normal_(mean=0.0, std=init_std)
            self.W_k_out.weight.normal_(mean=0.0, std=init_std)

    def compress_v(self, v: torch.Tensor) -> torch.Tensor:
        """V stays uncompressed (identity)."""
        return v

    def expand_v(self, v_latent: torch.Tensor, v_original: torch.Tensor = None) -> torch.Tensor:
        """V expansion is identity."""
        return v_latent

    def compress_k(self, k: torch.Tensor) -> torch.Tensor:
        """
        Compute K in latent space for residual connection.
        """
        self._original_k = k
        return self.W_k(k)

    def expand_k(
        self, k_latent: torch.Tensor, k_original: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Expand K with mode-dependent behavior.

        Residual mode: K_eff = K + init_scale * ΔK
        Full mode: K_eff = init_scale * ΔK
        Cache mode: K_eff = W_k_out(k_latent) (pure low-rank reconstruction)
        """
        if self.mode == "cache":
            return self.W_k_out(k_latent)

        delta_k = self.W_k_out(k_latent)

        if self.mode == "full":
            return self.init_scale * delta_k

        # Residual mode
        k_orig = k_original if k_original is not None else self._original_k

        if k_orig is not None:
            result = k_orig + self.init_scale * delta_k
            if k_original is None:
                self._original_k = None
            return result
        else:
            return self.init_scale * delta_k


def get_k_residual_operator_matrix(adapter: KResidualAdapter) -> torch.Tensor:
    """
    Extract the linear operator M_K from a trained K-only residual adapter.

    M_K = I + s * W_k_out @ W_k

    where s = init_scale.

    Args:
        adapter: Trained KResidualAdapter in residual mode

    Returns:
        M_K as a [d_head, d_head] matrix
    """
    if adapter.mode != "residual":
        raise ValueError(f"Expected mode='residual', got '{adapter.mode}'")

    with torch.no_grad():
        d = adapter.d_head
        s = adapter.init_scale
        W_k = adapter.W_k.weight  # [rank, d_head]
        W_k_out = adapter.W_k_out.weight  # [d_head, rank]

        I = torch.eye(d, dtype=W_k.dtype, device=W_k.device)
        M_K = I + s * (W_k_out @ W_k)

        return M_K


def analyze_k_residual_operator(adapter: KResidualAdapter) -> Dict:
    """
    Analyze the singular value spectrum of K residual adapter's operator M_K.
    """
    with torch.no_grad():
        M = get_k_residual_operator_matrix(adapter)
        U, S, Vh = torch.linalg.svd(M.float())

        total_energy = (S ** 2).sum()
        cumsum_energy = torch.cumsum(S ** 2, dim=0) / total_energy

        eff_rank_90 = (cumsum_energy < 0.90).sum().item() + 1
        eff_rank_99 = (cumsum_energy < 0.99).sum().item() + 1
        eff_rank_999 = (cumsum_energy < 0.999).sum().item() + 1

        return {
            "singular_values": S.cpu().numpy().tolist(),
            "top_5_sv": S[:5].cpu().numpy().tolist(),
            "effective_rank_90": eff_rank_90,
            "effective_rank_99": eff_rank_99,
            "effective_rank_999": eff_rank_999,
            "energy_in_top_16": cumsum_energy[15].item() if len(S) > 15 else 1.0,
            "energy_in_top_32": cumsum_energy[31].item() if len(S) > 31 else 1.0,
            "condition_number": (S[0] / S[-1]).item() if S[-1] > 0 else float("inf"),
        }


class NeuralKVCompressor(KVCompressor):
    """
    Mode C: Neural decomposition with nonlinear MLPs.

    Learned nonlinear compress/expand networks.
    More expressive than linear PCA/learned modes.
    """

    def __init__(
        self,
        d_head: int,
        rank: int,
        hidden_dim: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(d_head, rank)
        hidden_dim = hidden_dim or max(rank, d_head // 2)

        # K compressor MLP
        self.compress_k_net = nn.Sequential(
            nn.Linear(d_head, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, rank, dtype=dtype),
        )

        # V compressor MLP
        self.compress_v_net = nn.Sequential(
            nn.Linear(d_head, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, rank, dtype=dtype),
        )

        # K expander MLP
        self.expand_k_net = nn.Sequential(
            nn.Linear(rank, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, d_head, dtype=dtype),
        )

        # V expander MLP
        self.expand_v_net = nn.Sequential(
            nn.Linear(rank, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, d_head, dtype=dtype),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def compress_k(self, k: torch.Tensor) -> torch.Tensor:
        """K: [B, H, T, d_head] -> [B, H, T, rank]"""
        return self.compress_k_net(k)

    def compress_v(self, v: torch.Tensor) -> torch.Tensor:
        """V: [B, H, T, d_head] -> [B, H, T, rank]"""
        return self.compress_v_net(v)

    def expand_k(self, k_latent: torch.Tensor) -> torch.Tensor:
        """K_latent: [B, H, T, rank] -> [B, H, T, d_head]"""
        return self.expand_k_net(k_latent)

    def expand_v(self, v_latent: torch.Tensor) -> torch.Tensor:
        """V_latent: [B, H, T, rank] -> [B, H, T, d_head]"""
        return self.expand_v_net(v_latent)


def create_compressor(
    mode: str,
    d_head: int,
    rank: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    **kwargs,
) -> KVCompressor:
    """
    Factory function to create KV compressor plugin.

    Args:
        mode: "learned", "pca", or "neural"
        d_head: Head dimension (e.g., 64 for GPT-2)
        rank: Latent dimension (e.g., 16, 32)
        dtype: Data type for parameters
        device: Device to place parameters on
        **kwargs: Additional mode-specific arguments

    Returns:
        KVCompressor plugin instance
    """
    if mode == "learned":
        return LearnedKVCompressor(
            d_head=d_head,
            rank=rank,
            use_layernorm=kwargs.get("use_layernorm", True),
            dtype=dtype,
        ).to(device)

    elif mode == "pca":
        return PCAKVCompressor(d_head=d_head, rank=rank, dtype=dtype, device=device)

    elif mode == "neural":
        return NeuralKVCompressor(
            d_head=d_head,
            rank=rank,
            hidden_dim=kwargs.get("hidden_dim", None),
            dtype=dtype,
        ).to(device)

    elif mode == "v_residual_adapter":
        return VResidualAdapter(
            d_head=d_head,
            rank=rank,
            init_std=kwargs.get("init_std", 1e-3),
            init_scale=kwargs.get("init_scale", 1e-3),
            mode=kwargs.get("adapter_mode", "residual"),
            dtype=dtype,
            device=device,
        )

    else:
        raise ValueError(f"Unknown compressor mode: {mode}")


def create_per_head_compressors(
    mode: str,
    num_heads: int,
    d_head: int,
    rank: int,
    rank_schedule: Optional[Dict[int, int]] = None,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    **kwargs,
) -> nn.ModuleDict:
    """
    Create per-head compressor plugins with adaptive rank allocation.

    Args:
        mode: "learned", "pca", or "neural"
        num_heads: Number of attention heads
        d_head: Head dimension
        rank: Default rank (used if rank_schedule not provided)
        rank_schedule: Optional per-head rank allocation {head_idx: rank}
        dtype: Data type for parameters
        device: Device to place parameters on
        **kwargs: Additional mode-specific arguments

    Returns:
        ModuleDict mapping head_idx -> KVCompressor
    """
    compressors = nn.ModuleDict()

    for head_idx in range(num_heads):
        # Get rank for this head (adaptive or uniform)
        head_rank = rank_schedule.get(head_idx, rank) if rank_schedule else rank

        # Create compressor for this head
        compressors[str(head_idx)] = create_compressor(
            mode=mode,
            d_head=d_head,
            rank=head_rank,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    return compressors


def get_residual_operator_matrix(adapter: VResidualAdapter) -> torch.Tensor:
    """
    Extract the 64x64 linear operator M from a trained residual adapter.

    For a V-only residual adapter, the effective transformation is:
        V_eff = V + init_scale * W_v_out(W_v(V))
              = (I + init_scale * W_v_out @ W_v) @ V
              = M @ V

    where M = I + s * W_v_out @ W_v is a [d_head, d_head] matrix.

    Args:
        adapter: Trained VResidualAdapter in residual mode

    Returns:
        M: [d_head, d_head] linear operator tensor
    """
    if adapter.mode != "residual":
        raise ValueError(f"Expected mode='residual', got '{adapter.mode}'")

    with torch.no_grad():
        d = adapter.d_head
        s = adapter.init_scale

        # W_v: Linear(d_head -> rank), weight shape [rank, d_head]
        # W_v_out: Linear(rank -> d_head), weight shape [d_head, rank]
        W_v = adapter.W_v.weight  # [rank, d_head]
        W_v_out = adapter.W_v_out.weight  # [d_head, rank]

        # M = I + s * W_v_out @ W_v
        # W_v_out @ W_v: [d_head, rank] @ [rank, d_head] = [d_head, d_head]
        I = torch.eye(d, dtype=W_v.dtype, device=W_v.device)
        M = I + s * (W_v_out @ W_v)

        return M


def bake_residual_to_cache(
    adapter: VResidualAdapter,
    rank: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None,
) -> VResidualAdapter:
    """
    Convert a trained residual adapter to cache-mode via SVD bake-down.

    Given a trained residual adapter that defines M = I + s * W_v_out @ W_v,
    find the best rank-r approximation M ≈ U_r Σ_r V_r^T and create new
    cache-mode weights:
        W_cache = Σ_r^{1/2} @ V_r^T      (compress: d_head -> rank)
        W_cache_out = U_r @ Σ_r^{1/2}    (expand: rank -> d_head)

    Such that W_cache_out @ W_cache ≈ M.

    This allows the cache-mode adapter to approximate the residual-corrected V
    while actually storing only the compressed representation.

    Args:
        adapter: Trained VResidualAdapter in residual mode
        rank: Target rank for cache mode (default: adapter.rank)
        dtype: Output dtype (default: adapter's dtype)
        device: Output device (default: adapter's device)

    Returns:
        New VResidualAdapter in cache mode with baked weights
    """
    if adapter.mode != "residual":
        raise ValueError(f"Expected mode='residual', got '{adapter.mode}'")

    d = adapter.d_head
    r = rank if rank is not None else adapter.rank
    r = min(r, d)

    _dtype = dtype if dtype is not None else adapter.W_v.weight.dtype
    _device = device if device is not None else adapter.W_v.weight.device

    with torch.no_grad():
        # Get the full residual operator M = I + s * W_v_out @ W_v
        M = get_residual_operator_matrix(adapter)  # [d, d]

        # SVD: M = U @ diag(S) @ Vh
        # Note: torch.linalg.svd returns Vh (V transpose), not V
        U, S, Vh = torch.linalg.svd(M.float())  # [d,d], [d], [d,d]

        # Take top-r components
        U_r = U[:, :r]  # [d, r]
        S_r = S[:r]  # [r]
        V_r = Vh[:r, :].T  # [d, r] (transpose of Vh[:r, :])

        # Build sqrt of singular values for symmetric factorization
        S_sqrt = torch.sqrt(S_r)  # [r]

        # W_cache: compress d_head -> rank
        # W_cache = Σ_r^{1/2} @ V_r^T = diag(S_sqrt) @ V_r^T
        # Shape: [r, d]
        W_cache = S_sqrt.unsqueeze(1) * V_r.T  # [r, d]

        # W_cache_out: expand rank -> d_head
        # W_cache_out = U_r @ Σ_r^{1/2} = U_r @ diag(S_sqrt)
        # Shape: [d, r]
        W_cache_out = U_r * S_sqrt.unsqueeze(0)  # [d, r]

        # Verify: W_cache_out @ W_cache ≈ U_r @ S_r @ V_r^T ≈ M
        # reconstruction_error = torch.norm(W_cache_out @ W_cache - M) / torch.norm(M)

        # Create new cache-mode adapter
        new_adapter = VResidualAdapter(
            d_head=d,
            rank=r,
            init_std=1.0,  # Not used, weights will be overwritten
            init_scale=1.0,  # Scale is baked into weights
            mode="cache",
            dtype=_dtype,
            device=_device,
        )

        # Copy baked weights
        # W_v: Linear(d_head -> rank), weight is [rank, d_head]
        # W_cache is [r, d] = [rank, d_head] - correct
        new_adapter.W_v.weight.data.copy_(W_cache.to(_dtype))
        # W_v_out: Linear(rank -> d_head), weight is [d_head, rank]
        # W_cache_out is [d, r] = [d_head, rank] - correct
        new_adapter.W_v_out.weight.data.copy_(W_cache_out.to(_dtype))

        return new_adapter


def analyze_residual_operator(adapter: VResidualAdapter) -> Dict:
    """
    Analyze the singular value spectrum of a residual adapter's operator M.

    Useful for understanding effective rank and potential for compression.

    Args:
        adapter: Trained VResidualAdapter in residual mode

    Returns:
        Dict with singular values, effective rank, energy distribution
    """
    with torch.no_grad():
        M = get_residual_operator_matrix(adapter)
        U, S, Vh = torch.linalg.svd(M.float())

        # Effective rank (using energy threshold)
        total_energy = (S ** 2).sum()
        cumsum_energy = torch.cumsum(S ** 2, dim=0) / total_energy

        # Find rank needed for 99%, 99.9%, 99.99% energy
        eff_rank_99 = (cumsum_energy < 0.99).sum().item() + 1
        eff_rank_999 = (cumsum_energy < 0.999).sum().item() + 1
        eff_rank_9999 = (cumsum_energy < 0.9999).sum().item() + 1

        return {
            "singular_values": S.cpu().numpy().tolist(),
            "top_5_sv": S[:5].cpu().numpy().tolist(),
            "effective_rank_99": eff_rank_99,
            "effective_rank_999": eff_rank_999,
            "effective_rank_9999": eff_rank_9999,
            "energy_in_top_16": cumsum_energy[15].item() if len(S) > 15 else 1.0,
            "energy_in_top_32": cumsum_energy[31].item() if len(S) > 31 else 1.0,
            "condition_number": (S[0] / S[-1]).item() if S[-1] > 0 else float("inf"),
        }
