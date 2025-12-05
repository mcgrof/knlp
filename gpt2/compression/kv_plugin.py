"""
Universal KV-Cache Compression Plugin v3.0

A unified pluggable framework for K/V cache compression on any HF-loadable model.

This module provides:
- Base KVCompressor class with compress/expand API
- Multiple compressor implementations (PCA, TopK, Hybrid, SVD)
- CompressedKVCache manager for storing compressed representations
- KVPlugin wrapper for patching HuggingFace models
- Presets for common deployment scenarios

Usage:
    from gpt2.compression.kv_plugin import KVPlugin

    # Load model
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # Create plugin from preset
    plugin = KVPlugin.from_preset("balanced", model)

    # Calibrate (optional for some compressors)
    plugin.calibrate(calibration_tokens)

    # Use model normally - cache is now compressed
    outputs = model.generate(inputs, use_cache=True)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class KVCompressorConfig:
    """Configuration for a KV compressor."""

    d_input: int  # Input dimension (head_dim or d_model)
    d_compressed: int  # Compressed dimension
    n_heads: int = 1  # Number of attention heads
    per_head: bool = True  # Whether to compress per-head or globally
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    # Quantization settings (optional)
    quant_bits: Optional[int] = None  # None = no quant, 8 = int8, 4 = int4
    quant_target: str = "v"  # "v" = V latent only, "kv" = both K and V
    quant_per_channel: bool = True  # per-feature scaling vs global
    quant_storage: bool = False  # True = real int storage, False = fake quant
    quant_backend: str = "torch"  # "torch" or "triton" for fused kernels


@dataclass
class KVPluginConfig:
    """Configuration for the KV plugin."""

    # Model info (auto-detected)
    n_layers: int = 12
    n_heads: int = 12
    head_dim: int = 64
    d_model: int = 768

    # Compression settings
    compressor_type: str = "hybrid"  # pca, topk, hybrid, svd, identity
    d_compressed: int = 128  # Target compressed dimension
    d_latent: int = 256  # Intermediate latent dim (for hybrid)

    # Gate settings (for topk/hybrid)
    gate_type: str = "topk"  # topk, soft, group
    topk_k: int = 128  # Number of channels to keep

    # Quantization settings (optional latent quantization)
    quant_bits: Optional[int] = None  # None = no quant, 8 = int8, 4 = int4
    quant_target: str = "v"  # "v" = V latent only, "kv" = both K and V
    quant_per_channel: bool = True  # per-feature scaling vs global
    quant_storage: bool = False  # True = real int storage, False = fake quant
    quant_backend: str = "torch"  # "torch" or "triton" for fused kernels

    # Per-layer customization
    layer_configs: Optional[Dict[int, Dict]] = None

    # Runtime settings
    expand_mode: str = "pre"  # pre (expand before FA), lazy (future)
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


# =============================================================================
# Quantization Utilities
# =============================================================================


@dataclass
class QuantizedTensor:
    """Container for quantized tensor with scale factors.

    Supports both int8 (unpacked) and int4 (packed, 2 values per byte).

    Attributes:
        data: Quantized data. For int8: same shape as original.
              For int4: last dim is halved (packed).
        scale: Scale factors for dequantization.
        bits: Quantization bits (4 or 8).
        per_channel: Whether scaling is per-channel or global.
        dim: Dimension along which scaling was computed.
        packed: Whether data is packed (True for int4).
        original_dim: Original size of last dimension (for int4 unpacking).
    """

    data: torch.Tensor  # int8 or packed uint8 (for int4)
    scale: torch.Tensor  # scale factors for dequantization
    bits: int = 8
    per_channel: bool = True
    dim: int = -1  # dimension along which scaling was computed
    packed: bool = False  # True for int4 (2 values per byte)
    original_dim: int = 0  # Original D for int4 unpacking

    def dequantize(self) -> torch.Tensor:
        """Convert back to floating point."""
        if self.bits == 4 and self.packed:
            return dequantize_from_int4(self)
        return (self.data.float() * self.scale).to(self.scale.dtype)

    @property
    def shape(self):
        """Return logical shape (unpacked for int4)."""
        if self.packed and self.original_dim > 0:
            return self.data.shape[:-1] + (self.original_dim,)
        return self.data.shape

    def numel(self):
        """Return logical number of elements (unpacked for int4)."""
        if self.packed and self.original_dim > 0:
            return self.data.numel() * 2  # 2 values per byte
        return self.data.numel()

    def element_size(self):
        """Return element size of underlying storage."""
        return self.data.element_size()

    def memory_bytes(self) -> int:
        """Total memory including scale factors."""
        return (
            self.data.numel() * self.data.element_size()
            + self.scale.numel() * self.scale.element_size()
        )


def quantize_to_int8(
    x: torch.Tensor,
    per_channel: bool = True,
    dim: int = -1,
) -> QuantizedTensor:
    """
    Quantize tensor to int8 with real integer storage.

    Args:
        x: Input tensor (float16/32)
        per_channel: Per-channel vs global scaling
        dim: Dimension for per-channel scaling

    Returns:
        QuantizedTensor with int8 data and scale factors
    """
    Q = 127  # int8 range: [-127, 127]

    with torch.no_grad():
        if per_channel:
            scale = x.abs().amax(dim=dim, keepdim=True) / Q
        else:
            scale = x.abs().amax() / Q

        scale = torch.clamp(scale, min=1e-8)

        # Quantize to int8
        x_int = torch.round(x / scale).clamp(-Q, Q).to(torch.int8)

    return QuantizedTensor(
        data=x_int,
        scale=scale,
        bits=8,
        per_channel=per_channel,
        dim=dim,
    )


def dequantize_from_int8(
    qt: QuantizedTensor, dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Dequantize int8 tensor back to floating point.

    Args:
        qt: QuantizedTensor from quantize_to_int8
        dtype: Target dtype for output

    Returns:
        Dequantized tensor in target dtype
    """
    return (qt.data.float() * qt.scale).to(dtype)


def quantize_to_int4(
    x: torch.Tensor,
    per_channel: bool = True,
    dim: int = -1,
) -> QuantizedTensor:
    """
    Quantize tensor to int4 with packed storage (2 values per byte).

    Args:
        x: Input tensor (float16/32). Last dim must be even.
        per_channel: Per-channel vs global scaling
        dim: Dimension for per-channel scaling

    Returns:
        QuantizedTensor with packed int4 data (uint8, half the size)
    """
    Q = 7  # int4 symmetric range: [-7, 7] (we use -7..7, not -8..7 for symmetry)
    D = x.shape[-1]

    if D % 2 != 0:
        raise ValueError(f"Last dimension must be even for int4 packing, got {D}")

    with torch.no_grad():
        if per_channel:
            scale = x.abs().amax(dim=dim, keepdim=True) / Q
        else:
            scale = x.abs().amax() / Q

        scale = torch.clamp(scale, min=1e-8)

        # Quantize to int4 range
        x_scaled = torch.round(x / scale).clamp(-Q, Q).to(torch.int8)

        # Pack two int4s into one uint8: low nibble + high nibble
        # x_scaled[..., 0::2] goes to low nibble, x_scaled[..., 1::2] to high nibble
        x_low = x_scaled[..., 0::2] & 0x0F  # Mask to 4 bits (handles negative)
        x_high = (x_scaled[..., 1::2] & 0x0F) << 4  # Shift to high nibble
        packed = (x_low | x_high).to(torch.uint8)

    return QuantizedTensor(
        data=packed,
        scale=scale,
        bits=4,
        per_channel=per_channel,
        dim=dim,
        packed=True,
        original_dim=D,
    )


def dequantize_from_int4(
    qt: QuantizedTensor, dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Dequantize packed int4 tensor back to floating point.

    Args:
        qt: QuantizedTensor from quantize_to_int4
        dtype: Target dtype for output

    Returns:
        Dequantized tensor in target dtype
    """
    packed = qt.data
    scale = qt.scale
    D = qt.original_dim

    # Unpack: low nibble and high nibble
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)

    # Convert from unsigned 4-bit [0..15] to signed [-8..7]
    # Values were stored as (signed_val & 0x0F), so we need to sign-extend
    # If bit 3 is set (val >= 8), it was negative: subtract 16
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)

    # Interleave back to original shape
    x_int = torch.empty(*packed.shape[:-1], D, dtype=torch.int8, device=packed.device)
    x_int[..., 0::2] = low
    x_int[..., 1::2] = high

    # Dequantize
    return (x_int.float() * scale).to(dtype)


def fake_quant(
    x: torch.Tensor,
    bits: int,
    per_channel: bool = True,
    dim: int = -1,
) -> torch.Tensor:
    """
    Fake quantization for simulation (STE-style).

    Simulates symmetric integer quantization without actually storing as int.
    Useful for evaluating quantization impact before implementing real int storage.

    Args:
        x: Input tensor to quantize
        bits: Number of bits (e.g., 8 for int8, 4 for int4)
        per_channel: If True, compute scale per-channel along dim; else global
        dim: Dimension for per-channel scaling (default: last dim = features)

    Returns:
        Fake-quantized tensor (same dtype as input, but values are quantized)
    """
    # Symmetric int quant: values in [-Q, Q]
    Q = 2 ** (bits - 1) - 1  # 127 for 8-bit, 7 for 4-bit

    with torch.no_grad():
        if per_channel:
            # Per-feature / per-head scaling
            scale = x.detach().abs().amax(dim=dim, keepdim=True) / Q
        else:
            # Global scaling
            scale = x.detach().abs().amax() / Q

        # Avoid division by zero
        scale = torch.clamp(scale, min=1e-8)

    # Quantize and dequantize (STE: gradient flows through)
    x_q = torch.round(x / scale).clamp(-Q, Q)
    return x_q * scale


def fake_quant_affine(
    x: torch.Tensor,
    bits: int,
    per_channel: bool = True,
    dim: int = -1,
) -> torch.Tensor:
    """
    Asymmetric fake quantization (affine).

    Uses both scale and zero-point for better dynamic range utilization.
    Slightly more complex but can be better for non-symmetric distributions.

    Args:
        x: Input tensor to quantize
        bits: Number of bits
        per_channel: Per-channel vs global scaling
        dim: Dimension for per-channel scaling

    Returns:
        Fake-quantized tensor
    """
    Q_min = 0
    Q_max = 2**bits - 1

    with torch.no_grad():
        if per_channel:
            x_min = x.detach().amin(dim=dim, keepdim=True)
            x_max = x.detach().amax(dim=dim, keepdim=True)
        else:
            x_min = x.detach().amin()
            x_max = x.detach().amax()

        scale = (x_max - x_min) / (Q_max - Q_min)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = Q_min - x_min / scale

    # Quantize and dequantize
    x_q = torch.round(x / scale + zero_point).clamp(Q_min, Q_max)
    return (x_q - zero_point) * scale


# =============================================================================
# Base Compressor Class
# =============================================================================


class KVCompressor(nn.Module, ABC):
    """
    Abstract base class for KV compressors.

    All compressors must implement:
    - compress(x): Convert full representation to compressed
    - expand(z): Convert compressed back to full representation

    Optional:
    - calibrate(samples): Compute compression parameters from data
    """

    def __init__(self, config: KVCompressorConfig):
        super().__init__()
        self.config = config
        self.d_input = config.d_input
        self.d_compressed = config.d_compressed
        self.calibrated = False

    @abstractmethod
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress input tensor.

        Args:
            x: [..., d_input] tensor

        Returns:
            z: [..., d_compressed] tensor
        """
        raise NotImplementedError

    @abstractmethod
    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """
        Expand compressed tensor back to full dimension.

        Args:
            z: [..., d_compressed] tensor

        Returns:
            x_hat: [..., d_input] tensor (reconstruction)
        """
        raise NotImplementedError

    def calibrate(self, samples: torch.Tensor) -> None:
        """
        Calibrate compressor from samples.

        Args:
            samples: [N, d_input] tensor of calibration samples
        """
        pass  # Default: no calibration needed

    @property
    def compression_ratio(self) -> float:
        """Return the compression ratio (input/compressed)."""
        return self.d_input / self.d_compressed

    def extra_repr(self) -> str:
        return f"d_input={self.d_input}, d_compressed={self.d_compressed}, ratio={self.compression_ratio:.1f}x"


# =============================================================================
# Identity Compressor (Baseline)
# =============================================================================


class IdentityCompressor(KVCompressor):
    """No compression - passes through unchanged."""

    def __init__(self, config: KVCompressorConfig):
        # Override d_compressed to match d_input
        config = KVCompressorConfig(
            d_input=config.d_input,
            d_compressed=config.d_input,
            n_heads=config.n_heads,
            per_head=config.per_head,
            device=config.device,
            dtype=config.dtype,
        )
        super().__init__(config)
        self.calibrated = True

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        return z


# =============================================================================
# PCA Compressor
# =============================================================================


class PCACompressor(KVCompressor):
    """
    PCA-based compression using offline-computed principal components.

    Requires calibration to compute PCA basis.
    """

    def __init__(self, config: KVCompressorConfig):
        super().__init__(config)

        # PCA components: [d_compressed, d_input]
        self.register_buffer(
            "components",
            torch.zeros(config.d_compressed, config.d_input, dtype=config.dtype),
        )
        # Mean for centering: [d_input]
        self.register_buffer("mean", torch.zeros(config.d_input, dtype=config.dtype))

    def calibrate(self, samples: torch.Tensor) -> None:
        """Compute PCA components from calibration samples."""
        samples = samples.float()

        # Center data
        mean = samples.mean(dim=0)
        centered = samples - mean

        # SVD for PCA
        # [N, d_input] -> U [N, k], S [k], Vh [k, d_input]
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)

        # Take top-k components
        components = Vh[: self.d_compressed]

        self.mean.copy_(mean.to(self.config.dtype))
        self.components.copy_(components.to(self.config.dtype))
        self.calibrated = True

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto PCA basis."""
        # x: [..., d_input]
        centered = x - self.mean
        # [..., d_input] @ [d_input, d_compressed] -> [..., d_compressed]
        return centered @ self.components.T

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from PCA basis."""
        # z: [..., d_compressed]
        # [..., d_compressed] @ [d_compressed, d_input] -> [..., d_input]
        return z @ self.components + self.mean


# =============================================================================
# TopK Energy Gate Compressor
# =============================================================================


class TopKCompressor(KVCompressor):
    """
    Energy-gated compression keeping top-k channels by energy.

    Requires calibration to determine which channels to keep.
    """

    def __init__(self, config: KVCompressorConfig):
        super().__init__(config)

        # Indices of top-k channels: [d_compressed]
        self.register_buffer(
            "keep_indices",
            torch.arange(config.d_compressed, dtype=torch.long),
        )
        # Energy scores for all channels: [d_input]
        self.register_buffer(
            "energy_scores",
            torch.ones(config.d_input, dtype=config.dtype),
        )

    def calibrate(self, samples: torch.Tensor) -> None:
        """Compute channel energy and select top-k."""
        samples = samples.float()

        # Compute per-channel energy (variance)
        energy = samples.var(dim=0)

        # Select top-k indices
        _, indices = energy.topk(self.d_compressed)
        indices = indices.sort().values  # Sort for consistent ordering

        self.energy_scores.copy_(energy.to(self.config.dtype))
        self.keep_indices.copy_(indices)
        self.calibrated = True

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Select top-k channels."""
        # x: [..., d_input] -> [..., d_compressed]
        return x[..., self.keep_indices]

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """Scatter back to full dimension (zeros elsewhere)."""
        # z: [..., d_compressed] -> [..., d_input]
        shape = list(z.shape[:-1]) + [self.d_input]
        output = torch.zeros(shape, dtype=z.dtype, device=z.device)
        output[..., self.keep_indices] = z
        return output


# =============================================================================
# SVD Low-Rank Compressor
# =============================================================================


class SVDCompressor(KVCompressor):
    """
    SVD-based low-rank compression.

    Similar to PCA but without mean centering.
    """

    def __init__(self, config: KVCompressorConfig):
        super().__init__(config)

        # Right singular vectors: [d_compressed, d_input]
        self.register_buffer(
            "V",
            torch.zeros(config.d_compressed, config.d_input, dtype=config.dtype),
        )

    def calibrate(self, samples: torch.Tensor) -> None:
        """Compute SVD basis from calibration samples."""
        samples = samples.float()

        # SVD: [N, d_input] -> U [N, k], S [k], Vh [k, d_input]
        _, _, Vh = torch.linalg.svd(samples, full_matrices=False)

        # Take top-k right singular vectors
        V = Vh[: self.d_compressed]

        self.V.copy_(V.to(self.config.dtype))
        self.calibrated = True

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto SVD basis."""
        return x @ self.V.T

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from SVD basis."""
        return z @ self.V


# =============================================================================
# Orthogonal Low-Rank Compressor (No Calibration)
# =============================================================================


class OrthogonalCompressor(KVCompressor):
    """
    Pure learned low-rank projection with orthogonal initialization.

    Based on the KVSplice approach from MLA training: uses two linear layers
    with orthogonal init and no calibration required. The expand weight is
    initialized as the transpose of compress weight, making it an approximate
    pseudoinverse.

    Advantages over PCA/SVD compressors:
    - No calibration data needed (works out of the box)
    - No SVD computation (faster setup)
    - Weights can be fine-tuned during training if needed
    - Similar reconstruction quality for random orthogonal basis

    Optional quantization:
    - Set config.quant_bits to enable latent quantization (8 or 4 bit)
    - Useful for evaluating memory savings vs quality tradeoff
    """

    def __init__(self, config: KVCompressorConfig):
        super().__init__(config)

        # Low-rank projection layers (no bias for efficiency)
        self.compress_proj = nn.Linear(config.d_input, config.d_compressed, bias=False)
        self.expand_proj = nn.Linear(config.d_compressed, config.d_input, bias=False)

        # Quantization settings
        self.quant_bits = config.quant_bits
        self.quant_per_channel = config.quant_per_channel
        self.quant_storage = getattr(config, "quant_storage", False)
        self.quant_backend = getattr(config, "quant_backend", "torch")

        # Initialize as approximate inverse pair
        nn.init.orthogonal_(self.compress_proj.weight)
        with torch.no_grad():
            # expand = compress.T makes expand a pseudoinverse
            self.expand_proj.weight.copy_(self.compress_proj.weight.T)

        # Mark as calibrated since no calibration needed
        self.calibrated = True

        # Move to correct device/dtype
        self.to(config.device)
        if config.dtype != torch.float32:
            self.compress_proj = self.compress_proj.to(config.dtype)
            self.expand_proj = self.expand_proj.to(config.dtype)

    def compress(self, x: torch.Tensor) -> Union[torch.Tensor, QuantizedTensor]:
        """Project to low-rank subspace, optionally with quantization.

        Returns:
            If quant_storage=True: QuantizedTensor (int8/int4 data + scale)
            Otherwise: torch.Tensor (float16/32, optionally fake-quantized)
        """
        latent = self.compress_proj(x)

        if self.quant_bits is not None:
            if self.quant_storage:
                # Real quantized storage - actual memory savings
                if self.quant_bits == 4:
                    return quantize_to_int4(latent, self.quant_per_channel)
                else:  # 8-bit
                    return quantize_to_int8(latent, self.quant_per_channel)
            else:
                # Fake quantization - simulation only
                latent = fake_quant(latent, self.quant_bits, self.quant_per_channel)

        return latent

    def expand(self, z: Union[torch.Tensor, QuantizedTensor]) -> torch.Tensor:
        """Reconstruct from low-rank representation.

        Args:
            z: Either float tensor or QuantizedTensor (auto-detected)

        Uses fused Triton kernel when quant_backend="triton" and input is
        QuantizedTensor, otherwise falls back to PyTorch dequant + matmul.
        """
        if isinstance(z, QuantizedTensor):
            if self.quant_backend == "triton":
                # Use fused Triton kernel (dequant + matmul in one pass)
                from gpt2.compression.triton_kernels import (
                    triton_expand_int4,
                    triton_expand_int8,
                )

                weight = self.expand_proj.weight.T  # [K, N] for matmul
                if z.bits == 4:
                    return triton_expand_int4(z.data, z.scale, weight, z.original_dim)
                else:  # int8
                    return triton_expand_int8(z.data, z.scale, weight)
            else:
                # PyTorch path: dequant then matmul
                z = z.dequantize()

        return self.expand_proj(z)

    def calibrate(self, samples: torch.Tensor) -> None:
        """
        Optional calibration: fit projections to data via SVD.

        This makes OrthogonalCompressor behave like SVDCompressor when
        calibration data is available, but still works without it.
        """
        samples = samples.float()

        # Fit via SVD like SVDCompressor
        _, _, Vh = torch.linalg.svd(samples, full_matrices=False)
        V = Vh[: self.d_compressed]

        # Update weights to match SVD basis
        with torch.no_grad():
            self.compress_proj.weight.copy_(V.to(self.config.dtype))
            self.expand_proj.weight.copy_(V.T.to(self.config.dtype))

        self.calibrated = True

    def extra_repr(self) -> str:
        base = super().extra_repr()
        if self.quant_bits is not None:
            mode = "int8" if self.quant_storage else "fake"
            return f"{base}, quant={self.quant_bits}bit({mode})"
        return base


# =============================================================================
# Hybrid Latent + Gate Compressor
# =============================================================================


class HybridCompressor(KVCompressor):
    """
    Two-stage compression: latent projection + energy gating.

    Stage 1: Linear projection to d_latent (MLA-style)
    Stage 2: TopK selection to d_compressed (EGG-style)

    Total compression: d_input -> d_latent -> d_compressed
    """

    def __init__(
        self,
        config: KVCompressorConfig,
        d_latent: int = 256,
    ):
        super().__init__(config)
        self.d_latent = d_latent

        # Stage 1: Linear compression
        self.to_latent = nn.Linear(config.d_input, d_latent, bias=False)
        self.from_latent = nn.Linear(d_latent, config.d_input, bias=False)

        # Stage 2: TopK gate on latent
        self.register_buffer(
            "keep_indices",
            torch.arange(config.d_compressed, dtype=torch.long),
        )
        self.register_buffer(
            "latent_energy",
            torch.ones(d_latent, dtype=config.dtype),
        )

        # Initialize projections
        nn.init.orthogonal_(self.to_latent.weight)
        nn.init.orthogonal_(self.from_latent.weight)

    def calibrate(self, samples: torch.Tensor) -> None:
        """
        Calibrate hybrid compressor:
        1. Fit latent projection via SVD
        2. Compute latent energy for gating
        """
        samples = samples.float()

        # Fit latent projection
        _, _, Vh = torch.linalg.svd(samples, full_matrices=False)
        to_latent = Vh[: self.d_latent]
        from_latent = to_latent.T

        self.to_latent.weight.data.copy_(to_latent.to(self.config.dtype))
        self.from_latent.weight.data.copy_(from_latent.to(self.config.dtype))

        # Compute latent representations
        with torch.no_grad():
            latent = samples @ to_latent.T

        # Compute per-channel energy in latent space
        energy = latent.var(dim=0)
        _, indices = energy.topk(self.d_compressed)
        indices = indices.sort().values

        self.latent_energy.copy_(energy.to(self.config.dtype))
        self.keep_indices.copy_(indices)
        self.calibrated = True

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Two-stage compression: project then gate."""
        # Stage 1: [.., d_input] -> [.., d_latent]
        latent = self.to_latent(x)
        # Stage 2: [.., d_latent] -> [.., d_compressed]
        return latent[..., self.keep_indices]

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """Two-stage expansion: ungate then project."""
        # Stage 2 inverse: scatter to latent
        shape = list(z.shape[:-1]) + [self.d_latent]
        latent = torch.zeros(shape, dtype=z.dtype, device=z.device)
        latent[..., self.keep_indices] = z
        # Stage 1 inverse: project to input
        return self.from_latent(latent)

    @property
    def compression_ratio(self) -> float:
        """Total compression ratio."""
        return self.d_input / self.d_compressed

    def extra_repr(self) -> str:
        return (
            f"d_input={self.d_input}, d_latent={self.d_latent}, "
            f"d_compressed={self.d_compressed}, ratio={self.compression_ratio:.1f}x"
        )


# =============================================================================
# Compressed KV Cache
# =============================================================================


class CompressedKVCache:
    """
    Manager for compressed KV cache storage.

    Stores compressed K and V tensors per layer, handles:
    - Append (next token generation)
    - Prefill (full sequence processing)
    - Retrieval with optional expansion
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d_compressed: int,
        max_seq_len: int = 8192,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_compressed = d_compressed
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # Cache storage: [layers][k/v] -> [batch, heads, seq, compressed]
        self._cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None for _ in range(n_layers)
        ]
        self._seq_len = 0

    def reset(self) -> None:
        """Clear all cached values."""
        self._cache = [None for _ in range(self.n_layers)]
        self._seq_len = 0

    def store(
        self,
        layer_idx: int,
        k_compressed: torch.Tensor,
        v_compressed: torch.Tensor,
    ) -> None:
        """
        Store compressed K, V for a layer.

        Args:
            layer_idx: Layer index
            k_compressed: [batch, heads, seq, d_compressed]
            v_compressed: [batch, heads, seq, d_compressed]
        """
        self._cache[layer_idx] = (k_compressed, v_compressed)
        self._seq_len = k_compressed.shape[2]

    def append(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new compressed K, V to existing cache.

        Args:
            layer_idx: Layer index
            k_new: [batch, heads, 1, d_compressed] new compressed K
            v_new: [batch, heads, 1, d_compressed] new compressed V

        Returns:
            Full K, V cache for this layer
        """
        if self._cache[layer_idx] is None:
            self._cache[layer_idx] = (k_new, v_new)
        else:
            k_old, v_old = self._cache[layer_idx]
            k_full = torch.cat([k_old, k_new], dim=2)
            v_full = torch.cat([v_old, v_new], dim=2)
            self._cache[layer_idx] = (k_full, v_full)

        self._seq_len = self._cache[layer_idx][0].shape[2]
        return self._cache[layer_idx]

    def get(
        self,
        layer_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get compressed K, V for a layer."""
        return self._cache[layer_idx]

    @property
    def seq_len(self) -> int:
        """Current sequence length in cache."""
        return self._seq_len

    def memory_bytes(self) -> int:
        """Total memory usage in bytes."""
        total = 0
        for layer_cache in self._cache:
            if layer_cache is not None:
                k, v = layer_cache
                total += k.numel() * k.element_size()
                total += v.numel() * v.element_size()
        return total


# =============================================================================
# Compressed Attention Wrapper
# =============================================================================


class CompressedAttentionWrapper(nn.Module):
    """
    Wraps a HuggingFace attention module to use compressed KV cache.

    Intercepts K, V projections, compresses before caching,
    expands before attention computation.
    """

    def __init__(
        self,
        attention: nn.Module,
        k_compressor: KVCompressor,
        v_compressor: KVCompressor,
        cache: CompressedKVCache,
        layer_idx: int,
        expand_mode: str = "pre",
    ):
        super().__init__()
        self.attention = attention
        self.k_compressor = k_compressor
        self.v_compressor = v_compressor
        self.cache = cache
        self.layer_idx = layer_idx
        self.expand_mode = expand_mode

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward with compressed KV cache.

        This is a generic wrapper that:
        1. Calls original attention to get K, V
        2. Compresses K, V
        3. Stores in compressed cache
        4. Expands for attention computation
        """
        # For now, use pre-expand mode: call original with full dims
        # The cache compression happens at store/retrieve boundaries

        # Get original output
        outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        # If using cache, compress and store
        if use_cache and len(outputs) > 1:
            # outputs typically: (hidden_states, present_key_value, [attentions])
            present_kv = outputs[1] if len(outputs) > 1 else None
            if present_kv is not None:
                k_full, v_full = present_kv

                # Compress K, V
                k_compressed = self.k_compressor.compress(k_full)
                v_compressed = self.v_compressor.compress(v_full)

                # Store in compressed cache
                self.cache.store(self.layer_idx, k_compressed, v_compressed)

                # Return with compressed cache
                outputs = (outputs[0], (k_compressed, v_compressed)) + outputs[2:]

        return outputs


# =============================================================================
# Main Plugin Class
# =============================================================================


class KVPlugin(nn.Module):
    """
    Universal KV-Cache Compression Plugin v3.0

    Wraps any HuggingFace model to use compressed KV cache.

    Supports:
    - GPT-2, GPT-Neo, GPT-J
    - LLaMA, LLaMA-2, LLaMA-3
    - Mistral, Mixtral
    - Qwen, Qwen2
    - Phi, Phi-2
    - Gemma

    Usage:
        plugin = KVPlugin.from_preset("balanced", model)
        plugin.calibrate(calibration_tokens)
        # Model now uses compressed cache
    """

    # Compressor registry
    COMPRESSORS: Dict[str, Type[KVCompressor]] = {
        "identity": IdentityCompressor,
        "pca": PCACompressor,
        "topk": TopKCompressor,
        "svd": SVDCompressor,
        "orthogonal": OrthogonalCompressor,
        "hybrid": HybridCompressor,
    }

    # Presets from v2.5
    PRESETS: Dict[str, Dict[str, Any]] = {
        "none": {
            "compressor_type": "identity",
            "description": "No compression (baseline)",
        },
        "conservative": {
            "compressor_type": "pca",
            "d_compressed": 48,
            "description": "6x compression, ~0% quality loss",
        },
        "balanced": {
            "compressor_type": "hybrid",
            "d_latent": 256,
            "d_compressed": 128,
            "description": "12x compression, ~2-5% quality loss",
        },
        "aggressive": {
            "compressor_type": "hybrid",
            "d_latent": 256,
            "d_compressed": 64,
            "description": "18x compression, ~5-10% quality loss",
        },
        "extreme": {
            "compressor_type": "hybrid",
            "d_latent": 256,
            "d_compressed": 32,
            "description": "24x compression, ~10-20% quality loss",
        },
        "laptop": {
            "compressor_type": "hybrid",
            "d_latent": 128,
            "d_compressed": 64,
            "description": "24x compression for memory-constrained devices",
        },
        "orthogonal": {
            "compressor_type": "orthogonal",
            "d_compressed": 128,
            "description": "12x compression, no calibration needed",
        },
        "orthogonal_aggressive": {
            "compressor_type": "orthogonal",
            "d_compressed": 64,
            "description": "18x compression, no calibration needed",
        },
        "orthogonal_q8": {
            "compressor_type": "orthogonal",
            "d_compressed": 128,
            "quant_bits": 8,
            "quant_target": "v",
            "quant_per_channel": True,
            "description": "12x compression + 8-bit V quantization",
        },
        "orthogonal_q4": {
            "compressor_type": "orthogonal",
            "d_compressed": 128,
            "quant_bits": 4,
            "quant_target": "v",
            "quant_per_channel": True,
            "description": "12x compression + 4-bit V quantization",
        },
        "orthogonal_q8_kv": {
            "compressor_type": "orthogonal",
            "d_compressed": 128,
            "quant_bits": 8,
            "quant_target": "kv",
            "quant_per_channel": True,
            "description": "12x compression + 8-bit K/V quantization",
        },
        "orthogonal_q4_kv": {
            "compressor_type": "orthogonal",
            "d_compressed": 128,
            "quant_bits": 4,
            "quant_target": "kv",
            "quant_per_channel": True,
            "description": "12x compression + 4-bit K/V quantization",
        },
        # Real int8 storage presets (actual memory savings)
        "orthogonal_int8": {
            "compressor_type": "orthogonal",
            "d_compressed": 128,
            "quant_bits": 8,
            "quant_target": "v",
            "quant_per_channel": True,
            "quant_storage": True,
            "description": "12x compression + real int8 V storage",
        },
        "orthogonal_int8_kv": {
            "compressor_type": "orthogonal",
            "d_compressed": 128,
            "quant_bits": 8,
            "quant_target": "kv",
            "quant_per_channel": True,
            "quant_storage": True,
            "description": "12x compression + real int8 K/V storage",
        },
        # Real int4 storage presets (packed, 4x memory savings)
        "orthogonal_int4": {
            "compressor_type": "orthogonal",
            "d_compressed": 128,
            "quant_bits": 4,
            "quant_target": "v",
            "quant_per_channel": True,
            "quant_storage": True,
            "description": "12x compression + real int4 V storage (24x total)",
        },
        "orthogonal_int4_kv": {
            "compressor_type": "orthogonal",
            "d_compressed": 128,
            "quant_bits": 4,
            "quant_target": "kv",
            "quant_per_channel": True,
            "quant_storage": True,
            "description": "12x compression + real int4 K/V storage (24x total)",
        },
        "cloud_single": {
            "compressor_type": "pca",
            "d_compressed": 48,
            "description": "6x compression, optimized for single GPU",
        },
        "batch_throughput": {
            "compressor_type": "hybrid",
            "d_latent": 256,
            "d_compressed": 128,
            "description": "12x compression for batched inference",
        },
    }

    def __init__(
        self,
        model: nn.Module,
        config: KVPluginConfig,
    ):
        super().__init__()
        self.model = model
        self.config = config

        # Detect model architecture
        self._detect_architecture()

        # Create compressors for each layer
        self.k_compressors = nn.ModuleList()
        self.v_compressors = nn.ModuleList()
        self._create_compressors()

        # Create compressed cache
        self.cache = CompressedKVCache(
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_compressed=config.d_compressed,
            device=config.device,
            dtype=config.dtype,
        )

        # Track patching state
        self._patched = False

    def _detect_architecture(self) -> None:
        """Auto-detect model architecture from HF model."""
        model_config = getattr(self.model, "config", None)

        if model_config is not None:
            # Common HF config attributes
            self.config.n_layers = getattr(
                model_config,
                "num_hidden_layers",
                getattr(model_config, "n_layer", self.config.n_layers),
            )
            self.config.n_heads = getattr(
                model_config,
                "num_attention_heads",
                getattr(model_config, "n_head", self.config.n_heads),
            )
            self.config.d_model = getattr(
                model_config,
                "hidden_size",
                getattr(model_config, "n_embd", self.config.d_model),
            )
            self.config.head_dim = self.config.d_model // self.config.n_heads

    def _create_compressors(self) -> None:
        """Create K and V compressors for each layer."""
        compressor_cls = self.COMPRESSORS.get(
            self.config.compressor_type, HybridCompressor
        )

        for layer_idx in range(self.config.n_layers):
            # Check for per-layer override
            layer_cfg = {}
            if self.config.layer_configs and layer_idx in self.config.layer_configs:
                layer_cfg = self.config.layer_configs[layer_idx]

            # Create config for this layer
            d_compressed = layer_cfg.get("d_compressed", self.config.d_compressed)
            d_input = self.config.n_heads * self.config.head_dim

            # Determine quantization for K and V based on quant_target
            # "v" = only V, "kv" = both K and V
            quant_target = self.config.quant_target
            k_quant_bits = self.config.quant_bits if quant_target == "kv" else None
            v_quant_bits = self.config.quant_bits  # V always gets quant if set

            k_config = KVCompressorConfig(
                d_input=d_input,
                d_compressed=d_compressed,
                n_heads=self.config.n_heads,
                per_head=True,
                device=self.config.device,
                dtype=self.config.dtype,
                quant_bits=k_quant_bits,
                quant_per_channel=self.config.quant_per_channel,
                quant_storage=self.config.quant_storage,
            )
            v_config = KVCompressorConfig(
                d_input=d_input,
                d_compressed=d_compressed,
                n_heads=self.config.n_heads,
                per_head=True,
                device=self.config.device,
                dtype=self.config.dtype,
                quant_bits=v_quant_bits,
                quant_per_channel=self.config.quant_per_channel,
                quant_storage=self.config.quant_storage,
            )

            # Create compressors
            if compressor_cls == HybridCompressor:
                d_latent = layer_cfg.get("d_latent", self.config.d_latent)
                k_comp = compressor_cls(k_config, d_latent=d_latent)
                v_comp = compressor_cls(v_config, d_latent=d_latent)
            else:
                k_comp = compressor_cls(k_config)
                v_comp = compressor_cls(v_config)

            self.k_compressors.append(k_comp)
            self.v_compressors.append(v_comp)

    def calibrate(
        self,
        calibration_data: Union[torch.Tensor, List[torch.Tensor]],
        max_samples: int = 10000,
    ) -> None:
        """
        Calibrate all compressors from data.

        Args:
            calibration_data: Tensor of token IDs or list of tensors
            max_samples: Maximum calibration samples per layer
        """
        # Collect K, V activations
        k_samples = [[] for _ in range(self.config.n_layers)]
        v_samples = [[] for _ in range(self.config.n_layers)]

        hooks = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                # Try to extract K, V from output
                if isinstance(output, tuple) and len(output) >= 2:
                    # Assume output[1] is (K, V) cache
                    kv = output[1]
                    if kv is not None and isinstance(kv, tuple):
                        k, v = kv
                        # Flatten: [B, H, T, D] -> [B*T, H*D]
                        k_flat = k.permute(0, 2, 1, 3).reshape(
                            -1, k.shape[1] * k.shape[3]
                        )
                        v_flat = v.permute(0, 2, 1, 3).reshape(
                            -1, v.shape[1] * v.shape[3]
                        )
                        k_samples[layer_idx].append(k_flat.detach().cpu())
                        v_samples[layer_idx].append(v_flat.detach().cpu())

            return hook

        # Register hooks on attention layers
        attention_layers = self._get_attention_layers()
        for layer_idx, attn in enumerate(attention_layers):
            hook = attn.register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)

        # Run forward passes
        self.model.eval()
        with torch.no_grad():
            if isinstance(calibration_data, list):
                for batch in calibration_data:
                    batch = batch.to(self.config.device)
                    self.model(batch, use_cache=True)
            else:
                self.model(calibration_data.to(self.config.device), use_cache=True)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Calibrate each compressor
        for layer_idx in range(self.config.n_layers):
            if k_samples[layer_idx]:
                k_data = torch.cat(k_samples[layer_idx], dim=0)[:max_samples]
                v_data = torch.cat(v_samples[layer_idx], dim=0)[:max_samples]

                self.k_compressors[layer_idx].calibrate(k_data)
                self.v_compressors[layer_idx].calibrate(v_data)

    def _get_attention_layers(self) -> List[nn.Module]:
        """Get list of attention modules from model."""
        attention_layers = []

        # Try common HF patterns
        if hasattr(self.model, "transformer"):
            # GPT-2 style
            blocks = getattr(self.model.transformer, "h", [])
            for block in blocks:
                if hasattr(block, "attn"):
                    attention_layers.append(block.attn)
        elif hasattr(self.model, "model"):
            # LLaMA/Mistral style
            layers = getattr(self.model.model, "layers", [])
            for layer in layers:
                if hasattr(layer, "self_attn"):
                    attention_layers.append(layer.self_attn)
        elif hasattr(self.model, "gpt_neox"):
            # GPT-NeoX style
            layers = getattr(self.model.gpt_neox, "layers", [])
            for layer in layers:
                if hasattr(layer, "attention"):
                    attention_layers.append(layer.attention)

        return attention_layers

    def patch_model(self) -> None:
        """Patch model to use compressed attention."""
        if self._patched:
            return

        attention_layers = self._get_attention_layers()

        for layer_idx, attn in enumerate(attention_layers):
            wrapper = CompressedAttentionWrapper(
                attention=attn,
                k_compressor=self.k_compressors[layer_idx],
                v_compressor=self.v_compressors[layer_idx],
                cache=self.cache,
                layer_idx=layer_idx,
                expand_mode=self.config.expand_mode,
            )

            # Replace in parent module
            self._replace_attention(layer_idx, wrapper)

        self._patched = True

    def _replace_attention(
        self,
        layer_idx: int,
        wrapper: CompressedAttentionWrapper,
    ) -> None:
        """Replace attention module with wrapper."""
        if hasattr(self.model, "transformer"):
            self.model.transformer.h[layer_idx].attn = wrapper
        elif hasattr(self.model, "model"):
            self.model.model.layers[layer_idx].self_attn = wrapper
        elif hasattr(self.model, "gpt_neox"):
            self.model.gpt_neox.layers[layer_idx].attention = wrapper

    def reset_cache(self) -> None:
        """Clear the compressed KV cache."""
        self.cache.reset()

    @classmethod
    def from_preset(
        cls,
        preset: str,
        model: nn.Module,
        **overrides,
    ) -> "KVPlugin":
        """
        Create plugin from a named preset.

        Args:
            preset: Preset name (none, conservative, balanced, aggressive, extreme)
            model: HuggingFace model to wrap
            **overrides: Override specific config values

        Returns:
            Configured KVPlugin instance
        """
        if preset not in cls.PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Available: {list(cls.PRESETS.keys())}"
            )

        preset_cfg = cls.PRESETS[preset].copy()
        preset_cfg.pop("description", None)
        preset_cfg.update(overrides)

        config = KVPluginConfig(**preset_cfg)
        return cls(model, config)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics for the compressed cache."""
        return {
            "cache_bytes": self.cache.memory_bytes(),
            "cache_mb": self.cache.memory_bytes() / (1024 * 1024),
            "seq_len": self.cache.seq_len,
            "compression_ratio": self.config.d_model
            * self.config.n_heads
            / self.config.d_compressed,
        }

    def forward(self, *args, **kwargs):
        """Forward through wrapped model."""
        return self.model(*args, **kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_compressor(
    compressor_type: str,
    d_input: int,
    d_compressed: int,
    d_latent: int = 256,
    **kwargs,
) -> KVCompressor:
    """
    Factory function to create a compressor.

    Args:
        compressor_type: One of 'identity', 'pca', 'topk', 'svd', 'hybrid'
        d_input: Input dimension
        d_compressed: Compressed dimension
        d_latent: Latent dimension (for hybrid)

    Returns:
        KVCompressor instance
    """
    config = KVCompressorConfig(
        d_input=d_input,
        d_compressed=d_compressed,
        **kwargs,
    )

    cls = KVPlugin.COMPRESSORS.get(compressor_type)
    if cls is None:
        raise ValueError(f"Unknown compressor type: {compressor_type}")

    if cls == HybridCompressor:
        return cls(config, d_latent=d_latent)
    return cls(config)


def wrap_model(
    model: nn.Module,
    preset: str = "balanced",
    **kwargs,
) -> KVPlugin:
    """
    Convenience function to wrap a model with KV compression.

    Args:
        model: HuggingFace model
        preset: Preset name
        **kwargs: Additional config overrides

    Returns:
        KVPlugin wrapping the model
    """
    plugin = KVPlugin.from_preset(preset, model, **kwargs)
    return plugin
