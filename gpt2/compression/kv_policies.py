"""
Policy layer for KV compression presets.

This module answers questions like:
  - "Given this model, VRAM budget, and context, which KV preset should I use?"
  - "Should I enable LN nullspace or not?"
  - "Should I use gamma-aware quantization?"

It does NOT:
  - Implement the compressors themselves
  - Implement calibration
  - Know details of Triton kernels

Instead, it:
  - Describes a small set of high-level policies
  - Maps (model_id, policy) -> preset name / config
  - Provides a simple heuristic for choosing a policy given VRAM + context

Intended usage:

    from gpt2.compression.kv_policies import (
        KVPolicyName,
        choose_policy_for_run,
        get_preset_for_policy,
    )

    policy = choose_policy_for_run(
        model_config=model.config,
        model_id="Qwen/Qwen2.5-7B-Instruct",
        vram_gb=80.0,
        batch_size=4,
        max_context=8192,
        ppl_budget=0.015,        # +1.5% PPL
    )

    preset_name = get_preset_for_policy(model_id, policy)
    cache = load_preset_cache(model, preset_name, policy=policy)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Policy definitions
# ---------------------------------------------------------------------------


class KVPolicyName(str, Enum):
    DEFAULT = "default"
    HIGH_COMPRESSION = "high_compression"
    MEMORY_EMERGENCY = "memory_emergency"


@dataclass
class KVPolicy:
    """High-level semantic info about a KV-compression operating mode."""

    name: KVPolicyName

    # Maximum acceptable perplexity increase vs baseline
    # (e.g. 0.01 = +1.0% PPL).
    ppl_budget: float

    # Maximum acceptable generation latency overhead vs baseline
    # as a fraction (e.g. 0.35 = +35% slower).
    latency_overhead_budget: float

    # Whether to prefer gamma-aware quantization when available.
    use_gamma_aware: bool = True

    # Whether to allow LN-nullspace compression on V.
    use_ln_nullspace_v: bool = False

    # Whether to allow LN-nullspace compression on K as well.
    #  - This is more sensitive and may add noticeable compute overhead.
    use_ln_nullspace_k: bool = False

    # Runtime-aware compression (v20):
    # Don't compress until seq_len >= this threshold.
    # Below this, behave as identity cache (no overhead for short contexts).
    compress_start_len: int = 0

    # Keep the last N tokens uncompressed (FP16) for fast access.
    # Only compress tokens older than this sliding window.
    uncompressed_tail: int = 0

    # Short human-readable description for docs / logs.
    description: str = ""


# Baseline policies: tune these numbers based on v14-v20 results.
DEFAULT_POLICY = KVPolicy(
    name=KVPolicyName.DEFAULT,
    ppl_budget=0.015,  # +1.5% PPL
    latency_overhead_budget=0.35,  # up to +35% OK
    use_gamma_aware=True,
    use_ln_nullspace_v=False,
    use_ln_nullspace_k=False,
    compress_start_len=512,  # v20: no overhead for short contexts
    uncompressed_tail=256,  # v20: fast access to recent tokens
    description=(
        "Default production policy: V-only low-rank + int8, gamma-aware enabled, "
        "no LN-nullspace. Runtime-aware compression (v20) reduces overhead for "
        "short contexts and keeps recent tokens uncompressed for fast access."
    ),
)

HIGH_COMPRESSION_POLICY = KVPolicy(
    name=KVPolicyName.HIGH_COMPRESSION,
    ppl_budget=0.03,  # +3% PPL
    latency_overhead_budget=0.4,  # up to +40% overhead OK
    use_gamma_aware=True,
    use_ln_nullspace_v=False,
    use_ln_nullspace_k=False,
    compress_start_len=256,  # v20: start compressing earlier
    uncompressed_tail=128,  # v20: smaller uncompressed window
    description=(
        "High-compression policy: more aggressive rank/bits, still gamma-aware. "
        "LN-nullspace left off by default. Smaller uncompressed window for "
        "maximum memory savings at longer contexts."
    ),
)

MEMORY_EMERGENCY_POLICY = KVPolicy(
    name=KVPolicyName.MEMORY_EMERGENCY,
    ppl_budget=0.02,  # keep quality decent, but accept slower runtime
    latency_overhead_budget=0.6,  # up to +60% slower is acceptable
    use_gamma_aware=True,
    use_ln_nullspace_v=True,  # turn on V-only LN nullspace
    use_ln_nullspace_k=False,  # K-side stays off unless you really want to suffer
    compress_start_len=0,  # v20: always compress from the start
    uncompressed_tail=128,  # v20: minimal uncompressed window
    description=(
        "Memory-emergency policy: target fitting VRAM over speed. "
        "Enables V-only LN-nullspace for extra ~0.8% KV savings. "
        "Always compresses from start. Use when you would otherwise OOM."
    ),
)


# ---------------------------------------------------------------------------
#  Preset registry
# ---------------------------------------------------------------------------


@dataclass
class KVPresetInfo:
    """Metadata for a preset used by a given policy and model."""

    name: str
    compression_factor: float  # approx KV memory reduction vs full FP16
    policy_name: KVPolicyName


# Preset registry: maps (model_id, policy) -> preset info
# Based on v9-v17 results
PRESET_REGISTRY: Dict[Tuple[str, KVPolicyName], KVPresetInfo] = {
    # Qwen2.5-7B
    ("Qwen/Qwen2.5-7B", KVPolicyName.DEFAULT): KVPresetInfo(
        name="kv_preset_qwen-qwen2.5-7b_v18_default",
        compression_factor=2.67,
        policy_name=KVPolicyName.DEFAULT,
    ),
    ("Qwen/Qwen2.5-7B", KVPolicyName.HIGH_COMPRESSION): KVPresetInfo(
        name="kv_preset_qwen-qwen2.5-7b_v18_highcomp",
        compression_factor=3.2,
        policy_name=KVPolicyName.HIGH_COMPRESSION,
    ),
    ("Qwen/Qwen2.5-7B", KVPolicyName.MEMORY_EMERGENCY): KVPresetInfo(
        name="kv_preset_qwen-qwen2.5-7b_v18_mem_emerg",
        compression_factor=2.7,  # same order, but with LN-nullspace enabled
        policy_name=KVPolicyName.MEMORY_EMERGENCY,
    ),
    # Qwen2.5-7B-Instruct (alias)
    ("Qwen/Qwen2.5-7B-Instruct", KVPolicyName.DEFAULT): KVPresetInfo(
        name="kv_preset_qwen-qwen2.5-7b_v18_default",
        compression_factor=2.67,
        policy_name=KVPolicyName.DEFAULT,
    ),
    ("Qwen/Qwen2.5-7B-Instruct", KVPolicyName.HIGH_COMPRESSION): KVPresetInfo(
        name="kv_preset_qwen-qwen2.5-7b_v18_highcomp",
        compression_factor=3.2,
        policy_name=KVPolicyName.HIGH_COMPRESSION,
    ),
    ("Qwen/Qwen2.5-7B-Instruct", KVPolicyName.MEMORY_EMERGENCY): KVPresetInfo(
        name="kv_preset_qwen-qwen2.5-7b_v18_mem_emerg",
        compression_factor=2.7,
        policy_name=KVPolicyName.MEMORY_EMERGENCY,
    ),
    # Mistral-7B
    ("mistralai/Mistral-7B-v0.1", KVPolicyName.DEFAULT): KVPresetInfo(
        name="kv_preset_mistralai-mistral-7b_v18_default",
        compression_factor=2.42,
        policy_name=KVPolicyName.DEFAULT,
    ),
    ("mistralai/Mistral-7B-v0.1", KVPolicyName.HIGH_COMPRESSION): KVPresetInfo(
        name="kv_preset_mistralai-mistral-7b_v18_highcomp",
        compression_factor=3.0,
        policy_name=KVPolicyName.HIGH_COMPRESSION,
    ),
    ("mistralai/Mistral-7B-v0.1", KVPolicyName.MEMORY_EMERGENCY): KVPresetInfo(
        name="kv_preset_mistralai-mistral-7b_v18_mem_emerg",
        compression_factor=2.5,
        policy_name=KVPolicyName.MEMORY_EMERGENCY,
    ),
    # Qwen2.5-0.5B
    ("Qwen/Qwen2.5-0.5B", KVPolicyName.DEFAULT): KVPresetInfo(
        name="kv_preset_qwen-qwen2.5-0.5b_v18_default",
        compression_factor=2.29,
        policy_name=KVPolicyName.DEFAULT,
    ),
    ("Qwen/Qwen2.5-0.5B", KVPolicyName.HIGH_COMPRESSION): KVPresetInfo(
        name="kv_preset_qwen-qwen2.5-0.5b_v18_highcomp",
        compression_factor=3.0,
        policy_name=KVPolicyName.HIGH_COMPRESSION,
    ),
    ("Qwen/Qwen2.5-0.5B", KVPolicyName.MEMORY_EMERGENCY): KVPresetInfo(
        name="kv_preset_qwen-qwen2.5-0.5b_v18_mem_emerg",
        compression_factor=2.3,
        policy_name=KVPolicyName.MEMORY_EMERGENCY,
    ),
    # Qwen2-1.5B
    ("Qwen/Qwen2-1.5B", KVPolicyName.DEFAULT): KVPresetInfo(
        name="kv_preset_qwen-qwen2-1.5b_v18_default",
        compression_factor=2.67,
        policy_name=KVPolicyName.DEFAULT,
    ),
    ("Qwen/Qwen2-1.5B", KVPolicyName.HIGH_COMPRESSION): KVPresetInfo(
        name="kv_preset_qwen-qwen2-1.5b_v18_highcomp",
        compression_factor=3.0,
        policy_name=KVPolicyName.HIGH_COMPRESSION,
    ),
    ("Qwen/Qwen2-1.5B", KVPolicyName.MEMORY_EMERGENCY): KVPresetInfo(
        name="kv_preset_qwen-qwen2-1.5b_v18_mem_emerg",
        compression_factor=2.7,
        policy_name=KVPolicyName.MEMORY_EMERGENCY,
    ),
}


def get_policy(policy_name: KVPolicyName) -> KVPolicy:
    """Return a KVPolicy object by name."""
    if policy_name == KVPolicyName.DEFAULT:
        return DEFAULT_POLICY
    if policy_name == KVPolicyName.HIGH_COMPRESSION:
        return HIGH_COMPRESSION_POLICY
    if policy_name == KVPolicyName.MEMORY_EMERGENCY:
        return MEMORY_EMERGENCY_POLICY
    raise ValueError(f"Unknown KVPolicyName: {policy_name}")


def get_preset_for_policy(
    model_id: str, policy_name: KVPolicyName
) -> Optional[KVPresetInfo]:
    """
    Look up a preset for a given model and policy.

    Returns None if no specific preset has been registered.
    """
    key = (model_id, policy_name)
    preset = PRESET_REGISTRY.get(key)
    if preset is None:
        logger.warning(
            "No preset found for model_id=%s, policy=%s. "
            "You may want to add an entry to PRESET_REGISTRY.",
            model_id,
            policy_name,
        )
    return preset


# ---------------------------------------------------------------------------
#  LN Nullspace warning
# ---------------------------------------------------------------------------


def warn_ln_nullspace_overhead():
    """Emit a warning about LN nullspace compute overhead."""
    warnings.warn(
        "LN nullspace reduces FLOPs but increases wall-clock latency by ~7% "
        "due to extra compression overhead. Intended for memory-constrained cases. "
        "If latency is critical, consider disabling LN nullspace.",
        UserWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
#  Heuristics: estimate KV memory & choose a policy
# ---------------------------------------------------------------------------


def estimate_kv_bytes_fp16(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    batch_size: int,
    max_context: int,
    dtype_bytes: int = 2,
) -> int:
    """
    Rough estimate of KV cache size in bytes for *uncompressed* FP16 KV.

    Formula:
      KV size = num_layers * 2 (K+V) * batch * num_heads * max_context * head_dim * bytes

    This is deliberately simple and pessimistic.
    """
    return (
        num_layers * 2 * batch_size * num_heads * max_context * head_dim * dtype_bytes
    )


def estimate_kv_bytes_compressed(
    base_bytes_fp16: int,
    compression_factor: float,
) -> int:
    """Apply a simple multiplicative compression factor to FP16 KV size."""
    if compression_factor <= 0:
        return base_bytes_fp16
    return int(base_bytes_fp16 / compression_factor)


def choose_policy_for_run(
    model_config,
    model_id: str,
    vram_gb: float,
    batch_size: int,
    max_context: int,
    ppl_budget: float = 0.015,
) -> KVPolicyName:
    """
    Heuristic to choose a policy for a given run.

    Inputs:
      - model_config: HF config-like object with num_hidden_layers, num_attention_heads, hidden_size
      - model_id: HuggingFace model id (for preset lookup)
      - vram_gb: available VRAM for model + KV, in GiB
      - batch_size: batch size during generation
      - max_context: maximum context length you intend to use
      - ppl_budget: max acceptable PPL increase (fraction, e.g. 0.015 for +1.5%)

    Strategy (simple first pass):
      1. Try DEFAULT preset:
         - estimate compressed KV usage; if it fits comfortably in VRAM, and ppl_budget >= default.ppl_budget, use DEFAULT.
      2. Otherwise try HIGH_COMPRESSION:
         - if it fits and ppl_budget >= highcomp.ppl_budget, use HIGH_COMPRESSION.
      3. Otherwise, fall back to MEMORY_EMERGENCY.
    """
    num_layers = getattr(model_config, "num_hidden_layers", None)
    num_heads = getattr(model_config, "num_attention_heads", None)
    hidden_size = getattr(model_config, "hidden_size", None)

    if None in (num_layers, num_heads, hidden_size):
        logger.warning(
            "Model config missing num_hidden_layers / num_attention_heads / hidden_size. "
            "Falling back to DEFAULT policy."
        )
        return KVPolicyName.DEFAULT

    head_dim = hidden_size // num_heads
    base_kv_bytes = estimate_kv_bytes_fp16(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        batch_size=batch_size,
        max_context=max_context,
        dtype_bytes=2,  # fp16
    )

    vram_bytes = int(vram_gb * (1024**3))

    def fits_with_preset(preset: KVPresetInfo) -> bool:
        compressed_bytes = estimate_kv_bytes_compressed(
            base_kv_bytes, preset.compression_factor
        )
        # Leave some slack for model weights, activations, optimizer states (if any).
        # Very rough: require KV <= 25% of total VRAM.
        return compressed_bytes <= int(0.25 * vram_bytes)

    # 1) Try DEFAULT
    default_policy = DEFAULT_POLICY
    default_preset = get_preset_for_policy(model_id, KVPolicyName.DEFAULT)
    if (
        default_preset is not None
        and ppl_budget >= default_policy.ppl_budget
        and fits_with_preset(default_preset)
    ):
        logger.info(
            "Choosing DEFAULT KV policy for model %s. "
            "Estimated KV=%.2f GiB, VRAM=%.2f GiB.",
            model_id,
            estimate_kv_bytes_compressed(
                base_kv_bytes, default_preset.compression_factor
            )
            / (1024**3),
            vram_gb,
        )
        return KVPolicyName.DEFAULT

    # 2) Try HIGH_COMPRESSION
    high_policy = HIGH_COMPRESSION_POLICY
    high_preset = get_preset_for_policy(model_id, KVPolicyName.HIGH_COMPRESSION)
    if (
        high_preset is not None
        and ppl_budget >= high_policy.ppl_budget
        and fits_with_preset(high_preset)
    ):
        logger.info(
            "Choosing HIGH_COMPRESSION KV policy for model %s. "
            "Estimated KV=%.2f GiB, VRAM=%.2f GiB.",
            model_id,
            estimate_kv_bytes_compressed(base_kv_bytes, high_preset.compression_factor)
            / (1024**3),
            vram_gb,
        )
        return KVPolicyName.HIGH_COMPRESSION

    # 3) Fallback: MEMORY_EMERGENCY
    mem_preset = get_preset_for_policy(model_id, KVPolicyName.MEMORY_EMERGENCY)
    if mem_preset is not None:
        logger.warning(
            "Falling back to MEMORY_EMERGENCY KV policy for model %s. "
            "Estimated KV=%.2f GiB with mem_emerg preset, VRAM=%.2f GiB.",
            model_id,
            estimate_kv_bytes_compressed(base_kv_bytes, mem_preset.compression_factor)
            / (1024**3),
            vram_gb,
        )
        return KVPolicyName.MEMORY_EMERGENCY

    # If all else fails, log and return DEFAULT.
    logger.warning(
        "No suitable preset found for model %s; using DEFAULT KV policy as fallback.",
        model_id,
    )
    return KVPolicyName.DEFAULT


# ---------------------------------------------------------------------------
#  Compressor builder
# ---------------------------------------------------------------------------


def build_compressor_config_for_policy(
    model_config,
    model_id: str,
    policy_name: KVPolicyName,
) -> Dict:
    """
    Build a compressor configuration dict for a given model and policy.

    Returns a dict with:
        - preset_name: name of the preset to load
        - use_gamma_aware: whether to use gamma-aware quantization
        - use_ln_nullspace_v: whether to use V-only LN nullspace
        - use_ln_nullspace_k: whether to use K-only LN nullspace
        - compress_start_len: threshold to start compression (v20)
        - uncompressed_tail: keep N recent tokens uncompressed (v20)
        - compression_factor: expected compression ratio

    This config can be passed to the cache loader.
    """
    policy = get_policy(policy_name)
    preset = get_preset_for_policy(model_id, policy_name)

    if preset is None:
        # No preset found, return a default config
        return {
            "preset_name": None,
            "use_gamma_aware": policy.use_gamma_aware,
            "use_ln_nullspace_v": policy.use_ln_nullspace_v,
            "use_ln_nullspace_k": policy.use_ln_nullspace_k,
            "compress_start_len": policy.compress_start_len,
            "uncompressed_tail": policy.uncompressed_tail,
            "compression_factor": 1.0,
            "policy": policy_name.value,
        }

    # Warn if LN nullspace is enabled
    if policy.use_ln_nullspace_v or policy.use_ln_nullspace_k:
        warn_ln_nullspace_overhead()

    return {
        "preset_name": preset.name,
        "use_gamma_aware": policy.use_gamma_aware,
        "use_ln_nullspace_v": policy.use_ln_nullspace_v,
        "use_ln_nullspace_k": policy.use_ln_nullspace_k,
        "compress_start_len": policy.compress_start_len,
        "uncompressed_tail": policy.uncompressed_tail,
        "compression_factor": preset.compression_factor,
        "policy": policy_name.value,
    }
