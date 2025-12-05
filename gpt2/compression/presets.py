"""
KV Compression Presets.

Provides easy-to-use presets for KV cache compression.

Usage:
    from gpt2.compression.presets import load_preset, apply_compression_preset

    # Load and apply a preset
    model = apply_compression_preset(model, "qwen-0.5b-v9-vonly-int8")

    # Or load preset info
    preset = load_preset("qwen-0.5b-v9-vonly-int8")
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    load_calibrated_compressors,
)


# Default preset directory
PRESET_DIR = Path(__file__).parent.parent.parent / "presets"

# Built-in presets (model -> preset name -> config)
BUILTIN_PRESETS = {
    "Qwen/Qwen2.5-0.5B": {
        "v9-vonly-int8": {
            "rank": 56,
            "target": "v",
            "bits": 8,
            "calibration_file": "kv_lowrank_calib_qwen-qwen2.5-0.5b_r56.pt",
            "compression": 2.29,
            "ppl_delta": 0.0406,
        },
        "v9-conservative": {
            "rank": 60,
            "target": "v",
            "bits": 16,
            "calibration_file": "kv_lowrank_calib_qwen-qwen2.5-0.5b_r60.pt",
            "compression": 1.07,
            "ppl_delta": -0.0115,
        },
    },
    "Qwen/Qwen2.5-7B": {
        "v9-vonly-int8": {
            "rank": 96,
            "target": "v",
            "bits": 8,
            "calibration_file": "kv_lowrank_calib_qwen-qwen2.5-7b_r96.pt",
            "compression": 2.67,
            "ppl_delta": 0.0099,
        },
        "v9-aggressive": {
            "rank": 80,
            "target": "v",
            "bits": 8,
            "calibration_file": "kv_lowrank_calib_qwen-qwen2.5-7b_r80.pt",
            "compression": 3.2,
            "ppl_delta": 0.0471,
        },
    },
}


def list_presets(model_name: str = None) -> Dict:
    """
    List available presets.

    Args:
        model_name: Filter by model (optional)

    Returns:
        Dict of available presets
    """
    if model_name:
        return BUILTIN_PRESETS.get(model_name, {})
    return BUILTIN_PRESETS


def get_preset_info(model_name: str, preset_name: str) -> Optional[Dict]:
    """
    Get preset configuration.

    Args:
        model_name: Model name (e.g., "Qwen/Qwen2.5-0.5B")
        preset_name: Preset name (e.g., "v9-vonly-int8")

    Returns:
        Preset config dict or None if not found
    """
    model_presets = BUILTIN_PRESETS.get(model_name, {})
    return model_presets.get(preset_name)


def load_preset_from_file(preset_path: str) -> Dict:
    """Load preset from JSON file."""
    with open(preset_path) as f:
        return json.load(f)


def create_compressors_from_preset(
    preset: Dict,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[List[nn.Module], List[nn.Module], int]:
    """
    Create compressors from a preset configuration.

    Args:
        preset: Preset config dict
        device: Device to use
        dtype: Data type

    Returns:
        (k_compressors, v_compressors, num_layers)
    """
    calib_path = preset["calibration_file"]
    target = preset.get("target", "v")
    bits = preset.get("bits", 16)
    quantize_bits = bits if bits < 16 else None

    # Load calibrated compressors
    k_comp, v_comp, metadata = load_calibrated_compressors(
        calib_path,
        device=torch.device(device),
        dtype=dtype,
        quantize_bits=quantize_bits,
    )

    num_layers = metadata["n_layers"]

    # Apply target filter
    if target == "v":
        k_comp = [IdentityCompressor() for _ in range(num_layers)]
    elif target == "k":
        v_comp = [IdentityCompressor() for _ in range(num_layers)]

    return k_comp, v_comp, num_layers


def create_cache_from_preset(
    preset: Dict,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> CompressedDynamicCache:
    """
    Create a CompressedDynamicCache from a preset.

    Args:
        preset: Preset config dict
        device: Device to use
        dtype: Data type

    Returns:
        CompressedDynamicCache instance
    """
    k_comp, v_comp, num_layers = create_compressors_from_preset(preset, device, dtype)
    return CompressedDynamicCache(k_comp, v_comp, num_layers)


def apply_preset_to_generate(
    model,
    preset: Dict,
    device: str = "cuda",
) -> "CompressedDynamicCache":
    """
    Create a cache from preset for use with model.generate().

    Example:
        cache = apply_preset_to_generate(model, preset)
        outputs = model.generate(input_ids, past_key_values=cache, use_cache=True)

    Args:
        model: The model (used to get num_layers)
        preset: Preset config dict
        device: Device

    Returns:
        CompressedDynamicCache ready for generation
    """
    return create_cache_from_preset(preset, device)


def print_preset_info(preset: Dict):
    """Print preset information."""
    print(f"KV Compression Preset")
    print(f"  Model: {preset.get('model', 'Unknown')}")
    print(f"  Rank: {preset.get('rank', 'Unknown')}")
    print(f"  Target: {preset.get('target', 'Unknown').upper()}")
    print(f"  Bits: {preset.get('bits', 'Unknown')}")
    print(
        f"  Total compression: {preset.get('compression', preset.get('total_compression', 'Unknown'))}x"
    )
    print(f"  PPL delta: {preset.get('ppl_delta', 'Unknown')}")
    if "calibration_file" in preset:
        print(f"  Calibration: {preset['calibration_file']}")
