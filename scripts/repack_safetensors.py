#!/usr/bin/env python3
"""
Repack safetensors files in forward-pass order for sequential I/O.

Takes a model and its forward-pass access order, then creates a new
safetensors file with tensors arranged for sequential reading during
inference.

No GPU required - pure file operations.
"""

import os
import sys
import json
import argparse
import shutil
from typing import Dict, List, Optional
from pathlib import Path

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""

import torch


def load_access_order(access_graph_path: str) -> List[str]:
    """Load forward-pass order from access graph JSON."""
    with open(access_graph_path, "r") as f:
        data = json.load(f)
    return data.get("forward_order", [])


def get_safetensors_path(model_name: str) -> Optional[str]:
    """Find safetensors file for a HuggingFace model."""
    from huggingface_hub import hf_hub_download, list_repo_files

    try:
        files = list_repo_files(model_name)
        safetensors_files = [f for f in files if f.endswith(".safetensors")]

        if not safetensors_files:
            return None

        # Prefer model.safetensors
        for f in safetensors_files:
            if f == "model.safetensors":
                return hf_hub_download(model_name, f)

        return hf_hub_download(model_name, safetensors_files[0])
    except Exception as e:
        print(f"Error finding safetensors: {e}")
        return None


def repack_safetensors(
    input_path: str,
    output_path: str,
    forward_order: List[str],
    name_mapping: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Repack safetensors file with tensors in forward-pass order.

    Args:
        input_path: Path to original safetensors file
        output_path: Path for repacked file
        forward_order: Tensor names in forward-pass order
        name_mapping: Optional mapping from forward_order names to safetensors names

    Returns:
        Statistics about the repacking
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    print(f"Loading tensors from {input_path}...")

    # Load all tensors
    tensors = {}
    tensor_names_in_file = []

    with safe_open(input_path, framework="pt") as f:
        tensor_names_in_file = list(f.keys())
        for name in tensor_names_in_file:
            tensors[name] = f.get_tensor(name)

    print(f"Loaded {len(tensors)} tensors")

    # Build name mapping if needed
    # safetensors often omits "transformer." prefix
    if name_mapping is None:
        name_mapping = {}
        for fwd_name in forward_order:
            if fwd_name in tensors:
                name_mapping[fwd_name] = fwd_name
            else:
                # Try removing common prefixes
                for prefix in ["transformer.", "model.", ""]:
                    alt_name = fwd_name.replace(prefix, "", 1) if prefix else fwd_name
                    if alt_name in tensors:
                        name_mapping[fwd_name] = alt_name
                        break

    # Reorder tensors
    ordered_tensors = {}
    reordered_count = 0

    for fwd_name in forward_order:
        tensor_name = name_mapping.get(fwd_name)
        if tensor_name and tensor_name in tensors:
            ordered_tensors[tensor_name] = tensors[tensor_name]
            reordered_count += 1

    # Add any remaining tensors not in forward_order
    remaining = 0
    for name in tensor_names_in_file:
        if name not in ordered_tensors:
            ordered_tensors[name] = tensors[name]
            remaining += 1

    print(f"Reordered {reordered_count} tensors, {remaining} unchanged")

    # Save repacked file
    print(f"Saving to {output_path}...")
    save_file(ordered_tensors, output_path)

    # Calculate statistics
    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)

    # Measure order change
    original_order = tensor_names_in_file
    new_order = list(ordered_tensors.keys())

    # Count how many tensors are now in different positions
    position_changes = sum(
        1
        for i, name in enumerate(new_order)
        if i < len(original_order) and original_order[i] != name
    )

    stats = {
        "input_path": input_path,
        "output_path": output_path,
        "input_size_bytes": input_size,
        "output_size_bytes": output_size,
        "num_tensors": len(tensors),
        "tensors_reordered": reordered_count,
        "tensors_unchanged": remaining,
        "position_changes": position_changes,
        "original_order": original_order[:10],  # First 10 for reference
        "new_order": new_order[:10],
    }

    return stats


def repack_from_model(
    model_name: str,
    output_path: str,
    access_graph_path: Optional[str] = None,
) -> Dict:
    """
    Repack a HuggingFace model's safetensors file.

    If no access graph provided, generates one first.

    Args:
        model_name: HuggingFace model name
        output_path: Path for repacked model
        access_graph_path: Optional path to pre-computed access graph

    Returns:
        Repacking statistics
    """
    # Get safetensors path
    print(f"Finding safetensors for {model_name}...")
    input_path = get_safetensors_path(model_name)
    if input_path is None:
        raise ValueError(f"Could not find safetensors for {model_name}")

    print(f"Found: {input_path}")

    # Get forward order
    if access_graph_path:
        print(f"Loading access order from {access_graph_path}...")
        forward_order = load_access_order(access_graph_path)
    else:
        print("Generating access order via forward pass...")
        # Import and use the access graph builder
        from build_weight_access_graph import WeightAccessTracer, load_public_model

        model, tokenizer = load_public_model(model_name)
        tracer = WeightAccessTracer(model)

        # Run forward pass to get order
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, 64), device="cpu")
        tracer.trace(input_ids)
        forward_order = tracer.get_forward_order()

        print(f"Generated order for {len(forward_order)} tensors")

    # Repack
    stats = repack_safetensors(input_path, output_path, forward_order)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Repack safetensors files in forward-pass order"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Direct path to input safetensors (overrides --model)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for repacked safetensors",
    )
    parser.add_argument(
        "--access-graph",
        type=str,
        default=None,
        help="Path to access graph JSON (from build_weight_access_graph.py)",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default=None,
        help="Path to save repacking statistics JSON",
    )

    args = parser.parse_args()

    if args.input:
        # Direct file repacking
        if not args.access_graph:
            print("ERROR: --access-graph required when using --input")
            sys.exit(1)

        forward_order = load_access_order(args.access_graph)
        stats = repack_safetensors(args.input, args.output, forward_order)
    else:
        # Model-based repacking
        stats = repack_from_model(args.model, args.output, args.access_graph)

    # Print summary
    print("\n" + "=" * 60)
    print("REPACKING SUMMARY")
    print("=" * 60)
    print(f"Input:  {stats['input_path']}")
    print(f"Output: {stats['output_path']}")
    print(
        f"Size:   {stats['input_size_bytes']/1e6:.1f} MB -> {stats['output_size_bytes']/1e6:.1f} MB"
    )
    print(f"Tensors: {stats['num_tensors']} ({stats['tensors_reordered']} reordered)")
    print(f"Position changes: {stats['position_changes']}")

    print(f"\nOriginal order (first 10): {stats['original_order']}")
    print(f"New order (first 10):      {stats['new_order']}")

    # Save stats
    if args.stats_output:
        with open(args.stats_output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to {args.stats_output}")


if __name__ == "__main__":
    main()
