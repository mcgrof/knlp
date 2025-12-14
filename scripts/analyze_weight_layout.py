#!/usr/bin/env python3
"""
Analyze weight layout in safetensors files for mobile packing optimization.

Compares the current file layout against optimal forward-pass order to
estimate read amplification from mmap-based inference.

No GPU required - pure file analysis operations.
"""

import os
import sys
import json
import argparse
import struct
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# No GPU needed for this analysis
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""


@dataclass
class TensorLayout:
    """Layout information for a tensor in safetensors file."""

    name: str
    dtype: str
    shape: List[int]
    data_offset: int  # Byte offset in file
    data_size: int  # Size in bytes


def parse_safetensors_header(file_path: str) -> Tuple[Dict, int]:
    """
    Parse safetensors header to extract tensor metadata.

    Safetensors format:
    - 8 bytes: header size (little-endian u64)
    - N bytes: JSON header
    - remaining: tensor data

    Args:
        file_path: Path to .safetensors file

    Returns:
        (header_dict, header_end_offset)
    """
    with open(file_path, "rb") as f:
        # Read header size
        header_size_bytes = f.read(8)
        header_size = struct.unpack("<Q", header_size_bytes)[0]

        # Read header JSON
        header_json = f.read(header_size)
        header = json.loads(header_json.decode("utf-8"))

        # Data starts after 8-byte size + header
        data_start = 8 + header_size

    return header, data_start


def get_tensor_layouts(file_path: str) -> List[TensorLayout]:
    """
    Extract tensor layout information from safetensors file.

    Args:
        file_path: Path to .safetensors file

    Returns:
        List of TensorLayout objects
    """
    header, data_start = parse_safetensors_header(file_path)

    layouts = []
    for name, info in header.items():
        if name == "__metadata__":
            continue

        dtype = info["dtype"]
        shape = info["shape"]
        offsets = info["data_offsets"]  # [start, end] relative to data section

        # Calculate absolute offset in file
        abs_offset = data_start + offsets[0]
        data_size = offsets[1] - offsets[0]

        layouts.append(
            TensorLayout(
                name=name,
                dtype=dtype,
                shape=shape,
                data_offset=abs_offset,
                data_size=data_size,
            )
        )

    # Sort by file offset
    layouts.sort(key=lambda x: x.data_offset)
    return layouts


def estimate_page_faults(
    layouts: List[TensorLayout],
    access_order: List[str],
    page_size: int = 4096,
) -> Dict[str, any]:
    """
    Estimate page fault metrics for given access order.

    Args:
        layouts: Tensor layouts sorted by file offset
        access_order: Order in which tensors will be accessed
        page_size: OS page size in bytes

    Returns:
        Dictionary with page fault metrics
    """
    # Build name -> layout map
    layout_map = {l.name: l for l in layouts}

    # Track which pages each tensor touches
    tensor_pages: Dict[str, set] = {}
    all_pages = set()

    for layout in layouts:
        start_page = layout.data_offset // page_size
        end_page = (layout.data_offset + layout.data_size - 1) // page_size
        pages = set(range(start_page, end_page + 1))
        tensor_pages[layout.name] = pages
        all_pages.update(pages)

    # Simulate access pattern
    pages_loaded = set()
    page_loads = 0  # Total page loads (may include reloads)
    unique_page_loads = 0  # First-time page loads
    cache_hits = 0

    access_sequence = []
    for name in access_order:
        if name not in layout_map:
            continue

        pages_needed = tensor_pages[name]
        new_pages = pages_needed - pages_loaded

        if new_pages:
            page_loads += len(new_pages)
            unique_page_loads += len(new_pages)
            pages_loaded.update(new_pages)
        else:
            cache_hits += len(pages_needed)

        access_sequence.append(
            {
                "tensor": name,
                "pages_needed": len(pages_needed),
                "new_pages": len(new_pages),
                "cached": len(new_pages) == 0,
            }
        )

    # Calculate metrics
    total_bytes = sum(l.data_size for l in layouts)
    bytes_accessed = sum(
        layout_map[n].data_size for n in access_order if n in layout_map
    )
    pages_touched = len(all_pages)

    # Read amplification: bytes loaded via pages / bytes actually needed
    bytes_loaded = page_loads * page_size
    read_amplification = bytes_loaded / bytes_accessed if bytes_accessed > 0 else 0

    return {
        "total_file_bytes": total_bytes,
        "bytes_accessed": bytes_accessed,
        "bytes_loaded_pages": bytes_loaded,
        "read_amplification": read_amplification,
        "page_size": page_size,
        "total_pages_in_file": pages_touched,
        "pages_loaded": page_loads,
        "unique_page_loads": unique_page_loads,
        "cache_hits": cache_hits,
        "tensors_accessed": len(access_order),
    }


def analyze_layout_efficiency(
    layouts: List[TensorLayout],
    forward_order: List[str],
    page_size: int = 4096,
) -> Dict[str, any]:
    """
    Compare current file layout vs optimal forward-order layout.

    Args:
        layouts: Current file layout
        forward_order: Optimal access order from forward pass
        page_size: OS page size

    Returns:
        Comparison metrics
    """
    # Current file order (by offset)
    file_order = [l.name for l in layouts]

    # Calculate metrics for current layout
    current_metrics = estimate_page_faults(layouts, forward_order, page_size)

    # Calculate metrics if file was laid out in forward order
    # (simulate by creating hypothetical sequential layout)
    hypothetical_layouts = []
    offset = layouts[0].data_offset if layouts else 0
    layout_map = {l.name: l for l in layouts}

    for name in forward_order:
        if name in layout_map:
            orig = layout_map[name]
            hypothetical_layouts.append(
                TensorLayout(
                    name=name,
                    dtype=orig.dtype,
                    shape=orig.shape,
                    data_offset=offset,
                    data_size=orig.data_size,
                )
            )
            offset += orig.data_size

    optimal_metrics = estimate_page_faults(
        hypothetical_layouts, forward_order, page_size
    )

    # Calculate improvement potential
    improvement = {
        "current_read_amplification": current_metrics["read_amplification"],
        "optimal_read_amplification": optimal_metrics["read_amplification"],
        "potential_improvement": (
            (
                current_metrics["read_amplification"]
                - optimal_metrics["read_amplification"]
            )
            / current_metrics["read_amplification"]
            * 100
            if current_metrics["read_amplification"] > 0
            else 0
        ),
        "current_pages_loaded": current_metrics["pages_loaded"],
        "optimal_pages_loaded": optimal_metrics["pages_loaded"],
        "pages_saved": current_metrics["pages_loaded"]
        - optimal_metrics["pages_loaded"],
    }

    return {
        "current_layout": current_metrics,
        "optimal_layout": optimal_metrics,
        "improvement": improvement,
        "file_order": file_order,
        "forward_order": forward_order,
    }


def get_hf_cache_path(model_name: str) -> Optional[str]:
    """
    Find the cached safetensors file for a HuggingFace model.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Path to safetensors file or None
    """
    import os
    from pathlib import Path

    # Common HF cache locations
    cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path(os.environ.get("HF_HOME", "")) / "hub",
        (
            Path(os.environ.get("TRANSFORMERS_CACHE", ""))
            if os.environ.get("TRANSFORMERS_CACHE")
            else None
        ),
    ]

    # Model directory name in cache
    model_dir_name = f"models--{model_name.replace('/', '--')}"

    for cache_dir in cache_dirs:
        if cache_dir is None or not cache_dir.exists():
            continue

        model_dir = cache_dir / model_dir_name
        if model_dir.exists():
            # Find snapshots
            snapshots_dir = model_dir / "snapshots"
            if snapshots_dir.exists():
                for snapshot in snapshots_dir.iterdir():
                    safetensors = snapshot / "model.safetensors"
                    if safetensors.exists():
                        return str(safetensors)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze safetensors layout for mobile packing optimization"
    )
    parser.add_argument(
        "--safetensors",
        type=str,
        help="Path to safetensors file (or use --model)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model name to find in cache",
    )
    parser.add_argument(
        "--access-graph",
        type=str,
        help="Path to access graph JSON (from build_weight_access_graph.py)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=4096,
        help="Page size in bytes (default: 4096)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="layout_analysis.json",
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Find safetensors file
    if args.safetensors:
        safetensors_path = args.safetensors
    else:
        print(f"Looking for cached safetensors for {args.model}...")
        safetensors_path = get_hf_cache_path(args.model)
        if safetensors_path is None:
            print(f"ERROR: Could not find cached safetensors for {args.model}")
            print("Please provide --safetensors path directly")
            sys.exit(1)

    print(f"Analyzing: {safetensors_path}")

    # Parse safetensors layout
    layouts = get_tensor_layouts(safetensors_path)
    print(f"Found {len(layouts)} tensors in file")

    # Get file size
    file_size = os.path.getsize(safetensors_path)
    print(f"File size: {file_size:,} bytes ({file_size/1e6:.1f} MB)")

    # Get forward order
    if args.access_graph:
        print(f"\nLoading forward order from {args.access_graph}")
        with open(args.access_graph, "r") as f:
            access_data = json.load(f)
        forward_order = access_data.get("forward_order", [])
    else:
        # Use file order as proxy (assume sequential access)
        print("\nNo access graph provided - using file order as access pattern")
        forward_order = [l.name for l in layouts]

    # Map HF weight names to safetensors names if needed
    # (safetensors might use slightly different naming)
    layout_names = {l.name for l in layouts}
    forward_order_mapped = []

    for name in forward_order:
        if name in layout_names:
            forward_order_mapped.append(name)
        else:
            # Try common transformations
            # HF uses "transformer.xxx", safetensors might omit prefix
            alt_name = name.replace("transformer.", "")
            if alt_name in layout_names:
                forward_order_mapped.append(alt_name)

    if len(forward_order_mapped) < len(forward_order):
        print(
            f"Warning: Only {len(forward_order_mapped)}/{len(forward_order)} weights found in safetensors"
        )

    # Analyze layout efficiency
    print(f"\nAnalyzing layout with page_size={args.page_size} bytes...")
    analysis = analyze_layout_efficiency(layouts, forward_order_mapped, args.page_size)

    # Build output
    results = {
        "file": safetensors_path,
        "file_size_bytes": file_size,
        "num_tensors": len(layouts),
        "page_size": args.page_size,
        "tensor_layouts": [asdict(l) for l in layouts],
        "analysis": analysis,
    }

    # Save results
    print(f"\nSaving results to {args.output}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("LAYOUT ANALYSIS SUMMARY")
    print("=" * 60)

    current = analysis["current_layout"]
    optimal = analysis["optimal_layout"]
    improvement = analysis["improvement"]

    print(f"\nFile: {safetensors_path}")
    print(f"Size: {file_size/1e6:.1f} MB ({len(layouts)} tensors)")
    print(f"Page size: {args.page_size} bytes")

    print(f"\n--- Current Layout (as stored in file) ---")
    print(f"Bytes accessed: {current['bytes_accessed']:,}")
    print(f"Bytes loaded (via pages): {current['bytes_loaded_pages']:,}")
    print(f"Read amplification: {current['read_amplification']:.2f}x")
    print(f"Pages loaded: {current['pages_loaded']}")

    print(f"\n--- Optimal Layout (forward-pass order) ---")
    print(f"Bytes loaded (via pages): {optimal['bytes_loaded_pages']:,}")
    print(f"Read amplification: {optimal['read_amplification']:.2f}x")
    print(f"Pages loaded: {optimal['pages_loaded']}")

    print(f"\n--- Improvement Potential ---")
    print(f"Read amplification reduction: {improvement['potential_improvement']:.1f}%")
    print(f"Pages saved: {improvement['pages_saved']}")

    # Show first 10 tensors in file order vs forward order
    print(f"\n--- Layout Comparison (first 10) ---")
    print(f"{'File Order':<40} {'Forward Order':<40}")
    print("-" * 80)
    file_order = analysis["file_order"]
    fwd_order = analysis["forward_order"]
    for i in range(min(10, len(file_order), len(fwd_order))):
        f_name = file_order[i][:38] if i < len(file_order) else ""
        o_name = fwd_order[i][:38] if i < len(fwd_order) else ""
        match = "✓" if f_name == o_name else "✗"
        print(f"{f_name:<40} {o_name:<40} {match}")


if __name__ == "__main__":
    main()
