#!/usr/bin/env python3
"""
Analyze GGUF file layout for mobile weight packing optimization.

GGUF is the llama.cpp model format, commonly used for quantized models.
This script parses the GGUF header to extract tensor metadata and analyze
the layout for read amplification.

No GPU required - pure file parsing operations.

GGUF Format Reference:
https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

import os
import sys
import json
import struct
import argparse
from typing import Dict, List, Tuple, Optional, BinaryIO
from dataclasses import dataclass, asdict
from collections import defaultdict
from enum import IntEnum

# No GPU needed
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""


class GGMLType(IntEnum):
    """GGML tensor data types."""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    BF16 = 30


# Bytes per element for each type (approximate for quantized types)
GGML_TYPE_SIZE = {
    GGMLType.F32: 4.0,
    GGMLType.F16: 2.0,
    GGMLType.BF16: 2.0,
    GGMLType.Q4_0: 0.5 + 2 / 32,  # 4 bits + scale per block of 32
    GGMLType.Q4_1: 0.5 + 4 / 32,  # 4 bits + scale + min per block
    GGMLType.Q5_0: 0.625 + 2 / 32,
    GGMLType.Q5_1: 0.625 + 4 / 32,
    GGMLType.Q8_0: 1.0 + 2 / 32,
    GGMLType.Q8_1: 1.0 + 4 / 32,
    GGMLType.Q2_K: 0.3125,  # ~2.5 bits effective
    GGMLType.Q3_K: 0.4375,  # ~3.5 bits effective
    GGMLType.Q4_K: 0.5625,  # ~4.5 bits effective
    GGMLType.Q5_K: 0.6875,  # ~5.5 bits effective
    GGMLType.Q6_K: 0.8125,  # ~6.5 bits effective
    GGMLType.Q8_K: 1.0625,  # ~8.5 bits effective
    GGMLType.I8: 1.0,
    GGMLType.I16: 2.0,
    GGMLType.I32: 4.0,
    GGMLType.I64: 8.0,
    GGMLType.F64: 8.0,
}

# Block sizes for quantized types
GGML_BLOCK_SIZE = {
    GGMLType.Q4_0: 32,
    GGMLType.Q4_1: 32,
    GGMLType.Q5_0: 32,
    GGMLType.Q5_1: 32,
    GGMLType.Q8_0: 32,
    GGMLType.Q8_1: 32,
    GGMLType.Q2_K: 256,
    GGMLType.Q3_K: 256,
    GGMLType.Q4_K: 256,
    GGMLType.Q5_K: 256,
    GGMLType.Q6_K: 256,
    GGMLType.Q8_K: 256,
}


@dataclass
class GGUFTensor:
    """Information about a tensor in GGUF file."""

    name: str
    n_dims: int
    shape: List[int]
    dtype: str
    dtype_id: int
    offset: int  # Offset from start of tensor data section
    size_bytes: int  # Estimated size in bytes


class GGUFReader:
    """Parse GGUF file format."""

    GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.version = 0
        self.n_tensors = 0
        self.n_kv = 0
        self.metadata: Dict = {}
        self.tensors: List[GGUFTensor] = []
        self.tensor_data_offset = 0

    def _read_string(self, f: BinaryIO) -> str:
        """Read a GGUF string (length-prefixed)."""
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    def _read_value(self, f: BinaryIO, value_type: int):
        """Read a GGUF metadata value based on type."""
        if value_type == 0:  # UINT8
            return struct.unpack("<B", f.read(1))[0]
        elif value_type == 1:  # INT8
            return struct.unpack("<b", f.read(1))[0]
        elif value_type == 2:  # UINT16
            return struct.unpack("<H", f.read(2))[0]
        elif value_type == 3:  # INT16
            return struct.unpack("<h", f.read(2))[0]
        elif value_type == 4:  # UINT32
            return struct.unpack("<I", f.read(4))[0]
        elif value_type == 5:  # INT32
            return struct.unpack("<i", f.read(4))[0]
        elif value_type == 6:  # FLOAT32
            return struct.unpack("<f", f.read(4))[0]
        elif value_type == 7:  # BOOL
            return struct.unpack("<B", f.read(1))[0] != 0
        elif value_type == 8:  # STRING
            return self._read_string(f)
        elif value_type == 9:  # ARRAY
            arr_type = struct.unpack("<I", f.read(4))[0]
            arr_len = struct.unpack("<Q", f.read(8))[0]
            return [self._read_value(f, arr_type) for _ in range(arr_len)]
        elif value_type == 10:  # UINT64
            return struct.unpack("<Q", f.read(8))[0]
        elif value_type == 11:  # INT64
            return struct.unpack("<q", f.read(8))[0]
        elif value_type == 12:  # FLOAT64
            return struct.unpack("<d", f.read(8))[0]
        else:
            raise ValueError(f"Unknown value type: {value_type}")

    def parse(self):
        """Parse the GGUF file header."""
        with open(self.file_path, "rb") as f:
            # Read magic
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != self.GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF magic: {hex(magic)}")

            # Read version
            self.version = struct.unpack("<I", f.read(4))[0]
            if self.version not in (2, 3):
                print(
                    f"Warning: GGUF version {self.version} may not be fully supported"
                )

            # Read counts
            self.n_tensors = struct.unpack("<Q", f.read(8))[0]
            self.n_kv = struct.unpack("<Q", f.read(8))[0]

            print(f"GGUF version: {self.version}")
            print(f"Tensors: {self.n_tensors}")
            print(f"Metadata entries: {self.n_kv}")

            # Read metadata key-value pairs
            for _ in range(self.n_kv):
                key = self._read_string(f)
                value_type = struct.unpack("<I", f.read(4))[0]
                value = self._read_value(f, value_type)
                self.metadata[key] = value

            # Read tensor info
            for _ in range(self.n_tensors):
                name = self._read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                shape = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                dtype_id = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]

                # Get dtype name
                try:
                    dtype_name = GGMLType(dtype_id).name
                except ValueError:
                    dtype_name = f"UNKNOWN_{dtype_id}"

                # Estimate size
                n_elements = 1
                for dim in shape:
                    n_elements *= dim

                bytes_per_elem = GGML_TYPE_SIZE.get(dtype_id, 4.0)
                size_bytes = int(n_elements * bytes_per_elem)

                self.tensors.append(
                    GGUFTensor(
                        name=name,
                        n_dims=n_dims,
                        shape=shape,
                        dtype=dtype_name,
                        dtype_id=dtype_id,
                        offset=offset,
                        size_bytes=size_bytes,
                    )
                )

            # Record where tensor data starts
            self.tensor_data_offset = f.tell()

            # Align to 32 bytes (GGUF requirement)
            alignment = self.metadata.get("general.alignment", 32)
            if self.tensor_data_offset % alignment != 0:
                self.tensor_data_offset += alignment - (
                    self.tensor_data_offset % alignment
                )

        # Sort tensors by offset
        self.tensors.sort(key=lambda t: t.offset)


def analyze_gguf_layout(
    tensors: List[GGUFTensor],
    data_offset: int,
    page_size: int = 4096,
) -> Dict:
    """
    Analyze GGUF tensor layout for read amplification.

    Args:
        tensors: List of tensors sorted by offset
        data_offset: Offset where tensor data starts in file
        page_size: OS page size

    Returns:
        Layout analysis results
    """
    # Calculate absolute offsets
    tensor_pages = {}
    all_pages = set()

    for tensor in tensors:
        abs_offset = data_offset + tensor.offset
        start_page = abs_offset // page_size
        end_page = (abs_offset + tensor.size_bytes - 1) // page_size
        pages = set(range(start_page, end_page + 1))
        tensor_pages[tensor.name] = {
            "start_page": start_page,
            "end_page": end_page,
            "num_pages": len(pages),
            "size_bytes": tensor.size_bytes,
            "abs_offset": abs_offset,
        }
        all_pages.update(pages)

    # Group tensors by type (for forward pass ordering)
    tensor_groups = defaultdict(list)
    for tensor in tensors:
        # Extract layer number and component type
        name = tensor.name
        if ".layers." in name or ".blk." in name:
            # Transformer layer tensor
            parts = name.replace(".layers.", ".blk.").split(".blk.")
            if len(parts) > 1:
                layer_part = parts[1].split(".")[0]
                try:
                    layer_num = int(layer_part)
                    tensor_groups[f"layer_{layer_num}"].append(tensor)
                except ValueError:
                    tensor_groups["other"].append(tensor)
            else:
                tensor_groups["other"].append(tensor)
        elif "token_embd" in name or "embed" in name.lower():
            tensor_groups["embedding"].append(tensor)
        elif "output" in name.lower() or "lm_head" in name.lower():
            tensor_groups["output"].append(tensor)
        else:
            tensor_groups["other"].append(tensor)

    # Estimate forward-pass order (embed -> layers -> output)
    forward_order = []
    if "embedding" in tensor_groups:
        forward_order.extend(tensor_groups["embedding"])

    # Add layers in order
    layer_keys = sorted(
        [k for k in tensor_groups.keys() if k.startswith("layer_")],
        key=lambda x: int(x.split("_")[1]),
    )
    for layer_key in layer_keys:
        forward_order.extend(tensor_groups[layer_key])

    if "output" in tensor_groups:
        forward_order.extend(tensor_groups["output"])
    if "other" in tensor_groups:
        forward_order.extend(tensor_groups["other"])

    # Calculate read amplification for file order vs forward order
    def calc_page_loads(tensor_list):
        pages_loaded = set()
        total_loads = 0
        for tensor in tensor_list:
            info = tensor_pages[tensor.name]
            pages = set(range(info["start_page"], info["end_page"] + 1))
            new_pages = pages - pages_loaded
            total_loads += len(new_pages)
            pages_loaded.update(pages)
        return total_loads, len(pages_loaded)

    file_order_loads, file_unique = calc_page_loads(tensors)
    forward_order_loads, forward_unique = calc_page_loads(forward_order)

    total_bytes = sum(t.size_bytes for t in tensors)
    file_bytes_loaded = file_order_loads * page_size
    forward_bytes_loaded = forward_order_loads * page_size

    return {
        "total_tensors": len(tensors),
        "total_bytes": total_bytes,
        "total_pages": len(all_pages),
        "page_size": page_size,
        "file_order": {
            "pages_loaded": file_order_loads,
            "bytes_loaded": file_bytes_loaded,
            "read_amplification": (
                file_bytes_loaded / total_bytes if total_bytes > 0 else 0
            ),
        },
        "forward_order": {
            "pages_loaded": forward_order_loads,
            "bytes_loaded": forward_bytes_loaded,
            "read_amplification": (
                forward_bytes_loaded / total_bytes if total_bytes > 0 else 0
            ),
        },
        "improvement_potential": {
            "pages_saved": file_order_loads - forward_order_loads,
            "reduction_percent": (
                (file_order_loads - forward_order_loads) / file_order_loads * 100
                if file_order_loads > 0
                else 0
            ),
        },
        "tensor_groups": {k: len(v) for k, v in tensor_groups.items()},
    }


def get_hf_gguf_path(repo_id: str, filename: str = None) -> Optional[str]:
    """Find or download a GGUF file from HuggingFace."""
    from huggingface_hub import hf_hub_download, list_repo_files

    if filename is None:
        # Find first GGUF file
        files = list_repo_files(repo_id)
        gguf_files = [f for f in files if f.endswith(".gguf")]
        if not gguf_files:
            return None
        # Prefer Q4_K_M if available
        for f in gguf_files:
            if "Q4_K_M" in f:
                filename = f
                break
        if filename is None:
            filename = gguf_files[0]

    return hf_hub_download(repo_id=repo_id, filename=filename)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GGUF file layout for mobile packing optimization"
    )
    parser.add_argument(
        "--gguf",
        type=str,
        help="Path to GGUF file",
    )
    parser.add_argument(
        "--repo",
        type=str,
        help="HuggingFace repo ID (e.g., TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="Specific GGUF filename in repo",
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
        default="gguf_layout_analysis.json",
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Find GGUF file
    if args.gguf:
        gguf_path = args.gguf
    elif args.repo:
        print(f"Finding GGUF in {args.repo}...")
        gguf_path = get_hf_gguf_path(args.repo, args.filename)
        if gguf_path is None:
            print("ERROR: No GGUF files found in repo")
            sys.exit(1)
    else:
        print("ERROR: Provide --gguf path or --repo ID")
        sys.exit(1)

    print(f"\nAnalyzing: {gguf_path}")
    print(f"File size: {os.path.getsize(gguf_path) / 1e6:.1f} MB")

    # Parse GGUF
    reader = GGUFReader(gguf_path)
    reader.parse()

    # Analyze layout
    print(f"\nAnalyzing layout with page_size={args.page_size}...")
    analysis = analyze_gguf_layout(
        reader.tensors, reader.tensor_data_offset, args.page_size
    )

    # Build output
    results = {
        "file": gguf_path,
        "file_size_bytes": os.path.getsize(gguf_path),
        "gguf_version": reader.version,
        "model_metadata": {
            k: v
            for k, v in reader.metadata.items()
            if k.startswith("general.") or k.startswith("llama.")
        },
        "tensors": [asdict(t) for t in reader.tensors],
        "analysis": analysis,
    }

    # Save results
    print(f"\nSaving results to {args.output}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("GGUF LAYOUT ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nFile: {os.path.basename(gguf_path)}")
    print(f"Size: {os.path.getsize(gguf_path) / 1e6:.1f} MB")
    print(f"Tensors: {analysis['total_tensors']}")
    print(f"Model data: {analysis['total_bytes'] / 1e6:.1f} MB")

    # Print quantization info
    dtype_counts = defaultdict(int)
    dtype_bytes = defaultdict(int)
    for t in reader.tensors:
        dtype_counts[t.dtype] += 1
        dtype_bytes[t.dtype] += t.size_bytes

    print(f"\n--- Quantization Distribution ---")
    for dtype, count in sorted(dtype_counts.items(), key=lambda x: -x[1]):
        mb = dtype_bytes[dtype] / 1e6
        print(f"  {dtype}: {count} tensors ({mb:.1f} MB)")

    print(f"\n--- Read Amplification Analysis ---")
    print(f"Page size: {args.page_size} bytes")
    print(f"Total pages: {analysis['total_pages']}")

    fo = analysis["file_order"]
    print(f"\nFile order (current layout):")
    print(f"  Pages loaded: {fo['pages_loaded']}")
    print(f"  Read amplification: {fo['read_amplification']:.2f}x")

    fwd = analysis["forward_order"]
    print(f"\nForward order (optimal for inference):")
    print(f"  Pages loaded: {fwd['pages_loaded']}")
    print(f"  Read amplification: {fwd['read_amplification']:.2f}x")

    imp = analysis["improvement_potential"]
    print(f"\nImprovement potential:")
    print(f"  Pages saved: {imp['pages_saved']}")
    print(f"  Reduction: {imp['reduction_percent']:.1f}%")

    # Show first 10 tensors
    print(f"\n--- First 10 Tensors (by file offset) ---")
    for i, t in enumerate(reader.tensors[:10]):
        size_kb = t.size_bytes / 1024
        print(f"  {i + 1}. {t.name}")
        print(f"      shape={t.shape}, dtype={t.dtype}, size={size_kb:.1f}KB")


if __name__ == "__main__":
    main()
