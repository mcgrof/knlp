#!/usr/bin/env python3
"""
Generate llama-quantize CLI commands from a quantization plan.

Uses the exact mechanics from llama.cpp discussion #12741:
- --tensor-type for tensor-wise quantization (TWQ)
- --tensor-type with regex for layer-wise quantization (LWQ)
- --output-tensor-type and --token-embedding-type for special tensors

Reference: https://github.com/ggml-org/llama.cpp/discussions/12741
"""

import argparse
import json
import re
from pathlib import Path


def layers_to_regex(layers: list[int]) -> str:
    """
    Convert a list of layer indices to a compact regex pattern.

    Examples:
        [0, 1, 2] -> "(0|1|2)"
        [16, 17, 18, 19] -> "(1[6-9])"
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29] -> "(2[0-9])"
    """
    if not layers:
        return ""

    layers = sorted(set(layers))

    # Try to find contiguous ranges
    ranges = []
    start = layers[0]
    end = layers[0]

    for i in range(1, len(layers)):
        if layers[i] == end + 1:
            end = layers[i]
        else:
            ranges.append((start, end))
            start = layers[i]
            end = layers[i]
    ranges.append((start, end))

    # Convert ranges to regex patterns
    patterns = []
    for start, end in ranges:
        if start == end:
            patterns.append(str(start))
        elif end - start <= 2:
            # Small range: enumerate
            patterns.extend(str(i) for i in range(start, end + 1))
        else:
            # Large range: try compact patterns
            if start // 10 == end // 10:
                # Same decade: use character class
                patterns.append(f"{start // 10}[{start % 10}-{end % 10}]")
            else:
                # Cross decade: split
                # First decade
                if start % 10 != 0:
                    patterns.append(f"{start // 10}[{start % 10}-9]")
                    start = (start // 10 + 1) * 10
                # Middle decades
                while start + 9 <= end:
                    patterns.append(f"{start // 10}[0-9]")
                    start += 10
                # Last partial decade
                if start <= end:
                    if start % 10 == 0 and end % 10 == 9:
                        patterns.append(f"{start // 10}[0-9]")
                    elif start == end:
                        patterns.append(str(start))
                    else:
                        patterns.append(f"{start // 10}[{start % 10}-{end % 10}]")

    if len(patterns) == 1:
        return patterns[0]
    return f"({"|".join(patterns)})"


def pattern_to_llama_cpp(pattern: str) -> str:
    """
    Convert our tensor group name to llama.cpp tensor pattern.

    Our patterns:      llama.cpp patterns:
    attn_q          -> attn_q / q_proj
    attn_k          -> attn_k / k_proj
    attn_v          -> attn_v / v_proj
    attn_output     -> attn_output / o_proj
    ffn_gate        -> ffn_gate / gate_proj
    ffn_up          -> ffn_up / up_proj
    ffn_down        -> ffn_down / down_proj
    """
    # llama.cpp uses these patterns directly
    mapping = {
        "attn_q": "attn_q",
        "attn_k": "attn_k",
        "attn_v": "attn_v",
        "attn_output": "attn_output",
        "ffn_gate": "ffn_gate",
        "ffn_up": "ffn_up",
        "ffn_down": "ffn_down",
    }
    return mapping.get(pattern, pattern)


def build_tensor_type_arg(pattern: str, layers: list[int] | None, qtype: str) -> str:
    """
    Build a --tensor-type argument in llama.cpp format.

    TWQ (all layers): --tensor-type ffn_down=q6_k
    LWQ (specific layers): --tensor-type "\\.(18|19|20)\\.ffn_down=q6_k"
    """
    llama_pattern = pattern_to_llama_cpp(pattern)

    if layers is None:
        # Global (TWQ) - applies to all layers
        return f"--tensor-type {llama_pattern}={qtype}"
    else:
        # Layer-specific (LWQ) - use regex
        layer_regex = layers_to_regex(layers)
        # Double-escape backslash for shell
        return f'--tensor-type "\\\\.{layer_regex}\\\\.{llama_pattern}={qtype}"'


def emit_command(
    plan: dict,
    input_gguf: str,
    output_gguf: str,
    imatrix: str | None = None,
) -> str:
    """
    Generate the complete llama-quantize command.

    Format matches #12741 examples:
    llama-quantize [options] --imatrix matrix.dat input.gguf output.gguf QUANT
    """
    parts = ["llama-quantize"]

    # Add imatrix if provided
    if imatrix:
        parts.append(f"--imatrix {imatrix}")

    # Add tensor-type overrides
    for override in plan.get("overrides", []):
        pattern = override["pattern"]
        layers = override.get("layers")
        qtype = override["qtype"]

        arg = build_tensor_type_arg(pattern, layers, qtype)
        parts.append(arg)

    # Add input/output files and base quant
    parts.append(input_gguf)
    parts.append(output_gguf)
    parts.append(plan["base_quant"])

    return " \\\n    ".join(parts)


def emit_comparison_commands(
    plan: dict,
    input_gguf: str,
    output_dir: str,
    imatrix: str | None = None,
) -> dict[str, str]:
    """
    Generate commands for both baseline (naive) and FIM-guided quantization.

    This enables apples-to-apples PPL comparison.
    """
    base_quant = plan["base_quant"]
    output_path = Path(output_dir)

    # Naive (baseline) command
    naive_output = output_path / f"model-{base_quant}-naive.gguf"
    naive_parts = ["llama-quantize"]
    if imatrix:
        naive_parts.append(f"--imatrix {imatrix}")
    naive_parts.extend([input_gguf, str(naive_output), base_quant])
    naive_cmd = " \\\n    ".join(naive_parts)

    # FIM-guided command
    fim_output = output_path / f"model-{base_quant}-fim.gguf"
    fim_cmd = emit_command(plan, input_gguf, str(fim_output), imatrix)

    return {
        "naive": naive_cmd,
        "naive_output": str(naive_output),
        "fim_guided": fim_cmd,
        "fim_output": str(fim_output),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate llama-quantize CLI from quantization plan"
    )
    parser.add_argument(
        "--plan",
        type=str,
        required=True,
        help="Path to quantization plan JSON from plan_quant.py",
    )
    parser.add_argument(
        "--input-gguf",
        type=str,
        required=True,
        help="Path to input GGUF file (e.g., model-f16.gguf)",
    )
    parser.add_argument(
        "--output-gguf",
        type=str,
        default=None,
        help="Path to output GGUF file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for comparison mode (generates naive + FIM)",
    )
    parser.add_argument(
        "--imatrix",
        type=str,
        default=None,
        help="Path to importance matrix file (optional)",
    )
    parser.add_argument(
        "--output-script",
        type=str,
        default=None,
        help="Write commands to shell script instead of stdout",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Generate both naive and FIM-guided commands for comparison",
    )
    args = parser.parse_args()

    # Load plan
    with open(args.plan) as f:
        plan = json.load(f)

    if args.comparison or args.output_dir:
        # Comparison mode: generate both naive and FIM commands
        output_dir = args.output_dir or "."
        commands = emit_comparison_commands(
            plan, args.input_gguf, output_dir, args.imatrix
        )

        script = f"""#!/bin/bash
# FIM-Guided Quantization Comparison Script
# Generated from plan: {args.plan}
# Base quantization: {plan['base_quant']}

set -e

echo "=== Step 1: Naive (baseline) quantization ==="
{commands['naive']}

echo ""
echo "=== Step 2: FIM-guided quantization ==="
{commands['fim_guided']}

echo ""
echo "=== Quantization complete ==="
echo "Naive output: {commands['naive_output']}"
echo "FIM output: {commands['fim_output']}"
echo ""
echo "Run PPL evaluation with:"
echo "  llama-perplexity -m {commands['naive_output']} -f test.txt"
echo "  llama-perplexity -m {commands['fim_output']} -f test.txt"
"""
    else:
        # Single command mode
        output_gguf = args.output_gguf or "model-quantized.gguf"
        cmd = emit_command(plan, args.input_gguf, output_gguf, args.imatrix)
        script = f"""#!/bin/bash
# FIM-Guided Quantization Command
# Generated from plan: {args.plan}

{cmd}
"""

    if args.output_script:
        output_path = Path(args.output_script)
        with open(output_path, "w") as f:
            f.write(script)
        output_path.chmod(0o755)
        print(f"Script written to {output_path}")
    else:
        print(script)


if __name__ == "__main__":
    main()
