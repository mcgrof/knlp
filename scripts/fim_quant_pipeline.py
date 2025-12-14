#!/usr/bin/env python3
"""
End-to-end FIM-guided quantization pipeline.

This orchestrates the full workflow:
1. Compute FIM scores from calibration data
2. Generate quantization plan based on FIM
3. Emit llama-quantize commands
4. Optionally run quantization and evaluation

Usage:
    # Step 1: Compute FIM scores (CPU-only, no llama.cpp needed)
    python scripts/fim_quant_pipeline.py --step fim \
        --model openai-community/gpt2 \
        --calibration-data calibration.txt

    # Step 2: Generate quantization plan and commands
    python scripts/fim_quant_pipeline.py --step plan \
        --fim-scores fim_scores.json \
        --input-gguf model-f16.gguf

    # Step 3: Run quantization (requires llama.cpp)
    python scripts/fim_quant_pipeline.py --step quantize \
        --plan quant_plan.json \
        --input-gguf model-f16.gguf

    # Step 4: Evaluate (requires llama.cpp)
    python scripts/fim_quant_pipeline.py --step eval \
        --naive-gguf model-naive.gguf \
        --fim-gguf model-fim.gguf \
        --test-file test.txt

    # Full pipeline (all steps)
    python scripts/fim_quant_pipeline.py --step all \
        --model openai-community/gpt2 \
        --input-gguf model-f16.gguf \
        --test-file test.txt

Reference: llama.cpp discussion #12741
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Force CPU-only for FIM computation
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""

SCRIPT_DIR = Path(__file__).parent


def run_fim_scoring(args) -> Path:
    """Run fim_score.py to compute FIM importance."""
    output = args.output_dir / "fim_scores.json"

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "fim_score.py"),
        "--model",
        args.model,
        "--output",
        str(output),
        "--num-samples",
        str(args.num_samples),
        "--seq-length",
        str(args.seq_length),
    ]

    if args.calibration_data:
        cmd.extend(["--calibration-data", args.calibration_data])

    print(f"\n{'='*60}")
    print("STEP 1: Computing FIM scores")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: FIM scoring failed")
        sys.exit(1)

    return output


def run_planning(args, fim_scores_path: Path) -> Path:
    """Run plan_quant.py to generate quantization plan."""
    output = args.output_dir / "quant_plan.json"

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "plan_quant.py"),
        "--fim-scores",
        str(fim_scores_path),
        "--output",
        str(output),
        "--base-quant",
        args.base_quant,
        "--upgrade-quant",
        args.upgrade_quant,
        "--ffn-top-pct",
        str(args.ffn_top_pct),
        "--strategy",
        args.strategy,
    ]

    if args.attn_top_pct > 0:
        cmd.extend(["--attn-top-pct", str(args.attn_top_pct)])

    print(f"\n{'='*60}")
    print("STEP 2: Generating quantization plan")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: Planning failed")
        sys.exit(1)

    return output


def run_emit_commands(args, plan_path: Path) -> Path:
    """Run emit_llama_quantize_cmd.py to generate shell script."""
    output = args.output_dir / "quantize.sh"

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "emit_llama_quantize_cmd.py"),
        "--plan",
        str(plan_path),
        "--input-gguf",
        args.input_gguf,
        "--output-dir",
        str(args.output_dir),
        "--output-script",
        str(output),
        "--comparison",
    ]

    if args.imatrix:
        cmd.extend(["--imatrix", args.imatrix])

    print(f"\n{'='*60}")
    print("STEP 3: Generating quantization commands")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: Command generation failed")
        sys.exit(1)

    return output


def run_quantization(args, script_path: Path):
    """Execute the quantization script."""
    print(f"\n{'='*60}")
    print("STEP 4: Running quantization")
    print(f"{'='*60}")

    if not args.run_quantize:
        print(f"Skipping quantization. Run manually with:")
        print(f"  bash {script_path}")
        return

    result = subprocess.run(["bash", str(script_path)])
    if result.returncode != 0:
        print("ERROR: Quantization failed")
        sys.exit(1)


def run_evaluation(args):
    """Run run_eval.py to compare models."""
    naive_gguf = args.naive_gguf or str(
        args.output_dir / f"model-{args.base_quant}-naive.gguf"
    )
    fim_gguf = args.fim_gguf or str(
        args.output_dir / f"model-{args.base_quant}-fim.gguf"
    )

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_eval.py"),
        "--models",
        f"naive:{naive_gguf}",
        f"fim:{fim_gguf}",
        "--test-file",
        args.test_file,
        "--output",
        str(args.output_dir / "eval_results.json"),
        "--output-wandb",
        str(args.output_dir / "eval_wandb.json"),
    ]

    if args.llama_perplexity:
        cmd.extend(["--llama-perplexity", args.llama_perplexity])

    print(f"\n{'='*60}")
    print("STEP 5: Evaluating models")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    if not args.run_eval:
        print(f"Skipping evaluation. Run manually with:")
        print(f"  {' '.join(cmd)}")
        return

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: Evaluation failed")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="FIM-guided quantization pipeline for llama.cpp"
    )

    # Pipeline control
    parser.add_argument(
        "--step",
        type=str,
        choices=["fim", "plan", "quantize", "eval", "all"],
        default="all",
        help="Which step(s) to run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="fim_quant_output",
        help="Output directory for all artifacts",
    )

    # FIM scoring options
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        default=None,
        help="Path to calibration text file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of calibration sequences",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=512,
        help="Sequence length for calibration",
    )
    parser.add_argument(
        "--fim-scores",
        type=str,
        default=None,
        help="Path to existing FIM scores (skip FIM step)",
    )

    # Planning options
    parser.add_argument(
        "--base-quant",
        type=str,
        default="Q4_K_M",
        help="Base quantization level",
    )
    parser.add_argument(
        "--upgrade-quant",
        type=str,
        default="q6_k",
        help="Quantization for high-FIM tensors",
    )
    parser.add_argument(
        "--ffn-top-pct",
        type=float,
        default=15.0,
        help="Percentage of FFN layers to upgrade",
    )
    parser.add_argument(
        "--attn-top-pct",
        type=float,
        default=0.0,
        help="Percentage of attention layers to upgrade",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["layerwise", "global"],
        default="layerwise",
        help="Selection strategy",
    )
    parser.add_argument(
        "--plan",
        type=str,
        default=None,
        help="Path to existing plan (skip planning step)",
    )

    # Quantization options
    parser.add_argument(
        "--input-gguf",
        type=str,
        default=None,
        help="Path to input GGUF file",
    )
    parser.add_argument(
        "--imatrix",
        type=str,
        default=None,
        help="Path to importance matrix",
    )
    parser.add_argument(
        "--run-quantize",
        action="store_true",
        help="Actually run quantization (requires llama-quantize)",
    )

    # Evaluation options
    parser.add_argument(
        "--naive-gguf",
        type=str,
        default=None,
        help="Path to naive quantized model",
    )
    parser.add_argument(
        "--fim-gguf",
        type=str,
        default=None,
        help="Path to FIM-guided quantized model",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Path to test text file for PPL",
    )
    parser.add_argument(
        "--llama-perplexity",
        type=str,
        default="llama-perplexity",
        help="Path to llama-perplexity binary",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Actually run evaluation (requires llama-perplexity)",
    )

    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FIM-Guided Quantization Pipeline")
    print("Based on llama.cpp discussion #12741")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")

    # Determine which steps to run
    run_fim = args.step in ("fim", "all") and not args.fim_scores
    run_plan = args.step in ("plan", "all") and not args.plan
    run_quant = args.step in ("quantize", "all")
    run_eval = args.step in ("eval", "all")

    # Execute pipeline
    fim_scores_path = Path(args.fim_scores) if args.fim_scores else None
    plan_path = Path(args.plan) if args.plan else None

    if run_fim:
        fim_scores_path = run_fim_scoring(args)
    elif not fim_scores_path:
        fim_scores_path = args.output_dir / "fim_scores.json"

    if run_plan:
        if not fim_scores_path.exists():
            print(f"ERROR: FIM scores not found at {fim_scores_path}")
            print("Run with --step fim first, or provide --fim-scores")
            sys.exit(1)
        plan_path = run_planning(args, fim_scores_path)
    elif not plan_path:
        plan_path = args.output_dir / "quant_plan.json"

    if run_quant:
        if not args.input_gguf:
            print("ERROR: --input-gguf required for quantization step")
            sys.exit(1)
        if not plan_path.exists():
            print(f"ERROR: Plan not found at {plan_path}")
            print("Run with --step plan first, or provide --plan")
            sys.exit(1)
        script_path = run_emit_commands(args, plan_path)
        run_quantization(args, script_path)

    if run_eval:
        if not args.test_file:
            print("ERROR: --test-file required for evaluation step")
            sys.exit(1)
        run_evaluation(args)

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")
    print(f"Artifacts in: {args.output_dir}")


if __name__ == "__main__":
    main()
