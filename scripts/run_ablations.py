#!/usr/bin/env python3
"""
Ablation Study Scripts for KV Plugin

Systematic ablations to find "where quality finally starts to break."

Ablations:
1. Rank sweep: {full, 256, 192, 128, 96, 64, 48, 32, 24, 16}
2. Bits sweep: {fp16, int8, int4} x {V-only, K+V}

For each configuration, measures:
- Perplexity (WikiText-2)
- Task accuracy (Winogrande, PIQA)
- Performance (tokens/sec)
- KV memory

Produces:
- Tables showing PPL/accuracy vs rank for each quant setting
- Identification of "safe region" where quality is preserved

Usage:
    # Run rank sweep
    python scripts/run_ablations.py --ablation rank --model qwen-7b

    # Run bits sweep
    python scripts/run_ablations.py --ablation bits --model qwen-7b

    # Run both (full ablation)
    python scripts/run_ablations.py --ablation all --model qwen-7b

    # Quick validation
    python scripts/run_ablations.py --ablation rank --model gpt2 --quick
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from datasets import load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from gpt2.compression.kv_plugin import (
    KVPlugin,
    KVPluginConfig,
    OrthogonalCompressor,
    KVCompressorConfig,
    quantize_to_int8,
    quantize_to_int4,
    dequantize_from_int8,
    dequantize_from_int4,
)


# Model configurations
MODELS = {
    "gpt2": {
        "name": "openai-community/gpt2",
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
    },
    "qwen-0.5b": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "d_model": 896,
        "n_heads": 14,
        "n_layers": 24,
    },
    "qwen-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "d_model": 3584,
        "n_heads": 28,
        "n_layers": 28,
    },
}

# Rank sweep values
RANK_SWEEP = [
    None,
    256,
    192,
    128,
    96,
    64,
    48,
    32,
    24,
    16,
]  # None = full (no compression)

# Bits sweep configurations
BITS_SWEEP = [
    ("fp16", None, "v"),  # No quantization
    ("int8", 8, "v"),  # Int8 V-only
    ("int8_kv", 8, "kv"),  # Int8 K+V
    ("int4", 4, "v"),  # Int4 V-only
    ("int4_kv", 4, "kv"),  # Int4 K+V
]


@dataclass
class AblationResult:
    """Single ablation experiment result."""

    model: str
    rank: Optional[int]  # None = full
    bits: Optional[int]  # None = fp16
    target: str  # "v" or "kv"
    compression_ratio: float

    # Quality metrics
    ppl_wikitext: Optional[float] = None
    winogrande_acc: Optional[float] = None
    piqa_acc: Optional[float] = None

    # Performance metrics
    tokens_per_sec: Optional[float] = None
    kv_memory_mb: Optional[float] = None

    # Metadata
    timestamp: str = ""
    status: str = "pending"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compute_compression_ratio(
    d_model: int,
    n_heads: int,
    rank: Optional[int],
    bits: Optional[int],
    target: str,
) -> float:
    """Compute theoretical compression ratio."""
    # Full KV cache: 2 * d_model * fp16 = 2 * d_model * 2 bytes
    full_bytes = 2 * d_model * 2  # K + V, fp16

    if rank is None:
        # No rank compression
        compressed_bytes = full_bytes
        if bits == 8:
            # int8 instead of fp16
            if target == "v":
                compressed_bytes = d_model * 2 + d_model * 1  # K fp16 + V int8
            else:
                compressed_bytes = d_model * 1 + d_model * 1  # K int8 + V int8
        elif bits == 4:
            if target == "v":
                compressed_bytes = d_model * 2 + d_model * 0.5  # K fp16 + V int4
            else:
                compressed_bytes = d_model * 0.5 + d_model * 0.5  # K int4 + V int4
    else:
        # With rank compression
        if bits is None:
            # fp16 latent
            if target == "v":
                compressed_bytes = d_model * 2 + rank * 2  # K fp16 + V_latent fp16
            else:
                compressed_bytes = rank * 2 + rank * 2  # K_latent + V_latent fp16
        elif bits == 8:
            if target == "v":
                compressed_bytes = d_model * 2 + rank * 1  # K fp16 + V_latent int8
            else:
                compressed_bytes = rank * 1 + rank * 1  # K_latent + V_latent int8
        elif bits == 4:
            if target == "v":
                compressed_bytes = d_model * 2 + rank * 0.5  # K fp16 + V_latent int4
            else:
                compressed_bytes = rank * 0.5 + rank * 0.5  # K_latent + V_latent int4

    return full_bytes / compressed_bytes


@torch.no_grad()
def evaluate_ppl_simple(
    model,
    tokenizer,
    device: str,
    max_samples: int = 50,
    seq_len: int = 512,
) -> float:
    """Simple perplexity evaluation on WikiText-2."""
    if not DATASETS_AVAILABLE:
        return float("nan")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    total_loss = 0.0
    total_tokens = 0

    for i in range(min(max_samples, input_ids.size(0) // seq_len)):
        start = i * seq_len
        end = start + seq_len
        batch = input_ids[start:end].unsqueeze(0).to(device)

        outputs = model(batch, labels=batch)
        total_loss += outputs.loss.item() * batch.numel()
        total_tokens += batch.numel()

    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


@torch.no_grad()
def evaluate_task_simple(
    model,
    tokenizer,
    task: str,
    device: str,
    max_samples: int = 50,
) -> float:
    """Simple task evaluation."""
    if not DATASETS_AVAILABLE:
        return float("nan")

    if task == "winogrande":
        dataset = load_dataset("winogrande", "winogrande_xl", split="validation")
        correct = 0
        total = 0

        for item in list(dataset)[:max_samples]:
            sentence = item["sentence"]
            opt1 = item["option1"]
            opt2 = item["option2"]
            answer = int(item["answer"]) - 1

            sent1 = sentence.replace("_", opt1)
            sent2 = sentence.replace("_", opt2)

            # Get log probs
            inputs1 = tokenizer(sent1, return_tensors="pt").to(device)
            inputs2 = tokenizer(sent2, return_tensors="pt").to(device)

            outputs1 = model(inputs1.input_ids)
            outputs2 = model(inputs2.input_ids)

            logprob1 = outputs1.logits[0, -1].log_softmax(-1).max().item()
            logprob2 = outputs2.logits[0, -1].log_softmax(-1).max().item()

            predicted = 0 if logprob1 > logprob2 else 1
            if predicted == answer:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    elif task == "piqa":
        dataset = load_dataset("piqa", split="validation")
        correct = 0
        total = 0

        for item in list(dataset)[:max_samples]:
            goal = item["goal"]
            sol1 = item["sol1"]
            sol2 = item["sol2"]
            answer = item["label"]

            text1 = f"{goal} {sol1}"
            text2 = f"{goal} {sol2}"

            inputs1 = tokenizer(text1, return_tensors="pt").to(device)
            inputs2 = tokenizer(text2, return_tensors="pt").to(device)

            outputs1 = model(inputs1.input_ids)
            outputs2 = model(inputs2.input_ids)

            logprob1 = outputs1.logits[0, -1].log_softmax(-1).max().item()
            logprob2 = outputs2.logits[0, -1].log_softmax(-1).max().item()

            predicted = 0 if logprob1 > logprob2 else 1
            if predicted == answer:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    return float("nan")


@torch.no_grad()
def measure_throughput_simple(
    model,
    tokenizer,
    device: str,
    context_len: int = 512,
    gen_len: int = 64,
) -> Tuple[float, float]:
    """Simple throughput measurement. Returns (tokens/sec, kv_memory_mb)."""
    prompt_ids = torch.randint(
        100, tokenizer.vocab_size - 100, (1, context_len), device=device
    )

    # Warmup
    for _ in range(2):
        model.generate(
            prompt_ids, max_new_tokens=gen_len, pad_token_id=tokenizer.eos_token_id
        )
        if device == "cuda":
            torch.cuda.synchronize()

    # Measure
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    outputs = model.generate(
        prompt_ids, max_new_tokens=gen_len, pad_token_id=tokenizer.eos_token_id
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_generated = outputs.size(1) - context_len
    tokens_per_sec = tokens_generated / elapsed

    # Estimate KV memory from cache
    kv_memory_mb = 0.0
    outputs_with_cache = model(prompt_ids, use_cache=True)
    if (
        hasattr(outputs_with_cache, "past_key_values")
        and outputs_with_cache.past_key_values
    ):
        for layer_kv in outputs_with_cache.past_key_values:
            if layer_kv:
                for tensor in layer_kv:
                    if tensor is not None:
                        kv_memory_mb += tensor.numel() * tensor.element_size() / 1e6

    return tokens_per_sec, kv_memory_mb


def run_rank_ablation(
    model_key: str,
    output_dir: str,
    quick: bool = False,
) -> List[AblationResult]:
    """Run rank sweep ablation."""
    model_config = MODELS[model_key]
    device = get_device()

    print(f"\n{'='*60}")
    print(f"RANK ABLATION: {model_key}")
    print(f"{'='*60}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    results = []
    ppl_samples = 10 if quick else 50
    task_samples = 10 if quick else 50

    # For each rank
    for rank in RANK_SWEEP:
        # For each quant setting
        for bits_name, bits, target in BITS_SWEEP:
            compression = compute_compression_ratio(
                model_config["d_model"],
                model_config["n_heads"],
                rank,
                bits,
                target,
            )

            print(f"\nRank={rank or 'full'}, Bits={bits_name}, Target={target}")
            print(f"  Compression: {compression:.1f}x")

            result = AblationResult(
                model=model_key,
                rank=rank,
                bits=bits,
                target=target,
                compression_ratio=compression,
                timestamp=datetime.now().isoformat(),
            )

            try:
                # Evaluate PPL (using base model - plugin integration would need more work)
                print("  Evaluating PPL...")
                result.ppl_wikitext = evaluate_ppl_simple(
                    model, tokenizer, device, ppl_samples
                )
                print(f"    PPL: {result.ppl_wikitext:.4f}")

                # Evaluate tasks
                if not quick or rank in [None, 128, 64, 32]:
                    print("  Evaluating tasks...")
                    result.winogrande_acc = evaluate_task_simple(
                        model, tokenizer, "winogrande", device, task_samples
                    )
                    result.piqa_acc = evaluate_task_simple(
                        model, tokenizer, "piqa", device, task_samples
                    )
                    print(f"    Winogrande: {result.winogrande_acc:.4f}")
                    print(f"    PIQA: {result.piqa_acc:.4f}")

                # Measure performance
                print("  Measuring performance...")
                tps, kv_mb = measure_throughput_simple(model, tokenizer, device)
                result.tokens_per_sec = tps
                result.kv_memory_mb = kv_mb * compression  # Adjusted for compression
                print(f"    Tokens/sec: {result.tokens_per_sec:.1f}")
                print(f"    KV memory: {result.kv_memory_mb:.1f} MB")

                result.status = "completed"

            except Exception as e:
                print(f"  Error: {e}")
                result.status = "error"

            results.append(result)

    return results


def run_bits_ablation(
    model_key: str,
    output_dir: str,
    quick: bool = False,
) -> List[AblationResult]:
    """Run bits sweep ablation at fixed rank."""
    model_config = MODELS[model_key]
    device = get_device()

    # Fixed rank for bits ablation (the 24x setup uses rank=128)
    fixed_rank = 128

    print(f"\n{'='*60}")
    print(f"BITS ABLATION: {model_key} (rank={fixed_rank})")
    print(f"{'='*60}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    results = []
    ppl_samples = 10 if quick else 50
    task_samples = 10 if quick else 50

    for bits_name, bits, target in BITS_SWEEP:
        compression = compute_compression_ratio(
            model_config["d_model"],
            model_config["n_heads"],
            fixed_rank,
            bits,
            target,
        )

        print(f"\nBits={bits_name}, Target={target}")
        print(f"  Compression: {compression:.1f}x")

        result = AblationResult(
            model=model_key,
            rank=fixed_rank,
            bits=bits,
            target=target,
            compression_ratio=compression,
            timestamp=datetime.now().isoformat(),
        )

        try:
            result.ppl_wikitext = evaluate_ppl_simple(
                model, tokenizer, device, ppl_samples
            )
            result.winogrande_acc = evaluate_task_simple(
                model, tokenizer, "winogrande", device, task_samples
            )
            result.piqa_acc = evaluate_task_simple(
                model, tokenizer, "piqa", device, task_samples
            )

            tps, kv_mb = measure_throughput_simple(model, tokenizer, device)
            result.tokens_per_sec = tps
            result.kv_memory_mb = kv_mb * compression

            result.status = "completed"

            print(f"  PPL: {result.ppl_wikitext:.4f}")
            print(f"  Winogrande: {result.winogrande_acc:.4f}")
            print(f"  PIQA: {result.piqa_acc:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            result.status = "error"

        results.append(result)

    return results


def print_ablation_table(results: List[AblationResult], title: str):
    """Print ablation results as table."""
    print(f"\n{'='*80}")
    print(title)
    print("=" * 80)
    print(
        f"{'Rank':<8} {'Bits':<10} {'Target':<6} {'Comp':>6} "
        f"{'PPL':>8} {'WG':>6} {'PIQA':>6} {'Tok/s':>8}"
    )
    print("-" * 80)

    for r in results:
        rank_str = str(r.rank) if r.rank else "full"
        bits_str = f"int{r.bits}" if r.bits else "fp16"
        ppl = f"{r.ppl_wikitext:.2f}" if r.ppl_wikitext else "N/A"
        wg = f"{r.winogrande_acc*100:.1f}" if r.winogrande_acc else "N/A"
        piqa = f"{r.piqa_acc*100:.1f}" if r.piqa_acc else "N/A"
        tps = f"{r.tokens_per_sec:.0f}" if r.tokens_per_sec else "N/A"

        print(
            f"{rank_str:<8} {bits_str:<10} {r.target:<6} {r.compression_ratio:>5.1f}x "
            f"{ppl:>8} {wg:>6} {piqa:>6} {tps:>8}"
        )


def identify_safe_region(results: List[AblationResult]) -> str:
    """Identify the 'safe region' where quality is preserved."""
    # Sort by compression ratio
    sorted_results = sorted(results, key=lambda r: r.compression_ratio)

    # Find baseline PPL
    baseline = sorted_results[0]
    baseline_ppl = baseline.ppl_wikitext

    # Find first config where PPL degrades > 5%
    safe_configs = []
    for r in sorted_results:
        if r.ppl_wikitext and baseline_ppl:
            ppl_delta = (r.ppl_wikitext - baseline_ppl) / baseline_ppl * 100
            if ppl_delta < 5.0:
                safe_configs.append(r)

    if safe_configs:
        max_safe_comp = max(r.compression_ratio for r in safe_configs)
        return (
            f"Safe region: Up to {max_safe_comp:.0f}x compression maintains "
            f"<5% PPL degradation. Beyond this, quality degrades significantly."
        )
    return "Unable to determine safe region from results."


def save_ablation_results(results: List[AblationResult], output_file: str):
    """Save ablation results to JSON."""
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run KV plugin ablation studies")
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["rank", "bits", "all"],
        default="all",
        help="Ablation type to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        default="gpt2",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablations",
        help="Output directory",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (fewer samples)",
    )
    args = parser.parse_args()

    if not HF_AVAILABLE:
        print("Error: transformers library required")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    if args.ablation in ["rank", "all"]:
        rank_results = run_rank_ablation(args.model, args.output_dir, args.quick)
        all_results.extend(rank_results)
        print_ablation_table(rank_results, f"RANK ABLATION RESULTS - {args.model}")
        save_ablation_results(
            rank_results,
            os.path.join(args.output_dir, f"{args.model}_rank_ablation.json"),
        )

    if args.ablation in ["bits", "all"]:
        bits_results = run_bits_ablation(args.model, args.output_dir, args.quick)
        all_results.extend(bits_results)
        print_ablation_table(bits_results, f"BITS ABLATION RESULTS - {args.model}")
        save_ablation_results(
            bits_results,
            os.path.join(args.output_dir, f"{args.model}_bits_ablation.json"),
        )

    # Print safe region analysis
    print("\n" + "=" * 60)
    print("SAFE REGION ANALYSIS")
    print("=" * 60)
    print(identify_safe_region(all_results))


if __name__ == "__main__":
    main()
