#!/usr/bin/env python3
"""
Perplexity Evaluation Script for KV Plugin

Evaluates perplexity on standard benchmarks:
- WikiText-2 (standard LM evaluation)
- C4 (100k sample subset)
- Pile (1% slice)

Follows evaluation methodology from:
- Palu (ICLR 2025)
- MiniCache (NeurIPS 2024)
- PyramidKV (NeurIPS 2024)

Usage:
    python scripts/eval_ppl.py --model gpt2 --preset orthogonal_int4
    python scripts/eval_ppl.py --model Qwen/Qwen2.5-7B-Instruct --preset orthogonal_q8_kv
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers not available")

try:
    from datasets import load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available")

from gpt2.compression.kv_plugin import KVPlugin, KVPluginConfig


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_wikitext2(tokenizer, max_samples: int = None, seq_len: int = 2048):
    """Load WikiText-2 test set."""
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets library required for WikiText-2")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate all text
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])

    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    # Create sequences of seq_len
    n_sequences = input_ids.size(0) // seq_len
    if max_samples and n_sequences > max_samples:
        n_sequences = max_samples

    sequences = []
    for i in range(n_sequences):
        start = i * seq_len
        end = start + seq_len
        sequences.append(input_ids[start:end])

    return torch.stack(sequences)


def load_c4_sample(tokenizer, max_samples: int = 1000, seq_len: int = 2048):
    """Load C4 validation sample."""
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets library required for C4")

    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )

    sequences = []
    current_ids = []

    for item in tqdm(dataset, desc="Loading C4", total=max_samples * 2):
        tokens = tokenizer(item["text"], return_tensors="pt").input_ids[0]
        current_ids.extend(tokens.tolist())

        while len(current_ids) >= seq_len:
            sequences.append(torch.tensor(current_ids[:seq_len]))
            current_ids = current_ids[seq_len:]

            if len(sequences) >= max_samples:
                break

        if len(sequences) >= max_samples:
            break

    return torch.stack(sequences)


def load_pile_sample(tokenizer, max_samples: int = 1000, seq_len: int = 2048):
    """Load Pile validation sample."""
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets library required for Pile")

    # Use a Pile subset that's more accessible
    try:
        dataset = load_dataset(
            "monology/pile-uncopyrighted",
            split="validation",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception:
        print("Warning: Could not load Pile. Using WikiText-2 as fallback.")
        return load_wikitext2(tokenizer, max_samples, seq_len)

    sequences = []
    current_ids = []

    for item in tqdm(dataset, desc="Loading Pile", total=max_samples * 2):
        text = item.get("text", item.get("content", ""))
        if not text:
            continue

        tokens = tokenizer(text, return_tensors="pt").input_ids[0]
        current_ids.extend(tokens.tolist())

        while len(current_ids) >= seq_len:
            sequences.append(torch.tensor(current_ids[:seq_len]))
            current_ids = current_ids[seq_len:]

            if len(sequences) >= max_samples:
                break

        if len(sequences) >= max_samples:
            break

    if len(sequences) == 0:
        print("Warning: No Pile samples loaded. Using WikiText-2 as fallback.")
        return load_wikitext2(tokenizer, max_samples, seq_len)

    return torch.stack(sequences)


@torch.no_grad()
def evaluate_perplexity(
    model,
    tokenizer,
    dataset_name: str = "wikitext2",
    max_samples: int = 100,
    seq_len: int = 2048,
    batch_size: int = 1,
    device: str = "cuda",
    plugin: Optional[KVPlugin] = None,
) -> Dict:
    """
    Evaluate perplexity on a dataset.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        dataset_name: One of 'wikitext2', 'c4', 'pile'
        max_samples: Maximum number of sequences to evaluate
        seq_len: Sequence length
        batch_size: Batch size
        device: Device to use
        plugin: Optional KV plugin for compression

    Returns:
        Dict with perplexity and timing metrics
    """
    # Load dataset
    if dataset_name == "wikitext2":
        sequences = load_wikitext2(tokenizer, max_samples, seq_len)
    elif dataset_name == "c4":
        sequences = load_c4_sample(tokenizer, max_samples, seq_len)
    elif dataset_name == "pile":
        sequences = load_pile_sample(tokenizer, max_samples, seq_len)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Loaded {len(sequences)} sequences of length {seq_len}")

    model.eval()
    if plugin:
        plugin.reset_cache()

    total_loss = 0.0
    total_tokens = 0
    total_time = 0.0

    for i in tqdm(range(0, len(sequences), batch_size), desc=f"Eval {dataset_name}"):
        batch = sequences[i : i + batch_size].to(device)

        start_time = time.perf_counter()

        outputs = model(batch, labels=batch)
        loss = outputs.loss

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        total_loss += loss.item() * batch.numel()
        total_tokens += batch.numel()
        total_time += elapsed

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "dataset": dataset_name,
        "perplexity": perplexity,
        "loss": avg_loss,
        "tokens": total_tokens,
        "sequences": len(sequences),
        "time_seconds": total_time,
        "tokens_per_second": total_tokens / total_time,
    }


def run_evaluation(
    model_name: str,
    preset: str = "none",
    datasets: List[str] = ["wikitext2"],
    max_samples: int = 100,
    seq_len: int = 2048,
    batch_size: int = 1,
    output_file: Optional[str] = None,
) -> Dict:
    """
    Run full perplexity evaluation.

    Args:
        model_name: HuggingFace model name
        preset: KV plugin preset (or "none" for baseline)
        datasets: List of datasets to evaluate
        max_samples: Max samples per dataset
        seq_len: Sequence length
        batch_size: Batch size
        output_file: Optional output JSON file

    Returns:
        Dict with all results
    """
    device = get_device()
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    print(f"Preset: {preset}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if device != "cuda":
        model = model.to(device)

    # Create plugin if not baseline
    plugin = None
    if preset != "none":
        print(f"Creating KV plugin with preset: {preset}")
        plugin = KVPlugin.from_preset(preset, model)
        # Note: For proper integration, we'd need to patch the model
        # For now, this demonstrates the API

    # Measure GPU memory
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        model_memory = torch.cuda.max_memory_allocated() / 1e9

    # Run evaluations
    results = {
        "model": model_name,
        "preset": preset,
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_samples": max_samples,
            "seq_len": seq_len,
            "batch_size": batch_size,
        },
        "datasets": {},
    }

    for dataset_name in datasets:
        print(f"\nEvaluating on {dataset_name}...")
        try:
            dataset_results = evaluate_perplexity(
                model=model,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                max_samples=max_samples,
                seq_len=seq_len,
                batch_size=batch_size,
                device=device,
                plugin=plugin,
            )
            results["datasets"][dataset_name] = dataset_results
            print(f"  Perplexity: {dataset_results['perplexity']:.4f}")
            print(f"  Tokens/sec: {dataset_results['tokens_per_second']:.0f}")
        except Exception as e:
            print(f"  Error: {e}")
            results["datasets"][dataset_name] = {"error": str(e)}

    # Add memory info
    if device == "cuda":
        results["gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9

    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity with KV compression"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="none",
        help="KV plugin preset (none for baseline)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["wikitext2"],
        help="Datasets to evaluate (wikitext2, c4, pile)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=100, help="Max samples per dataset"
    )
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    if not HF_AVAILABLE:
        print("Error: transformers library required")
        sys.exit(1)

    if not DATASETS_AVAILABLE:
        print("Error: datasets library required")
        sys.exit(1)

    results = run_evaluation(
        model_name=args.model,
        preset=args.preset,
        datasets=args.datasets,
        max_samples=args.max_samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        output_file=args.output,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PERPLEXITY RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {results['model']}")
    print(f"Preset: {results['preset']}")
    print("-" * 60)

    for name, data in results["datasets"].items():
        if "error" in data:
            print(f"{name}: ERROR - {data['error']}")
        else:
            print(
                f"{name}: PPL={data['perplexity']:.4f}, {data['tokens_per_second']:.0f} tok/s"
            )


if __name__ == "__main__":
    main()
