#!/usr/bin/env python3
"""
Inference benchmark for comparing trained MLA models.

Compares:
1. Text generation quality
2. Extended lm-eval benchmarks
3. Inference memory usage

Supports:
- GPT2_MLA: Base MLA with 6x KV cache compression
- GPT2_MLA_KV: MLA + KVSplice with 12x cache compression
"""

import argparse
import json
import os
import sys
import time
import torch
import tiktoken

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.mla import (
    GPT2_MLA,
    GPT2_MLA_KV,
    MLA_Config,
)


def load_model(checkpoint_path, device="cuda"):
    """Load a model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine model type from checkpoint or path
    # Check more specific patterns first (MLAKV before MLA)
    if "stepMLAKV" in checkpoint_path or "MLAKV" in checkpoint_path:
        model_type = "mlakv"
    elif "stepMLA" in checkpoint_path or "MLA" in checkpoint_path:
        model_type = "mla"
    else:
        model_type = "mla"  # default

    # Create config
    cfg = MLA_Config(
        d_model=768,
        n_heads=12,
        n_layers=12,
        block_size=1024,
        d_latent=256,
        head_dim=64,
    )

    # Create model
    if model_type == "mla":
        model = GPT2_MLA(cfg)
    elif model_type == "mlakv":
        model = GPT2_MLA_KV(cfg, compression_ratio=0.5)

    # Load weights
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()

    return model, model_type


@torch.no_grad()
def generate_text(
    model, prompt, max_tokens=100, temperature=0.8, top_k=40, device="cuda"
):
    """Generate text from a prompt."""
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_tokens):
        # Get logits
        if tokens.size(1) > 1024:
            tokens = tokens[:, -1024:]

        logits, _ = model(tokens)
        logits = logits[:, -1, :] / temperature

        # Top-k sampling
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

        # Stop at EOS
        if next_token.item() == enc.eot_token:
            break

    return enc.decode(tokens[0].tolist())


def measure_inference_memory(model, seq_len=512, batch_size=1, device="cuda"):
    """Measure inference memory usage."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Create input
    x = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(3):
        _ = model(x)

    torch.cuda.reset_peak_memory_stats()

    # Measure
    start = time.perf_counter()
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 10

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

    # Estimate KV cache size for this model
    cache_info = estimate_cache_size(model, seq_len, batch_size)

    return {
        "peak_memory_gb": peak_memory,
        "latency_ms": elapsed * 1000,
        "tokens_per_sec": seq_len / elapsed,
        **cache_info,
    }


def estimate_cache_size(model, seq_len, batch_size):
    """Estimate KV cache memory for autoregressive generation."""
    # Get model config
    if hasattr(model, "cfg"):
        cfg = model.cfg
        n_layers = cfg.n_layers
        n_heads = cfg.n_heads
        head_dim = cfg.head_dim
        d_latent = cfg.d_latent
    else:
        # Default GPT-2 config
        n_layers = 12
        n_heads = 12
        head_dim = 64
        d_latent = 256

    # Standard KV cache: store K and V for each layer
    # Shape: [batch, n_heads, seq_len, head_dim] * 2 (K and V) * n_layers
    standard_cache_bytes = (
        batch_size * n_layers * 2 * n_heads * seq_len * head_dim * 2
    )  # float16
    standard_cache_mb = standard_cache_bytes / 1024**2

    # Check if model uses KVSplice (compressed cache)
    cache_type = "standard"
    compressed_cache_mb = standard_cache_mb
    compression_ratio = 1.0

    if hasattr(model, "compression_ratio"):
        # RAMLAKV uses compressed latent cache
        compression_ratio = model.compression_ratio
        d_compressed = int(d_latent * compression_ratio)
        # Cache stores compressed latents instead of full K/V
        compressed_cache_bytes = (
            batch_size * n_layers * seq_len * d_compressed * 2
        )  # float16
        compressed_cache_mb = compressed_cache_bytes / 1024**2
        cache_type = "kvsplice"

    return {
        "cache_type": cache_type,
        "standard_cache_mb": standard_cache_mb,
        "actual_cache_mb": compressed_cache_mb,
        "cache_compression": compression_ratio,
        "cache_savings_pct": (
            (1 - compressed_cache_mb / standard_cache_mb) * 100
            if standard_cache_mb > 0
            else 0
        ),
    }


def run_lm_eval(model, tasks, limit=100, device="cuda"):
    """Run lm-eval benchmarks."""
    try:
        from lm_eval import evaluator
        from lm_eval.api.model import LM
        import tiktoken
    except ImportError:
        print("lm-eval not installed, skipping benchmarks")
        return {}

    enc = tiktoken.get_encoding("gpt2")

    class ModelWrapper(LM):
        def __init__(self, model, enc, device):
            super().__init__()
            self._model = model
            self._enc = enc
            self._device = device

        @property
        def eot_token_id(self):
            return self._enc.eot_token

        @property
        def max_length(self):
            return 1024

        @property
        def max_gen_toks(self):
            return 256

        @property
        def batch_size(self):
            return 8

        @property
        def device(self):
            return self._device

        def tok_encode(self, string, **kwargs):
            return self._enc.encode(string)

        def tok_decode(self, tokens, **kwargs):
            return self._enc.decode(tokens)

        def _model_call(self, inps):
            with torch.no_grad():
                logits, _ = self._model(inps)
            return logits

        def _model_generate(self, context, max_length, eos_token_id):
            return context  # Not used for acc tasks

        def loglikelihood(self, requests):
            results = []
            for context, continuation in [req.args for req in requests]:
                ctx_tokens = self._enc.encode(context)
                cont_tokens = self._enc.encode(continuation)

                full_tokens = ctx_tokens + cont_tokens
                if len(full_tokens) > 1024:
                    full_tokens = full_tokens[-1024:]
                    ctx_len = max(
                        0, len(ctx_tokens) - (len(ctx_tokens) + len(cont_tokens) - 1024)
                    )
                else:
                    ctx_len = len(ctx_tokens)

                input_ids = torch.tensor([full_tokens], device=self._device)

                with torch.no_grad():
                    logits, _ = self._model(input_ids)

                log_probs = torch.log_softmax(logits, dim=-1)

                cont_log_prob = 0.0
                for i, token in enumerate(cont_tokens):
                    pos = ctx_len + i - 1
                    if pos >= 0 and pos < log_probs.size(1):
                        cont_log_prob += log_probs[0, pos, token].item()

                is_greedy = True  # Simplified
                results.append((cont_log_prob, is_greedy))

            return results

        def loglikelihood_rolling(self, requests):
            return self.loglikelihood(requests)

        def generate_until(self, requests):
            return [""] * len(requests)

    wrapper = ModelWrapper(model, enc, device)

    results = evaluator.simple_evaluate(
        model=wrapper,
        tasks=tasks,
        num_fewshot=0,
        limit=limit,
    )

    return results.get("results", {})


def main():
    parser = argparse.ArgumentParser(
        description="Inference benchmark for MLA/RAMLA/SBA models"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory with test results (e.g., test_matrix_results_20251122_220931)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--generate", action="store_true", help="Run text generation comparison"
    )
    parser.add_argument(
        "--memory", action="store_true", help="Measure inference memory"
    )
    parser.add_argument("--lm-eval", action="store_true", help="Run extended lm-eval")
    parser.add_argument(
        "--lm-eval-tasks",
        type=str,
        default="hellaswag,winogrande,arc_easy",
        help="Tasks for lm-eval",
    )
    parser.add_argument(
        "--lm-eval-limit", type=int, default=100, help="Samples per task"
    )
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if args.all:
        args.generate = True
        args.memory = True
        args.lm_eval = True

    # Find all model checkpoints
    checkpoints = {}
    for subdir in os.listdir(args.results_dir):
        subdir_path = os.path.join(args.results_dir, subdir)
        if os.path.isdir(subdir_path):
            ckpt_path = os.path.join(subdir_path, "final_model_stepV0.pt")
            if os.path.exists(ckpt_path):
                # Extract step name (check more specific patterns first)
                if "stepMLAKV0" in subdir:
                    name = "MLAKV0"
                elif "stepMLA0" in subdir:
                    name = "MLA0"
                elif "stepRAMLAKV0" in subdir:
                    name = "RAMLAKV0"
                elif "stepRAMLA0" in subdir:
                    name = "RAMLA0"
                elif "stepSBASS0" in subdir:
                    name = "SBASS0"
                elif "stepSBAKV0" in subdir:
                    name = "SBAKV0"
                elif "stepSBA0" in subdir:
                    name = "SBA0"
                else:
                    continue
                checkpoints[name] = ckpt_path

    print(f"Found {len(checkpoints)} model checkpoints:")
    for name, path in sorted(checkpoints.items()):
        print(f"  {name}: {path}")
    print()

    results = {}

    # Text generation comparison
    if args.generate:
        print("=" * 60)
        print("TEXT GENERATION COMPARISON")
        print("=" * 60)

        prompts = [
            "The meaning of life is",
            "In the beginning, there was",
            "Scientists have discovered that",
        ]

        for name, ckpt_path in sorted(checkpoints.items()):
            print(f"\n--- {name} ---")
            try:
                model, model_type = load_model(ckpt_path, args.device)
                print(f"Loaded {model_type} model")

                for prompt in prompts:
                    output = generate_text(
                        model, prompt, max_tokens=50, device=args.device
                    )
                    print(f"\nPrompt: {prompt}")
                    print(f"Output: {output}")

                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error: {e}")
        print()

    # Memory measurement
    if args.memory:
        print("=" * 60)
        print("INFERENCE MEMORY BENCHMARK")
        print("=" * 60)

        memory_results = {}
        for name, ckpt_path in sorted(checkpoints.items()):
            try:
                model, model_type = load_model(ckpt_path, args.device)
                stats = measure_inference_memory(model, device=args.device)
                memory_results[name] = stats
                cache_info = ""
                if stats.get("cache_savings_pct", 0) > 0:
                    cache_info = f", cache: {stats['actual_cache_mb']:.1f}MB ({stats['cache_savings_pct']:.0f}% savings)"
                else:
                    cache_info = f", cache: {stats['standard_cache_mb']:.1f}MB"
                print(
                    f"{name}: {stats['peak_memory_gb']:.2f} GB, "
                    f"{stats['latency_ms']:.1f} ms, "
                    f"{stats['tokens_per_sec']:.0f} tok/s{cache_info}"
                )
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"{name}: Error - {e}")

        results["memory"] = memory_results
        print()

    # Extended lm-eval
    if args.lm_eval:
        print("=" * 60)
        print("LM-EVAL BENCHMARKS")
        print("=" * 60)

        tasks = [t.strip() for t in args.lm_eval_tasks.split(",")]
        print(f"Tasks: {tasks}, Limit: {args.lm_eval_limit}")

        eval_results = {}
        for name, ckpt_path in sorted(checkpoints.items()):
            print(f"\n--- {name} ---")
            try:
                model, model_type = load_model(ckpt_path, args.device)
                task_results = run_lm_eval(
                    model, tasks, args.lm_eval_limit, args.device
                )
                eval_results[name] = task_results

                for task_name, task_metrics in task_results.items():
                    acc = task_metrics.get(
                        "acc_norm,none", task_metrics.get("acc,none", 0)
                    )
                    print(f"  {task_name}: {acc:.4f}")

                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error: {e}")
                import traceback

                traceback.print_exc()

        results["lm_eval"] = eval_results

    # Save results
    output_path = os.path.join(args.results_dir, "inference_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
