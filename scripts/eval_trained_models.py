#!/usr/bin/env python3
"""
Post-training evaluation script for trained models.

Loads trained models from test_matrix_results directory and runs
comprehensive evaluation:
- Inference benchmarks (throughput, latency)
- lm-eval (hellaswag, etc.)
- KV cache analysis
- KVSplice parameter analysis (if applicable)

Results are uploaded to W&B under the original run IDs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Add parent to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


def find_wandb_run_id(test_dir):
    """Extract W&B run ID from training logs."""
    log_file = test_dir / "output.log"
    if not log_file.exists():
        return None

    with open(log_file, "r") as f:
        for line in f:
            if "wandb: Run data is saved locally in" in line:
                # Extract run ID from path like: wandb/run-20251127_170049-jbqnzyte
                parts = line.split("run-")
                if len(parts) > 1:
                    run_id = parts[1].split()[0].split("-")[-1]
                    return run_id
    return None


def load_model_and_config(checkpoint_path):
    """Load model from checkpoint."""
    print(f"\nLoading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get model config from checkpoint
    model_args = checkpoint.get("model_args", {})
    config = checkpoint.get("config", {})

    # Determine model type
    mla_variant = config.get("MLA_VARIANT", "")

    # Import appropriate model
    if "mla_kv" in mla_variant:
        from gpt2.mla import GPT2_MLA_KV, MLA_Config

        print("  Model type: GPT2_MLA_KV (MLA + KVSplice)")
        model_class = GPT2_MLA_KV
    elif "mla" in mla_variant or config.get("ENABLE_MLA"):
        from gpt2.mla import GPT2_MLA, MLA_Config

        print("  Model type: GPT2_MLA (base MLA)")
        model_class = GPT2_MLA
    else:
        # Standard GPT-2
        print("  Model type: GPT2 (standard)")
        # TODO: Import standard GPT2 if needed
        return None, None

    # Create model config
    cfg = MLA_Config(
        d_model=model_args.get("n_embd", 768),
        n_heads=model_args.get("n_head", 12),
        head_dim=model_args.get("n_embd", 768) // model_args.get("n_head", 12),
        d_latent=config.get("MLA_D_LATENT", 256),
        block_size=model_args.get("block_size", 1024),
        n_layers=model_args.get("n_layer", 12),
        dropout=model_args.get("dropout", 0.0),
    )

    # Instantiate model
    if "mla_kv" in mla_variant:
        compression_ratio = float(config.get("MLA_COMPRESSION_RATIO", 0.5))
        model = model_class(cfg, compression_ratio=compression_ratio)
    else:
        model = model_class(cfg)

    # Load weights
    model.load_state_dict(checkpoint["model"])

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model, cfg


def run_kv_cache_analysis(model, cfg):
    """Compute KV cache memory metrics."""
    print("\n--- KV Cache Analysis ---")

    n_layers = cfg.n_layers
    n_heads = cfg.n_heads
    head_dim = cfg.head_dim
    seq_lengths = [512, 1024, 2048, 4096]
    batch_size = 1

    metrics = {}

    for seq_len in seq_lengths:
        # Standard KV cache
        standard_cache_bytes = batch_size * n_layers * 2 * n_heads * seq_len * head_dim * 2
        standard_cache_mb = standard_cache_bytes / 1024**2

        # Check if model uses compressed KV cache
        if hasattr(cfg, "d_latent"):
            d_latent = cfg.d_latent

            # Check for KVSplice
            if hasattr(model, "compression_ratio"):
                compression_ratio = model.compression_ratio
                d_compressed = int(d_latent * compression_ratio)
                compressed_cache_bytes = (
                    batch_size * n_layers * seq_len * d_compressed * 2
                )
                actual_cache_mb = compressed_cache_bytes / 1024**2
                cache_type = "kvsplice"
            else:
                # MLA without KVSplice
                latent_cache_bytes = batch_size * n_layers * seq_len * d_latent * 2
                actual_cache_mb = latent_cache_bytes / 1024**2
                cache_type = "mla"

            savings_pct = (1 - actual_cache_mb / standard_cache_mb) * 100
        else:
            actual_cache_mb = standard_cache_mb
            cache_type = "standard"
            savings_pct = 0.0

        metrics[f"kv_cache/seq{seq_len}_standard_mb"] = standard_cache_mb
        metrics[f"kv_cache/seq{seq_len}_actual_mb"] = actual_cache_mb
        metrics[f"kv_cache/seq{seq_len}_savings_pct"] = savings_pct

        print(
            f"  seq={seq_len}: {actual_cache_mb:.1f} MB / {standard_cache_mb:.1f} MB "
            f"({savings_pct:.0f}% savings)"
        )

    metrics["kv_cache/type"] = (
        {"kvsplice": 2.0, "mla": 1.0, "standard": 0.0}[cache_type]
    )

    return metrics


def run_kvsplice_analysis(model):
    """Analyze KVSplice scale/shift parameters."""
    print("\n--- KVSplice Parameter Analysis ---")

    # Check if model has kvsplice
    if not hasattr(model, "layers"):
        print("  Model has no layers attribute")
        return None

    first_layer = model.layers[0] if hasattr(model, "layers") else None
    if not first_layer or not hasattr(first_layer, "attn"):
        print("  Model layers have no attn attribute")
        return None

    if not hasattr(first_layer.attn, "kvsplice"):
        print("  Model does not have KVSplice (MLA only)")
        return None

    import torch.nn.functional as F
    import numpy as np

    n_layers = len(model.layers)
    all_scales = []
    all_shifts = []

    for layer_idx, layer in enumerate(model.layers):
        if not hasattr(layer.attn, "kvsplice"):
            continue

        kvsplice = layer.attn.kvsplice
        scale_raw = kvsplice.transform_scale.data
        shift = kvsplice.transform_shift.data

        scale = F.softplus(scale_raw).cpu().numpy()
        all_scales.append(scale)
        all_shifts.append(shift.cpu().numpy())

    if not all_scales:
        return None

    # Average across layers
    avg_scale = np.mean(all_scales, axis=0)
    avg_shift = np.mean(all_shifts, axis=0)

    metrics = {
        "kvsplice/scale_mean": float(np.mean(avg_scale)),
        "kvsplice/scale_std": float(np.std(avg_scale)),
        "kvsplice/scale_min": float(np.min(avg_scale)),
        "kvsplice/scale_max": float(np.max(avg_scale)),
        "kvsplice/shift_mean": float(np.mean(avg_shift)),
        "kvsplice/shift_std": float(np.std(avg_shift)),
    }

    # Count prunable dimensions (scale < 0.1)
    low_scale_dims = int(np.sum(avg_scale < 0.1))
    metrics["kvsplice/prunable_dims"] = low_scale_dims
    metrics["kvsplice/prunable_pct"] = 100 * low_scale_dims / len(avg_scale)

    print(f"  Scale mean: {metrics['kvsplice/scale_mean']:.3f}")
    print(f"  Scale std: {metrics['kvsplice/scale_std']:.3f}")
    print(f"  Prunable dims: {low_scale_dims} ({metrics['kvsplice/prunable_pct']:.1f}%)")

    return metrics


def run_inference_benchmark(model, device="cuda"):
    """Run inference throughput/latency benchmarks."""
    print("\n--- Inference Benchmarks ---")

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping")
        return None

    import time
    import numpy as np

    model = model.to(device)
    model.eval()

    metrics = {}
    batch_sizes = [1, 4, 16]
    seq_len = 128
    warmup_iters = 5
    measure_iters = 20

    for batch_size in batch_sizes:
        # Create dummy input
        x = torch.randint(0, 50257, (batch_size, seq_len), device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = model(x)

        torch.cuda.synchronize()

        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(measure_iters):
                start = time.perf_counter()
                _ = model(x)
                torch.cuda.synchronize()
                latencies.append(time.perf_counter() - start)

        latencies = np.array(latencies)
        total_tokens = batch_size * seq_len * measure_iters
        total_time = latencies.sum()
        tokens_per_sec = total_tokens / total_time

        metrics[f"inference/tokens_per_sec_bs{batch_size}"] = tokens_per_sec
        metrics[f"inference/latency_ms_bs{batch_size}"] = np.median(latencies) * 1000
        metrics[f"inference/latency_p95_ms_bs{batch_size}"] = (
            np.percentile(latencies, 95) * 1000
        )

        print(
            f"  batch_size={batch_size}: {tokens_per_sec:.0f} tok/s, "
            f"latency={np.median(latencies)*1000:.1f}ms"
        )

    model.cpu()
    return metrics


def run_lm_eval(model, device="cuda", tasks=["hellaswag"], limit=100):
    """Run lm-eval benchmarks."""
    print("\n--- lm-eval Benchmarks ---")

    try:
        from lm_eval import evaluator
        from lm_eval.api.model import LM
        import tiktoken
        import torch.nn.functional as F
    except ImportError as e:
        print(f"  lm-eval not available: {e}")
        return None

    model = model.to(device)
    model.eval()

    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Wrapper class
    class GPT2ModelWrapper(LM):
        def __init__(wrapper_self, model, device, tokenizer, block_size):
            super().__init__()
            wrapper_self._model = model
            wrapper_self._device = device
            wrapper_self._tokenizer = tokenizer
            wrapper_self._block_size = block_size
            wrapper_self.batch_size_per_gpu = 1

        @property
        def eot_token_id(wrapper_self):
            return wrapper_self._tokenizer.eot_token

        @property
        def max_length(wrapper_self):
            return wrapper_self._block_size

        @property
        def max_gen_toks(wrapper_self):
            return 256

        @property
        def batch_size(wrapper_self):
            return 1

        @property
        def device(wrapper_self):
            return wrapper_self._device

        def tok_encode(wrapper_self, string, **kwargs):
            return wrapper_self._tokenizer.encode(
                string, allowed_special={"<|endoftext|>"}
            )

        def tok_decode(wrapper_self, tokens, **kwargs):
            return wrapper_self._tokenizer.decode(tokens)

        def _loglikelihood_tokens(wrapper_self, requests, disable_tqdm=False):
            results = []
            for context, continuation in requests:
                ctx_tensor = torch.tensor([context], device=wrapper_self._device)
                with torch.no_grad():
                    logits, _ = wrapper_self._model(ctx_tensor)
                log_probs = F.log_softmax(
                    logits[0, -len(continuation) - 1 : -1], dim=-1
                )
                ll = sum(
                    log_probs[i, continuation[i]].item()
                    for i in range(min(len(continuation), log_probs.size(0)))
                )
                results.append((ll, True))
            return results

        def loglikelihood(wrapper_self, requests):
            new_reqs = []
            for req in requests:
                context = wrapper_self.tok_encode(req.args[0])
                continuation = wrapper_self.tok_encode(req.args[1])
                new_reqs.append((context, continuation))
            return wrapper_self._loglikelihood_tokens(new_reqs)

        def loglikelihood_rolling(wrapper_self, requests):
            results = []
            for req in requests:
                tokens = wrapper_self.tok_encode(req.args[0])
                if len(tokens) < 2:
                    results.append((0.0, True))
                    continue
                results.append(wrapper_self._loglikelihood_tokens([(tokens[:-1], tokens[1:])])[0])
            return results

        def generate_until(wrapper_self, requests):
            results = []
            for req in requests:
                context = wrapper_self.tok_encode(req.args[0])
                max_tokens = req.args[1].get("max_gen_toks", 256)
                ctx_tensor = torch.tensor([context], device=wrapper_self._device)

                with torch.no_grad():
                    for _ in range(max_tokens):
                        logits, _ = wrapper_self._model(ctx_tensor)
                        next_token = logits[0, -1].argmax().item()
                        if next_token == wrapper_self.eot_token_id:
                            break
                        ctx_tensor = torch.cat(
                            [ctx_tensor, torch.tensor([[next_token]], device=wrapper_self._device)],
                            dim=1,
                        )

                results.append(wrapper_self.tok_decode(ctx_tensor[0].tolist()))
            return results

    wrapper = GPT2ModelWrapper(model, device, enc, block_size=1024)

    print(f"  Running tasks: {tasks} (limit={limit})")
    results = evaluator.simple_evaluate(
        model=wrapper,
        tasks=tasks,
        num_fewshot=0,
        limit=limit,
    )

    metrics = {}
    for task in tasks:
        if task in results["results"]:
            task_results = results["results"][task]
            for metric_name, value in task_results.items():
                if isinstance(value, (int, float)):
                    metrics[f"lm_eval/{task}_{metric_name}"] = value
                    print(f"  {task}/{metric_name}: {value:.4f}")

    model.cpu()
    return metrics


def upload_to_wandb(run_id, project, metrics, model_name):
    """Upload metrics to existing W&B run."""
    import wandb

    print(f"\n--- Uploading to W&B ---")
    print(f"  Project: {project}")
    print(f"  Run ID: {run_id}")

    # Resume the run
    run = wandb.init(
        project=project,
        id=run_id,
        resume="allow",
    )

    # Log metrics with "eval_" prefix to distinguish from training
    eval_metrics = {f"eval_{k}": v for k, v in metrics.items()}
    wandb.log(eval_metrics)

    wandb.finish()
    print("  ✓ Uploaded to W&B")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models and upload results to W&B"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to test_matrix_results directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--lm-eval-tasks",
        type=str,
        default="hellaswag",
        help="Comma-separated lm-eval tasks",
    )
    parser.add_argument(
        "--lm-eval-limit",
        type=int,
        default=100,
        help="Limit per task for lm-eval",
    )
    parser.add_argument(
        "--skip-wandb",
        action="store_true",
        help="Skip W&B upload (just print metrics)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        return 1

    # Find all test directories
    test_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("gpt2_")]

    if not test_dirs:
        print(f"No test directories found in {results_dir}")
        return 1

    print(f"Found {len(test_dirs)} test(s) to evaluate")

    # Load project name from config
    config_file = results_dir / "config.txt"
    project = "gpt2-evaluation"  # default
    if config_file.exists():
        with open(config_file) as f:
            for line in f:
                if "CONFIG_TRACKER_PROJECT" in line:
                    project = line.split("=")[1].strip().strip('"')
                    break

    for test_dir in sorted(test_dirs):
        print(f"\n{'='*60}")
        print(f"Evaluating: {test_dir.name}")
        print(f"{'='*60}")

        # Find checkpoint
        checkpoint_files = list(test_dir.glob("final_model*.pt"))
        if not checkpoint_files:
            print(f"  No checkpoint found in {test_dir}, skipping")
            continue

        checkpoint_path = checkpoint_files[0]

        # Load model
        model, cfg = load_model_and_config(checkpoint_path)
        if model is None:
            print("  Failed to load model, skipping")
            continue

        # Run all evaluations
        all_metrics = {}

        # KV cache analysis
        kv_metrics = run_kv_cache_analysis(model, cfg)
        if kv_metrics:
            all_metrics.update(kv_metrics)

        # KVSplice analysis
        kvsplice_metrics = run_kvsplice_analysis(model)
        if kvsplice_metrics:
            all_metrics.update(kvsplice_metrics)

        # Inference benchmarks
        if args.device == "cuda" and torch.cuda.is_available():
            inference_metrics = run_inference_benchmark(model, args.device)
            if inference_metrics:
                all_metrics.update(inference_metrics)

        # lm-eval
        tasks = [t.strip() for t in args.lm_eval_tasks.split(",")]
        lm_eval_metrics = run_lm_eval(
            model, args.device, tasks=tasks, limit=args.lm_eval_limit
        )
        if lm_eval_metrics:
            all_metrics.update(lm_eval_metrics)

        # Print summary
        print(f"\n--- Summary ---")
        print(f"  Total metrics: {len(all_metrics)}")

        # Upload to W&B
        if not args.skip_wandb:
            run_id = find_wandb_run_id(test_dir)
            if run_id:
                upload_to_wandb(run_id, project, all_metrics, test_dir.name)
            else:
                print("  Warning: Could not find W&B run ID, skipping upload")
                print("  Run with --skip-wandb to just print metrics")

        # Save metrics to JSON
        metrics_file = test_dir / "eval_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"  Saved metrics to {metrics_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
