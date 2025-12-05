#!/usr/bin/env python3
"""
KV Plugin H100 Inference Benchmark Suite

Comprehensive benchmarking for KV cache compression on H100 GPUs.
Tests memory scaling, batch throughput, long-context performance,
and quality across serious models.

Usage:
    # Quick test with default model
    python scripts/kv_plugin_h100_benchmark.py --quick

    # Full benchmark with W&B logging
    python scripts/kv_plugin_h100_benchmark.py --model meta-llama/Llama-2-7b-hf --wandb

    # Sweep all presets on a model
    python scripts/kv_plugin_h100_benchmark.py --model mistralai/Mistral-7B-v0.1 --all-presets --wandb

    # Memory scaling analysis
    python scripts/kv_plugin_h100_benchmark.py --memory-scaling --wandb

    # Batch throughput sweep
    python scripts/kv_plugin_h100_benchmark.py --batch-sweep --wandb
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent.parent))

from gpt2.compression.kv_plugin import KVPlugin, KVPluginConfig


# Models suitable for H100 testing (fit in 80GB)
SERIOUS_MODELS = {
    "small": [
        "openai-community/gpt2",
        "microsoft/phi-2",
        "Qwen/Qwen2.5-0.5B",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ],
    "medium": [
        "meta-llama/Llama-2-7b-hf",
        "mistralai/Mistral-7B-v0.1",
        "Qwen/Qwen2.5-7B",
        "google/gemma-7b",
    ],
    "large": [
        "meta-llama/Llama-2-13b-hf",
        "Qwen/Qwen2.5-14B",
    ],
    "xlarge": [
        # These need 4-bit quantization on single H100
        "meta-llama/Llama-2-70b-hf",
        "mistralai/Mixtral-8x7B-v0.1",
    ],
}


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    model_name: str = "openai-community/gpt2"
    preset: str = "balanced"
    device: str = "cuda"
    dtype: str = "float16"  # float16, bfloat16, float32
    # Sequence lengths
    context_lengths: List[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    max_new_tokens: int = 128
    # Batch sizes
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    # Trials
    warmup_iters: int = 3
    benchmark_iters: int = 10
    # W&B
    wandb_project: str = "kv-plugin-h100-benchmark"
    wandb_entity: Optional[str] = None
    # Quantization
    use_4bit: bool = False
    # Calibration
    calibration_samples: int = 100


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    model: str
    preset: str
    batch_size: int
    context_length: int
    # Timing
    ttft_ms: float  # Time to first token
    ttft_std_ms: float
    tpot_ms: float  # Time per output token
    throughput_tok_s: float  # Tokens per second
    # Memory
    peak_memory_gb: float
    kv_cache_mb: float
    kv_cache_compressed_mb: float
    compression_ratio: float
    # Quality
    perplexity: Optional[float] = None
    # Meta
    timestamp: str = ""
    gpu_name: str = ""
    dtype: str = ""


class WandBLogger:
    """W&B logging wrapper with graceful fallback."""

    def __init__(
        self,
        enabled: bool = False,
        project: str = "kv-plugin-benchmark",
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
    ):
        self.enabled = enabled
        self.run = None

        if not enabled:
            return

        try:
            import wandb

            self.wandb = wandb

            self.run = wandb.init(
                project=project,
                entity=entity,
                config=config or {},
                name=run_name,
                reinit=True,
            )
            print(f"W&B initialized: {wandb.run.url}")
        except ImportError:
            print("Warning: wandb not installed, logging disabled")
            self.enabled = False
        except Exception as e:
            print(f"Warning: Failed to init W&B: {e}")
            self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if not self.enabled or self.run is None:
            return
        try:
            if step is not None:
                self.wandb.log(metrics, step=step)
            else:
                self.wandb.log(metrics)
        except Exception as e:
            print(f"Warning: Failed to log to W&B: {e}")

    def log_summary(self, metrics: Dict[str, Any]):
        if not self.enabled or self.run is None:
            return
        try:
            for k, v in metrics.items():
                self.wandb.run.summary[k] = v
        except Exception as e:
            print(f"Warning: Failed to log summary: {e}")

    def finish(self):
        if self.enabled and self.run is not None:
            try:
                self.wandb.finish()
            except Exception:
                pass


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"device": "cpu"}

    props = torch.cuda.get_device_properties(0)
    return {
        "device": torch.cuda.get_device_name(0),
        "total_memory_gb": props.total_memory / 1e9,
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
    }


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype_str, torch.float16)


def load_model(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    use_4bit: bool = False,
) -> Tuple[nn.Module, AutoTokenizer]:
    """Load model with optional quantization."""
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    return model, tokenizer


def measure_ttft(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup: int = 3,
    trials: int = 10,
) -> Tuple[float, float]:
    """Measure time-to-first-token (prefill latency)."""
    device = input_ids.device

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids, use_cache=True)
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(input_ids, use_cache=True)

        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std


def measure_generation_throughput(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    warmup: int = 2,
    trials: int = 5,
) -> Tuple[float, float]:
    """Measure generation throughput (tokens/sec) and TPOT."""
    # Check model's max sequence length and adjust max_new_tokens
    model_config = getattr(model, "config", None)
    max_position = getattr(
        model_config,
        "max_position_embeddings",
        getattr(model_config, "n_positions", 2048),
    )
    input_len = input_ids.shape[1]
    available_tokens = max_position - input_len

    if available_tokens <= 0:
        # Can't generate, input already at max length
        return 0.0, 0.0

    actual_max_new_tokens = min(max_new_tokens, available_tokens)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=min(10, actual_max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()

    # Benchmark
    times = []
    tokens_generated = []

    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=actual_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        tokens_generated.append(outputs.shape[1] - input_ids.shape[1])

    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)

    throughput = avg_tokens / avg_time if avg_time > 0 else 0  # tok/s
    tpot = (avg_time * 1000) / avg_tokens if avg_tokens > 0 else 0  # ms/tok

    return throughput, tpot


def measure_kv_cache_size(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> Tuple[float, int]:
    """Measure KV cache size in MB."""
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    past_kv = outputs.past_key_values
    if past_kv is None:
        return 0.0, 0

    total_bytes = 0
    n_layers = len(past_kv)

    for layer_kv in past_kv:
        for tensor in layer_kv:
            if tensor is not None:
                total_bytes += tensor.numel() * tensor.element_size()

    return total_bytes / (1024**2), n_layers


def measure_perplexity(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int = 2048,
) -> float:
    """Compute perplexity on text."""
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    input_ids = encodings.input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    return torch.exp(outputs.loss).item()


def create_input_ids(
    tokenizer: AutoTokenizer,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Create valid input IDs using actual tokenization."""
    # Use real text that tokenizes well
    sample_text = (
        """
    The transformer architecture has revolutionized natural language processing.
    Attention mechanisms allow models to focus on relevant parts of the input.
    Key-value caching enables efficient autoregressive generation by storing
    intermediate states. Compression techniques can reduce memory requirements
    while maintaining quality. Large language models have become increasingly
    powerful, capable of generating coherent text and reasoning about complex
    problems. The computational requirements continue to grow, driving research
    into efficiency improvements and novel architectures.
    """
        * 20
    )  # Repeat to get enough tokens

    # Tokenize
    tokens = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        max_length=context_length,
        padding="max_length",
    ).input_ids

    # Expand to batch size
    if tokens.shape[1] < context_length:
        # Pad with repeated tokens if needed
        tokens = tokens.repeat(1, (context_length // tokens.shape[1]) + 1)[
            :, :context_length
        ]

    # Repeat for batch
    input_ids = tokens.repeat(batch_size, 1).to(device)
    return input_ids


def run_single_benchmark(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    preset: str,
    batch_size: int,
    context_length: int,
    max_new_tokens: int,
    warmup: int,
    trials: int,
    eval_text: Optional[str] = None,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Create input with valid token IDs using proper tokenization
    input_ids = create_input_ids(tokenizer, batch_size, context_length, device)

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Measure TTFT
    ttft_avg, ttft_std = measure_ttft(model, input_ids, warmup, trials)

    # Measure throughput (only for batch_size=1, skip in quick mode to avoid OOM)
    # Also skip if context is close to max position
    model_config = getattr(model, "config", None)
    max_pos = getattr(
        model_config,
        "max_position_embeddings",
        getattr(model_config, "n_positions", 2048),
    )
    room_for_gen = max_pos - context_length
    if batch_size == 1 and room_for_gen >= 64 and max_new_tokens > 0:
        throughput, tpot = measure_generation_throughput(
            model,
            tokenizer,
            input_ids,
            min(max_new_tokens, room_for_gen - 32),
            warmup,
            max(1, trials // 2),
        )
    else:
        throughput, tpot = 0.0, 0.0

    # Measure KV cache
    kv_cache_mb, n_layers = measure_kv_cache_size(model, input_ids)

    # Peak memory
    peak_memory = torch.cuda.max_memory_allocated() / (1e9)

    # Perplexity (only for batch_size=1)
    ppl = None
    if batch_size == 1 and eval_text:
        try:
            ppl = measure_perplexity(model, tokenizer, eval_text)
        except Exception:
            pass

    # Get compression info from preset
    preset_info = KVPlugin.PRESETS.get(preset, {})
    d_compressed = preset_info.get("d_compressed", 0)
    # Estimate compression ratio
    if d_compressed > 0 and kv_cache_mb > 0:
        # Get d_model from model config
        model_config = getattr(model, "config", None)
        d_model = getattr(model_config, "hidden_size", 768)
        compression_ratio = d_model / d_compressed
        kv_compressed_mb = kv_cache_mb / compression_ratio
    else:
        compression_ratio = 1.0
        kv_compressed_mb = kv_cache_mb

    return BenchmarkResult(
        model=str(getattr(model, "name_or_path", "unknown")),
        preset=preset,
        batch_size=batch_size,
        context_length=context_length,
        ttft_ms=ttft_avg,
        ttft_std_ms=ttft_std,
        tpot_ms=tpot,
        throughput_tok_s=throughput,
        peak_memory_gb=peak_memory,
        kv_cache_mb=kv_cache_mb,
        kv_cache_compressed_mb=kv_compressed_mb,
        compression_ratio=compression_ratio,
        perplexity=ppl,
        timestamp=datetime.now().isoformat(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        dtype=str(dtype),
    )


def benchmark_memory_scaling(
    model_name: str,
    presets: List[str],
    context_lengths: List[int],
    config: BenchmarkConfig,
    logger: WandBLogger,
    skip_calibration: bool = False,
) -> List[BenchmarkResult]:
    """Benchmark memory scaling across context lengths."""
    results = []
    dtype = get_dtype(config.dtype)

    model, tokenizer = load_model(model_name, config.device, dtype, config.use_4bit)
    model.name_or_path = model_name

    # Get model's max sequence length and filter context lengths
    model_config = getattr(model, "config", None)
    max_position = getattr(
        model_config,
        "max_position_embeddings",
        getattr(model_config, "n_positions", 2048),
    )
    # Filter context lengths to not exceed model max (leave room for generation)
    valid_context_lengths = [c for c in context_lengths if c <= max_position - 32]
    if not valid_context_lengths:
        valid_context_lengths = [min(max_position - 32, 512)]
    print(
        f"Model max position: {max_position}, testing context lengths: {valid_context_lengths}"
    )

    # Evaluation text for perplexity
    eval_text = (
        """
    The transformer architecture revolutionized natural language processing.
    Attention mechanisms allow models to focus on relevant input parts.
    """
        * 50
    )

    for preset in presets:
        print(f"\n{'='*60}")
        print(f"Testing preset: {preset}")
        print(f"{'='*60}")

        # Apply compression plugin if not baseline
        plugin = None
        if preset != "none":
            try:
                plugin = KVPlugin.from_preset(preset, model, device=config.device)
                # Skip calibration for orthogonal presets or if explicitly requested
                is_orthogonal = preset.startswith("orthogonal")
                if not skip_calibration and not is_orthogonal:
                    cal_tokens = tokenizer(
                        eval_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=2048,
                    ).input_ids.to(config.device)
                    plugin.calibrate([cal_tokens])
                plugin.patch_model()
            except Exception as e:
                print(f"  Warning: Failed to apply preset: {e}")
                continue

        for ctx_len in valid_context_lengths:
            print(f"  Context length: {ctx_len}")
            try:
                result = run_single_benchmark(
                    model=model,
                    tokenizer=tokenizer,
                    preset=preset,
                    batch_size=1,
                    context_length=ctx_len,
                    max_new_tokens=config.max_new_tokens,
                    warmup=config.warmup_iters,
                    trials=config.benchmark_iters,
                    eval_text=eval_text,
                )
                results.append(result)

                # Log to W&B
                logger.log(
                    {
                        "preset": preset,
                        "context_length": ctx_len,
                        "ttft_ms": result.ttft_ms,
                        "throughput_tok_s": result.throughput_tok_s,
                        "peak_memory_gb": result.peak_memory_gb,
                        "kv_cache_mb": result.kv_cache_mb,
                        "compression_ratio": result.compression_ratio,
                        "perplexity": result.perplexity,
                    }
                )

                print(
                    f"    TTFT: {result.ttft_ms:.2f}ms, Memory: {result.peak_memory_gb:.2f}GB"
                )

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at context_length={ctx_len}")
                torch.cuda.empty_cache()
                break
            except (torch.AcceleratorError, RuntimeError) as e:
                if "CUDA" in str(e) or "device-side assert" in str(e):
                    print(f"    CUDA error - aborting remaining tests")
                    break
                print(f"    Error: {e}")
            except Exception as e:
                print(f"    Error: {e}")

        # Unpatch for next preset
        if plugin is not None:
            try:
                plugin.unpatch_model()
            except Exception:
                pass
            del plugin
            torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


def benchmark_batch_throughput(
    model_name: str,
    presets: List[str],
    batch_sizes: List[int],
    context_length: int,
    config: BenchmarkConfig,
    logger: WandBLogger,
) -> List[BenchmarkResult]:
    """Benchmark throughput across batch sizes."""
    results = []
    dtype = get_dtype(config.dtype)

    model, tokenizer = load_model(model_name, config.device, dtype, config.use_4bit)
    model.name_or_path = model_name

    for preset in presets:
        print(f"\n{'='*60}")
        print(f"Testing preset: {preset}")
        print(f"{'='*60}")

        plugin = None
        if preset != "none":
            try:
                plugin = KVPlugin.from_preset(preset, model, device=config.device)
                cal_tokens = create_input_ids(tokenizer, 1, 2048, config.device)
                plugin.calibrate([cal_tokens])
                plugin.patch_model()
            except Exception as e:
                print(f"  Warning: Failed to apply preset: {e}")
                continue

        for bs in batch_sizes:
            print(f"  Batch size: {bs}")
            try:
                result = run_single_benchmark(
                    model=model,
                    tokenizer=tokenizer,
                    preset=preset,
                    batch_size=bs,
                    context_length=context_length,
                    max_new_tokens=config.max_new_tokens,
                    warmup=config.warmup_iters,
                    trials=config.benchmark_iters,
                )
                results.append(result)

                logger.log(
                    {
                        "preset": preset,
                        "batch_size": bs,
                        "ttft_ms": result.ttft_ms,
                        "peak_memory_gb": result.peak_memory_gb,
                        "kv_cache_mb": result.kv_cache_mb,
                    }
                )

                print(
                    f"    TTFT: {result.ttft_ms:.2f}ms, Memory: {result.peak_memory_gb:.2f}GB"
                )

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at batch_size={bs}")
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(f"    Error: {e}")

        if plugin is not None:
            try:
                plugin.unpatch_model()
            except Exception:
                pass
            del plugin
            torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


def benchmark_model_sweep(
    models: List[str],
    preset: str,
    config: BenchmarkConfig,
    logger: WandBLogger,
) -> List[BenchmarkResult]:
    """Benchmark a preset across multiple models."""
    results = []
    dtype = get_dtype(config.dtype)

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        try:
            model, tokenizer = load_model(
                model_name, config.device, dtype, config.use_4bit
            )
            model.name_or_path = model_name
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        plugin = None
        if preset != "none":
            try:
                plugin = KVPlugin.from_preset(preset, model, device=config.device)
                cal_tokens = create_input_ids(tokenizer, 1, 2048, config.device)
                plugin.calibrate([cal_tokens])
                plugin.patch_model()
            except Exception as e:
                print(f"  Warning: Failed to apply preset: {e}")

        for ctx_len in [512, 1024, 2048]:
            try:
                result = run_single_benchmark(
                    model=model,
                    tokenizer=tokenizer,
                    preset=preset,
                    batch_size=1,
                    context_length=ctx_len,
                    max_new_tokens=config.max_new_tokens,
                    warmup=config.warmup_iters,
                    trials=config.benchmark_iters,
                )
                results.append(result)

                logger.log(
                    {
                        "model": model_name,
                        "context_length": ctx_len,
                        "ttft_ms": result.ttft_ms,
                        "throughput_tok_s": result.throughput_tok_s,
                        "peak_memory_gb": result.peak_memory_gb,
                    }
                )

                print(
                    f"  ctx={ctx_len}: TTFT={result.ttft_ms:.2f}ms, {result.throughput_tok_s:.1f} tok/s"
                )

            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at context_length={ctx_len}")
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(f"  Error: {e}")

        if plugin is not None:
            try:
                plugin.unpatch_model()
            except Exception:
                pass
            del plugin

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    return results


def print_results_table(results: List[BenchmarkResult]):
    """Print results as a formatted table."""
    if not results:
        print("No results to display")
        return

    print("\n" + "=" * 120)
    print("BENCHMARK RESULTS")
    print("=" * 120)
    print(
        f"{'Model':<25} {'Preset':<12} {'BS':>3} {'Ctx':>5} {'TTFT(ms)':>10} {'Tok/s':>8} {'Mem(GB)':>8} {'KV(MB)':>8} {'Ratio':>6}"
    )
    print("-" * 120)

    for r in results:
        model_short = r.model.split("/")[-1][:24]
        ppl_str = f"{r.perplexity:.2f}" if r.perplexity else "-"
        print(
            f"{model_short:<25} {r.preset:<12} {r.batch_size:>3} {r.context_length:>5} "
            f"{r.ttft_ms:>10.2f} {r.throughput_tok_s:>8.1f} {r.peak_memory_gb:>8.2f} "
            f"{r.kv_cache_mb:>8.2f} {r.compression_ratio:>5.1f}x"
        )


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save results to JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "gpu_info": get_gpu_info(),
        "results": [asdict(r) for r in results],
    }

    with open(output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="KV Plugin H100 Inference Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark with GPT-2
  python scripts/kv_plugin_h100_benchmark.py --quick

  # Full benchmark with LLaMA-7B and W&B logging
  python scripts/kv_plugin_h100_benchmark.py --model meta-llama/Llama-2-7b-hf --wandb

  # Memory scaling analysis
  python scripts/kv_plugin_h100_benchmark.py --memory-scaling --model mistralai/Mistral-7B-v0.1 --wandb

  # Batch throughput sweep
  python scripts/kv_plugin_h100_benchmark.py --batch-sweep --model Qwen/Qwen2.5-7B

  # Test all presets on a model
  python scripts/kv_plugin_h100_benchmark.py --all-presets --model meta-llama/Llama-2-7b-hf

  # Model comparison sweep
  python scripts/kv_plugin_h100_benchmark.py --model-sweep --preset balanced --wandb
        """,
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="balanced",
        choices=list(KVPlugin.PRESETS.keys()),
        help="Compression preset",
    )
    parser.add_argument("--all-presets", action="store_true", help="Test all presets")

    # Benchmark modes
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark (small model, few configs)",
    )
    parser.add_argument(
        "--memory-scaling",
        action="store_true",
        help="Memory scaling analysis across context lengths",
    )
    parser.add_argument(
        "--batch-sweep", action="store_true", help="Batch size throughput sweep"
    )
    parser.add_argument(
        "--model-sweep", action="store_true", help="Sweep across multiple models"
    )

    # Configuration
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="Context lengths to test",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128, help="Max tokens to generate"
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--trials", type=int, default=10, help="Benchmark trials")

    # Hardware
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--use-4bit", action="store_true", help="Use 4-bit quantization"
    )

    # W&B
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="kv-plugin-h100-benchmark",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="W&B entity/team"
    )

    # Output
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    # GPU info
    gpu_info = get_gpu_info()
    print("=" * 60)
    print("KV Plugin H100 Inference Benchmark")
    print("=" * 60)
    print(f"GPU: {gpu_info.get('device', 'N/A')}")
    print(f"VRAM: {gpu_info.get('total_memory_gb', 0):.1f} GB")
    print(f"Compute: {gpu_info.get('compute_capability', 'N/A')}")
    print(f"SMs: {gpu_info.get('multi_processor_count', 0)}")
    print("=" * 60)

    # Build config
    config = BenchmarkConfig(
        model_name=args.model,
        preset=args.preset,
        dtype=args.dtype,
        context_lengths=args.context_lengths,
        batch_sizes=args.batch_sizes,
        max_new_tokens=args.max_new_tokens,
        warmup_iters=args.warmup,
        benchmark_iters=args.trials,
        use_4bit=args.use_4bit,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    # Determine presets to test
    if args.all_presets:
        presets = list(KVPlugin.PRESETS.keys())
    else:
        presets = [args.preset]

    # Initialize W&B
    run_name = f"{args.model.split('/')[-1]}_{args.preset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = WandBLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=asdict(config),
        run_name=run_name,
    )

    # Run benchmarks
    results = []

    if args.quick:
        # Quick mode: small model, few configs
        # Use "orthogonal" instead of "balanced" - orthogonal needs no calibration
        config.context_lengths = [256, 512, 768]
        config.batch_sizes = [1, 4]
        config.benchmark_iters = 5
        results = benchmark_memory_scaling(
            "openai-community/gpt2",
            ["none", "orthogonal"],
            config.context_lengths,
            config,
            logger,
            skip_calibration=True,  # Skip calibration for orthogonal
        )

    elif args.memory_scaling:
        results = benchmark_memory_scaling(
            args.model,
            presets,
            args.context_lengths,
            config,
            logger,
        )

    elif args.batch_sweep:
        results = benchmark_batch_throughput(
            args.model,
            presets,
            args.batch_sizes,
            1024,  # Fixed context for batch sweep
            config,
            logger,
        )

    elif args.model_sweep:
        # Test preset across models
        models_to_test = SERIOUS_MODELS["small"] + SERIOUS_MODELS["medium"][:2]
        results = benchmark_model_sweep(
            models_to_test,
            args.preset,
            config,
            logger,
        )

    else:
        # Default: memory scaling on single model
        results = benchmark_memory_scaling(
            args.model,
            presets,
            args.context_lengths,
            config,
            logger,
        )

    # Print results
    print_results_table(results)

    # Save results
    if args.output:
        save_results(results, args.output)
    else:
        # Default output
        model_short = args.model.split("/")[-1]
        output_path = f"key_results/h100_benchmark_{model_short}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(results, output_path)

    # Log summary to W&B
    if results:
        summary = {
            "total_configs_tested": len(results),
            "best_throughput_tok_s": (
                max(r.throughput_tok_s for r in results if r.throughput_tok_s > 0)
                if any(r.throughput_tok_s > 0 for r in results)
                else 0
            ),
            "best_compression_ratio": max(r.compression_ratio for r in results),
            "min_ttft_ms": min(r.ttft_ms for r in results),
        }
        logger.log_summary(summary)

    logger.finish()
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
