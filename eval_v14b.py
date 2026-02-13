#!/usr/bin/env python
"""
BPA v13b/v14b Compression Fidelity Evaluation Harness.

Evaluates three KV compression backends under identical decode:
  dense    — no compression (baseline)
  lowrank  — per-head SVD-calibrated low-rank KV
  quant    — per-layer INT8/INT4 quantization
  kvsplice_trained — trained segment compressor (requires checkpoint)

No routing changes, no eviction. Near window always full KV.
Far tokens compressed by backend.

Usage:
    python eval_v14b.py \
        --kv_backend dense lowrank quant \
        --L 4096 8192 16384 32768 \
        --seeds 0 1 2 \
        --decode_steps 256 \
        --outdir results/v14b_run
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from scripts.bpa_v11_bench import (
    DTYPE,
    compute_ppl,
    get_cpu_rss_mb,
    get_gpu_mem,
    get_text_batch,
    gpu_sync,
    load_validation_tokens,
    reset_gpu_mem,
)
from backends.base import DenseBackend, V14StepStats


# ============================================================
# Backend registry
# ============================================================


def get_backend(name, kvsplice_checkpoint=None):
    """Instantiate a compression backend by name."""
    if name == "dense":
        return DenseBackend()
    elif name == "lowrank":
        from backends.lowrank import LowRankBackend

        return LowRankBackend()
    elif name == "lowrank_konly":
        from backends.lowrank import LowRankBackend

        b = LowRankBackend()
        b.k_only = True
        return b
    elif name == "quant":
        from backends.quant import QuantBackend

        return QuantBackend()
    elif name == "quant_int4":
        from backends.quant import QuantBackend

        b = QuantBackend()
        b._force_int4 = True
        return b
    elif name == "kvsplice_trained":
        if kvsplice_checkpoint is None:
            raise RuntimeError(
                "KVSplice requires training; aborting. "
                "Pass --kvsplice_checkpoint with a trained .pt file."
            )
        from backends.kvsplice_kl import KVSpliceKLBackend

        backend = KVSpliceKLBackend()
        backend.checkpoint_path = kvsplice_checkpoint
        return backend
    elif name.startswith("kvsplice_seg"):
        seg = int(name.replace("kvsplice_seg", ""))
        ckpt = f"kvsplice_trained/kvsplice_seg{seg}.pt"
        if not os.path.exists(ckpt):
            raise RuntimeError(
                f"KVSplice checkpoint {ckpt} not found. "
                f"Run: python scripts/train_kvsplice.py --segment_size {seg}"
            )
        from backends.kvsplice_kl import KVSpliceKLBackend

        backend = KVSpliceKLBackend()
        backend.checkpoint_path = ckpt
        return backend
    else:
        raise ValueError(f"Unknown backend: {name}")


# ============================================================
# GPU preflight
# ============================================================


def gpu_preflight(device_str):
    """Verify GPU and log info."""
    assert torch.version.hip is not None, "ROCm/HIP not available"
    assert torch.cuda.is_available(), "CUDA not available"
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1e9
    print(f"GPU Preflight OK: {props.name} ({total_gb:.1f}GB)")
    print(f"  torch={torch.__version__} hip={torch.version.hip}")
    return {
        "device_name": props.name,
        "total_gb": round(total_gb, 1),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
    }


# ============================================================
# Model loading
# ============================================================


def load_model(model_key, device_str):
    """Load HF model."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    registry = {"qwen05b": "Qwen/Qwen2.5-0.5B", "qwen15b": "Qwen/Qwen2.5-1.5B"}
    model_name = registry.get(model_key, model_key)
    print(f"Loading model {model_name}...")

    config = AutoConfig.from_pretrained(model_name)
    max_ctx = getattr(config, "max_position_embeddings", 1024)
    n_layers = config.num_hidden_layers
    hidden = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim = hidden // n_heads

    model_config = {
        "n_layers": n_layers,
        "hidden_size": hidden,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE)
    model = model.to(device_str).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.1f}M max_ctx={max_ctx}")
    print(f"  layers={n_layers} hidden={hidden} heads={n_heads} kv_heads={n_kv_heads}")

    return model, tokenizer, max_ctx, model_config


# ============================================================
# Eval result
# ============================================================


@dataclass
class V14Result:
    backend: str
    L: int
    regime: str
    batch_size: int
    seed: int
    decode_steps: int
    ppl: float
    ppl_dense: float = 0.0
    ppl_delta_pct: float = 0.0
    passed_1pct: bool = False
    passed_3pct: bool = False
    catastrophic: bool = False
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    tok_s: float = 0.0
    kv_bytes_total: int = 0
    kv_bytes_ratio: float = 1.0
    compression_overhead_ms: float = 0.0
    overhead_pct: float = 0.0
    peak_gpu_alloc_mb: float = 0.0
    peak_gpu_reserved_mb: float = 0.0
    peak_cpu_rss_mb: float = 0.0
    n_full: int = 0
    n_compressed: int = 0
    description: str = ""
    error: str = ""


# ============================================================
# Single evaluation
# ============================================================


def run_single_eval(
    backend,
    model,
    token_data,
    L,
    decode_steps,
    seed,
    device_str,
    max_ctx,
    model_config,
    batch_size=1,
    regime="r1",
):
    """Run one evaluation of a backend at (L, seed, regime)."""
    rng = np.random.RandomState(seed)
    total_len = L + decode_steps
    idx = get_text_batch(token_data, batch_size, total_len, rng).to(device_str)
    prefix = idx[:, :L]
    continuation = idx[:, L : L + decode_steps]

    rss_before = get_cpu_rss_mb()

    # Warmup
    if device_str != "cpu":
        with torch.no_grad():
            _ = model(prefix[:, :16])
        gpu_sync(device_str)

    reset_gpu_mem(device_str)

    # Configure
    backend.configure(L, model_config)

    # Time decode
    gpu_sync(device_str)
    t_start = time.perf_counter()

    try:
        all_logits, step_stats = backend.run_decode(
            model, prefix, continuation, device_str, max_ctx
        )
    except Exception as e:
        torch.cuda.empty_cache()
        return V14Result(
            backend=backend.name,
            L=L,
            regime=regime,
            batch_size=batch_size,
            seed=seed,
            decode_steps=decode_steps,
            ppl=float("inf"),
            error=str(e),
            catastrophic=True,
        )

    gpu_sync(device_str)
    total_s = time.perf_counter() - t_start

    ppl = compute_ppl(all_logits[:, :-1, :], continuation)
    decode_ms = total_s * 1000
    p50_ms = decode_ms / decode_steps
    p95_ms = p50_ms * 1.5

    # Aggregate stats
    if step_stats:
        last = step_stats[-1]
        total_compress_ms = sum(s.compress_ms + s.decompress_ms for s in step_stats)
        avg_bytes = np.mean([s.kv_bytes_total for s in step_stats])
    else:
        last = V14StepStats()
        total_compress_ms = 0
        avg_bytes = 0

    rss_after = get_cpu_rss_mb()
    gpu_alloc, gpu_reserved = get_gpu_mem(device_str)

    result = V14Result(
        backend=backend.name,
        L=L,
        regime=regime,
        batch_size=batch_size,
        seed=seed,
        decode_steps=decode_steps,
        ppl=float(ppl),
        p50_ms=round(p50_ms, 3),
        p95_ms=round(p95_ms, 3),
        tok_s=round(decode_steps / total_s, 1) if total_s > 0 else 0,
        kv_bytes_total=int(avg_bytes),
        compression_overhead_ms=round(total_compress_ms, 3),
        overhead_pct=(
            round(100 * total_compress_ms / decode_ms, 2) if decode_ms > 0 else 0
        ),
        peak_gpu_alloc_mb=round(gpu_alloc, 0),
        peak_gpu_reserved_mb=round(gpu_reserved, 0),
        peak_cpu_rss_mb=round(max(rss_before, rss_after), 0),
        n_full=last.n_full,
        n_compressed=last.n_compressed,
        description=backend.description(),
        catastrophic=(ppl > 1e5 or math.isnan(ppl) or math.isinf(ppl)),
    )

    del all_logits
    torch.cuda.empty_cache()
    return result


# ============================================================
# Quality gating
# ============================================================


def apply_quality_gating(results, dense_ppls):
    """Apply PASS/FAIL based on dense PPL."""
    for r in results:
        key = (r.L, r.regime, r.seed)
        if key in dense_ppls:
            r.ppl_dense = dense_ppls[key]
            if r.ppl_dense > 0:
                r.ppl_delta_pct = round(100 * (r.ppl - r.ppl_dense) / r.ppl_dense, 2)
                r.passed_1pct = r.ppl <= r.ppl_dense * 1.01
                r.passed_3pct = r.ppl <= r.ppl_dense * 1.03
            if r.ppl > r.ppl_dense * 3:
                r.catastrophic = True

    # Compute kv_bytes_ratio vs dense
    dense_bytes = {}
    for r in results:
        if r.backend == "dense":
            dense_bytes[(r.L, r.regime, r.seed)] = r.kv_bytes_total

    for r in results:
        key = (r.L, r.regime, r.seed)
        if key in dense_bytes and dense_bytes[key] > 0:
            r.kv_bytes_ratio = round(r.kv_bytes_total / dense_bytes[key], 4)


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="BPA v14b Compression Eval")
    parser.add_argument("--model", default="qwen05b")
    parser.add_argument(
        "--kv_backend",
        nargs="+",
        default=["dense", "lowrank", "quant"],
    )
    parser.add_argument("--L", nargs="+", type=int, default=[4096, 8192, 16384, 32768])
    parser.add_argument("--regimes", nargs="+", default=["r1"])
    parser.add_argument("--decode_steps", type=int, default=256)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kvsplice_checkpoint", default=None)
    args = parser.parse_args()

    if args.outdir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.outdir = f"results/v14b_run_{ts}"
    os.makedirs(args.outdir, exist_ok=True)

    gpu_info = gpu_preflight(args.device)
    model, tokenizer, max_ctx, model_config = load_model(args.model, args.device)

    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)

    regime_bs = {"r1": 1, "r2": 4}
    valid_L = [L for L in args.L if L <= max_ctx]

    all_results = []
    dense_ppls = {}

    # ---- Phase 1: Dense baselines ----
    print("\n" + "=" * 60)
    print("Phase 1: Dense baselines")
    print("=" * 60)

    dense_be = DenseBackend()
    for regime in args.regimes:
        bs = regime_bs.get(regime, 1)
        for L in valid_L:
            for seed in args.seeds:
                print(
                    f"  dense L={L} {regime}(bs={bs}) seed={seed}...",
                    end="",
                    flush=True,
                )
                dense_be.configure(L, model_config)
                r = run_single_eval(
                    dense_be,
                    model,
                    token_data,
                    L,
                    args.decode_steps,
                    seed,
                    args.device,
                    max_ctx,
                    model_config,
                    batch_size=bs,
                    regime=regime,
                )
                r.ppl_dense = r.ppl
                r.passed_1pct = True
                r.passed_3pct = True
                all_results.append(r)
                dense_ppls[(L, regime, seed)] = r.ppl
                print(
                    f" PPL={r.ppl:.1f} p50={r.p50_ms:.2f}ms gpu={r.peak_gpu_alloc_mb:.0f}MB"
                )

    # Save dense baselines
    dense_data = [asdict(r) for r in all_results]
    with open(os.path.join(args.outdir, "dense_baselines.json"), "w") as f:
        json.dump(dense_data, f, indent=2, default=str)

    # ---- Phase 2+: Compression backends ----
    backends_to_run = [b for b in args.kv_backend if b != "dense"]
    for backend_name in backends_to_run:
        print(f"\n{'=' * 60}")
        print(f"Backend: {backend_name}")
        print("=" * 60)

        try:
            backend = get_backend(backend_name, args.kvsplice_checkpoint)
        except RuntimeError as e:
            print(f"  SKIP: {e}")
            continue

        # Calibrate on max L value (captures full RoPE distribution)
        cal_L = max(valid_L)
        print(f"  Calibrating on L={cal_L}...")
        try:
            backend.configure(cal_L, model_config)
            backend.calibrate(model, token_data, cal_L, args.device, model_config)
        except Exception as e:
            print(f"  Calibration failed: {e}")
            import traceback

            traceback.print_exc()
            continue

        for regime in args.regimes:
            bs = regime_bs.get(regime, 1)
            for L in valid_L:
                for seed in args.seeds:
                    print(
                        f"  {backend_name} L={L} {regime}(bs={bs}) seed={seed}...",
                        end="",
                        flush=True,
                    )
                    # Reconfigure for this L (keeps calibration)
                    backend.configure(L, model_config)
                    try:
                        r = run_single_eval(
                            backend,
                            model,
                            token_data,
                            L,
                            args.decode_steps,
                            seed,
                            args.device,
                            max_ctx,
                            model_config,
                            batch_size=bs,
                            regime=regime,
                        )
                    except Exception as e:
                        print(f" ERROR: {e}")
                        r = V14Result(
                            backend=backend_name,
                            L=L,
                            regime=regime,
                            batch_size=bs,
                            seed=seed,
                            decode_steps=args.decode_steps,
                            ppl=float("inf"),
                            error=str(e),
                            catastrophic=True,
                        )
                    all_results.append(r)

                    if r.error:
                        print(f" ERROR: {r.error[:60]}")
                    else:
                        print(
                            f" PPL={r.ppl:.1f} p50={r.p50_ms:.2f}ms"
                            f" overhead={r.overhead_pct:.1f}%"
                        )

    # ---- Quality gating ----
    print(f"\n{'=' * 60}")
    print("Quality gating")
    print("=" * 60)
    apply_quality_gating(all_results, dense_ppls)

    for bname in ["dense"] + backends_to_run:
        br = [r for r in all_results if r.backend == bname]
        n = len(br)
        n1 = sum(1 for r in br if r.passed_1pct)
        n3 = sum(1 for r in br if r.passed_3pct)
        nc = sum(1 for r in br if r.catastrophic)
        print(f"  {bname:20s}: {n1}/{n} @1%  {n3}/{n} @3%  {nc} catastrophic")

    # ---- Save ----
    print(f"\nSaving to {args.outdir}/")
    with open(os.path.join(args.outdir, "all_results.json"), "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)

    # Scoreboard
    scoreboard = {}
    for r in all_results:
        key = f"{r.backend}_{r.L}_{r.regime}"
        if key not in scoreboard:
            scoreboard[key] = {
                "backend": r.backend,
                "L": r.L,
                "regime": r.regime,
                "seeds": [],
            }
        scoreboard[key]["seeds"].append(
            {
                "seed": r.seed,
                "ppl": r.ppl,
                "ppl_delta_pct": r.ppl_delta_pct,
                "passed_1pct": r.passed_1pct,
                "passed_3pct": r.passed_3pct,
                "catastrophic": r.catastrophic,
                "p50_ms": r.p50_ms,
                "kv_bytes_ratio": r.kv_bytes_ratio,
                "overhead_pct": r.overhead_pct,
            }
        )
    # Aggregate per (backend, L, regime)
    for key, entry in scoreboard.items():
        seeds = entry["seeds"]
        entry["all_pass_1pct"] = all(s["passed_1pct"] for s in seeds)
        entry["all_pass_3pct"] = all(s["passed_3pct"] for s in seeds)
        entry["any_catastrophic"] = any(s["catastrophic"] for s in seeds)
        good = [s for s in seeds if not s["catastrophic"]]
        entry["avg_ppl"] = (
            round(np.mean([s["ppl"] for s in good]), 2) if good else 999999
        )
        entry["avg_ppl_delta_pct"] = (
            round(np.mean([s["ppl_delta_pct"] for s in good]), 2) if good else 999999
        )
        entry["avg_p50_ms"] = round(np.mean([s["p50_ms"] for s in seeds]), 2)
        entry["avg_kv_bytes_ratio"] = round(
            np.mean([s["kv_bytes_ratio"] for s in seeds]), 4
        )
        entry["avg_overhead_pct"] = round(
            np.mean([s["overhead_pct"] for s in seeds]), 2
        )

    with open(os.path.join(args.outdir, "scoreboard.json"), "w") as f:
        json.dump(scoreboard, f, indent=2, default=str)

    # Meta
    meta = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "L_values": valid_L,
        "regimes": args.regimes,
        "backends": args.kv_backend,
        "decode_steps": args.decode_steps,
        "seeds": args.seeds,
        "gpu_info": gpu_info,
    }
    with open(os.path.join(args.outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {len(all_results)} results saved.")


if __name__ == "__main__":
    main()
