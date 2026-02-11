#!/usr/bin/env python
"""
BPA v11: Model probe — find a suitable long-context model for W7900.

Tries candidate HF models, runs smoke tests at multiple L values,
records peak memory and throughput. Writes model_probe_results.json.

Usage:
    python scripts/bpa_v11_model_probe.py [--device cuda]
"""

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np
import psutil
import torch


def gpu_preflight():
    """Phase 0.1: strict GPU preflight. Fail fast if anything is wrong."""
    info = {}
    info["torch_version"] = torch.__version__
    info["hip_version"] = getattr(torch.version, "hip", None)
    info["cuda_available"] = torch.cuda.is_available()

    if not torch.cuda.is_available():
        print("FATAL: torch.cuda.is_available() == False")
        sys.exit(1)

    info["device_name"] = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    info["total_memory_gb"] = round(props.total_memory / 1e9, 1)
    info["device_count"] = torch.cuda.device_count()
    info["hostname"] = socket.gethostname()
    info["cpu"] = platform.processor() or platform.machine()

    print("=== GPU Preflight ===")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Shell checks
    for cmd, desc in [
        ("rocminfo 2>/dev/null | head -n 40", "rocminfo"),
        (
            "rocm-smi --showproductname --showuse --showmemuse 2>/dev/null",
            "rocm-smi",
        ),
    ]:
        try:
            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            if r.returncode == 0 and r.stdout.strip():
                info[desc] = r.stdout.strip()[:500]
                print(f"  {desc}: OK")
            else:
                print(f"  {desc}: not available (rc={r.returncode})")
        except Exception as e:
            print(f"  {desc}: error ({e})")

    print("=== Preflight OK ===\n")
    return info


def detect_max_ctx(config):
    """Phase 0.2: detect max context from HF model config."""
    for attr in [
        "max_position_embeddings",
        "n_positions",
        "max_seq_len",
    ]:
        val = getattr(config, attr, None)
        if val is not None:
            return int(val)
    return None


def probe_model(model_name, device, test_lengths):
    """Try loading a model and running decode at various L values."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    result = {
        "model": model_name,
        "status": "unknown",
        "max_ctx": None,
        "params_m": None,
        "tests": [],
    }

    # Load config first
    try:
        config = AutoConfig.from_pretrained(model_name)
        max_ctx = detect_max_ctx(config)
        result["max_ctx"] = max_ctx
        n_layers = getattr(config, "num_hidden_layers", None)
        hidden = getattr(config, "hidden_size", None)
        n_heads = getattr(config, "num_attention_heads", None)
        n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
        result["config"] = {
            "n_layers": n_layers,
            "hidden_size": hidden,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "model_type": getattr(config, "model_type", "unknown"),
        }
    except Exception as e:
        result["status"] = f"config_error: {e}"
        return result

    # Load model
    try:
        print(f"  Loading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
        params = sum(p.numel() for p in model.parameters())
        result["params_m"] = round(params / 1e6, 1)
        model = model.to(device)
        model.eval()
        print(f"  Loaded: {result['params_m']}M params")
    except Exception as e:
        result["status"] = f"load_error: {e}"
        return result

    # Try tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        vocab_size = tokenizer.vocab_size
    except Exception:
        vocab_size = getattr(config, "vocab_size", 50257)

    # Smoke test at each L
    for L in test_lengths:
        if max_ctx is not None and L > max_ctx:
            result["tests"].append({"L": L, "status": "skip_exceeds_max_ctx"})
            continue

        test_entry = {"L": L, "status": "unknown"}
        try:
            torch.cuda.reset_peak_memory_stats()
            input_ids = torch.randint(0, vocab_size, (1, L), device=device)

            # Prefill
            with torch.no_grad():
                out = model(input_ids, use_cache=True)
                past = out.past_key_values
                next_tok = out.logits[:, -1:].argmax(dim=-1)

            # Decode 64 tokens
            times = []
            for _ in range(64):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model(next_tok, past_key_values=past, use_cache=True)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000
                times.append(dt)
                past = out.past_key_values
                next_tok = out.logits[:, -1:].argmax(dim=-1)

            peak_gpu = torch.cuda.max_memory_allocated() / 1e6
            test_entry["status"] = "OK"
            test_entry["p50_ms"] = round(float(np.median(times)), 2)
            test_entry["p95_ms"] = round(float(np.percentile(times, 95)), 2)
            test_entry["tok_s"] = round(1000 / float(np.median(times)), 0)
            test_entry["peak_gpu_mb"] = round(peak_gpu, 0)
            print(
                f"  L={L}: p50={test_entry['p50_ms']:.1f}ms "
                f"tok/s={test_entry['tok_s']:.0f} "
                f"peak={test_entry['peak_gpu_mb']:.0f}MB"
            )
            del past, out, input_ids
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            test_entry["status"] = "OOM"
            print(f"  L={L}: OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            test_entry["status"] = f"error: {str(e)[:200]}"
            print(f"  L={L}: error: {e}")
            torch.cuda.empty_cache()

        result["tests"].append(test_entry)

    # Check if all tests pass
    ok_tests = [t for t in result["tests"] if t["status"] == "OK"]
    if ok_tests:
        result["status"] = "OK"
    else:
        result["status"] = "all_tests_failed"

    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="BPA v11 Model Probe")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="bpa_v11_results/model_probe_results.json")
    args = parser.parse_args()

    preflight = gpu_preflight()

    candidates = [
        "Qwen/Qwen2.5-0.5B",
        "meta-llama/Llama-3.2-1B-Instruct",
    ]

    test_lengths = [512, 1024, 2048, 4096, 8192]

    all_results = []
    for name in candidates:
        print(f"\n--- Probing {name} ---")
        r = probe_model(name, args.device, test_lengths)
        all_results.append(r)

    # Rank: prefer models with most OK tests, then lowest p50 at L=4096
    def rank_key(r):
        ok_count = sum(1 for t in r["tests"] if t["status"] == "OK")
        p50_4k = 999
        for t in r["tests"]:
            if t.get("L") == 4096 and t["status"] == "OK":
                p50_4k = t["p50_ms"]
        return (-ok_count, p50_4k)

    all_results.sort(key=rank_key)

    print("\n=== RANKED RESULTS ===")
    for i, r in enumerate(all_results):
        ok = sum(1 for t in r["tests"] if t["status"] == "OK")
        print(
            f"  #{i+1} {r['model']}: {r['status']} "
            f"({ok}/{len(r['tests'])} OK) "
            f"max_ctx={r['max_ctx']} params={r['params_m']}M"
        )
        for t in r["tests"]:
            extra = ""
            if t["status"] == "OK":
                extra = (
                    f" p50={t['p50_ms']}ms tok/s={t['tok_s']}"
                    f" peak={t['peak_gpu_mb']}MB"
                )
            print(f"    L={t['L']}: {t['status']}{extra}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "preflight": preflight,
        "candidates": all_results,
        "recommendation": all_results[0]["model"] if all_results else None,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")
    print(f"Recommendation: {output['recommendation']}")


if __name__ == "__main__":
    main()
