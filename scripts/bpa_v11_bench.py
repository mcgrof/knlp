#!/usr/bin/env python
"""
BPA v11: Decode benchmark for HuggingFace models with long context.

Uses an off-the-shelf HF model (default: Qwen2.5-0.5B, max_ctx=32768)
to test BPA at L in {512, 1024, 2048, 4096, 8192} — all in-range.

BPA is implemented as a KV cache eviction layer: after each decode step,
the controller decides which KV entries to keep (local window + far chunks)
and prunes the rest from past_key_values. Dense baseline keeps all KV.

Usage:
    # Benchmark
    python scripts/bpa_v11_bench.py bench --L 512,1024,2048,4096 \\
        --steps 256 --seeds 1,2,3 --device cuda

    # Matched-quality tuning
    python scripts/bpa_v11_bench.py tune --L 512,1024,2048,4096 \\
        --tol 1,3 --steps 256 --device cuda

    # Stress tests
    python scripts/bpa_v11_bench.py stress --L 512,1024,2048,4096 \\
        --steps 256 --seeds 1,2 --device cuda
"""

import argparse
import csv
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Constants
# ============================================================

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DTYPE = torch.float16
CHUNK_SIZE = 64  # chunk granularity for far selection

# ============================================================
# GPU preflight
# ============================================================


def gpu_preflight(device_str):
    """Strict GPU preflight. Fail fast."""
    if device_str == "cpu":
        return {}
    if not torch.cuda.is_available():
        print("FATAL: device=cuda but torch.cuda.is_available() == False")
        sys.exit(1)
    info = {
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        ),
        "device_count": torch.cuda.device_count(),
    }
    print("GPU Preflight OK:", info["device_name"], f"({info['total_memory_gb']}GB)")
    return info


# ============================================================
# Run metadata
# ============================================================


def get_git_sha():
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def build_run_meta(device_str, model_name, max_ctx, model_config):
    meta = {
        "git_sha": get_git_sha(),
        "hostname": socket.gethostname(),
        "cpu": platform.processor() or platform.machine(),
        "torch_version": torch.__version__,
        "dtype": str(DTYPE),
        "device": device_str,
        "model_name": model_name,
        "max_ctx": max_ctx,
        "model_config": model_config,
    }
    if device_str != "cpu" and torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["gpu_count"] = torch.cuda.device_count()
        meta["gpu_mem_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
    return meta


def context_regime(L, max_ctx):
    return "in_range" if L <= max_ctx else "extrapolated"


# ============================================================
# Memory helpers
# ============================================================


def get_cpu_rss_mb():
    return psutil.Process().memory_info().rss / 1e6


def get_gpu_mem(device_str):
    if device_str == "cpu" or not torch.cuda.is_available():
        return 0.0, 0.0
    alloc = torch.cuda.max_memory_allocated() / 1e6
    reserved = torch.cuda.max_memory_reserved() / 1e6
    return alloc, reserved


def reset_gpu_mem(device_str):
    if device_str != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def gpu_sync(device_str):
    if device_str != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================
# Model loading
# ============================================================


def load_hf_model(model_name, device_str):
    """Load an HF model and detect its configuration."""
    config = AutoConfig.from_pretrained(model_name)
    max_ctx = None
    for attr in ["max_position_embeddings", "n_positions"]:
        val = getattr(config, attr, None)
        if val is not None:
            max_ctx = int(val)
            break

    n_layers = getattr(config, "num_hidden_layers", None)
    hidden = getattr(config, "hidden_size", None)
    n_heads = getattr(config, "num_attention_heads", None)
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim = hidden // n_heads if hidden and n_heads else None

    model_config = {
        "n_layers": n_layers,
        "hidden_size": hidden,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "model_type": getattr(config, "model_type", "unknown"),
        "vocab_size": getattr(config, "vocab_size", 50257),
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE)
    model = model.to(device_str)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}")
    print(f"  Params: {params/1e6:.1f}M")
    print(f"  max_ctx: {max_ctx}")
    print(f"  layers={n_layers} hidden={hidden} heads={n_heads} kv_heads={n_kv_heads}")

    return model, tokenizer, max_ctx, model_config


# ============================================================
# Data loading
# ============================================================


def load_validation_tokens(tokenizer, n_tokens=500000):
    """Load validation tokens from wikitext or generate random ones."""
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        text = "\n\n".join(ds["text"])
        tokens = tokenizer.encode(text)
        if len(tokens) >= n_tokens:
            return np.array(tokens[:n_tokens], dtype=np.int64)
    except Exception:
        pass

    # Fallback: use random tokens from vocab
    print("  Using random tokens as validation data")
    rng = np.random.RandomState(42)
    return rng.randint(0, tokenizer.vocab_size, size=n_tokens).astype(np.int64)


def get_text_batch(token_data, batch_size, seq_len, rng):
    """Get a batch of token sequences from the data."""
    max_start = len(token_data) - seq_len
    if max_start <= 0:
        raise ValueError(f"Token data too short: {len(token_data)} < {seq_len}")
    starts = rng.randint(0, max_start, size=batch_size)
    batch = np.stack([token_data[s : s + seq_len] for s in starts])
    return torch.from_numpy(batch)


# ============================================================
# KV cache manipulation for BPA
# ============================================================


def kv_cache_len(past_key_values):
    """Get sequence length from past_key_values."""
    if past_key_values is None:
        return 0
    # past_key_values is tuple of (key, value) per layer
    # key shape: [batch, n_kv_heads, seq_len, head_dim]
    return past_key_values[0][0].shape[2]


def evict_kv_cache(past_key_values, keep_mask):
    """Evict KV cache entries not in keep_mask.

    Args:
        past_key_values: tuple of (key, value) per layer
        keep_mask: bool tensor [seq_len] of positions to keep

    Returns:
        new past_key_values with evicted entries removed
    """
    # keep_mask is [seq_len], expand to [1, 1, seq_len, 1] for gather
    indices = keep_mask.nonzero(as_tuple=True)[0]  # [n_kept]
    new_past = []
    for k, v in past_key_values:
        # k, v: [batch, n_kv_heads, seq_len, head_dim]
        k_new = k[:, :, indices, :]
        v_new = v[:, :, indices, :]
        new_past.append((k_new, v_new))
    return tuple(new_past)


def build_keep_mask(total_len, local_window, far_chunks, chunk_size):
    """Build a boolean mask of positions to keep in KV cache.

    Keeps: local window (last W tokens) + selected far chunks.
    """
    mask = torch.zeros(total_len, dtype=torch.bool)
    # Local window
    local_start = max(0, total_len - local_window)
    mask[local_start:] = True
    # Far chunks
    far_end = max(0, total_len - local_window)
    for c in far_chunks:
        c_start = c * chunk_size
        c_end = min((c + 1) * chunk_size, far_end)
        if c_start < far_end:
            mask[c_start:c_end] = True
    return mask


def select_far_chunks_random(n_chunks, far_end_chunk, k_far, rng):
    """Select k_far random chunks from the far region."""
    if far_end_chunk <= 0 or k_far <= 0:
        return []
    available = list(range(far_end_chunk))
    k = min(k_far, len(available))
    return sorted(rng.choice(available, size=k, replace=False).tolist())


# ============================================================
# Adaptive controller (same as v10 with W duty cycle logging)
# ============================================================


class AdaptiveController:
    """Controls local window W(t) and far budget k_far(t) per step."""

    def __init__(
        self,
        W_min=64,
        W_max=512,
        W_decay=0.95,
        W_ramp_factor=2.0,
        W_pressure_thresh=0.45,
        B_far_max=8,
        B_far_target=2.0,
        B_far_scale=8.0,
        B_far_bias=0.0,
        ema_alpha=0.2,
        hysteresis=0.3,
        gate_every_k=4,
        pi_kp=0.02,
        pi_ki=0.002,
        hyst_persist=2,
    ):
        self.W_min = W_min
        self.W_max = W_max
        self.W_decay = W_decay
        self.W_ramp_factor = W_ramp_factor
        self.W_pressure_thresh = W_pressure_thresh
        self.B_far_max = B_far_max
        self.B_far_target = B_far_target
        self.B_far_scale = B_far_scale
        self.B_far_bias = B_far_bias
        self.ema_alpha = ema_alpha
        self.hysteresis = hysteresis
        self.gate_every_k = gate_every_k
        self.pi_kp = pi_kp
        self.pi_ki = pi_ki
        self.hyst_persist = hyst_persist

        self._pending_k_far = None
        self._persist_count = 0

        self.reset()

    def reset(self):
        self.W_current = float(self.W_min)
        self.B_far_raw = float(self.B_far_target)
        self.k_far_current = int(round(self.B_far_target))
        self.pressure_ema = 0.0
        self.above_thresh = False
        self.B_far_cumsum = 0.0
        self.n_steps = 0
        self.pi_integral = 0.0
        self._pending_k_far = None
        self._persist_count = 0
        self.W_log = []
        self.B_far_raw_log = []
        self.k_far_log = []
        self.pressure_log = []
        self.entropy_log = []

    def compute_pressure(self, logits_t, residual_norm):
        """Compute pressure from logits entropy + residual norm."""
        probs = F.softmax(logits_t.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        max_entropy = math.log(probs.shape[-1])
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        norm_resid = min(residual_norm / 100.0, 1.0)
        pressure = 0.7 * norm_entropy + 0.3 * norm_resid
        self.entropy_log.append(entropy)
        return pressure

    def step(self, pressure):
        """Update W and k_far based on pressure."""
        self.pressure_ema = (
            self.ema_alpha * pressure + (1 - self.ema_alpha) * self.pressure_ema
        )

        if self.above_thresh:
            trigger = self.pressure_ema > (self.W_pressure_thresh - self.hysteresis)
        else:
            trigger = self.pressure_ema > self.W_pressure_thresh
        self.above_thresh = trigger

        if trigger:
            self.W_current = min(self.W_current * self.W_ramp_factor, self.W_max)
        else:
            self.W_current = max(self.W_current * self.W_decay, self.W_min)

        W_int = int(round(self.W_current))
        W_int = max(self.W_min, min(self.W_max, W_int))

        # Adaptive B_far: raw pressure with fast EMA
        raw_B = self.B_far_scale * pressure + self.B_far_bias
        raw_B = max(0.0, min(float(self.B_far_max), raw_B))
        b_alpha = 0.4
        self.B_far_raw = b_alpha * raw_B + (1 - b_alpha) * self.B_far_raw

        # PI governor
        self.n_steps += 1
        avg_k = self.B_far_cumsum / self.n_steps if self.n_steps > 0 else 0
        error = avg_k - self.B_far_target
        self.pi_integral += error
        pi_correction = self.pi_kp * error + self.pi_ki * self.pi_integral
        adjusted_B = self.B_far_raw - pi_correction
        adjusted_B = max(0.0, min(float(self.B_far_max), adjusted_B))

        # Unbiased rounding
        floor_B = int(math.floor(adjusted_B))
        frac = adjusted_B - floor_B
        candidate_k = floor_B + (1 if frac >= 0.5 else 0)
        candidate_k = max(0, min(self.B_far_max, candidate_k))

        # Hysteresis persistence
        if candidate_k != self.k_far_current:
            if self._pending_k_far == candidate_k:
                self._persist_count += 1
            else:
                self._pending_k_far = candidate_k
                self._persist_count = 1
            if self._persist_count >= self.hyst_persist:
                self.k_far_current = candidate_k
                self._pending_k_far = None
                self._persist_count = 0
        else:
            self._pending_k_far = None
            self._persist_count = 0

        self.B_far_cumsum += self.k_far_current

        self.W_log.append(W_int)
        self.B_far_raw_log.append(float(self.B_far_raw))
        self.k_far_log.append(self.k_far_current)
        self.pressure_log.append(float(self.pressure_ema))

        return W_int, self.k_far_current

    def get_summary(self):
        return {
            "W_mean": float(np.mean(self.W_log)) if self.W_log else 0,
            "W_min_obs": int(min(self.W_log)) if self.W_log else 0,
            "W_max_obs": int(max(self.W_log)) if self.W_log else 0,
            "W_std": float(np.std(self.W_log)) if self.W_log else 0,
            "B_far_raw_mean": (
                float(np.mean(self.B_far_raw_log)) if self.B_far_raw_log else 0
            ),
            "k_far_mean": float(np.mean(self.k_far_log)) if self.k_far_log else 0,
            "k_far_min": int(min(self.k_far_log)) if self.k_far_log else 0,
            "k_far_max": int(max(self.k_far_log)) if self.k_far_log else 0,
            "k_far_std": float(np.std(self.k_far_log)) if self.k_far_log else 0,
            "pressure_mean": (
                float(np.mean(self.pressure_log)) if self.pressure_log else 0
            ),
            "entropy_mean": (
                float(np.mean(self.entropy_log)) if self.entropy_log else 0
            ),
            "n_steps": self.n_steps,
        }

    def get_w_duty_cycle(self):
        """Return distribution of W values as percentage of time at each."""
        if not self.W_log:
            return {}
        from collections import Counter

        counts = Counter(self.W_log)
        total = len(self.W_log)
        return {str(w): round(c / total * 100, 1) for w, c in sorted(counts.items())}


# ============================================================
# Stress test generators
# ============================================================


def make_late_binding_sequence(token_data, seq_len, rng):
    """Key at 0.2*L, reference at 0.8*L."""
    offset = rng.randint(0, len(token_data) - seq_len * 2)
    tokens = token_data[offset : offset + seq_len].copy()
    key_pos = int(0.2 * seq_len)
    ref_pos = int(0.8 * seq_len)
    tokens[ref_pos : ref_pos + 16] = tokens[key_pos : key_pos + 16]
    return torch.from_numpy(tokens.astype(np.int64)).unsqueeze(0)


def make_kv_retrieval_sequence(token_data, seq_len, rng):
    """8 key-value pairs early, query for first pair late."""
    offset = rng.randint(0, len(token_data) - seq_len * 2)
    tokens = token_data[offset : offset + seq_len].copy()
    n_pairs = 8
    pair_len = 32
    for i in range(n_pairs):
        start = i * pair_len + 16
        if start + pair_len < seq_len // 2:
            src_start = rng.randint(0, len(token_data) - pair_len)
            tokens[start : start + pair_len] = token_data[
                src_start : src_start + pair_len
            ]
    query_pos = int(0.85 * seq_len)
    tokens[query_pos : query_pos + 16] = tokens[16 : 16 + 16]
    return torch.from_numpy(tokens.astype(np.int64)).unsqueeze(0)


def make_forced_far_sequence(token_data, seq_len, rng):
    """Force far retrieval: key at 16, distractors in middle, ref at 0.9*L."""
    offset = rng.randint(0, len(token_data) - seq_len * 2)
    tokens = token_data[offset : offset + seq_len].copy()
    key_chunk = tokens[16 : 16 + 32].copy()
    for i in range(10):
        dist_pos = int(0.1 * seq_len) + i * int(0.07 * seq_len)
        if dist_pos + 32 < int(0.85 * seq_len):
            src = rng.randint(0, len(token_data) - 32)
            tokens[dist_pos : dist_pos + 32] = token_data[src : src + 32]
    ref_pos = int(0.9 * seq_len)
    tokens[ref_pos : ref_pos + 32] = key_chunk
    return torch.from_numpy(tokens.astype(np.int64)).unsqueeze(0)


STRESS_GENERATORS = {
    "late_binding": make_late_binding_sequence,
    "kv_retrieval": make_kv_retrieval_sequence,
    "forced_far": make_forced_far_sequence,
}

STRESS_SPANS = {
    "late_binding": {"easy": (0.3, 0.6), "hard": (0.75, 1.0)},
    "kv_retrieval": {"easy": (0.3, 0.6), "hard": (0.8, 1.0)},
    "forced_far": {"easy": (0.3, 0.7), "hard": (0.85, 1.0)},
}


# ============================================================
# PPL computation
# ============================================================


def compute_ppl(logits, targets):
    """Compute perplexity from logits and targets."""
    # logits: [B, T, V], targets: [B, T]
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(-1, V).float(), targets.reshape(-1), reduction="mean"
    )
    return math.exp(min(loss.item(), 20))  # cap at exp(20) to avoid overflow


# ============================================================
# KV size estimation
# ============================================================


def kv_bytes_per_token(model_config):
    """Estimate KV bytes read per token for a given number of kept tokens."""
    n_layers = model_config["n_layers"]
    n_kv_heads = model_config["n_kv_heads"]
    head_dim = model_config["head_dim"]
    # 2 for K+V, 2 for fp16 bytes
    return 2 * n_layers * n_kv_heads * head_dim * 2


# ============================================================
# DecodeResult
# ============================================================


@dataclass
class DecodeResult:
    method: str
    seq_len: int
    decode_steps: int
    seed: int
    context_regime: str
    prefill_ms: float
    decode_per_token_ms: float
    decode_p95_ms: float
    gate_pct_of_total: float
    throughput_toks_per_sec: float
    ppl: float
    kv_kept_mean: float
    kv_mb_per_tok: float
    kv_ratio: float
    peak_cpu_rss_mb: float
    peak_gpu_alloc_mb: float
    peak_gpu_reserved_mb: float
    W_mean: float = 0.0
    k_far_mean: float = 0.0
    k_far_max: int = 0
    pressure_mean: float = 0.0
    stress_mode: str = "control"
    region_ppl_easy: float = 0.0
    region_ppl_hard: float = 0.0
    quality_failed: bool = False
    ppl_delta_pct: float = 0.0
    w_duty_cycle: dict = field(default_factory=dict)
    bpa_params: dict = field(default_factory=dict)
    run_meta: dict = field(default_factory=dict)
    batch_size: int = 1


# ============================================================
# Dense decode runner
# ============================================================


@torch.no_grad()
def run_dense_decode(
    model,
    token_data,
    seq_len,
    decode_steps,
    seed,
    device_str,
    model_config,
    max_ctx,
    batch_size=1,
):
    rng = np.random.RandomState(seed)
    idx = get_text_batch(token_data, batch_size, seq_len + decode_steps, rng).to(
        device_str
    )
    prefix = idx[:, :seq_len]
    continuation = idx[:, seq_len : seq_len + decode_steps]

    rss_before = get_cpu_rss_mb()

    # Warmup
    if device_str != "cpu":
        with torch.no_grad():
            _ = model(prefix[:, :16])
        gpu_sync(device_str)

    reset_gpu_mem(device_str)

    # Prefill
    t0 = time.perf_counter()
    out = model(prefix, use_cache=True)
    past = out.past_key_values
    gpu_sync(device_str)
    prefill_ms = (time.perf_counter() - t0) * 1000

    # Decode
    decode_latencies = []
    all_logits = [out.logits[:, -1:, :]]

    for step in range(decode_steps):
        next_token = continuation[:, step : step + 1]
        gpu_sync(device_str)
        t0 = time.perf_counter()
        out = model(next_token, past_key_values=past, use_cache=True)
        gpu_sync(device_str)
        dt = (time.perf_counter() - t0) * 1000
        decode_latencies.append(dt)
        past = out.past_key_values
        all_logits.append(out.logits)

    all_logits_cat = torch.cat(all_logits, dim=1)
    ppl = compute_ppl(all_logits_cat[:, :-1, :], continuation)

    # Region PPL
    half = decode_steps // 2
    if half > 1:
        ppl_easy = compute_ppl(all_logits_cat[:, :half, :], continuation[:, :half])
        ppl_hard = compute_ppl(all_logits_cat[:, half:-1, :], continuation[:, half:])
    else:
        ppl_easy = ppl_hard = ppl

    cache_len = kv_cache_len(past)
    kv_bpt = kv_bytes_per_token(model_config)
    kv_mb = cache_len * kv_bpt / 1e6

    rss_after = get_cpu_rss_mb()
    peak_cpu_rss = max(rss_after, rss_before)
    gpu_alloc, gpu_reserved = get_gpu_mem(device_str)

    assert peak_cpu_rss > 0, f"peak_cpu_rss_mb={peak_cpu_rss} must be > 0"
    if device_str != "cpu":
        assert gpu_alloc > 0, f"peak_gpu_alloc_mb={gpu_alloc} must be > 0 on CUDA"

    decode_arr = np.array(decode_latencies)
    total_decode_ms = sum(decode_latencies)

    del past, out
    torch.cuda.empty_cache()

    return DecodeResult(
        method="dense",
        seq_len=seq_len,
        decode_steps=decode_steps,
        seed=seed,
        context_regime=context_regime(seq_len, max_ctx),
        prefill_ms=prefill_ms,
        decode_per_token_ms=float(np.median(decode_arr)),
        decode_p95_ms=float(np.percentile(decode_arr, 95)),
        gate_pct_of_total=0.0,
        throughput_toks_per_sec=(
            decode_steps / (total_decode_ms / 1000) if total_decode_ms > 0 else 0
        ),
        ppl=ppl,
        kv_kept_mean=float(cache_len),
        kv_mb_per_tok=kv_mb,
        kv_ratio=1.0,
        peak_cpu_rss_mb=peak_cpu_rss,
        peak_gpu_alloc_mb=gpu_alloc,
        peak_gpu_reserved_mb=gpu_reserved,
        region_ppl_easy=ppl_easy,
        region_ppl_hard=ppl_hard,
        batch_size=batch_size,
    )


# ============================================================
# BPA decode runner (KV cache eviction)
# ============================================================


@torch.no_grad()
def run_bpa_decode(
    model,
    token_data,
    seq_len,
    decode_steps,
    seed,
    device_str,
    model_config,
    max_ctx,
    chunk_size=CHUNK_SIZE,
    gate_every_k=4,
    W_min=64,
    W_max=512,
    W_pressure_thresh=0.45,
    W_decay=0.95,
    B_far_max=8,
    B_far_target=2.0,
    B_far_scale=8.0,
    B_far_bias=0.0,
    stress_mode="control",
    stress_rng=None,
    batch_size=1,
):
    rng = np.random.RandomState(seed)

    if stress_mode in STRESS_GENERATORS:
        gen = STRESS_GENERATORS[stress_mode]
        idx = gen(token_data, seq_len + decode_steps, stress_rng or rng).to(device_str)
    else:
        idx = get_text_batch(token_data, batch_size, seq_len + decode_steps, rng).to(
            device_str
        )

    prefix = idx[:, :seq_len]
    continuation = idx[:, seq_len : seq_len + decode_steps]

    controller = AdaptiveController(
        W_min=W_min,
        W_max=W_max,
        W_decay=W_decay,
        W_pressure_thresh=W_pressure_thresh,
        B_far_max=B_far_max,
        B_far_target=B_far_target,
        B_far_scale=B_far_scale,
        B_far_bias=B_far_bias,
        gate_every_k=gate_every_k,
    )

    rss_before = get_cpu_rss_mb()

    # Warmup
    if device_str != "cpu":
        with torch.no_grad():
            _ = model(prefix[:, :16])
        gpu_sync(device_str)

    reset_gpu_mem(device_str)

    # Prefill (dense — we need full KV for initial state)
    t0 = time.perf_counter()
    out = model(prefix, use_cache=True)
    past = out.past_key_values
    gpu_sync(device_str)
    prefill_ms = (time.perf_counter() - t0) * 1000

    # Compute initial pressure from last prefill token
    last_logits = out.logits[:, -1, :]
    # Use logits norm as residual proxy (we don't have direct residual access)
    resid_norm = last_logits.float().norm().item()
    pressure = controller.compute_pressure(last_logits[0], resid_norm)
    controller.step(pressure)

    decode_latencies = []
    gate_latencies = []
    kept_tokens_log = []
    all_logits = [out.logits[:, -1:, :]]
    sel_rng = np.random.RandomState(seed + 1000)

    W_t = controller.W_log[-1]
    k_far_t = controller.k_far_log[-1]

    for step in range(decode_steps):
        next_token = continuation[:, step : step + 1]

        # Gate update
        if step % gate_every_k == 0:
            t_gate = time.perf_counter()
            W_t = controller.W_log[-1]
            k_far_t = controller.k_far_log[-1]

            # Evict KV cache based on current W and k_far
            cache_len = kv_cache_len(past)
            if cache_len > W_t:
                n_chunks = (cache_len + chunk_size - 1) // chunk_size
                far_end_chunk = max(0, (cache_len - W_t) // chunk_size)
                far_chunks = select_far_chunks_random(
                    n_chunks, far_end_chunk, k_far_t, sel_rng
                )
                keep_mask = build_keep_mask(cache_len, W_t, far_chunks, chunk_size)
                keep_mask = keep_mask.to(device_str)
                past = evict_kv_cache(past, keep_mask)

            gpu_sync(device_str)
            gate_dt = (time.perf_counter() - t_gate) * 1000
            gate_latencies.append(gate_dt)

        kept_tokens_log.append(kv_cache_len(past))

        # Decode step
        gpu_sync(device_str)
        t0 = time.perf_counter()
        out = model(next_token, past_key_values=past, use_cache=True)
        gpu_sync(device_str)
        dt = (time.perf_counter() - t0) * 1000
        decode_latencies.append(dt)
        past = out.past_key_values
        all_logits.append(out.logits)

        # Update pressure
        logits_t = out.logits[:, -1, :]
        resid_norm = logits_t.float().norm().item()
        pressure = controller.compute_pressure(logits_t[0], resid_norm)
        controller.step(pressure)

    all_logits_cat = torch.cat(all_logits, dim=1)
    ppl = compute_ppl(all_logits_cat[:, :-1, :], continuation)

    # Region PPL
    half = decode_steps // 2
    if half > 1:
        ppl_easy = compute_ppl(all_logits_cat[:, :half, :], continuation[:, :half])
        ppl_hard = compute_ppl(all_logits_cat[:, half:-1, :], continuation[:, half:])
    else:
        ppl_easy = ppl_hard = ppl

    # For stress tests, use defined spans
    if stress_mode in STRESS_SPANS and decode_steps > 4:
        spans = STRESS_SPANS[stress_mode]
        e_start = int(spans["easy"][0] * decode_steps)
        e_end = int(spans["easy"][1] * decode_steps)
        h_start = int(spans["hard"][0] * decode_steps)
        h_end = min(int(spans["hard"][1] * decode_steps), decode_steps)
        if e_end > e_start + 1 and h_end > h_start + 1:
            ppl_easy = compute_ppl(
                all_logits_cat[:, e_start : e_end - 1, :],
                continuation[:, e_start : e_end - 1],
            )
            ppl_hard = compute_ppl(
                all_logits_cat[:, h_start : h_end - 1, :],
                continuation[:, h_start : h_end - 1],
            )

    kv_kept_mean = float(np.mean(kept_tokens_log)) if kept_tokens_log else 0
    kv_bpt = kv_bytes_per_token(model_config)
    kv_mb = kv_kept_mean * kv_bpt / 1e6
    dense_kept = seq_len + decode_steps / 2
    kv_ratio = kv_kept_mean / dense_kept if dense_kept > 0 else 1.0

    rss_after = get_cpu_rss_mb()
    peak_cpu_rss = max(rss_after, rss_before)
    gpu_alloc, gpu_reserved = get_gpu_mem(device_str)

    assert peak_cpu_rss > 0, f"peak_cpu_rss_mb={peak_cpu_rss} must be > 0"
    if device_str != "cpu":
        assert gpu_alloc > 0, f"peak_gpu_alloc_mb={gpu_alloc} must be > 0 on CUDA"

    decode_arr = np.array(decode_latencies)
    total_decode_ms = sum(decode_latencies)
    total_gate_ms = sum(gate_latencies) if gate_latencies else 0
    gate_pct = (
        total_gate_ms / (total_decode_ms + total_gate_ms) * 100
        if (total_decode_ms + total_gate_ms) > 0
        else 0
    )

    ctrl = controller.get_summary()

    del past, out
    torch.cuda.empty_cache()

    return DecodeResult(
        method="bpa_v11",
        seq_len=seq_len,
        decode_steps=decode_steps,
        seed=seed,
        context_regime=context_regime(seq_len, max_ctx),
        prefill_ms=prefill_ms,
        decode_per_token_ms=float(np.median(decode_arr)),
        decode_p95_ms=float(np.percentile(decode_arr, 95)),
        gate_pct_of_total=gate_pct,
        throughput_toks_per_sec=(
            decode_steps / (total_decode_ms / 1000) if total_decode_ms > 0 else 0
        ),
        ppl=ppl,
        kv_kept_mean=kv_kept_mean,
        kv_mb_per_tok=kv_mb,
        kv_ratio=kv_ratio,
        peak_cpu_rss_mb=peak_cpu_rss,
        peak_gpu_alloc_mb=gpu_alloc,
        peak_gpu_reserved_mb=gpu_reserved,
        W_mean=ctrl["W_mean"],
        k_far_mean=ctrl["k_far_mean"],
        k_far_max=ctrl["k_far_max"],
        pressure_mean=ctrl["pressure_mean"],
        stress_mode=stress_mode,
        region_ppl_easy=ppl_easy,
        region_ppl_hard=ppl_hard,
        w_duty_cycle=controller.get_w_duty_cycle(),
        bpa_params={
            "W_min": W_min,
            "W_max": W_max,
            "W_decay": W_decay,
            "W_pressure_thresh": W_pressure_thresh,
            "B_far_max": B_far_max,
            "B_far_target": B_far_target,
            "B_far_scale": B_far_scale,
            "B_far_bias": B_far_bias,
            "gate_every_k": gate_every_k,
            "chunk_size": chunk_size,
        },
        batch_size=batch_size,
    )


# ============================================================
# Static sparse decode runner
# ============================================================


@torch.no_grad()
def run_static_sparse_decode(
    model,
    token_data,
    seq_len,
    decode_steps,
    seed,
    device_str,
    model_config,
    max_ctx,
    local_window=256,
    far_budget=2,
    chunk_size=CHUNK_SIZE,
    batch_size=1,
):
    rng = np.random.RandomState(seed)
    idx = get_text_batch(token_data, batch_size, seq_len + decode_steps, rng).to(
        device_str
    )
    prefix = idx[:, :seq_len]
    continuation = idx[:, seq_len : seq_len + decode_steps]

    rss_before = get_cpu_rss_mb()

    if device_str != "cpu":
        with torch.no_grad():
            _ = model(prefix[:, :16])
        gpu_sync(device_str)

    reset_gpu_mem(device_str)

    t0 = time.perf_counter()
    out = model(prefix, use_cache=True)
    past = out.past_key_values
    gpu_sync(device_str)
    prefill_ms = (time.perf_counter() - t0) * 1000

    decode_latencies = []
    kept_tokens_log = []
    all_logits = [out.logits[:, -1:, :]]
    sel_rng = np.random.RandomState(seed + 2000)

    for step in range(decode_steps):
        next_token = continuation[:, step : step + 1]

        # Evict every gate_every_k steps (fixed allocation)
        if step % 4 == 0:
            cache_len = kv_cache_len(past)
            if cache_len > local_window:
                n_chunks = (cache_len + chunk_size - 1) // chunk_size
                far_end_chunk = max(0, (cache_len - local_window) // chunk_size)
                far_chunks = select_far_chunks_random(
                    n_chunks, far_end_chunk, far_budget, sel_rng
                )
                keep_mask = build_keep_mask(
                    cache_len, local_window, far_chunks, chunk_size
                )
                keep_mask = keep_mask.to(device_str)
                past = evict_kv_cache(past, keep_mask)

        kept_tokens_log.append(kv_cache_len(past))

        gpu_sync(device_str)
        t0 = time.perf_counter()
        out = model(next_token, past_key_values=past, use_cache=True)
        gpu_sync(device_str)
        dt = (time.perf_counter() - t0) * 1000
        decode_latencies.append(dt)
        past = out.past_key_values
        all_logits.append(out.logits)

    all_logits_cat = torch.cat(all_logits, dim=1)
    ppl = compute_ppl(all_logits_cat[:, :-1, :], continuation)

    half = decode_steps // 2
    if half > 1:
        ppl_easy = compute_ppl(all_logits_cat[:, :half, :], continuation[:, :half])
        ppl_hard = compute_ppl(all_logits_cat[:, half:-1, :], continuation[:, half:])
    else:
        ppl_easy = ppl_hard = ppl

    kv_kept_mean = float(np.mean(kept_tokens_log)) if kept_tokens_log else 0
    kv_bpt = kv_bytes_per_token(model_config)
    kv_mb = kv_kept_mean * kv_bpt / 1e6
    dense_kept = seq_len + decode_steps / 2
    kv_ratio = kv_kept_mean / dense_kept if dense_kept > 0 else 1.0

    rss_after = get_cpu_rss_mb()
    peak_cpu_rss = max(rss_after, rss_before)
    gpu_alloc, gpu_reserved = get_gpu_mem(device_str)

    assert peak_cpu_rss > 0
    if device_str != "cpu":
        assert gpu_alloc > 0

    decode_arr = np.array(decode_latencies)
    total_decode_ms = sum(decode_latencies)

    del past, out
    torch.cuda.empty_cache()

    return DecodeResult(
        method="static_sparse",
        seq_len=seq_len,
        decode_steps=decode_steps,
        seed=seed,
        context_regime=context_regime(seq_len, max_ctx),
        prefill_ms=prefill_ms,
        decode_per_token_ms=float(np.median(decode_arr)),
        decode_p95_ms=float(np.percentile(decode_arr, 95)),
        gate_pct_of_total=0.0,
        throughput_toks_per_sec=(
            decode_steps / (total_decode_ms / 1000) if total_decode_ms > 0 else 0
        ),
        ppl=ppl,
        kv_kept_mean=kv_kept_mean,
        kv_mb_per_tok=kv_mb,
        kv_ratio=kv_ratio,
        peak_cpu_rss_mb=peak_cpu_rss,
        peak_gpu_alloc_mb=gpu_alloc,
        peak_gpu_reserved_mb=gpu_reserved,
        region_ppl_easy=ppl_easy,
        region_ppl_hard=ppl_hard,
        bpa_params={
            "local_window": local_window,
            "far_budget": far_budget,
            "chunk_size": chunk_size,
        },
        batch_size=batch_size,
    )


# ============================================================
# Auto-threshold from pressure distribution
# ============================================================


@torch.no_grad()
def auto_thresholds(model, token_data, seq_len, device_str, seed=1):
    """Compute pressure distribution quantiles for threshold selection."""
    rng = np.random.RandomState(seed)
    idx = get_text_batch(token_data, 1, seq_len + 32, rng).to(device_str)
    prefix = idx[:, :seq_len]

    out = model(prefix)
    logits = out.logits  # [1, seq_len, vocab]
    pressures = []
    for t in range(min(seq_len, 256)):
        probs = F.softmax(logits[0, t].float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        max_ent = math.log(probs.shape[-1])
        norm_ent = entropy / max_ent if max_ent > 0 else 0
        # Use logits norm as residual proxy
        norm_res = min(logits[0, t].float().norm().item() / 100.0, 1.0)
        pressures.append(0.7 * norm_ent + 0.3 * norm_res)

    p = np.array(pressures)
    return [float(np.percentile(p, q)) for q in [60, 75, 90]]


# ============================================================
# Tuning harness
# ============================================================

TUNE_SPACE = {
    "W_min": [32, 64, 128, 256],
    "W_max": [256, 512, 1024, 2048],
    "W_decay": [0.90, 0.95, 0.98],
    "gate_every_k": [2, 4, 8],
    "B_far_target": [1.0, 2.0, 3.0],
    "B_far_max": [4, 8, 16],
}


def run_tuning(
    model,
    token_data,
    seq_len,
    decode_steps,
    tol_pct,
    device_str,
    model_config,
    max_ctx,
    dense_ppl,
    output_dir,
):
    """Search for BPA configs meeting quality constraint."""
    thresholds = auto_thresholds(model, token_data, seq_len, device_str)
    print(f"  Auto thresholds for L={seq_len}: {thresholds}")

    # Build search space — scale W_max relative to seq_len
    w_max_candidates = [w for w in TUNE_SPACE["W_max"] if w < seq_len]
    if not w_max_candidates:
        w_max_candidates = [seq_len // 2]

    combos = []
    for W_min in TUNE_SPACE["W_min"]:
        for W_max in w_max_candidates:
            if W_max <= W_min:
                continue
            for thresh in thresholds:
                for decay in [0.95]:
                    for gate_k in [4]:
                        for B_target in [2.0]:
                            for B_max in [8]:
                                combos.append(
                                    {
                                        "W_min": W_min,
                                        "W_max": W_max,
                                        "W_pressure_thresh": thresh,
                                        "W_decay": decay,
                                        "gate_every_k": gate_k,
                                        "B_far_target": B_target,
                                        "B_far_max": B_max,
                                    }
                                )

    print(f"  Search space: {len(combos)} configurations")
    ppl_limit = dense_ppl * (1 + tol_pct / 100.0)

    results = []
    feasible = []
    eval_seeds = [1, 2]

    for i, params in enumerate(combos):
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(combos)}] feasible so far: {len(feasible)}")

        seed_ppls = []
        seed_p50s = []
        seed_kv = []
        seed_kept = []
        for s in eval_seeds:
            r = run_bpa_decode(
                model,
                token_data,
                seq_len,
                decode_steps,
                seed=s,
                device_str=device_str,
                model_config=model_config,
                max_ctx=max_ctx,
                **params,
            )
            seed_ppls.append(r.ppl)
            seed_p50s.append(r.decode_per_token_ms)
            seed_kv.append(r.kv_mb_per_tok)
            seed_kept.append(r.kv_kept_mean)

        avg_ppl = float(np.mean(seed_ppls))
        ppl_delta = (avg_ppl - dense_ppl) / dense_ppl * 100

        entry = {
            **params,
            "ppl": avg_ppl,
            "ppl_delta_pct": ppl_delta,
            "decode_p50_ms": float(np.mean(seed_p50s)),
            "kv_mb_per_tok": float(np.mean(seed_kv)),
            "kv_kept_mean": float(np.mean(seed_kept)),
            "feasible": avg_ppl <= ppl_limit,
        }
        results.append(entry)

        if avg_ppl <= ppl_limit:
            feasible.append(entry)

    # Save search results
    os.makedirs(os.path.join(output_dir, "search_results_v11"), exist_ok=True)
    csv_path = os.path.join(
        output_dir, "search_results_v11", f"L{seq_len}_tol{tol_pct}.csv"
    )
    if results:
        keys = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"  Search results: {csv_path} ({len(feasible)}/{len(results)} feasible)")

    # Select best: lowest p50 decode latency, tie-break by KV
    best = None
    if feasible:
        feasible.sort(key=lambda x: (x["decode_p50_ms"], x["kv_mb_per_tok"]))
        best = feasible[0]

    os.makedirs(os.path.join(output_dir, "selected_config_v11"), exist_ok=True)
    config_path = os.path.join(
        output_dir, "selected_config_v11", f"L{seq_len}_tol{tol_pct}.json"
    )

    if best:
        selected = {
            "L": seq_len,
            "tol_pct": tol_pct,
            "dense_ppl": dense_ppl,
            "ppl_limit": ppl_limit,
            "status": "PASS",
            "params": {
                k: v
                for k, v in best.items()
                if k in TUNE_SPACE or k == "W_pressure_thresh"
            },
            "metrics": {
                "ppl": best["ppl"],
                "ppl_delta_pct": best["ppl_delta_pct"],
                "decode_p50_ms": best["decode_p50_ms"],
                "kv_mb_per_tok": best["kv_mb_per_tok"],
                "kv_kept_mean": best["kv_kept_mean"],
            },
        }
    else:
        if results:
            results.sort(key=lambda x: x["ppl"])
            be = results[0]
            selected = {
                "L": seq_len,
                "tol_pct": tol_pct,
                "dense_ppl": dense_ppl,
                "ppl_limit": ppl_limit,
                "status": "FAIL",
                "quality_failed": True,
                "best_effort": {
                    "ppl": be["ppl"],
                    "ppl_delta_pct": be["ppl_delta_pct"],
                },
            }
        else:
            selected = {"L": seq_len, "tol_pct": tol_pct, "status": "FAIL"}

    with open(config_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"  Selected: {config_path} -> {selected.get('status')}")
    return selected


def tune_static_sparse(
    model,
    token_data,
    seq_len,
    decode_steps,
    tol_pct,
    device_str,
    model_config,
    max_ctx,
    dense_ppl,
    bpa_k_far_mean,
    output_dir,
):
    """Tune static_sparse baseline."""
    ppl_limit = dense_ppl * (1 + tol_pct / 100.0)
    far_budget = max(1, int(round(bpa_k_far_mean)))

    best = None
    for W in [64, 128, 256, 384, 512, 768, 1024]:
        if W >= seq_len:
            continue
        r = run_static_sparse_decode(
            model,
            token_data,
            seq_len,
            decode_steps,
            seed=1,
            device_str=device_str,
            model_config=model_config,
            max_ctx=max_ctx,
            local_window=W,
            far_budget=far_budget,
        )
        if r.ppl <= ppl_limit:
            if best is None or r.decode_per_token_ms < best.decode_per_token_ms:
                best = r

    config_path = os.path.join(
        output_dir, "selected_config_v11", f"static_L{seq_len}_tol{tol_pct}.json"
    )
    if best:
        selected = {
            "L": seq_len,
            "tol_pct": tol_pct,
            "method": "static_sparse",
            "status": "PASS",
            "metrics": {
                "ppl": best.ppl,
                "decode_p50_ms": best.decode_per_token_ms,
                "kv_kept_mean": best.kv_kept_mean,
            },
        }
    else:
        selected = {
            "L": seq_len,
            "tol_pct": tol_pct,
            "method": "static_sparse",
            "status": "FAIL",
        }

    with open(config_path, "w") as f:
        json.dump(selected, f, indent=2)
    return selected


# ============================================================
# Save / print helpers
# ============================================================


def save_all_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "decode_results_v11.json")
    data = [asdict(r) for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(results)} results to {path}")


def print_summary(results):
    print("\n" + "=" * 150)
    hdr = (
        f"{'Method':16s} {'L':>5s} {'BS':>3s} {'Regime':>10s} "
        f"{'Prefill':>8s} {'p50':>8s} {'p95':>8s} {'Gate%':>6s} "
        f"{'PPL':>10s} {'KV_kept':>8s} {'KV_MB':>7s} {'KV_rat':>7s} "
        f"{'tok/s':>7s} {'CPU_MB':>7s} {'GPU_MB':>7s}"
    )
    print(hdr)
    print("-" * 150)
    for r in sorted(results, key=lambda x: (x.seq_len, x.method)):
        print(
            f"{r.method:16s} {r.seq_len:5d} {r.batch_size:3d} "
            f"{r.context_regime:>10s} "
            f"{r.prefill_ms:7.0f}ms "
            f"{r.decode_per_token_ms:7.2f}ms "
            f"{r.decode_p95_ms:7.2f}ms "
            f"{r.gate_pct_of_total:5.1f}% "
            f"{r.ppl:10.1f} "
            f"{r.kv_kept_mean:7.0f} "
            f"{r.kv_mb_per_tok:6.2f} "
            f"{r.kv_ratio:6.2f}x "
            f"{r.throughput_toks_per_sec:6.0f} "
            f"{r.peak_cpu_rss_mb:6.0f} "
            f"{r.peak_gpu_alloc_mb:6.0f}"
        )
    print("=" * 150)


# ============================================================
# Commands
# ============================================================


def cmd_bench(args):
    """Benchmark: dense + bpa_v11 + static_sparse across L values."""
    device_str = args.device
    preflight_info = gpu_preflight(device_str)

    if device_str != "cpu" and not preflight_info:
        print("FATAL: GPU required for headline benchmarks")
        sys.exit(1)

    seq_lens = [int(x) for x in args.L.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    methods = (
        ["dense", "bpa_v11", "static_sparse"] if args.method == "all" else [args.method]
    )
    batch_size = args.batch_size

    print(f"Loading model {args.model}...")
    model, tokenizer, max_ctx, model_config = load_hf_model(args.model, device_str)
    run_meta = build_run_meta(device_str, args.model, max_ctx, model_config)

    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)
    print(f"  {len(token_data)} tokens loaded")

    regimes = [(L, context_regime(L, max_ctx)) for L in seq_lens]
    print(f"Context regimes: {regimes}")

    results = []
    for method in methods:
        for L in seq_lens:
            for seed in seeds:
                regime = context_regime(L, max_ctx)
                total_len = L + args.steps
                print(
                    f"  {method:16s} L={L:5d} seed={seed} "
                    f"regime={regime} bs={batch_size}...",
                    end="",
                    flush=True,
                )

                if method == "dense":
                    r = run_dense_decode(
                        model,
                        token_data,
                        L,
                        args.steps,
                        seed,
                        device_str,
                        model_config,
                        max_ctx,
                        batch_size=batch_size,
                    )
                elif method == "bpa_v11":
                    r = run_bpa_decode(
                        model,
                        token_data,
                        L,
                        args.steps,
                        seed,
                        device_str,
                        model_config,
                        max_ctx,
                        batch_size=batch_size,
                    )
                elif method == "static_sparse":
                    r = run_static_sparse_decode(
                        model,
                        token_data,
                        L,
                        args.steps,
                        seed,
                        device_str,
                        model_config,
                        max_ctx,
                        batch_size=batch_size,
                    )
                else:
                    continue

                r.run_meta = run_meta
                results.append(r)
                print(
                    f" PPL={r.ppl:.1f} p50={r.decode_per_token_ms:.2f}ms "
                    f"kept={r.kv_kept_mean:.0f} "
                    f"cpu={r.peak_cpu_rss_mb:.0f}MB "
                    f"gpu={r.peak_gpu_alloc_mb:.0f}MB"
                )

    print_summary(results)
    save_all_results(results, args.output_dir)


def cmd_tune(args):
    """Matched-quality tuning."""
    device_str = args.device
    gpu_preflight(device_str)

    seq_lens = [int(x) for x in args.L.split(",")]
    tols = [float(x) for x in args.tol.split(",")]

    print(f"Loading model {args.model}...")
    model, tokenizer, max_ctx, model_config = load_hf_model(args.model, device_str)

    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)

    # Get dense baselines first
    dense_ppls = {}
    for L in seq_lens:
        ppls = []
        for s in [1, 2]:
            r = run_dense_decode(
                model, token_data, L, args.steps, s, device_str, model_config, max_ctx
            )
            ppls.append(r.ppl)
            print(f"  Dense L={L} seed={s}: PPL={r.ppl:.1f}")
        dense_ppls[L] = float(np.mean(ppls))
        print(f"  Dense L={L} avg PPL={dense_ppls[L]:.1f}")

    # Tune BPA
    all_tuning = []
    for L in seq_lens:
        for tol in tols:
            print(f"\n=== Tuning BPA L={L} tol={tol}% ===")
            sel = run_tuning(
                model,
                token_data,
                L,
                args.steps,
                tol,
                device_str,
                model_config,
                max_ctx,
                dense_ppls[L],
                args.output_dir,
            )
            all_tuning.append(sel)

            # Also tune static_sparse
            print(f"  Tuning static_sparse L={L} tol={tol}%...")
            k_far_mean = 2.0
            if sel.get("status") == "PASS" and "params" in sel:
                k_far_mean = sel["params"].get("B_far_target", 2.0)
            ss = tune_static_sparse(
                model,
                token_data,
                L,
                args.steps,
                tol,
                device_str,
                model_config,
                max_ctx,
                dense_ppls[L],
                k_far_mean,
                args.output_dir,
            )
            all_tuning.append(ss)

    # Save tuning summary
    summary_path = os.path.join(args.output_dir, "tuning_summary_v11.json")
    with open(summary_path, "w") as f:
        json.dump(all_tuning, f, indent=2)
    print(f"\nSaved tuning summary to {summary_path}")


def cmd_stress(args):
    """Stress tests with region-wise evaluation."""
    device_str = args.device
    gpu_preflight(device_str)

    seq_lens = [int(x) for x in args.L.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    stress_modes = ["control", "late_binding", "kv_retrieval", "forced_far"]

    print(f"Loading model {args.model}...")
    model, tokenizer, max_ctx, model_config = load_hf_model(args.model, device_str)

    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)

    results = []
    for L in seq_lens:
        for seed in seeds:
            for stress in stress_modes:
                stress_rng = np.random.RandomState(seed * 100 + hash(stress) % 1000)
                print(
                    f"  bpa_v11 L={L:5d} seed={seed} stress={stress}...",
                    end="",
                    flush=True,
                )
                r = run_bpa_decode(
                    model,
                    token_data,
                    L,
                    args.steps,
                    seed,
                    device_str,
                    model_config,
                    max_ctx,
                    stress_mode=stress,
                    stress_rng=stress_rng,
                )
                results.append(r)
                print(
                    f" PPL={r.ppl:.1f} k_far={r.k_far_mean:.1f}"
                    f"(max={r.k_far_max}) W={r.W_mean:.0f}"
                )
                if stress != "control":
                    print(
                        f"    easy={r.region_ppl_easy:.0f} "
                        f"hard={r.region_ppl_hard:.0f}"
                    )

    # Print summary
    print("\n" + "=" * 100)
    print("STRESS TEST SUMMARY")
    print("=" * 100)
    print(
        f"{'L':>5s} {'Stress':>15s} {'Seed':>5s} "
        f"{'PPL':>10s} {'Easy':>10s} {'Hard':>10s} "
        f"{'k_far':>6s} {'k_max':>6s} {'W_mean':>7s} {'KV_kept':>8s}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r.seq_len:5d} {r.stress_mode:>15s} {r.seed:5d} "
            f"{r.ppl:10.0f} {r.region_ppl_easy:10.0f} {r.region_ppl_hard:10.0f} "
            f"{r.k_far_mean:6.1f} {r.k_far_max:6d} "
            f"{r.W_mean:7.0f} {r.kv_kept_mean:8.0f}"
        )

    save_all_results(results, args.output_dir + "_stress")


def main():
    parser = argparse.ArgumentParser(description="BPA v11 Decode Benchmark")
    sub = parser.add_subparsers(dest="command")

    # Bench
    p_bench = sub.add_parser("bench")
    p_bench.add_argument("--model", default=DEFAULT_MODEL)
    p_bench.add_argument("--L", default="512,1024,2048,4096")
    p_bench.add_argument("--steps", type=int, default=256)
    p_bench.add_argument("--seeds", default="1,2,3")
    p_bench.add_argument("--method", default="all")
    p_bench.add_argument("--device", default="cuda")
    p_bench.add_argument("--output-dir", default="bpa_v11_results")
    p_bench.add_argument("--batch-size", type=int, default=1)

    # Tune
    p_tune = sub.add_parser("tune")
    p_tune.add_argument("--model", default=DEFAULT_MODEL)
    p_tune.add_argument("--L", default="512,1024,2048,4096")
    p_tune.add_argument("--tol", default="1,3")
    p_tune.add_argument("--steps", type=int, default=256)
    p_tune.add_argument("--device", default="cuda")
    p_tune.add_argument("--output-dir", default="bpa_v11_results")

    # Stress
    p_stress = sub.add_parser("stress")
    p_stress.add_argument("--model", default=DEFAULT_MODEL)
    p_stress.add_argument("--L", default="512,1024,2048,4096")
    p_stress.add_argument("--steps", type=int, default=256)
    p_stress.add_argument("--seeds", default="1,2")
    p_stress.add_argument("--device", default="cuda")
    p_stress.add_argument("--output-dir", default="bpa_v11_results")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "bench":
        cmd_bench(args)
    elif args.command == "tune":
        cmd_tune(args)
    elif args.command == "stress":
        cmd_stress(args)


if __name__ == "__main__":
    main()
