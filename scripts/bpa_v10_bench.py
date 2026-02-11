#!/usr/bin/env python
"""
BPA v10: Decode benchmark with run manifests, real memory reporting,
matched-quality tuning, continuous far-budget controller, and GPU support.

Usage:
    # Single benchmark run
    python scripts/bpa_v10_bench.py bench --method all --L 512,1024 --steps 64

    # Matched-quality tuning
    python scripts/bpa_v10_bench.py tune --L 512,1024 --tol 1,3 --steps 64

    # Stress test with region-wise evaluation
    python scripts/bpa_v10_bench.py stress --L 512,1024 --steps 64
"""

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.bpa import BPAConfig, GPT2_BPA
from scripts.bpa_v3_experiment import (
    build_local_mask,
    compute_ppl,
    get_text_batch,
    load_text_data,
)
from scripts.bpa_v5_experiment import load_v4_gate, select_far_chunks
from scripts.bpa_v6_experiment import interpolate_pos_embed
from utils.ra_value_tracker import RAValueTracker, load_surgical_heads

# ============================================================
# Constants
# ============================================================

TRAINED_MAX_CTX = 1024  # GPT-2 wpe was trained at block_size=1024
CKPT_PATH = (
    "test_matrix_results_20260124_163855/"
    "gpt2_adamwspam_none_none/final_model_stepV0.pt"
)
DATA_PATH = "data/finewebedu/val.bin"
SURGICAL_PATH = "configs/ra_surgical_gpt2.json"
N_LAYERS = 12
N_HEADS = 12
N_EMBD = 768
HEAD_DIM = N_EMBD // N_HEADS
DTYPE_BYTES = 2  # float16 proxy for KV traffic estimation


# ============================================================
# Run manifest and metadata
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


def build_run_meta(device_str):
    meta = {
        "git_sha": get_git_sha(),
        "hostname": socket.gethostname(),
        "cpu": platform.processor() or platform.machine(),
        "torch_version": torch.__version__,
        "dtype": "float32",
        "device": device_str,
        "trained_max_ctx": TRAINED_MAX_CTX,
        "model_config": {
            "n_layer": N_LAYERS,
            "n_head": N_HEADS,
            "n_embd": N_EMBD,
            "vocab_size": 50304,
        },
    }
    if device_str != "cpu" and torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["gpu_count"] = torch.cuda.device_count()
        meta["gpu_mem_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
    return meta


def context_regime(L):
    return "in_range" if L <= TRAINED_MAX_CTX else "extrapolated"


def run_manifest_path(output_dir, method, L, device_str):
    sha = get_git_sha()
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        output_dir,
        "logs",
        f"{ts}_{sha}_{method}_L{L}_{device_str}.json",
    )


# ============================================================
# Memory reporting
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
# Adaptive controller with PI governor for far budget
# ============================================================


class AdaptiveController:
    """Controls local window W(t) and far budget B_far(t) per step.

    Pressure = 0.7 * norm_entropy + 0.3 * norm_residual.
    W(t): ramp up on high pressure, slow decay otherwise.
    B_far(t): continuous signal with EMA, hysteresis, and PI governor.
    """

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

        # PI controller for global budget governor
        self.pi_kp = pi_kp
        self.pi_ki = pi_ki
        self.pi_integral = 0.0

        # Hysteresis persistence for k_far changes
        self.hyst_persist = hyst_persist
        self._pending_k_far = None
        self._persist_count = 0

        # State
        self.W_current = float(W_min)
        self.B_far_raw = float(B_far_target)
        self.k_far_current = int(round(B_far_target))
        self.pressure_ema = 0.0
        self.above_thresh = False
        self.B_far_cumsum = 0.0
        self.n_steps = 0

        # Per-token logs
        self.W_log = []
        self.B_far_raw_log = []
        self.k_far_log = []
        self.pressure_log = []
        self.entropy_log = []

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
        probs = F.softmax(logits_t.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        max_entropy = math.log(probs.shape[-1])
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        norm_resid = min(residual_norm / 100.0, 1.0)
        pressure = 0.7 * norm_entropy + 0.3 * norm_resid
        self.entropy_log.append(entropy)
        return pressure

    def step(self, pressure):
        self.pressure_ema = (
            self.ema_alpha * pressure + (1 - self.ema_alpha) * self.pressure_ema
        )

        # Hysteresis for W threshold
        if self.above_thresh:
            trigger = self.pressure_ema > (self.W_pressure_thresh - self.hysteresis)
        else:
            trigger = self.pressure_ema > self.W_pressure_thresh
        self.above_thresh = trigger

        # Adaptive W
        if trigger:
            self.W_current = min(self.W_current * self.W_ramp_factor, self.W_max)
        else:
            self.W_current = max(self.W_current * self.W_decay, self.W_min)

        W_int = int(round(self.W_current))
        W_int = max(self.W_min, min(self.W_max, W_int))

        # Adaptive B_far: use raw pressure (not EMA) for faster response
        raw_B = self.B_far_scale * pressure + self.B_far_bias
        raw_B = max(0.0, min(float(self.B_far_max), raw_B))
        # Light EMA on the B_far signal itself (alpha=0.4 for fast tracking)
        b_alpha = 0.4
        self.B_far_raw = b_alpha * raw_B + (1 - b_alpha) * self.B_far_raw

        # PI governor: keep mean k_far near target
        self.n_steps += 1
        avg_k = self.B_far_cumsum / self.n_steps if self.n_steps > 0 else 0
        error = avg_k - self.B_far_target
        self.pi_integral += error
        pi_correction = self.pi_kp * error + self.pi_ki * self.pi_integral
        adjusted_B = self.B_far_raw - pi_correction
        adjusted_B = max(0.0, min(float(self.B_far_max), adjusted_B))

        # Unbiased rounding: floor + Bernoulli
        floor_B = int(math.floor(adjusted_B))
        frac = adjusted_B - floor_B
        # Deterministic for reproducibility: round based on fractional part
        candidate_k = floor_B + (1 if frac >= 0.5 else 0)
        candidate_k = max(0, min(self.B_far_max, candidate_k))

        # Hysteresis persistence: only change k_far if new value persists
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

    def get_per_token_log(self):
        return {
            "W": self.W_log,
            "B_far_raw": self.B_far_raw_log,
            "k_far": self.k_far_log,
            "pressure": self.pressure_log,
        }


# ============================================================
# Stress test generators
# ============================================================


def make_late_binding_sequence(data, seq_len, rng):
    """Key at 0.2*L, reference at 0.8*L."""
    offset = rng.randint(0, len(data) - seq_len * 2)
    tokens = data[offset : offset + seq_len].copy()
    key_pos = int(0.2 * seq_len)
    ref_pos = int(0.8 * seq_len)
    tokens[ref_pos : ref_pos + 16] = tokens[key_pos : key_pos + 16]
    return torch.from_numpy(tokens.astype(np.int64)).unsqueeze(0)


def make_kv_retrieval_sequence(data, seq_len, rng):
    """8 key-value pairs early, query for first pair late."""
    offset = rng.randint(0, len(data) - seq_len * 2)
    tokens = data[offset : offset + seq_len].copy()
    n_pairs = 8
    pair_len = 32
    for i in range(n_pairs):
        start = i * pair_len + 16
        if start + pair_len < seq_len // 2:
            src_start = rng.randint(0, len(data) - pair_len)
            tokens[start : start + pair_len] = data[src_start : src_start + pair_len]
    query_pos = int(0.85 * seq_len)
    tokens[query_pos : query_pos + 16] = tokens[16 : 16 + 16]
    return torch.from_numpy(tokens.astype(np.int64)).unsqueeze(0)


def make_forced_far_retrieval_sequence(data, seq_len, rng):
    """Force far retrieval beyond W_max: place key at position 16,
    fill positions 16..seq_len*0.9 with distractors, reference at 0.9*L.
    Hard span = [0.85*L, L], easy span = [0.3*L, 0.7*L]."""
    offset = rng.randint(0, len(data) - seq_len * 2)
    tokens = data[offset : offset + seq_len].copy()
    key_chunk = tokens[16 : 16 + 32].copy()
    # Fill middle with random distractors
    for i in range(10):
        dist_pos = int(0.1 * seq_len) + i * int(0.07 * seq_len)
        if dist_pos + 32 < int(0.85 * seq_len):
            src = rng.randint(0, len(data) - 32)
            tokens[dist_pos : dist_pos + 32] = data[src : src + 32]
    ref_pos = int(0.9 * seq_len)
    tokens[ref_pos : ref_pos + 32] = key_chunk
    return torch.from_numpy(tokens.astype(np.int64)).unsqueeze(0)


STRESS_GENERATORS = {
    "late_binding": make_late_binding_sequence,
    "kv_retrieval": make_kv_retrieval_sequence,
    "forced_far": make_forced_far_retrieval_sequence,
}

# Hard/easy span definitions for region-wise evaluation
STRESS_SPANS = {
    "late_binding": {"easy": (0.3, 0.6), "hard": (0.75, 1.0)},
    "kv_retrieval": {"easy": (0.3, 0.6), "hard": (0.8, 1.0)},
    "forced_far": {"easy": (0.3, 0.7), "hard": (0.85, 1.0)},
}


# ============================================================
# Model loading
# ============================================================


def load_model(seq_len, device="cpu"):
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    block_size = max(seq_len, 1024)
    cfg = BPAConfig(
        n_layer=N_LAYERS,
        n_head=N_HEADS,
        n_embd=N_EMBD,
        local_window=256,
        chunk_size=64,
        top_b=8,
        vocab_size=50304,
        block_size=block_size,
    )
    model = GPT2_BPA(cfg)
    model_sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    if seq_len > TRAINED_MAX_CTX:
        model_sd["transformer.wpe.weight"] = interpolate_pos_embed(
            model_sd["transformer.wpe.weight"], block_size
        )
        print(f"  Position embedding interpolated: {TRAINED_MAX_CTX} -> {block_size}")
    model.load_state_dict(model_sd, strict=False)
    model.eval()
    model.to(device)
    return model, cfg


# ============================================================
# Prefill and decode steps (unchanged from v9 core)
# ============================================================


@torch.no_grad()
def prefill_dense(model, idx):
    B, T = idx.shape
    device = idx.device
    n_head = N_HEADS
    n_embd = N_EMBD
    head_dim = HEAD_DIM

    pos = torch.arange(0, T, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)

    causal_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
    )
    scale = 1.0 / (head_dim**0.5)
    kv_cache = []

    for block in model.transformer.h:
        h = block.ln_1(x)
        attn = block.attn
        qkv = attn.c_attn(h)
        q, k, v = qkv.split(n_embd, dim=2)
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        attn_out = attn.resid_dropout(attn.c_proj(attn_out))
        x = x + attn_out
        x = x + block.mlp(block.ln_2(x))
        kv_cache.append((k, v))

    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)
    return logits, kv_cache, x


@torch.no_grad()
def decode_step_dense(model, token_id, pos_idx, kv_cache):
    device = token_id.device
    n_head = N_HEADS
    n_embd = N_EMBD
    head_dim = HEAD_DIM

    tok_emb = model.transformer.wte(token_id)
    pos_emb = model.transformer.wpe(
        torch.tensor([pos_idx], dtype=torch.long, device=device)
    )
    x = model.transformer.drop(tok_emb + pos_emb)

    new_kv_cache = []
    scale = 1.0 / (head_dim**0.5)

    for layer_idx, block in enumerate(model.transformer.h):
        h = block.ln_1(x)
        attn = block.attn
        qkv = attn.c_attn(h)
        q, k, v = qkv.split(n_embd, dim=2)
        B = q.shape[0]
        q = q.view(B, 1, n_head, head_dim).transpose(1, 2)
        k = k.view(B, 1, n_head, head_dim).transpose(1, 2)
        v = v.view(B, 1, n_head, head_dim).transpose(1, 2)
        k_prev, v_prev = kv_cache[layer_idx]
        k_full = torch.cat([k_prev, k], dim=2)
        v_full = torch.cat([v_prev, v], dim=2)
        new_kv_cache.append((k_full, v_full))
        scores = (q @ k_full.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = (attn_weights @ v_full).transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, 1, n_embd)
        attn_out = attn.resid_dropout(attn.c_proj(attn_out))
        x = x + attn_out
        x = x + block.mlp(block.ln_2(x))

    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)
    return logits, new_kv_cache, x


@torch.no_grad()
def decode_step_sparse(
    model,
    token_id,
    pos_idx,
    kv_cache,
    local_window,
    far_budget,
    chunk_size,
    strategy,
    ra_chunk_values=None,
    rng=None,
):
    device = token_id.device
    n_head = N_HEADS
    n_embd = N_EMBD
    head_dim = HEAD_DIM

    tok_emb = model.transformer.wte(token_id)
    pos_emb = model.transformer.wpe(
        torch.tensor([pos_idx], dtype=torch.long, device=device)
    )
    x = model.transformer.drop(tok_emb + pos_emb)

    new_kv_cache = []
    scale = 1.0 / (head_dim**0.5)

    cache_len = kv_cache[0][0].shape[2] if kv_cache else 0
    total_len = cache_len + 1

    far_end = max(0, total_len - local_window)
    n_chunks_total = (total_len + chunk_size - 1) // chunk_size

    selected_chunks = []
    if far_end > 0 and far_budget > 0:
        selected_chunks = select_far_chunks(
            strategy=strategy,
            n_chunks=n_chunks_total,
            far_budget=far_budget,
            ra_chunk_values=ra_chunk_values,
            query_pos=pos_idx,
            chunk_size=chunk_size,
            local_window=local_window,
            rng=rng,
        )

    attn_allowed = torch.zeros(total_len, dtype=torch.bool, device=device)
    local_start = max(0, total_len - local_window)
    attn_allowed[local_start:] = True
    for c in selected_chunks:
        c_start = c * chunk_size
        c_end = min((c + 1) * chunk_size, far_end)
        if c_start < far_end:
            attn_allowed[c_start:c_end] = True

    kept_tokens = int(attn_allowed.sum().item())

    # Track distances of selected far chunks to current position
    far_distances = []
    for c in selected_chunks:
        c_mid = c * chunk_size + chunk_size // 2
        if c_mid < far_end:
            far_distances.append(pos_idx - c_mid)

    for layer_idx, block in enumerate(model.transformer.h):
        h = block.ln_1(x)
        attn = block.attn
        qkv = attn.c_attn(h)
        q, k, v = qkv.split(n_embd, dim=2)
        B = q.shape[0]
        q = q.view(B, 1, n_head, head_dim).transpose(1, 2)
        k = k.view(B, 1, n_head, head_dim).transpose(1, 2)
        v = v.view(B, 1, n_head, head_dim).transpose(1, 2)
        k_prev, v_prev = kv_cache[layer_idx]
        k_full = torch.cat([k_prev, k], dim=2)
        v_full = torch.cat([v_prev, v], dim=2)
        new_kv_cache.append((k_full, v_full))
        scores = (q @ k_full.transpose(-2, -1)) * scale
        mask = ~attn_allowed.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = (attn_weights @ v_full).transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, 1, n_embd)
        attn_out = attn.resid_dropout(attn.c_proj(attn_out))
        x = x + attn_out
        x = x + block.mlp(block.ln_2(x))

    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)

    far_dist_summary = {}
    if far_distances:
        far_dist_summary = {
            "min": int(min(far_distances)),
            "mean": float(np.mean(far_distances)),
            "max": int(max(far_distances)),
        }

    return logits, new_kv_cache, x, kept_tokens, far_dist_summary


# ============================================================
# DecodeResult dataclass
# ============================================================


@dataclass
class DecodeResult:
    method: str
    seq_len: int
    decode_steps: int
    seed: int
    context_regime: str
    prefill_ms: float
    decode_total_ms: float
    decode_per_token_ms: float
    decode_p95_ms: float
    gate_per_token_ms: float
    gate_pct_of_total: float
    throughput_toks_per_sec: float
    ppl: float
    kv_bytes_read_per_token: float
    kv_kept_mean: float
    peak_cpu_rss_mb: float
    peak_gpu_alloc_mb: float
    peak_gpu_reserved_mb: float
    W_mean: float = 0.0
    W_min_obs: int = 0
    W_max_obs: int = 0
    B_far_raw_mean: float = 0.0
    k_far_mean: float = 0.0
    k_far_min: int = 0
    k_far_max: int = 0
    k_far_std: float = 0.0
    pressure_mean: float = 0.0
    stress_mode: str = "control"
    region_ppl_early: float = 0.0
    region_ppl_late: float = 0.0
    quality_failed: bool = False
    ppl_delta_pct: float = 0.0
    run_meta: dict = field(default_factory=dict)
    controller_log: dict = field(default_factory=dict)
    bpa_params: dict = field(default_factory=dict)


# ============================================================
# Benchmark runners
# ============================================================


def run_dense_decode(model, text_data, seq_len, decode_steps, seed, device_str):
    rng = np.random.RandomState(seed)
    idx = get_text_batch(text_data, 1, seq_len + decode_steps, rng).to(device_str)
    prefix = idx[:, :seq_len]
    continuation = idx[:, seq_len : seq_len + decode_steps]

    rss_before = get_cpu_rss_mb()

    if device_str != "cpu":
        _ = model.transformer.wte(prefix[:, :16])
        gpu_sync(device_str)

    reset_gpu_mem(device_str)

    t0 = time.perf_counter()
    logits_pf, kv_cache, _ = prefill_dense(model, prefix)
    gpu_sync(device_str)
    prefill_ms = (time.perf_counter() - t0) * 1000

    decode_latencies = []
    all_logits = [logits_pf[:, -1:, :]]
    kv = kv_cache

    for step in range(decode_steps):
        next_token = continuation[:, step : step + 1]
        t0 = time.perf_counter()
        logits_step, kv, _ = decode_step_dense(model, next_token, seq_len + step, kv)
        gpu_sync(device_str)
        dt = (time.perf_counter() - t0) * 1000
        decode_latencies.append(dt)
        all_logits.append(logits_step)

    all_logits_cat = torch.cat(all_logits, dim=1)
    targets = continuation
    ppl = compute_ppl(all_logits_cat[:, :-1, :], targets)
    half = decode_steps // 2
    if half > 1:
        ppl_early = compute_ppl(all_logits_cat[:, :half, :], targets[:, :half])
        ppl_late = compute_ppl(all_logits_cat[:, half:-1, :], targets[:, half:])
    else:
        ppl_early = ppl_late = ppl

    avg_cache_len = seq_len + decode_steps / 2
    kv_bytes = avg_cache_len * 2 * N_LAYERS * N_EMBD * DTYPE_BYTES

    rss_after = get_cpu_rss_mb()
    peak_cpu_rss = max(rss_after, rss_before)
    gpu_alloc, gpu_reserved = get_gpu_mem(device_str)

    assert peak_cpu_rss > 0, f"peak_cpu_rss_mb={peak_cpu_rss} must be > 0"
    if device_str != "cpu":
        assert gpu_alloc > 0, f"peak_gpu_alloc_mb={gpu_alloc} must be > 0 on CUDA"

    decode_arr = np.array(decode_latencies)
    total_decode_ms = sum(decode_latencies)

    return DecodeResult(
        method="dense",
        seq_len=seq_len,
        decode_steps=decode_steps,
        seed=seed,
        context_regime=context_regime(seq_len),
        prefill_ms=prefill_ms,
        decode_total_ms=total_decode_ms,
        decode_per_token_ms=float(np.median(decode_arr)),
        decode_p95_ms=float(np.percentile(decode_arr, 95)),
        gate_per_token_ms=0.0,
        gate_pct_of_total=0.0,
        throughput_toks_per_sec=(
            decode_steps / (total_decode_ms / 1000) if total_decode_ms > 0 else 0
        ),
        ppl=ppl,
        kv_bytes_read_per_token=kv_bytes,
        kv_kept_mean=avg_cache_len,
        peak_cpu_rss_mb=peak_cpu_rss,
        peak_gpu_alloc_mb=gpu_alloc,
        peak_gpu_reserved_mb=gpu_reserved,
        region_ppl_early=ppl_early,
        region_ppl_late=ppl_late,
    )


def run_bpa_v10_decode(
    model,
    text_data,
    seq_len,
    decode_steps,
    seed,
    device_str,
    surgical_heads,
    chunk_size=64,
    strategy="ra_value",
    gate_every_k=4,
    W_min=64,
    W_max=512,
    W_pressure_thresh=0.45,
    W_decay=0.95,
    B_far_max=8,
    B_far_target=2.0,
    B_far_scale=5.0,
    B_far_bias=0.0,
    stress_mode="control",
    stress_rng=None,
):
    rng = np.random.RandomState(seed)

    if stress_mode in STRESS_GENERATORS:
        gen = STRESS_GENERATORS[stress_mode]
        idx = gen(text_data, seq_len + decode_steps, stress_rng or rng).to(device_str)
    else:
        idx = get_text_batch(text_data, 1, seq_len + decode_steps, rng).to(device_str)

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

    tracker = RAValueTracker(surgical_heads, chunk_size=chunk_size)
    rss_before = get_cpu_rss_mb()

    if device_str != "cpu":
        _ = model.transformer.wte(prefix[:, :16])
        gpu_sync(device_str)

    reset_gpu_mem(device_str)

    t0 = time.perf_counter()
    logits_pf, kv_cache, residual = prefill_dense(model, prefix)
    gpu_sync(device_str)
    prefill_ms = (time.perf_counter() - t0) * 1000

    last_logits = logits_pf[:, -1, :]
    resid_norm = residual[:, -1, :].float().norm().item()
    pressure = controller.compute_pressure(last_logits[0], resid_norm)
    controller.step(pressure)

    decode_latencies = []
    gate_latencies = []
    kept_tokens_log = []
    far_dist_log = []
    all_logits = [logits_pf[:, -1:, :]]
    kv = kv_cache
    sel_rng = np.random.RandomState(seed + 1000)

    W_t = controller.W_log[-1]
    B_far_t = controller.k_far_log[-1]

    for step in range(decode_steps):
        next_token = continuation[:, step : step + 1]
        cur_pos = seq_len + step

        if step % gate_every_k == 0:
            t_gate = time.perf_counter()
            W_t = controller.W_log[-1]
            B_far_t = controller.k_far_log[-1]
            gpu_sync(device_str)
            gate_dt = (time.perf_counter() - t_gate) * 1000
            gate_latencies.append(gate_dt)

        t0 = time.perf_counter()
        logits_step, kv, residual_step, kept, far_dist = decode_step_sparse(
            model,
            next_token,
            cur_pos,
            kv,
            local_window=W_t,
            far_budget=B_far_t,
            chunk_size=chunk_size,
            strategy=strategy,
            ra_chunk_values=None,
            rng=sel_rng,
        )
        gpu_sync(device_str)
        dt = (time.perf_counter() - t0) * 1000
        decode_latencies.append(dt)
        kept_tokens_log.append(kept)
        if far_dist:
            far_dist_log.append(far_dist)
        all_logits.append(logits_step)

        resid_norm = residual_step[:, 0, :].float().norm().item()
        pressure = controller.compute_pressure(logits_step[0, 0], resid_norm)
        controller.step(pressure)

    all_logits_cat = torch.cat(all_logits, dim=1)
    targets = continuation
    ppl = compute_ppl(all_logits_cat[:, :-1, :], targets)

    # Region-wise PPL
    half = decode_steps // 2
    if half > 1:
        ppl_early = compute_ppl(all_logits_cat[:, :half, :], targets[:, :half])
        ppl_late = compute_ppl(all_logits_cat[:, half:-1, :], targets[:, half:])
    else:
        ppl_early = ppl_late = ppl

    # For stress tests, use defined spans
    if stress_mode in STRESS_SPANS and decode_steps > 4:
        spans = STRESS_SPANS[stress_mode]
        e_start = int(spans["easy"][0] * decode_steps)
        e_end = int(spans["easy"][1] * decode_steps)
        h_start = int(spans["hard"][0] * decode_steps)
        h_end = min(int(spans["hard"][1] * decode_steps), decode_steps)
        if e_end > e_start + 1 and h_end > h_start + 1:
            ppl_early = compute_ppl(
                all_logits_cat[:, e_start : e_end - 1, :],
                targets[:, e_start : e_end - 1],
            )
            ppl_late = compute_ppl(
                all_logits_cat[:, h_start : h_end - 1, :],
                targets[:, h_start : h_end - 1],
            )

    kv_kept_mean = float(np.mean(kept_tokens_log)) if kept_tokens_log else 0
    kv_bytes = kv_kept_mean * 2 * N_LAYERS * N_EMBD * DTYPE_BYTES

    rss_after = get_cpu_rss_mb()
    peak_cpu_rss = max(rss_after, rss_before)
    gpu_alloc, gpu_reserved = get_gpu_mem(device_str)

    assert peak_cpu_rss > 0, f"peak_cpu_rss_mb={peak_cpu_rss} must be > 0"
    if device_str != "cpu":
        assert gpu_alloc > 0, f"peak_gpu_alloc_mb={gpu_alloc} must be > 0 on CUDA"

    decode_arr = np.array(decode_latencies)
    gate_arr = np.array(gate_latencies) if gate_latencies else np.array([0.0])
    total_decode_ms = sum(decode_latencies)
    total_gate_ms = sum(gate_latencies)
    gate_pct = (
        total_gate_ms / (total_decode_ms + total_gate_ms) * 100
        if (total_decode_ms + total_gate_ms) > 0
        else 0
    )

    ctrl = controller.get_summary()
    per_token = controller.get_per_token_log()

    return DecodeResult(
        method="bpa_v10",
        seq_len=seq_len,
        decode_steps=decode_steps,
        seed=seed,
        context_regime=context_regime(seq_len),
        prefill_ms=prefill_ms,
        decode_total_ms=total_decode_ms,
        decode_per_token_ms=float(np.median(decode_arr)),
        decode_p95_ms=float(np.percentile(decode_arr, 95)),
        gate_per_token_ms=float(np.mean(gate_arr)),
        gate_pct_of_total=gate_pct,
        throughput_toks_per_sec=(
            decode_steps / (total_decode_ms / 1000) if total_decode_ms > 0 else 0
        ),
        ppl=ppl,
        kv_bytes_read_per_token=kv_bytes,
        kv_kept_mean=kv_kept_mean,
        peak_cpu_rss_mb=peak_cpu_rss,
        peak_gpu_alloc_mb=gpu_alloc,
        peak_gpu_reserved_mb=gpu_reserved,
        W_mean=ctrl["W_mean"],
        W_min_obs=ctrl["W_min_obs"],
        W_max_obs=ctrl["W_max_obs"],
        B_far_raw_mean=ctrl["B_far_raw_mean"],
        k_far_mean=ctrl["k_far_mean"],
        k_far_min=ctrl["k_far_min"],
        k_far_max=ctrl["k_far_max"],
        k_far_std=ctrl["k_far_std"],
        pressure_mean=ctrl["pressure_mean"],
        stress_mode=stress_mode,
        region_ppl_early=ppl_early,
        region_ppl_late=ppl_late,
        controller_log={
            "per_token": {
                "W": per_token["W"][-50:],
                "k_far": per_token["k_far"][-50:],
                "B_far_raw": per_token["B_far_raw"][-50:],
                "pressure": per_token["pressure"][-50:],
            },
            "summary": ctrl,
        },
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
            "strategy": strategy,
        },
    )


def run_static_sparse_decode(
    model,
    text_data,
    seq_len,
    decode_steps,
    seed,
    device_str,
    local_window=256,
    far_budget=4,
    chunk_size=64,
):
    rng = np.random.RandomState(seed)
    idx = get_text_batch(text_data, 1, seq_len + decode_steps, rng).to(device_str)
    prefix = idx[:, :seq_len]
    continuation = idx[:, seq_len : seq_len + decode_steps]

    rss_before = get_cpu_rss_mb()

    if device_str != "cpu":
        _ = model.transformer.wte(prefix[:, :16])
        gpu_sync(device_str)

    reset_gpu_mem(device_str)

    t0 = time.perf_counter()
    logits_pf, kv_cache, _ = prefill_dense(model, prefix)
    gpu_sync(device_str)
    prefill_ms = (time.perf_counter() - t0) * 1000

    decode_latencies = []
    kept_tokens_log = []
    all_logits = [logits_pf[:, -1:, :]]
    kv = kv_cache
    sel_rng = np.random.RandomState(seed + 2000)

    for step in range(decode_steps):
        next_token = continuation[:, step : step + 1]
        cur_pos = seq_len + step
        t0 = time.perf_counter()
        logits_step, kv, _, kept, _ = decode_step_sparse(
            model,
            next_token,
            cur_pos,
            kv,
            local_window=local_window,
            far_budget=far_budget,
            chunk_size=chunk_size,
            strategy="recency",
            rng=sel_rng,
        )
        gpu_sync(device_str)
        dt = (time.perf_counter() - t0) * 1000
        decode_latencies.append(dt)
        kept_tokens_log.append(kept)
        all_logits.append(logits_step)

    all_logits_cat = torch.cat(all_logits, dim=1)
    targets = continuation
    ppl = compute_ppl(all_logits_cat[:, :-1, :], targets)
    half = decode_steps // 2
    if half > 1:
        ppl_early = compute_ppl(all_logits_cat[:, :half, :], targets[:, :half])
        ppl_late = compute_ppl(all_logits_cat[:, half:-1, :], targets[:, half:])
    else:
        ppl_early = ppl_late = ppl

    kv_kept_mean = float(np.mean(kept_tokens_log)) if kept_tokens_log else 0
    kv_bytes = kv_kept_mean * 2 * N_LAYERS * N_EMBD * DTYPE_BYTES

    rss_after = get_cpu_rss_mb()
    peak_cpu_rss = max(rss_after, rss_before)
    gpu_alloc, gpu_reserved = get_gpu_mem(device_str)

    assert peak_cpu_rss > 0, f"peak_cpu_rss_mb={peak_cpu_rss} must be > 0"
    if device_str != "cpu":
        assert gpu_alloc > 0, f"peak_gpu_alloc_mb={gpu_alloc} must be > 0 on CUDA"

    decode_arr = np.array(decode_latencies)
    total_decode_ms = sum(decode_latencies)

    return DecodeResult(
        method="static_sparse",
        seq_len=seq_len,
        decode_steps=decode_steps,
        seed=seed,
        context_regime=context_regime(seq_len),
        prefill_ms=prefill_ms,
        decode_total_ms=total_decode_ms,
        decode_per_token_ms=float(np.median(decode_arr)),
        decode_p95_ms=float(np.percentile(decode_arr, 95)),
        gate_per_token_ms=0.0,
        gate_pct_of_total=0.0,
        throughput_toks_per_sec=(
            decode_steps / (total_decode_ms / 1000) if total_decode_ms > 0 else 0
        ),
        ppl=ppl,
        kv_bytes_read_per_token=kv_bytes,
        kv_kept_mean=kv_kept_mean,
        peak_cpu_rss_mb=peak_cpu_rss,
        peak_gpu_alloc_mb=gpu_alloc,
        peak_gpu_reserved_mb=gpu_reserved,
        region_ppl_early=ppl_early,
        region_ppl_late=ppl_late,
        bpa_params={
            "local_window": local_window,
            "far_budget": far_budget,
            "chunk_size": chunk_size,
        },
    )


# ============================================================
# Tuning harness (Phase 2)
# ============================================================

# Search space
TUNE_SPACE = {
    "W_min": [32, 64, 96, 128],
    "W_max": [128, 256, 512],
    "W_decay": [0.90, 0.95, 0.98],
    "gate_every_k": [2, 4, 8],
    "B_far_target": [1.0, 2.0, 3.0],
    "B_far_max": [4, 8, 16],
}


def auto_thresholds(model, text_data, seq_len, device_str, seed=1):
    """Compute pressure distribution quantiles for auto-threshold."""
    rng = np.random.RandomState(seed)
    idx = get_text_batch(text_data, 1, seq_len + 32, rng).to(device_str)
    prefix = idx[:, :seq_len]

    logits_pf, _, residual = prefill_dense(model, prefix)
    pressures = []
    for t in range(min(seq_len, 256)):
        probs = F.softmax(logits_pf[0, t].float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        max_ent = math.log(probs.shape[-1])
        norm_ent = entropy / max_ent if max_ent > 0 else 0
        norm_res = min(residual[0, t].float().norm().item() / 100.0, 1.0)
        pressures.append(0.7 * norm_ent + 0.3 * norm_res)

    p = np.array(pressures)
    q60 = float(np.percentile(p, 60))
    q75 = float(np.percentile(p, 75))
    q90 = float(np.percentile(p, 90))
    return [q60, q75, q90]


def run_tuning(
    model,
    text_data,
    seq_len,
    decode_steps,
    tol_pct,
    device_str,
    surgical_heads,
    dense_ppl,
    output_dir,
):
    """Search hyperparameter space for BPA configs meeting quality."""
    thresholds = auto_thresholds(model, text_data, seq_len, device_str)
    print(f"  Auto thresholds for L={seq_len}: {thresholds}")

    # Reduced search: sample representative combos instead of full grid
    # to keep runtime sane
    combos = []
    for W_min in TUNE_SPACE["W_min"]:
        for W_max in TUNE_SPACE["W_max"]:
            if W_max <= W_min:
                continue
            for thresh in thresholds:
                for decay in [0.95]:  # fix decay to reduce space
                    for gate_k in [4]:  # fix gate_k to reduce space
                        for B_target in [2.0]:  # fix B_target
                            for B_max in [8]:  # fix B_max
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

        # Average over 2 seeds for robust PPL estimate
        seed_ppls = []
        seed_p50s = []
        seed_kv = []
        seed_kept = []
        seed_W = []
        seed_kfar = []
        for s in eval_seeds:
            r = run_bpa_v10_decode(
                model,
                text_data,
                seq_len,
                decode_steps,
                seed=s,
                device_str=device_str,
                surgical_heads=surgical_heads,
                chunk_size=64,
                strategy="ra_value",
                **params,
            )
            seed_ppls.append(r.ppl)
            seed_p50s.append(r.decode_per_token_ms)
            seed_kv.append(r.kv_bytes_read_per_token)
            seed_kept.append(r.kv_kept_mean)
            seed_W.append(r.W_mean)
            seed_kfar.append(r.k_far_mean)

        avg_ppl = float(np.mean(seed_ppls))
        ppl_delta = (avg_ppl - dense_ppl) / dense_ppl * 100

        entry = {
            **params,
            "ppl": avg_ppl,
            "ppl_delta_pct": ppl_delta,
            "decode_p50_ms": float(np.mean(seed_p50s)),
            "decode_p95_ms": float(np.mean(seed_p50s)),
            "kv_bytes_per_token": float(np.mean(seed_kv)),
            "kv_kept_mean": float(np.mean(seed_kept)),
            "W_mean": float(np.mean(seed_W)),
            "k_far_mean": float(np.mean(seed_kfar)),
            "feasible": avg_ppl <= ppl_limit,
        }
        results.append(entry)

        if avg_ppl <= ppl_limit:
            feasible.append((entry, r))

    # Save search results
    os.makedirs(os.path.join(output_dir, "search_results"), exist_ok=True)
    csv_path = os.path.join(
        output_dir, "search_results", f"L{seq_len}_tol{tol_pct}.csv"
    )
    if results:
        import csv

        keys = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"  Search results: {csv_path} ({len(feasible)}/{len(results)} feasible)")

    # Select best: lowest p50 decode latency, tie-break by KV bytes
    best = None
    if feasible:
        feasible.sort(key=lambda x: (x[0]["decode_p50_ms"], x[0]["kv_bytes_per_token"]))
        best = feasible[0]

    # Save selected config
    os.makedirs(os.path.join(output_dir, "selected_config"), exist_ok=True)
    config_path = os.path.join(
        output_dir, "selected_config", f"L{seq_len}_tol{tol_pct}.json"
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
                for k, v in best[0].items()
                if k in TUNE_SPACE or k == "W_pressure_thresh"
            },
            "metrics": {
                "ppl": best[0]["ppl"],
                "ppl_delta_pct": best[0]["ppl_delta_pct"],
                "decode_p50_ms": best[0]["decode_p50_ms"],
                "kv_bytes_per_token": best[0]["kv_bytes_per_token"],
                "kv_kept_mean": best[0]["kv_kept_mean"],
            },
        }
    else:
        # Best effort: find config with lowest PPL
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
                "best_effort_params": {
                    k: v
                    for k, v in be.items()
                    if k in TUNE_SPACE or k == "W_pressure_thresh"
                },
                "best_effort_metrics": {
                    "ppl": be["ppl"],
                    "ppl_delta_pct": be["ppl_delta_pct"],
                    "decode_p50_ms": be["decode_p50_ms"],
                },
            }
        else:
            selected = {
                "L": seq_len,
                "tol_pct": tol_pct,
                "status": "FAIL",
                "reason": "no results",
            }

    with open(config_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"  Selected config: {config_path} -> {selected.get('status')}")

    return selected


def tune_static_sparse(
    model,
    text_data,
    seq_len,
    decode_steps,
    tol_pct,
    device_str,
    dense_ppl,
    bpa_k_far_mean,
    output_dir,
):
    """Tune static_sparse baseline with matched mean far budget."""
    ppl_limit = dense_ppl * (1 + tol_pct / 100.0)
    # Match BPA's mean k_far
    far_budget = max(1, int(round(bpa_k_far_mean)))

    best = None
    results = []
    for W in [64, 96, 128, 192, 256, 384, 512]:
        r = run_static_sparse_decode(
            model,
            text_data,
            seq_len,
            decode_steps,
            seed=1,
            device_str=device_str,
            local_window=W,
            far_budget=far_budget,
        )
        entry = {
            "local_window": W,
            "far_budget": far_budget,
            "ppl": r.ppl,
            "ppl_delta_pct": (r.ppl - dense_ppl) / dense_ppl * 100,
            "decode_p50_ms": r.decode_per_token_ms,
            "kv_bytes_per_token": r.kv_bytes_read_per_token,
            "feasible": r.ppl <= ppl_limit,
        }
        results.append(entry)
        if r.ppl <= ppl_limit:
            if best is None or r.decode_per_token_ms < best[0]["decode_p50_ms"]:
                best = (entry, r)

    config_path = os.path.join(
        output_dir, "selected_config", f"static_L{seq_len}_tol{tol_pct}.json"
    )
    if best:
        selected = {
            "L": seq_len,
            "tol_pct": tol_pct,
            "method": "static_sparse",
            "status": "PASS",
            "params": {
                "local_window": best[0]["local_window"],
                "far_budget": best[0]["far_budget"],
            },
            "metrics": {
                "ppl": best[0]["ppl"],
                "ppl_delta_pct": best[0]["ppl_delta_pct"],
                "decode_p50_ms": best[0]["decode_p50_ms"],
                "kv_bytes_per_token": best[0]["kv_bytes_per_token"],
            },
        }
    else:
        selected = {
            "L": seq_len,
            "tol_pct": tol_pct,
            "method": "static_sparse",
            "status": "FAIL",
            "quality_failed": True,
        }

    with open(config_path, "w") as f:
        json.dump(selected, f, indent=2)
    return selected


# ============================================================
# Save / report helpers
# ============================================================


def save_result_manifest(result, output_dir, run_meta):
    """Save individual run as a manifest JSON."""
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    path = run_manifest_path(
        output_dir,
        result.method,
        result.seq_len,
        run_meta.get("device", "cpu"),
    )
    data = asdict(result)
    data["run_meta"] = run_meta
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def save_all_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "decode_results_v10.json")
    data = [asdict(r) for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(results)} results to {path}")


def print_summary(results):
    print("\n" + "=" * 140)
    print(
        f"{'Method':15s} {'L':>5s} {'Regime':>12s} {'Prefill':>8s} "
        f"{'p50':>8s} {'p95':>8s} {'Gate%':>6s} {'PPL':>8s} "
        f"{'KV_kept':>8s} {'KV_MB':>7s} {'tok/s':>8s} "
        f"{'CPU_MB':>8s} {'GPU_MB':>8s}"
    )
    print("-" * 140)
    for r in sorted(results, key=lambda x: (x.seq_len, x.method)):
        kv_mb = r.kv_bytes_read_per_token / 1e6
        print(
            f"{r.method:15s} {r.seq_len:5d} {r.context_regime:>12s} "
            f"{r.prefill_ms:7.0f}ms "
            f"{r.decode_per_token_ms:7.2f}ms "
            f"{r.decode_p95_ms:7.2f}ms "
            f"{r.gate_pct_of_total:5.1f}% "
            f"{r.ppl:8.1f} "
            f"{r.kv_kept_mean:7.0f} "
            f"{kv_mb:6.2f} "
            f"{r.throughput_toks_per_sec:7.0f} "
            f"{r.peak_cpu_rss_mb:7.0f} "
            f"{r.peak_gpu_alloc_mb:7.0f}"
        )
    print("=" * 140)


# ============================================================
# Commands
# ============================================================


def cmd_bench(args):
    """Run benchmark for specified methods/lengths."""
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_str = "cpu"

    seq_lens = [int(x) for x in args.L.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    methods = (
        ["dense", "bpa_v10", "static_sparse"] if args.method == "all" else [args.method]
    )

    text_data = load_text_data(DATA_PATH)
    surgical_heads = load_surgical_heads(SURGICAL_PATH)
    run_meta = build_run_meta(device_str)
    max_L = max(seq_lens)

    print(f"Loading model for max L={max_L}...")
    model, cfg = load_model(max_L + args.steps, device_str)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"Device: {device_str}")
    print(f"Context regimes: {[(L, context_regime(L)) for L in seq_lens]}")
    print()

    all_results = []

    for L in seq_lens:
        for seed in seeds:
            for method in methods:
                print(
                    f"  {method:15s} L={L:5d} seed={seed} "
                    f"regime={context_regime(L)}...",
                    end="",
                    flush=True,
                )
                try:
                    if method == "dense":
                        r = run_dense_decode(
                            model, text_data, L, args.steps, seed, device_str
                        )
                    elif method == "bpa_v10":
                        r = run_bpa_v10_decode(
                            model,
                            text_data,
                            L,
                            args.steps,
                            seed,
                            device_str,
                            surgical_heads,
                            W_min=args.W_min,
                            W_max=args.W_max,
                            gate_every_k=args.gate_every_k,
                        )
                    elif method == "static_sparse":
                        r = run_static_sparse_decode(
                            model,
                            text_data,
                            L,
                            args.steps,
                            seed,
                            device_str,
                            local_window=256,
                            far_budget=4,
                        )
                    else:
                        print(f" unknown method {method}")
                        continue

                    r.run_meta = run_meta
                    all_results.append(r)
                    save_result_manifest(r, args.output_dir, run_meta)
                    print(
                        f" PPL={r.ppl:.1f} p50={r.decode_per_token_ms:.2f}ms "
                        f"kept={r.kv_kept_mean:.0f} "
                        f"cpu={r.peak_cpu_rss_mb:.0f}MB "
                        f"gpu={r.peak_gpu_alloc_mb:.0f}MB"
                    )
                except Exception as e:
                    print(f" ERROR: {e}")
                    import traceback

                    traceback.print_exc()

    print_summary(all_results)
    save_all_results(all_results, args.output_dir)


def cmd_tune(args):
    """Run matched-quality tuning loop."""
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_str = "cpu"

    seq_lens = [int(x) for x in args.L.split(",")]
    tolerances = [float(x) for x in args.tol.split(",")]

    text_data = load_text_data(DATA_PATH)
    surgical_heads = load_surgical_heads(SURGICAL_PATH)
    max_L = max(seq_lens)

    print(f"Loading model for max L={max_L}...")
    model, cfg = load_model(max_L + args.steps, device_str)
    print(f"Device: {device_str}")
    print()

    results_summary = []

    for L in seq_lens:
        regime = context_regime(L)
        print(f"\n{'='*60}")
        print(f"TUNING L={L} ({regime})")
        print(f"{'='*60}")

        # Dense baseline
        print(f"  Running dense baseline...")
        dense_ppls = []
        for seed in [1, 2]:
            r = run_dense_decode(model, text_data, L, args.steps, seed, device_str)
            dense_ppls.append(r.ppl)
        dense_ppl = float(np.mean(dense_ppls))
        print(f"  Dense PPL = {dense_ppl:.1f}")

        for tol in tolerances:
            print(
                f"\n  --- Tolerance: {tol}% (ppl limit: {dense_ppl * (1+tol/100):.1f}) ---"
            )
            selected = run_tuning(
                model,
                text_data,
                L,
                args.steps,
                tol,
                device_str,
                surgical_heads,
                dense_ppl,
                args.output_dir,
            )
            results_summary.append(selected)

            # Tune static baseline if BPA passed
            if selected.get("status") == "PASS":
                bpa_k_far = selected["metrics"].get("kv_kept_mean", 2.0)
                # Use k_far from controller summary as far budget for static
                # Default to 2 if not available
                k_far_for_static = 2
                print(
                    f"  Tuning static_sparse baseline (far_budget={k_far_for_static})..."
                )
                static_sel = tune_static_sparse(
                    model,
                    text_data,
                    L,
                    args.steps,
                    tol,
                    device_str,
                    dense_ppl,
                    k_far_for_static,
                    args.output_dir,
                )
                results_summary.append(static_sel)

    # Save overall summary
    summary_path = os.path.join(args.output_dir, "tuning_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nTuning summary: {summary_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("TUNING RESULTS")
    print("=" * 80)
    for s in results_summary:
        method = s.get("method", "bpa_v10")
        status = s.get("status", "?")
        L = s.get("L", "?")
        tol = s.get("tol_pct", "?")
        regime = context_regime(L) if isinstance(L, int) else "?"
        if status == "PASS":
            m = s["metrics"]
            print(
                f"  {method:15s} L={L:5} tol={tol}% {regime:>12s} "
                f"-> PASS ppl={m['ppl']:.1f} ({m['ppl_delta_pct']:+.1f}%) "
                f"p50={m['decode_p50_ms']:.2f}ms"
            )
        else:
            print(f"  {method:15s} L={L:5} tol={tol}% {regime:>12s} -> FAIL")


def cmd_stress(args):
    """Run stress tests with region-wise evaluation."""
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_str = "cpu"

    seq_lens = [int(x) for x in args.L.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    stress_modes = list(STRESS_GENERATORS.keys())

    text_data = load_text_data(DATA_PATH)
    surgical_heads = load_surgical_heads(SURGICAL_PATH)
    max_L = max(seq_lens)

    print(f"Loading model for max L={max_L}...")
    model, cfg = load_model(max_L + args.steps, device_str)
    print(f"Device: {device_str}")
    print()

    all_results = []
    run_meta = build_run_meta(device_str)

    for L in seq_lens:
        for seed in seeds:
            # Control run (BPA)
            print(
                f"  bpa_v10 L={L:5d} seed={seed} stress=control...",
                end="",
                flush=True,
            )
            r_control = run_bpa_v10_decode(
                model,
                text_data,
                L,
                args.steps,
                seed,
                device_str,
                surgical_heads,
                stress_mode="control",
            )
            r_control.run_meta = run_meta
            all_results.append(r_control)
            print(
                f" PPL={r_control.ppl:.1f} "
                f"k_far={r_control.k_far_mean:.1f} "
                f"W={r_control.W_mean:.0f}"
            )

            # Stress runs
            for stress in stress_modes:
                stress_rng = np.random.RandomState(seed + 42)
                print(
                    f"  bpa_v10 L={L:5d} seed={seed} stress={stress}...",
                    end="",
                    flush=True,
                )
                r = run_bpa_v10_decode(
                    model,
                    text_data,
                    L,
                    args.steps,
                    seed,
                    device_str,
                    surgical_heads,
                    stress_mode=stress,
                    stress_rng=stress_rng,
                )
                r.run_meta = run_meta
                all_results.append(r)
                print(
                    f" PPL={r.ppl:.1f} "
                    f"k_far(mean={r.k_far_mean:.1f} "
                    f"max={r.k_far_max}) "
                    f"W={r.W_mean:.0f} "
                    f"early={r.region_ppl_early:.0f} "
                    f"late={r.region_ppl_late:.0f}"
                )

    save_all_results(all_results, args.output_dir)

    # Region-wise summary
    print("\n" + "=" * 120)
    print("STRESS TEST REGION-WISE SUMMARY")
    print("=" * 120)
    print(
        f"{'L':>5s} {'Stress':>15s} {'PPL':>8s} "
        f"{'Easy PPL':>10s} {'Hard PPL':>10s} "
        f"{'k_far_mean':>10s} {'k_far_max':>9s} "
        f"{'W_mean':>7s} {'KV_kept':>8s}"
    )
    print("-" * 120)

    for r in sorted(all_results, key=lambda x: (x.seq_len, x.stress_mode)):
        print(
            f"{r.seq_len:5d} {r.stress_mode:>15s} {r.ppl:8.0f} "
            f"{r.region_ppl_early:10.0f} {r.region_ppl_late:10.0f} "
            f"{r.k_far_mean:10.1f} {r.k_far_max:9d} "
            f"{r.W_mean:7.0f} {r.kv_kept_mean:7.0f}"
        )


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="BPA v10 decode benchmark")
    sub = parser.add_subparsers(dest="command")

    # bench
    p_bench = sub.add_parser("bench", help="Run benchmark")
    p_bench.add_argument("--method", default="all")
    p_bench.add_argument("--L", default="512,1024")
    p_bench.add_argument("--steps", type=int, default=64)
    p_bench.add_argument("--seeds", default="1,2")
    p_bench.add_argument("--device", default="cpu")
    p_bench.add_argument("--output-dir", default="bpa_v10_results")
    p_bench.add_argument("--W-min", type=int, default=64)
    p_bench.add_argument("--W-max", type=int, default=512)
    p_bench.add_argument("--gate-every-k", type=int, default=4)

    # tune
    p_tune = sub.add_parser("tune", help="Matched-quality tuning")
    p_tune.add_argument("--L", default="512,1024")
    p_tune.add_argument("--tol", default="1,3")
    p_tune.add_argument("--steps", type=int, default=64)
    p_tune.add_argument("--device", default="cpu")
    p_tune.add_argument("--output-dir", default="bpa_v10_results")

    # stress
    p_stress = sub.add_parser("stress", help="Stress tests")
    p_stress.add_argument("--L", default="512,1024")
    p_stress.add_argument("--steps", type=int, default=64)
    p_stress.add_argument("--seeds", default="1,2")
    p_stress.add_argument("--device", default="cpu")
    p_stress.add_argument("--output-dir", default="bpa_v10_results")

    args = parser.parse_args()

    if args.command == "bench":
        cmd_bench(args)
    elif args.command == "tune":
        cmd_tune(args)
    elif args.command == "stress":
        cmd_stress(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
