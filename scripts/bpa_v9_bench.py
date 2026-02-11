#!/usr/bin/env python
"""
BPA v9: Decode benchmark + adaptive W/B_far controller.

Measures prefill + autoregressive decode latency, memory, KV traffic
for dense vs BPA v9 (adaptive) vs static top-k baseline.

Usage:
    python scripts/bpa_v9_bench.py --method dense --L 1024,2048 --steps 128
    python scripts/bpa_v9_bench.py --method bpa_v9 --L 1024,2048 --steps 128
    python scripts/bpa_v9_bench.py --method static_topk --L 1024,2048 --steps 128
    python scripts/bpa_v9_bench.py --method all --L 1024,2048 --steps 128
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
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
# Adaptive controller for W(t) and B_far(t)
# ============================================================


class AdaptiveController:
    """Controls local window W(t) and far budget B_far(t) per step.

    Pressure signals:
    - entropy: next-token entropy from model logits
    - residual_norm: norm of last residual stream (cheap proxy for difficulty)

    Controller:
    - W(t): if pressure > thresh -> ramp up, else decay toward W_min
    - B_far(t): clamp(a * pressure + b, 0, B_max), EMA smoothed
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
        B_far_scale=5.0,
        B_far_bias=0.0,
        ema_alpha=0.1,
        hysteresis=0.3,
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

        # State
        self.W_current = float(W_min)
        self.B_far_current = float(B_far_target)
        self.pressure_ema = 0.0
        self.above_thresh = False
        self.B_far_cumsum = 0.0
        self.n_steps = 0

        # Logs
        self.W_log = []
        self.B_far_log = []
        self.pressure_log = []
        self.entropy_log = []

    def reset(self):
        self.W_current = float(self.W_min)
        self.B_far_current = float(self.B_far_target)
        self.pressure_ema = 0.0
        self.above_thresh = False
        self.B_far_cumsum = 0.0
        self.n_steps = 0
        self.W_log = []
        self.B_far_log = []
        self.pressure_log = []
        self.entropy_log = []

    def compute_pressure(self, logits_t: torch.Tensor, residual_norm: float):
        """Compute pressure from logits at position t.

        logits_t: [vocab_size] raw logits for position t.
        residual_norm: scalar norm of residual stream at t.
        """
        probs = F.softmax(logits_t.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        max_entropy = math.log(probs.shape[-1])
        norm_entropy = entropy / max_entropy
        norm_resid = min(residual_norm / 100.0, 1.0)
        pressure = 0.7 * norm_entropy + 0.3 * norm_resid
        self.entropy_log.append(entropy)
        return pressure

    def step(self, pressure: float):
        """Update W and B_far based on pressure signal."""
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

        # Adaptive B_far
        raw_B = self.B_far_scale * self.pressure_ema + self.B_far_bias
        raw_B = max(0.0, min(float(self.B_far_max), raw_B))
        self.B_far_current = (
            self.ema_alpha * raw_B + (1 - self.ema_alpha) * self.B_far_current
        )
        B_far_int = max(0, min(self.B_far_max, int(round(self.B_far_current))))

        # Global cap enforcement: if average B_far exceeds target, reduce
        self.B_far_cumsum += B_far_int
        self.n_steps += 1
        avg_B = self.B_far_cumsum / self.n_steps
        if avg_B > self.B_far_target * 1.5 and B_far_int > 0:
            B_far_int = max(0, B_far_int - 1)

        self.W_log.append(W_int)
        self.B_far_log.append(B_far_int)
        self.pressure_log.append(float(self.pressure_ema))

        return W_int, B_far_int

    def get_summary(self):
        return {
            "W_mean": float(np.mean(self.W_log)) if self.W_log else 0,
            "W_min": int(min(self.W_log)) if self.W_log else 0,
            "W_max": int(max(self.W_log)) if self.W_log else 0,
            "W_std": float(np.std(self.W_log)) if self.W_log else 0,
            "B_far_mean": float(np.mean(self.B_far_log)) if self.B_far_log else 0,
            "B_far_min": int(min(self.B_far_log)) if self.B_far_log else 0,
            "B_far_max_obs": int(max(self.B_far_log)) if self.B_far_log else 0,
            "pressure_mean": (
                float(np.mean(self.pressure_log)) if self.pressure_log else 0
            ),
            "entropy_mean": float(np.mean(self.entropy_log)) if self.entropy_log else 0,
            "n_steps": self.n_steps,
        }


# ============================================================
# Stress test generators
# ============================================================


def make_late_binding_sequence(data, seq_len, rng):
    """Late-binding pointer: key defined early, referenced late.

    Structure: [context_A ... KEY_DEF ... context_B ... KEY_REF ...]
    The KEY_DEF appears around position 0.2*L, KEY_REF at 0.8*L.
    Quality should depend on whether the model retrieves KEY_DEF.
    """
    offset = rng.randint(0, len(data) - seq_len * 2)
    tokens = data[offset : offset + seq_len].copy()
    key_pos = int(0.2 * seq_len)
    ref_pos = int(0.8 * seq_len)
    tokens[ref_pos : ref_pos + 16] = tokens[key_pos : key_pos + 16]
    return torch.from_numpy(tokens.astype(np.int64)).unsqueeze(0)


def make_kv_retrieval_sequence(data, seq_len, rng):
    """Key-value retrieval with distractors.

    Structure: [key1:val1 key2:val2 ... keyN:valN ... query_key1]
    Many similar key-value pairs early, query late.
    """
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


# ============================================================
# Model loading
# ============================================================


def load_model(seq_len, device="cpu"):
    """Load GPT2_BPA model from checkpoint."""
    ckpt_path = (
        "test_matrix_results_20260124_163855/"
        "gpt2_adamwspam_none_none/final_model_stepV0.pt"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    block_size = max(seq_len, 1024)

    cfg = BPAConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        local_window=256,
        chunk_size=64,
        top_b=8,
        vocab_size=50304,
        block_size=block_size,
    )

    model = GPT2_BPA(cfg)
    model_sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    if seq_len > 1024:
        model_sd["transformer.wpe.weight"] = interpolate_pos_embed(
            model_sd["transformer.wpe.weight"], block_size
        )
        print(f"  Position embedding interpolated: 1024 -> {block_size}")

    model.load_state_dict(model_sd, strict=False)
    model.eval()
    model.to(device)
    return model, cfg


# ============================================================
# Single-pass prefill (full forward, all positions at once)
# ============================================================


@torch.no_grad()
def prefill_dense(model, idx):
    """Standard dense prefill: full forward on all tokens."""
    B, T = idx.shape
    device = idx.device
    cfg = model.config
    n_head = cfg.n_head
    n_embd = cfg.n_embd
    head_dim = n_embd // n_head

    pos = torch.arange(0, T, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)

    causal_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
    )

    scale = 1.0 / (head_dim**0.5)
    kv_cache = []

    for layer_idx, block in enumerate(model.transformer.h):
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


# ============================================================
# Autoregressive decode step
# ============================================================


@torch.no_grad()
def decode_step_dense(model, token_id, pos_idx, kv_cache):
    """Single dense decode step with KV cache."""
    device = token_id.device
    cfg = model.config
    n_head = cfg.n_head
    n_embd = cfg.n_embd
    head_dim = n_embd // n_head

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

        # Append to cache
        k_prev, v_prev = kv_cache[layer_idx]
        k_full = torch.cat([k_prev, k], dim=2)
        v_full = torch.cat([v_prev, v], dim=2)
        new_kv_cache.append((k_full, v_full))

        # Attend to all cached KV
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
    """Single sparse decode step: attend local_window + selected far chunks."""
    device = token_id.device
    cfg = model.config
    n_head = cfg.n_head
    n_embd = cfg.n_embd
    head_dim = n_embd // n_head

    tok_emb = model.transformer.wte(token_id)
    pos_emb = model.transformer.wpe(
        torch.tensor([pos_idx], dtype=torch.long, device=device)
    )
    x = model.transformer.drop(tok_emb + pos_emb)

    new_kv_cache = []
    scale = 1.0 / (head_dim**0.5)

    cache_len = kv_cache[0][0].shape[2] if kv_cache else 0
    total_len = cache_len + 1

    # Determine which positions to attend
    far_end = max(0, total_len - local_window)
    n_chunks_total = (total_len + chunk_size - 1) // chunk_size

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
    else:
        selected_chunks = []

    # Build attention mask over cache: 1 = attend, 0 = mask
    attn_allowed = torch.zeros(total_len, dtype=torch.bool, device=device)
    # Always attend to local window
    local_start = max(0, total_len - local_window)
    attn_allowed[local_start:] = True
    # Attend to selected far chunks
    for c in selected_chunks:
        c_start = c * chunk_size
        c_end = min((c + 1) * chunk_size, far_end)
        if c_start < far_end:
            attn_allowed[c_start:c_end] = True

    kept_tokens = int(attn_allowed.sum().item())

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

        # Sparse attention: only attend to allowed positions
        scores = (q @ k_full.transpose(-2, -1)) * scale
        # Mask out disallowed positions
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
    return logits, new_kv_cache, x, kept_tokens


# ============================================================
# Benchmark runners
# ============================================================


@dataclass
class DecodeResult:
    method: str
    seq_len: int
    decode_steps: int
    seed: int
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
    peak_mem_mb: float
    W_mean: float = 0.0
    W_min: int = 0
    W_max: int = 0
    B_far_mean: float = 0.0
    pressure_mean: float = 0.0
    stress_mode: str = "control"
    region_ppl_early: float = 0.0
    region_ppl_late: float = 0.0
    extra: dict = field(default_factory=dict)


def run_dense_decode(model, text_data, seq_len, decode_steps, seed, device):
    """Run dense prefill + decode benchmark."""
    rng = np.random.RandomState(seed)
    idx = get_text_batch(text_data, 1, seq_len + decode_steps, rng).to(device)
    prefix = idx[:, :seq_len]
    continuation = idx[:, seq_len : seq_len + decode_steps]

    cfg = model.config
    n_layers = cfg.n_layer
    n_embd = cfg.n_embd
    dtype_bytes = 2  # float16 proxy

    # Warmup
    if device != "cpu":
        _ = model.transformer.wte(prefix[:, :16])
        torch.cuda.synchronize()

    # Prefill
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    logits_pf, kv_cache, _ = prefill_dense(model, prefix)
    if device != "cpu":
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    # Decode loop
    decode_latencies = []
    all_logits = [logits_pf[:, -1:, :]]  # last token logits from prefill
    kv = kv_cache

    for step in range(decode_steps):
        next_token = continuation[:, step : step + 1]
        t0 = time.perf_counter()
        logits_step, kv, _ = decode_step_dense(model, next_token, seq_len + step, kv)
        if device != "cpu":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        decode_latencies.append(dt)
        all_logits.append(logits_step)

    # PPL on continuation
    all_logits_cat = torch.cat(all_logits, dim=1)  # [1, decode_steps+1, V]
    targets = continuation  # [1, decode_steps]
    ppl = compute_ppl(all_logits_cat[:, :-1, :], targets)
    # Region-wise PPL
    half = decode_steps // 2
    if half > 1:
        ppl_early = compute_ppl(all_logits_cat[:, :half, :], targets[:, :half])
        ppl_late = compute_ppl(all_logits_cat[:, half:-1, :], targets[:, half:])
    else:
        ppl_early = ppl_late = ppl

    # KV accounting: dense reads all cached tokens
    avg_cache_len = seq_len + decode_steps / 2
    kv_bytes = avg_cache_len * 2 * n_layers * n_embd * dtype_bytes

    peak_mem = 0.0
    if device != "cpu":
        peak_mem = torch.cuda.max_memory_allocated() / 1e6

    decode_arr = np.array(decode_latencies)
    total_decode_ms = sum(decode_latencies)

    return DecodeResult(
        method="dense",
        seq_len=seq_len,
        decode_steps=decode_steps,
        seed=seed,
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
        peak_mem_mb=peak_mem,
        region_ppl_early=ppl_early,
        region_ppl_late=ppl_late,
    )


def run_bpa_v9_decode(
    model,
    text_data,
    seq_len,
    decode_steps,
    seed,
    device,
    surgical_heads,
    chunk_size=64,
    strategy="ra_value",
    gate_every_k=4,
    W_min=64,
    W_max=512,
    B_far_max=8,
    B_far_target=2.0,
    stress_mode="control",
    stress_rng=None,
):
    """Run BPA v9 adaptive decode benchmark."""
    rng = np.random.RandomState(seed)

    if stress_mode == "late_binding":
        idx = make_late_binding_sequence(
            text_data, seq_len + decode_steps, stress_rng or rng
        ).to(device)
    elif stress_mode == "kv_retrieval":
        idx = make_kv_retrieval_sequence(
            text_data, seq_len + decode_steps, stress_rng or rng
        ).to(device)
    else:
        idx = get_text_batch(text_data, 1, seq_len + decode_steps, rng).to(device)

    prefix = idx[:, :seq_len]
    continuation = idx[:, seq_len : seq_len + decode_steps]

    cfg = model.config
    n_layers = cfg.n_layer
    n_embd = cfg.n_embd
    dtype_bytes = 2

    controller = AdaptiveController(
        W_min=W_min,
        W_max=W_max,
        B_far_max=B_far_max,
        B_far_target=B_far_target,
    )

    tracker = RAValueTracker(surgical_heads, chunk_size=chunk_size)

    # Warmup
    if device != "cpu":
        _ = model.transformer.wte(prefix[:, :16])
        torch.cuda.synchronize()

    # Prefill (dense — cache everything)
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    logits_pf, kv_cache, residual = prefill_dense(model, prefix)
    if device != "cpu":
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    # Initialize controller from prefill
    last_logits = logits_pf[:, -1, :]
    resid_norm = residual[:, -1, :].float().norm().item()
    pressure = controller.compute_pressure(last_logits[0], resid_norm)
    controller.step(pressure)

    # Decode loop with adaptive W and B_far
    decode_latencies = []
    gate_latencies = []
    kept_tokens_log = []
    all_logits = [logits_pf[:, -1:, :]]
    kv = kv_cache
    sel_rng = np.random.RandomState(seed + 1000)

    for step in range(decode_steps):
        next_token = continuation[:, step : step + 1]
        cur_pos = seq_len + step

        # Gate decision: update every k steps
        if step % gate_every_k == 0:
            t_gate = time.perf_counter()
            W_t, B_far_t = int(controller.W_log[-1]), int(controller.B_far_log[-1])
            if device != "cpu":
                torch.cuda.synchronize()
            gate_dt = (time.perf_counter() - t_gate) * 1000
            gate_latencies.append(gate_dt)

        # Decode step with sparse attention
        t0 = time.perf_counter()
        logits_step, kv, residual_step, kept = decode_step_sparse(
            model,
            next_token,
            cur_pos,
            kv,
            local_window=W_t,
            far_budget=B_far_t,
            chunk_size=chunk_size,
            strategy=strategy,
            ra_chunk_values=None,  # Use recency within ra_value for decode
            rng=sel_rng,
        )
        if device != "cpu":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        decode_latencies.append(dt)
        kept_tokens_log.append(kept)
        all_logits.append(logits_step)

        # Update controller
        resid_norm = residual_step[:, 0, :].float().norm().item()
        pressure = controller.compute_pressure(logits_step[0, 0], resid_norm)
        controller.step(pressure)

    # PPL
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
    kv_bytes = kv_kept_mean * 2 * n_layers * n_embd * dtype_bytes

    peak_mem = 0.0
    if device != "cpu":
        peak_mem = torch.cuda.max_memory_allocated() / 1e6

    decode_arr = np.array(decode_latencies)
    gate_arr = np.array(gate_latencies) if gate_latencies else np.array([0.0])
    total_decode_ms = sum(decode_latencies)
    total_gate_ms = sum(gate_latencies)
    gate_pct = (
        total_gate_ms / (total_decode_ms + total_gate_ms) * 100
        if (total_decode_ms + total_gate_ms) > 0
        else 0
    )

    ctrl_summary = controller.get_summary()

    return DecodeResult(
        method="bpa_v9",
        seq_len=seq_len,
        decode_steps=decode_steps,
        seed=seed,
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
        peak_mem_mb=peak_mem,
        W_mean=ctrl_summary["W_mean"],
        W_min=ctrl_summary["W_min"],
        W_max=ctrl_summary["W_max"],
        B_far_mean=ctrl_summary["B_far_mean"],
        pressure_mean=ctrl_summary["pressure_mean"],
        stress_mode=stress_mode,
        region_ppl_early=ppl_early,
        region_ppl_late=ppl_late,
        extra={
            "controller": ctrl_summary,
            "W_log": controller.W_log[-20:],
            "B_far_log": controller.B_far_log[-20:],
        },
    )


def run_static_topk_decode(
    model,
    text_data,
    seq_len,
    decode_steps,
    seed,
    device,
    local_window=256,
    far_budget=4,
    chunk_size=64,
):
    """Static top-k baseline: fixed W and B_far, recency selection."""
    rng = np.random.RandomState(seed)
    idx = get_text_batch(text_data, 1, seq_len + decode_steps, rng).to(device)
    prefix = idx[:, :seq_len]
    continuation = idx[:, seq_len : seq_len + decode_steps]

    cfg = model.config
    n_layers = cfg.n_layer
    n_embd = cfg.n_embd
    dtype_bytes = 2

    if device != "cpu":
        _ = model.transformer.wte(prefix[:, :16])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    logits_pf, kv_cache, _ = prefill_dense(model, prefix)
    if device != "cpu":
        torch.cuda.synchronize()
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
        logits_step, kv, _, kept = decode_step_sparse(
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
        if device != "cpu":
            torch.cuda.synchronize()
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
    kv_bytes = kv_kept_mean * 2 * n_layers * n_embd * dtype_bytes

    peak_mem = 0.0
    if device != "cpu":
        peak_mem = torch.cuda.max_memory_allocated() / 1e6

    decode_arr = np.array(decode_latencies)
    total_decode_ms = sum(decode_latencies)

    return DecodeResult(
        method="static_topk",
        seq_len=seq_len,
        decode_steps=decode_steps,
        seed=seed,
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
        peak_mem_mb=peak_mem,
        region_ppl_early=ppl_early,
        region_ppl_late=ppl_late,
    )


# ============================================================
# Summary / report
# ============================================================


def print_summary_table(results: List[DecodeResult]):
    """Print a compact summary table."""
    print("\n" + "=" * 110)
    print(
        f"{'Method':15s} {'L':>5s} {'Steps':>5s} {'Prefill':>8s} "
        f"{'Decode':>8s} {'p95':>8s} {'Gate%':>6s} {'PPL':>8s} "
        f"{'KV_kept':>8s} {'KV_MB':>7s} {'tok/s':>8s} {'Mem_MB':>8s}"
    )
    print("-" * 110)
    for r in sorted(results, key=lambda x: (x.seq_len, x.method)):
        kv_mb = r.kv_bytes_read_per_token / 1e6
        print(
            f"{r.method:15s} {r.seq_len:5d} {r.decode_steps:5d} "
            f"{r.prefill_ms:7.1f}ms "
            f"{r.decode_per_token_ms:7.2f}ms "
            f"{r.decode_p95_ms:7.2f}ms "
            f"{r.gate_pct_of_total:5.1f}% "
            f"{r.ppl:8.1f} "
            f"{r.kv_kept_mean:7.0f} "
            f"{kv_mb:6.2f} "
            f"{r.throughput_toks_per_sec:7.0f} "
            f"{r.peak_mem_mb:7.0f}"
        )
    print("=" * 110)


def save_results(results: List[DecodeResult], output_dir: str):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "decode_results.json")
    data = [asdict(r) for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(results)} results to {path}")


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="BPA v9 decode benchmark")
    parser.add_argument(
        "--method",
        default="all",
        help="dense, bpa_v9, static_topk, or all",
    )
    parser.add_argument("--L", default="512,1024,2048", help="Sequence lengths")
    parser.add_argument("--steps", type=int, default=128, help="Decode steps")
    parser.add_argument("--seeds", default="1,2", help="Random seeds")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--output-dir", default="bpa_v9_results", help="Output directory"
    )
    parser.add_argument(
        "--stress",
        default="control",
        help="control, late_binding, kv_retrieval, all",
    )
    parser.add_argument("--W-min", type=int, default=64)
    parser.add_argument("--W-max", type=int, default=512)
    parser.add_argument("--B-far-max", type=int, default=8)
    parser.add_argument("--B-far-target", type=float, default=2.0)
    parser.add_argument(
        "--gate-every-k", type=int, default=4, help="Gate update frequency"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=64, help="Chunk size for selection"
    )
    args = parser.parse_args()

    seq_lens = [int(x) for x in args.L.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    methods = (
        ["dense", "bpa_v9", "static_topk"] if args.method == "all" else [args.method]
    )
    stress_modes = (
        ["control", "late_binding", "kv_retrieval"]
        if args.stress == "all"
        else [args.stress]
    )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    text_data = load_text_data("data/finewebedu/val.bin")
    surgical_heads = load_surgical_heads("configs/ra_surgical_gpt2.json")

    all_results = []
    max_L = max(seq_lens)

    print(f"Loading model for max L={max_L}...")
    model, cfg = load_model(max_L + args.steps, device)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"Device: {device}")
    print(f"Methods: {methods}")
    print(f"Seq lens: {seq_lens}")
    print(f"Decode steps: {args.steps}")
    print(f"Seeds: {seeds}")
    print(f"Stress modes: {stress_modes}")
    print()

    for L in seq_lens:
        for seed in seeds:
            for stress in stress_modes:
                stress_rng = np.random.RandomState(seed + 42)

                for method in methods:
                    print(
                        f"  {method:15s} L={L:5d} seed={seed} " f"stress={stress}...",
                        end="",
                        flush=True,
                    )

                    try:
                        if method == "dense":
                            if stress != "control":
                                print(" skip (dense only on control)")
                                continue
                            r = run_dense_decode(
                                model, text_data, L, args.steps, seed, device
                            )
                        elif method == "bpa_v9":
                            r = run_bpa_v9_decode(
                                model,
                                text_data,
                                L,
                                args.steps,
                                seed,
                                device,
                                surgical_heads,
                                chunk_size=args.chunk_size,
                                gate_every_k=args.gate_every_k,
                                W_min=args.W_min,
                                W_max=args.W_max,
                                B_far_max=args.B_far_max,
                                B_far_target=args.B_far_target,
                                stress_mode=stress,
                                stress_rng=stress_rng,
                            )
                        elif method == "static_topk":
                            if stress != "control":
                                print(" skip (static only on control)")
                                continue
                            r = run_static_topk_decode(
                                model,
                                text_data,
                                L,
                                args.steps,
                                seed,
                                device,
                                local_window=256,
                                far_budget=4,
                                chunk_size=args.chunk_size,
                            )
                        else:
                            print(f" unknown method {method}")
                            continue

                        r.stress_mode = stress
                        all_results.append(r)
                        print(
                            f" PPL={r.ppl:.1f} "
                            f"decode={r.decode_per_token_ms:.2f}ms "
                            f"kept={r.kv_kept_mean:.0f}"
                        )
                    except Exception as e:
                        print(f" ERROR: {e}")
                        import traceback

                        traceback.print_exc()

    print_summary_table(all_results)
    save_results(all_results, args.output_dir)

    # Quality comparison
    print("\n=== Quality Comparison (PPL vs Dense) ===")
    for L in seq_lens:
        dense_ppls = [
            r.ppl
            for r in all_results
            if r.method == "dense" and r.seq_len == L and r.stress_mode == "control"
        ]
        if not dense_ppls:
            continue
        dense_ppl = np.mean(dense_ppls)
        print(f"\nL={L}: Dense PPL = {dense_ppl:.1f}")
        for method in ["bpa_v9", "static_topk"]:
            method_ppls = [
                r.ppl
                for r in all_results
                if r.method == method and r.seq_len == L and r.stress_mode == "control"
            ]
            if method_ppls:
                m_ppl = np.mean(method_ppls)
                delta = (m_ppl - dense_ppl) / dense_ppl * 100
                within_1 = "YES" if abs(delta) <= 1 else "NO"
                within_3 = "YES" if abs(delta) <= 3 else "NO"
                print(
                    f"  {method:15s}: PPL={m_ppl:.1f} "
                    f"({delta:+.1f}%) "
                    f"within 1%: {within_1}, "
                    f"within 3%: {within_3}"
                )


if __name__ == "__main__":
    main()
