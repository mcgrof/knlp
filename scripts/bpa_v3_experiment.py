#!/usr/bin/env python
"""
BPA v3: Measure KV Cache Memory, FLOPs, and Time Gains.

Runs a grid of 4 variants across sequence lengths and seeds:
  V0: Dense baseline (full causal attention)
  V1: BPA local-only (far-context disabled always)
  V2: BPA learned-gate (v2 gate decides per-position)
  V3: BPA random-gate (matched enabled_rate to V2)

Measures:
  - val_ppl (cross-entropy perplexity)
  - tokens_per_query (mean, p50, p95, p99)
  - enabled_rate (fraction with far-context enabled)
  - effective_kept_tokens
  - kv_bytes_written/read/total per token (analytic)
  - flops_proxy
  - peak_kv_bytes (analytic; dense cache = unchanged)
  - wall_time ms/token

Usage:
    python scripts/bpa_v3_experiment.py \\
        --checkpoint <path> \\
        --seq-lens 512,1024 \\
        --seeds 1,2,3 \\
        --n-eval 50 \\
        --output-dir bpa_v3_results
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.bpa import BPAConfig, GPT2_BPA
from utils.kv_accounting import compute_kv_accounting, kv_bytes_per_token


# ---------------------------------------------------------------------------
# Gate loading from BPA v2
# ---------------------------------------------------------------------------
def load_v2_gate(gate_dir: str, seed: int = 1):
    """Load trained gate from BPA v2 Phase 1 results."""
    from scripts.bpa_v2_train_gate import MLPGate, N_FEATURES

    result_path = os.path.join(gate_dir, "phase1_results.json")
    if not os.path.exists(result_path):
        return None, None, None

    with open(result_path) as f:
        results = json.load(f)

    # Load gate checkpoint
    ckpt_path = os.path.join(gate_dir, f"mlp_gate_seed{seed}.pt")
    if not os.path.exists(ckpt_path):
        return None, None, None

    gate = MLPGate(N_FEATURES, hidden=128, n_layers=2)
    gate.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    gate.eval()

    # Load normalization stats
    stats_path = os.path.join(gate_dir, f"mlp_norm_stats_seed{seed}.npz")
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        return gate, stats["mean"], stats["std"]

    return gate, None, None


def train_quick_gate(data_dir: str, seed: int):
    """Train a quick gate from Phase 0 data if no saved gate exists."""
    from scripts.bpa_v2_train_gate import load_dataset, MLPGate, N_FEATURES

    features, labels = load_dataset(data_dir)

    torch.manual_seed(seed)
    np.random.seed(seed)

    thr = np.percentile(labels, 75)
    labels_binary = (labels > thr).astype(np.float32)
    pos_rate = labels_binary.mean()
    pos_weight = (1 - pos_rate) / (pos_rate + 1e-10)

    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0) + 1e-8
    features_norm = (features - feat_mean) / (feat_std + 1e-8)

    gate = MLPGate(N_FEATURES, hidden=128, n_layers=2)
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)
    pw = torch.tensor([pos_weight])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    train_x = torch.tensor(features_norm, dtype=torch.float32)
    train_y = torch.tensor(labels_binary, dtype=torch.float32)

    gate.train()
    for epoch in range(20):
        optimizer.zero_grad()
        chunk = min(10000, len(train_x))
        perm = torch.randperm(len(train_x))[:chunk]
        logits = gate(train_x[perm])
        loss = criterion(logits, train_y[perm])
        loss.backward()
        optimizer.step()

    gate.eval()
    return gate, feat_mean, feat_std


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_text_data(path: str) -> np.ndarray:
    """Load pre-tokenized text data."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_text_batch(
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    rng: np.random.RandomState,
) -> torch.Tensor:
    """Get a batch of real tokenized text."""
    max_start = len(data) - seq_len - 1
    starts = rng.randint(0, max_start, size=batch_size)
    batch = torch.stack(
        [torch.from_numpy(data[s : s + seq_len].astype(np.int64)) for s in starts]
    )
    return batch


# ---------------------------------------------------------------------------
# Manual forward pass with flexible attention masking
# ---------------------------------------------------------------------------
def build_local_mask(T: int, local_window: int, device: torch.device) -> torch.Tensor:
    """Build local-only attention mask. True = mask out."""
    local_mask = torch.ones(T, T, dtype=torch.bool, device=device)
    for t in range(T):
        start = max(0, t - local_window + 1)
        local_mask[t, start : t + 1] = False
    return local_mask


@torch.no_grad()
def manual_forward(
    model: GPT2_BPA,
    idx: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run manual forward with optional attention mask.

    Args:
        model: GPT2_BPA model
        idx: [B, T] input token ids
        attn_mask: [T, T] bool, True = mask out. If None, full causal.

    Returns:
        logits: [B, T, vocab_size]
    """
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
    if attn_mask is not None:
        combined_mask = causal_mask | attn_mask
    else:
        combined_mask = causal_mask

    for block in model.transformer.h:
        h = block.ln_1(x)
        attn = block.attn

        qkv = attn.c_attn(h)
        q, k, v = qkv.split(n_embd, dim=2)
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        scale = 1.0 / (head_dim**0.5)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(
            combined_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        attn_out = attn.resid_dropout(attn.c_proj(attn_out))

        x = x + attn_out
        x = x + block.mlp(block.ln_2(x))

    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)
    return logits


@torch.no_grad()
def manual_forward_with_per_position_mask(
    model: GPT2_BPA,
    idx: torch.Tensor,
    gate_decisions: np.ndarray,
    local_window: int,
) -> torch.Tensor:
    """Forward pass with per-position gating.

    For positions where gate_decisions is False, restrict attention to
    local_window. For positions where True, allow full causal attention.

    gate_decisions: [n_valid] bool array for positions local_window..T-1

    Returns:
        logits: [B, T, vocab_size]
    """
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

    # Build per-position causal mask with local restriction
    # causal_mask[t, s] = True means mask out (s > t)
    causal_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
    )

    # Build local mask for positions where gate is off
    local_restrict = torch.zeros(T, T, dtype=torch.bool, device=device)
    for ti, t in enumerate(range(local_window, T)):
        if not gate_decisions[ti]:
            # Mask out far-context for this position
            far_end = max(0, t - local_window + 1)
            if far_end > 0:
                local_restrict[t, :far_end] = True

    combined_mask = causal_mask | local_restrict

    for block in model.transformer.h:
        h = block.ln_1(x)
        attn = block.attn

        qkv = attn.c_attn(h)
        q, k, v = qkv.split(n_embd, dim=2)
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        scale = 1.0 / (head_dim**0.5)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(
            combined_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        attn_out = attn.resid_dropout(attn.c_proj(attn_out))

        x = x + attn_out
        x = x + block.mlp(block.ln_2(x))

    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)
    return logits


# ---------------------------------------------------------------------------
# Gate feature extraction (simplified from v2 frontier)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_gate_features(
    model: GPT2_BPA,
    idx: torch.Tensor,
    local_window: int,
) -> np.ndarray:
    """Extract local-only features for gate prediction.

    Returns features array of shape [n_valid, 7] where
    n_valid = T - local_window, averaged across batch and heads.
    """
    B, T = idx.shape
    device = idx.device
    cfg = model.config
    n_head = cfg.n_head
    n_embd = cfg.n_embd
    head_dim = n_embd // n_head
    n_valid = max(T - local_window, 0)

    if n_valid == 0:
        return np.zeros((0, 7))

    # We only need Q and local attention for features
    pos = torch.arange(0, T, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)

    causal_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
    )
    local_mask = build_local_mask(T, local_window, device)
    combined_mask = causal_mask | local_mask

    # Accumulate features across layers
    all_feats = np.zeros((n_valid, 7))
    n_layers_counted = 0

    for block in model.transformer.h:
        h = block.ln_1(x)
        attn = block.attn
        qkv = attn.c_attn(h)
        q, k, v = qkv.split(n_embd, dim=2)
        q_heads = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k_heads = k.view(B, T, n_head, head_dim).transpose(1, 2)

        scale = 1.0 / (head_dim**0.5)
        scores = (q_heads @ k_heads.transpose(-2, -1)) * scale
        scores = scores.masked_fill(
            combined_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        local_attn_probs = F.softmax(scores, dim=-1)

        for ti, t in enumerate(range(local_window, T)):
            local_start = max(0, t - local_window + 1)
            local_end = t + 1
            pos_norm = t / max(T - 1, 1)
            bucket = 0 if pos_norm < 0.33 else (1 if pos_norm < 0.66 else 2)
            boundary_band = min(8, local_window // 4)

            # Average across batch and heads
            w = local_attn_probs[:, :, t, local_start:local_end]  # [B, H, W]
            w_norm = w / (w.sum(dim=-1, keepdim=True) + 1e-10)

            ent = -(w_norm * (w_norm + 1e-10).log()).sum(dim=-1)  # [B, H]
            max_ent = math.log(local_end - local_start + 1e-10)
            feat_entropy = (ent / (max_ent + 1e-10)).mean().item()
            feat_max = w_norm.max(dim=-1)[0].mean().item()

            band_end = min(local_start + boundary_band, local_end)
            feat_band = (
                local_attn_probs[:, :, t, local_start:band_end]
                .sum(dim=-1)
                .mean()
                .item()
            )

            q_norm_val = q_heads[:, :, t, :].norm(dim=-1).mean().item()

            all_feats[ti] += np.array(
                [feat_entropy, feat_max, feat_band, q_norm_val, 1.0, pos_norm, bucket]
            )

        n_layers_counted += 1
        x, _ = block(x)

    # Average across layers
    all_feats /= n_layers_counted
    return all_feats


# ---------------------------------------------------------------------------
# Per-variant evaluation
# ---------------------------------------------------------------------------
@dataclass
class RunMetrics:
    """Metrics for a single experiment run."""

    variant: str
    seq_len: int
    seed: int
    # PPL
    ppl_mean: float
    ppl_std: float
    # Gate
    enabled_rate: float
    # Tokens per query
    tokens_per_query_mean: float
    tokens_per_query_p50: float
    tokens_per_query_p95: float
    tokens_per_query_p99: float
    # KV accounting (analytic)
    kv_bytes_written_per_token: float
    kv_bytes_read_per_token: float
    kv_bytes_total_per_token: float
    peak_kv_bytes: float
    effective_kept_tokens: float
    # FLOPs
    flops_proxy: float
    flops_relative: float  # relative to V0 dense
    # Time
    wall_ms_per_token: float
    tokens_per_sec: float
    # Tokens seen
    tokens_seen: int
    n_eval_batches: int


def compute_ppl(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute perplexity from logits and targets."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
    )
    return float(torch.exp(loss))


def evaluate_variant(
    model: GPT2_BPA,
    text_data: np.ndarray,
    variant: str,
    seq_len: int,
    seed: int,
    n_eval: int,
    batch_size: int,
    local_window: int,
    gate=None,
    feat_mean=None,
    feat_std=None,
    gate_threshold: float = 0.5,
    target_enabled_rate: float = 0.5,
) -> RunMetrics:
    """Evaluate a single variant and return metrics."""
    cfg = model.config
    n_layers = cfg.n_layer
    d_model = cfg.n_embd
    rng = np.random.RandomState(seed)

    ppls = []
    tokens_per_query_list = []
    enabled_rates = []
    wall_times = []
    tokens_seen = 0

    # Warmup
    warmup_idx = get_text_batch(text_data, batch_size, seq_len, rng)
    _ = manual_forward(model, warmup_idx)

    for i in range(n_eval):
        idx = get_text_batch(text_data, batch_size, seq_len, rng)
        B, T = idx.shape

        t0 = time.perf_counter()

        if variant == "V0_dense":
            logits = manual_forward(model, idx)
            enabled_rate_batch = 1.0
            kept = float(T)

        elif variant == "V1_local_only":
            local_mask = build_local_mask(T, local_window, idx.device)
            logits = manual_forward(model, idx, attn_mask=local_mask)
            enabled_rate_batch = 0.0
            kept = float(min(T, local_window))

        elif variant == "V2_learned_gate":
            # Extract features
            feats = compute_gate_features(model, idx, local_window)
            n_valid = len(feats)
            if n_valid > 0 and gate is not None:
                feats_norm = (feats - feat_mean) / (feat_std + 1e-8)
                gate_logits = (
                    gate(torch.tensor(feats_norm, dtype=torch.float32))
                    .detach()
                    .numpy()
                    .flatten()
                )
                gate_probs = 1.0 / (1.0 + np.exp(-np.clip(gate_logits, -500, 500)))
                gate_decisions = gate_probs > gate_threshold
                enabled_rate_batch = float(gate_decisions.mean())
            else:
                gate_decisions = np.zeros(max(T - local_window, 0), dtype=bool)
                enabled_rate_batch = 0.0

            logits = manual_forward_with_per_position_mask(
                model, idx, gate_decisions, local_window
            )
            # Effective kept: local for all, plus full for enabled positions
            n_valid_pos = max(T - local_window, 0)
            n_local_only = T - n_valid_pos  # first local_window positions
            avg_enabled_far = enabled_rate_batch * (T - local_window)
            kept = float(local_window + avg_enabled_far)
            # But cap: positions before local_window attend full (up to T)
            # weighted average across all positions
            if T > 0:
                # Positions 0..local_window-1: attend up to min(t+1, T)
                # Positions local_window..T-1: attend local_window or T
                total_attended = 0
                for t in range(T):
                    if t < local_window:
                        total_attended += min(t + 1, T)
                    else:
                        ti = t - local_window
                        if gate_decisions[ti]:
                            total_attended += min(t + 1, T)
                        else:
                            total_attended += local_window
                kept = total_attended / T

        elif variant == "V3_random_gate":
            n_valid = max(T - local_window, 0)
            gate_decisions = rng.random(n_valid) < target_enabled_rate
            enabled_rate_batch = float(gate_decisions.mean())

            logits = manual_forward_with_per_position_mask(
                model, idx, gate_decisions, local_window
            )
            # Same kept calculation as V2
            if T > 0:
                total_attended = 0
                for t in range(T):
                    if t < local_window:
                        total_attended += min(t + 1, T)
                    else:
                        ti = t - local_window
                        if gate_decisions[ti]:
                            total_attended += min(t + 1, T)
                        else:
                            total_attended += local_window
                kept = total_attended / T
            else:
                kept = 0.0
        else:
            raise ValueError(f"Unknown variant: {variant}")

        t1 = time.perf_counter()

        ppl = compute_ppl(logits, idx)
        ppls.append(ppl)
        tokens_per_query_list.append(kept)
        enabled_rates.append(enabled_rate_batch)
        wall_times.append((t1 - t0) * 1000)  # ms
        tokens_seen += B * T

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"    [{i+1}/{n_eval}] "
                f"PPL={ppl:.1f} "
                f"kept={kept:.0f} "
                f"rate={enabled_rate_batch:.3f} "
                f"ms={wall_times[-1]:.0f}"
            )

    # Aggregate
    tpq = np.array(tokens_per_query_list)
    wall = np.array(wall_times)
    mean_enabled_rate = float(np.mean(enabled_rates))
    mean_kept = float(np.mean(tpq))

    # KV accounting
    kv = compute_kv_accounting(
        seq_len=seq_len,
        n_layers=n_layers,
        d_model=d_model,
        local_window=local_window if variant != "V0_dense" else seq_len,
        enabled_rate=(
            mean_enabled_rate
            if variant not in ("V0_dense", "V1_local_only")
            else (1.0 if variant == "V0_dense" else 0.0)
        ),
        bytes_per_elem=2,
    )

    # Dense baseline FLOPs for relative computation
    dense_flops = seq_len * d_model * n_layers

    # Time per token
    total_tokens_per_batch = batch_size * seq_len
    ms_per_token = float(np.median(wall)) / total_tokens_per_batch

    return RunMetrics(
        variant=variant,
        seq_len=seq_len,
        seed=seed,
        ppl_mean=float(np.mean(ppls)),
        ppl_std=float(np.std(ppls)),
        enabled_rate=mean_enabled_rate,
        tokens_per_query_mean=mean_kept,
        tokens_per_query_p50=float(np.percentile(tpq, 50)),
        tokens_per_query_p95=float(np.percentile(tpq, 95)),
        tokens_per_query_p99=float(np.percentile(tpq, 99)),
        kv_bytes_written_per_token=kv.kv_bytes_written_per_token,
        kv_bytes_read_per_token=kv.kv_bytes_read_per_token,
        kv_bytes_total_per_token=kv.kv_bytes_total_per_token,
        peak_kv_bytes=kv.peak_kv_bytes,
        effective_kept_tokens=mean_kept,
        flops_proxy=kv.flops_proxy,
        flops_relative=kv.flops_proxy / dense_flops if dense_flops > 0 else 1.0,
        wall_ms_per_token=ms_per_token,
        tokens_per_sec=1000.0 / ms_per_token if ms_per_token > 0 else 0.0,
        tokens_seen=tokens_seen,
        n_eval_batches=n_eval,
    )


# ---------------------------------------------------------------------------
# Main grid runner
# ---------------------------------------------------------------------------
def run_grid(
    checkpoint: str,
    seq_lens: List[int],
    seeds: List[int],
    n_eval: int,
    batch_size: int,
    local_window: int,
    chunk_size: int,
    top_b: int,
    gate_data_dir: str,
    text_data_path: str,
    output_dir: str,
    gate_threshold: float = 0.5,
) -> Dict:
    """Run the full experiment grid."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "per_run_metrics"), exist_ok=True)

    # Load text data
    text_data = load_text_data(text_data_path)
    print(f"Text data: {len(text_data)} tokens from {text_data_path}")

    # Load or train gate
    gate, feat_mean, feat_std = None, None, None
    if os.path.exists(gate_data_dir):
        print(f"Training gate from {gate_data_dir}...")
        gate, feat_mean, feat_std = train_quick_gate(gate_data_dir, seed=1)
        print("  Gate trained.")

    variants = ["V0_dense", "V1_local_only", "V2_learned_gate", "V3_random_gate"]
    all_results = []

    for seq_len in seq_lens:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"SEQ_LEN={seq_len}, SEED={seed}")
            print(f"{'='*60}")

            # Load model
            block_size = seq_len
            if checkpoint:
                ckpt_probe = torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                )
                wpe_size = ckpt_probe["model"]["transformer.wpe.weight"].shape[0]
                block_size = max(seq_len, wpe_size)
                del ckpt_probe

            cfg = BPAConfig(
                block_size=block_size,
                vocab_size=50304,
                n_layer=12,
                n_head=12,
                n_embd=768,
                local_window=local_window,
                chunk_size=chunk_size,
                top_b=top_b,
            )

            torch.manual_seed(seed)
            model = GPT2_BPA(cfg)

            if checkpoint:
                ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt["model"])
                print(f"  Loaded checkpoint (iter={ckpt.get('iter_num', '?')})")
                del ckpt

            model.eval()

            # First pass: get V2 enabled_rate for V3 matching
            v2_rate = 0.5  # default
            seed_results = {}

            for variant in variants:
                print(f"\n  --- {variant} ---")

                target_rate = v2_rate if variant == "V3_random_gate" else 0.5
                metrics = evaluate_variant(
                    model=model,
                    text_data=text_data,
                    variant=variant,
                    seq_len=seq_len,
                    seed=seed,
                    n_eval=n_eval,
                    batch_size=batch_size,
                    local_window=local_window,
                    gate=gate,
                    feat_mean=feat_mean,
                    feat_std=feat_std,
                    gate_threshold=gate_threshold,
                    target_enabled_rate=target_rate,
                )

                if variant == "V2_learned_gate":
                    v2_rate = metrics.enabled_rate
                    print(f"  V2 enabled_rate={v2_rate:.3f} (V3 will match)")

                seed_results[variant] = metrics
                all_results.append(metrics)

                # Save per-run metrics
                run_key = f"{variant}_L{seq_len}_S{seed}"
                run_path = os.path.join(
                    output_dir, "per_run_metrics", f"{run_key}.json"
                )
                with open(run_path, "w") as f:
                    json.dump(asdict(metrics), f, indent=2)

            # Print comparison table for this (seq_len, seed)
            print(
                f"\n  {'Variant':<20} {'PPL':>8} {'Rate':>6} {'Kept':>6} "
                f"{'KV_R_KB':>8} {'FLOPs%':>7} {'ms/tok':>7}"
            )
            print(f"  {'-'*68}")
            dense_flops = seed_results["V0_dense"].flops_proxy
            for v in variants:
                m = seed_results[v]
                print(
                    f"  {v:<20} {m.ppl_mean:>8.1f} {m.enabled_rate:>6.3f} "
                    f"{m.effective_kept_tokens:>6.0f} "
                    f"{m.kv_bytes_read_per_token/1024:>8.1f} "
                    f"{m.flops_proxy/dense_flops*100:>6.1f}% "
                    f"{m.wall_ms_per_token:>7.3f}"
                )

    # Save all results
    raw_path = os.path.join(output_dir, "raw_results.json")
    with open(raw_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nAll results saved to {raw_path}")

    return {"results": all_results, "raw_path": raw_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BPA v3: KV Cache Memory, FLOPs, and Time Measurement"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Trained model checkpoint"
    )
    parser.add_argument("--seq-lens", type=str, default="512", help="Sequence lengths")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Seeds")
    parser.add_argument("--n-eval", type=int, default=50, help="Eval batches")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--local-window", type=int, default=256, help="Local window")
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunk size")
    parser.add_argument("--top-b", type=int, default=8, help="Top-B budget")
    parser.add_argument(
        "--gate-data-dir",
        type=str,
        default="bpa_v2_trained_dataset",
        help="Phase 0 gate training data",
    )
    parser.add_argument(
        "--text-data",
        type=str,
        default="gpt2/data/finewebedu/val.bin",
        help="Tokenized text for eval",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bpa_v3_results",
        help="Output directory",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=0.5,
        help="Gate threshold",
    )
    args = parser.parse_args()

    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 70)
    print("BPA v3: KV Cache Memory, FLOPs, and Time Measurement")
    print("=" * 70)
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Seq lens:     {seq_lens}")
    print(f"  Seeds:        {seeds}")
    print(f"  n_eval:       {args.n_eval}")
    print(f"  batch_size:   {args.batch_size}")
    print(f"  local_window: {args.local_window}")
    print(f"  text_data:    {args.text_data}")
    print(f"  output_dir:   {args.output_dir}")

    run_grid(
        checkpoint=args.checkpoint,
        seq_lens=seq_lens,
        seeds=seeds,
        n_eval=args.n_eval,
        batch_size=args.batch_size,
        local_window=args.local_window,
        chunk_size=args.chunk_size,
        top_b=args.top_b,
        gate_data_dir=args.gate_data_dir,
        text_data_path=args.text_data,
        output_dir=args.output_dir,
        gate_threshold=args.gate_threshold,
    )


if __name__ == "__main__":
    main()
