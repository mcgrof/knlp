#!/usr/bin/env python
"""
BPA v5: RA-guided far-context selection for Boundary-Pressure Attention.

Tests whether RA inbound mass (computed from surgical heads) improves
far-context chunk selection compared to recency or random baselines.

At a fixed far budget (number of chunks retrieved when gate enables far),
different selection strategies pick WHICH far chunks to attend to:

  1. recency: most recent chunks first (baseline)
  2. random: random chunk selection
  3. ra_value: chunks with highest RA inbound mass
  4. ra_blend: RA_value * exp(-age/tau) (value + recency blend)

The boundary_pressure gate controls WHETHER far is enabled (demand side).
The selection strategy controls WHICH far chunks to use (supply side).

Usage:
    python scripts/bpa_v5_experiment.py \
        --checkpoint <path> \
        --gate-dir bpa_v4_gate_results \
        --seq-lens 512,1024 \
        --seeds 1,2,3 \
        --n-eval 50 \
        --far-budget 4 \
        --output-dir bpa_v5_results
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.bpa import BPAConfig, GPT2_BPA
from scripts.bpa_v3_experiment import (
    RunMetrics,
    build_local_mask,
    compute_gate_features,
    compute_ppl,
    get_text_batch,
    load_text_data,
)
from scripts.bpa_v4_gate import V4Gate, find_budget_threshold
from utils.kv_accounting import compute_kv_accounting
from utils.ra_value_tracker import (
    RAValueTracker,
    load_surgical_heads,
)


def load_v4_gate(gate_dir: str, seed: int = 1):
    """Load a trained v4 gate and normalization stats."""
    ckpt_path = os.path.join(gate_dir, f"v4_best_gate_seed{seed}.pt")
    stats_path = os.path.join(gate_dir, f"v4_norm_stats_seed{seed}.npz")

    if not os.path.exists(ckpt_path):
        print(f"  WARNING: No v4 gate at {ckpt_path}")
        return None, None, None

    results_path = os.path.join(gate_dir, f"v4_gate_seed{seed}_detail.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            detail = json.load(f)
        n_features = len(detail["normalization"]["mean"])
    else:
        n_features = 7

    # Detect architecture from checkpoint
    ckpt_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    first_weight = ckpt_sd["net.0.weight"]
    hidden = first_weight.shape[0]
    n_layers_gate = sum(1 for k in ckpt_sd if k.endswith(".weight")) - 1
    gate = V4Gate(n_features, hidden=hidden, n_layers=n_layers_gate)
    gate.load_state_dict(ckpt_sd)
    gate.eval()

    stats = np.load(stats_path)
    return gate, stats["mean"], stats["std"]


def select_far_chunks(
    strategy: str,
    n_chunks: int,
    far_budget: int,
    ra_chunk_values: Optional[torch.Tensor],
    query_pos: int,
    chunk_size: int,
    local_window: int,
    tau: float = 4.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Select which far chunks to attend to.

    Args:
        strategy: "recency", "random", "ra_value", or "ra_blend"
        n_chunks: total number of chunks in sequence
        far_budget: how many chunks to select
        ra_chunk_values: [n_chunks] RA value scores (or None)
        query_pos: current query position
        chunk_size: tokens per chunk
        local_window: local attention window size
        tau: decay constant for ra_blend (in chunk units)
        rng: random state for "random" strategy

    Returns:
        Array of selected chunk indices (sorted)
    """
    # Determine which chunks are "far" (not in local window)
    local_end_pos = max(0, query_pos - local_window + 1)
    local_end_chunk = local_end_pos // chunk_size
    # Far chunks are [0, local_end_chunk)
    far_chunks = list(range(local_end_chunk))

    if len(far_chunks) == 0:
        return np.array([], dtype=np.int64)

    budget = min(far_budget, len(far_chunks))

    if strategy == "recency":
        # Most recent far chunks first
        selected = far_chunks[-budget:]
    elif strategy == "random":
        if rng is not None:
            selected = rng.choice(far_chunks, size=budget, replace=False)
        else:
            selected = np.random.choice(far_chunks, size=budget, replace=False)
    elif strategy == "ra_value":
        if ra_chunk_values is not None and len(ra_chunk_values) > 0:
            # Select chunks with highest RA inbound mass
            values = ra_chunk_values.cpu().numpy()
            far_values = [(c, values[c]) for c in far_chunks if c < len(values)]
            far_values.sort(key=lambda x: x[1], reverse=True)
            selected = [c for c, _ in far_values[:budget]]
        else:
            # Fallback to recency if no RA values
            selected = far_chunks[-budget:]
    elif strategy == "ra_blend":
        if ra_chunk_values is not None and len(ra_chunk_values) > 0:
            values = ra_chunk_values.cpu().numpy()
            query_chunk = query_pos // chunk_size
            scores = []
            for c in far_chunks:
                if c < len(values):
                    age = query_chunk - c
                    score = values[c] * math.exp(-age / tau)
                    scores.append((c, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            selected = [c for c, _ in scores[:budget]]
        else:
            selected = far_chunks[-budget:]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return np.sort(np.array(selected, dtype=np.int64))


@torch.no_grad()
def forward_with_chunk_selection(
    model: GPT2_BPA,
    idx: torch.Tensor,
    gate_decisions: np.ndarray,
    local_window: int,
    chunk_size: int,
    strategy: str,
    far_budget: int,
    ra_chunk_values: Optional[torch.Tensor] = None,
    tau: float = 4.0,
    rng: Optional[np.random.RandomState] = None,
    tracker: Optional[RAValueTracker] = None,
) -> Tuple[torch.Tensor, dict]:
    """Forward pass with per-position gating AND chunk-level selection.

    When gate_decisions[t] is True, select far_budget chunks using the
    given strategy. When False, restrict to local window only.

    Also collects RA inbound mass stats if tracker is provided.

    Returns:
        logits: [B, T, vocab_size]
        info: dict with attended_chunks, ra_stats
    """
    B, T = idx.shape
    device = idx.device
    cfg = model.config
    n_head = cfg.n_head
    n_embd = cfg.n_embd
    head_dim = n_embd // n_head
    n_chunks = (T + chunk_size - 1) // chunk_size

    pos = torch.arange(0, T, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)

    # Build per-position attention mask
    # For each position t >= local_window:
    #   if gate ON: allow local + selected far chunks
    #   if gate OFF: allow local only
    causal_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
    )

    # Start with causal + local-only restriction
    local_restrict = torch.zeros(T, T, dtype=torch.bool, device=device)
    chunks_attended = []

    for ti, t in enumerate(range(local_window, T)):
        far_end = max(0, t - local_window + 1)
        if far_end > 0:
            if gate_decisions[ti]:
                # Gate ON: select specific far chunks
                selected = select_far_chunks(
                    strategy=strategy,
                    n_chunks=n_chunks,
                    far_budget=far_budget,
                    ra_chunk_values=ra_chunk_values,
                    query_pos=t,
                    chunk_size=chunk_size,
                    local_window=local_window,
                    tau=tau,
                    rng=rng,
                )
                chunks_attended.append(len(selected))

                # Mask out all far positions first
                local_restrict[t, :far_end] = True
                # Then unmask selected chunks
                for c in selected:
                    c_start = c * chunk_size
                    c_end = min((c + 1) * chunk_size, far_end)
                    if c_start < far_end:
                        local_restrict[t, c_start:c_end] = False
            else:
                # Gate OFF: local only
                local_restrict[t, :far_end] = True
                chunks_attended.append(0)

    combined_mask = causal_mask | local_restrict

    if tracker is not None:
        tracker.reset()

    for layer_idx, block in enumerate(model.transformer.h):
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

        if tracker is not None and layer_idx in tracker.heads_by_layer:
            tracker.accumulate(layer_idx, attn_weights)

        attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        attn_out = attn.resid_dropout(attn.c_proj(attn_out))

        x = x + attn_out
        x = x + block.mlp(block.ln_2(x))

    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)

    info = {
        "chunks_attended_mean": (
            float(np.mean(chunks_attended)) if chunks_attended else 0.0
        ),
    }
    if tracker is not None:
        info["ra_stats"] = tracker.get_stats()

    return logits, info


def evaluate_v5_variant(
    model: GPT2_BPA,
    text_data: np.ndarray,
    variant: str,
    strategy: str,
    seq_len: int,
    seed: int,
    n_eval: int,
    batch_size: int,
    local_window: int,
    chunk_size: int,
    far_budget: int,
    gate=None,
    feat_mean=None,
    feat_std=None,
    target_enabled_rate: float = 0.7,
    tau: float = 4.0,
    tracker: Optional[RAValueTracker] = None,
) -> RunMetrics:
    """Evaluate a v5 variant with chunk-level selection."""
    cfg = model.config
    n_layers = cfg.n_layer
    d_model = cfg.n_embd
    rng = np.random.RandomState(seed)

    ppls = []
    tokens_per_query_list = []
    enabled_rates = []
    wall_times = []
    tokens_seen = 0
    ra_stats_accum = {}

    # Warmup: run one batch to collect initial RA values
    warmup_idx = get_text_batch(text_data, batch_size, seq_len, rng)
    if tracker is not None:
        _, _ = forward_with_chunk_selection(
            model,
            warmup_idx,
            np.ones(max(seq_len - local_window, 0), dtype=bool),
            local_window,
            chunk_size,
            "recency",
            far_budget,
            tracker=tracker,
        )
        ra_values = tracker.get_chunk_values()
    else:
        ra_values = None

    for i in range(n_eval):
        idx = get_text_batch(text_data, batch_size, seq_len, rng)
        B, T = idx.shape
        n_valid = max(T - local_window, 0)

        t0 = time.perf_counter()

        # Gate decisions (use v4 gate at target_enabled_rate)
        if variant == "V0_dense":
            gate_decisions = np.ones(n_valid, dtype=bool)
        elif variant == "V1_local_only":
            gate_decisions = np.zeros(n_valid, dtype=bool)
        elif gate is not None and n_valid > 0:
            feats = compute_gate_features(model, idx, local_window)
            feats_norm = (feats - feat_mean) / (feat_std + 1e-8)
            gate_logits = (
                gate(torch.tensor(feats_norm, dtype=torch.float32))
                .detach()
                .numpy()
                .flatten()
            )
            gate_probs = 1.0 / (1.0 + np.exp(-np.clip(gate_logits, -500, 500)))
            threshold = find_budget_threshold(gate_probs, target_enabled_rate)
            gate_decisions = gate_probs >= threshold
        else:
            gate_decisions = np.ones(n_valid, dtype=bool)

        enabled_rate_batch = float(gate_decisions.mean()) if n_valid > 0 else 0.0

        logits, info = forward_with_chunk_selection(
            model,
            idx,
            gate_decisions,
            local_window,
            chunk_size,
            strategy,
            far_budget,
            ra_chunk_values=ra_values,
            tau=tau,
            rng=rng,
            tracker=tracker,
        )

        # Update RA values after each batch
        if tracker is not None:
            ra_values = tracker.get_chunk_values()

        t1 = time.perf_counter()

        # Compute effective kept tokens
        if T > 0:
            total_attended = 0
            for t in range(T):
                if t < local_window:
                    total_attended += min(t + 1, T)
                else:
                    ti = t - local_window
                    if gate_decisions[ti]:
                        total_attended += local_window + far_budget * chunk_size
                    else:
                        total_attended += local_window
            kept = total_attended / T
        else:
            kept = 0.0

        ppl = compute_ppl(logits, idx)
        ppls.append(ppl)
        tokens_per_query_list.append(kept)
        enabled_rates.append(enabled_rate_batch)
        wall_times.append((t1 - t0) * 1000)
        tokens_seen += B * T

        # Accumulate RA stats
        if "ra_stats" in info:
            for k, v in info["ra_stats"].items():
                ra_stats_accum.setdefault(k, []).append(v)

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"    [{i+1}/{n_eval}] "
                f"PPL={ppl:.1f} "
                f"kept={kept:.0f} "
                f"rate={enabled_rate_batch:.3f} "
                f"chunks={info.get('chunks_attended_mean', 0):.1f} "
                f"ms={wall_times[-1]:.0f}"
            )

    tpq = np.array(tokens_per_query_list)
    wall = np.array(wall_times)
    mean_enabled_rate = float(np.mean(enabled_rates))
    mean_kept = float(np.mean(tpq))

    kv = compute_kv_accounting(
        seq_len=seq_len,
        n_layers=n_layers,
        d_model=d_model,
        local_window=local_window,
        enabled_rate=mean_enabled_rate,
        bytes_per_elem=2,
    )

    dense_flops = seq_len * d_model * n_layers
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
        flops_relative=(kv.flops_proxy / dense_flops if dense_flops > 0 else 1.0),
        wall_ms_per_token=ms_per_token,
        tokens_per_sec=1000.0 / ms_per_token if ms_per_token > 0 else 0.0,
        tokens_seen=tokens_seen,
        n_eval_batches=n_eval,
    )


def run_v5_grid(
    checkpoint: str,
    seq_lens: List[int],
    seeds: List[int],
    n_eval: int,
    batch_size: int,
    local_window: int,
    chunk_size: int,
    top_b: int,
    far_budget: int,
    v4_gate_dir: str,
    text_data_path: str,
    output_dir: str,
    target_enabled_rate: float = 0.7,
    tau: float = 4.0,
) -> Dict:
    """Run the full v5 experiment grid."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "per_run_metrics"), exist_ok=True)

    text_data = load_text_data(text_data_path)
    print(f"Text data: {len(text_data)} tokens from {text_data_path}")

    # Load surgical heads for RA tracker
    surgical_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "ra_surgical_gpt2.json",
    )
    surgical_heads = load_surgical_heads(surgical_path)
    print(f"Surgical heads: {len(surgical_heads)} heads from {surgical_path}")

    all_results = []
    strategies = ["recency", "random", "ra_value", "ra_blend"]

    for seq_len in seq_lens:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"SEQ_LEN={seq_len}, SEED={seed}")
            print(f"{'='*60}")

            # Load model
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            vocab_size = ckpt["model"]["transformer.wte.weight"].shape[0]

            cfg = BPAConfig(
                n_layer=12,
                n_head=12,
                n_embd=768,
                local_window=local_window,
                chunk_size=chunk_size,
                top_b=top_b,
                vocab_size=vocab_size,
                block_size=max(seq_len, 1024),
            )
            model = GPT2_BPA(cfg)
            model_sd = {
                k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()
            }
            model.load_state_dict(model_sd, strict=False)
            model.eval()
            del ckpt

            device = torch.device("cpu")
            model = model.to(device)

            # Load gate
            gate, feat_mean, feat_std = load_v4_gate(v4_gate_dir, seed)

            # V0: Dense baseline
            print(f"\n  V0_dense (full causal)")
            tracker = RAValueTracker(surgical_heads, chunk_size=chunk_size)
            metrics = evaluate_v5_variant(
                model,
                text_data,
                "V0_dense",
                "recency",
                seq_len,
                seed,
                n_eval,
                batch_size,
                local_window,
                chunk_size,
                far_budget=9999,
                target_enabled_rate=1.0,
                tracker=tracker,
            )
            all_results.append(metrics)
            fname = f"V0_dense_L{seq_len}_S{seed}.json"
            with open(os.path.join(output_dir, "per_run_metrics", fname), "w") as f:
                json.dump(asdict(metrics), f, indent=2)

            # V1: Local only
            print(f"\n  V1_local_only")
            metrics = evaluate_v5_variant(
                model,
                text_data,
                "V1_local_only",
                "recency",
                seq_len,
                seed,
                n_eval,
                batch_size,
                local_window,
                chunk_size,
                far_budget=0,
                target_enabled_rate=0.0,
            )
            all_results.append(metrics)
            fname = f"V1_local_only_L{seq_len}_S{seed}.json"
            with open(os.path.join(output_dir, "per_run_metrics", fname), "w") as f:
                json.dump(asdict(metrics), f, indent=2)

            # Selection strategy variants
            for strat in strategies:
                variant = f"V5_{strat}"
                print(
                    f"\n  {variant} (budget={far_budget}, rate={target_enabled_rate})"
                )
                tracker = RAValueTracker(surgical_heads, chunk_size=chunk_size)
                metrics = evaluate_v5_variant(
                    model,
                    text_data,
                    variant,
                    strat,
                    seq_len,
                    seed,
                    n_eval,
                    batch_size,
                    local_window,
                    chunk_size,
                    far_budget=far_budget,
                    gate=gate,
                    feat_mean=feat_mean,
                    feat_std=feat_std,
                    target_enabled_rate=target_enabled_rate,
                    tau=tau,
                    tracker=tracker if strat in ("ra_value", "ra_blend") else None,
                )
                all_results.append(metrics)
                fname = f"{variant}_L{seq_len}_S{seed}.json"
                with open(os.path.join(output_dir, "per_run_metrics", fname), "w") as f:
                    json.dump(asdict(metrics), f, indent=2)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save all results
    results_path = os.path.join(output_dir, "raw_results.json")
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nSaved {len(all_results)} results to {results_path}")

    return {"results": [asdict(r) for r in all_results]}


def main():
    parser = argparse.ArgumentParser(description="BPA v5 experiment")
    parser.add_argument(
        "--checkpoint",
        default="test_matrix_results_20260206_184612/gpt2_adamwspam_none_none/final_model_stepV0.pt",
    )
    parser.add_argument("--gate-dir", default="bpa_v4_gate_results")
    parser.add_argument("--seq-lens", default="512,1024")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--top-b", type=int, default=8)
    parser.add_argument("--far-budget", type=int, default=4)
    parser.add_argument("--target-rate", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=4.0)
    parser.add_argument("--output-dir", default="bpa_v5_results")
    parser.add_argument(
        "--text-data",
        default="data/finewebedu/val.bin",
    )

    args = parser.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    run_v5_grid(
        checkpoint=args.checkpoint,
        seq_lens=seq_lens,
        seeds=seeds,
        n_eval=args.n_eval,
        batch_size=args.batch_size,
        local_window=args.local_window,
        chunk_size=args.chunk_size,
        top_b=args.top_b,
        far_budget=args.far_budget,
        v4_gate_dir=args.gate_dir,
        text_data_path=args.text_data,
        output_dir=args.output_dir,
        target_enabled_rate=args.target_rate,
        tau=args.tau,
    )


if __name__ == "__main__":
    main()
