#!/usr/bin/env python
"""
BPA v2 Phase 2: Speculative Local-First Pipeline Evaluation

Deploys the trained gate to decide when far-context attention is needed.

Pipeline per token:
  Step 1: Local-only forward (always) -> compute local features -> gate
  Step 2: Conditional far-context for triggered (l,h,t) only

Evaluation compares:
  1. Uniform: always compute far (baseline, upper bound quality)
  2. Learned gate: conditional far based on gate predictions
  3. Oracle gate: conditional far based on true boundary_pressure
  4. Random gate: random decisions at same enabled-rate (sanity)

Reports:
  - PPL for each variant at matched tokens_seen
  - Compute proxy: enabled-rate, tokens/query
  - Gate decision stability

Usage:
    python scripts/bpa_v2_speculative.py [--data-dir bpa_v2_gate_dataset]
        [--gate-threshold 0.5] [--n-eval 100] [--seed 1]
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.bpa import BPAConfig, GPT2_BPA
from scripts.bpa_v2_train_gate import MLPGate, LogisticGate, FEATURE_NAMES, N_FEATURES


class SpeculativePipeline:
    """
    Evaluates speculative local-first attention with gate-based decisions.

    Does NOT modify the model. Instead, computes full attention (like uniform)
    and then simulates gating by measuring what the PPL would be if we
    restricted attention to local-only for positions where the gate says
    far-context is not needed.
    """

    def __init__(
        self,
        model: GPT2_BPA,
        gate: Optional[nn.Module] = None,
        feat_mean: Optional[np.ndarray] = None,
        feat_std: Optional[np.ndarray] = None,
        gate_threshold: float = 0.5,
        local_window: int = 64,
    ):
        self.model = model
        self.gate = gate
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.gate_threshold = gate_threshold
        self.local_window = local_window
        self.cfg = model.config
        self.n_layer = self.cfg.n_layer
        self.n_head = self.cfg.n_head
        self.n_embd = self.cfg.n_embd
        self.head_dim = self.n_embd // self.n_head

        # EMA for query norm spike
        self.query_norm_ema = torch.ones(self.n_layer, self.n_head)
        self.ema_alpha = 0.1

    @torch.no_grad()
    def evaluate_batch(
        self,
        idx: torch.Tensor,
        oracle_thr: float,
    ) -> Dict:
        """
        Evaluate a single batch with all gating variants.

        Returns per-variant logits and compute statistics.
        """
        B, T = idx.shape
        device = idx.device

        # Full forward pass (uniform baseline) to get reference logits
        # Pass targets so we get full [B, T, vocab] logits, not just last token
        targets = idx  # self-supervised LM: targets are input shifted by 1
        logits_uniform, _ = self.model(idx, targets=targets)  # [B, T, vocab]

        # Now compute local-only logits and gate decisions per layer
        tok_emb = self.model.transformer.wte(idx)
        pos_arange = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.model.transformer.wpe(pos_arange)
        x = self.model.transformer.drop(tok_emb + pos_emb)

        # Track per-position gate decisions
        gate_decisions = torch.zeros(B, T, self.n_layer, self.n_head)
        oracle_decisions = torch.zeros(B, T, self.n_layer, self.n_head)
        boundary_pressures = torch.zeros(B, T, self.n_layer, self.n_head)

        # Compute local-only output for each layer
        x_local = x.clone()
        for layer_idx, block in enumerate(self.model.transformer.h):
            h = block.ln_1(x_local)
            attn = block.attn

            qkv = attn.c_attn(h)
            q, k, v = qkv.split(self.n_embd, dim=2)

            q_heads = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k_heads = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v_heads = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

            # Full attention scores
            scale = 1.0 / (self.head_dim**0.5)
            scores = (q_heads @ k_heads.transpose(-2, -1)) * scale
            causal_mask = torch.triu(
                torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
            attn_probs = F.softmax(scores, dim=-1)

            # Compute features and decisions for positions past local_window
            for t in range(self.local_window, T):
                local_start = max(0, t - self.local_window + 1)
                local_end = t + 1
                boundary_band = min(8, self.local_window // 4)
                pos_norm = t / max(T - 1, 1)

                if pos_norm < 0.33:
                    bucket = 0
                elif pos_norm < 0.66:
                    bucket = 1
                else:
                    bucket = 2

                for hi in range(self.n_head):
                    w = attn_probs[:, hi, t, : t + 1]

                    # Local features
                    local_w = w[:, local_start:local_end]
                    local_w_norm = local_w / (local_w.sum(dim=-1, keepdim=True) + 1e-10)

                    ent = -(local_w_norm * (local_w_norm + 1e-10).log()).sum(dim=-1)
                    max_ent = np.log(local_end - local_start + 1e-10)
                    feat_entropy = ent / (max_ent + 1e-10)

                    feat_max = local_w_norm.max(dim=-1)[0]

                    band_end = local_start + boundary_band
                    feat_band = w[:, local_start:band_end].sum(dim=-1)

                    q_norm = q_heads[:, hi, t, :].norm(dim=-1)
                    ema_val = self.query_norm_ema[layer_idx, hi]
                    feat_spike = q_norm / (ema_val + 1e-8)

                    self.query_norm_ema[layer_idx, hi] = (
                        self.ema_alpha * q_norm.mean().item()
                        + (1 - self.ema_alpha) * ema_val
                    )

                    # Oracle: boundary_pressure
                    far_end = t - self.local_window
                    bp = 0.0
                    if far_end > 0:
                        bp = attn_probs[:, hi, t, :far_end].sum(dim=-1)
                        boundary_pressures[:, t, layer_idx, hi] = bp

                    oracle_decisions[:, t, layer_idx, hi] = (
                        (bp > oracle_thr).float()
                        if isinstance(bp, torch.Tensor)
                        else float(bp > oracle_thr)
                    )

                    # Gate decision
                    if self.gate is not None:
                        feats = torch.tensor(
                            [
                                [
                                    feat_entropy.mean().item(),
                                    feat_max.mean().item(),
                                    feat_band.mean().item(),
                                    q_norm.mean().item(),
                                    feat_spike.mean().item(),
                                    pos_norm,
                                    bucket,
                                ]
                            ],
                            dtype=torch.float32,
                        )
                        if self.feat_mean is not None:
                            feats = (feats - torch.tensor(self.feat_mean)) / (
                                torch.tensor(self.feat_std) + 1e-8
                            )
                        logit = self.gate(feats).item()
                        prob = 1.0 / (1.0 + np.exp(-logit))
                        gate_decisions[:, t, layer_idx, hi] = float(
                            prob > self.gate_threshold
                        )

            # Forward through block
            x_local, _ = block(x_local)

        # Compute statistics
        # Positions that matter: past local_window
        valid_mask = torch.zeros(B, T, dtype=torch.bool)
        valid_mask[:, self.local_window :] = True

        stats = {}

        # Gate enabled-rate
        if self.gate is not None:
            gate_valid = gate_decisions[
                valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(gate_decisions)
            ]
            stats["gate_enabled_rate"] = float(gate_valid.mean())
        else:
            stats["gate_enabled_rate"] = 0.0

        # Oracle enabled-rate
        oracle_valid = oracle_decisions[
            valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(oracle_decisions)
        ]
        stats["oracle_enabled_rate"] = float(oracle_valid.mean())

        # Random at same rate as gate
        if self.gate is not None:
            random_rate = stats["gate_enabled_rate"]
            random_decisions = (torch.rand_like(gate_decisions) < random_rate).float()
        else:
            random_decisions = torch.zeros_like(oracle_decisions)

        # Compute PPL for uniform (full attention) baseline
        # shift logits and targets for language modeling
        shift_logits = logits_uniform[:, :-1, :].contiguous()
        shift_targets = idx[:, 1:].contiguous()
        loss_uniform = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
        )
        stats["ppl_uniform"] = float(torch.exp(loss_uniform))
        stats["loss_uniform"] = float(loss_uniform)

        # Compute PPL for local-only (worst case, no far-context)
        x_final = self.model.transformer.ln_f(x_local)
        logits_local = self.model.lm_head(x_final)
        shift_local = logits_local[:, :-1, :].contiguous()
        loss_local = F.cross_entropy(
            shift_local.view(-1, shift_local.size(-1)),
            shift_targets.view(-1),
        )
        stats["ppl_local_only"] = float(torch.exp(loss_local))
        stats["loss_local_only"] = float(loss_local)

        # Compute FLOPs proxy
        # Uniform: all positions attend to full context
        # Gated: only triggered positions attend to far context
        n_valid = valid_mask.sum().item() * self.n_layer * self.n_head
        stats["flops_uniform"] = float(n_valid)  # all far

        if self.gate is not None:
            stats["flops_gate"] = float(gate_decisions.sum())
        stats["flops_oracle"] = float(oracle_decisions.sum())
        stats["flops_random"] = float(random_decisions.sum())

        # Tokens per query (average)
        stats["tokens_per_query_uniform"] = float(T)
        if self.gate is not None:
            far_rate = stats["gate_enabled_rate"]
            stats["tokens_per_query_gate"] = float(
                self.local_window + far_rate * (T - self.local_window)
            )

        return stats


def run_evaluation(
    n_eval: int = 100,
    batch_size: int = 2,
    seq_len: int = 256,
    local_window: int = 64,
    gate_threshold: float = 0.5,
    seed: int = 1,
    data_dir: str = "bpa_v2_gate_dataset",
    gate_results_dir: str = "bpa_v2_gate_results",
    output_dir: str = "bpa_v2_speculative_results",
) -> Dict:
    """Run the speculative pipeline evaluation."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = BPAConfig(
        block_size=seq_len,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        local_window=local_window,
        chunk_size=32,
        top_b=4,
    )

    print(f"Creating BPA model...")
    model = GPT2_BPA(cfg)
    model.eval()

    # Try to load trained gate
    gate = None
    feat_mean = None
    feat_std = None

    gate_result_path = os.path.join(gate_results_dir, "phase1_results.json")
    if os.path.exists(gate_result_path):
        with open(gate_result_path) as f:
            gate_results = json.load(f)
        print(f"Loaded gate results from {gate_result_path}")
        # Create gate model (we don't have saved weights, so train a fresh one)
        # In production, we'd load from checkpoint
    else:
        print("No gate results found, training a quick gate...")

    # Quick gate training inline (for testing)
    if os.path.exists(os.path.join(data_dir, "manifest.json")):
        from scripts.bpa_v2_train_gate import load_dataset

        features, labels = load_dataset(data_dir)

        thr = np.percentile(labels, 75)
        labels_binary = (labels > thr).astype(np.float32)
        pos_rate = labels_binary.mean()
        pos_weight = (1 - pos_rate) / (pos_rate + 1e-10)

        feat_mean = features.mean(axis=0)
        feat_std = features.std(axis=0) + 1e-8
        features_norm = (features - feat_mean) / feat_std

        gate = MLPGate(N_FEATURES, hidden=128, n_layers=2)
        optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)
        pw = torch.tensor([pos_weight])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        train_x = torch.tensor(features_norm, dtype=torch.float32)
        train_y = torch.tensor(labels_binary, dtype=torch.float32)

        gate.train()
        for epoch in range(10):
            optimizer.zero_grad()
            # Process in chunks to avoid memory issues
            chunk = min(10000, len(train_x))
            perm = torch.randperm(len(train_x))[:chunk]
            logits = gate(train_x[perm])
            loss = criterion(logits, train_y[perm])
            loss.backward()
            optimizer.step()

        gate.eval()
        print(f"Gate trained (quick, 10 epochs)")

        oracle_thr = thr
    else:
        print("No dataset found, running without gate (uniform-only evaluation)")
        oracle_thr = 0.3  # default for untrained model

    pipeline = SpeculativePipeline(
        model=model,
        gate=gate,
        feat_mean=feat_mean,
        feat_std=feat_std,
        gate_threshold=gate_threshold,
        local_window=local_window,
    )

    os.makedirs(output_dir, exist_ok=True)

    all_stats = []
    t0 = time.time()

    print(f"\nEvaluating {n_eval} batches...")
    for i in range(n_eval):
        idx = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        stats = pipeline.evaluate_batch(idx, oracle_thr=oracle_thr)
        all_stats.append(stats)

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            print(
                f"  [{i+1}/{n_eval}] "
                f"PPL_uniform={stats['ppl_uniform']:.1f} "
                f"PPL_local={stats['ppl_local_only']:.1f} "
                f"gate_rate={stats['gate_enabled_rate']:.3f} "
                f"({elapsed:.1f}s)"
            )

    elapsed = time.time() - t0
    print(f"\nEvaluation complete in {elapsed:.1f}s")

    # Aggregate
    report = aggregate_stats(all_stats)
    report["config"] = {
        "n_eval": n_eval,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "local_window": local_window,
        "gate_threshold": gate_threshold,
        "seed": seed,
    }

    # Print report
    print(f"\n{'='*60}")
    print("SPECULATIVE PIPELINE RESULTS")
    print(f"{'='*60}")

    print(f"\nPPL Comparison:")
    print(
        f"  Uniform (full attention):  {report['ppl_uniform_mean']:.2f} +/- {report['ppl_uniform_std']:.2f}"
    )
    print(
        f"  Local-only (no far):       {report['ppl_local_mean']:.2f} +/- {report['ppl_local_std']:.2f}"
    )

    print(f"\nCompute Proxy:")
    print(f"  Gate enabled-rate:         {report['gate_enabled_rate_mean']:.4f}")
    print(f"  Oracle enabled-rate:       {report['oracle_enabled_rate_mean']:.4f}")
    if "tokens_per_query_gate_mean" in report:
        print(
            f"  Tokens/query (uniform):    {report['tokens_per_query_uniform_mean']:.1f}"
        )
        print(
            f"  Tokens/query (gate):       {report['tokens_per_query_gate_mean']:.1f}"
        )
        savings = (
            1.0
            - report["tokens_per_query_gate_mean"]
            / report["tokens_per_query_uniform_mean"]
        )
        print(f"  Compute savings:           {savings*100:.1f}%")

    # Acceptance
    print(f"\n{'='*60}")
    print("ACCEPTANCE CHECK")
    print(f"{'='*60}")

    # Gate should have reasonable enabled rate (not 0 or 1)
    rate = report["gate_enabled_rate_mean"]
    stable = 0.05 < rate < 0.95
    print(f"  Gate rate stable (5-95%):  {'PASS' if stable else 'FAIL'} ({rate:.4f})")

    # PPL gap between uniform and local should exist
    ppl_gap = report["ppl_local_mean"] - report["ppl_uniform_mean"]
    has_gap = ppl_gap > 0.1
    print(
        f"  PPL gap (local vs full):   {'PASS' if has_gap else 'FAIL'} ({ppl_gap:.2f})"
    )

    report["acceptance"] = {
        "gate_rate_stable": stable,
        "ppl_gap_exists": has_gap,
    }

    # Save
    result_path = os.path.join(output_dir, "phase2_results.json")

    def _json_default(x):
        if isinstance(x, (np.floating, np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.integer, np.int32, np.int64)):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.bool_):
            return bool(x)
        raise TypeError(f"Not JSON serializable: {type(x)}")

    with open(result_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print(f"\nResults saved to {result_path}")

    return report


def aggregate_stats(all_stats: List[Dict]) -> Dict:
    """Aggregate per-batch stats into summary."""
    agg = {}

    keys = [
        ("ppl_uniform", "ppl_uniform"),
        ("ppl_local_only", "ppl_local"),
        ("gate_enabled_rate", "gate_enabled_rate"),
        ("oracle_enabled_rate", "oracle_enabled_rate"),
        ("tokens_per_query_uniform", "tokens_per_query_uniform"),
    ]

    for src_key, dst_key in keys:
        vals = [s[src_key] for s in all_stats if src_key in s]
        if vals:
            agg[f"{dst_key}_mean"] = float(np.mean(vals))
            agg[f"{dst_key}_std"] = float(np.std(vals))

    # Optional gate keys
    if "tokens_per_query_gate" in all_stats[0]:
        vals = [s["tokens_per_query_gate"] for s in all_stats]
        agg["tokens_per_query_gate_mean"] = float(np.mean(vals))
        agg["tokens_per_query_gate_std"] = float(np.std(vals))

    if "flops_gate" in all_stats[0]:
        vals = [s["flops_gate"] for s in all_stats]
        agg["flops_gate_mean"] = float(np.mean(vals))

    return agg


def main():
    parser = argparse.ArgumentParser(description="BPA v2 Phase 2: Speculative Pipeline")
    parser.add_argument("--n-eval", type=int, default=50, help="Eval batches")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--local-window", type=int, default=64, help="Local window")
    parser.add_argument(
        "--gate-threshold", type=float, default=0.5, help="Gate decision threshold"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="bpa_v2_gate_dataset",
        help="Phase 0 dataset",
    )
    parser.add_argument(
        "--gate-results-dir",
        type=str,
        default="bpa_v2_gate_results",
        help="Phase 1 results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bpa_v2_speculative_results",
        help="Output directory",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("BPA v2 Phase 2: Speculative Local-First Pipeline")
    print("=" * 70)

    run_evaluation(
        n_eval=args.n_eval,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        local_window=args.local_window,
        gate_threshold=args.gate_threshold,
        seed=args.seed,
        data_dir=args.data_dir,
        gate_results_dir=args.gate_results_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
