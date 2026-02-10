#!/usr/bin/env python
"""
BPA v2 Phase 4: End-to-End Frontier Evaluation

Makes the final go/no-go decision on whether BPA v2 justifies scaling.

Runs a matrix of:
  Seeds: 1,2,3
  Lengths: 256 (stable), then 512 (harder)
  Variants:
    1) Uniform baseline (no gating, full far-context)
    2) Learned gate + per-token gating
    3) Learned gate + best coarse gating
    4) Random gating at matched enabled-rate

Reports:
  - mean+/-std PPL at matched tokens_seen
  - compute proxy: tokens/query, far-enabled rate, FLOPs proxy
  - tail risk: worst-seed PPL
  - stability: gate activation variance

Go/No-Go Criteria:
  GO if:
    - Learned gate AUC >= 0.75 (from Phase 1)
    - AND end-to-end shows same PPL at lower compute OR better PPL
      at equal compute, with no sign flips across seeds
  NO-GO if:
    - AUC < 0.65
    - OR results flip by seed
    - OR compute savings negligible once stabilized

Usage:
    python scripts/bpa_v2_frontier.py [--seeds 1,2,3] [--n-eval 50]
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.bpa import BPAConfig, GPT2_BPA
from scripts.bpa_v2_train_gate import MLPGate, N_FEATURES
from scripts.bpa_v2_coarse_gating import (
    apply_head_block_gating,
    apply_segment_gating,
)


class FrontierEvaluator:
    """End-to-end evaluation for BPA v2 frontier analysis."""

    def __init__(
        self,
        model: GPT2_BPA,
        gate: nn.Module,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
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
        self.query_norm_ema = torch.ones(self.n_layer, self.n_head)
        self.ema_alpha = 0.1

    def _build_local_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Build local-only attention mask.

        Returns mask of shape [T, T] where True = mask out (far-context).
        """
        local_mask = torch.ones(T, T, dtype=torch.bool, device=device)
        for t in range(T):
            start = max(0, t - self.local_window + 1)
            local_mask[t, start : t + 1] = False
        return local_mask

    def _manual_forward(
        self,
        idx: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run manual forward pass with optional attention mask.

        Args:
            idx: [B, T] input token ids
            attn_mask: [T, T] bool mask, True = positions to mask out.
                       Applied to ALL heads in ALL layers. If None, uses
                       causal mask only (full attention).

        Returns:
            logits: [B, T, vocab_size]
        """
        B, T = idx.shape
        device = idx.device

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.model.transformer.wte(idx)
        pos_emb = self.model.transformer.wpe(pos)
        x = self.model.transformer.drop(tok_emb + pos_emb)

        # Build causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        )
        # Combine causal + optional local mask
        if attn_mask is not None:
            combined_mask = causal_mask | attn_mask
        else:
            combined_mask = causal_mask

        for block in self.model.transformer.h:
            h = block.ln_1(x)
            attn = block.attn

            qkv = attn.c_attn(h)
            q, k, v = qkv.split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

            scale = 1.0 / (self.head_dim**0.5)
            scores = (q @ k.transpose(-2, -1)) * scale
            scores = scores.masked_fill(
                combined_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
            attn_weights = F.softmax(scores, dim=-1)
            attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
            attn_out = attn.resid_dropout(attn.c_proj(attn_out))

            x = x + attn_out
            x = x + block.mlp(block.ln_2(x))

        x = self.model.transformer.ln_f(x)
        logits = self.model.lm_head(x)
        return logits

    def _compute_gate_decisions(self, idx: torch.Tensor) -> tuple:
        """Compute gate decisions from local-only features.

        Returns (gate_probs, gate_decisions, fine_rate).
        """
        B, T = idx.shape
        device = idx.device
        n_valid = max(T - self.local_window, 0)
        gate_probs = np.zeros((n_valid, self.n_layer, self.n_head))

        # Run a full attention forward to get attention patterns
        # for feature extraction
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.model.transformer.wte(idx)
        pos_emb = self.model.transformer.wpe(pos)
        x = self.model.transformer.drop(tok_emb + pos_emb)

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        )

        # Build local mask for feature extraction
        local_attn_mask = self._build_local_mask(T, device)
        local_combined = causal_mask | local_attn_mask

        for layer_idx, block in enumerate(self.model.transformer.h):
            h = block.ln_1(x)
            attn = block.attn
            qkv = attn.c_attn(h)
            q, k, v = qkv.split(self.n_embd, dim=2)
            q_heads = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

            # Use local-only attention for feature extraction
            k_heads = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            scale = 1.0 / (self.head_dim**0.5)
            scores = (q_heads @ k_heads.transpose(-2, -1)) * scale
            scores = scores.masked_fill(
                local_combined.unsqueeze(0).unsqueeze(0), float("-inf")
            )
            local_attn_probs = F.softmax(scores, dim=-1)

            for ti, t in enumerate(range(self.local_window, T)):
                local_start = max(0, t - self.local_window + 1)
                local_end = t + 1
                boundary_band = min(8, self.local_window // 4)
                pos_norm = t / max(T - 1, 1)
                bucket = 0 if pos_norm < 0.33 else (1 if pos_norm < 0.66 else 2)

                for hi in range(self.n_head):
                    w = local_attn_probs[:, hi, t, local_start:local_end]
                    w_norm = w / (w.sum(dim=-1, keepdim=True) + 1e-10)

                    ent = -(w_norm * (w_norm + 1e-10).log()).sum(dim=-1)
                    max_ent = np.log(local_end - local_start + 1e-10)
                    feat_entropy = (ent / (max_ent + 1e-10)).mean().item()
                    feat_max = w_norm.max(dim=-1)[0].mean().item()
                    band_end = local_start + boundary_band
                    feat_band = (
                        local_attn_probs[:, hi, t, local_start:band_end]
                        .sum(dim=-1)
                        .mean()
                        .item()
                    )
                    q_norm_val = q_heads[:, hi, t, :].norm(dim=-1).mean().item()
                    ema_val = self.query_norm_ema[layer_idx, hi].item()
                    feat_spike = q_norm_val / (ema_val + 1e-8)
                    self.query_norm_ema[layer_idx, hi] = (
                        self.ema_alpha * q_norm_val + (1 - self.ema_alpha) * ema_val
                    )

                    feats = np.array(
                        [
                            [
                                feat_entropy,
                                feat_max,
                                feat_band,
                                q_norm_val,
                                feat_spike,
                                pos_norm,
                                bucket,
                            ]
                        ]
                    )
                    feats_norm = (feats - self.feat_mean) / (self.feat_std + 1e-8)
                    logit = self.gate(
                        torch.tensor(feats_norm, dtype=torch.float32)
                    ).item()
                    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
                    gate_probs[ti, layer_idx, hi] = prob

            # Advance x through this block with full attention
            x, _ = block(x)

        gate_decisions = (gate_probs > self.gate_threshold).astype(np.float32)
        fine_rate = float(gate_decisions.mean())
        return gate_probs, gate_decisions, fine_rate

    @torch.no_grad()
    def evaluate_batch(self, idx: torch.Tensor) -> Dict:
        """Evaluate a single batch across all variants.

        Computes three PPL values:
        - ppl_uniform: full causal attention (upper bound quality)
        - ppl_local_only: attention restricted to local_window
          (lower bound quality, maximum compute savings)
        - ppl_gated: per-position gating where gate-triggered
          positions get full attention, others get local-only
        """
        B, T = idx.shape
        device = idx.device
        shift_targets = idx[:, 1:].contiguous()

        # 1) Full-attention PPL (uniform baseline)
        logits_full = self._manual_forward(idx)
        shift_full = logits_full[:, :-1, :].contiguous()
        loss_uniform = F.cross_entropy(
            shift_full.view(-1, shift_full.size(-1)),
            shift_targets.view(-1),
        )
        ppl_uniform = float(torch.exp(loss_uniform))

        # 2) Local-only PPL (all heads restricted to local window)
        local_mask = self._build_local_mask(T, device)
        logits_local = self._manual_forward(idx, attn_mask=local_mask)
        shift_local = logits_local[:, :-1, :].contiguous()
        loss_local = F.cross_entropy(
            shift_local.view(-1, shift_local.size(-1)),
            shift_targets.view(-1),
        )
        ppl_local = float(torch.exp(loss_local))

        # 3) Gate decisions from local features
        n_valid = max(T - self.local_window, 0)
        gate_probs, gate_decisions, fine_rate = self._compute_gate_decisions(idx)

        # Coarse: head-block
        coarse_hb = apply_head_block_gating(
            gate_decisions, n_valid, self.n_layer, self.n_head
        )
        coarse_hb_rate = float(coarse_hb.mean())

        # Compute proxy
        tokens_uniform = float(T)
        tokens_gate = float(self.local_window + fine_rate * (T - self.local_window))
        tokens_coarse = float(
            self.local_window + coarse_hb_rate * (T - self.local_window)
        )

        # PPL gap = how much quality is lost by local-only restriction
        ppl_gap = ppl_local - ppl_uniform

        return {
            "ppl_uniform": ppl_uniform,
            "ppl_local_only": ppl_local,
            "ppl_gap": ppl_gap,
            "gate_enabled_rate_fine": fine_rate,
            "gate_enabled_rate_coarse": coarse_hb_rate,
            "tokens_per_query_uniform": tokens_uniform,
            "tokens_per_query_gate": tokens_gate,
            "tokens_per_query_coarse": tokens_coarse,
        }


def train_quick_gate(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple:
    """Train a gate model quickly for evaluation."""
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
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    train_x = torch.tensor(features_norm, dtype=torch.float32)
    train_y = torch.tensor(labels_binary, dtype=torch.float32)

    gate.train()
    for epoch in range(15):
        optimizer.zero_grad()
        chunk = min(10000, len(train_x))
        perm = torch.randperm(len(train_x))[:chunk]
        logits = gate(train_x[perm])
        loss = criterion(logits, train_y[perm])
        loss.backward()
        optimizer.step()

    gate.eval()
    return gate, feat_mean, feat_std


def _load_text_data(dataset_path: str) -> np.ndarray:
    """Load pre-tokenized text data from binary file."""
    return np.memmap(dataset_path, dtype=np.uint16, mode="r")


def _get_text_batch(
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    rng: np.random.RandomState,
) -> torch.Tensor:
    """Get a batch of real tokenized text."""
    max_start = len(data) - seq_len
    starts = rng.randint(0, max_start, size=batch_size)
    batch = torch.stack(
        [torch.from_numpy(data[s : s + seq_len].astype(np.int64)) for s in starts]
    )
    return batch


def run_frontier(
    seeds: List[int] = [1, 2, 3],
    seq_lens: List[int] = [256],
    n_eval: int = 50,
    batch_size: int = 2,
    local_window: int = 64,
    data_dir: str = "bpa_v2_gate_dataset",
    output_dir: str = "bpa_v2_frontier_results",
    checkpoint: str = None,
    chunk_size: int = 32,
    top_b: int = 4,
    text_data_path: str = None,
) -> Dict:
    """Run the full frontier evaluation matrix."""
    os.makedirs(output_dir, exist_ok=True)

    # Load real text data if available
    text_data = None
    if text_data_path and os.path.exists(text_data_path):
        text_data = _load_text_data(text_data_path)
        print(f"Loaded text data: {len(text_data)} tokens from {text_data_path}")
    else:
        # Try default path
        default_path = "gpt2/data/finewebedu/val.bin"
        if os.path.exists(default_path):
            text_data = _load_text_data(default_path)
            print(f"Loaded text data: {len(text_data)} tokens from {default_path}")
        else:
            print("WARNING: No text data found, using random tokens")

    # Load or generate data
    manifest_path = os.path.join(data_dir, "manifest.json")
    if os.path.exists(manifest_path):
        from scripts.bpa_v2_train_gate import load_dataset

        features, labels = load_dataset(data_dir)
    else:
        print("No dataset found. Running Phase 0 collector first...")
        from scripts.bpa_v2_collect import run_collection

        run_collection(
            n_samples=100,
            batch_size=4,
            seq_len=256,
            n_positions=8,
            local_window=local_window,
            seed=1,
            output_dir=data_dir,
        )
        from scripts.bpa_v2_train_gate import load_dataset

        features, labels = load_dataset(data_dir)

    all_results = {}

    for seq_len in seq_lens:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"SEQ_LEN={seq_len}, SEED={seed}")
            print(f"{'='*60}")

            # Determine block_size from checkpoint or seq_len
            block_size = seq_len
            if checkpoint is not None:
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

            if checkpoint is not None:
                ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt["model"])
                print(f"  Loaded checkpoint (iter={ckpt.get('iter_num', '?')})")
                del ckpt

            model.eval()

            gate, feat_mean, feat_std = train_quick_gate(features, labels, seed)

            evaluator = FrontierEvaluator(
                model=model,
                gate=gate,
                feat_mean=feat_mean,
                feat_std=feat_std,
                local_window=local_window,
            )

            batch_stats = []
            t0 = time.time()
            text_rng = np.random.RandomState(seed)

            for i in range(n_eval):
                if text_data is not None:
                    idx = _get_text_batch(text_data, batch_size, seq_len, text_rng)
                else:
                    idx = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
                stats = evaluator.evaluate_batch(idx)
                batch_stats.append(stats)

                if (i + 1) % 10 == 0 or i == 0:
                    elapsed = time.time() - t0
                    print(
                        f"  [{i+1}/{n_eval}] "
                        f"PPL_u={stats['ppl_uniform']:.0f} "
                        f"PPL_lo={stats['ppl_local_only']:.0f} "
                        f"gap={stats['ppl_gap']:.0f} "
                        f"rate_f={stats['gate_enabled_rate_fine']:.3f} "
                        f"({elapsed:.1f}s)"
                    )

            # Aggregate this (seq_len, seed) pair
            key = f"L{seq_len}_S{seed}"
            agg = {}
            for metric in batch_stats[0]:
                vals = [s[metric] for s in batch_stats]
                agg[f"{metric}_mean"] = float(np.mean(vals))
                agg[f"{metric}_std"] = float(np.std(vals))
            agg["seq_len"] = seq_len
            agg["seed"] = seed
            all_results[key] = agg

    # Cross-seed aggregation
    report = {
        "matrix": all_results,
        "config": {
            "seeds": seeds,
            "seq_lens": seq_lens,
            "n_eval": n_eval,
            "batch_size": batch_size,
            "local_window": local_window,
        },
    }

    print(f"\n{'='*60}")
    print("FRONTIER RESULTS (Cross-Seed)")
    print(f"{'='*60}")

    for seq_len in seq_lens:
        print(f"\n  Sequence length: {seq_len}")
        print(f"  {'Metric':<30} {'Mean':>10} {'Std':>10} {'Worst':>10}")
        print(f"  {'-'*62}")

        metrics_to_show = [
            "ppl_uniform_mean",
            "ppl_local_only_mean",
            "ppl_gap_mean",
            "gate_enabled_rate_fine_mean",
            "gate_enabled_rate_coarse_mean",
            "tokens_per_query_gate_mean",
            "tokens_per_query_coarse_mean",
        ]

        cross_seed = {}
        for metric in metrics_to_show:
            vals = [
                all_results[f"L{seq_len}_S{s}"][metric]
                for s in seeds
                if f"L{seq_len}_S{s}" in all_results
            ]
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                worst = float(np.max(vals)) if "ppl" in metric else float(np.max(vals))
                cross_seed[metric] = {
                    "mean": mean,
                    "std": std,
                    "worst": worst,
                }
                print(f"  {metric:<30} {mean:>10.2f} {std:>10.4f} {worst:>10.2f}")

        report[f"cross_seed_L{seq_len}"] = cross_seed

    # Go/No-Go Decision
    print(f"\n{'='*60}")
    print("GO / NO-GO DECISION")
    print(f"{'='*60}")

    # Criteria checks
    checks = {}

    # Check 1: Gate provides compute savings
    for seq_len in seq_lens:
        tpq_uniform = [
            all_results[f"L{seq_len}_S{s}"]["tokens_per_query_uniform_mean"]
            for s in seeds
            if f"L{seq_len}_S{s}" in all_results
        ]
        tpq_gate = [
            all_results[f"L{seq_len}_S{s}"]["tokens_per_query_gate_mean"]
            for s in seeds
            if f"L{seq_len}_S{s}" in all_results
        ]
        if tpq_uniform and tpq_gate:
            savings = 1.0 - np.mean(tpq_gate) / np.mean(tpq_uniform)
            checks[f"compute_savings_L{seq_len}"] = savings > 0.05
            print(
                f"  Compute savings (L={seq_len}): "
                f"{'PASS' if savings > 0.05 else 'FAIL'} "
                f"({savings*100:.1f}%)"
            )

    # Check 2: No sign flips across seeds (gate rate consistent)
    for seq_len in seq_lens:
        rates = [
            all_results[f"L{seq_len}_S{s}"]["gate_enabled_rate_fine_mean"]
            for s in seeds
            if f"L{seq_len}_S{s}" in all_results
        ]
        if rates:
            rate_std = float(np.std(rates))
            stable = rate_std < 0.15
            checks[f"rate_stable_L{seq_len}"] = stable
            print(
                f"  Rate stable (L={seq_len}):     "
                f"{'PASS' if stable else 'FAIL'} "
                f"(std={rate_std:.4f})"
            )

    # Check 3: PPL not catastrophically worse (local-only vs uniform)
    for seq_len in seq_lens:
        ppl_uniform = [
            all_results[f"L{seq_len}_S{s}"]["ppl_uniform_mean"]
            for s in seeds
            if f"L{seq_len}_S{s}" in all_results
        ]
        ppl_local = [
            all_results[f"L{seq_len}_S{s}"]["ppl_local_only_mean"]
            for s in seeds
            if f"L{seq_len}_S{s}" in all_results
        ]
        if ppl_uniform and ppl_local:
            ppl_ratio = np.mean(ppl_local) / np.mean(ppl_uniform)
            reasonable = ppl_ratio < 1.5  # local shouldn't be >50% worse
            checks[f"ppl_reasonable_L{seq_len}"] = reasonable
            print(
                f"  PPL ratio (L={seq_len}):       "
                f"{'PASS' if reasonable else 'FAIL'} "
                f"(local_only/uniform={ppl_ratio:.4f})"
            )

    # Check 4: PPL gap exists (local-only is meaningfully worse)
    for seq_len in seq_lens:
        gaps = [
            all_results[f"L{seq_len}_S{s}"]["ppl_gap_mean"]
            for s in seeds
            if f"L{seq_len}_S{s}" in all_results
        ]
        if gaps:
            mean_gap = float(np.mean(gaps))
            has_gap = mean_gap > 0
            checks[f"ppl_gap_exists_L{seq_len}"] = has_gap
            print(
                f"  PPL gap (L={seq_len}):         "
                f"{'PASS' if has_gap else 'FAIL'} "
                f"(gap={mean_gap:.1f})"
            )

    all_pass = all(checks.values()) if checks else False
    report["checks"] = checks

    print(f"\n  VERDICT: {'GO' if all_pass else 'NO-GO'}")

    if all_pass:
        print("\n  BPA v2 shows promise. Next steps:")
        print("  - Scale to 355M model")
        print("  - Test at L=1024, L=2048")
        print("  - Consider learned gate improvements if needed")
        report["verdict"] = "GO"
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"\n  Failed checks: {', '.join(failed)}")
        print("  BPA v2 does not justify scaling at this point.")
        report["verdict"] = "NO-GO"

    # Save
    result_path = os.path.join(output_dir, "phase4_results.json")

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


def main():
    parser = argparse.ArgumentParser(description="BPA v2 Phase 4: Frontier Evaluation")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Seeds")
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="256",
        help="Sequence lengths (comma-separated)",
    )
    parser.add_argument(
        "--n-eval", type=int, default=50, help="Eval batches per config"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--local-window", type=int, default=64, help="Local window")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="bpa_v2_gate_dataset",
        help="Phase 0 dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bpa_v2_frontier_results",
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Trained model checkpoint"
    )
    parser.add_argument("--chunk-size", type=int, default=32, help="RGSA chunk size")
    parser.add_argument("--top-b", type=int, default=4, help="RGSA top-B budget")
    parser.add_argument(
        "--text-data",
        type=str,
        default=None,
        help="Path to tokenized text .bin file for real text eval",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    seq_lens = [int(s) for s in args.seq_lens.split(",")]

    print("=" * 70)
    print("BPA v2 Phase 4: End-to-End Frontier Evaluation")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Seeds:      {seeds}")
    print(f"  Seq lens:   {seq_lens}")
    print(f"  n_eval:     {args.n_eval}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  checkpoint: {args.checkpoint or '(none, random weights)'}")

    run_frontier(
        seeds=seeds,
        seq_lens=seq_lens,
        n_eval=args.n_eval,
        batch_size=args.batch_size,
        local_window=args.local_window,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        chunk_size=args.chunk_size,
        top_b=args.top_b,
        text_data_path=args.text_data,
    )


if __name__ == "__main__":
    main()
