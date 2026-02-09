#!/usr/bin/env python
"""
BPA v1 Phase 3: Cheap Signal Predictors

Test whether boundary_pressure can be predicted WITHOUT computing
far-context attention scores. If yes, BPA can deliver real compute savings.

Candidate cheap predictors:
C1) Local-boundary score gap: concentration at boundary vs elsewhere
C2) Local-only entropy: output entropy using local-only attention
C3) Attempted escape proxy: attention concentration near boundary
C4) Residual norm spike: per-token norm changes

Evaluation:
- Use oracle boundary_pressure as label
- Compute AUC for predicting boundary_pressure > threshold

Acceptance criteria (R3):
- A cheap predictor achieves AUC >= 0.75 for predicting high boundary_pressure
- Overhead is negligible (no far-KV computation)
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.bpa import BPAConfig, GPT2_BPA


class CheapSignalComputer:
    """Compute cheap predictors for boundary_pressure without far-context."""

    def __init__(self, model: GPT2_BPA):
        self.model = model
        self.cfg = model.config

    @torch.no_grad()
    def compute_signals(
        self,
        idx: torch.Tensor,
        positions: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cheap signals that don't require far-context scores.

        Args:
            idx: Input tokens [B, T]
            positions: Positions to compute signals at

        Returns:
            Dict of signal tensors, each [B, len(positions), n_layer, n_head]
        """
        B, T = idx.shape
        n_layer = self.cfg.n_layer
        n_head = self.cfg.n_head
        n_pos = len(positions)
        local_window = self.cfg.local_window

        # Get embeddings (RGSA uses transformer.wte/wpe)
        tok_emb = self.model.transformer.wte(idx)  # [B, T, n_embd]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.model.transformer.wpe(pos)  # [T, n_embd]
        x = self.model.transformer.drop(tok_emb + pos_emb)

        # Track signals
        signals = {
            "local_boundary_gap": torch.zeros(B, n_pos, n_layer, n_head),
            "local_entropy": torch.zeros(B, n_pos, n_layer, n_head),
            "escape_proxy": torch.zeros(B, n_pos, n_layer, n_head),
            "residual_spike": torch.zeros(B, n_pos, n_layer),
        }

        # Track residual norms for spike detection
        prev_norm = x.norm(dim=-1, keepdim=True)  # [B, T, 1]

        for layer_idx, block in enumerate(self.model.transformer.h):
            # Get attention module
            attn = block.attn

            # Apply layer norm (as the block does)
            x_normed = block.ln_1(x)

            # Compute Q, K for this layer
            n_embd = self.cfg.n_embd
            head_dim = n_embd // n_head

            # Get QKV projections
            qkv = attn.c_attn(x_normed)  # [B, T, 3 * n_embd]
            q, k, v = qkv.split(n_embd, dim=2)

            # Reshape for multi-head
            q = q.view(B, T, n_head, head_dim).transpose(1, 2)  # [B, H, T, D]
            k = k.view(B, T, n_head, head_dim).transpose(1, 2)

            # Compute attention scores (full, then mask to local)
            scale = 1.0 / (head_dim ** 0.5)
            attn_scores = (q @ k.transpose(-2, -1)) * scale  # [B, H, T, T]

            # Apply causal mask
            causal_mask = torch.triu(
                torch.ones(T, T, device=idx.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

            # For each position, compute cheap signals
            for pos_idx, pos in enumerate(positions):
                if pos < local_window:
                    continue  # Skip positions with no far-context

                # Local window: [pos - local_window, pos]
                local_start = max(0, pos - local_window + 1)
                local_end = pos + 1
                boundary_start = local_start
                boundary_zone = min(10, local_window // 4)  # Boundary region

                for head_idx in range(n_head):
                    # Get attention weights for this position and head
                    pos_scores = attn_scores[:, head_idx, pos, :pos+1]  # [B, pos+1]
                    pos_weights = F.softmax(pos_scores, dim=-1)  # [B, pos+1]

                    # C1: Local-boundary gap
                    # Attention mass at boundary vs elsewhere in local window
                    boundary_mass = pos_weights[:, boundary_start:boundary_start+boundary_zone].sum(dim=-1)
                    interior_mass = pos_weights[:, boundary_start+boundary_zone:local_end].sum(dim=-1)
                    gap = boundary_mass / (interior_mass + 1e-10)
                    signals["local_boundary_gap"][:, pos_idx, layer_idx, head_idx] = gap

                    # C2: Local entropy
                    # How spread out is attention within local window
                    local_weights = pos_weights[:, local_start:local_end]
                    local_weights = local_weights / (local_weights.sum(dim=-1, keepdim=True) + 1e-10)
                    entropy = -(local_weights * (local_weights + 1e-10).log()).sum(dim=-1)
                    max_entropy = np.log(local_end - local_start + 1e-10)
                    norm_entropy = entropy / (max_entropy + 1e-10)
                    signals["local_entropy"][:, pos_idx, layer_idx, head_idx] = norm_entropy

                    # C3: Escape proxy
                    # How much attention concentrates at boundary
                    # High concentration at boundary = "trying to escape"
                    boundary_concentration = pos_weights[:, boundary_start:boundary_start+boundary_zone].max(dim=-1)[0]
                    signals["escape_proxy"][:, pos_idx, layer_idx, head_idx] = boundary_concentration

                # C4: Residual spike (layer-level, not head-level)
                curr_norm = x[:, pos].norm(dim=-1)
                spike = (curr_norm - prev_norm[:, pos, 0].abs()) / (prev_norm[:, pos, 0].abs() + 1e-10)
                signals["residual_spike"][:, pos_idx, layer_idx] = spike

            # Pass through block for next layer (block returns tuple)
            x, _ = block(x)
            prev_norm = x.norm(dim=-1, keepdim=True)

        return signals


def compute_oracle_boundary_pressure(
    model: GPT2_BPA,
    idx: torch.Tensor,
    positions: List[int],
) -> torch.Tensor:
    """Compute true boundary_pressure using full attention."""
    signals = model.compute_conditional_signals(idx, positions=positions)
    return signals["boundary_pressure"]  # [B, n_pos, n_layer, n_head]


def evaluate_predictor(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold_percentile: float = 75,
) -> Dict:
    """Evaluate a cheap predictor against oracle labels."""
    threshold = np.percentile(labels, threshold_percentile)
    binary_labels = (labels > threshold).astype(int)

    if len(np.unique(binary_labels)) < 2:
        return {"auc": 0.5, "pr_auc": 0.5}

    try:
        auc_score = roc_auc_score(binary_labels, predictions)
        precision, recall, _ = precision_recall_curve(binary_labels, predictions)
        pr_auc = auc(recall, precision)

        corr, _ = scipy_stats.spearmanr(predictions, labels)

        return {
            "roc_auc": float(auc_score),
            "pr_auc": float(pr_auc),
            "spearman_r": float(corr) if not np.isnan(corr) else 0.0,
        }
    except ValueError:
        return {"roc_auc": 0.5, "pr_auc": 0.5, "spearman_r": 0.0}


def run_cheap_signal_test(
    model_size: str = "124M",
    seq_len: int = 256,
    n_samples: int = 20,
    n_positions: int = 8,
    seed: int = 1,
) -> Dict:
    """Test cheap predictors against oracle boundary_pressure."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    configs = {
        "124M": dict(n_layer=12, n_head=12, n_embd=768),
    }
    params = configs[model_size]

    cfg = BPAConfig(
        block_size=seq_len,
        vocab_size=50304,
        n_layer=params["n_layer"],
        n_head=params["n_head"],
        n_embd=params["n_embd"],
        local_window=64,
        chunk_size=32,
        top_b=4,
    )

    model = GPT2_BPA(cfg)
    model.eval()

    cheap_computer = CheapSignalComputer(model)

    # Collect predictions and labels
    all_predictions = {
        "local_boundary_gap": [],
        "local_entropy": [],
        "escape_proxy": [],
    }
    all_labels = []

    print(f"\nCollecting {n_samples} samples...")
    for sample_idx in range(n_samples):
        if (sample_idx + 1) % 5 == 0:
            print(f"  Sample {sample_idx + 1}/{n_samples}")

        idx = torch.randint(0, cfg.vocab_size, (2, seq_len))

        # Sample positions past local_window
        positions = sorted(np.random.choice(
            range(cfg.local_window, seq_len),
            size=min(n_positions, seq_len - cfg.local_window),
            replace=False
        ).tolist())

        # Get oracle boundary_pressure
        oracle_bp = compute_oracle_boundary_pressure(model, idx, positions)  # [B, n_pos, n_layer, n_head]

        # Get cheap signals
        cheap_signals = cheap_computer.compute_signals(idx, positions)

        # Flatten and collect
        for name in ["local_boundary_gap", "local_entropy", "escape_proxy"]:
            pred = cheap_signals[name].numpy().flatten()
            all_predictions[name].extend(pred.tolist())

        labels = oracle_bp.numpy().flatten()
        all_labels.extend(labels.tolist())

    # Convert to numpy
    all_labels = np.array(all_labels)
    for name in all_predictions:
        all_predictions[name] = np.array(all_predictions[name])

    print(f"\nCollected {len(all_labels)} measurements")

    # Evaluate each predictor
    results = {
        "seed": seed,
        "n_measurements": len(all_labels),
        "oracle_stats": {
            "mean": float(np.mean(all_labels)),
            "std": float(np.std(all_labels)),
            "p75": float(np.percentile(all_labels, 75)),
        },
        "predictors": {},
    }

    print("\nPredictor Evaluation:")
    print("-" * 60)
    print(f"{'Predictor':<25} {'ROC-AUC':>10} {'PR-AUC':>10} {'Spearman':>10}")
    print("-" * 60)

    for name, predictions in all_predictions.items():
        eval_result = evaluate_predictor(predictions, all_labels)
        results["predictors"][name] = eval_result

        print(f"{name:<25} {eval_result['roc_auc']:>10.4f} "
              f"{eval_result['pr_auc']:>10.4f} {eval_result['spearman_r']:>10.4f}")

    print("-" * 60)

    # Find best predictor
    best_name = max(results["predictors"],
                   key=lambda n: results["predictors"][n]["roc_auc"])
    best_auc = results["predictors"][best_name]["roc_auc"]

    results["best_predictor"] = {
        "name": best_name,
        "roc_auc": best_auc,
        "passes_threshold": best_auc >= 0.75,
    }

    return results


def run_multi_seed(seeds: List[int], n_samples: int = 15) -> Dict:
    """Run cheap signal test across multiple seeds."""
    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        result = run_cheap_signal_test(n_samples=n_samples, seed=seed)
        all_results.append(result)

    # Aggregate
    predictor_names = list(all_results[0]["predictors"].keys())
    aggregated = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "predictors": {},
    }

    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")
    print(f"{'Predictor':<25} {'Mean AUC':>10} {'Std AUC':>10}")
    print("-" * 60)

    best_name = None
    best_auc = 0

    for name in predictor_names:
        aucs = [r["predictors"][name]["roc_auc"] for r in all_results]
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))

        aggregated["predictors"][name] = {
            "mean_roc_auc": mean_auc,
            "std_roc_auc": std_auc,
        }

        print(f"{name:<25} {mean_auc:>10.4f} {std_auc:>10.4f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_name = name

    print("-" * 60)

    aggregated["best_predictor"] = {
        "name": best_name,
        "mean_roc_auc": best_auc,
        "passes_threshold": best_auc >= 0.75,
    }

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="BPA v1 Phase 3: Cheap Signal Test")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Seeds to test")
    parser.add_argument("--samples", type=int, default=15, help="Samples per seed")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 70)
    print("BPA v1 Phase 3: Cheap Signal Predictors")
    print("=" * 70)
    print("\nAcceptance criteria (R3):")
    print("  - A cheap predictor achieves ROC-AUC >= 0.75")
    print("  - Signal computable without far-context scores")
    print()

    results = run_multi_seed(seeds, n_samples=args.samples)

    # Verdict
    best = results["best_predictor"]
    print(f"\nBest predictor: {best['name']}")
    print(f"Mean ROC-AUC: {best['mean_roc_auc']:.4f}")

    if best["passes_threshold"]:
        print("\nVERDICT: R3 PASSES - cheap predictor found")
        print(f"'{best['name']}' can predict boundary_pressure without far-context")
        print("Proceed to Phase 4 (end-to-end PPL)")
    else:
        print(f"\nVERDICT: R3 FAILS - best AUC ({best['mean_roc_auc']:.4f}) < 0.75")
        print("No cheap predictor achieves sufficient accuracy")
        print("BPA cannot deliver compute savings without oracle access")

    # Save results
    os.makedirs("bpa_v1_results", exist_ok=True)
    with open("bpa_v1_results/phase3_cheap.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to bpa_v1_results/phase3_cheap.json")


if __name__ == "__main__":
    main()
