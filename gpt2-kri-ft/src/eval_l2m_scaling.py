"""L2M-style scaling evaluation for KRI-aware fine-tuned models.

Measure how the minimum KV history needed to stay within epsilon
nats/token of the full-attention teacher scales with context length L.

For each (model, L, policy, budget):
  - NLL_full (no pruning) and NLL_sparse (under the policy/budget)
  - KL(pruned_logits || full_logits) over all tokens
  - KL restricted to t > prefill_split (sparse-region-only KL)
  - NLL bucketed by position (0-25%, 25-50%, 50-75%, 75-100% of L)
  - Actual retained-KV tokens (mean, p50, p95) and bytes
  - Decode-region unique retention (size of the union of attended
    positions across all decode-region queries — what a real serving
    system would need to keep in cache if it evicts unattended slots)
  - Selected blocks per layer/head if available

Then for each (model, policy, epsilon) we find B_min(L, eps) by
bisecting over budgets and fit log(B_min) = gamma * log(L) + const.
Three L points → wide CI bars; the fit is *internal-range gamma* for
GPT-2 small (max L = 1024). The Phase 2.9 modern-model port at long
context is what makes the gamma fit a real number.

Output:
  - results/l2m_scaling/<run>/eval_rows.jsonl  raw per-row eval data
  - results/l2m_scaling/<run>/bmin_table.csv   B_min(L, model, policy, eps)
  - results/l2m_scaling/<run>/gamma_fit.csv    fitted gamma + R^2 + 95% CI
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DataConfig, collate, get_tokenizer, get_train_val_streams  # noqa: E402
from src.eval_pruned_ppl import load_model  # noqa: E402
from src.kri_mask import fixed_policy_mask  # noqa: E402
from src.model_gpt2_kri import GPT2KRI  # noqa: E402
from src.utils import pick_device, pick_dtype, report_device, set_seed  # noqa: E402


@dataclass
class BudgetSpec:
    kind: str   # "fixed_blocks" | "fixed_fraction" | "power_law"
    label: str  # human-readable
    topk_blocks: int
    local_window_tokens: int
    sink_blocks: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, required=True,
                   help="Comma-separated list of HF names or checkpoint paths.")
    p.add_argument("--context_lengths", type=str, default="256,512,1024")
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--policies", type=str, default="full,recent,sink_recent,random_global,kri")
    p.add_argument("--sink_blocks", type=int, default=1)
    p.add_argument("--fixed_blocks", type=str, default="4,8,16",
                   help="Fixed top-k block counts to sweep.")
    p.add_argument("--fixed_fractions", type=str, default="0.5,0.25,0.125,0.0625",
                   help="Fixed retained fractions (baseline budgets).")
    p.add_argument("--power_gammas", type=str, default="0.0,0.25,0.5,0.75,1.0",
                   help="Power-law exponents for B(L) = ceil(c * L^gamma / block_size).")
    p.add_argument("--power_c_grid", type=str, default="0.5,1.0,2.0,4.0",
                   help="Constants c in the power-law spec.")
    p.add_argument("--epsilon", type=str, default="0.01,0.03,0.1,0.3",
                   help="nats/token gap thresholds for B_min(L, eps).")
    p.add_argument("--n_batches", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--streaming", type=str, default="false")
    p.add_argument("--val_split", type=str, default="validation")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--precision", type=str, default="auto",
                   choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer_name", type=str, default="openai-community/gpt2",
                   help="Must match the eval'd models' vocab. For SmolLM2 "
                        "evals pass HuggingFaceTB/SmolLM2-360M (or similar).")
    return p.parse_args()


def _truthy(s: str) -> bool:
    return str(s).lower() in ("1", "true", "t", "yes", "y")


def _csv_floats(s: str):
    return tuple(float(x) for x in s.split(","))


def _csv_ints(s: str):
    return tuple(int(x) for x in s.split(","))


def build_budget_specs(L: int, args, block_size: int) -> List[BudgetSpec]:
    """Enumerate the budget specs to sweep at this context length."""
    specs: List[BudgetSpec] = []
    sink_b = args.sink_blocks
    # Local window per spec: when a spec implies a "total cache" budget,
    # we split: sink_b blocks for sink, half of the remaining tokens for
    # local window, other half for top-k global blocks. The split keeps
    # the local window and topk roughly balanced.
    NB = (L + block_size - 1) // block_size

    # 1) fixed top-k counts
    for k in _csv_ints(args.fixed_blocks):
        local_W = max(block_size, L // 8)  # default ~12.5% local
        specs.append(BudgetSpec(
            kind="fixed_blocks",
            label=f"fixedB_{k}",
            topk_blocks=k,
            local_window_tokens=local_W,
            sink_blocks=sink_b,
        ))

    # 2) fixed fractions (baseline curve)
    for frac in _csv_floats(args.fixed_fractions):
        target = max(block_size, int(round(frac * L)))
        sink_tokens = sink_b * block_size
        rem = max(block_size, target - sink_tokens)
        local_W = max(block_size, rem // 3)
        for_kri = max(0, rem - local_W)
        k = max(0, for_kri // block_size)
        specs.append(BudgetSpec(
            kind="fixed_fraction",
            label=f"fixedF_{frac:.4f}",
            topk_blocks=k,
            local_window_tokens=local_W,
            sink_blocks=sink_b,
        ))

    # 3) power-law: B(L) = ceil(c * L^gamma / block_size)
    for gamma in _csv_floats(args.power_gammas):
        for c in _csv_floats(args.power_c_grid):
            k = max(1, int(math.ceil(c * (L ** gamma) / block_size)))
            local_W = max(block_size, L // 8)
            specs.append(BudgetSpec(
                kind="power_law",
                label=f"pow_g{gamma:.2f}_c{c:.2f}",
                topk_blocks=k,
                local_window_tokens=local_W,
                sink_blocks=sink_b,
            ))
    return specs


@torch.no_grad()
def _full_attention_forward(model: GPT2KRI, ids: torch.Tensor, dtype: torch.dtype,
                            device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, log_probs) under full causal attention."""
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
        logits, _ = model(ids)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return logits, log_probs


@torch.no_grad()
def _sparse_attention_forward(model: GPT2KRI, ids: torch.Tensor, mask: torch.Tensor,
                              dtype: torch.dtype, device: torch.device
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
        logits, _ = model(ids, attn_mask=mask)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return logits, log_probs


def _nll_from_logprobs(log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-position NLL (the negative log-prob of the next token).

    log_probs[b, t, :] predicts label[b, t+1]. Returns [B, T-1].
    """
    shift_lp = log_probs[:, :-1, :]
    shift_lab = labels[:, 1:]
    nll = -shift_lp.gather(-1, shift_lab.unsqueeze(-1)).squeeze(-1)
    return nll


def _kl_from_log_probs(p_log: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    """KL(p || q) given log-probabilities of each. Returns [B, T]."""
    p = p_log.exp()
    return (p * (p_log - q_log)).sum(-1)


def _decode_region_unique_retained(mask: torch.Tensor, prefill_split: int) -> int:
    """Size of the union of attended positions across all decode-region
    queries. mask: [B, H, T, T] bool. Returns the average across
    (B, H) of the per-(batch, head) union size.
    """
    B, H, T, _ = mask.shape
    # Restrict to rows in decode region
    sub = mask[:, :, prefill_split + 1 :, :]  # [B, H, T_dec, T]
    union = sub.any(dim=2)  # [B, H, T]  True if any decode query touches it
    return float(union.sum(dim=-1).float().mean().item())


def _kv_bytes(retained_tokens: float, model: GPT2KRI, dtype: torch.dtype) -> float:
    """Estimated KV-cache bytes for `retained_tokens` worth of cache,
    per sequence. Assumes K and V both stored at this dtype, n_layer
    layers, n_head heads of head_dim dims."""
    bytes_per_elt = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}.get(dtype, 4)
    head_dim = model.cfg.n_embd // model.cfg.n_head
    return retained_tokens * model.cfg.n_layer * model.cfg.n_head * head_dim * 2 * bytes_per_elt


def eval_one(model: GPT2KRI, batches: List[dict], policy: str, spec: BudgetSpec,
             L: int, prefill_split: int, block_size: int, device: torch.device,
             dtype: torch.dtype) -> dict:
    """Run one (policy, budget) eval over n_batches cached batches.

    Returns a dict with all the metrics named at the top of this file.
    """
    H = model.cfg.n_head
    nlls_full = []
    nlls_sparse = []
    kls = []
    kls_sparse_region = []
    nlls_by_bin = [[] for _ in range(4)]
    retained_means = []
    retained_p50s = []
    retained_p95s = []
    unique_retentions = []
    n_tokens = 0

    bin_edges = [0, L // 4, L // 2, 3 * L // 4, L]

    for batch in batches:
        ids = batch["input_ids"][:, :L].to(device)
        labels = batch["labels"][:, :L].to(device)
        B, T = ids.shape
        if T < L:
            continue

        full_logits, full_lp = _full_attention_forward(model, ids, dtype, device)
        if policy == "full":
            sparse_lp = full_lp
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device)).view(1, 1, T, T).expand(B, H, T, T)
        else:
            need_kv = policy == "kri" and spec.topk_blocks > 0
            kvs = model.collect_kv(ids) if need_kv else None
            if need_kv:
                k_per = [kv[0] for kv in kvs]
                v_per = [kv[1] for kv in kvs]
                q_per = [kv[0] for kv in kvs]
            else:
                k_per = v_per = q_per = None
            mask, _ = fixed_policy_mask(
                policy=policy, seq_len=T, batch_size=B, n_head=H,
                block_size=block_size,
                local_window_tokens=spec.local_window_tokens,
                sink_blocks=spec.sink_blocks,
                topk_blocks=spec.topk_blocks,
                device=device,
                k_per_layer=k_per, v_per_layer=v_per, q_per_layer=q_per,
            )
            _, sparse_lp = _sparse_attention_forward(model, ids, mask, dtype, device)

        nll_full = _nll_from_logprobs(full_lp, labels)        # [B, T-1]
        nll_sparse = _nll_from_logprobs(sparse_lp, labels)
        nlls_full.append(nll_full)
        nlls_sparse.append(nll_sparse)

        # KL between sparse and full (KL(sparse || full)). All tokens.
        kl_all = _kl_from_log_probs(sparse_lp, full_lp)        # [B, T]
        kls.append(kl_all[:, :-1])

        # Sparse-region-only KL (positions strictly past prefill_split)
        if prefill_split < T - 1:
            kls_sparse_region.append(kl_all[:, prefill_split + 1 : -1])

        # Position-binned NLL
        for i in range(4):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if lo < T - 1:
                slice_ = nll_sparse[:, max(0, lo - 1) : max(0, hi - 1)]
                nlls_by_bin[i].append(slice_)

        # Retention stats
        keep_per_q = mask.sum(-1).float()  # [B, H, T]
        dense_avg = T / 2.0
        frac = keep_per_q / dense_avg
        retained_means.append(float(frac.mean().item()))
        retained_p50s.append(float(torch.quantile(frac.flatten(), 0.5).item()))
        retained_p95s.append(float(torch.quantile(frac.flatten(), 0.95).item()))
        unique_retentions.append(_decode_region_unique_retained(mask, prefill_split))
        n_tokens += B * (T - 1)

    def mean(xs):
        return sum(xs) / max(1, len(xs))

    mean_nll_full = float(torch.cat(nlls_full).mean().item())
    mean_nll_sparse = float(torch.cat(nlls_sparse).mean().item())
    mean_kl = float(torch.cat(kls).mean().item()) if kls else 0.0
    mean_kl_sparse_region = (
        float(torch.cat(kls_sparse_region).mean().item()) if kls_sparse_region else 0.0
    )
    nll_by_bin = [
        float(torch.cat(b).mean().item()) if b else float("nan")
        for b in nlls_by_bin
    ]

    mean_retained_tokens = mean(retained_means) * (L / 2.0)
    return {
        "context_length": L,
        "prefill_split": prefill_split,
        "policy": policy,
        "budget_kind": spec.kind,
        "budget_label": spec.label,
        "topk_blocks": spec.topk_blocks,
        "local_window_tokens": spec.local_window_tokens,
        "sink_blocks": spec.sink_blocks,
        "block_size": block_size,
        "n_eval_tokens": n_tokens,
        "nll_full": mean_nll_full,
        "nll_sparse": mean_nll_sparse,
        "nll_gap": mean_nll_sparse - mean_nll_full,
        "ppl_full": math.exp(mean_nll_full),
        "ppl_sparse": math.exp(mean_nll_sparse) if mean_nll_sparse < 30 else float("inf"),
        "kl_to_full_mean": mean_kl,
        "kl_sparse_region_only": mean_kl_sparse_region,
        "nll_bin_0": nll_by_bin[0],
        "nll_bin_1": nll_by_bin[1],
        "nll_bin_2": nll_by_bin[2],
        "nll_bin_3": nll_by_bin[3],
        "actual_mean_retained_frac": mean(retained_means),
        "actual_p50_retained_frac": mean(retained_p50s),
        "actual_p95_retained_frac": mean(retained_p95s),
        "actual_mean_retained_tokens": mean_retained_tokens,
        "actual_mean_retained_bytes": _kv_bytes(mean_retained_tokens, model, dtype),
        "decode_region_unique_retained_tokens": mean(unique_retentions),
    }


def _budget_axis_for_policy(policy: str) -> str:
    """The independent variable to fit gamma against, per policy.

    - kri / random_global: number of selected top-k global blocks
    - recent / sink_recent: size of the local window in tokens
    - full: not applicable
    """
    if policy in ("kri", "random_global"):
        return "topk_blocks"
    if policy in ("recent", "sink_recent"):
        return "local_window_tokens"
    return "actual_mean_retained_tokens"


def bisect_bmin(rows: List[dict], epsilons: List[float]) -> List[dict]:
    """For each (model, policy, L, epsilon), find the smallest budget
    (per `_budget_axis_for_policy`) such that nll_gap <= epsilon.

    Falls back to `actual_mean_retained_tokens` as the budget axis when
    the policy is full or unknown. If no evaluated budget reaches the
    threshold, B_min is reported as None and `achieved=False`.
    """
    by_key: Dict[Tuple[str, str, int], List[dict]] = {}
    for r in rows:
        if r["policy"] == "full":
            continue
        key = (r["model"], r["policy"], r["context_length"])
        by_key.setdefault(key, []).append(r)
    out = []
    for (model, policy, L), group in by_key.items():
        axis = _budget_axis_for_policy(policy)
        # Sort ascending by the policy-specific budget axis. Smaller
        # is better; first one to achieve the gap is the minimum.
        group_sorted = sorted(group, key=lambda r: r[axis])
        for eps in epsilons:
            achieved = False
            best = None
            for r in group_sorted:
                if r["nll_gap"] <= eps:
                    achieved = True
                    best = r
                    break
            out.append({
                "model": model,
                "policy": policy,
                "context_length": L,
                "epsilon": eps,
                "B_min_axis": axis,
                "B_min": (best[axis] if achieved else None),
                "B_min_topk_blocks": (best["topk_blocks"] if achieved else None),
                "B_min_local_window": (best["local_window_tokens"] if achieved else None),
                "B_min_actual_tokens": (best["actual_mean_retained_tokens"] if achieved else None),
                "B_min_actual_frac": (best["actual_mean_retained_frac"] if achieved else None),
                "achieved": achieved,
                "nll_gap_at_bmin": (best["nll_gap"] if achieved else None),
            })
    return out


def fit_gamma(bmin_rows: List[dict]) -> List[dict]:
    """For each (model, policy, epsilon), fit log(B_min) = gamma * log(L) + const.
    Only rows with achieved=True contribute. Requires >=3 L points for a fit.
    """
    by_key: Dict[Tuple[str, str, float], List[dict]] = {}
    for r in bmin_rows:
        if not r["achieved"] or r.get("B_min") is None:
            continue
        key = (r["model"], r["policy"], r["epsilon"])
        by_key.setdefault(key, []).append(r)
    out = []
    for (model, policy, eps), group in by_key.items():
        if len(group) < 2:
            out.append({"model": model, "policy": policy, "epsilon": eps,
                        "n_points": len(group), "gamma": None, "intercept": None,
                        "r2": None, "note": "insufficient L points"})
            continue
        xs = [math.log(r["context_length"]) for r in group]
        ys = [math.log(max(1, r["B_min"])) for r in group]
        n = len(xs)
        xb, yb = sum(xs) / n, sum(ys) / n
        num = sum((xs[i] - xb) * (ys[i] - yb) for i in range(n))
        den = sum((xs[i] - xb) ** 2 for i in range(n))
        if den == 0:
            out.append({"model": model, "policy": policy, "epsilon": eps,
                        "n_points": n, "gamma": None, "intercept": None,
                        "r2": None, "note": "degenerate xs"})
            continue
        gamma = num / den
        intercept = yb - gamma * xb
        ss_res = sum((ys[i] - (gamma * xs[i] + intercept)) ** 2 for i in range(n))
        ss_tot = sum((ys[i] - yb) ** 2 for i in range(n))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        out.append({
            "model": model, "policy": policy, "epsilon": eps,
            "n_points": n, "gamma": gamma, "intercept": intercept, "r2": r2, "note": "",
        })
    return out


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    report_device()
    device = pick_device()
    dtype = pick_dtype(args.precision)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl = out_dir / "eval_rows.jsonl"
    bmin_csv = out_dir / "bmin_table.csv"
    gamma_csv = out_dir / "gamma_fit.csv"
    summary_path = out_dir / "summary.txt"

    context_lengths = _csv_ints(args.context_lengths)
    policies = args.policies.split(",")
    epsilons = list(_csv_floats(args.epsilon))
    model_paths = args.models.split(",")

    # Stream rows incrementally so a crash mid-sweep doesn't wipe out
    # the work that completed.
    rows: List[dict] = []
    eval_jsonl_fh = eval_jsonl.open("w")

    for m_path in model_paths:
        print(f"\n=== model: {m_path} ===")
        model, tag = load_model(m_path, device)
        tok = get_tokenizer(args.tokenizer_name)

        for L in context_lengths:
            print(f"\n  -- context_length L={L}")
            data_cfg = DataConfig(
                dataset_name=args.dataset_name,
                dataset_config=args.dataset_config,
                text_column=args.text_column,
                streaming=_truthy(args.streaming),
                train_split=args.train_split,
                val_split=args.val_split,
                seq_len=L,
            )
            _, val_ds = get_train_val_streams(data_cfg, tok)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                    collate_fn=collate, num_workers=0)
            cached = []
            for i, b in enumerate(val_loader):
                if i >= args.n_batches:
                    break
                cached.append(b)
            print(f"    cached {len(cached)} batches at L={L}")

            prefill_split = L // 4  # decode region is the latter 3/4

            specs = build_budget_specs(L, args, args.block_size)

            for policy in policies:
                if policy == "full":
                    res = eval_one(model, cached, "full",
                                   BudgetSpec(kind="full", label="full",
                                              topk_blocks=0,
                                              local_window_tokens=L,
                                              sink_blocks=0),
                                   L, prefill_split, args.block_size, device, dtype)
                    res["model"] = tag
                    rows.append(res)
                    eval_jsonl_fh.write(json.dumps(res) + "\n")
                    eval_jsonl_fh.flush()
                    print(f"    full L={L}: nll={res['nll_full']:.4f} ppl={res['ppl_full']:.2f}")
                    continue
                for spec in specs:
                    res = eval_one(model, cached, policy, spec, L, prefill_split,
                                   args.block_size, device, dtype)
                    res["model"] = tag
                    rows.append(res)
                    eval_jsonl_fh.write(json.dumps(res) + "\n")
                    eval_jsonl_fh.flush()
                    print(
                        f"    {policy:14s} L={L} {spec.label:20s} "
                        f"topk={spec.topk_blocks:3d} W={spec.local_window_tokens:4d} "
                        f"nll_gap={res['nll_gap']:.4f} "
                        f"kl={res['kl_to_full_mean']:.4f} "
                        f"act_frac={res['actual_mean_retained_frac']:.3f}"
                    )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    eval_jsonl_fh.close()

    # B_min lookup table and gamma fit
    bmin_rows = bisect_bmin(rows, epsilons)
    with bmin_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bmin_rows[0].keys()) if bmin_rows else
                           ["model", "policy", "context_length", "epsilon",
                            "B_min_axis", "B_min", "B_min_topk_blocks",
                            "B_min_local_window", "B_min_actual_tokens",
                            "B_min_actual_frac", "achieved", "nll_gap_at_bmin"])
        w.writeheader()
        for r in bmin_rows:
            w.writerow(r)

    gamma_rows = fit_gamma(bmin_rows)
    with gamma_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gamma_rows[0].keys()) if gamma_rows else
                           ["model", "policy", "epsilon", "n_points", "gamma",
                            "intercept", "r2", "note"])
        w.writeheader()
        for r in gamma_rows:
            w.writerow(r)

    with summary_path.open("w") as fh:
        fh.write(f"L2M scaling eval — {len(rows)} eval rows, "
                 f"{len(bmin_rows)} B_min rows, {len(gamma_rows)} gamma fits\n\n")
        fh.write(f"context lengths: {context_lengths}\n")
        fh.write(f"models:          {model_paths}\n")
        fh.write(f"policies:        {policies}\n")
        fh.write(f"epsilons:        {epsilons}\n\n")
        fh.write("=== gamma fits ===\n")
        for r in sorted(gamma_rows, key=lambda r: (r["model"], r["policy"], r["epsilon"])):
            g = r["gamma"]
            g_s = f"{g:+.3f}" if g is not None else "  n/a"
            r2 = r["r2"]
            r2_s = f"{r2:.3f}" if r2 is not None else "  n/a"
            fh.write(f"  {r['model'][:40]:40s}  policy={r['policy']:14s}  "
                     f"eps={r['epsilon']:.2f}  n={r['n_points']}  "
                     f"gamma={g_s}  r2={r2_s}  {r['note']}\n")
    print(f"\nwrote {len(rows)} eval rows -> {eval_jsonl}")
    print(f"wrote {len(bmin_rows)} B_min rows -> {bmin_csv}")
    print(f"wrote {len(gamma_rows)} gamma fits -> {gamma_csv}")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    # Bypass the GIL race on shutdown that bit train_kri.py
    import os
    rc = main()
    os._exit(rc)
