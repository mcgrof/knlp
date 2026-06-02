"""Phase 2.6.2 / task #31 — KRI-FT vs canonical KRI-family routers.

Run the four canonical KRI routers (kri_q, kri_q_window, kri_g,
kri_d) from src/canonical_kri.py against vanilla / dense-FT /
KRI-FT, on the standard pruned-PPL eval format. The output is a
table per (model, router, K).

If KRI-FT outperforms vanilla / dense-FT under the canonical
routers — particularly kri_q_window K=8 — that strengthens the
cross-router generalisation claim. If KRI-FT only wins under our
internal KRI-Q+N policy, the claim narrows to "model is
compatible with the specific training-time mask schedule."
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.canonical_kri import canonical_router_mask  # noqa: E402
from src.data import DataConfig, collate, get_tokenizer, get_train_val_streams  # noqa: E402
from src.eval_pruned_ppl import load_model  # noqa: E402
from src.model_gpt2_kri import GPT2KRI  # noqa: E402
from src.utils import pick_device, pick_dtype, report_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, required=True)
    p.add_argument("--routers", type=str, default="kri_q,kri_q_window,kri_g,kri_d")
    p.add_argument("--topks", type=str, default="1,4,8,16")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--local_window_tokens", type=int, default=128)
    p.add_argument("--sink_blocks", type=int, default=1)
    p.add_argument("--score_layer_index", type=int, default=0,
                   help="Layer at which to extract Key centroids and query "
                        "probe. 0 matches the prior KRI-Q work; the "
                        "training-time mask in this repo defaults to layer 6.")
    p.add_argument("--n_batches", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--streaming", type=str, default="false")
    p.add_argument("--val_split", type=str, default="validation")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--precision", type=str, default="auto",
                   choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument(
        "--tokenizer_name",
        type=str,
        default="openai-community/gpt2",
        help=(
            "Tokenizer for the validation stream. Must match the model "
            "vocab (SmolLM2: HuggingFaceTB/SmolLM2-360M)."
        ),
    )
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _truthy(s: str) -> bool:
    return str(s).lower() in ("1", "true", "t", "yes", "y")


def _csv_ints(s: str):
    return tuple(int(x) for x in s.split(","))


@torch.no_grad()
def _full_log_probs(model, ids, dtype, device):
    with torch.autocast(device_type=device.type, dtype=dtype,
                        enabled=dtype != torch.float32):
        logits, _ = model(ids)
    return F.log_softmax(logits.float(), dim=-1), logits


@torch.no_grad()
def _stats_under_mask(model, ids, labels, mask, full_lp, dtype, device,
                      prefill_split: int):
    with torch.autocast(device_type=device.type, dtype=dtype,
                        enabled=dtype != torch.float32):
        logits, _ = model(ids, attn_mask=mask)
    pruned_lp = F.log_softmax(logits.float(), dim=-1)
    nll_full = -(full_lp[:, :-1, :].gather(-1, labels[:, 1:].unsqueeze(-1)).squeeze(-1))
    nll_sparse = -(pruned_lp[:, :-1, :].gather(-1, labels[:, 1:].unsqueeze(-1)).squeeze(-1))
    p = pruned_lp.exp()
    kl = (p * (pruned_lp - full_lp)).sum(-1)
    kl_decode = kl[:, prefill_split + 1 :]
    return {
        "nll_full": float(nll_full.mean().item()),
        "nll_sparse": float(nll_sparse.mean().item()),
        "kl_mean": float(kl.mean().item()),
        "kl_decode": float(kl_decode.mean().item()) if kl_decode.numel() else 0.0,
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    report_device()
    device = pick_device()
    dtype = pick_dtype(args.precision)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_out = out.with_suffix(".csv")

    routers = args.routers.split(",")
    topks = _csv_ints(args.topks)
    models = args.models.split(",")
    L = args.seq_len
    prefill_split = L // 4

    tok = get_tokenizer(args.tokenizer_name)
    rows = []
    fh = out.open("w")

    for m_path in models:
        print(f"\n=== model: {m_path} ===")
        model, tag = load_model(m_path, device)
        H = model.cfg.n_head
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
        if not cached:
            print("  no batches; skipping model")
            del model
            continue
        print(f"  cached {len(cached)} batches at L={L}")

        for router in routers:
            for topk in topks:
                agg = {"kl_mean": 0.0, "kl_decode": 0.0,
                       "nll_full": 0.0, "nll_sparse": 0.0}
                for batch in cached:
                    ids = batch["input_ids"][:, :L].to(device)
                    labels = batch["labels"][:, :L].to(device)
                    full_lp, _ = _full_log_probs(model, ids, dtype, device)
                    kvs = model.collect_kv(ids)
                    k_per = [kv[0] for kv in kvs]
                    v_per = [kv[1] for kv in kvs]
                    q_per = [kv[0] for kv in kvs]  # K stands in for the q-probe
                    mask = canonical_router_mask(
                        router,
                        k_per_layer=k_per, v_per_layer=v_per, q_per_layer=q_per,
                        seq_len=L, block_size=args.block_size,
                        local_window_tokens=args.local_window_tokens,
                        sink_blocks=args.sink_blocks,
                        topk_blocks=topk,
                        score_layer_index=args.score_layer_index,
                        device=device,
                    )
                    stats = _stats_under_mask(model, ids, labels, mask, full_lp,
                                                dtype, device, prefill_split)
                    for k in agg:
                        agg[k] += stats[k]
                for k in agg:
                    agg[k] /= max(1, len(cached))
                ppl = math.exp(agg["nll_sparse"]) if agg["nll_sparse"] < 30 else float("inf")
                row = {
                    "model": tag, "router": router, "topk": topk,
                    "local_window_tokens": args.local_window_tokens,
                    "sink_blocks": args.sink_blocks,
                    "block_size": args.block_size,
                    "score_layer_index": args.score_layer_index,
                    "seq_len": L, "prefill_split": prefill_split,
                    **agg, "ppl_sparse": ppl,
                }
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"  {router:14s} K={topk:>3d}  KL={agg['kl_mean']:.4f} "
                      f"KL_decode={agg['kl_decode']:.4f} PPL={ppl:.3f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fh.close()
    if rows:
        with csv_out.open("w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {out} and {csv_out}")
    return 0


if __name__ == "__main__":
    import os
    rc = main()
    os._exit(rc)
