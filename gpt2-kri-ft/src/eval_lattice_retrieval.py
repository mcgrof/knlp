"""Retrieval check for the Lattice / Trellis routers.

Does trellis_kri's query-AGNOSTIC win on KL/PPL survive a query-SPECIFIC
retrieval task? Reuses the synthetic key=value retrieval from
eval_synthetic_retrieval (a doc of `key_i = adj noun` pairs, query
`lookup key_j =`, rank the gold value above distractors) but routes the KV
cache through the route_mask dispatch (per-position surface) so trellis_kri /
lattice_* / kri_q(+novelty) compare head-to-head at matched retained KV.

The prediction under test: trellis_kri keeps the highest-SURPRISE blocks
regardless of the query, so it should drop the needle block (the queried
key's pair) more often than query-conditioned KRI-Q — winning PPL/KL but
losing retrieval. key_position=early places the needle far from the query so
the local window can't trivially cover it; that is the routing-dependent case.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import get_tokenizer  # noqa: E402
from src.eval_lattice_kri import route_mask  # noqa: E402
from src.eval_pruned_ppl import load_model  # noqa: E402
from src.eval_synthetic_retrieval import make_example  # noqa: E402
from src.utils import pick_device, pick_dtype, report_device, set_seed  # noqa: E402

DEFAULT_ROUTERS = (
    "full,sink_recent,kri_q,kri_q_novelty,lattice_kri_residual_rel,trellis_kri"
)


@torch.no_grad()
def _cont_logprob(
    model, tok, prefix, cont, device, dtype, seq_len, router, topk, bs, lw, sink, latkw
):
    """Summed logprob of `cont` given `prefix` under one router, plus the
    actual retained fraction of the mask."""
    ids = tok(
        f"{prefix} {cont}".strip(), return_tensors="pt", add_special_tokens=False
    )["input_ids"][:, :seq_len].to(device)
    pre_len = min(
        tok(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].size(1),
        ids.size(1),
    )
    T = ids.size(1)
    if pre_len >= T:
        return float("-inf"), 0.0
    H = model.cfg.n_head
    if router == "full":
        mask, ret = None, 1.0
    else:
        kvs = model.collect_kv(ids)
        li = min(6, len(kvs) - 1)
        k, v = kvs[li][0], kvs[li][1]
        # prefill_split=0 routes every position (incl. the query/continuation),
        # so the test really probes whether the router keeps the needle block.
        mask, _ = route_mask(
            router,
            k,
            v,
            T,
            bs,
            lw,
            sink,
            topk,
            device,
            latkw,
            surface="per_position",
            prefill_split=0,
        )
        ret = float(mask.sum(-1).float().mean() / (T / 2.0))
    with torch.autocast(
        device_type=device.type, dtype=dtype, enabled=dtype != torch.float32
    ):
        logits, _ = model(ids, attn_mask=mask)
    lp = F.log_softmax(logits.float(), dim=-1)
    summed = sum(
        float(lp[0, t - 1, ids[0, t].item()].item()) for t in range(pre_len, T)
    )
    return summed, ret


def _score_example(
    model, tok, ex, device, dtype, seq_len, router, topk, bs, lw, sink, latkw
):
    gold_lp, ret = _cont_logprob(
        model,
        tok,
        ex["prefix"],
        ex["gold"],
        device,
        dtype,
        seq_len,
        router,
        topk,
        bs,
        lw,
        sink,
        latkw,
    )
    dlps = [
        _cont_logprob(
            model,
            tok,
            ex["prefix"],
            d,
            device,
            dtype,
            seq_len,
            router,
            topk,
            bs,
            lw,
            sink,
            latkw,
        )[0]
        for d in ex["distractors"]
    ]
    rank = 1 + sum(1 for x in dlps if x > gold_lp)
    return {
        "top1": int(rank == 1),
        "rank": rank,
        "margin": gold_lp - max(dlps) if dlps else float("inf"),
        "ret": ret,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True)
    p.add_argument("--routers", default=DEFAULT_ROUTERS)
    p.add_argument("--topks", default="4,8,16")
    p.add_argument("--num_examples", type=int, default=150)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--local_window_tokens", type=int, default=64)
    p.add_argument("--sink_blocks", type=int, default=1)
    p.add_argument("--n_distractors", type=int, default=8)
    p.add_argument("--num_pairs_choices", default="48,64")
    p.add_argument(
        "--key_position", default="early", choices=["random", "early", "middle", "late"]
    )
    p.add_argument("--lambda_orth", type=float, default=0.25)
    p.add_argument("--summary_mode", default="k")
    p.add_argument(
        "--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"]
    )
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    report_device()
    device = pick_device()
    dtype = pick_dtype(args.precision)
    tok = get_tokenizer("openai-community/gpt2")
    rng = random.Random(args.seed)
    pair_choices = [int(x) for x in args.num_pairs_choices.split(",")]
    examples = [
        make_example(
            rng,
            pair_choices[i % len(pair_choices)],
            args.key_position,
            args.n_distractors,
        )
        for i in range(args.num_examples)
    ]
    print(
        f"generated {len(examples)} retrieval examples "
        f"(key_position={args.key_position}, pairs in {pair_choices})",
        flush=True,
    )
    latkw = dict(lambda_orth=args.lambda_orth, summary_mode=args.summary_mode)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, fh = [], out.open("w")
    for m_path in args.models.split(","):
        print(f"\n=== model: {m_path} ===", flush=True)
        model, tag = load_model(m_path, device)
        for router in args.routers.split(","):
            budgets = (
                [0]
                if router in ("full", "sink_recent")
                else [int(x) for x in args.topks.split(",")]
            )
            for topk in budgets:
                st = [
                    _score_example(
                        model,
                        tok,
                        ex,
                        device,
                        dtype,
                        args.seq_len,
                        router,
                        topk,
                        args.block_size,
                        args.local_window_tokens,
                        args.sink_blocks,
                        latkw,
                    )
                    for ex in examples
                ]
                n = len(st)
                row = {
                    "model": tag,
                    "router": router,
                    "topk": topk,
                    "key_position": args.key_position,
                    "seq_len": args.seq_len,
                    "seed": args.seed,
                    "top1_accuracy": sum(s["top1"] for s in st) / n,
                    "mean_rank": sum(s["rank"] for s in st) / n,
                    "retained_frac": sum(s["ret"] for s in st) / n,
                }
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(
                    f"  {router:26s} K={topk:>3d} ret={row['retained_frac']:.3f} "
                    f"top1={row['top1_accuracy']:.3f} rank={row['mean_rank']:.2f}",
                    flush=True,
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    if rows:
        with out.with_suffix(".csv").open("w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    import os

    os._exit(main())
