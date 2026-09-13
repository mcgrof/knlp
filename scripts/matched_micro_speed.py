#!/usr/bin/env python3
"""Throughput ladder for the matched micro arms on one GPU.

For each arm: build it, run warm-up training steps (kernel compiles
land here), then time a fixed number of training steps at one batch
size and report tokens per second, peak memory, and parameters. Also
times an evaluation-mode forward on 64-token chunks, the length at
which fla's layers switch to their recurrent kernels, as a proxy for
decode-side cost per token. Writes one JSON.

    python3 scripts/matched_micro_speed.py --arms gdn3,mom3 --batch 32
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matched_micro_train import (
    ARMS,
    CONTRACT,
    TokenStream,
    build_arm,
    ce_loss,
)  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--data-dir", default="matched-micro-data")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--decode-batch", type=int, default=64)
    ap.add_argument("--decode-iters", type=int, default=50)
    ap.add_argument("--out", default="speed.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    CONTRACT["batch_size"] = args.batch
    stream = TokenStream(args.data_dir, CONTRACT["seq_len"], args.batch)
    rows = []
    for arm in args.arms.split(","):
        cfg = ARMS[arm]
        if cfg["kind"] != "stack":
            continue
        torch.cuda.reset_peak_memory_stats()
        model = build_arm(arm, device)
        params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=CONTRACT["lr"])
        model.train()

        def step(i):
            idx = stream.train_batch(i, device)
            opt.zero_grad(set_to_none=True)
            loss = ce_loss(model(idx), idx)
            aux = model.load_balance_loss() if cfg.get("aux_loss_scale") else None
            (loss if aux is None else loss + cfg["aux_loss_scale"] * aux).backward()
            opt.step()

        for i in range(args.warmup):
            step(i)
        sync()
        t0 = time.time()
        for i in range(args.warmup, args.warmup + args.steps):
            step(i)
        sync()
        train_s = time.time() - t0
        tok_s = args.steps * args.batch * CONTRACT["seq_len"] / train_s
        peak = torch.cuda.max_memory_allocated() / 2**30
        # decode proxy: eval-mode forward on 64-token chunks
        model.eval()
        with torch.no_grad():
            x = stream.train_batch(0, device)[: args.decode_batch, :64]
            for _ in range(5):
                model(x)
            sync()
            t0 = time.time()
            for _ in range(args.decode_iters):
                model(x)
            sync()
            chunk_ms = (time.time() - t0) / args.decode_iters * 1000
        row = dict(
            arm=arm,
            params=params,
            batch=args.batch,
            train_tok_s=round(tok_s),
            train_peak_gb=round(peak, 2),
            eval64_ms_per_chunk_batch=round(chunk_ms, 2),
            eval64_us_per_token=round(chunk_ms * 1000 / (args.decode_batch * 64), 2),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, opt
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(
            dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, rows=rows),
            f,
            indent=1,
        )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
