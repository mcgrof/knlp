#!/usr/bin/env python
"""
Train KVSplice segment compressor for v14b evaluation.

Trains a learned linear projection that compresses consecutive KV tokens
into segment representations, optimized via attention KL divergence and
value reconstruction loss against the dense teacher.

Saves checkpoint to kvsplice_trained/kvsplice_seg{S}.pt

Usage:
    python scripts/train_kvsplice.py \
        --segment_size 4 \
        --train_steps 5000 \
        --lr 1e-3 \
        --device cuda
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backends.kvsplice_kl import SegmentCompressor
from scripts.bpa_v11_bench import (
    DTYPE,
    get_text_batch,
    load_validation_tokens,
)


def train_kvsplice(args):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device = args.device

    # Load model
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim = config.hidden_size // n_heads

    print(f"  layers={n_layers} kv_heads={n_kv_heads} head_dim={head_dim}")

    # Load calibration data
    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)

    # Create compressor
    seg = args.segment_size
    compressor = SegmentCompressor(head_dim, seg, n_kv_heads, device, DTYPE)

    # Collect training data from multiple calibration sequences
    W_min = 1024
    W_sink = 4
    train_data = []

    print("Collecting training data from multiple sequences...")
    for cal_seed in range(args.n_cal_seqs):
        rng = np.random.RandomState(42 + cal_seed)
        cal_len = min(args.cal_len, 8192)
        idx = get_text_batch(token_data, 1, cal_len, rng).to(device)

        with torch.no_grad():
            out = model(idx, use_cache=True)
            past = out.past_key_values

        # Sample layers
        train_layers = list(range(0, n_layers, max(1, n_layers // 6)))[:6]

        for li in train_layers:
            k_full, v_full = past[li]
            near_start = max(W_sink, cal_len - W_min)
            far_end = near_start
            n_far = far_end - W_sink

            if n_far < seg:
                continue

            k_far = k_full[:, :, W_sink:far_end, :].detach()
            v_far = v_full[:, :, W_sink:far_end, :].detach()

            n_segs = n_far // seg
            k_far = k_far[:, :, : n_segs * seg, :]
            v_far = v_far[:, :, : n_segs * seg, :]

            # Use K from near window as pseudo-queries
            k_near = k_full[:, :, far_end:, :].detach()
            q_samples = k_near[:, :, : min(64, k_near.shape[2]), :]

            train_data.append((q_samples, k_far, v_far, li))

        del past, out
        torch.cuda.empty_cache()

    if not train_data:
        print("ERROR: no training data collected")
        return

    print(f"  Collected {len(train_data)} training batches")

    # Train
    optimizer = torch.optim.Adam(compressor.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.train_steps
    )

    best_loss = float("inf")
    temperature = args.temperature
    log_every = max(1, args.train_steps // 20)

    print(f"\nTraining for {args.train_steps} steps...")
    t0 = time.perf_counter()

    for step in range(args.train_steps):
        # Sample a random training batch
        idx = step % len(train_data)
        q, k_far, v_far, li = train_data[idx]

        k_seg, v_seg = compressor(k_far, v_far)

        q_f = q.float()
        scale = head_dim**0.5

        # Teacher logits
        logits_t = torch.matmul(q_f, k_far.float().transpose(-2, -1)) / scale
        # Student logits
        logits_s = torch.matmul(q_f, k_seg.float().transpose(-2, -1)) / scale

        # Attention outputs
        attn_t = F.softmax(logits_t / temperature, dim=-1)
        out_t = torch.matmul(attn_t, v_far.float())

        attn_s = F.softmax(logits_s / temperature, dim=-1)
        out_s = torch.matmul(attn_s, v_seg.float())

        # Value reconstruction loss
        value_loss = F.mse_loss(out_s, out_t.detach())

        # Segment-level KL divergence
        T_far = k_far.shape[2]
        n_segs_cur = k_seg.shape[2]

        attn_t_segs = (
            attn_t[:, :, :, : n_segs_cur * seg]
            .reshape(1, q.shape[1], q.shape[2], n_segs_cur, seg)
            .sum(dim=-1)
        )

        attn_t_norm = attn_t_segs / (attn_t_segs.sum(dim=-1, keepdim=True) + 1e-8)
        attn_s_norm = attn_s / (attn_s.sum(dim=-1, keepdim=True) + 1e-8)

        kl_loss = F.kl_div(
            (attn_s_norm + 1e-8).log(),
            attn_t_norm,
            reduction="batchmean",
        )

        loss = value_loss + 0.1 * kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(compressor.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if step % log_every == 0 or step == args.train_steps - 1:
            elapsed = time.perf_counter() - t0
            print(
                f"  step {step:5d}/{args.train_steps}: "
                f"loss={loss.item():.6f} val={value_loss.item():.6f} "
                f"kl={kl_loss.item():.6f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"[{elapsed:.0f}s]"
            )

    elapsed = time.perf_counter() - t0
    compressor.eval()

    # Save checkpoint
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, f"kvsplice_seg{seg}.pt")
    torch.save(
        {
            "compressor_state": compressor.state_dict(),
            "segment_size": seg,
            "head_dim": head_dim,
            "n_kv_heads": n_kv_heads,
            "train_steps": args.train_steps,
            "best_loss": best_loss,
            "final_loss": loss.item(),
            "lr": args.lr,
            "temperature": temperature,
            "cal_len": args.cal_len,
            "n_cal_seqs": args.n_cal_seqs,
        },
        ckpt_path,
    )
    print(f"\nSaved checkpoint: {ckpt_path}")
    print(f"  best_loss={best_loss:.6f} final_loss={loss.item():.6f}")
    print(f"  training time: {elapsed:.1f}s")

    # Save training meta
    meta_path = os.path.join(args.outdir, f"train_meta_seg{seg}.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "segment_size": seg,
                "train_steps": args.train_steps,
                "best_loss": best_loss,
                "final_loss": loss.item(),
                "lr": args.lr,
                "temperature": temperature,
                "elapsed_s": round(elapsed, 1),
                "n_cal_seqs": args.n_cal_seqs,
                "cal_len": args.cal_len,
            },
            f,
            indent=2,
        )

    return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment_size", type=int, default=4)
    parser.add_argument("--train_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cal_len", type=int, default=8192)
    parser.add_argument("--n_cal_seqs", type=int, default=4)
    parser.add_argument("--outdir", default="kvsplice_trained")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train_kvsplice(args)


if __name__ == "__main__":
    main()
