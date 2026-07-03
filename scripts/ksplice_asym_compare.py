#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
ksplice-asym (ASKV) equal-byte comparison on FineWebEdu.

Trains a small GPT-2 with several KV-cache contracts under an identical
budget and reports validation perplexity against the *cache bytes per token*
each contract costs. The point is not SOTA perplexity; it is to test one
architectural claim from the asymmetric-decode result:

  Keys sit on the serial score path before softmax and need fidelity; values
  overlap the reduction and tolerate aggressive compression. So a cache that
  spends its bytes asymmetrically (wide/high-precision K lane, narrow/low-
  precision V lane) should beat a joint latent at equal bytes, and beat the
  "wrong way" contract (quantize K hard, keep V full) even when the wrong-way
  contract is given MORE bytes.

Arms (all at d_model=512, 8 heads, so full K+V = 1024 dims = 2048 B/tok/layer):

  mla_joint    MLA joint latent d_latent=256               -> 512 B  (baseline)
  askv_k16v8   split K=192(16b) / V=128(8b)                -> 512 B  (equal, proposed)
  askv_split16 split K=128(16b) / V=128(16b), no quant     -> 512 B  (equal, symmetric)
  askv_k8v16   split K=192(8b)  / V=128(16b)  WRONG WAY    -> 448 B  (fewer bytes, but K hurt)

If askv_k16v8 <= mla_joint and askv_k16v8 < askv_k8v16 (despite k8v16 being no
cheaper in the lane that matters), the asymmetric contract is the right call.

Runs on a single GPU (W7900 via HIP). Usage:
  python scripts/ksplice_asym_compare.py --data-dir <finewebedu dir> \
      --out-dir <results dir> --iters 1500 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

# Make the repo root importable when run as `python scripts/ksplice_asym_compare.py`
# (sys.path[0] would otherwise be scripts/, not the repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gpt2.mla import MLA_Config, GPT2_MLA, GPT2_MLA_KV_ASYM


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
            for i in ix
        ]
    )
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def eval_ppl(model, val_data, block_size, batch_size, device, n_batches, ctx):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val_data, block_size, batch_size, device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    mean = float(np.mean(losses))
    return mean, math.exp(mean)


def build_model(arm, base_cfg_kwargs, vocab_size):
    cfg = MLA_Config(**base_cfg_kwargs)
    if arm == "mla_joint":
        cfg.d_latent = 256
        return GPT2_MLA(cfg, vocab_size=vocab_size), None
    # ASKV arms differ only in lane widths / precisions
    if arm == "askv_k16v8":
        cfg.d_k_latent, cfg.d_v_latent = 192, 128
        cfg.k_quant_bits, cfg.v_quant_bits = 16, 8
    elif arm == "askv_split16":
        cfg.d_k_latent, cfg.d_v_latent = 128, 128
        cfg.k_quant_bits, cfg.v_quant_bits = 16, 16
    elif arm == "askv_k8v16":  # wrong way: quantize the key lane, keep V full
        cfg.d_k_latent, cfg.d_v_latent = 192, 128
        cfg.k_quant_bits, cfg.v_quant_bits = 8, 16
    else:
        raise ValueError(f"unknown arm {arm}")
    m = GPT2_MLA_KV_ASYM(cfg, vocab_size=vocab_size)
    return m, m.get_compression_stats()


def mla_cache_bytes(cfg):
    # MLA joint latent stored bf16
    return cfg.d_latent * 2.0


def train_arm(arm, args, train_data, val_data, device, ctx, base_cfg_kwargs):
    torch.manual_seed(1337)
    model, stats = build_model(arm, dict(base_cfg_kwargs), args.vocab_size)
    model.to(device)
    if arm == "mla_joint":
        cache_bytes = mla_cache_bytes(model.cfg)
    else:
        cache_bytes = stats["cache_bytes_per_token_per_layer"]

    n_params = model.get_num_params()
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp_dtype == "float16"))

    warmup = max(10, args.iters // 20)

    def lr_at(it):
        if it < warmup:
            return args.lr * it / warmup
        # cosine to 10%
        prog = (it - warmup) / max(1, args.iters - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    model.train()
    t0 = time.time()
    running = []
    for it in range(args.iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            running.append(loss.item() * args.grad_accum)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if it % args.log_every == 0 or it == args.iters - 1:
            avg = float(np.mean(running[-args.log_every :]))
            print(
                f"  [{arm}] it {it:4d}/{args.iters} "
                f"loss {avg:.4f} lr {lr_at(it):.2e} "
                f"{(time.time()-t0):.0f}s",
                flush=True,
            )
    train_time = time.time() - t0
    val_loss, val_ppl = eval_ppl(
        model,
        val_data,
        args.block_size,
        args.batch_size,
        device,
        args.eval_batches,
        ctx,
    )
    result = {
        "arm": arm,
        "n_params": n_params,
        "cache_bytes_per_token_per_layer": cache_bytes,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "train_time_s": train_time,
        "final_train_loss": float(np.mean(running[-args.log_every :])),
        "stats": stats,
    }
    del model, opt
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dir with train.bin/val.bin")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--vocab-size", type=int, default=50304)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--amp-dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument(
        "--arms",
        default="mla_joint,askv_k16v8,askv_split16,askv_k8v16",
        help="comma-separated arm list",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if device.startswith("cuda")
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    )

    train_data = np.memmap(
        os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r"
    )
    print(f"train tokens {len(train_data):,}  val tokens {len(val_data):,}")

    base_cfg_kwargs = dict(
        d_model=args.d_model,
        n_heads=args.n_heads,
        head_dim=args.d_model // args.n_heads,
        n_layers=args.n_layers,
        block_size=args.block_size,
        dropout=0.0,
    )
    full_kv_bytes = 2 * args.n_heads * (args.d_model // args.n_heads) * 2.0
    print(
        f"config: d_model={args.d_model} heads={args.n_heads} layers={args.n_layers} "
        f"block={args.block_size}; full K+V cache = {full_kv_bytes:.0f} B/tok/layer"
    )

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    results = []
    for arm in arms:
        print(f"\n=== training arm: {arm} ===", flush=True)
        r = train_arm(arm, args, train_data, val_data, device, ctx, base_cfg_kwargs)
        r["full_kv_bytes"] = full_kv_bytes
        r["compression_vs_full"] = full_kv_bytes / r["cache_bytes_per_token_per_layer"]
        results.append(r)
        print(
            f"  -> {arm}: val_ppl {r['val_ppl']:.2f}  "
            f"cache {r['cache_bytes_per_token_per_layer']:.0f} B/tok/layer  "
            f"({r['compression_vs_full']:.2f}x vs full)",
            flush=True,
        )
        with open(os.path.join(args.out_dir, "results.json"), "w") as f:
            json.dump(
                {"args": vars(args), "config": base_cfg_kwargs, "results": results},
                f,
                indent=2,
            )

    # Summary table
    print("\n" + "=" * 74)
    print(
        f"{'arm':<14}{'val_ppl':>10}{'val_loss':>10}"
        f"{'bytes/tok':>12}{'x vs full':>11}{'train_s':>9}"
    )
    print("-" * 74)
    for r in sorted(results, key=lambda z: z["val_ppl"]):
        print(
            f"{r['arm']:<14}{r['val_ppl']:>10.2f}{r['val_loss']:>10.4f}"
            f"{r['cache_bytes_per_token_per_layer']:>12.0f}"
            f"{r['compression_vs_full']:>11.2f}{r['train_time_s']:>9.0f}"
        )
    print("=" * 74)
    print(f"results -> {os.path.join(args.out_dir, 'results.json')}")


if __name__ == "__main__":
    main()
