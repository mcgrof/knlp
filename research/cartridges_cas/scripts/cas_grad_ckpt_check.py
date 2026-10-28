#!/usr/bin/env python3
"""Check that activation recomputation leaves the cartridge trainer's
loss and gradient unchanged, and measure what it costs.

``patches/grad_checkpoint.py`` lets ``CARTRIDGES_GRAD_CKPT=1`` recompute
decoder activations in the backward pass so one packed 2048-token
micro-batch of Qwen3-8B fits a 24 GiB GPU. Recomputation is supposed to
change no arithmetic. This script measures that on the target GPU
instead of assuming it:

1. On a short packed sequence (``SHORT_SEQ``, default 512) that fits
   with and without recomputation, run the trainer's exact forward and
   loss (``-p log q`` over the stored top-20 entries) and one backward
   pass into the cartridge, both ways. Report the loss difference and
   the gradient difference (max abs, relative L2, cosine). Each way is
   also run twice, so the kernel nondeterminism floor is reported next
   to the patch's effect.
2. On the trainer's real sequence length (``FULL_SEQ``, default 2048),
   try one unpatched micro-batch (expected to run out of memory on a
   24 GiB card; the failure is caught and reported), then time
   ``N_TIME`` patched micro-batches and report peak memory and the
   implied seconds per optimizer step at ``GLOBAL_BS`` micro-batches.

Env:
    MODEL      Qwen/Qwen3-8B (default)
    PARQUET    training parquet (any self-study split)
    CART       cartridge file for the cache init (first-tokens anchor)
    ROWS       rows to load (default 64)
    SHORT_SEQ  default 512; FULL_SEQ default 2048
    N_TIME     patched micro-batches to time (default 4)
    GLOBAL_BS  micro-batches per optimizer step (default 128)
    OUT_JSON   results (default grad_ckpt_check.json)

Prints ``GRADCKPT,<key>,<value>`` lines and ``GRADCKPT_DONE``.
"""

import json
import os
import sys
import time

os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/cas_grad_ckpt_check")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.datasets import TrainDataset, DataSource
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cas_cart_init import load_cart  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
PARQUET = os.environ["PARQUET"]
CART = os.environ["CART"]
ROWS = int(os.environ.get("ROWS", "64"))
SHORT_SEQ = int(os.environ.get("SHORT_SEQ", "512"))
FULL_SEQ = int(os.environ.get("FULL_SEQ", "2048"))
N_TIME = int(os.environ.get("N_TIME", "4"))
GLOBAL_BS = int(os.environ.get("GLOBAL_BS", "128"))
OUT_JSON = os.environ.get("OUT_JSON", "grad_ckpt_check.json")
DEVICE = "cuda"


def build_dataset(tokenizer, seq_len):
    cfg = TrainDataset.Config(
        data_sources=[DataSource(path=PARQUET, type="local", limit=ROWS)],
        top_k_logits=20,
        packed_seq_length=seq_len,
        packing_mode="truncate",
    )
    return cfg.instantiate(tokenizer=tokenizer, seed=42)


def make_cache(attn):
    keys, values, nfrozen = load_cart(CART, device=DEVICE)
    return TrainableCache(
        config=attn, init_keys=keys, init_values=values, num_frozen_tokens=nfrozen
    ).to(DEVICE)


def grad_pass(model, cache, batch, ckpt):
    """The trainer's forward, loss and backward for one micro-batch.
    Returns (loss, flat fp32 cartridge gradient on CPU, seconds, peak GiB)."""
    os.environ["CARTRIDGES_GRAD_CKPT"] = "1" if ckpt else "0"
    for p in cache.parameters():
        p.grad = None
    cache.clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=batch.input_ids.to(DEVICE),
            seq_ids=batch.element_ids.to(DEVICE),
            position_ids=batch.position_ids.to(DEVICE),
            use_cache=True,
            past_key_values=cache,
        )
        pred = F.log_softmax(out.logits, dim=-1)[
            0, batch.topk_token_idxs.to(DEVICE) - 1, batch.topk_token_ids.to(DEVICE)
        ]
        loss = (-batch.topk_logprobs.to(DEVICE).exp() * pred).mean()
    loss.backward()
    torch.cuda.synchronize()
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    # the frozen sink slots carry no gradient; the trainable ones must
    grads = [
        p.grad.detach().float().flatten().cpu()
        for p in cache.parameters()
        if p.requires_grad and p.grad is not None
    ]
    n_trainable = sum(1 for p in cache.parameters() if p.requires_grad)
    assert len(grads) == n_trainable, f"{len(grads)} grads for {n_trainable} params"
    grads = torch.cat(grads)
    del out, pred
    return float(loss), grads, secs, peak


def compare(ga, gb):
    return {
        "max_abs": float((ga - gb).abs().max()),
        "rel_l2": float((ga - gb).norm() / ga.norm()),
        "cosine": float(F.cosine_similarity(ga, gb, dim=0)),
        "norm": float(ga.norm()),
    }


def main():
    res = {"model": MODEL, "parquet": PARQUET, "cart": CART, "gpu": None}
    res["gpu"] = torch.cuda.get_device_name(0)
    res["total_gib"] = torch.cuda.get_device_properties(0).total_memory / 2**30
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = (
        FlexQwen3ForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        .to(DEVICE)
        .to(torch.bfloat16)
    )
    for p in model.parameters():
        p.requires_grad = False
    attn = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
    )
    cache = make_cache(attn)
    print(f"GRADCKPT,weights_gib,{torch.cuda.memory_allocated() / 2**30:.2f}")

    # 1. parity on the short sequence
    ds = build_dataset(tok, SHORT_SEQ)
    b = ds[0]
    runs = {}
    for name, ckpt in (("off", False), ("off2", False), ("on", True), ("on2", True)):
        loss, g, secs, peak = grad_pass(model, cache, b, ckpt)
        runs[name] = (loss, g)
        res[f"short_{name}"] = {"loss": loss, "secs": secs, "peak_gib": peak}
        print(f"GRADCKPT,short_{name},loss={loss:.8f},secs={secs:.2f},peak={peak:.2f}")
    res["short_seq"] = SHORT_SEQ
    res["parity_on_vs_off"] = compare(runs["off"][1], runs["on"][1])
    res["floor_off_vs_off2"] = compare(runs["off"][1], runs["off2"][1])
    res["floor_on_vs_on2"] = compare(runs["on"][1], runs["on2"][1])
    res["loss_diff_on_vs_off"] = runs["on"][0] - runs["off"][0]
    res["loss_diff_off_vs_off2"] = runs["off2"][0] - runs["off"][0]
    for k in ("parity_on_vs_off", "floor_off_vs_off2", "floor_on_vs_on2"):
        v = res[k]
        print(
            f"GRADCKPT,{k},max_abs={v['max_abs']:.3e},rel_l2={v['rel_l2']:.3e},"
            f"cosine={v['cosine']:.8f}"
        )
    print(
        f"GRADCKPT,loss_diff,on_vs_off={res['loss_diff_on_vs_off']:.3e},"
        f"off_vs_off2={res['loss_diff_off_vs_off2']:.3e}"
    )
    del ds, b, runs
    torch.cuda.empty_cache()

    # 2. the real sequence length: memory and time
    ds = build_dataset(tok, FULL_SEQ)
    res["full_seq"] = FULL_SEQ
    try:
        loss, g, secs, peak = grad_pass(model, cache, ds[0], False)
        res["full_off"] = {"loss": loss, "secs": secs, "peak_gib": peak, "oom": False}
        print(f"GRADCKPT,full_off,loss={loss:.6f},secs={secs:.2f},peak={peak:.2f}")
    except torch.OutOfMemoryError as e:
        res["full_off"] = {"oom": True, "error": str(e)[:200]}
        print("GRADCKPT,full_off,oom=1")
    for p in cache.parameters():
        p.grad = None
    torch.cuda.empty_cache()
    times, peaks = [], []
    for i in range(N_TIME):
        loss, g, secs, peak = grad_pass(model, cache, ds[i % len(ds)], True)
        times.append(secs)
        peaks.append(peak)
        print(f"GRADCKPT,full_on_{i},loss={loss:.6f},secs={secs:.2f},peak={peak:.2f}")
    mean_secs = sum(times[1:]) / max(1, len(times) - 1) if len(times) > 1 else times[0]
    res["full_on"] = {
        "secs_each": times,
        "peak_gib": max(peaks),
        "mean_secs_after_first": mean_secs,
        "step_secs_at_global_bs": mean_secs * GLOBAL_BS,
        "global_bs": GLOBAL_BS,
    }
    print(
        f"GRADCKPT,full_on,peak={max(peaks):.2f},secs={mean_secs:.2f},"
        f"step_secs={mean_secs * GLOBAL_BS:.0f}"
    )
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1)
    print(f"GRADCKPT_DONE {OUT_JSON}")


if __name__ == "__main__":
    main()
