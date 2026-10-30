#!/usr/bin/env python3
"""Check and time several packed sequences per forward pass for the
cartridge trainer.

Two questions, in order. First, does a pass over m merged packed
sequences (``cas_microbatch.py``) give the same loss and cartridge
gradient as m separate passes, the library's arithmetic? For the same
P packed sequences the script accumulates the gradient both ways and
reports the loss difference, the cosine between the two gradients and
their relative norm difference, against the repeat spread of the
one-sequence path itself (the same passes run twice). Second, how fast
is each m on this GPU? For each m it runs warmup passes and then three
timed blocks of passes, each pass a forward and backward exactly as in
training, and reports processed tokens per second, seconds per
optimizer-step equivalent (128 sequences), peak memory and the time of
one Adam step on the cartridge. It stops at the first m that runs out
of memory. As a yardstick for the parity numbers, the one-sequence path
is also run with activation recomputation flipped: a different but
equally valid computation, so a merged-pass difference no larger than
that one is numerical, not a change of objective.

Env:
    MODEL       reader (default Qwen/Qwen3-8B)
    PARQUET     Self-Study parquet
    CART        cartridge file (frozen sink + trainable layout)
    MICROS      comma list of m (default 1,2,4,8)
    PARITY_P    sequences in the parity check (default 8; 0 skips it)
    WARMUP      warmup passes per m (default 3)
    BLOCK       timed passes per block (default 8); three blocks
    SEED        packing seed (default 42)
    OUT_JSON    results (default microbatch_bench.json)

Set CARTRIDGES_GRAD_CKPT and CARTRIDGES_COMPILE_FLEX as for training;
the JSON records both. Prints ``MB_PARITY,<m>,...`` and
``MB_SPEED,<m>,...`` lines and ``MB_DONE``.
"""

import json
import os
import sys
import time

os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/cas_microbatch_bench")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.datasets import TrainDataset, DataSource
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cas_cart_init import load_cart  # noqa: E402
from cas_microbatch import entry_weights, merge_batches  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
PARQUET = os.environ["PARQUET"]
CART = os.environ["CART"]
MICROS = [int(x) for x in os.environ.get("MICROS", "1,2,4,8").split(",")]
PARITY_P = int(os.environ.get("PARITY_P", "8"))
WARMUP = int(os.environ.get("WARMUP", "3"))
BLOCK = int(os.environ.get("BLOCK", "8"))
SEED = int(os.environ.get("SEED", "42"))
OUT_JSON = os.environ.get("OUT_JSON", "microbatch_bench.json")
DEV = "cuda"
GLOBAL = 128


def pass_loss(model, cache, batch, counts, m):
    """The library's training forward for one pass, with the loss
    weighted so a merged pass equals the sum of per-sequence means / m
    (one unmerged pass: its mean)."""
    cache.clear()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=batch.input_ids.to(DEV),
            seq_ids=batch.element_ids.to(DEV),
            position_ids=batch.position_ids.to(DEV),
            use_cache=True,
            past_key_values=cache,
        )
        pred = F.log_softmax(out.logits, dim=-1)[
            0,
            batch.topk_token_idxs.to(DEV) - 1,
            batch.topk_token_ids.to(DEV),
        ]
        ce = -batch.topk_logprobs.to(DEV).exp() * pred
        if counts is None:
            loss = ce.mean()
        else:
            loss = (ce * entry_weights(counts, m, ce.device, ce.dtype)).sum()
    return loss


def grads(cache):
    ps = list(cache.trainable_keys) + list(cache.trainable_values)
    return torch.cat([p.grad.float().reshape(-1) for p in ps])


def zero(cache):
    for p in list(cache.trainable_keys) + list(cache.trainable_values):
        p.grad = None


def accumulate(model, cache, dataset, idxs, m):
    """Gradient of sum_s mean_s / len(idxs) over the packed sequences
    idxs, in passes of m merged sequences."""
    zero(cache)
    total = 0.0
    P = len(idxs)
    for g in range(0, P, m):
        subs = [dataset[i] for i in idxs[g : g + m]]
        if m == 1:
            loss = pass_loss(model, cache, subs[0], None, 1) / P
        else:
            b = merge_batches(subs)
            loss = pass_loss(model, cache, b, b._cr_sub_entries, m) * m / P
        loss.backward()
        total += float(loss)
    return total, grads(cache)


def main():
    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    dataset = TrainDataset.Config(
        data_sources=[DataSource(path=PARQUET, type="local")],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ).instantiate(tokenizer=tok, seed=SEED)
    n = len(dataset)
    t0 = time.time()
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    model = model.to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    load_s = time.time() - t0
    attn = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
    )
    keys, values, nfrozen = load_cart(CART, device=DEV)
    cache = TrainableCache(
        config=attn, init_keys=keys, init_values=values, num_frozen_tokens=nfrozen
    ).to(DEV)
    rng = np.random.default_rng(1234)
    res = dict(
        model=MODEL,
        parquet=os.path.abspath(PARQUET),
        cart=os.path.abspath(CART),
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        grad_ckpt=os.environ.get("CARTRIDGES_GRAD_CKPT", "0"),
        compile_flex=os.environ.get("CARTRIDGES_COMPILE_FLEX", "unset"),
        n_packed=n,
        model_load_s=load_s,
        parity={},
        speed={},
    )

    def save():
        with open(OUT_JSON, "w") as f:
            json.dump(res, f, indent=1)

    if PARITY_P:
        idxs = [int(x) for x in rng.choice(n, PARITY_P, replace=False)]
        l_ref, g_ref = accumulate(model, cache, dataset, idxs, 1)
        l_rep, g_rep = accumulate(model, cache, dataset, idxs, 1)
        floor = dict(
            loss_diff=abs(l_rep - l_ref),
            cos=float(F.cosine_similarity(g_ref, g_rep, dim=0)),
            rel_norm=float((g_rep - g_ref).norm() / g_ref.norm()),
        )
        res["parity"]["repeat_m1"] = floor
        print(f"MB_PARITY,1rep,{floor}", flush=True)
        # yardstick: the one-sequence path with activation recomputation
        # flipped, a different but equally valid computation of the same
        # gradient (the patched decoder reads the variable per forward)
        ck = os.environ.get("CARTRIDGES_GRAD_CKPT", "0")
        os.environ["CARTRIDGES_GRAD_CKPT"] = "0" if ck == "1" else "1"
        try:
            l_y, g_y = accumulate(model, cache, dataset, idxs, 1)
            yard = dict(
                loss_diff=abs(l_y - l_ref),
                cos=float(F.cosine_similarity(g_ref, g_y, dim=0)),
                rel_norm=float((g_y - g_ref).norm() / g_ref.norm()),
            )
        except torch.OutOfMemoryError:
            yard = "oom"
            torch.cuda.empty_cache()
        os.environ["CARTRIDGES_GRAD_CKPT"] = ck
        res["parity"]["yardstick_ckpt_flip_m1"] = yard
        print(f"MB_PARITY,1ckptflip,{yard}", flush=True)
        for m in MICROS:
            if m == 1 or PARITY_P % m:
                continue
            try:
                l_m, g_m = accumulate(model, cache, dataset, idxs, m)
            except torch.OutOfMemoryError:
                res["parity"][str(m)] = "oom"
                torch.cuda.empty_cache()
                break
            r = dict(
                loss_ref=l_ref,
                loss=l_m,
                loss_diff=abs(l_m - l_ref),
                cos=float(F.cosine_similarity(g_ref, g_m, dim=0)),
                rel_norm=float((g_m - g_ref).norm() / g_ref.norm()),
            )
            res["parity"][str(m)] = r
            print(f"MB_PARITY,{m},{r}", flush=True)
            save()
        zero(cache)

    opt = torch.optim.Adam(cache.parameters(), lr=0.1)
    for m in MICROS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            times = []
            k = 0
            for blk in range(-1, 3):
                count = WARMUP if blk < 0 else BLOCK
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(count):
                    idx = [int(x) for x in rng.choice(n, m, replace=False)]
                    subs = [dataset[i] for i in idx]
                    if m == 1:
                        loss = pass_loss(model, cache, subs[0], None, 1)
                    else:
                        b = merge_batches(subs)
                        loss = pass_loss(model, cache, b, b._cr_sub_entries, m)
                    loss.backward()
                    k += 1
                torch.cuda.synchronize()
                if blk >= 0:
                    times.append((time.time() - t0) / count)
            torch.cuda.synchronize()
            t0 = time.time()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            opt_s = time.time() - t0
        except torch.OutOfMemoryError:
            res["speed"][str(m)] = "oom"
            print(f"MB_SPEED,{m},oom", flush=True)
            save()
            break
        per_pass = float(np.mean(times))
        r = dict(
            sec_per_pass_blocks=times,
            sec_per_pass=per_pass,
            tokens_per_s=m * 2048 / per_pass,
            sec_per_128_seq=per_pass * GLOBAL / m,
            peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            adam_step_s=opt_s,
        )
        res["speed"][str(m)] = r
        print(
            f"MB_SPEED,{m},{r['tokens_per_s']:.0f} tok/s,"
            f"{r['sec_per_128_seq']:.1f} s per 128 seq,"
            f"peak {r['peak_gib']:.1f} GiB,blocks {[round(x, 3) for x in times]}",
            flush=True,
        )
        save()
    res["seconds"] = time.time() - t_start
    save()
    print(f"MB_DONE {OUT_JSON} ({res['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
