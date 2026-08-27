#!/usr/bin/env python3
"""Optimizer ablation for cartridge training: AdamW vs SOAP, matched arms.

Trains ONE isolated cartridge from a shared truncation init with the
stored sparse top-k distillation objective (the recipe that produced the
best local cartridges), swapping ONLY the optimizer between arms.
Everything else is matched: the starting KV state (built once, saved,
and loaded by every subsequent arm), the data order (same seeded
sampler), steps, gradient accumulation, learning rate, and gradient
clipping.

Why this comparison is interesting: cartridge training backpropagates
through a large frozen model to update a small trainable KV prefix, so
the optimizer step is a small fraction of total step cost.  That is the
regime where a second-order method's per-step progress is nearly free.
The deliverable is a loss curve and an honest wall-clock split per arm
(CUDA-synchronized around the optimizer step), not a quality claim; the
strict letter evaluator (opcart_reeval.py) scores the saved checkpoints
separately.

The objective is intentionally the stored recipe byte-for-byte,
including the known [sampled + top-k] duplicate-row layout of the
synthesis parquet: the ablation varies the optimizer, never the loss.

Env:
  MODEL              HF id (default Qwen/Qwen3-0.6B)
  OPTIMIZER          adamw | soap
  PATIENT            LongHealth patient id (e.g. patient_02)
  DATA_PARQUET       stored self-study parquet (teacher top-k targets)
  RECORDS_DIR        dir holding <patient>.txt (truncation-init source)
  KV_TOKENS          cartridge size in KV tokens (default 512)
  INIT_CART          shared starting checkpoint; built+saved if missing
  STEPS, ACCUM, LR, SEED, CHECKPOINT_AT, OUT_DIR
  SOAP_PRECOND_FREQ  SOAP eigenbasis refresh period (default 10)
  KNLP_ROOT          knlp checkout root (locates the vendored SOAP)
"""

import importlib.util
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridges")
OUT_DIR = os.environ.get("OUT_DIR", "/root/cas_out/opt_ablation")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", OUT_DIR)
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.datasets import TrainDataset, DataSource
from cartridges.initialization.tokenization_utils import (
    MODEL_TO_SYSTEM_PROMPT_TOKENIZER,
)
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
OPTIMIZER = os.environ["OPTIMIZER"]
PATIENT = os.environ["PATIENT"]
DATA_PARQUET = os.environ["DATA_PARQUET"]
RECORDS_DIR = os.environ.get("RECORDS_DIR", "/root/cart_records")
KV_TOKENS = int(os.environ.get("KV_TOKENS", "512"))
INIT_CART = os.environ["INIT_CART"]
STEPS = int(os.environ.get("STEPS", "300"))
ACCUM = int(os.environ.get("ACCUM", "8"))
LR = float(os.environ.get("LR", "2e-2"))
SEED = int(os.environ.get("SEED", "0"))
CHECKPOINT_AT = [int(x) for x in os.environ.get("CHECKPOINT_AT", "").split(",") if x]
SOAP_PRECOND_FREQ = int(os.environ.get("SOAP_PRECOND_FREQ", "10"))
KNLP_ROOT = os.environ.get("KNLP_ROOT", "")
DEVICE = "cuda"

assert OPTIMIZER in ("adamw", "soap"), OPTIMIZER

torch.manual_seed(SEED)
rng = random.Random(SEED)


def load_soap_class():
    """Load the vendored reference SOAP from the knlp tree by path, so
    the script keeps working after bootstrap copies it out of the repo."""
    assert KNLP_ROOT, "OPTIMIZER=soap needs KNLP_ROOT pointing at a knlp checkout"
    soap_py = Path(KNLP_ROOT) / "fim" / "fisher_pruning" / "soap.py"
    assert soap_py.is_file(), f"vendored SOAP not found at {soap_py}"
    spec = importlib.util.spec_from_file_location("knlp_vendored_soap", soap_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SOAP


# ---------------------------------------------------------------------------
# model and shared cartridge init
# ---------------------------------------------------------------------------

print(f"[cart-opt] optimizer={OPTIMIZER} model={MODEL} patient={PATIENT}")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
model.eval()
model.requires_grad_(False)
attn_config = AttnConfig(
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_key_value_heads,
    head_dim=model.config.head_dim,
)


@torch.no_grad()
def build_init_cart(path):
    """Truncation init (KVFromText path): forward the first KV_TOKENS
    system-prompt tokens of the record through the frozen model and keep
    the resulting KV.  The first token is saved as a frozen attention
    sink (the isolated-cartridge geometry the strict evaluator expects);
    the rest trains.  Saved once so every arm starts from identical KV."""
    record = (Path(RECORDS_DIR) / f"{PATIENT}.txt").read_text()
    tok_fn = MODEL_TO_SYSTEM_PROMPT_TOKENIZER[tok.name_or_path.lower()]
    ids = tok_fn(tokenizer=tok, content=record, max_tokens=KV_TOKENS).squeeze(0)
    assert ids.shape[0] >= KV_TOKENS, (
        f"{PATIENT}: record gives only {ids.shape[0]} tokens "
        f"(< KV_TOKENS={KV_TOKENS}); lower KV_TOKENS"
    )
    ids = ids[:KV_TOKENS].to(DEVICE)
    tmp = TrainableCache(config=attn_config)
    seq_ids = torch.zeros_like(ids, dtype=torch.long)
    pos = torch.arange(ids.shape[-1], dtype=torch.long, device=DEVICE)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(
            input_ids=ids,
            seq_ids=seq_ids,
            position_ids=pos,
            use_cache=True,
            past_key_values=tmp,
            mode="generate",
        )
    keys = [t.detach().to(torch.bfloat16).cpu() for t in tmp._keys]
    values = [t.detach().to(torch.bfloat16).cpu() for t in tmp._values]
    torch.save(
        {
            "trainable_keys": [k[:, :, 1:].contiguous() for k in keys],
            "trainable_values": [v[:, :, 1:].contiguous() for v in values],
            "frozen_keys": [k[:, :, :1].contiguous() for k in keys],
            "frozen_values": [v[:, :, :1].contiguous() for v in values],
        },
        path,
    )
    print(f"[cart-opt] built shared init cart: {path}")


def load_cart_as_trainable(path):
    """Rebuild a library-format checkpoint as a TrainableCache preserving
    the frozen/trainable split (same loader contract as opcart_train)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)

    def t(p):
        return torch.as_tensor(p.data if hasattr(p, "data") else p).to(torch.bfloat16)

    tk = [t(p) for p in ck["trainable_keys"]]
    tv = [t(p) for p in ck["trainable_values"]]
    fk = ck.get("frozen_keys") or []
    fv = ck.get("frozen_values") or []
    if fk:
        ik = [torch.cat([t(fk[i]), tk[i]], dim=2) for i in range(len(tk))]
        iv = [torch.cat([t(fv[i]), tv[i]], dim=2) for i in range(len(tv))]
        nfrozen = t(fk[0]).shape[2]
    else:
        ik, iv, nfrozen = tk, tv, 0
    cache = TrainableCache(
        config=attn_config, init_keys=ik, init_values=iv, num_frozen_tokens=nfrozen
    ).to(DEVICE)
    return cache, nfrozen


if not Path(INIT_CART).is_file():
    Path(INIT_CART).parent.mkdir(parents=True, exist_ok=True)
    build_init_cart(INIT_CART)
cache, NUM_FROZEN = load_cart_as_trainable(INIT_CART)

for p in model.parameters():
    assert not p.requires_grad, "frozen base model has requires_grad param"
assert any(p.requires_grad for p in cache.parameters()), "cartridge not trainable"

dataset = TrainDataset.Config(
    data_sources=[DataSource(path=DATA_PARQUET, type="local")],
    top_k_logits=20,
    packed_seq_length=2048,
    packing_mode="truncate",
).instantiate(tokenizer=tok, seed=SEED)
print(f"[cart-opt] dataset elements: {len(dataset.elements)}")


# ---------------------------------------------------------------------------
# the stored objective (held fixed across arms)
# ---------------------------------------------------------------------------


def stored_topk_loss(el):
    """The stored recipe's sparse top-k distillation CE, byte-identical
    to the opcart_train baseline arm."""
    ids = el.input_ids.to(DEVICE)
    idxs = el.topk_token_idxs.to(DEVICE)
    tids = el.topk_token_ids.to(DEVICE)
    tlp = el.topk_logprobs.to(DEVICE)
    seq_ids = torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE)
    pos = torch.arange(ids.shape[0], device=DEVICE)
    cache.clear()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=ids,
            seq_ids=seq_ids,
            position_ids=pos,
            use_cache=True,
            past_key_values=cache,
        )
        vocab = out.logits.shape[-1]
        gi = idxs - 1
        valid = (gi >= 0) & (gi < ids.shape[0]) & (tids >= 0) & (tids < vocab)
        if int(valid.sum()) == 0:
            return None
        lp = F.log_softmax(out.logits.float(), dim=-1)[0, gi[valid], tids[valid]]
        return (-tlp[valid].exp() * lp).mean()


def build_optimizer():
    if OPTIMIZER == "adamw":
        return torch.optim.AdamW(
            cache.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0
        )
    soap_cls = load_soap_class()
    return soap_cls(
        cache.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        precondition_frequency=SOAP_PRECOND_FREQ,
    )


def save_cart(path):
    """Save in library format, preserving the frozen sink so the strict
    evaluator reconstructs the same geometry."""
    ck = torch.load(INIT_CART, map_location="cpu", weights_only=False)
    torch.save(
        {
            "trainable_keys": [
                p.detach().contiguous().cpu() for p in cache.trainable_keys
            ],
            "trainable_values": [
                p.detach().contiguous().cpu() for p in cache.trainable_values
            ],
            "frozen_keys": ck.get("frozen_keys"),
            "frozen_values": ck.get("frozen_values"),
        },
        path,
    )


# ---------------------------------------------------------------------------
# matched training loop with an honest wall-clock split
# ---------------------------------------------------------------------------


def main():
    out_dir = Path(OUT_DIR) / OPTIMIZER
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = build_optimizer()
    n_trainable = sum(p.numel() for p in cache.parameters() if p.requires_grad)
    run = dict(
        optimizer=OPTIMIZER,
        model=MODEL,
        patient=PATIENT,
        parquet=DATA_PARQUET,
        init_cart=INIT_CART,
        kv_tokens=KV_TOKENS,
        num_frozen=NUM_FROZEN,
        trainable_params=n_trainable,
        steps=STEPS,
        accum=ACCUM,
        lr=LR,
        seed=SEED,
        soap_precond_freq=SOAP_PRECOND_FREQ if OPTIMIZER == "soap" else None,
        history=[],
        cost=dict(fwdbwd_s=0.0, update_s=0.0, train_tokens=0),
    )
    print(f"[cart-opt] trainable cartridge params: {n_trainable}")

    t_start = time.time()
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        accum_loss = 0.0
        n_used = 0
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(ACCUM):
            el = dataset.elements[rng.randrange(len(dataset.elements))]
            loss = stored_topk_loss(el)
            if loss is None:
                continue
            run["cost"]["train_tokens"] += int(el.input_ids.shape[0])
            (loss / ACCUM).backward()
            accum_loss += float(loss.detach()) / ACCUM
            n_used += 1
        torch.cuda.synchronize()
        t1 = time.time()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(cache.parameters(), 1.0))
        opt.step()
        torch.cuda.synchronize()
        t2 = time.time()
        run["cost"]["fwdbwd_s"] += t1 - t0
        run["cost"]["update_s"] += t2 - t1
        run["history"].append(
            dict(
                step=step,
                loss=accum_loss,
                used=n_used,
                grad_norm=grad_norm,
                fwdbwd_s=round(t1 - t0, 4),
                update_s=round(t2 - t1, 4),
            )
        )
        if step % 10 == 0 or step == 1:
            print(
                f"[cart-opt] {OPTIMIZER} step {step:4d} loss={accum_loss:.4f} "
                f"update={t2 - t1:.3f}s wall={time.time() - t_start:.0f}s",
                flush=True,
            )
        if step in CHECKPOINT_AT:
            save_cart(out_dir / f"{PATIENT}_step{step}.pt")

    run["cost"]["total_wall_s"] = time.time() - t_start
    run["cost"]["update_frac"] = run["cost"]["update_s"] / max(
        run["cost"]["total_wall_s"], 1e-9
    )
    run["cost"]["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    save_cart(out_dir / f"{PATIENT}.pt")
    with open(out_dir / "run.json", "w") as f:
        json.dump(run, f, indent=1)
    print(
        f"CART_OPT_DONE optimizer={OPTIMIZER} final_loss={accum_loss:.4f} "
        f"update_frac={run['cost']['update_frac']:.4f} "
        f"wall={run['cost']['total_wall_s']:.0f}s out={out_dir}"
    )


if __name__ == "__main__":
    main()
