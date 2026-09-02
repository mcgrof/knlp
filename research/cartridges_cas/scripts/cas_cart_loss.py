#!/usr/bin/env python3
"""Score many cartridges on one fixed slice of self-study conversations
with the cartridge trainer's own distillation loss.

The 20-question accuracy read that the CAS (Cartridges at Scale) line
reports has a dynamic range of about five questions and is mostly
deterministic per question, so it cannot resolve the effects the
meta-initialization comparisons look for. The dense signal that can is
the objective itself: the top-20 distillation cross-entropy on
conversations the cartridge never trained on. This script computes it
for any number of cartridge files with the model loaded once, using the
same dataset packing and the same per-entry formula as
``cartridges.train.evaluate_perplexity`` (mean over stored top-k
entries of -p * log q), so a number here is directly comparable to the
"Eval loss" lines the trainer prints during a run.

Two refinements the trainer does not offer:

* ``NOCART=1`` also scores the frozen model with no cartridge at all.
  That pass marks each target position as *document-informative* when
  the no-cartridge model's argmax disagrees with the teacher's argmax,
  and every cartridge is then reported on all entries, on the
  informative subset, and on the rest. A cartridge that only learned the
  self-study format moves the rest; one that learned the document moves
  the informative subset.
* Per-position losses are written to a ``.npz`` next to the JSON so two
  cartridges can be compared paired by position rather than by their
  means alone.

Env:
    MODEL      Qwen/Qwen3-8B (default)
    PARQUET    conversations to score (a val split from cas_split_val.py)
    LIMIT      rows to use (default 256; the trainer's VAL_LIMIT)
    SEED       packing seed (default 42, the trainer's)
    CARTS      comma list of ``name=path`` (or bare paths)
    NOCART     1 to add the no-cartridge pass and the informative mask
    OUT_JSON   results (default cart_loss.json); ``.npz`` sibling written
    DEVICE     cuda (default)

Prints one ``CART_LOSS,<name>,<loss>,<loss_info>,<loss_rest>,<entries>``
line per cartridge and ``CART_LOSS_DONE`` at the end.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/cas_cart_loss")
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

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
PARQUET = os.environ["PARQUET"]
LIMIT = int(os.environ.get("LIMIT", "256"))
SEED = int(os.environ.get("SEED", "42"))
CARTS = [c for c in os.environ.get("CARTS", "").split(",") if c]
NOCART = os.environ.get("NOCART", "0") == "1"
OUT_JSON = os.environ.get("OUT_JSON", "cart_loss.json")
DEVICE = os.environ.get("DEVICE", "cuda")
SEQ_LEN = 2048


def parse_carts(items):
    out = []
    for it in items:
        if "=" in it:
            name, path = it.split("=", 1)
        else:
            name, path = Path(it).stem, it
        out.append((name, path))
    return out


def build_dataset(tokenizer):
    cfg = TrainDataset.Config(
        data_sources=[DataSource(path=PARQUET, type="local", limit=LIMIT)],
        top_k_logits=20,
        packed_seq_length=SEQ_LEN,
        packing_mode="truncate",
    )
    return cfg.instantiate(tokenizer=tokenizer, seed=SEED)


def teacher_argmax(batch, n_pos):
    """Token id of the teacher's most likely token at each predicting
    position (-1 where the position has no stored targets)."""
    pos = batch.topk_token_idxs - 1
    lp = batch.topk_logprobs
    best = torch.full((n_pos,), -float("inf"))
    best = best.scatter_reduce(0, pos, lp, reduce="amax", include_self=True)
    is_best = lp == best[pos]
    out = torch.full((n_pos,), -1, dtype=torch.long)
    out[pos[is_best]] = batch.topk_token_ids[is_best]
    return out


@torch.no_grad()
def score(model, dataset, cache):
    """Return per-position (ce_sum, n_entries, model_argmax) arrays over the
    packed dataset, in the trainer's entry units."""
    n_batches = len(dataset)
    ce_sum = torch.zeros(n_batches * SEQ_LEN, dtype=torch.float64)
    n_ent = torch.zeros(n_batches * SEQ_LEN, dtype=torch.long)
    argmax = torch.full((n_batches * SEQ_LEN,), -1, dtype=torch.long)
    for b in range(n_batches):
        batch = dataset[b]
        if cache is not None:
            cache.clear()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=batch.input_ids.to(DEVICE),
                seq_ids=batch.element_ids.to(DEVICE),
                position_ids=batch.position_ids.to(DEVICE),
                use_cache=cache is not None,
                past_key_values=cache,
            )
            logits = out.logits[0]
            idx = batch.topk_token_idxs.to(DEVICE) - 1
            ids = batch.topk_token_ids.to(DEVICE)
            # the trainer's formula, unchanged: -p(x) log q(x) per entry
            pred_lp = F.log_softmax(logits, dim=-1)[idx, ids]
            ce = -batch.topk_logprobs.to(DEVICE).exp() * pred_lp
        base = b * SEQ_LEN
        ce_sum.index_add_(0, base + idx.cpu(), ce.double().cpu())
        n_ent.index_add_(0, base + idx.cpu(), torch.ones_like(idx.cpu()))
        argmax[base : base + SEQ_LEN] = logits.argmax(dim=-1).cpu()
        del out, logits
    return ce_sum.numpy(), n_ent.numpy(), argmax.numpy()


def summarize(ce_sum, n_ent, info_mask):
    has = n_ent > 0
    tot = float(ce_sum[has].sum() / n_ent[has].sum())
    res = {"loss": tot, "n_entries": int(n_ent.sum()), "n_positions": int(has.sum())}
    if info_mask is not None:
        i = has & info_mask
        r = has & ~info_mask
        res["loss_info"] = float(ce_sum[i].sum() / max(1, n_ent[i].sum()))
        res["loss_rest"] = float(ce_sum[r].sum() / max(1, n_ent[r].sum()))
        res["n_info_entries"] = int(n_ent[i].sum())
        res["n_info_positions"] = int(i.sum())
    return res


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    dataset = build_dataset(tok)
    print(
        f"[cart-loss] {PARQUET}: {len(dataset.elements)} rows -> "
        f"{len(dataset)} packed sequences of {SEQ_LEN}",
        flush=True,
    )
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
    model.eval()
    attn = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
    )
    print(f"[cart-loss] model up in {time.time() - t0:.0f}s", flush=True)

    # teacher argmax per position, from the stored targets alone
    n_pos = len(dataset) * SEQ_LEN
    t_arg = np.full(n_pos, -1, dtype=np.int64)
    for b in range(len(dataset)):
        t_arg[b * SEQ_LEN : (b + 1) * SEQ_LEN] = teacher_argmax(
            dataset[b], SEQ_LEN
        ).numpy()

    results = {
        "model": MODEL,
        "parquet": os.path.abspath(PARQUET),
        "limit": LIMIT,
        "seed": SEED,
        "n_rows": len(dataset.elements),
        "n_packed": len(dataset),
        "carts": {},
    }
    arrays = {"teacher_argmax": t_arg}
    info_mask = None
    if NOCART:
        t1 = time.time()
        ce, n, am = score(model, dataset, None)
        info_mask = (n > 0) & (am != t_arg)
        arrays.update(nocart_ce=ce, nocart_n=n, nocart_argmax=am, info_mask=info_mask)
        results["nocart"] = summarize(ce, n, info_mask)
        results["nocart"]["seconds"] = time.time() - t1
        r = results["nocart"]
        print(
            f"CART_LOSS,nocart,{r['loss']:.6f},{r['loss_info']:.6f},"
            f"{r['loss_rest']:.6f},{r['n_entries']}  "
            f"(informative positions {r['n_info_positions']}/{r['n_positions']})",
            flush=True,
        )

    for name, path in parse_carts(CARTS):
        t1 = time.time()
        keys, values, nfrozen = load_cart(path, device=DEVICE)
        cache = TrainableCache(
            config=attn, init_keys=keys, init_values=values, num_frozen_tokens=nfrozen
        ).to(DEVICE)
        ce, n, am = score(model, dataset, cache)
        r = summarize(ce, n, info_mask)
        r.update(path=os.path.abspath(path), trainable=keys[0].shape[2] - nfrozen)
        r["frozen"] = nfrozen
        r["seconds"] = time.time() - t1
        results["carts"][name] = r
        arrays[f"{name}__ce"] = ce
        arrays[f"{name}__n"] = n
        li = f"{r['loss_info']:.6f}" if "loss_info" in r else "nan"
        lr = f"{r['loss_rest']:.6f}" if "loss_rest" in r else "nan"
        print(
            f"CART_LOSS,{name},{r['loss']:.6f},{li},{lr},{r['n_entries']}",
            flush=True,
        )
        del cache, keys, values
        torch.cuda.empty_cache()

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=1)
    np.savez_compressed(str(Path(OUT_JSON).with_suffix(".npz")), **arrays)
    print(f"CART_LOSS_DONE {OUT_JSON} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
