#!/usr/bin/env python3
"""If the untrained init beats training, which init should we build?

Four documents now agree that a cartridge's document specificity is
highest before any training and never recovers, so the best cartridge
this pipeline produces is an initialization.  That makes the choice of
initialization the lever, and it has never been compared under this
instrument.

The one in use takes the record's first KV_TOKENS tokens, which covers
about 4% of a 12k-token record, contiguously, from the beginning.  The
Cartridges paper's sampled-chunk initialization spends the same budget
spread across the whole document instead.  If coverage of the document
matters more than contiguity, that difference should be visible
immediately, with no training at all.

Methods, all spending exactly KV_TOKENS tokens:

  first_k         the incumbent: one contiguous block from the start
  uniform_chunks  evenly spaced chunks across the whole record, so
                  coverage is deterministic and reproducible
  random_chunks   the paper's sampling, seeded; two seeds say whether
                  chunk placement matters or only chunk spread does
  last_k          a control: same contiguity, opposite end.  If first_k
                  wins only because records open with a summary, this
                  catches it

Nothing here trains.  Each cartridge is one forward pass, so the whole
comparison is minutes, and it is scored with the same forced-choice
instrument as everything else in this lane.

Env: MODEL, RECORDS_DIR, PATIENTS (comma list), KV_TOKENS, OUT_DIR,
     METHODS (comma list), CHUNK (tokens per chunk, default 32).
"""

import os
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", "/data/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/initcmp")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.initialization.tokenization_utils import (
    MODEL_TO_SYSTEM_PROMPT_TOKENIZER,
)
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
RECORDS_DIR = os.environ["RECORDS_DIR"]
PATIENTS = [p.strip() for p in os.environ["PATIENTS"].split(",") if p.strip()]
KV_TOKENS = int(os.environ.get("KV_TOKENS", "512"))
OUT_DIR = os.environ["OUT_DIR"]
METHODS = [
    m.strip()
    for m in os.environ.get(
        "METHODS", "first_k,uniform_chunks,random_chunks_s0,random_chunks_s1,last_k"
    ).split(",")
    if m.strip()
]
CHUNK = int(os.environ.get("CHUNK", "32"))
DEVICE = "cuda"

tok = AutoTokenizer.from_pretrained(MODEL)
model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
model.eval()
model.requires_grad_(False)
ac = AttnConfig(
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_key_value_heads,
    head_dim=model.config.head_dim,
)
TOK_FN = MODEL_TO_SYSTEM_PROMPT_TOKENIZER[tok.name_or_path.lower()]


def select(method, full, k):
    """Choose k token positions from the record under one policy.

    Chunks are kept contiguous because a KV cache built from shredded
    single tokens would test something else entirely -- local context
    is what makes a key worth caching.
    """
    n = full.shape[0]
    if method == "first_k":
        return full[:k]
    if method == "last_k":
        return full[-k:]
    n_chunks = max(1, k // CHUNK)
    size = k // n_chunks
    if method == "uniform_chunks":
        span = max(1, (n - size) // max(1, n_chunks - 1)) if n_chunks > 1 else 0
        starts = [min(i * span, n - size) for i in range(n_chunks)]
    elif method.startswith("random_chunks"):
        seed = (
            int(method.rsplit("s", 1)[-1]) if method.rsplit("s", 1)[-1].isdigit() else 0
        )
        g = torch.Generator().manual_seed(seed)
        starts = sorted(
            torch.randint(0, max(1, n - size), (n_chunks,), generator=g).tolist()
        )
    else:
        raise ValueError(f"unknown method {method}")
    parts = [full[s : s + size] for s in starts]
    out = torch.cat(parts)
    # integer division can leave the budget a few tokens short; top up
    # from the front so every method spends exactly the same budget
    if out.shape[0] < k:
        out = torch.cat([out, full[: k - out.shape[0]]])
    return out[:k]


@torch.no_grad()
def build(ids):
    ids = ids.to(DEVICE)
    cache = TrainableCache(config=ac)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(
            input_ids=ids,
            seq_ids=torch.zeros_like(ids, dtype=torch.long),
            position_ids=torch.arange(ids.shape[-1], dtype=torch.long, device=DEVICE),
            use_cache=True,
            past_key_values=cache,
            mode="generate",
        )
    return cache


def save(cache, path):
    """Byte-for-byte the layout build_init_cart writes, so these
    cartridges load through the same path as every other one here and
    the first token becomes the frozen attention sink."""
    keys = [t.detach().to(torch.bfloat16).cpu() for t in cache._keys]
    values = [t.detach().to(torch.bfloat16).cpu() for t in cache._values]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trainable_keys": [k[:, :, 1:].contiguous() for k in keys],
            "trainable_values": [v[:, :, 1:].contiguous() for v in values],
            "frozen_keys": [k[:, :, :1].contiguous() for k in keys],
            "frozen_values": [v[:, :, :1].contiguous() for v in values],
        },
        path,
    )


def main():
    for patient in PATIENTS:
        record = (Path(RECORDS_DIR) / f"{patient}.txt").read_text()
        full = TOK_FN(tokenizer=tok, content=record, max_tokens=10**7).squeeze(0)
        print(f"[init] {patient}: record is {full.shape[0]} tokens", flush=True)
        for m in METHODS:
            ids = select(m, full, KV_TOKENS)
            assert ids.shape[0] == KV_TOKENS, f"{m} gave {ids.shape[0]} tokens"
            path = f"{OUT_DIR}/{patient}/{m}.pt"
            save(build(ids), path)
            print(f"[init]   {m}: {path}", flush=True)
            torch.cuda.empty_cache()
    print("INIT_COMPARE_DONE")


if __name__ == "__main__":
    main()
