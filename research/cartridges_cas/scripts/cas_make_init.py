#!/usr/bin/env python3
"""Build step-0 cartridges of the kinds the CAS (Cartridges at Scale)
paper ablates, as files that ``cas_train_isolated.py`` can start from
(``CART_INIT=``) and ``cas_cart_loss.py`` can score.

The paper's initialization ablation compares three starting points for a
cartridge of p tokens: the KV state of the first p tokens of the
document (its default, and ours), the KV state of p random tokens drawn
from the document, and random vectors. Every trained cartridge in this
line starts from the first kind, and the meta-initialization study needs
the other two as rulers: a learned correction that also lifts a
random-token start is document-agnostic, and one that only lifts the
first-p start is tied to the content it was fitted on.

All text-derived kinds use the same system-prompt layout as the
library's ``KVFromText`` (three header tokens, content, then the
``<|im_end|>\\n`` pair), so slot i means the same thing in every file,
and all carry the library's default one-token frozen sink at slot 0.

Env:
    MODEL        Qwen/Qwen3-8B (default)
    RECORD       path to the document text (records/<patient>.txt)
    KV_TOKENS    p, the total cartridge length (frozen sink included);
                 "auto" = ceil(doc_tokens / KV_DIVISOR) like the trainer
    KV_DIVISOR   20 (default)
    KIND         first | randdoc | randvocab | randvec
    SEED         0 (default; randdoc/randvocab/randvec draws)
    OUT          cartridge file to write
    DEVICE       cuda (default)

Prints ``MAKE_INIT_DONE <kind> <p> <out>``.
"""

import os
import random

os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/cas_make_init")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import sys

import torch
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.initialization.tokenization_utils import (
    MODEL_TO_SYSTEM_PROMPT_TOKENIZER,
)
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cas_cart_init import save_cart  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
RECORD = os.environ.get("RECORD", "")
KIND = os.environ.get("KIND", "first")
SEED = int(os.environ.get("SEED", "0"))
OUT = os.environ.get("OUT", "")  # required in main(); optional for imports
DEVICE = os.environ.get("DEVICE", "cuda")
NFROZEN = 1
# the library's Qwen system-prompt layout
HEADER = [151644, 8948, 198]  # <|im_start|>system\n
FOOTER = [151645, 198]  # <|im_end|>\n
FIRST_SPECIAL = 151643


def resolve_p(tok, doc_ids):
    raw = os.environ.get("KV_TOKENS", "auto")
    if raw != "auto":
        return int(raw)
    div = int(os.environ.get("KV_DIVISOR", "20"))
    lo = int(os.environ.get("KV_MIN", "256"))
    hi = int(os.environ.get("KV_MAX", "2048"))
    return max(lo, min(hi, -(-len(doc_ids) // div)))


def layout(content_ids, p):
    """The trainer's exact truncation: first p-2 tokens of the templated
    prompt, then the closing pair."""
    ids = HEADER + list(content_ids)
    if len(ids) + len(FOOTER) > p:
        ids = ids[: p - len(FOOTER)] + FOOTER
    else:
        ids = ids + FOOTER
    return ids


def make_ids(tok, kind, p, doc_ids, text):
    rng = random.Random(SEED)
    if kind == "first":
        # the library's own path, so this file is bit-comparable to the
        # init the trainer writes with SAVE_INIT=1
        fn = MODEL_TO_SYSTEM_PROMPT_TOKENIZER[tok.name_or_path.lower()]
        ids = fn(tokenizer=tok, content=text, max_tokens=p).squeeze(0).tolist()
        assert ids[: len(HEADER)] == HEADER and ids[-len(FOOTER) :] == FOOTER
        return ids
    if kind == "randdoc":
        n = p - len(HEADER) - len(FOOTER)
        pool = list(doc_ids)
        pick = rng.sample(pool, n) if len(pool) >= n else rng.choices(pool, k=n)
        return layout(pick, p)
    if kind == "randvocab":
        n = p - len(HEADER) - len(FOOTER)
        return layout([rng.randrange(FIRST_SPECIAL) for _ in range(n)], p)
    raise ValueError(kind)


@torch.no_grad()
def kv_from_ids(model, attn, ids):
    """Replicates KVFromText.initialize_kv_cache for explicit token ids."""
    init_cache = TrainableCache(config=attn)
    input_ids = torch.tensor(ids, dtype=torch.long, device=model.device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(
            input_ids=input_ids,
            seq_ids=torch.zeros_like(input_ids),
            position_ids=torch.arange(len(ids), dtype=torch.long, device=model.device),
            use_cache=True,
            past_key_values=init_cache,
            mode="generate",
        )
    keys = [k.detach().to(torch.bfloat16).cpu() for k in init_cache._keys]
    values = [v.detach().to(torch.bfloat16).cpu() for v in init_cache._values]
    return keys, values


def main():
    assert OUT and RECORD, "OUT= and RECORD= are required"
    tok = AutoTokenizer.from_pretrained(MODEL)
    with open(RECORD) as f:
        text = f.read()
    # the trainer counts document tokens with the bare tokenizer
    doc_ids = tok(text)["input_ids"]
    p = resolve_p(tok, doc_ids)
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
    model.eval()
    attn = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
    )
    if KIND == "randvec":
        g = torch.Generator().manual_seed(SEED)
        shape = (1, attn.n_heads, p, attn.head_dim)
        keys = [
            torch.randn(shape, generator=g).to(torch.bfloat16)
            for _ in range(attn.n_layers)
        ]
        values = [
            torch.randn(shape, generator=g).to(torch.bfloat16)
            for _ in range(attn.n_layers)
        ]
    else:
        ids = make_ids(tok, KIND, p, doc_ids, text)
        assert len(ids) == p, (len(ids), p)
        keys, values = kv_from_ids(model, attn, ids)
    assert keys[0].shape[2] == p
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    save_cart(keys, values, NFROZEN, OUT)
    print(
        f"MAKE_INIT_DONE {KIND} p={p} (trainable {p - NFROZEN}, frozen {NFROZEN}) "
        f"doc_tokens={len(doc_ids)} seed={SEED} -> {OUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
