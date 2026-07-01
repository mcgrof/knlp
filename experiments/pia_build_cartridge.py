# SPDX-License-Identifier: GPL-2.0
"""Build a synthetic prefix-cache cartridge for PIA codec/drift runs.

A knlp cartridge is torch.save(list[(k, v)]) with one (K, V) tuple per layer,
plus a sibling meta.json and prefix_token_ids.json. This builds one by
prefilling a long references/bibliography document through the model and saving
the resulting KV cache -- the same shape the codec and semantic-drift modes
expect. It is deterministic given the model and the document, so the codec
comparison is reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch


def references_doc(n_entries):
    """A plausible ~bibliography prefix: numbered citations the queries ask
    about. Deterministic, no randomness."""
    venues = ["NeurIPS", "ICML", "ICLR", "ACL", "EMNLP", "CVPR", "JMLR"]
    topics = [
        "attention mechanisms",
        "the transformer architecture",
        "layer normalization",
        "residual learning",
        "batch normalization",
        "adaptive optimization",
        "dropout regularization",
        "word embeddings",
        "sequence to sequence learning",
        "neural machine translation",
        "language model pretraining",
        "mixture of experts",
        "sparse attention",
        "rotary position embeddings",
        "state space models",
        "retrieval augmentation",
        "knowledge distillation",
        "quantized inference",
        "key-value cache compression",
        "long-context modeling",
    ]
    lines = ["References", ""]
    for i in range(n_entries):
        t = topics[i % len(topics)]
        v = venues[i % len(venues)]
        yr = 2015 + (i % 10)
        lines.append(
            f"[{i + 1}] A. Researcher, B. Coauthor, and C. Lastname. "
            f"A study of {t} for large scale models. "
            f"In Proceedings of {v} {yr}, pages {100 + i}-{120 + i}. "
            f"BibTeX key: ref{i + 1}_{t.split()[0]}{yr}."
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--target-tokens", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        .to(device)
        .eval()
    )

    text = references_doc(140)
    ids = tok(text, return_tensors="pt").input_ids[:, : args.target_tokens]
    ids = ids.to(device)
    prefix_len = ids.shape[1]

    with torch.no_grad():
        out = model(ids, use_cache=True)
    pkv = out.past_key_values
    # keep the leading batch dim: the drift module's DynamicCache.update wants
    # 4D [batch, kv_heads, seq, head_dim] tensors.
    if hasattr(pkv, "layers"):
        layers = [(l.keys.cpu(), l.values.cpu()) for l in pkv.layers]
    else:
        layers = [(k.cpu(), v.cpu()) for k, v in pkv]

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(layers, os.path.join(args.out_dir, "cartridge.pt"))

    tok_ids = ids[0].cpu().tolist()
    sha = hashlib.sha256(json.dumps(tok_ids).encode("utf-8")).hexdigest()
    meta = {
        "model": args.model,
        "dtype": "bfloat16",
        "n_layers": len(layers),
        "budget_tokens": prefix_len,
        "block_size": args.block_size,
        "prefix_token_ids_sha256": sha,
    }
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    json.dump(tok_ids, open(os.path.join(args.out_dir, "prefix_token_ids.json"), "w"))
    print(
        f"cartridge: {len(layers)} layers, prefix {prefix_len} tokens, "
        f"blocks {prefix_len // args.block_size}, sha {sha[:12]}",
        flush=True,
    )
    print("wrote", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
