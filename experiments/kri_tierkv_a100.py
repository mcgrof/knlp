# SPDX-License-Identifier: GPL-2.0
"""KRI-TierKV milestone-1 emulation experiment.

Runs a real model over a long context, computes per-block KRI-D-sum from the KV
cache and the oracle attention mass from the attention weights, then evaluates
each eviction/retrieval policy: how much attention mass its selection captures,
how many high-mass blocks it drops, and how many slow-tier bytes it moves per
token. The question this answers is whether KRI-D-sum picks the old blocks that
carry attention mass better than FIFO or recent-only, before any offload path is
built.

Emulation only: it does not modify attention. The attention-mass oracle needs the
full attention matrices, so it runs at a bounded context (default 2048); KRI-D-sum
scoring, policy geometry, and byte accounting hold at any length. On A100, FP8 is
fake/storage quant only -- no native FP8 speed claim.

Example:
  python -m experiments.kri_tierkv_a100 \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda --dtype bfloat16 \
    --context-len 2048 --block-size 128 --fast-window-tokens 512 \
    --protect-prefix-tokens 128 --slow-topk-blocks 4 \
    --quant none --output-dir runs/kri-tierkv/a100-emu
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kri_tierkv.config import (
    TierKVConfig,
    EvictionPolicy,
    RetrievalPolicy,
    QuantPolicy,
)
from kri_tierkv.block_index import BlockIndex
from kri_tierkv import scoring, eviction, fake_quant, trace


POLICIES = [
    ("dense_reference", EvictionPolicy.KRI_D_SUM, RetrievalPolicy.DENSE_REFERENCE),
    ("recent_only", EvictionPolicy.RECENCY, RetrievalPolicy.RECENT_ONLY),
    ("fifo_kri_topk", EvictionPolicy.FIFO, RetrievalPolicy.KRI_TOPK),
    ("kri_d_sum_kri_topk", EvictionPolicy.KRI_D_SUM, RetrievalPolicy.KRI_TOPK),
    ("oracle_topk", EvictionPolicy.KRI_D_SUM, RetrievalPolicy.ORACLE_TOPK),
]


def build_context(tok, target_tokens):
    """A long distractor context with one needle, trimmed to ~target tokens."""
    needle = (
        "IMPORTANT: the vault code for building 7 is MAGENTA-4471, set by "
        "director Priya Venkatesan on 2026-03-14. "
    )
    filler = "Routine log entry with no salient facts, status nominal. "
    text = filler * 40 + needle + filler * 400
    ids = tok(text, return_tensors="pt").input_ids[:, :target_tokens]
    return ids


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = (
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager"
    ).to(device)
    model.eval()

    ids = build_context(tok, args.context_len).to(device)
    seq_len = ids.shape[1]
    with torch.no_grad():
        out = model(ids, use_cache=True, output_attentions=True)
    kv = [(k, v) for k, v in out.past_key_values]
    attns = list(out.attentions)

    cfg0 = TierKVConfig(
        block_size=args.block_size,
        fast_window_tokens=args.fast_window_tokens,
        protect_prefix_tokens=args.protect_prefix_tokens,
        slow_topk_blocks=args.slow_topk_blocks,
        quant_policy=QuantPolicy(args.quant),
    )
    num_blocks = math.ceil(seq_len / args.block_size)

    # optional fake quant applied to the KV before scoring (storage sim)
    kb, vb = cfg0.resolved_bits()
    quant_rel_err = None
    if (kb, vb) != (16, 16):
        errs = []
        newkv = []
        for k, v in kv:
            kq, vq = fake_quant.fake_quant_kv(k, v, kb, vb)
            errs.append(fake_quant.quant_error(v, vq))
            newkv.append((kq, vq))
        kv = newkv
        quant_rel_err = sum(errs) / len(errs)

    kri_scores = scoring.kri_d_sum_scores(kv, args.block_size)
    mass = scoring.attention_mass_scores(attns, args.block_size, num_blocks)

    n_layers = len(kv)
    n_kv_heads = kv[0][0].shape[1]
    head_dim = kv[0][0].shape[3]

    os.makedirs(args.output_dir, exist_ok=True)
    tw = trace.TraceWriter(os.path.join(args.output_dir, "trace.jsonl"))
    rows = []
    for name, evp, rtp in POLICIES:
        cfg = TierKVConfig(
            block_size=args.block_size,
            fast_window_tokens=args.fast_window_tokens,
            protect_prefix_tokens=args.protect_prefix_tokens,
            slow_topk_blocks=args.slow_topk_blocks,
            eviction_policy=evp,
            retrieval_policy=rtp,
            quant_policy=cfg0.quant_policy,
        )
        idx = BlockIndex(seq_len, cfg)
        idx.set_scores(kri_scores)
        score_vec = mass if rtp == RetrievalPolicy.ORACLE_TOPK else kri_scores
        selected = eviction.retrieve(idx, cfg, score_vec)
        recall = trace.attention_mass_recall(selected, mass)
        fnr = trace.high_mass_false_negative_rate(selected, mass)
        bmoved = trace.bytes_moved_per_token(
            selected, idx, n_layers, n_kv_heads, head_dim, kb, vb
        )
        row = {
            "policy": name,
            "eviction": evp.value,
            "retrieval": rtp.value,
            "n_selected": len(selected),
            "num_blocks": num_blocks,
            "attention_mass_recall": round(recall, 4),
            "high_mass_false_negative_rate": round(fnr, 4),
            "bytes_moved_per_token": bmoved,
            "blocks_fetched_per_step": sum(
                1 for b in selected if b < num_blocks and idx.blocks[b].tier == "slow"
            ),
        }
        rows.append(row)
        tw.write(row)
        print(
            f"[{name:20}] recall={recall:.3f} fnr={fnr:.3f} "
            f"fetched_slow={row['blocks_fetched_per_step']}",
            flush=True,
        )
    tw.close()

    metrics = {
        "model": args.model,
        "device_name": (torch.cuda.get_device_name() if device == "cuda" else "cpu"),
        "context_len": seq_len,
        "block_size": args.block_size,
        "num_blocks": num_blocks,
        "fast_window_tokens": args.fast_window_tokens,
        "slow_topk_blocks": args.slow_topk_blocks,
        "quant": args.quant,
        "k_bits": kb,
        "v_bits": vb,
        "quant_rel_err": quant_rel_err,
        "policies": rows,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if args.html_report:
        from kri_tierkv.report import write_html

        write_html(metrics, os.path.join(args.output_dir, "report.html"))
    print("wrote", args.output_dir, flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--context-len", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--fast-window-tokens", type=int, default=512)
    ap.add_argument("--protect-prefix-tokens", type=int, default=128)
    ap.add_argument("--slow-topk-blocks", type=int, default=4)
    ap.add_argument("--quant", default="none", choices=[q.value for q in QuantPolicy])
    ap.add_argument("--html-report", action="store_true", default=True)
    ap.add_argument("--output-dir", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
