# SPDX-License-Identifier: GPL-2.0
"""Parity check: the cheap qk selector vs the output_attentions selector.

At a context small enough for output_attentions, both should surface the same
prefix tokens. Reports top-k set Jaccard and whether both keep the needle.
"""

import argparse

import torch

from pia_qk_selector import qk_top_tokens
from pia_reuse_replay import NEEDLES, answer, build_context, query_aware_kept


def _found(ans, expected):
    n = lambda s: "".join(c for c in s.upper() if c.isalnum())
    return n(expected) in n(ans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--context-len", type=int, default=2048)
    ap.add_argument("--keep-ratio", type=float, default=0.08)
    ap.add_argument("--obs", type=int, default=10)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        .to(dev)
        .eval()
    )
    needles = NEEDLES[:6]
    ctx_text, qa = build_context(tok, needles, args.context_len)
    ctx_ids = tok(ctx_text, return_tensors="pt").input_ids.to(dev)
    prefix_len = ctx_ids.shape[1]
    keep = max(1, int(prefix_len * args.keep_ratio))

    jac, attn_acc, qk_acc, full_acc = [], [], [], []
    for q, expected in qa:
        q_ids = tok(
            "\n\nQuestion: " + q + "\nAnswer with only the code:",
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(dev)
        full = torch.cat([ctx_ids, q_ids], dim=1)
        attn_set = query_aware_kept(model, tok, ctx_ids, q, keep, dev, obs=args.obs)
        qk_set = qk_top_tokens(model, full, prefix_len, keep, args.obs, dev)
        jac.append(len(attn_set & qk_set) / max(1, len(attn_set | qk_set)))
        full_acc.append(_found(answer(model, tok, ctx_ids, None, q, dev), expected))
        attn_acc.append(_found(answer(model, tok, ctx_ids, attn_set, q, dev), expected))
        qk_acc.append(_found(answer(model, tok, ctx_ids, qk_set, q, dev), expected))

    def m(v):
        return sum(v) / len(v)

    print(f"prefix {prefix_len}, keep {keep}, {len(qa)} queries")
    print(f"top-k set Jaccard: mean={m(jac):.3f}")
    print(
        f"answer accuracy  full={m(full_acc):.3f}  "
        f"attn-selector={m(attn_acc):.3f}  qk-selector={m(qk_acc):.3f}"
    )


if __name__ == "__main__":
    main()
