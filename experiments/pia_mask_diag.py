# SPDX-License-Identifier: GPL-2.0
"""Diagnostic for the masked-cache answering path used by the P1 replay.

Plants ONE unambiguous needle in a short prefix and checks three things:
  full            -- dense KV, no mask (must find the needle).
  honest_keep     -- keep a window that INCLUDES the needle tokens (must find).
  drop_needle     -- keep a window that EXCLUDES the needle tokens (must miss).

If full and honest_keep both find it and drop_needle misses, the masking +
position-pinning path is sound and the P1 weakness is a needle-design problem,
not a harness bug.
"""

from __future__ import annotations

import argparse

import torch


def answer(model, tok, ctx_ids, kept_prefix, q_text, device, gen=16):
    prefix_len = ctx_ids.shape[1]
    q_ids = tok(
        "\n\nQuestion: " + q_text + "\nAnswer:",
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    pmask = torch.zeros(prefix_len, dtype=torch.long, device=device)
    if kept_prefix is None:
        pmask[:] = 1
    else:
        idx = torch.tensor(sorted(kept_prefix), device=device)
        pmask[idx] = 1
    with torch.no_grad():
        base = model(ctx_ids, use_cache=True)
    cache = base.past_key_values
    q_len = q_ids.shape[1]
    attn = torch.cat([pmask, torch.ones(q_len, dtype=torch.long, device=device)])
    pos = torch.arange(prefix_len, prefix_len + q_len, device=device).unsqueeze(0)
    out = model(
        q_ids,
        past_key_values=cache,
        attention_mask=attn.unsqueeze(0),
        position_ids=pos,
        use_cache=True,
    )
    gen_ids = []
    with torch.no_grad():
        for step in range(gen):
            nxt = int(out.logits[0, -1].argmax())
            gen_ids.append(nxt)
            attn = torch.cat([attn, torch.ones(1, dtype=torch.long, device=device)])
            p = torch.tensor([[prefix_len + q_len + step]], device=device)
            out = model(
                torch.tensor([[nxt]], device=device),
                past_key_values=out.past_key_values,
                attention_mask=attn.unsqueeze(0),
                position_ids=p,
                use_cache=True,
            )
    return tok.decode(gen_ids)


def needle_token_span(tok, ctx_ids, needle_text, device):
    """Positions of the needle sentence, by token-subsequence match."""
    ids = ctx_ids[0].tolist()
    # tokenize the needle standalone (both with and without a leading space,
    # since BPE differs on word boundaries) and find it as a subsequence.
    for variant in (needle_text, " " + needle_text):
        pat = tok(variant, add_special_tokens=False).input_ids
        for i in range(len(ids) - len(pat) + 1):
            if ids[i : i + len(pat)] == pat:
                return set(range(i, i + len(pat)))
    # fall back: match just the code token(s)
    code = needle_text.split()[-1].strip(".")
    cpat = tok(" " + code, add_special_tokens=False).input_ids
    for i in range(len(ids) - len(cpat) + 1):
        if ids[i : i + len(cpat)] == cpat:
            return set(range(max(0, i - 12), i + len(cpat)))
    return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="eager"
        )
        .to(device)
        .eval()
    )

    filler = "Routine log entry with no salient facts, status nominal. "
    needle = "The vault code for the north tower is MAGENTA-4471."
    text = filler * 40 + needle + " " + filler * 40
    ctx_ids = tok(text, return_tensors="pt").input_ids.to(device)
    prefix_len = ctx_ids.shape[1]
    q = "What is the vault code for the north tower?"

    span = needle_token_span(tok, ctx_ids, needle, device)
    print(
        f"prefix {prefix_len} tokens, needle span {sorted(span)[:1]}..{sorted(span)[-1:]} "
        f"({len(span)} toks)",
        flush=True,
    )

    # honest keep: needle span + a sink + recent, well under full
    keep = set(range(64)) | set(range(prefix_len - 64, prefix_len)) | span
    # drop: same budget but WITHOUT the needle span
    drop = set(range(64)) | set(range(prefix_len - 64, prefix_len))
    drop -= span

    a_full = answer(model, tok, ctx_ids, None, q, device)
    a_keep = answer(model, tok, ctx_ids, keep, q, device)
    a_drop = answer(model, tok, ctx_ids, drop, q, device)

    def f(a):
        return "MAGENTA-4471" in a

    print(f"full        found={f(a_full)}  ::{a_full!r}", flush=True)
    print(f"honest_keep found={f(a_keep)}  ::{a_keep!r}", flush=True)
    print(f"drop_needle found={f(a_drop)}  ::{a_drop!r}", flush=True)
    ok = f(a_full) and f(a_keep) and not f(a_drop)
    print(
        "DIAG",
        "PASS masking path sound" if ok else "FAIL masking path suspect",
        flush=True,
    )


if __name__ == "__main__":
    main()
