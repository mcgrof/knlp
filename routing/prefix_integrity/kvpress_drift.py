# SPDX-License-Identifier: GPL-2.0
"""Measure output drift from actual KVPress cache compression methods.

Compare the next-token distribution and a short greedy answer with a dense
baseline for four synthetic questions over a shared context. Report KL
divergence, top-1 agreement, answer exact-match, and expected-answer retrieval.

Each press compresses the context before the question is appended. This
measures query-independent compression drift, not cross-query prefix reuse.
The cache-contract checks live in the separate prefix-integrity harness.

Requires torch, kvpress, transformers, and accelerate. Run with
``python -m routing.prefix_integrity.kvpress_drift --out kvpress.json``.
"""

import argparse
import json
import os


def build_context(n_filler=180, needle_pos=90):
    """A long distractor context with one needle fact and a magic number."""
    filler = [
        f"Record {i}: routine log entry with no salient facts, item code "
        f"{1000 + i}, status nominal, reviewed by desk {i % 7}."
        for i in range(n_filler)
    ]
    needle = (
        "IMPORTANT FACT: The archival vault access code for building 7 is "
        "MAGENTA-4471, authorized by director Priya Venkatesan on 2026-03-14."
    )
    filler[needle_pos] = needle
    return "\n".join(filler)


PROBES = [
    ("What is the archival vault access code for building 7?", "MAGENTA-4471"),
    ("Who authorized the building 7 vault access code?", "Priya Venkatesan"),
    ("On what date was the vault access authorized?", "2026-03-14"),
    ("What status do the routine log entries report?", "nominal"),
]


def _dist_and_answer(model, tok, prompt_ids, cache, device, gen=12):
    """Return (next-token softmax, greedy answer text) continuing from cache."""
    import torch

    with torch.no_grad():
        out = model(prompt_ids.to(device), past_key_values=cache, use_cache=True)
    logits = out.logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    # greedy continue
    cur = cache
    ids = [int(logits.argmax())]
    nxt = torch.tensor([[ids[0]]], device=device)
    with torch.no_grad():
        o = out
        for _ in range(gen - 1):
            o = model(nxt, past_key_values=o.past_key_values, use_cache=True)
            t = int(o.logits[0, -1].argmax())
            ids.append(t)
            nxt = torch.tensor([[t]], device=device)
    return probs, tok.decode(ids)


def _kl(p, q, eps=1e-8):
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return float((p * (p / q).log()).sum())


def run(model, tok, press_ctor, context, device, compression_ratio):
    """Measure drift of one press (None = dense baseline reference build)."""
    import torch

    ctx_ids = tok(context, return_tensors="pt").input_ids.to(device)
    results = []
    for question, expected in PROBES:
        q = f"\n\nQuestion: {question}\nAnswer:"
        q_ids = tok(q, return_tensors="pt", add_special_tokens=False).input_ids

        # dense cache (full context)
        with torch.no_grad():
            dense_out = model(ctx_ids, use_cache=True)
        dense_cache = dense_out.past_key_values
        p_dense, a_dense = _dist_and_answer(model, tok, q_ids, dense_cache, device)

        if press_ctor is None:
            results.append(
                {
                    "q": question,
                    "expected": expected,
                    "answer": a_dense,
                    "found": expected.lower() in a_dense.lower(),
                    "kl": 0.0,
                    "top1": 1,
                }
            )
            continue

        press = press_ctor(compression_ratio=compression_ratio)
        with torch.no_grad():
            with press(model):
                comp_out = model(ctx_ids, use_cache=True)
        comp_cache = comp_out.past_key_values
        p_comp, a_comp = _dist_and_answer(model, tok, q_ids, comp_cache, device)

        results.append(
            {
                "q": question,
                "expected": expected,
                "answer": a_comp,
                "found": expected.lower() in a_comp.lower(),
                "answer_match": a_comp.strip() == a_dense.strip(),
                "kl": _kl(p_dense, p_comp),
                "top1": int(p_dense.argmax().item() == p_comp.argmax().item()),
            }
        )
    n = len(results)
    return {
        "kl_mean": sum(r["kl"] for r in results) / n,
        "kl_max": max(r["kl"] for r in results),
        "top1_agreement": sum(r["top1"] for r in results) / n,
        "needle_found_rate": sum(1 for r in results if r["found"]) / n,
        "answer_match_rate": sum(1 for r in results if r.get("answer_match")) / n,
        "per_probe": results,
    }


def load_presses():
    """Import the KVPress presses that exist in this install."""
    import kvpress

    names = [
        ("streaming_llm", "StreamingLLMPress"),
        ("knorm", "KnormPress"),
        ("snapkv", "SnapKVPress"),
        ("expected_attention", "ExpectedAttentionPress"),
        ("tova", "TOVAPress"),
        ("h2o", "ObservedAttentionPress"),
        ("pyramidkv", "PyramidKVPress"),
        ("adakv", "AdaKVPress"),
    ]
    out = {}
    for key, cls in names:
        p = getattr(kvpress, cls, None)
        if p is not None:
            out[key] = p
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--compression-ratio", type=float, default=0.75)
    ap.add_argument("--n-filler", type=int, default=180)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    model.eval()

    context = build_context(n_filler=args.n_filler)
    ctx_len = tok(context, return_tensors="pt").input_ids.shape[1]
    presses = load_presses()
    print(
        f"context {ctx_len} tokens; presses: {sorted(presses)}; "
        f"compression_ratio={args.compression_ratio}",
        flush=True,
    )

    out = {
        "model": args.model,
        "context_tokens": int(ctx_len),
        "compression_ratio": args.compression_ratio,
        "methods": {},
    }

    # dense reference
    dense = run(model, tok, None, context, device, args.compression_ratio)
    out["methods"]["dense"] = dense
    print(f"[dense] needle_found={dense['needle_found_rate']:.2f}", flush=True)

    for key, ctor in presses.items():
        try:
            r = run(model, tok, ctor, context, device, args.compression_ratio)
            out["methods"][key] = r
            print(
                f"[{key}] kl_mean={r['kl_mean']:.4f} top1={r['top1_agreement']:.3f} "
                f"needle_found={r['needle_found_rate']:.2f} "
                f"answer_match={r['answer_match_rate']:.2f}",
                flush=True,
            )
        except Exception as e:  # keep going if one press API differs
            out["methods"][key] = {"error": str(e)}
            print(f"[{key}] ERROR {e}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
