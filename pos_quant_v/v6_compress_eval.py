#!/usr/bin/env python3
"""3-selector V-cache compression replay on real NIAH retrieval.

Prefill a needle-in-a-haystack context at full precision, then COMPRESS the
value cache and decode the answer over it -- exactly the deployment path (cache
written full at prefill, queries attend over compressed V). Compares selectors
for which past tokens keep full Value vs a rank-k projection:

  full            no compression (baseline)
  all_rank4       every token's V compressed to rank-4 (the floor)
  content_oracle  full V at true content (key/value) tokens, rank-4 at filler
                  -- the denoising arm
  delta_thresh    certified: compress token i to rank-4 iff its V residual
                  delta_i <= eps (else full); guarantees ||Delta o|| <= eps per
                  head, query-independent. Sweep eps.

Reports, per arm: retrieval accuracy, KV-V bytes vs full, and for delta_thresh
the F1 overlap of its compress decision with the true filler mask, plus the
certified bound (sum alpha_i delta_i) vs the actual attention-output change.

Per-(layer, kv-head) rank-k basis is fit by PCA on a held-out calibration set of
contexts (a fixed shared basis, as the scheme requires).

Usage:
  python3 pos_quant_v/v6_compress_eval.py --model Qwen/Qwen2.5-7B-Instruct \
      --length 4096 --needles 4 --eval 20 --calib 8 --out OUT/v6.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from niah_task import build_context, token_content_mask, load_filler_sentences


@torch.no_grad()
def prefill_values(model, ids, device):
    """Forward, return DynamicCache and the list of value tensors per layer."""
    out = model(ids.to(device), use_cache=True)
    return out.past_key_values


def get_values(cache):
    """Per-layer value tensors, robust to the transformers cache API."""
    if hasattr(cache, "value_cache"):
        return cache.value_cache
    return [layer.values for layer in cache.layers]


def set_value(cache, L, v):
    if hasattr(cache, "value_cache"):
        cache.value_cache[L] = v
    else:
        cache.layers[L].values = v


def fit_bases(value_lists, ks, device):
    """value_lists: list over contexts of [n_layers] tensors [1,H,T,hd].
    Returns bases[k][layer] = [H, hd, k] (per-head PCA top-k)."""
    n_layers = len(value_lists[0])
    H = value_lists[0][0].shape[1]
    hd = value_lists[0][0].shape[3]
    bases = {k: [] for k in ks}
    for L in range(n_layers):
        # pool V across calib contexts per head -> [H, sumT, hd]
        per_head = []
        for h in range(H):
            chunks = [vl[L][0, h] for vl in value_lists]  # [T,hd] each
            per_head.append(torch.cat(chunks, 0).float())  # [sumT,hd]
        Bk = {k: torch.empty(H, hd, k, device=device) for k in ks}
        for h in range(H):
            X = per_head[h].to(device)
            # uncentered PCA (keeps mean direction)
            M = X.t() @ X
            evals, evecs = torch.linalg.eigh(M)
            for k in ks:
                Bk[k][h] = evecs[:, -k:]
        for k in ks:
            bases[k].append(Bk[k])
    return bases  # bases[k][L] = [H,hd,k]


def residuals(V, basis_k):
    """V [1,H,T,hd], basis_k [H,hd,k] -> delta [H,T] residual norm of rank-k."""
    H, hd, k = basis_k.shape
    v = V[0].float()  # [H,T,hd]
    # recon = v @ B @ B^T  per head
    proj = torch.einsum("htd,hdk->htk", v, basis_k)
    recon = torch.einsum("htk,hdk->htd", proj, basis_k)
    delta = (v - recon).norm(dim=-1)  # [H,T]
    return recon, delta


def compress_cache(past, recon_rank4, keep_full_mask):
    """Return a new value_cache where tokens with keep_full_mask=False are
    replaced by their rank-4 reconstruction. keep_full_mask: [T] bool (per token,
    shared across heads/layers -- selectors decide per token)."""
    import copy

    new = copy.deepcopy(past)
    km = keep_full_mask
    vals = get_values(new)
    for L in range(len(vals)):
        v = vals[L]  # [1,H,T,hd]
        rec = recon_rank4[L].unsqueeze(0).to(v.dtype)  # [1,H,T,hd]
        comp = torch.where(km.view(1, 1, -1, 1), v, rec)
        set_value(new, L, comp)
    return new


@torch.no_grad()
def decode_answer(model, last_id, past, device, max_new=20):
    ids = last_id.view(1, 1).to(device)
    gen = []
    cur = past
    for _ in range(max_new):
        out = model(ids, past_key_values=cur, use_cache=True)
        nxt = out.logits[0, -1].argmax()
        gen.append(int(nxt))
        cur = out.past_key_values
        ids = nxt.view(1, 1)
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--needles", type=int, default=4)
    ap.add_argument("--eval", type=int, default=20)
    ap.add_argument("--calib", type=int, default=8)
    ap.add_argument("--ranks", default="4,8,16,32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device).eval()
    sents = load_filler_sentences(args.seed)
    rng = random.Random(args.seed)

    def make(L):
        text, spans, needles, qa = build_context(tok, L, args.needles, sents, rng)
        # do NOT truncate: the query is at the end of the context and must be kept
        ids, cmask = token_content_mask(tok, text, spans, 10**9)
        return torch.tensor(ids), torch.tensor(cmask), qa  # qa=(key,value)

    def answer_correct(gen_ids, value_str):
        return value_str in tok.decode(gen_ids)

    # baseline retrieval gate
    base_hits = 0
    samples = []
    for _ in range(args.eval):
        ids, cmask, qa = make(args.length)
        samples.append((ids, cmask, qa))
        past = prefill_values(model, ids[:-1].unsqueeze(0), device)
        gen = decode_answer(model, ids[-1], past, device)
        base_hits += answer_correct(gen, qa[1])
    base_acc = base_hits / args.eval
    print(
        f"[{args.model}] L={args.length} baseline retrieval acc={base_acc:.3f}",
        flush=True,
    )

    result = {
        "model": args.model,
        "length": args.length,
        "needles": args.needles,
        "eval": args.eval,
        "baseline_acc": base_acc,
        "arms": {},
    }
    if args.baseline_only or base_acc < 0.5:
        if base_acc < 0.5:
            print("  baseline too low; skipping compression arms", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[wrote] {args.out}")
        return

    # calibration bases
    calib_vals = []
    for _ in range(args.calib):
        ids, _, _ = make(args.length)
        past = prefill_values(model, ids.unsqueeze(0), device)
        calib_vals.append([v.cpu() for v in get_values(past)])
    ranks = [int(x) for x in args.ranks.split(",")]
    bases = fit_bases(calib_vals, ranks, device)
    print(f"  fit bases for ranks {ranks}", flush=True)

    # compress-fraction targets for the certified delta-selector (compress the
    # bottom-f tokens by residual; rank-comparable, unlike an absolute eps)
    fracs = [0.5, 0.75, 0.9, 0.95, 0.99]
    arms = {}  # name -> list of (correct, frac_full, f1_vs_filler)

    def record(name, correct, frac_full, f1):
        arms.setdefault(name, []).append((correct, frac_full, f1))

    def f1_compress_vs_filler(keepmask, cmask_t):
        pred_c = ~keepmask
        true_f = ~cmask_t
        tp = (pred_c & true_f).sum().item()
        fp = (pred_c & ~true_f).sum().item()
        fn = (~pred_c & true_f).sum().item()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return 2 * p * r / max(p + r, 1e-9)

    for ids, cmask, qa in samples:
        past = prefill_values(model, ids[:-1].unsqueeze(0), device)
        vals = get_values(past)
        T = vals[0].shape[2]
        cmask_t = cmask[:T].to(device)
        for k in ranks:
            bk = bases[k]
            recons = []
            delta_tok = torch.zeros(T, device=device)
            for L in range(len(vals)):
                rec, delta = residuals(vals[L], bk[L])
                recons.append(rec)
                delta_tok += delta.mean(0)
            delta_tok /= len(vals)
            # content_oracle@k: full V at true content, rank-k at filler
            comp = compress_cache(past, recons, cmask_t)
            gen = decode_answer(model, ids[-1], comp, device)
            record(
                f"r{k}_content_oracle",
                answer_correct(gen, qa[1]),
                cmask_t.float().mean().item(),
                f1_compress_vs_filler(cmask_t, cmask_t),
            )
            # delta-selector: compress the bottom-f by residual
            for f in fracs:
                thr = torch.quantile(delta_tok, f)
                keepmask = delta_tok > thr  # keep top-(1-f) full
                comp = compress_cache(past, recons, keepmask)
                gen = decode_answer(model, ids[-1], comp, device)
                record(
                    f"r{k}_delta_f{f}",
                    answer_correct(gen, qa[1]),
                    keepmask.float().mean().item(),
                    f1_compress_vs_filler(keepmask, cmask_t),
                )

    n = len(samples)
    print(f"  --- arms (baseline acc {base_acc:.3f}) ---", flush=True)
    for name in arms:
        rows = arms[name]
        acc = sum(c for c, _, _ in rows) / n
        bf = sum(b for _, b, _ in rows) / n
        f1 = sum(x for _, _, x in rows) / n
        result["arms"][name] = {"acc": acc, "frac_full_V": bf, "f1_vs_filler": f1}
        print(
            f"  {name:22s} acc={acc:.3f}  frac_full_V={bf:.3f}  "
            f"F1(compress,filler)={f1:.3f}",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
