"""Brick C: oracle V-page budget for SSD-offload viability (Qwen2.5-7B).

The real gate (per the protocol): keep FP8 keys in HBM, compute EXACT attention, then
ask how many 64KB V pages per layer per token an IDEAL policy must fetch to keep the
attention OUTPUT error tiny -- not "fraction of mass". 64KB page = 128 tokens (Qwen2.5
-7B FP8 V = 4 kv_heads x 128 dim x 1 byte = 512 bytes/token/layer). The I/O gate from
Brick B (layer-serial burst) is ~10-14 pages/layer for a 3 ms budget.

GQA care: a 64KB V page holds 128 tokens x ALL kv_heads for one layer, so a page is
shared across all 28 query heads. The byte budget is the UNION of token-pages any head
needs -- which can blow up if heads disagree. We rank pages by total cross-head mass,
fetch the top-B, RENORMALIZE attention over the fetched tokens per head, and report the
worst-head output relative error. Sampled queries = the last positions (late decode,
the offload-relevant case). Reports p50/p90/p99/max over (layer, query).
"""

import argparse
import sys

import torch
import torch.nn.functional as F

PAGE = 128  # tokens per 64KB V page (4 kv_heads * 128 dim * 1 byte = 512 B/tok/layer)
BUDGETS = [4, 8, 12, 16, 20, 32, 64, 128, 256]
_REC = []  # (layer, query_idx, {B: max_head_relerr})
_CFG = {}


def analysis(module, q, k, v):
    # q:[B,Hq,T,D] k,v:[B,Hkv,T,D] post-RoPE. Sampled queries = last n_sample positions.
    Hq = q.shape[1]
    Hkv = k.shape[1]
    g = Hq // Hkv
    T = k.shape[2]
    D = q.shape[-1]
    ns = min(_CFG["n_sample"], T)
    sidx = torch.arange(T - ns, T, device=q.device)
    qs = q[0, :, sidx, :]  # [Hq, ns, D]
    ke = k[0].repeat_interleave(g, dim=0)  # [Hq, T, D]
    ve = v[0].repeat_interleave(g, dim=0)  # [Hq, T, D]
    scale = 1.0 / (D**0.5)
    scores = torch.einsum("hsd,htd->hst", qs.float(), ke.float()) * scale  # [Hq,ns,T]
    alpha = torch.softmax(scores, dim=-1)  # exact attention
    o_full = torch.einsum("hst,htd->hsd", alpha, ve.float())  # [Hq,ns,D]
    npages = (T + PAGE - 1) // PAGE
    padT = npages * PAGE
    am = F.pad(alpha, (0, padT - T)).view(Hq, ns, npages, PAGE).sum(-1)  # [Hq,ns,np]
    page_score = am.sum(0)  # [ns, np] total cross-head mass per page (the shared page)
    li = module.layer_idx
    for s in range(ns):
        rec = {}
        order = torch.argsort(page_score[s], descending=True)
        for B in BUDGETS:
            if B >= npages:
                rec[B] = 0.0
                continue
            sel = order[:B]
            tok_mask = torch.zeros(padT, device=q.device, dtype=torch.bool)
            for p in sel.tolist():
                tok_mask[p * PAGE : (p + 1) * PAGE] = True
            tok_mask = tok_mask[:T]
            a = alpha[:, s, :] * tok_mask  # [Hq,T]
            a = a / a.sum(-1, keepdim=True).clamp(min=1e-9)  # renormalize per head
            o_p = torch.einsum("ht,htd->hd", a, ve.float())  # [Hq,D]
            relerr = (o_p - o_full[:, s]).norm(dim=-1) / o_full[:, s].norm(
                dim=-1
            ).clamp(min=1e-9)
            rec[B] = relerr.max().item()  # worst head
        _REC.append((li, int(sidx[s].item()), rec))


def install(model):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    impl = model.config._attn_implementation
    orig = ALL_ATTENTION_FUNCTIONS[impl]

    def wrapper(module, q, k, v, *a, **kw):
        if getattr(module, "layer_idx", None) is not None:
            with torch.no_grad():
                analysis(module, q, k, v)
        return orig(module, q, k, v, *a, **kw)

    ALL_ATTENTION_FUNCTIONS[impl] = wrapper
    return impl, orig


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--n-sample", type=int, default=16)
    ap.add_argument("--workload", default="wikitext", choices=["wikitext", "needle"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    _CFG["n_sample"] = args.n_sample
    sys.path.insert(0, "/tmp")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device)
    model.eval()
    install(model)

    if args.workload == "needle":
        filler = "The garden was quiet and the leaves drifted slowly. " * 4000
        needle = " The secret passcode is bluefin-7732. "
        text = filler[: len(filler) // 2] + needle + filler[len(filler) // 2 :]
        text += " The secret passcode is"
    else:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(t for t in ds["text"] if t and not t.isspace())
    ids = tok(text, return_tensors="pt").input_ids[:, : args.ctx].to(args.device)
    print(f"[pageoracle] {args.model} ctx={ids.shape[1]} workload={args.workload}")
    with torch.no_grad():
        model(ids)

    npages = (ids.shape[1] + PAGE - 1) // PAGE
    print(f"context = {ids.shape[1]} tokens = {npages} V-pages/layer (64KB each)")
    print(f"samples = {len(_REC)} (layer x query) records\n")
    print(f"{'budget B':>9}{'%ctx':>7}{'p50 err':>10}{'p90':>9}{'p99':>9}{'max':>9}")
    print("-" * 53)
    for B in BUDGETS:
        errs = [r[2][B] for r in _REC]
        print(
            f"{B:>9}{100*B/npages:>6.1f}%{pct(errs,50):>10.3f}{pct(errs,90):>9.3f}"
            f"{pct(errs,99):>9.3f}{max(errs):>9.3f}"
        )
    # min B for worst-head err < threshold, per record -> distribution
    for thr in (0.05, 0.01):
        need = []
        for _, _, rec in _REC:
            b = next((B for B in BUDGETS if rec[B] < thr), BUDGETS[-1] + 1)
            need.append(b)
        print(
            f"\npages/layer for worst-head err < {thr}: "
            f"p50={pct(need,50)} p90={pct(need,90)} p99={pct(need,99)} max={max(need)}"
        )


if __name__ == "__main__":
    main()
