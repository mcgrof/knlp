"""Brick C pre-step (prune, free): DECODE-mode V-page budget at 32K/64K.

The eager OOM only hits PREFILL (full T x T attention). DECODE attends with a SINGLE
query over the cached context, so exact QK is [1, T] -- cheap at any length. This
harness prefills with flash (memory-feasible), then measures one decode step under
top-B V-page selection vs full-V, across several context chunks, at 32K and 64K. It
validates the decode harness for the 128K pod and extends the page-budget trend.

Per decode step the selected pages ARE the shared byte budget (B pages, ranked by
cross-head mass), so we sweep B and report the min B for top-1 match / low KL across
chunks (p50/p90/p99). Quality = top-1 agreement + KL(full || paged) at the decode
position. Same metric family as the 4-12K end-model result, now at decode + long ctx.
"""

import argparse
import sys

import torch
import torch.nn.functional as F

PAGE = 128
_PV = {"B": 0, "orig": None}


def harness_attn(module, q, k, v, attention_mask, scaling, dropout=0.0, **kw):
    if q.shape[2] > 1:  # PREFILL: real attention (flash/sdpa), no paging
        return _PV["orig"](module, q, k, v, attention_mask, scaling, dropout, **kw)
    # DECODE: single query, exact QK over the full cache, optional V-paging
    g = module.num_key_value_groups
    ks = k.repeat_interleave(g, dim=1)
    vs = v.repeat_interleave(g, dim=1)
    Tk = ks.shape[2]
    aw = torch.matmul(q.float(), ks.float().transpose(2, 3)) * scaling  # [B,Hq,1,Tk]
    if attention_mask is not None:
        aw = aw + attention_mask[:, :, :, :Tk].float()
    aw = torch.softmax(aw, dim=-1)
    B = _PV["B"]
    if B and B > 0:
        npg = (Tk + PAGE - 1) // PAGE
        if B < npg:
            pad = npg * PAGE - Tk
            pm = (
                F.pad(aw, (0, pad)).view(aw.shape[0], aw.shape[1], 1, npg, PAGE).sum(-1)
            )
            tot = pm.sum(1)[:, 0, :]  # [B, npg] cross-head page mass
            sel = tot.topk(B, dim=-1).indices  # [B, B]
            psel = torch.zeros(aw.shape[0], npg, dtype=torch.bool, device=aw.device)
            psel.scatter_(1, sel, True)
            keymask = psel.repeat_interleave(PAGE, dim=1)[:, :Tk]  # [B, Tk]
            aw = aw * keymask[:, None, None, :]
            aw = aw / aw.sum(-1, keepdim=True).clamp(min=1e-9)
    o = torch.matmul(aw.to(q.dtype), vs).transpose(1, 2).contiguous()
    return o, aw


def install(model):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    impl = model.config._attn_implementation
    _PV["orig"] = ALL_ATTENTION_FUNCTIONS[impl]
    ALL_ATTENTION_FUNCTIONS[impl] = harness_attn


@torch.no_grad()
def decode_logit(model, ids, T, B):
    # prefill [0:T-1] (full), then ONE decode step at position T-1 under budget B
    _PV["B"] = 0
    out = model(ids[:, : T - 1], use_cache=True)
    pkv = out.past_key_values
    _PV["B"] = B
    out = model(ids[:, T - 1 : T], past_key_values=pkv, use_cache=True)
    return out.logits[0, -1].float()


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--budgets", default="4,8,12,16,20,24,32,48,64")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device)
    model.eval()
    install(model)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    txt = "\n".join(t for t in ds["text"][:200000] if t and not t.isspace())
    toks = tok(txt)["input_ids"]
    budgets = [int(x) for x in args.budgets.split(",")]
    print(f"[decode-pagev] {args.model} ctx={args.ctx} chunks={args.chunks}")
    npg = (args.ctx + PAGE - 1) // PAGE
    print(f"context pages = {npg}")

    agg = {B: {"t1": [], "kl": []} for B in budgets}
    for c in range(args.chunks):
        s = c * args.ctx
        if s + args.ctx + 1 > len(toks):
            break
        ids = torch.tensor(toks[s : s + args.ctx]).unsqueeze(0).to(args.device)
        full = decode_logit(model, ids, args.ctx, 0)
        fa = full.argmax().item()
        flp = F.log_softmax(full, dim=-1)
        fp = flp.exp()
        for B in budgets:
            lg = decode_logit(model, ids, args.ctx, B)
            agg[B]["t1"].append(int(lg.argmax().item() == fa))
            lp = F.log_softmax(lg, dim=-1)
            agg[B]["kl"].append((fp * (flp - lp)).sum().item())
        torch.cuda.empty_cache()

    n = len(agg[budgets[0]]["t1"])
    print(f"\nmeasured {n} decode positions")
    print(
        f"{'pages B':>8}{'%ctx':>7}{'top1 match':>12}{'KL p50':>9}{'KL p90':>9}{'KL p99':>9}"
    )
    print("-" * 54)
    for B in budgets:
        t1 = sum(agg[B]["t1"]) / max(1, n)
        kl = agg[B]["kl"]
        print(
            f"{B:>8}{100*B/npg:>6.1f}%{t1:>12.3f}"
            f"{pct(kl,50):>9.3f}{pct(kl,90):>9.3f}{pct(kl,99):>9.3f}"
        )


if __name__ == "__main__":
    main()
