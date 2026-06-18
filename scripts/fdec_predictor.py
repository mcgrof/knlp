"""Brick C2 Variant 4: prefetch-predictor hit-rate for sparse V-page decode.

The page-budget oracle (fdec_decode_pagev) uses EXACT per-layer QK selection -- it has
each layer's query when it picks pages. A real prefetcher must pick pages BEFORE it has
the query (to overlap fetch with compute, InfiniGen/Scout-style), so it predicts. This
measures whether the selection is predictable enough for that to work:

  - temporal hit-rate: how much does decode step t-1's selected page set overlap step t's
    exact set? (reuse-previous-token predictor)
  - layer-ahead hit-rate: how much does layer L's selected set overlap layer L+1's within
    the same step? (predict next layer from current, Scout-style)

High hit-rate -> prefetch is cheap and the oracle budget is ~achievable. Low hit-rate ->
prefetch misses -> synchronous stalls -> the marginal byte budget is unreachable. Reports
mean + the LOW percentiles (the misses cause the stalls, so p10/p50 matter).
"""

import argparse
import sys

import torch
import torch.nn.functional as F

PAGE = 128
_PV = {"B": 16, "local": 16, "sink": 1, "orig": None}
_STEP = {}  # layer_idx -> set(selected older-page indices) for the current decode step


def harness_attn(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
    if q.shape[2] > 1:
        return _PV["orig"](
            module, q, k, v, attention_mask, dropout=dropout, scaling=scaling, **kw
        )
    g = module.num_key_value_groups
    ks = k.repeat_interleave(g, dim=1)
    vs = v.repeat_interleave(g, dim=1)
    Tk = ks.shape[2]
    aw = torch.matmul(q.float(), ks.float().transpose(2, 3)) * scaling
    if attention_mask is not None:
        aw = aw + attention_mask[:, :, :, :Tk].float()
    aw = torch.softmax(aw, dim=-1)
    B, local, sink = _PV["B"], _PV["local"], _PV["sink"]
    npg = (Tk + PAGE - 1) // PAGE
    if B + local + sink < npg:
        pad = npg * PAGE - Tk
        pm = F.pad(aw, (0, pad)).view(aw.shape[0], aw.shape[1], 1, npg, PAGE).sum(-1)
        tot = pm.sum(1)[:, 0, :]
        resident = torch.zeros(aw.shape[0], npg, dtype=torch.bool, device=aw.device)
        if sink:
            resident[:, :sink] = True
        if local:
            resident[:, npg - local :] = True
        tot = tot.masked_fill(resident, float("-inf"))
        sel = tot.topk(B, dim=-1).indices[0]  # the fetched older pages this layer/step
        _STEP[module.layer_idx] = set(sel.tolist())
        keep = resident.clone()
        keep.scatter_(1, sel.unsqueeze(0), True)
        keymask = keep.repeat_interleave(PAGE, dim=1)[:, :Tk]
        aw = aw * keymask[:, None, None, :]
        aw = aw / aw.sum(-1, keepdim=True).clamp(min=1e-9)
    o = torch.matmul(aw.to(q.dtype), vs).transpose(1, 2).contiguous()
    return o, aw


def install(model):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    impl = model.config._attn_implementation
    _PV["orig"] = ALL_ATTENTION_FUNCTIONS[impl]
    ALL_ATTENTION_FUNCTIONS[impl] = harness_attn


def jacc(a, b):  # |a & b| / |b| -- fraction of b's pages already in a (prefetch hit)
    return len(a & b) / max(1, len(b))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--local-pages", type=int, default=16)
    ap.add_argument("--sink-pages", type=int, default=1)
    ap.add_argument("--page-size", type=int, default=128)
    ap.add_argument("--prefill-chunk", type=int, default=8192)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    global PAGE
    PAGE = args.page_size
    _PV.update(B=args.B, local=args.local_pages, sink=args.sink_pages)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device)
    model.eval()
    install(model)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    txt = "\n".join(t for t in ds["text"][:300000] if t and not t.isspace())
    toks = tok(txt)["input_ids"]
    ids = torch.tensor(toks[: args.ctx]).unsqueeze(0).to(args.device)
    print(f"[predictor] ctx={args.ctx} gen={args.gen} B={args.B} page={PAGE}")

    # chunked prefill
    pkv = None
    for i in range(0, args.ctx, args.prefill_chunk):
        out = model(
            ids[:, i : i + args.prefill_chunk], past_key_values=pkv, use_cache=True
        )
        pkv = out.past_key_values
    cur = out.logits[:, -1:].argmax(-1)

    history = []  # per gen step: dict layer-> set(pages)
    for _ in range(args.gen):
        _STEP.clear()
        out = model(cur, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        cur = out.logits[:, -1:].argmax(-1)
        history.append(dict(_STEP))

    layers = sorted(history[0].keys())
    temporal, layerahead = [], []
    for t in range(1, len(history)):
        for L in layers:
            if L in history[t] and L in history[t - 1]:
                temporal.append(jacc(history[t - 1][L], history[t][L]))
    for t in range(len(history)):
        for i in range(len(layers) - 1):
            L, Ln = layers[i], layers[i + 1]
            if L in history[t] and Ln in history[t]:
                layerahead.append(jacc(history[t][L], history[t][Ln]))

    def report(name, xs):
        if not xs:
            print(f"  {name}: (no data)")
            return
        print(
            f"  {name}: mean={sum(xs)/len(xs):.3f}  p10={pct(xs,10):.3f} "
            f"p50={pct(xs,50):.3f} p90={pct(xs,90):.3f}  (n={len(xs)})"
        )

    print(f"\nprefetch hit-rate (fraction of needed pages already predicted), B={args.B}")
    report("temporal (reuse prev token)", temporal)
    report("layer-ahead (predict L+1 from L)", layerahead)


if __name__ == "__main__":
    main()
