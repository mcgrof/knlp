"""ScoutAttention-faithful layer-ahead V-page predictor (the within-token prefetch gate).

The naive layer-ahead predictor (reuse layer L's selected pages for L+1) scored only 0.23 --
consecutive layers attend to nearly disjoint page sets, so it cannot drive a within-token
prefetcher. ScoutAttention's actual trick is different: it PREDICTS the next layer's QUERY
early. At decode, to hide layer L's V-fetch you must start it ~one layer ahead, while you
still only have the residual that fed layer L-1. So predict layer L's query from that stale
residual:

    q_hat[L] = RoPE( W_Q[L]( input_layernorm[L]( h_{L-1} ) ) )

where h_{L-1} is the residual that was the INPUT to layer L-1 (one full transformer layer of
lead time: q_hat[L]'s pages can be fetched while L-1's attention+MLP run). The exact query
uses the input to layer L. We measure (a) cosine(q_hat[L], q_exact[L]) per decode step, and
(b) PAGE RECALL: with q_hat selecting top-(B x overfetch) pages, what fraction of the pages
the EXACT query needs are covered. Recall ~1.0 at small overfetch => layer-ahead prefetch is
reachable and the oracle byte budget is achievable; low recall => within-token prefetch is
not rescued and the budget stays an oracle-only number. Also reports the EXTRA bytes the
overfetch costs, so the fetch-budget accounting stays honest.
"""

import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

PAGE = 32
_PV = {"B": 64, "local": 64, "sink": 4, "orig": None}
_H = {}  # layer_idx -> hidden state that was the INPUT to that layer, this decode step
_LAYER = {}  # layer_idx -> decoder layer module (for input_layernorm)
_ROT = {"emb": None}
_R = {"cos": [], "recall": defaultdict(list), "exact_sz": []}
OVERFETCH = (1.0, 1.25, 1.5, 2.0)


def select_pages(pm, npg, B, local, sink):
    resident = torch.zeros(npg, dtype=torch.bool, device=pm.device)
    if sink:
        resident[:sink] = True
    if local:
        resident[npg - local :] = True
    masked = pm.masked_fill(resident, float("-inf"))
    k = min(B, int(resident.numel() - resident.sum().item()))
    if k <= 0:
        return set()
    return set(masked.topk(k).indices.tolist())


def harness_attn(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
    if q.shape[2] > 1:  # PREFILL
        return _PV["orig"](
            module, q, k, v, attention_mask, dropout=dropout, scaling=scaling, **kw
        )
    L = module.layer_idx
    g = module.num_key_value_groups
    ks = k.repeat_interleave(g, dim=1)
    vs = v.repeat_interleave(g, dim=1)
    Tk = ks.shape[2]
    npg = (Tk + PAGE - 1) // PAGE
    pad = npg * PAGE - Tk
    B, local, sink = _PV["B"], _PV["local"], _PV["sink"]

    def pagemass(query):
        aw = torch.matmul(query.float(), ks.float().transpose(2, 3)) * scaling
        if attention_mask is not None:
            aw = aw + attention_mask[:, :, :, :Tk].float()
        aw = torch.softmax(aw, dim=-1)
        pm = F.pad(aw, (0, pad)).view(1, aw.shape[1], 1, npg, PAGE).sum(-1)
        return pm.sum(1)[0, 0, :], aw  # [npg], full aw for the exact path

    exact_pm, aw = pagemass(q)
    exact_sel = select_pages(exact_pm, npg, B, local, sink)

    # ScoutAttention layer-ahead: predict THIS layer's query from the residual that fed the
    # PREVIOUS layer (one full layer of prefetch lead time).
    if (L - 1) in _H and L >= 1 and len(exact_sel) > 0:
        h_proxy = _H[L - 1]
        dl = _LAYER[L]
        qa = module.q_proj(dl.input_layernorm(h_proxy))  # [1,1,Hq*D]
        D = q.shape[-1]
        Hq = q.shape[1]
        qa = qa.view(1, 1, Hq, D).transpose(1, 2)  # [1,Hq,1,D]
        pos = torch.arange(Tk - 1, Tk, device=q.device).unsqueeze(0)
        c, s = _ROT["emb"](qa, pos)
        qa, _ = apply_rotary_pos_emb(qa, qa, c, s)
        approx_pm, _ = pagemass(qa)
        _R["cos"].append(F.cosine_similarity(qa.flatten(), q.flatten(), dim=0).item())
        _R["exact_sz"].append(len(exact_sel))
        for of in OVERFETCH:
            asel = select_pages(approx_pm, npg, int(round(B * of)), local, sink)
            _R["recall"][of].append(len(exact_sel & asel) / len(exact_sel))

    # keep decode coherent: apply the EXACT sparse-V mask (same as the budget harness)
    if B + local + sink < npg:
        resident = torch.zeros(1, npg, dtype=torch.bool, device=aw.device)
        if sink:
            resident[:, :sink] = True
        if local:
            resident[:, npg - local :] = True
        keep = resident.clone()
        keep[:, list(exact_sel)] = True
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
    _ROT["emb"] = model.model.rotary_emb
    for i, dl in enumerate(model.model.layers):
        _LAYER[i] = dl

        def mk(idx):
            def pre(mod, args, kwargs):
                hs = kwargs.get("hidden_states", args[0] if args else None)
                if hs is not None and hs.shape[1] == 1:  # decode only
                    _H[idx] = hs.detach()
                return None

            return pre

        dl.register_forward_pre_hook(mk(i), with_kwargs=True)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--local-pages", type=int, default=64)
    ap.add_argument("--sink-pages", type=int, default=4)
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--page-kb", type=float, default=16.0)
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
    print(
        f"[scout] ctx={args.ctx} gen={args.gen} B={args.B} page={PAGE} ({args.page_kb}KB) "
        f"local={args.local_pages} sink={args.sink_pages}"
    )

    pkv = None
    for i in range(0, args.ctx, args.prefill_chunk):
        out = model(
            ids[:, i : i + args.prefill_chunk], past_key_values=pkv, use_cache=True
        )
        pkv = out.past_key_values
    cur = out.logits[:, -1:].argmax(-1)
    for _ in range(args.gen):
        _H.clear()
        out = model(cur, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        cur = out.logits[:, -1:].argmax(-1)

    cosv = _R["cos"]
    print(f"\nn samples (layer x step) = {len(cosv)}")
    print(
        f"cosine(q_hat[L], q_exact[L]): mean={sum(cosv)/len(cosv):.3f} "
        f"p10={pct(cosv,10):.3f} p50={pct(cosv,50):.3f} p90={pct(cosv,90):.3f}"
    )
    sz = _R["exact_sz"]
    print(f"exact selected pages/layer: mean={sum(sz)/len(sz):.1f}\n")
    print(
        "layer-ahead page RECALL (frac of exact-needed pages q_hat covers) + fetch cost:"
    )
    print(
        f"{'overfetch':>10}{'pages fetched':>15}{'recall mean':>13}{'p10':>8}{'p50':>8}{'p90':>8}"
    )
    for of in OVERFETCH:
        r = _R["recall"][of]
        fetched = int(round(args.B * of))
        mb = fetched * args.page_kb / 1024
        print(
            f"{of:>9.2f}x{f'{fetched}pg/{mb:.2f}MB':>15}"
            f"{sum(r)/len(r):>13.3f}{pct(r,10):>8.3f}{pct(r,50):>8.3f}{pct(r,90):>8.3f}"
        )


if __name__ == "__main__":
    main()
