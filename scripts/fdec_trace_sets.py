"""Measured page-set traces: validate the last estimated variable (temporal+Scout UNION).

The deadline model used a SCALAR prefetch hit-rate (0.76 temporal measured, 0.92 Scout
measured, 0.96 union ESTIMATED) and an analytic overfetch-FP guess. This captures the actual
page SETS per (decode token, layer) so every prefetch number is measured, not estimated:

  exact[t,L]    = the top-B pages the exact query needs (the demand)
  temporal[t,L] = exact[t-1,L]              (reuse previous token's pages -- token-ahead)
  scout[t,L]    = top-(overfetch x B) pages from q_hat[L]=RoPE(W_Q[L](RMSNorm[L](h_{L-1})))
                  (predict THIS layer's query from the one-layer-stale residual -- layer-ahead)
  union[t,L]    = temporal U scout                                  (what a real prefetcher issues)

It then reports, over all (t>=1, L), on natural-text AND retrieval prompts:
  - recall of temporal / scout / union vs exact (the real union hit-rate)
  - false-positive overfetch: union minus exact (pages fetched but not needed)
  - a UNION-FETCHED residency sim (LRU cache admits union U exact, the REAL pollution): per
    (t,L) prefetch-new reads (overlapped) and sync misses (stall), HBM evictions/token
  - a charged TPOT using those MEASURED sync/prefetch counts (no scalar hit-rate, no FP guess)

If the measured union recall and charged TPOT still beat ~3.2-3.3 ms p99, build the trace-replay
transport prototype next; if not, stop. Dumps the per-(t,L) sync/prefetch matrix to JSON.
"""

import argparse
import json
from collections import OrderedDict

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

PAGE = 32
_PV = {"B": 64, "local": 64, "sink": 4, "of": 1.5, "orig": None}
_H = {}
_LAYER = {}
_ROT = {"emb": None}
_EXACT = []  # per step: {layer: frozenset(exact older-page ids)}
_SCOUT = []  # per step: {layer: frozenset(scout top-(of*B) older-page ids)}

BURST = [
    (4, 2.85, 83),
    (8, 4.82, 101),
    (12, 6.22, 119),
    (16, 7.14, 138),
    (20, 8.29, 150),
    (32, 8.76, 229),
    (64, 10.45, 389),
]


def burst_bw(d):
    if d <= BURST[0][0]:
        return BURST[0][1]
    if d >= BURST[-1][0]:
        return BURST[-1][1]
    for (d0, b0, _), (d1, b1, _) in zip(BURST, BURST[1:]):
        if d0 <= d <= d1:
            return b0 + (b1 - b0) * (d - d0) / (d1 - d0)
    return BURST[-1][1]


def topset(pm, npg, k, local, sink):
    resident = torch.zeros(npg, dtype=torch.bool, device=pm.device)
    if sink:
        resident[:sink] = True
    if local:
        resident[npg - local :] = True
    masked = pm.masked_fill(resident, float("-inf"))
    k = min(k, int(resident.numel() - resident.sum().item()))
    if k <= 0:
        return frozenset()
    return frozenset(masked.topk(k).indices.tolist())


def harness_attn(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
    if q.shape[2] > 1:
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
    B, local, sink, of = _PV["B"], _PV["local"], _PV["sink"], _PV["of"]

    def pagemass(query):
        aw = torch.matmul(query.float(), ks.float().transpose(2, 3)) * scaling
        if attention_mask is not None:
            aw = aw + attention_mask[:, :, :, :Tk].float()
        aw = torch.softmax(aw, dim=-1)
        pm = F.pad(aw, (0, pad)).view(1, aw.shape[1], 1, npg, PAGE).sum(-1)
        return pm.sum(1)[0, 0, :], aw

    exact_pm, aw = pagemass(q)
    exact_sel = topset(exact_pm, npg, B, local, sink)
    _EXACT[-1][L] = exact_sel

    if (L - 1) in _H and L >= 1:
        qa = module.q_proj(_LAYER[L].input_layernorm(_H[L - 1]))
        D, Hq = q.shape[-1], q.shape[1]
        qa = qa.view(1, 1, Hq, D).transpose(1, 2)
        pos = torch.arange(Tk - 1, Tk, device=q.device).unsqueeze(0)
        c, s = _ROT["emb"](qa, pos)
        qa, _ = apply_rotary_pos_emb(qa, qa, c, s)
        scout_pm, _ = pagemass(qa)
        _SCOUT[-1][L] = topset(scout_pm, npg, int(round(of * B)), local, sink)

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
                if hs is not None and hs.shape[1] == 1:
                    _H[idx] = hs.detach()

            return pre

        dl.register_forward_pre_hook(mk(i), with_kwargs=True)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else 0.0


def build_retrieval_prompt(tok, ctx, n_needles=8):
    # filler with planted "fact N: code is XXXXX" needles spread through the context, then a
    # question -- decode attends back to the planted lines (sharp retrieval attention).
    filler = "The weather report noted mild conditions across the region throughout the day. "
    base = filler * 4000
    ids = tok(base)["input_ids"]
    needles = []
    out = []
    step = max(1, len(ids) // (n_needles + 1))
    for i in range(n_needles):
        code = 10000 + i * 1111
        needle = tok(f" Important fact {i}: the secret code is {code}. ")["input_ids"]
        needles.append((i, code))
        out += ids[i * step : (i + 1) * step] + needle
    out += tok(" What are all the secret codes mentioned above? The codes are")[
        "input_ids"
    ]
    return out[:ctx]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--overfetch", type=float, default=1.5)
    ap.add_argument("--local-pages", type=int, default=64)
    ap.add_argument("--sink-pages", type=int, default=4)
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--page-kb", type=float, default=16.0)
    ap.add_argument("--capacity-mult", type=int, default=2, help="LRU cap = mult x B")
    ap.add_argument("--prompt", choices=["natural", "retrieval"], default="natural")
    ap.add_argument("--prefill-chunk", type=int, default=8192)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dump-trace", default=None)
    # charged-deadline knobs (same as fdec_deadline --charged)
    ap.add_argument("--pcie-gbps", type=float, default=20.0)
    ap.add_argument("--pcie-lat-us", type=float, default=12.0)
    ap.add_argument("--kernel-tax-us", type=float, default=10.0)
    ap.add_argument("--selector-tax-us", type=float, default=5.0)
    args = ap.parse_args()
    global PAGE
    PAGE = args.page_size
    _PV.update(
        B=args.B, local=args.local_pages, sink=args.sink_pages, of=args.overfetch
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device)
    model.eval()
    install(model)

    if args.prompt == "retrieval":
        toks = build_retrieval_prompt(tok, args.ctx)
    else:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        txt = "\n".join(t for t in ds["text"][:300000] if t and not t.isspace())
        toks = tok(txt)["input_ids"]
    ids = torch.tensor(toks[: args.ctx]).unsqueeze(0).to(args.device)
    print(
        f"[trace-sets] prompt={args.prompt} ctx={ids.shape[1]} gen={args.gen} B={args.B} "
        f"overfetch={args.overfetch} page={PAGE} cap={args.capacity_mult}xB"
    )

    pkv = None
    for i in range(0, ids.shape[1], args.prefill_chunk):
        out = model(
            ids[:, i : i + args.prefill_chunk], past_key_values=pkv, use_cache=True
        )
        pkv = out.past_key_values
    cur = out.logits[:, -1:].argmax(-1)
    for _ in range(args.gen):
        _H.clear()
        _EXACT.append({})
        _SCOUT.append({})
        out = model(cur, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        cur = out.logits[:, -1:].argmax(-1)

    layers = sorted(_EXACT[0].keys())
    nL = len(layers)
    cap = args.capacity_mult * args.B
    kb = args.page_kb

    # ---- recall + false-positive overfetch (measured, not estimated) ----
    rec = {"temporal": [], "scout": [], "union": []}
    fp_overfetch = (
        []
    )  # |union \ exact| / B  (false-positive pages per layer, vs budget)
    for t in range(1, len(_EXACT)):
        for L in layers:
            ex = _EXACT[t].get(L)
            if not ex:
                continue
            temp = _EXACT[t - 1].get(L, frozenset())
            sc = _SCOUT[t].get(L, frozenset())
            un = temp | sc
            rec["temporal"].append(len(ex & temp) / len(ex))
            rec["scout"].append(len(ex & sc) / len(ex) if sc else 0.0)
            rec["union"].append(len(ex & un) / len(ex))
            fp_overfetch.append(len(un - ex) / args.B)

    # ---- union-fetched residency (REAL pollution): admit union U exact, LRU evict ----
    caches = {L: OrderedDict() for L in layers}
    sync_mat, hidden_mat = [], []  # per token: {layer: count}
    evict_per_tok = []
    for t in range(1, len(_EXACT)):
        srow, hrow = {}, {}
        ev = 0
        for L in layers:
            ex = _EXACT[t].get(L)
            if ex is None:
                continue
            temp = _EXACT[t - 1].get(L, frozenset())
            sc = _SCOUT[t].get(L, frozenset())
            un = temp | sc
            c = caches[L]
            resident_before = set(c.keys())
            prefetch_new = len(un - resident_before)  # overlapped reads
            sync_miss = len(ex - resident_before - un)  # needed, unpredicted -> stall
            hrow[L] = prefetch_new
            srow[L] = sync_miss
            for p in un | ex:  # admit everything fetched into HBM
                c[p] = t
                c.move_to_end(p)
            while len(c) > cap:
                c.popitem(last=False)
                ev += 1
        sync_mat.append(srow)
        hidden_mat.append(hrow)
        evict_per_tok.append(ev)

    # ---- charged TPOT using MEASURED sync/hidden ----
    page_bytes = kb * 1024
    w = 7.0e9
    k_bytes = nL * 4 * 128 * 1 * args.ctx
    v_bytes = nL * cap * page_bytes
    floor = (w + k_bytes + v_bytes) / (3350.0e9) * 1e3
    layer_tax = (args.kernel_tax_us + args.selector_tax_us) / 1e3
    pcie_lat = args.pcie_lat_us / 1e3
    pcie_bw = args.pcie_gbps
    floor_c = floor + layer_tax * nL
    cpl = floor / nL + layer_tax

    def eff_bw(d):
        return min(burst_bw(d), pcie_bw)

    tpot, ssd_mb = [], []
    for t in range(len(sync_mat)):
        stall = 0.0
        bytes_t = 0
        for L in layers:
            sync = sync_mat[t].get(L, 0)
            hidden = hidden_mat[t].get(L, 0)
            bytes_t += (sync + hidden) * page_bytes
            if sync > 0:
                stall += (
                    sync * page_bytes / (eff_bw(max(1, sync)) * 1e9) * 1e3 + pcie_lat
                )
            if hidden > 0:
                rt = hidden * page_bytes / (eff_bw(max(1, hidden)) * 1e9) * 1e3
                stall += max(0.0, rt - cpl)
        tpot.append(floor_c + stall)
        ssd_mb.append(bytes_t / 1024 / 1024)

    print(
        f"\nmeasured recall (frac of exact pages the predictor covers), n={len(rec['union'])}:"
    )
    for name in ("temporal", "scout", "union"):
        xs = rec[name]
        print(
            f"  {name:<9} mean={sum(xs)/len(xs):.3f} p10={pct(xs,10):.3f} "
            f"p50={pct(xs,50):.3f} p90={pct(xs,90):.3f}"
        )
    print(
        f"false-positive overfetch (union pages not needed, /B): "
        f"mean={sum(fp_overfetch)/len(fp_overfetch):.2f} p90={pct(fp_overfetch,90):.2f}"
    )
    sync_tok = [sum(r.values()) / nL for r in sync_mat]
    hid_tok = [sum(r.values()) / nL for r in hidden_mat]
    print(f"\nunion-fetched residency (cap {args.capacity_mult}xB={cap} pg/layer):")
    print(
        f"  sync misses/layer (STALL): p50={pct(sync_tok,50):.2f} p90={pct(sync_tok,90):.2f} "
        f"p99={pct(sync_tok,99):.2f}"
    )
    print(
        f"  prefetch-new/layer (overlapped): p50={pct(hid_tok,50):.1f} p99={pct(hid_tok,99):.1f}"
    )
    print(
        f"  HBM evictions/token: p50={pct(evict_per_tok,50):.0f} p99={pct(evict_per_tok,99):.0f}"
    )
    print(
        f"  new SSD MB/token: p50={pct(ssd_mb,50):.2f} p90={pct(ssd_mb,90):.2f} "
        f"p99={pct(ssd_mb,99):.2f}"
    )
    budget = 3.0
    over = sum(1 for x in tpot if x > budget) / len(tpot)
    bar = 3.3
    overbar = sum(1 for x in tpot if x > bar) / len(tpot)
    print(
        f"\nCHARGED TPOT (measured sync/hidden, floor {floor_c:.2f} ms): "
        f"p50={pct(tpot,50):.2f} p90={pct(tpot,90):.2f} p99={pct(tpot,99):.2f} ms"
    )
    print(
        f"  over 3.0 ms: {over*100:.0f}%   |   over {bar} ms (the stop-bar): {overbar*100:.0f}%"
    )
    print(
        f"  VERDICT: {'PASS -> build trace-replay transport prototype' if pct(tpot,99) <= bar else 'FAIL -> stop, do not fork'}"
        f" (p99 {pct(tpot,99):.2f} vs {bar} ms bar)"
    )

    if args.dump_trace:
        json.dump(
            {
                "prompt": args.prompt,
                "ctx": args.ctx,
                "gen": args.gen,
                "B": args.B,
                "overfetch": args.overfetch,
                "page_kb": kb,
                "n_layers": nL,
                "cap_mult": args.capacity_mult,
                "recall": {k: (sum(v) / len(v)) for k, v in rec.items()},
                "fp_overfetch_mean": sum(fp_overfetch) / len(fp_overfetch),
                "sync_matrix": [
                    {str(L): r.get(L, 0) for L in layers} for r in sync_mat
                ],
                "hidden_matrix": [
                    {str(L): r.get(L, 0) for L in layers} for r in hidden_mat
                ],
                "evict_per_tok": evict_per_tok,
                "tpot_p99": pct(tpot, 99),
            },
            open(args.dump_trace, "w"),
        )
        print(f"[dump] {args.dump_trace}")


if __name__ == "__main__":
    main()
