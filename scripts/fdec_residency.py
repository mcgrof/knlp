"""Stateful V-page residency simulator: SSD MISSES, not selected pages.

The byte-budget work counted every SELECTED page as a fetch. But with ~76% temporal
overlap, a PERSISTENT HBM residency of fetched pages means most of a token's demand is
already on-GPU -- the real SSD reads are only the MISSES (new pages). This generates the
per-(token, layer) exact page-demand trace from a real decode run, then simulates per-layer
LRU residency caches at several capacities and reports miss pages/bytes per token
(p50/p90/p99) -- the number that actually hits the SSD. If misses are ~0.24x the selected
set, 128K may move from marginal to viable. Compute-only (no fetch); pairs with the
measured SSD bandwidth for the deadline model (next brick).
"""

import argparse
import sys
from collections import OrderedDict

import torch
import torch.nn.functional as F

PAGE = 128
_PV = {"B": 16, "local": 16, "sink": 1, "orig": None}
_STEP = {}


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
        sel = tot.topk(B, dim=-1).indices[0]
        # record ABSOLUTE older-page indices (stable as context grows -> the SSD-resident
        # region); these are the demand for the residency cache
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


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def sim_residency(history, layers, capacity, per_layer=False):
    # per-layer LRU cache; return list of miss counts per token (summed across layers).
    # per_layer=True also returns the per-token per-layer miss matrix (for the deadline
    # model, which needs the layer-serial fetch pattern, not just the token total).
    caches = {L: OrderedDict() for L in layers}
    miss_per_tok = []
    matrix = []
    for step in range(len(history)):
        misses = 0
        row = {}
        for L in layers:
            if L not in history[step]:
                continue
            demand = history[step][L]
            c = caches[L]
            m = len(demand - set(c.keys()))
            misses += m
            row[L] = m
            for p in demand:
                c[p] = step
                c.move_to_end(p)
            while len(c) > capacity:
                c.popitem(last=False)
        miss_per_tok.append(misses)
        matrix.append(row)
    if per_layer:
        return miss_per_tok, matrix
    return miss_per_tok


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--local-pages", type=int, default=64)
    ap.add_argument("--sink-pages", type=int, default=4)
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument(
        "--page-kb", type=float, default=16.0, help="bytes/page for reporting"
    )
    ap.add_argument("--prefill-chunk", type=int, default=8192)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--dump-trace",
        default=None,
        help="JSON path: raw per-token demand trace + per-layer miss matrices "
        "(at 1x/2x/4x capacity) for the deadline model",
    )
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
        f"[residency] ctx={args.ctx} gen={args.gen} B={args.B} page={PAGE} ({args.page_kb}KB)"
    )

    pkv = None
    for i in range(0, args.ctx, args.prefill_chunk):
        out = model(
            ids[:, i : i + args.prefill_chunk], past_key_values=pkv, use_cache=True
        )
        pkv = out.past_key_values
    cur = out.logits[:, -1:].argmax(-1)
    history = []
    for _ in range(args.gen):
        _STEP.clear()
        out = model(cur, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        cur = out.logits[:, -1:].argmax(-1)
        history.append(dict(_STEP))

    layers = sorted(history[0].keys())
    nL = len(layers)
    sel_per_tok = (
        args.B * nL
    )  # selected pages/token across all layers (the old "fetch")
    kb = args.page_kb

    def line(name, miss):
        # skip first token (cold cache)
        m = miss[1:] if len(miss) > 1 else miss
        print(
            f"  {name:<26} miss p50={pct(m,50)/nL:.1f} p90={pct(m,90)/nL:.1f} "
            f"p99={pct(m,99)/nL:.1f} pages/layer  |  "
            f"new SSD MB/tok p50={pct(m,50)*kb/1024:.2f} p99={pct(m,99)*kb/1024:.2f}"
        )

    print(
        f"\nselected (old baseline) = {args.B} pages/layer/tok = "
        f"{sel_per_tok*kb/1024:.2f} MB/tok total ({args.B*kb/1024:.3f} MB/layer)"
    )
    print("residency miss rate (what actually hits SSD), per-layer LRU cache:")
    matrices = {}
    for mult in (1, 2, 4):
        miss, mat = sim_residency(history, layers, mult * args.B, per_layer=True)
        matrices[mult] = mat
        line(f"capacity {mult}xB ({mult*args.B} pg/layer)", miss)

    if args.dump_trace:
        import json

        # per-token list of {layer: [sorted absolute page ids]}; the deadline model
        # replays this against any predictor/cache policy without a GPU.
        demand = [
            {str(L): sorted(history[s][L]) for L in history[s]}
            for s in range(len(history))
        ]
        miss_mat = {
            str(mult): [{str(L): mat[s][L] for L in mat[s]} for s in range(len(mat))]
            for mult, mat in matrices.items()
        }
        with open(args.dump_trace, "w") as f:
            json.dump(
                {
                    "ctx": args.ctx,
                    "gen": args.gen,
                    "B": args.B,
                    "page_kb": kb,
                    "n_layers": nL,
                    "layers": layers,
                    "demand": demand,
                    "miss_matrix": miss_mat,
                },
                f,
            )
        print(f"[dump] wrote trace -> {args.dump_trace}")


if __name__ == "__main__":
    main()
