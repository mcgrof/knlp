#!/usr/bin/env python3
"""Standalone KV-budget bench: full / SnapKV / Quest / RocketKV.

Point it at any HF causal LM and a list of KV budgets and it reports task
accuracy on a needle-in-haystack retrieval task plus the resident-HBM and
per-step read cost of each method. Runs on the validated masked-decode rig: the
full cache is kept and a per-(layer, kv-head) attention mask simulates
eviction/selection -- proven bit-identical to physical cache slicing in fp32, so
the accuracy numbers are faithful while the footprint is accounted analytically.

The budget B is the number of context pages a method may attend to per decode
step, on top of the always-on sink and recent windows. Each method spends B
differently -- that spread IS the comparison:

  full      reference: every page resident and read.
  snapkv    permanent eviction. Keep the top-B pages once, scored by SnapKV
            obs-window attention mass (1D max-pooled). resident = B, no cold
            reads. A MEMORY method -- evicted pages are gone for good.
  quest     dynamic read. Full cache stays resident; each step read the top-B
            pages by per-page min/max box score for THIS step's query.
            resident = 100%, read = B. A BANDWIDTH method.
  rocketkv  two-stage. A SnapKV permanent hot tier of H pages (stage 1) plus,
            each step, the top-(B-H) cold pages by SparQ box-on-top-|q|
            coordinates (stage 2). resident = H, cold-read = B-H. A
            MEMORY+BANDWIDTH method -- the only one that can RECOVER a page it
            did not bank, by paying a cold read.

So at equal budget B every method attends the same B+sink+recent pages (same
accuracy-relevant working set) but pays differently: snapkv the least resident
HBM but cannot recover a wrongly-evicted page; quest no memory saving at all;
rocketkv banks only H but buys back accuracy with cold reads. The headline
question the bench answers: does rocketkv's cold read recover enough accuracy
over snapkv at the same resident footprint to justify a cold tier?

Everything is page-granular so the four methods share one budget axis. No
calibration phase is needed (the box/SparQ scores come straight from the keys).

Example:
  python3 rocketkv_eval.py --model Qwen/Qwen2.5-1.5B \\
      --length 4096 --needles 8 --budget 4,8,16,32 --eval 80 \\
      --out /tmp/rocketkv_qwen.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from niah_task import build_context, load_filler_sentences
from niah_evict_perhead import answer_block_indices, get_keys
from diag_perhead_oracle import capture_q, get_values

NEG = -1e9
_B = {"on": False}


def budget_attention(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    """Patched eager attention. When a per-step budget state is installed it
    builds a per-(layer,kv-head) page-keep mask and applies it as an additive
    bias before softmax."""
    g = module.num_key_value_groups
    ks = key.repeat_interleave(g, dim=1)
    vs = value.repeat_interleave(g, dim=1)
    aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
    if attention_mask is not None:
        aw = aw + attention_mask[:, :, :, : ks.shape[-2]]
    st = getattr(module, "_b", None)
    if _B["on"] and st is not None and query.shape[2] == 1:
        Bn, Hq, _, Kc = aw.shape
        Hkv = key.shape[1]
        T, NB = st["T"], st["NB"]
        method = st["method"]
        q = query[:, :, 0, :].view(Bn, Hkv, g, -1).mean(2).float()  # [B,Hkv,D]
        selp = st["selectable"]  # [NB] bool: pages eligible to spend budget on
        if method == "snapkv":
            keep = st["hot"].unsqueeze(0).expand(Bn, -1, -1).clone()  # [B,Hkv,NB]
        elif method == "quest":
            qk = q.unsqueeze(2)
            ub = torch.where(qk > 0, qk * st["kmax"], qk * st["kmin"]).sum(-1)
            ub = ub.masked_fill(~selp.view(1, 1, NB), NEG)
            top = ub.topk(min(st["B"], NB), dim=-1).indices
            keep = torch.zeros(Bn, Hkv, NB, dtype=torch.bool, device=aw.device)
            keep.scatter_(2, top, True)
        else:  # rocketkv: SnapKV hot tier + SparQ cold read
            k1 = st["k1"]
            sel = q.abs().topk(min(k1, q.shape[-1]), dim=-1).indices
            qs = torch.gather(q, 2, sel)
            kmx = torch.gather(
                st["kmax"].unsqueeze(0).expand(Bn, -1, -1, -1),
                3,
                sel.unsqueeze(2).expand(-1, -1, NB, -1),
            )
            kmn = torch.gather(
                st["kmin"].unsqueeze(0).expand(Bn, -1, -1, -1),
                3,
                sel.unsqueeze(2).expand(-1, -1, NB, -1),
            )
            qe = qs.unsqueeze(2)
            ub = torch.where(qe > 0, qe * kmx, qe * kmn).sum(-1)
            hot = st["hot"].unsqueeze(0).expand(Bn, -1, -1)
            ub = ub.masked_fill(hot | ~selp.view(1, 1, NB), NEG)  # cold, non-hot only
            ncold = max(0, st["B"] - st["H"])
            keep = hot.clone()
            if ncold > 0:
                top = ub.topk(min(ncold, NB), dim=-1).indices
                keep.scatter_(2, top, True)
        # expand page-keep -> token-keep, force sink + recent + generated
        tk = keep.gather(2, st["tok2page"].view(1, 1, T).expand(Bn, Hkv, T))
        tk[:, :, : st["sink"]] = True
        tk[:, :, st["recent"] :] = True
        keepq = tk.repeat_interleave(g, dim=1)
        z = aw.new_zeros(())
        nz = aw.new_full((), NEG)
        bias = aw.new_zeros(Bn, Hq, Kc)
        bias[:, :, :T] = torch.where(keepq, z, nz)
        aw = aw + bias.view(Bn, Hq, 1, Kc)
    aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(aw, vs).transpose(1, 2).contiguous()
    return out, aw


def install(model):
    model.config._attn_implementation = "eager"
    mods = set()
    for m in model.modules():
        if hasattr(m, "num_key_value_groups") and hasattr(m, "layer_idx"):
            m._b = None
            mods.add(type(m).__module__)
    for mn in mods:
        mod = sys.modules.get(mn)
        if mod is not None and hasattr(mod, "eager_attention_forward"):
            mod.eager_attention_forward = budget_attention


def attn_mods(model):
    return [
        m
        for m in model.modules()
        if hasattr(m, "num_key_value_groups") and hasattr(m, "layer_idx")
    ]


def clone_cache(past):
    from transformers import DynamicCache

    new = DynamicCache()
    for i, layer in enumerate(past.layers):
        new.update(layer.keys.clone(), layer.values.clone(), i)
    return new


@torch.no_grad()
def decode(model, last_id, past, device, T, max_new):
    ids = torch.tensor([[int(last_id)]], device=device)
    cur, gen = past, []
    _B["on"] = True
    for s in range(max_new):
        out = model(
            ids,
            past_key_values=cur,
            position_ids=torch.tensor([[T + s]], device=device),
            cache_position=torch.tensor([T + s], device=device),
            use_cache=True,
        )
        nxt = int(out.logits[0, -1].argmax())
        gen.append(nxt)
        cur = out.past_key_values
        ids = torch.tensor([[nxt]], device=device)
    _B["on"] = False
    return gen


def page_box(kl, tok2page, Hkv, NB, d):
    """Per-page min/max of the keys (Quest box)."""
    kmin = kl.new_full((Hkv, NB, d), 1e30)
    kmax = kl.new_full((Hkv, NB, d), -1e30)
    t2 = tok2page.view(1, -1, 1).expand(Hkv, kl.shape[1], d)
    kmin.scatter_reduce_(1, t2, kl, reduce="amin", include_self=True)
    kmax.scatter_reduce_(1, t2, kl, reduce="amax", include_self=True)
    return kmin, kmax


def snap_hot(qobs, kl, tok2page, Hkv, g, NB, d, nrecent, n_keep, pool, selectable):
    """SnapKV page importance from obs-window attention mass (1D max-pooled),
    then the top-n_keep pages restricted to the selectable set (sink/recent are
    kept separately, so the budget must not be wasted on them)."""
    qo = qobs.view(Hkv, g, nrecent, -1)  # [Hkv,g,W,D]
    sc = torch.einsum("hgwd,htd->hgwt", qo, kl) / math.sqrt(d)
    imp = sc.softmax(dim=-1).sum(dim=(1, 2))  # [Hkv,T] obs-attention mass
    if pool > 1:
        imp = F.max_pool1d(
            imp.unsqueeze(1), kernel_size=pool, stride=1, padding=pool // 2
        ).squeeze(1)[:, : imp.shape[1]]
    pimp = imp.new_full((Hkv, NB), 0.0)
    pimp.scatter_reduce_(
        1,
        tok2page.view(1, -1).expand(Hkv, imp.shape[1]),
        imp,
        reduce="sum",
        include_self=True,
    )
    pimp = pimp.masked_fill(~selectable.view(1, NB), -1e30)
    hotp = pimp.topk(min(n_keep, NB), dim=-1).indices
    hot = torch.zeros(Hkv, NB, dtype=torch.bool, device=kl.device)
    hot.scatter_(1, hotp, True)
    return hot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--needles", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument(
        "--budget", default="4,8,16,32", help="pages/step beyond sink+recent"
    )
    ap.add_argument("--eval", type=int, default=80)
    ap.add_argument(
        "--k1", type=int, default=32, help="SparQ top-|q| coords (rocketkv)"
    )
    ap.add_argument(
        "--rocket-hot-frac",
        type=float,
        default=0.5,
        help="fraction of B kept as hot tier",
    )
    ap.add_argument("--snap-pool", type=int, default=13, help="SnapKV max-pool kernel")
    ap.add_argument("--sink", type=int, default=1, help="sink pages (always resident)")
    ap.add_argument(
        "--recent", type=int, default=8, help="recent pages (always resident)"
    )
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument(
        "--methods", default="full,snapkv,quest,rocketkv", help="comma-separated"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager"
    ).to(device)
    model.eval()
    torch.set_grad_enabled(False)
    install(model)
    mods = {m.layer_idx: m for m in attn_mods(model)}
    bs = args.block_size
    d = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )
    Hq = model.config.num_attention_heads
    Hkv = getattr(model.config, "num_key_value_heads", Hq)
    g = Hq // Hkv
    budgets = [int(x) for x in args.budget.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    sents = load_filler_sentences(args.seed)
    rng = random.Random(args.seed)
    W = args.recent * bs  # obs/recent window in tokens

    acc = {m: {B: [0, 0] for B in budgets} for m in methods}
    n_used = 0
    NB_last = 0
    for ei in range(args.eval):
        text, spans, needles_l, (qk, qv) = build_context(
            tok, args.length, args.needles, sents, rng
        )
        ids, ans = answer_block_indices(tok, text, qv, spans, bs, 10**9)
        ids_t = torch.tensor(ids)[:-1]
        T = ids_t.shape[0]
        NB = (T + bs - 1) // bs
        if not ans or max(ans) >= NB:
            continue
        recent = list(range(max(0, T - W), T))
        last_id = ids[-1]
        qs, past = capture_q(model, ids_t, sorted(set(recent + [T - 1])), device)
        keys = get_keys(past)
        tok2page = torch.arange(T, device=device) // bs
        ridx = torch.tensor(list(range(len(recent))), device=device)
        # pages eligible to spend budget on = not sink, not recent
        selectable = torch.ones(NB, dtype=torch.bool, device=device)
        selectable[: args.sink] = False
        selectable[max(0, NB - args.recent) :] = False
        # precompute per-layer box + (for snapkv/rocketkv) obs queries
        layer = []
        for li in range(len(keys)):
            kl = keys[li][0].float()
            kmin, kmax = page_box(kl, tok2page, Hkv, NB, d)
            qobs = qs[li][
                :, ridx, :
            ]  # [Hkv*g? ] -> shape [Hq?]; capture_q gives [Hkv,...]
            layer.append((kl, kmin, kmax, qobs))
        n_used += 1
        NB_last = NB
        for B in budgets:
            for method in methods:
                if method == "full":
                    for m in mods.values():
                        m._b = None
                else:
                    H = (
                        max(1, round(args.rocket_hot_frac * B))
                        if method == "rocketkv"
                        else 0
                    )
                    for li, m in mods.items():
                        kl, kmin, kmax, qobs = layer[li]
                        if method in ("snapkv", "rocketkv"):
                            nkeep = B if method == "snapkv" else H
                            hot = snap_hot(
                                qobs,
                                kl,
                                tok2page,
                                Hkv,
                                g,
                                NB,
                                d,
                                len(recent),
                                nkeep,
                                args.snap_pool,
                                selectable,
                            )
                        else:
                            hot = None
                        m._b = dict(
                            method=method,
                            B=B,
                            H=H,
                            k1=args.k1,
                            kmin=kmin,
                            kmax=kmax,
                            hot=hot,
                            selectable=selectable,
                            tok2page=tok2page,
                            NB=NB,
                            T=T,
                            sink=args.sink * bs,
                            recent=max(0, T - args.recent * bs),
                        )
                gen = decode(model, last_id, clone_cache(past), device, T, args.max_new)
                ok = qv in tok.decode(gen)
                acc[method][B][0] += int(ok)
                acc[method][B][1] += 1
        for m in mods.values():
            m._b = None
        del past, layer
        torch.cuda.empty_cache()
        if (ei + 1) % 10 == 0:
            print(f"  {n_used}/{args.eval}", flush=True)

    NB = NB_last
    extra = args.sink + args.recent  # always-resident pages

    def footprint(method, B):
        H = max(1, round(args.rocket_hot_frac * B))
        if method == "full":
            return NB, NB, 0
        if method == "snapkv":
            return B + extra, B + extra, 0  # resident, read, cold-read
        if method == "quest":
            return NB, B + extra, 0  # full resident, read from resident
        # rocketkv
        return H + extra, B + extra, max(0, B - H)

    res = {
        "model": args.model,
        "head_dim": d,
        "block_size": bs,
        "NB": NB,
        "eval_used": n_used,
        "budgets": budgets,
        "sink": args.sink,
        "recent": args.recent,
        "snap_pool": args.snap_pool,
        "rocket_hot_frac": args.rocket_hot_frac,
        "k1": args.k1,
        "results": {},
    }
    print(f"\n[{args.model}] NB={NB} pages, sink={args.sink} recent={args.recent}")
    print(
        f"{'method':9s} {'budget':>7s} {'acc':>6s} {'resid%':>7s} "
        f"{'read/step':>10s} {'cold/step':>10s}"
    )
    for method in methods:
        res["results"][method] = {}
        for B in budgets:
            a = acc[method][B][0] / max(1, acc[method][B][1])
            resid, read, cold = footprint(method, B)
            residpct = 100.0 * resid / NB
            res["results"][method][B] = {
                "acc": a,
                "resident_pages": resid,
                "resident_pct": residpct,
                "read_per_step": read,
                "cold_read_per_step": cold,
            }
            print(
                f"{method:9s} {B:7d} {a:6.2f} {residpct:6.1f}% "
                f"{read:10d} {cold:10d}"
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
