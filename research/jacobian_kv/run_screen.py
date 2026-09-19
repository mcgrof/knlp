# SPDX-License-Identifier: GPL-2.0
"""Stage 0: does a receiver-sensitive second moment rank real cache damage?

The free kill test.  Nothing downstream in the plan happens if this fails, and
failing it cheaply is a success of the plan rather than a disappointment.

What it does, in one sentence: build a bank of controlled KV-cache errors whose
*size* is held fixed within groups, measure what each one actually costs the
frozen receiver by exact teacher-forced KL, then ask which candidate metric
ranks those costs best.

Six metrics are compared, per tensor kind and per model:

  ``mse``              raw cache reconstruction error -- the metric to beat
  ``wo``               the per-head output-projection metric already tested and
                       already beaten by plain SVD in this program
  ``attn_local``       CacheBridge-style attention-local sensitivity
  ``meanj``            the signed mean-Jacobian control, built by averaging the
                       VJP across contexts *before* squaring
  ``jtfj``             the proposed second moment, fitted on calibration
                       contexts and applied to held-out ones -- the deployable
                       arm, and the one the gate judges
  ``jtfj_oracle``      the same second moment computed on the scored context
                       itself.  Not deployable: it needs a backward pass at
                       scoring time.  It is the ceiling, and it is what
                       separates "the receiver signal does not exist" from "it
                       exists but does not survive being averaged into a global
                       metric" -- two very different negative results.

Two views of every correlation are reported.  Pooled across groups, error size
varies and ``mse`` is a strong baseline.  Within a matched-norm group, size is
constant, ``mse`` has exactly zero ranking power, and what remains is the only
question that matters: given the bytes already spent, does the metric know
which direction to spend them in.

Env: HIP_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
Run:
    python -m research.jacobian_kv.run_screen \\
        --model Qwen/Qwen3-0.6B --ctx 512 --seed 0 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.jacobian_kv import artifacts as art  # noqa: E402
from research.jacobian_kv.evaluate import (  # noqa: E402
    argmax_flip_rate,
    blocked_spearman,
    cluster_bootstrap_blocked_delta,
    delta_nll,
    evaluate_gate,
    kendall_distance,
    method_ranking_distance,
    spearman,
    teacher_forced_kl,
)
from research.jacobian_kv.perturb import build_bank, matched_norm_report  # noqa: E402
from research.jacobian_kv.sensitivity import (  # noqa: E402
    fisher_probe,
    freeze_model,
    score_attention_local,
    score_mse,
    vjp_from_probe,
)

PROBE_COUNTS = (1, 2, 4, 8)
MAX_PROBES = max(PROBE_COUNTS)

# The signed mean-Jacobian control estimates E_x[J]' F E_x[J] from probe
# directions held fixed across contexts, so the rank of its estimate is capped
# by the number of those directions.  The candidate, drawing fresh probes per
# context, accumulates n_contexts * MAX_PROBES directions and is effectively
# full rank.  Giving the control only MAX_PROBES directions would decide the
# plan's central contrast by estimator budget rather than by the mathematics,
# so the control gets its own larger budget and the candidate is additionally
# reported truncated to the control's rank for a matched head-to-head.
MEANJ_PROBES = 32


# ---------------------------------------------------------------------------
# cache helpers
#
# DynamicCache.update rebinds (`self.keys = torch.cat([...])`) rather than
# mutating in place, so one set of base tensors can back thousands of
# perturbation forwards with a single copy for the perturbed layer.  Cloning
# the whole cache per perturbation would move ~120 MiB each time for nothing.
# ---------------------------------------------------------------------------


def make_cache(keys, values):
    from transformers import DynamicCache

    c = DynamicCache()
    for i, (k, v) in enumerate(zip(keys, values)):
        c.update(k, v, i)
    return c


def base_cache_tensors(model, input_ids):
    from transformers import DynamicCache

    c = DynamicCache()
    with torch.no_grad():
        model(input_ids=input_ids, past_key_values=c, use_cache=True)
    keys = [layer.keys.detach() for layer in c.layers]
    values = [layer.values.detach() for layer in c.layers]
    return keys, values


class QKCapture:
    """Record post-RoPE queries and keys at chosen layers, then restore.

    ``eager_attention_forward`` is a module-level function in transformers
    4.56, dispatched by name rather than through the attention registry, so a
    temporary rebinding is the way to see the tensors attention actually
    consumes: post-RoPE, post any query/key normalisation, and before the
    grouped-query expansion.  Restoring in ``__exit__`` matters because the
    binding is process global.
    """

    def __init__(self, model, layers):
        self.layers = set(layers)
        self.mod = sys.modules[type(model.model.layers[0].self_attn).__module__]
        self.orig = self.mod.eager_attention_forward
        self.q = {}
        self.k = {}

    def __enter__(self):
        orig, want, q, k = self.orig, self.layers, self.q, self.k

        def patched(module, query, key, value, attention_mask, scaling, **kw):
            idx = getattr(module, "layer_idx", None)
            if idx in want and idx not in q:
                q[idx] = query.detach()
                k[idx] = key.detach()
            return orig(module, query, key, value, attention_mask, scaling, **kw)

        self.mod.eager_attention_forward = patched
        return self

    def __exit__(self, *exc):
        self.mod.eager_attention_forward = self.orig
        return False


def prefill_with_attention(model, input_ids, layers):
    """Prompt self-attention and post-RoPE queries, both available at encode time.

    A metric meant to run when a cache is written can only use what exists
    then.  The continuation's attention over the prompt does not, so anything
    built from it is an oracle and is labelled as one.
    """
    from transformers import DynamicCache

    cache = DynamicCache()
    with QKCapture(model, layers) as cap, torch.no_grad():
        out = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
        )
    attn = {li: out.attentions[li].detach() for li in layers}
    q = {li: cap.q[li] for li in layers if li in cap.q}
    keys = [layer.keys.detach() for layer in cache.layers]
    values = [layer.values.detach() for layer in cache.layers]
    del out
    return keys, values, attn, q


def score_qk_local(error, q_post_rope, attn_probs, n_kv_heads):
    """First-order key sensitivity: the attention logit each key error moves.

    A key error acts only through the query inner product, so the natural cheap
    baseline for keys is the squared logit perturbation weighted by the
    attention mass at that position.  The output-projection metric is a
    value-space construct and says nothing about a key, which left keys with
    only one real competitor before this was added.
    """
    e = error.float()[0]  # [H_kv, T, D]
    qq = q_post_rope.float()[0]  # [H_q, Tq, D]
    a = attn_probs.float()[0]  # [H_q, Tq, Tkv]
    grp = qq.shape[0] // n_kv_heads
    qq = qq.reshape(n_kv_heads, grp, qq.shape[1], qq.shape[2])
    a = a.reshape(n_kv_heads, grp, a.shape[1], a.shape[2])[..., : e.shape[1]]
    # logit change at (kv head h, query j, key t) is q_hj . e_ht
    dlogit = torch.einsum("hgjd,htd->hgjt", qq, e)
    return float((a.pow(2) * dlogit.pow(2)).sum())


def forward_continuation(model, keys, values, cont_ids, prompt_len, want_attn=False):
    cache = make_cache(keys, values)
    total = prompt_len + cont_ids.shape[1]
    attn = torch.ones(1, total, dtype=torch.long, device=cont_ids.device)
    pos = torch.arange(prompt_len, total, device=cont_ids.device).unsqueeze(0)
    out = model(
        input_ids=cont_ids,
        attention_mask=attn,
        position_ids=pos,
        past_key_values=cache,
        use_cache=True,
        output_attentions=want_attn,
    )
    return out


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def wikitext_chunks(tok, n_chunks, ctx, cont_len, seed):
    """Disjoint fixed-length chunks from wikitext-2 test.  Offsets recorded."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    span = ctx + cont_len
    n_avail = ids.shape[0] // span
    if n_avail < n_chunks:
        raise RuntimeError(f"only {n_avail} chunks of {span} tokens available")
    rng = np.random.default_rng(seed)
    picks = rng.choice(n_avail, size=n_chunks, replace=False)
    return [(int(p), ids[p * span : (p + 1) * span].clone()) for p in sorted(picks)]


NEEDLE_FACTS = [
    ("the harbour master at Vardo", "Ingeborg Rasmussen"),
    ("the reserve fuel pump code", "four seven two nine"),
    ("the winter ferry berth number", "berth nineteen"),
    ("the lighthouse relief keeper", "Tomas Lindqvist"),
    ("the cold-store access phrase", "northern gannet"),
]


def needle_chunks(tok, n_chunks, ctx, cont_len, seed):
    """Long-range retrieval contexts: a fact planted early, asked about late.

    Pretraining-like text under-weights the one thing a cache is for, so these
    contexts make retrieval damage visible in the same divergence column.

    The filler is real, *disjoint* prose rather than one paragraph repeated.
    An earlier version tiled a single 42-token block, which made every context
    the same token stream with one fact spliced in, so a metric fitted on the
    calibration contexts was evaluated on the same text it was fitted on.  That
    corpus produced the only gate pass in a preliminary matrix, which is
    exactly the sort of result a repeated corpus manufactures.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    body_ids = tok(text, return_tensors="pt").input_ids[0].tolist()

    rng = np.random.default_rng(seed + 9001)
    span = ctx + 64
    n_avail = len(body_ids) // span
    if n_avail < n_chunks:
        raise RuntimeError(f"only {n_avail} filler spans of {span} tokens available")
    picks = rng.choice(n_avail, size=n_chunks, replace=False)

    out = []
    for i, pick in enumerate(sorted(int(p) for p in picks)):
        subject, answer = NEEDLE_FACTS[int(rng.integers(len(NEEDLE_FACTS)))]
        depth = float(rng.uniform(0.05, 0.45))
        fact = f" Note carefully: {subject} is {answer}. "
        ids_fact = tok(fact, add_special_tokens=False).input_ids
        need = ctx - len(ids_fact)
        filler = body_ids[pick * span : pick * span + need]
        at = max(0, int(depth * need))
        ids = filler[:at] + ids_fact + filler[at:]
        ids = ids[:ctx]
        q = f"\nQuestion: what is {subject}?\nAnswer: {answer}"
        ids_q = tok(q, add_special_tokens=False).input_ids
        while len(ids_q) < cont_len:
            ids_q = (
                ids_q
                + tok(f" {subject} is {answer}.", add_special_tokens=False).input_ids
            )
        out.append(
            (10_000 + pick, torch.tensor(ids + ids_q[:cont_len], dtype=torch.long))
        )
    return out


# ---------------------------------------------------------------------------
# metric fitting on calibration contexts
# ---------------------------------------------------------------------------


def fit_global_metrics(model, tok, chunks, ctx, cont_len, layers, device, seed, log):
    """Accumulate the per-(layer, kind, head) second moment and its control.

    ``jtfj`` averages ``u u'`` over contexts, probes and positions.  ``meanj``
    holds the probe directions *fixed across contexts* and averages ``u``
    before squaring, which is what builds ``E[J]' F E[J]`` -- the object that
    loses any direction cancelling across contexts.  The contrast between them
    is the plan's actual question.
    """
    acc2: dict = {}
    accm: dict = {}
    accpos: dict = {}
    counts: dict = {}
    fixed_probes = None
    t0 = time.time()
    tokens_seen = 0

    for ci, (_, ids) in enumerate(chunks):
        ids = ids.to(device)
        prompt, cont = ids[:ctx].unsqueeze(0), ids[ctx : ctx + cont_len].unsqueeze(0)
        keys, values = base_cache_tensors(model, prompt)
        leaves = {}
        kk = [k.clone() for k in keys]
        vv = [v.clone() for v in values]
        for li in layers:
            kk[li] = kk[li].detach().clone().requires_grad_(True)
            vv[li] = vv[li].detach().clone().requires_grad_(True)
            leaves[(li, "k")] = kk[li]
            leaves[(li, "v")] = vv[li]

        with torch.enable_grad():
            out = forward_continuation(model, kk, vv, cont, ctx)
            logits = out.logits
            if fixed_probes is None:
                g0 = torch.Generator(device=device).manual_seed(seed + 555)
                fixed_probes = [
                    fisher_probe(logits, g0).detach() for _ in range(MEANJ_PROBES)
                ]
            g = torch.Generator(device=device).manual_seed(seed * 1000 + ci)
            for p in range(MAX_PROBES):
                z = fisher_probe(logits, g)
                u = vjp_from_probe(logits, leaves, z, retain=True)
                for key, ug in u.items():
                    x = ug.float()[0]  # [H, T, D]
                    acc2.setdefault((key, p), 0)
                    acc2[(key, p)] = acc2[(key, p)] + torch.einsum("htd,hte->hde", x, x)
                    accpos.setdefault((key, p), 0)
                    accpos[(key, p)] = accpos[(key, p)] + x.pow(2).sum(dim=-1)
                    counts[key] = counts.get(key, 0) + x.shape[1]
            # The control averages the product across contexts *before*
            # squaring, and that is its only intended difference from the
            # candidate.  It therefore keeps the position axis exactly as the
            # candidate does; collapsing positions here would make the control
            # a different, lower-rank object and the contrast unreadable.
            for p in range(MEANJ_PROBES):
                zf = fixed_probes[p].to(logits.dtype)
                um = vjp_from_probe(logits, leaves, zf, retain=(p < MEANJ_PROBES - 1))
                for key, ug in um.items():
                    accm.setdefault((key, p), 0)
                    accm[(key, p)] = accm[(key, p)] + ug.float()[0]  # [H, T, D]
        tokens_seen += ctx + cont_len
        del kk, vv, leaves, out, logits
        if (ci + 1) % 8 == 0:
            log(f"  fitted on {ci + 1}/{len(chunks)} calibration contexts")

    n_ctx = len(chunks)
    # jtfj: cumulative over probe counts so 1/2/4/8 are all available
    G2: dict = {}
    for pc in PROBE_COUNTS:
        for key in {k for (k, _) in acc2}:
            tot = sum(acc2[(key, p)] for p in range(pc))
            G2[(key, pc)] = tot / float(counts[key] / MAX_PROBES * pc)
    Gm: dict = {}
    Wm: dict = {}
    for key in {k for (k, _) in accm}:
        ubar = torch.stack(
            [accm[(key, p)] / n_ctx for p in range(MEANJ_PROBES)], 0
        )  # [P, H, T, D]
        n_pos = ubar.shape[2]
        Gm[key] = torch.einsum("phtd,phte->hde", ubar, ubar) / (MEANJ_PROBES * n_pos)
        em = ubar.pow(2).sum(-1).mean(dim=0)  # [H, T]
        Wm[key] = em / em.mean(dim=-1, keepdim=True).clamp(min=1e-30)
    # Positional energy profile, normalised to mean one per head.  Contexts are
    # a fixed length here, so absolute position transfers between them; this is
    # the cheapest transferable thing the Jacobian knows that a position-uniform
    # metric throws away, and isolating it is what makes a later win
    # attributable to the matrix shape rather than to the position weighting.
    # Two forms are kept.  The normalised profile scales a metric that already
    # carries per-head magnitude in G, so the product counts that magnitude
    # once.  The raw profile is what the shape-free control must use: its
    # per-head mean is the trace of G for that head, so using the normalised
    # form there would strip the control of per-head and per-layer sensitivity
    # and credit the matrix shape with it.
    W: dict = {}
    Wraw: dict = {}
    for pc in PROBE_COUNTS:
        for key in {k for (k, _) in accpos}:
            tot = sum(accpos[(key, p)] for p in range(pc)) / float(n_ctx * pc)
            Wraw[(key, pc)] = tot / float(tot.shape[-1])
            W[(key, pc)] = tot / tot.mean(dim=-1, keepdim=True).clamp(min=1e-30)
    # The candidate truncated to the control's rank, so the two can be compared
    # at matched estimator budget instead of at matched label.
    Gr: dict = {}
    for pc in PROBE_COUNTS:
        for key in {k for (k, _) in acc2}:
            Gr[(key, pc)] = _truncate_rank(G2[(key, pc)], MEANJ_PROBES)
    return (
        G2,
        Gm,
        W,
        Wraw,
        Wm,
        Gr,
        {"fit_s": time.time() - t0, "fit_tokens": tokens_seen},
    )


def _truncate_rank(G: torch.Tensor, rank: int) -> torch.Tensor:
    """Keep the top ``rank`` eigendirections of each head's metric."""
    evals, evecs = torch.linalg.eigh(G.double())
    r = int(min(rank, evals.shape[-1]))
    keep = evals[..., -r:].clamp(min=0.0)
    vecs = evecs[..., -r:]
    return torch.einsum("hdr,hr,her->hde", vecs, keep, vecs).to(G.dtype)


def quad_score(error: torch.Tensor, G: torch.Tensor, w=None) -> float:
    """``sum_{h,t} w_ht * e_ht' G_h e_ht`` -- the block-diagonal receiver score.

    ``G`` is the per-head *shape* of the receiver metric; ``w`` is an optional
    per-(head, position) *scale*.  Keeping them apart is deliberate.  The shape
    is what the second moment uniquely claims to contribute; the scale is
    available from cheaper sources -- a fitted positional profile, or the
    receiver's own attention mass -- so separating them makes any win
    attributable to one or the other rather than only to their product.
    """
    e = error.float()[0]  # [H, T, D]
    q = torch.einsum("htd,hde,hte->ht", e, G, e)
    if w is not None:
        q = q * w[..., : q.shape[-1]]
    return float(q.sum())


def energy_score(error: torch.Tensor, w: torch.Tensor) -> float:
    """``sum_{h,t} w_ht ||e_ht||^2`` -- a positional profile with no matrix shape.

    The control that decides whether the second moment's matrix structure earns
    its keep, or whether a scalar per-position weight was the whole story.
    """
    e = error.float()[0]
    return float((e.pow(2).sum(-1) * w[..., : e.shape[1]]).sum())


def attention_position_weight(attn_layer, n_kv_heads, n_positions):
    """Per-(kv head, position) attention mass, normalised to mean one.

    Free at encode time -- it needs no backward pass -- which is what makes the
    ``jtfj_attn`` arm a deployable instantiation rather than an oracle.
    """
    a = attn_layer.float().mean(dim=2)[0]  # [n_q, T_kv]
    if a.shape[0] != n_kv_heads:
        grp = a.shape[0] // n_kv_heads
        a = a.reshape(n_kv_heads, grp, a.shape[-1]).mean(dim=1)
    a = a[:, :n_positions]
    return a / a.mean(dim=-1, keepdim=True).clamp(min=1e-30)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None, help="pinned snapshot hash")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--cont-len", type=int, default=64)
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--eval", type=int, default=12)
    ap.add_argument("--layers", default="", help="comma list; default = 5 spread")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--context-source", default="wikitext", choices=["wikitext", "needle"]
    )
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument(
        "--min-level",
        type=float,
        default=0.0,
        help="drop perturbation levels below this fraction of the block norm; "
        "a robustness check, since the smallest level carries rows whose "
        "measured damage is near the estimator's own resolution",
    )
    ap.add_argument(
        "--analyse-only",
        action="store_true",
        help="recompute the gates from an existing perturbations.jsonl; "
        "measured data is never re-measured just to change a statistic",
    )
    args = ap.parse_args()

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    if args.analyse_only:
        return _analyse_only(args, log)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    t_start = time.time()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager", revision=args.revision
    ).to(device)
    freeze_model(model)
    cfg = model.config
    n_layer = int(cfg.num_hidden_layers)
    n_q = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_q))
    head_dim = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q)
    group = n_q // n_kv

    layers = (
        [int(x) for x in args.layers.split(",") if x != ""]
        if args.layers
        else sorted({0, n_layer // 4, n_layer // 2, (3 * n_layer) // 4, n_layer - 1})
    )
    log(f"{args.model}: {n_layer}L n_q={n_q} n_kv={n_kv} D={head_dim}; layers {layers}")

    src = wikitext_chunks if args.context_source == "wikitext" else needle_chunks
    all_chunks = src(tok, args.calib + args.eval, args.ctx, args.cont_len, args.seed)
    calib = all_chunks[: args.calib]
    held = all_chunks[args.calib :]
    calib_idx = tuple(int(i) for i, _ in calib)
    eval_idx = tuple(int(i) for i, _ in held)
    assert not (set(calib_idx) & set(eval_idx)), "calibration and evaluation overlap"
    log(f"{len(calib)} calibration / {len(held)} held-out contexts, ctx={args.ctx}")

    run_id = args.run_id or (
        f"{args.model.replace('/', '_')}_ctx{args.ctx}_{args.context_source}_s{args.seed}"
    )
    out_dir = art.ensure_run_dir(args.out_dir, run_id)
    rows_path = os.path.join(out_dir, "perturbations.jsonl")

    # ---- fit the global metrics on calibration contexts ------------------
    log("fitting global receiver metrics on calibration contexts")
    G2, Gm, Wpos, Wposraw, Wmean, Grank, fit_stats = fit_global_metrics(
        model, tok, calib, args.ctx, args.cont_len, layers, device, args.seed, log
    )
    log(f"fit done in {fit_stats['fit_s']:.1f}s")

    wo = {
        li: model.model.layers[li].self_attn.o_proj.weight.detach().float()
        for li in layers
    }

    # ---- score the held-out perturbation bank ----------------------------
    records = []
    writer = art.JsonlWriter(rows_path)
    eval_tokens = 0
    t_eval = time.time()
    try:
        for ci, (chunk_id, ids) in enumerate(held):
            ids = ids.to(device)
            prompt = ids[: args.ctx].unsqueeze(0)
            cont = ids[args.ctx : args.ctx + args.cont_len].unsqueeze(0)
            # Prefill attention and post-RoPE queries are what a cache writer
            # can actually see.  The continuation's attention over the prompt is
            # the future, so arms built from it are kept but labelled oracle and
            # are judged only against baselines allowed the same information.
            keys, values, pre_attn, pre_q = prefill_with_attention(
                model, prompt, layers
            )

            with torch.no_grad():
                clean_out = forward_continuation(
                    model, keys, values, cont, args.ctx, want_attn=True
                )
            clean_logits = clean_out.logits.detach().float()
            attns = [a.detach() for a in clean_out.attentions]
            del clean_out
            attn_w = {
                li: attention_position_weight(attns[li], n_kv, args.ctx)
                for li in layers
            }
            pre_attn_w = {
                li: attention_position_weight(pre_attn[li], n_kv, args.ctx)
                for li in layers
            }

            # Numerical floor.  Re-prefilling from the tokens reaches the same
            # cache by an independent path, so this measures real run-to-run
            # resolution.  Comparing two identical forwards over the same
            # tensors, as an earlier version did, is bit-identical by
            # construction and reports 0.0 whatever the estimator's precision
            # actually is.
            rk, rv = base_cache_tensors(model, prompt)
            with torch.no_grad():
                ctrl = forward_continuation(model, rk, rv, cont, args.ctx)
            ctrl_kl = teacher_forced_kl(clean_logits, ctrl.logits.float())
            floor_kl = float(ctrl_kl.abs().mean())
            floor_kl_max = float(ctrl_kl.abs().max())
            del ctrl, rk, rv

            # oracle sketch for this context: one backward set, all layers
            kk = [k.clone() for k in keys]
            vv = [v.clone() for v in values]
            leaves = {}
            for li in layers:
                kk[li] = kk[li].detach().clone().requires_grad_(True)
                vv[li] = vv[li].detach().clone().requires_grad_(True)
                leaves[(li, "k")] = kk[li]
                leaves[(li, "v")] = vv[li]
            oracle: dict = {}
            with torch.enable_grad():
                o = forward_continuation(model, kk, vv, cont, args.ctx)
                g = torch.Generator(device=device).manual_seed(args.seed * 7717 + ci)
                for p in range(MAX_PROBES):
                    z = fisher_probe(o.logits, g)
                    u = vjp_from_probe(o.logits, leaves, z, retain=(p < MAX_PROBES - 1))
                    for key, ug in u.items():
                        oracle.setdefault(key, []).append(ug.float().reshape(-1).cpu())
            del kk, vv, leaves, o

            oracle_U = {k: torch.stack(v, 0) for k, v in oracle.items()}

            for li in layers:
                for kind in ("k", "v"):
                    block = (keys if kind == "k" else values)[li]
                    gen = torch.Generator(device=device).manual_seed(
                        args.seed * 104729 + ci * 1013 + li * 17 + (kind == "v")
                    )
                    bank = build_bank(
                        block,
                        kind=kind,
                        layer=li,
                        generator=gen,
                        context_tag=f"c{chunk_id}",
                    )
                    for pert in bank:
                        pk = list(keys)
                        pv = list(values)
                        if kind == "k":
                            pk[li] = block + pert.delta
                        else:
                            pv[li] = block + pert.delta
                        with torch.no_grad():
                            po = forward_continuation(model, pk, pv, cont, args.ctx)
                        pl = po.logits.float()
                        del po

                        kl = float(teacher_forced_kl(clean_logits, pl).mean())
                        dnll = float(delta_nll(clean_logits, pl, cont).mean())
                        flips = argmax_flip_rate(clean_logits, pl)
                        del pl

                        e = pert.delta
                        scores = {
                            "mse": score_mse(e),
                            # deployable attention weighting: the prompt's own
                            # self-attention, which exists when a cache is written
                            "attn_prefill": score_attention_local(
                                e, pre_attn[li], kind
                            ),
                            "attn_prefill_lin": score_attention_local(
                                e, pre_attn[li], kind, power=1.0
                            ),
                            # oracle-tier: built from the continuation's attention
                            "attn_future": score_attention_local(e, attns[li], kind),
                            "attn_future_lin": score_attention_local(
                                e, attns[li], kind, power=1.0
                            ),
                            "meanj": quad_score(e, Gm[(li, kind)]),
                            "meanj_pos": quad_score(
                                e, Gm[(li, kind)], Wmean[(li, kind)]
                            ),
                        }
                        if kind == "v":
                            # the output-projection metric is a value-space
                            # construct; a key never passes through it
                            scores["wo"] = _wo_score(e, wo[li], group, head_dim)
                        else:
                            scores["qk_local"] = score_qk_local(
                                e, pre_q[li], pre_attn[li], n_kv
                            )
                        for pc in PROBE_COUNTS:
                            G = G2[((li, kind), pc)]
                            wfit = Wpos[((li, kind), pc)]
                            wraw = Wposraw[((li, kind), pc)]
                            scores[f"jtfj_p{pc}"] = quad_score(e, G)
                            scores[f"jtfj_pos_p{pc}"] = quad_score(e, G, wfit)
                            scores[f"jtfj_prefill_p{pc}"] = quad_score(
                                e, G, pre_attn_w[li]
                            )
                            scores[f"jtfj_future_p{pc}"] = quad_score(e, G, attn_w[li])
                            scores[f"jtfj_rank_p{pc}"] = quad_score(
                                e, Grank[((li, kind), pc)], wfit
                            )
                            scores[f"pos_energy_p{pc}"] = energy_score(e, wraw)
                        U = oracle_U[(li, kind)]
                        ev = e.float().reshape(-1).cpu()
                        for pc in PROBE_COUNTS:
                            scores[f"jtfj_oracle_p{pc}"] = float(
                                (U[:pc] @ ev).pow(2).mean()
                            )

                        row = {
                            "context": int(chunk_id),
                            "layer": li,
                            "kind": kind,
                            "family": pert.family,
                            "params": pert.params,
                            "norm_group": pert.norm_group,
                            "frob": pert.frob,
                            "exact_kl": kl,
                            "delta_nll": dnll,
                            "argmax_flip_rate": flips,
                            "floor_kl": floor_kl,
                            "floor_kl_max": floor_kl_max,
                            "scores": scores,
                        }
                        writer.write(row)
                        records.append(row)
                        eval_tokens += args.ctx + args.cont_len
            del attns, attn_w, pre_attn, pre_attn_w, pre_q
            del clean_logits, oracle, oracle_U
            torch.cuda.empty_cache() if device == "cuda" else None
            log(
                f"  context {ci + 1}/{len(held)} done "
                f"({len(records)} perturbations, floor KL {floor_kl:.2e})"
            )
    finally:
        writer.close()

    eval_s = time.time() - t_eval
    log(f"scored {len(records)} perturbations in {eval_s:.1f}s")

    # ---- gates -----------------------------------------------------------
    summary = analyse(records, args, model_name=args.model)
    summary["fit_stats"] = fit_stats
    summary["matched_norm_check"] = _norm_check(records)

    wall_h = (time.time() - t_start) / 3600.0
    man = art.JKVManifest(
        run_id=run_id,
        stage="stage0",
        target_model_id=args.model,
        target_model_revision=args.revision or _resolve_revision(args.model),
        dtype=args.dtype,
        attn_backend="eager",
        device=device,
        rope_status="post_rope",
        rope_theta_target=float(getattr(cfg, "rope_theta", 0.0) or 0.0),
        rope_scaling_target=getattr(cfg, "rope_scaling", None),
        kv_split="both",
        context_length=int(args.ctx),
        continuation_length=int(args.cont_len),
        selected_layers=tuple(layers),
        probe_count=MAX_PROBES,
        measurement_level="fake_quant_teacher_forced",
        calib_dataset=f"{args.context_source}:calib",
        calib_indices=calib_idx,
        eval_dataset=f"{args.context_source}:eval",
        eval_indices=eval_idx,
        seed=args.seed,
        versions=art.versions(),
        gpu_model=(torch.cuda.get_device_name(0) if device == "cuda" else "cpu"),
        gpu_hours=wall_h,
        wall_hours=wall_h,
        evaluated_tokens=eval_tokens,
        observed_tokens_per_s=eval_tokens / max(eval_s, 1e-9),
        hourly_price_usd=0.0,
        billed_cost_usd=0.0,
        peak_mem_bytes=(
            int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
        ),
        cmdline=" ".join(sys.argv),
    )
    man.code_commit, man.code_dirty = art.git_state()
    man.config_hash = art.config_hash(vars(args))
    art.write_json(os.path.join(out_dir, "manifest.json"), man.finalize())
    art.write_json(os.path.join(out_dir, "summary.json"), summary)

    _print_table(summary, log)
    return 0


def _analyse_only(args, log):
    """Redo the statistics on rows already measured, touching no GPU."""
    run_id = args.run_id
    if not run_id:
        raise SystemExit("--analyse-only needs --run-id")
    out_dir = os.path.join(args.out_dir, run_id)
    rows_path = os.path.join(out_dir, "perturbations.jsonl")
    # Identity comes from the run's own manifest, never from this invocation's
    # flags.  Re-analysing with a different --seed would otherwise relabel the
    # rows, and the replication requirement reads exactly that field.
    man_path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(man_path):
        raise SystemExit(f"{man_path} missing; refusing to label rows from the CLI")
    with open(man_path) as fh:
        man = json.load(fh)
    args.seed = int(man["seed"])
    args.ctx = int(man["context_length"])
    args.cont_len = int(man["continuation_length"])
    src = man.get("eval_dataset", "")
    if ":" in src:
        args.context_source = src.split(":", 1)[0]
    args.model = man["target_model_id"]
    with open(rows_path) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    if args.min_level > 0:
        before = len(records)
        records = [
            r for r in records if float(r["params"].get("level", 0)) >= args.min_level
        ]
        log(f"dropped {before - len(records)} rows below level {args.min_level}")
    log(f"re-analysing {len(records)} rows from {rows_path} (seed {args.seed})")
    summary = analyse(records, args, model_name=args.model)
    summary["matched_norm_check"] = _norm_check(records)
    summary["reanalysed_from"] = rows_path
    summary["min_level"] = args.min_level
    art.write_json(os.path.join(out_dir, "summary.json"), summary)
    _print_table(summary, log)
    return 0


def _wo_score(e, w_o, group, head_dim):
    """Expand a kv-head error to its query heads, then push through o_proj."""
    b, h, t, d = e.shape
    eq = e.float().repeat_interleave(group, dim=1)  # [B, n_q, T, D]
    flat = eq.permute(0, 2, 1, 3).reshape(b, t, -1)
    return float((flat @ w_o.transpose(0, 1)).pow(2).mean())


def _resolve_revision(model_id):
    """Best-effort pinned revision from the local snapshot path."""
    try:
        from huggingface_hub import snapshot_download

        p = snapshot_download(model_id, local_files_only=True)
        return os.path.basename(p.rstrip("/"))
    except Exception:  # noqa: BLE001
        return "local-snapshot-unresolved"


def _norm_check(records):
    groups: dict = {}
    for r in records:
        groups.setdefault(r["norm_group"], []).append(r["frob"])
    worst = 0.0
    for norms in groups.values():
        lo, hi = min(norms), max(norms)
        worst = max(worst, (hi - lo) / max(lo, 1e-30))
    return {"n_groups": len(groups), "worst_relative_norm_spread": worst}


# Baselines are tiered by what information they are allowed.  A candidate that
# needs the continuation's attention -- which does not exist when a cache is
# written -- is judged against baselines allowed the same thing, so neither
# side is credited for seeing the future.  The signed mean Jacobian is a
# baseline, not a candidate: the plan lists it among the minimum serious
# comparison, and leaving it out of the tier would let a candidate clear the
# bar without ever having to beat the control the plan cares most about.
# The plan's promotion rule is written against the *cheap* incumbents, so that
# is what the gate compares to.  The signed mean-Jacobian control is a separate
# question and gets its own reported comparison: the plan's decision tree says
# that if a simpler method matches, the simpler method is what ships, so the
# control has to be answered even when the cheap gate passes.
DEPLOYABLE_BASELINES = ("mse", "wo", "qk_local", "attn_prefill", "attn_prefill_lin")
ORACLE_BASELINES = DEPLOYABLE_BASELINES + ("attn_future", "attn_future_lin")
CONTROL_BASELINES = ("meanj", "meanj_pos")

DEPLOYABLE_CANDIDATES = (
    "jtfj_p4",
    "jtfj_p8",
    "jtfj_pos_p4",
    "jtfj_pos_p8",
    "jtfj_prefill_p8",
    "jtfj_rank_p8",
    "pos_energy_p8",
)
ORACLE_CANDIDATES = ("jtfj_future_p8", "jtfj_oracle_p4", "jtfj_oracle_p8")


def _probe_family(scores, pc):
    """Every probe-varying arm re-expressed at probe count ``pc``."""
    out = {}
    for name in scores:
        if "_p" in name and name.rsplit("_p", 1)[-1].isdigit():
            stem = name.rsplit("_p", 1)[0]
            alt = f"{stem}_p{pc}"
            if alt in scores:
                out[name] = scores[alt]
    return out


def analyse(records, args, model_name):
    """The blocked gate, per tensor kind, with pooled figures kept as context."""
    out = {
        "model": model_name,
        "ctx": args.ctx,
        "seed": args.seed,
        "context_source": args.context_source,
        "n_records": len(records),
        "statistic": "blocked_spearman_fisher_mean",
        "cells": {},
    }
    if not records:
        return out
    score_names = sorted(records[0]["scores"].keys())

    for kind in ("k", "v"):
        rs = [r for r in records if r["kind"] == kind]
        if len(rs) < 8:
            continue
        truth = [r["exact_kl"] for r in rs]
        groups = [r["norm_group"] for r in rs]
        clusters = [r["context"] for r in rs]
        names = [n for n in score_names if n in rs[0]["scores"]]
        scores = {n: [r["scores"][n] for r in rs] for n in names}

        cell = {
            "n": len(rs),
            "n_groups": len(set(groups)),
            "n_clusters": len(set(clusters)),
            "blocked_rho": {
                n: blocked_spearman(scores[n], truth, groups) for n in names
            },
            "pooled_rho": {n: spearman(scores[n], truth) for n in names},
            "floor_kl_mean_max": max(r["floor_kl"] for r in rs),
            "floor_kl_max": max(r.get("floor_kl_max", 0.0) for r in rs),
            "exact_kl_median": float(np.median(truth)),
            "exact_kl_min": float(np.min(truth)),
            "gates": {},
        }

        fam4, fam8 = _probe_family(scores, 4), _probe_family(scores, 8)
        md = method_ranking_distance(scores, truth, groups, fam4, fam8)

        for cand, tier, bl in [
            *[(c, "deployable", DEPLOYABLE_BASELINES) for c in DEPLOYABLE_CANDIDATES],
            *[(c, "oracle", ORACLE_BASELINES) for c in ORACLE_CANDIDATES],
        ]:
            if cand not in scores:
                continue
            item = {}
            a4 = scores.get(cand.replace("_p8", "_p4"))
            a8 = scores.get(cand.replace("_p4", "_p8"))
            if a4 is not None and a8 is not None and a4 is not a8:
                item["item_ordering_distance_4_to_8"] = kendall_distance(a4, a8)
            g = evaluate_gate(
                scores,
                truth,
                groups,
                clusters,
                model=model_name,
                kind=kind,
                candidate=cand,
                baselines=[b for b in bl if b in scores],
                baseline_tier=tier,
                probe_stability={"method_ranking_distance_4_to_8": md, **item},
                n_boot=args.n_boot,
                seed=args.seed,
            )
            cell["gates"][cand] = g.to_dict()
            controls = [b for b in CONTROL_BASELINES if b in scores]
            if controls and cand not in CONTROL_BASELINES:
                gc = evaluate_gate(
                    scores,
                    truth,
                    groups,
                    clusters,
                    model=model_name,
                    kind=kind,
                    candidate=cand,
                    baselines=controls,
                    baseline_tier="mean_jacobian_control",
                    probe_stability={
                        "method_ranking_distance_4_to_8": md,
                        **item,
                    },
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
                cell.setdefault("control_gates", {})[cand] = gc.to_dict()

        cell["attribution"] = _attribution(scores, truth, groups, clusters, args)
        out["cells"][kind] = cell
    return out


def _attribution(scores, truth, groups, clusters, args):
    """Where any ranking power actually comes from.

    Paired comparisons whose two arms differ in exactly one ingredient, each
    with its own clustered interval, so a verdict is an account rather than a
    yes or no. Pairs whose arms differ in more than one thing are named for
    what they confound rather than for what they would isolate.
    """
    pairs = {
        # the matrix shape, against a profile carrying the same per-head and
        # per-layer scale -- the only difference left is the shape itself
        "shape_over_position": ("jtfj_pos_p8", "pos_energy_p8"),
        # the fitted positional weighting, over the position-uniform metric
        "position_over_uniform": ("jtfj_pos_p8", "jtfj_p8"),
        # the second moment against the signed mean Jacobian, at matched rank
        "second_moment_over_mean_jacobian": ("jtfj_rank_p8", "meanj_pos"),
        # full-rank candidate over its own rank-matched form: how much of the
        # candidate's lead is estimator budget rather than the object
        "full_rank_over_matched_rank": ("jtfj_pos_p8", "jtfj_rank_p8"),
        # attention mass a cache writer can see, against the fitted profile
        "prefill_attention_over_fitted": ("jtfj_prefill_p8", "jtfj_pos_p8"),
        # confounds three axes at once: per-context fitting, full coupling
        # rather than block diagonal, and 8 samples against a fitted estimate
        "oracle_over_global_confounded": ("jtfj_oracle_p8", "jtfj_pos_p8"),
    }
    out = {}
    for name, (a, b) in pairs.items():
        if a not in scores or b not in scores:
            continue
        d, lo, hi = cluster_bootstrap_blocked_delta(
            scores[a],
            scores[b],
            truth,
            groups,
            clusters,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        out[name] = {"a": a, "b": b, "delta": d, "ci_lo": lo, "ci_hi": hi}
    return out


def _print_table(summary, log):
    for kind, cell in summary.get("cells", {}).items():
        log(
            f"--- {summary['model']} kind={kind} n={cell['n']} "
            f"groups={cell['n_groups']} contexts={cell['n_clusters']}"
        )
        br, pr = cell["blocked_rho"], cell["pooled_rho"]
        log(f"    {'metric':22s} {'blocked rho':>12s} {'pooled rho':>11s}")
        for n in sorted(br, key=lambda x: -br[x] if not math.isnan(br[x]) else 1):
            log(f"    {n:22s} {br[n]:12.4f} {pr.get(n, float('nan')):11.4f}")
        for name, at in cell.get("attribution", {}).items():
            log(
                f"    attrib {name:34s} {at['delta']:+.4f} "
                f"CI[{at['ci_lo']:+.4f},{at['ci_hi']:+.4f}]"
            )
        for cand, g in list(cell.get("gates", {}).items()) + list(
            cell.get("control_gates", {}).items()
        ):
            verdict = "PASS" if g["passed"] else "fail"
            log(
                f"    gate[{g['baseline_tier'][:4]}] {cand:18s} {verdict}  "
                f"delta={g['delta']:+.4f} CI[{g['ci_lo']:+.4f},{g['ci_hi']:+.4f}] "
                f"p={g['p_value']:.4f} vs {g['best_baseline']}"
            )
            for r in g["reasons"]:
                log(f"        - {r}")
        log(
            f"    numerical floor: mean |KL| {cell['floor_kl_mean_max']:.3e}, "
            f"worst {cell['floor_kl_max']:.3e}; smallest measured KL "
            f"{cell['exact_kl_min']:.3e}, median {cell['exact_kl_median']:.3e}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
