"""Lattice-KRI training-free screen — KL/PPL vs *actual* retained KV.

Phase 2 + 3 harness. Compares the Lattice-KRI variants against the KRI
family and the structural baselines on one shared surface, measuring KL to
the full (dense) cache and the actual fraction of KV retained — not the
nominal budget. The question the screen answers: at matched *actual*
retained KV, does any Lattice variant beat KRI-Q+N?

Surface. Every router here is a block *selector* feeding the same
sink/recent/current/causal assembly (_causal_base_mask + _select_blocks_to_mask
from canonical_kri). KRI-Q / KRI-Q+N / KRI reuse select_kri_blocks; the
Lattice variants use select_lattice_blocks; the structural baselines emit the
base mask with no global blocks. The query probe is the last-position Key
(collect_kv exposes K/V, not Q — this matches eval_canonical_kri), so every
router is handicapped identically and the comparison is apples-to-apples.
The strict per-position no-future-leak path (build_kri_mask + select_fn) is
proven by tests/test_lattice_kri.py and is the deployable contract; this
screen uses the cheaper single-probe surface for a fair cross-router sweep.

Negative control. trellis_kri_gated (the parked mask-only Trellis-KRI) is not
present in this repo, so random_global — the existing iso-budget random
control — stands in: if a scored router cannot beat random block selection at
matched KV, its scoring rule is not earning its keep.

Output is one JSONL/CSV row per (model, router, topk) with KL, PPL, the
sparse/full NLL gap, and both overall and decode-region actual retained
fraction (normalised to the dense-causal baseline T/2, as elsewhere in the
repo). Plotting + diagnostics live in plot_lattice_kri_curves.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.canonical_kri import _causal_base_mask, _select_blocks_to_mask  # noqa: E402
from src.data import (  # noqa: E402
    DataConfig,
    collate,
    get_tokenizer,
    get_train_val_streams,
)
from src.eval_canonical_kri import _full_log_probs, _stats_under_mask  # noqa: E402
from src.eval_pruned_ppl import load_model  # noqa: E402
from src.kri_mask import (  # noqa: E402
    KRIConfig,
    build_kri_mask,
    num_blocks,
    select_kri_blocks,
)
from src.lattice_kri import (
    LatticeConfig,
    _VARIANTS,
    select_lattice_blocks,
)  # noqa: E402
from src.trellis_kri import score_trellis_blocks, select_trellis_blocks  # noqa: E402
from src.kri_mask import _block_summaries  # noqa: E402
from src.portfolio import (  # noqa: E402
    compose,
    parse_pf_name,
    pf_valid,
    qualified_pool,
    spec_n,
)
from src.utils import pick_device, pick_dtype, report_device, set_seed  # noqa: E402

# The default comparison set. Order = baselines, KRI family, Lattice, controls.
DEFAULT_ROUTERS = (
    ["full", "recent", "sink_recent", "kri_q", "kri_q_novelty", "kri"]
    + [f"lattice_kri_{v}" for v in _VARIANTS]
    + ["trellis_kri", "random_global"]
)


def _kri_config(name, block_size, local_window, topk, sink_blocks):
    """KRI-family selector configs, all via the native select_kri_blocks."""
    common = dict(
        block_size=block_size,
        local_window_tokens=local_window,
        global_topk_blocks=topk,
        protected_blocks=tuple(range(sink_blocks)),
    )
    if name == "kri_q":  # cosine-only KRI-Q
        return KRIConfig(
            **common, w_cos=1.0, w_value_energy=0.0, w_recency=0.0, use_novelty=False
        )
    if name == "kri_q_novelty":  # KRI-Q + the repo's novelty term (the baseline)
        return KRIConfig(
            **common,
            w_cos=1.0,
            w_value_energy=0.0,
            w_recency=0.0,
            use_novelty=True,
            w_novelty=0.5,
        )
    if (
        name == "kri"
    ):  # the full internal KRI-Q+N curriculum (cos+value+recency+novelty)
        return KRIConfig(**common)
    raise ValueError(name)


def capture_q_probe(model, ids, li, recent_window=16):
    """Pre-RoPE Q probes at score layer `li`, GQA-aggregated to kv heads.

    Mirrors collect_kv's pre-hook trick (k_proj on the layer's hidden states),
    so probe and key centroids live in the same PRE-RoPE space the canonical
    surface already uses. Returns dict with 'last' and 'recent_mean' probes,
    each [B, Hkv, D] L2-normalised.
    """
    grabbed = {}
    Hq, Hkv = model.cfg_n_head, model.cfg_n_kv_head
    attn = list(model._attn_module_iter())[li]

    def hook(module, args, kwargs):
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if hidden is None:
            return
        B, T, _ = hidden.shape
        q = module.q_proj(hidden)
        D = q.shape[-1] // Hq
        grabbed["q"] = q.view(B, T, Hq, D).transpose(1, 2).detach()  # [B,Hq,T,D]

    h = attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model.forward(ids)
    finally:
        h.remove()
    q = grabbed["q"].float()
    B, _, T, D = q.shape
    g = Hq // Hkv
    qn = F.normalize(q.view(B, Hkv, g, T, D), dim=-1)
    last = F.normalize(qn[:, :, :, -1, :].mean(2), dim=-1)  # [B,Hkv,D]
    w = min(recent_window, T)
    recent = F.normalize(qn[:, :, :, T - w :, :].mean(dim=(2, 3)), dim=-1)
    return {"last": last, "recent_mean": recent}


def capture_attn_block_mass(model, ids, li, block_size):
    """EXACT accumulated attention mass per (kv-head, block) at layer `li`,
    from the eager-attention softmax itself (output_attentions) -- the true
    H2O statistic, replacing the K-as-query proxy. Returns [B, Hkv, NB]."""
    Hq, Hkv = model.cfg_n_head, model.cfg_n_kv_head
    with torch.no_grad():
        out = model.hf(input_ids=ids, output_attentions=True, use_cache=False)
    a = out.attentions[li].float()  # [B,Hq,T,T] post-softmax
    del out
    B, _, T, _ = a.shape
    tokmass = a.sum(dim=2)  # mass received per key token [B,Hq,T]
    del a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    g = Hq // Hkv
    tokmass = tokmass.view(B, Hkv, g, T).sum(2)  # [B,Hkv,T]
    nb = (T + block_size - 1) // block_size
    if nb * block_size > T:
        tokmass = torch.nn.functional.pad(tokmass, (0, nb * block_size - T))
    return tokmass.view(B, Hkv, nb, block_size).sum(-1)  # [B,Hkv,NB]


def _route_per_position(
    name,
    k,
    v,
    L,
    block_size,
    local_window,
    sink_blocks,
    topk,
    device,
    lattice_kwargs,
    prefill_split,
):
    """Per-position surface: each decode position routes with its OWN query
    probe (q_t = K[:, :, t, :]) via build_kri_mask, reusing its sink/recent/
    current/prefill/causal assembly and no-future-leak guarantee. This is the
    faithful surface the spec asks for and where KRI routing actually works;
    the single last-position probe (canonical) is too weak to beat random on
    easy data."""
    B, H, T, D = k.shape
    if name == "full":
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
        return causal.view(1, 1, L, L).expand(B, H, L, L).clone(), None
    kw = dict(k_per_layer=[k], v_per_layer=[v], q_per_layer=[k], device=device)
    if name in ("recent", "sink_recent"):
        sb = 0 if name == "recent" else sink_blocks
        cfg = KRIConfig(
            block_size=block_size,
            local_window_tokens=local_window,
            global_topk_blocks=0,
            prefill_split=prefill_split,
            protected_blocks=tuple(range(sb)),
        )
        return build_kri_mask(cfg, L, B, H, **kw), None
    if name == "random_global":
        return (
            _per_position_random(
                k,
                L,
                block_size,
                local_window,
                sink_blocks,
                topk,
                prefill_split,
                device,
            ),
            None,
        )
    if name == "h2o":
        return (
            _per_position_h2o(
                k,
                L,
                block_size,
                local_window,
                sink_blocks,
                topk,
                prefill_split,
                device,
            ),
            None,
        )
    if name == "trellis_kri":
        return (
            _per_position_trellis(
                k,
                v,
                L,
                block_size,
                local_window,
                sink_blocks,
                topk,
                prefill_split,
                device,
            ),
            None,
        )
    if name.startswith("lattice_kri"):
        variant = "rel_orth" if name == "lattice_kri" else name[len("lattice_kri_") :]
        cfg = LatticeConfig(
            block_size=block_size,
            local_window_tokens=local_window,
            global_topk_blocks=topk,
            prefill_split=prefill_split,
            protected_blocks=tuple(range(sink_blocks)),
            variant=variant,
            **(lattice_kwargs or {}),
        )
        sel_fn = select_lattice_blocks
    else:  # kri family
        cfg = _kri_config(name, block_size, local_window, topk, sink_blocks)
        cfg.prefill_split = prefill_split
        sel_fn = select_kri_blocks
    return build_kri_mask(cfg, L, B, H, select_fn=sel_fn, **kw), None


def _per_position_h2o(
    k, L, block_size, local_window, sink_blocks, topk, prefill_split, device
):
    """Per-position H2O (Heavy-Hitter Oracle) baseline: keep the top-K blocks by
    ACCUMULATED attention mass (summed attention from every prior query up to t),
    plus the sink+recent base. Attention uses K as the query proxy (the eval's
    convention) and a shared-over-(B,H) ranking, like the random control. This is
    the real deployed-SOTA bar -- residual_rel has to beat this, not just KRI."""
    import math

    B, H, T, D = k.shape
    bs = block_size
    cfg = KRIConfig(
        block_size=bs,
        local_window_tokens=local_window,
        global_topk_blocks=0,
        prefill_split=prefill_split,
        protected_blocks=tuple(range(sink_blocks)),
    )
    mask = build_kri_mask(
        cfg, L, B, H, k_per_layer=[k], v_per_layer=[k], q_per_layer=[k], device=device
    )
    nb = (T + bs - 1) // bs
    scores = torch.einsum("bhid,bhjd->bhij", k, k) * (1.0 / math.sqrt(D))
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    scores = scores.masked_fill(~causal.view(1, 1, T, T), float("-inf"))
    a = torch.softmax(scores, dim=-1)  # [B,H,T,T]
    pad = nb * bs - T
    if pad:
        a = torch.nn.functional.pad(a, (0, pad))
    a_blk = a.view(B, H, T, nb, bs).sum(-1)  # block-sum over keys -> [B,H,T,NB]
    cum = a_blk.cumsum(dim=2).mean(dim=(0, 1))  # accumulated, shared -> [T,NB]
    for t in range(prefill_split + 1, L):
        local_first = max(0, t - local_window + 1) // bs
        elig = [b for b in range(local_first) if b >= sink_blocks]
        if not elig:
            continue
        sc = cum[t, torch.tensor(elig, device=device)]
        for i in sc.topk(min(topk, len(elig))).indices.tolist():
            lo = elig[i] * bs
            hi = min(L, (elig[i] + 1) * bs)
            mask[:, :, t, lo:hi] = True
    return mask


def _per_position_random(
    k, L, block_size, local_window, sink_blocks, topk, prefill_split, device
):
    """Per-position random-global control: base + topk random eligible blocks
    chosen fresh per decode position (the negative control on this surface)."""
    B, H, T, D = k.shape
    bs = block_size
    cfg = KRIConfig(
        block_size=bs,
        local_window_tokens=local_window,
        global_topk_blocks=0,
        prefill_split=prefill_split,
        protected_blocks=tuple(range(sink_blocks)),
    )
    mask = build_kri_mask(
        cfg,
        L,
        B,
        H,
        k_per_layer=[k],
        v_per_layer=[k],
        q_per_layer=[k],
        device=device,
    )
    for t in range(prefill_split + 1, L):
        local_first = max(0, t - local_window + 1) // bs
        elig = [b for b in range(local_first) if b >= sink_blocks]
        if not elig:
            continue
        for i in torch.randperm(len(elig))[: min(topk, len(elig))].tolist():
            lo = elig[i] * bs
            hi = min(L, (elig[i] + 1) * bs)
            mask[:, :, t, lo:hi] = True
    return mask


def _per_position_trellis(
    k, v, L, block_size, local_window, sink_blocks, topk, prefill_split, device
):
    """Per-position Trellis-as-selector: query-agnostic block write-surprise
    (scored once, O(T)) then top-k eligible blocks per decode position. The
    surprise is reduced over (batch, heads) to a shared per-block score, the
    same union-style collapse build_kri_mask uses for the KRI routers."""
    B, H, T, D = k.shape
    bs = block_size
    scores = score_trellis_blocks(k, v, bs).mean(dim=(0, 1))  # [NB], shared
    cfg = KRIConfig(
        block_size=bs,
        local_window_tokens=local_window,
        global_topk_blocks=0,
        prefill_split=prefill_split,
        protected_blocks=tuple(range(sink_blocks)),
    )
    mask = build_kri_mask(
        cfg, L, B, H, k_per_layer=[k], v_per_layer=[k], q_per_layer=[k], device=device
    )
    for t in range(prefill_split + 1, L):
        local_first = max(0, t - local_window + 1) // bs
        if local_first <= sink_blocks:
            continue
        elig = list(range(sink_blocks, local_first))
        es = scores[torch.tensor(elig, device=scores.device)]
        for j in es.topk(min(topk, len(elig))).indices.tolist():
            lo = elig[j] * bs
            hi = min(L, (elig[j] + 1) * bs)
            mask[:, :, t, lo:hi] = True
    return mask


def route_mask(
    name,
    k,
    v,
    L,
    block_size,
    local_window,
    sink_blocks,
    topk,
    device,
    lattice_kwargs=None,
    surface="canonical",
    prefill_split=0,
    q_override=None,
    mass_override=None,
):
    """Return (mask [B,H,L,L] bool, sel [B,H,K] or None) for one router.

    surface="canonical": one selection per (B,H) from the last-position probe
    (cheap, matches eval_canonical_kri). surface="per_position": route each
    decode position by its own query via build_kri_mask (faithful).

    Probe-test routers (Phase A-minus part 2): `relq_recent`/`residq_recent`
    select with `q_override` (pre-RoPE recent-Q group-mean probe, [B,H,D])
    instead of the last-position Key; `h2o_true` ranks blocks by
    `mass_override` (exact attention mass from the eager softmax)."""
    B, H, T, D = k.shape
    if surface == "per_position":
        if name in ("relq_recent", "residq_recent", "h2o_true") or name.startswith(
            "pf_"
        ):
            raise ValueError(f"{name} supports the canonical surface only")
        return _route_per_position(
            name,
            k,
            v,
            L,
            block_size,
            local_window,
            sink_blocks,
            topk,
            device,
            lattice_kwargs,
            prefill_split,
        )
    base, causal = _causal_base_mask(
        L, block_size, local_window, sink_blocks, B, H, device
    )
    if name == "full":
        return causal.view(1, 1, L, L).expand(B, H, L, L).clone(), None
    if name == "recent":
        b0, _ = _causal_base_mask(L, block_size, local_window, 0, B, H, device)
        return b0, None
    if name == "sink_recent":
        return base, None

    q_probe = k[:, :, -1, :]  # last-position Key as the query probe
    if name == "random_global":
        local_first = max(0, (L - 1) - local_window) // block_size
        elig = [b for b in range(local_first) if b >= sink_blocks]
        if not elig:
            return base, None
        keff = min(topk, len(elig))
        order = torch.randperm(len(elig))[:keff].tolist()
        sel = torch.tensor([elig[i] for i in order], device=device)
        sel = sel.view(1, 1, keff).expand(B, H, keff)
    elif name == "h2o":
        # Heavy-Hitter Oracle: top-K eligible blocks by TOTAL accumulated
        # attention mass (per head, K as query proxy). The deployed-SOTA bar
        # residual_rel must beat -- not just the internal KRI baselines.
        sc = torch.einsum("bhid,bhjd->bhij", k, k) * (1.0 / math.sqrt(D))
        cm = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        a = torch.softmax(sc.masked_fill(~cm.view(1, 1, T, T), float("-inf")), dim=-1)
        nb = (T + block_size - 1) // block_size
        if nb * block_size > T:
            a = torch.nn.functional.pad(a, (0, nb * block_size - T))
        mass = a.view(B, H, T, nb, block_size).sum(-1).sum(2)  # [B,H,NB]
        lf = max(0, (L - 1) - local_window) // block_size
        keff = min(topk, max(0, lf - sink_blocks))
        if keff == 0:
            return base, None
        mass[:, :, lf:] = -1.0
        if sink_blocks:
            mass[:, :, :sink_blocks] = -1.0
        sel = mass.topk(keff, dim=-1).indices  # [B,H,keff]
    elif name in ("relq_recent", "residq_recent"):
        assert q_override is not None, f"{name} needs q_override"
        cfg = LatticeConfig(
            block_size=block_size,
            local_window_tokens=local_window,
            global_topk_blocks=topk,
            protected_blocks=tuple(range(sink_blocks)),
            variant="rel_only" if name == "relq_recent" else "residual_rel",
            **(lattice_kwargs or {}),
        )
        sel = select_lattice_blocks(q_override.to(k.dtype), k, v, cfg, t_query=L - 1)
    elif name == "h2o_true":
        assert mass_override is not None, "h2o_true needs mass_override"
        mass = mass_override.clone()
        lf = max(0, (L - 1) - local_window) // block_size
        keff = min(topk, max(0, lf - sink_blocks))
        if keff == 0:
            return base, None
        mass[:, :, lf:] = -1.0
        if sink_blocks:
            mass[:, :, :sink_blocks] = -1.0
        sel = mass.topk(keff, dim=-1).indices
    elif name.startswith("pf_"):
        # Phase-A protected portfolio: R reserve (multi-probe) -> H reserve
        # (exact mass) -> D fill (residual order, relevance-qualified pool).
        assert q_override is not None and mass_override is not None
        ms, hs = parse_pf_name(name)
        m_n, h_n = spec_n(ms, topk), spec_n(hs, topk)
        NBv = (T + block_size - 1) // block_size
        lf = max(0, (L - 1) - local_window) // block_size
        elig = torch.zeros(NBv, dtype=torch.bool, device=device)
        elig[:lf] = True
        if sink_blocks:
            elig[:sink_blocks] = False
        NE = int(elig.sum())
        keff = min(topk, NE)
        if keff == 0:
            return base, None
        cent = _block_summaries(k, k, block_size)[0].float()  # [B,H,NB,D]
        nbig = float("-inf")

        def order_from(scores):
            s = scores.masked_fill(~elig.view(1, 1, NBv), nbig)
            return s.argsort(dim=-1, descending=True)[:, :, :NE]

        r1o = order_from(
            torch.einsum("bhd,bhnd->bhn", F.normalize(q_override.float(), dim=-1), cent)
        )
        r2o = order_from(
            torch.einsum(
                "bhd,bhnd->bhn", F.normalize(k[:, :, -1, :].float(), dim=-1), cent
            )
        )
        ho = order_from(mass_override.float())
        # qualified pool per head: union of each relevance list's top-2K
        pool_mask = torch.zeros(B, H, NBv, dtype=torch.bool, device=device)
        pool_mask.scatter_(2, r1o[:, :, : 2 * keff], True)
        pool_mask.scatter_(2, r2o[:, :, : 2 * keff], True)
        # D order: vectorised residual sequence over the pool (last-K probe)
        q_resid = F.normalize(k[:, :, -1, :].float(), dim=-1)
        chosen = torch.zeros(B, H, NBv, dtype=torch.bool, device=device)
        d_seq = []
        for _ in range(keff):
            sc = torch.einsum("bhd,bhnd->bhn", F.normalize(q_resid, dim=-1), cent)
            sc = sc.masked_fill(~(pool_mask & elig.view(1, 1, NBv)) | chosen, nbig)
            pick = sc.argmax(dim=-1)  # [B,H]
            d_seq.append(pick)
            chosen.scatter_(2, pick.unsqueeze(-1), True)
            kc = torch.gather(
                cent, 2, pick.view(B, H, 1, 1).expand(B, H, 1, cent.shape[-1])
            ).squeeze(2)
            q_resid = q_resid - (q_resid * kc).sum(-1, keepdim=True) * kc
        d_seq = torch.stack(d_seq, dim=-1)  # [B,H,keff]
        sel_rows = []
        for b in range(B):
            row = []
            for hh in range(H):
                r1l = r1o[b, hh].tolist()
                r2l = r2o[b, hh].tolist()
                kept = compose(
                    r1l,
                    r2l,
                    ho[b, hh].tolist(),
                    d_seq[b, hh].tolist(),
                    keff,
                    m_n,
                    h_n,
                    pool=qualified_pool(r1l, r2l, keff),
                )
                row.append(kept)
            sel_rows.append(row)
        sel = torch.tensor(sel_rows, dtype=torch.long, device=device)  # [B,H,keff]
    elif name == "trellis_kri":
        cfg = KRIConfig(
            block_size=block_size,
            local_window_tokens=local_window,
            global_topk_blocks=topk,
            protected_blocks=tuple(range(sink_blocks)),
        )
        sel = select_trellis_blocks(q_probe, k, v, cfg, t_query=L - 1)
    elif name.startswith("lattice_kri"):
        variant = "rel_orth" if name == "lattice_kri" else name[len("lattice_kri_") :]
        cfg = LatticeConfig(
            block_size=block_size,
            local_window_tokens=local_window,
            global_topk_blocks=topk,
            protected_blocks=tuple(range(sink_blocks)),
            variant=variant,
            **(lattice_kwargs or {}),
        )
        sel = select_lattice_blocks(q_probe, k, v, cfg, t_query=L - 1)
    else:  # kri_q | kri_q_novelty | kri
        cfg = _kri_config(name, block_size, local_window, topk, sink_blocks)
        sel = select_kri_blocks(q_probe, k, v, cfg, t_query=L - 1)

    if sel.numel() == 0:
        return base, sel
    return _select_blocks_to_mask(sel, L, block_size, B, H, base, causal), sel


def _retained(mask, prefill_split):
    """Actual retained fraction, normalised to the dense-causal baseline
    (T/2), reported overall and over the decode region. Plus mean kept
    tokens per query (the un-normalised count)."""
    B, H, T, _ = mask.shape
    keep = mask.sum(-1).float()  # [B,H,T]
    dense_avg = T / 2.0
    overall = float(keep.mean().item() / dense_avg)
    if T > prefill_split + 1:
        decode = float(keep[:, :, prefill_split + 1 :].mean().item() / dense_avg)
    else:
        decode = overall
    return overall, decode, float(keep.mean().item())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        type=str,
        required=True,
        help="comma-separated HF names or checkpoint paths",
    )
    p.add_argument("--routers", type=str, default=",".join(DEFAULT_ROUTERS))
    p.add_argument(
        "--topks",
        type=str,
        default="1,2,4,8,16",
        help="global block budgets to sweep (x-axis becomes actual frac)",
    )
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--local_window_tokens", type=int, default=128)
    p.add_argument("--sink_blocks", type=int, default=1)
    p.add_argument("--n_batches", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--streaming", type=str, default="false")
    p.add_argument("--val_split", type=str, default="validation")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--tokenizer_name", type=str, default="openai-community/gpt2")
    p.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
    )
    # Lattice knobs (apply to all lattice_* routers in this run).
    p.add_argument("--lambda_orth", type=float, default=0.25)
    p.add_argument("--lambda_redun", type=float, default=0.25)
    p.add_argument(
        "--summary_mode", type=str, default="k", choices=["k", "v", "kv_cat", "kv_sum"]
    )
    p.add_argument("--include_sink_recent_in_basis", type=str, default="false")
    p.add_argument(
        "--surface",
        type=str,
        default="canonical",
        choices=["canonical", "per_position"],
        help="canonical: one last-position probe per (B,H); per_position: "
        "route each decode token by its own query (faithful, slower)",
    )
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _truthy(s):
    return str(s).lower() in ("1", "true", "t", "yes", "y")


def main():
    args = parse_args()
    set_seed(args.seed)
    report_device()
    device = pick_device()
    dtype = pick_dtype(args.precision)
    L = args.seq_len
    prefill_split = L // 4
    routers = args.routers.split(",")
    topks = [int(x) for x in args.topks.split(",")]
    models = args.models.split(",")
    lattice_kwargs = dict(
        lambda_orth=args.lambda_orth,
        lambda_redun=args.lambda_redun,
        summary_mode=args.summary_mode,
        include_sink_recent_in_basis=_truthy(args.include_sink_recent_in_basis),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok = get_tokenizer(args.tokenizer_name)
    rows = []
    fh = out.open("w")

    for m_path in models:
        print(f"\n=== model: {m_path} ===", flush=True)
        model, tag = load_model(m_path, device)
        data_cfg = DataConfig(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            text_column=args.text_column,
            streaming=_truthy(args.streaming),
            train_split=args.train_split,
            val_split=args.val_split,
            seq_len=L,
        )
        _, val_ds = get_train_val_streams(data_cfg, tok)
        loader = DataLoader(
            val_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=0
        )
        # Seed selects a DISJOINT validation window so multi-seed runs give a
        # real variance estimate (the routers are otherwise deterministic, so
        # reusing the same first-N batches would make every seed identical).
        skip = args.seed * args.n_batches
        cached = []
        for i, b in enumerate(loader):
            if i < skip:
                continue
            if i >= skip + args.n_batches:
                break
            cached.append(b)
        if not cached:
            print("  no batches; skipping", flush=True)
            del model
            continue
        print(
            f"  cached {len(cached)} batches at L={L} (seed window "
            f"[{skip}:{skip + len(cached)}])",
            flush=True,
        )

        # Phase A-minus part 2: precompute per-batch probe/mass overrides once
        # (they depend on the model + batch, not the router/topk).
        needs_q = any(
            r in ("relq_recent", "residq_recent") or r.startswith("pf_")
            for r in routers
        )
        needs_mass = "h2o_true" in routers or any(r.startswith("pf_") for r in routers)
        overrides = []
        if needs_q or needs_mass:
            li0 = min(KRIConfig().score_layer_index, model.cfg_n_layer - 1)
            for batch in cached:
                ids = batch["input_ids"][:, :L].to(device)
                ov = {}
                if needs_q:
                    ov["q"] = capture_q_probe(model, ids, li0)["recent_mean"]
                if needs_mass:
                    ov["mass"] = capture_attn_block_mass(
                        model, ids, li0, args.block_size
                    )
                overrides.append(ov)
            print(f"  precomputed probe/mass overrides at layer {li0}", flush=True)
        else:
            overrides = [{} for _ in cached]

        for router in routers:
            # routers with no global-block budget don't vary with topk
            budgets = topks if router not in ("full", "recent", "sink_recent") else [0]
            for topk in budgets:
                if router.startswith("pf_") and not pf_valid(router, topk):
                    continue  # m+h > 0.75K -- rejected combo
                agg = {
                    "kl_mean": 0.0,
                    "kl_decode": 0.0,
                    "nll_full": 0.0,
                    "nll_sparse": 0.0,
                    "ret": 0.0,
                    "ret_decode": 0.0,
                    "kept_tok": 0.0,
                }
                for bi, batch in enumerate(cached):
                    ids = batch["input_ids"][:, :L].to(device)
                    labels = batch["labels"][:, :L].to(device)
                    full_lp, _ = _full_log_probs(model, ids, dtype, device)
                    kvs = model.collect_kv(ids)
                    k_per = [kv[0] for kv in kvs]
                    v_per = [kv[1] for kv in kvs]
                    li = min(KRIConfig().score_layer_index, len(k_per) - 1)
                    mask, _sel = route_mask(
                        router,
                        k_per[li],
                        v_per[li],
                        L,
                        args.block_size,
                        args.local_window_tokens,
                        args.sink_blocks,
                        topk,
                        device,
                        lattice_kwargs,
                        surface=args.surface,
                        prefill_split=prefill_split,
                        q_override=overrides[bi].get("q"),
                        mass_override=overrides[bi].get("mass"),
                    )
                    stats = _stats_under_mask(
                        model, ids, labels, mask, full_lp, dtype, device, prefill_split
                    )
                    ov, dc, kt = _retained(mask, prefill_split)
                    agg["kl_mean"] += stats["kl_mean"]
                    agg["kl_decode"] += stats["kl_decode"]
                    agg["nll_full"] += stats["nll_full"]
                    agg["nll_sparse"] += stats["nll_sparse"]
                    agg["ret"] += ov
                    agg["ret_decode"] += dc
                    agg["kept_tok"] += kt
                n = max(1, len(cached))
                for kk in agg:
                    agg[kk] /= n
                ppl = (
                    math.exp(agg["nll_sparse"])
                    if agg["nll_sparse"] < 30
                    else float("inf")
                )
                row = {
                    "model": tag,
                    "router": router,
                    "topk": topk,
                    "seq_len": L,
                    "block_size": args.block_size,
                    "local_window_tokens": args.local_window_tokens,
                    "sink_blocks": args.sink_blocks,
                    "prefill_split": prefill_split,
                    "dataset": args.dataset_name,
                    "seed": args.seed,
                    "kl_mean": agg["kl_mean"],
                    "kl_decode": agg["kl_decode"],
                    "nll_full": agg["nll_full"],
                    "nll_sparse": agg["nll_sparse"],
                    "ppl_sparse": ppl,
                    "retained_frac": agg["ret"],
                    "retained_frac_decode": agg["ret_decode"],
                    "mean_kept_tokens": agg["kept_tok"],
                }
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(
                    f"  {router:24s} K={topk:>3d}  ret={agg['ret']:.3f} "
                    f"KL_dec={agg['kl_decode']:.4f} PPL={ppl:.3f}",
                    flush=True,
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fh.close()
    if rows:
        csv_out = out.with_suffix(".csv")
        with csv_out.open("w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {out} and {csv_out}", flush=True)
    return 0


if __name__ == "__main__":
    import os

    os._exit(main())
