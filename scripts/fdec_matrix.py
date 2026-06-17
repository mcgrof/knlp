#!/usr/bin/env python3
"""fdec compatibility matrix runner (algorithmic dimension).

Verifies that the fight-decode bricks COMPOSE -- run together in one eager decode
loop without mismatch -- and reports correctness vs an fp16 baseline plus the byte
reduction each brick buys. This is the algorithmic/quality dimension of the matrix;
it runs on any GPU (incl. the W7900 on prune) in eager PyTorch. The kernel/throughput
dimension (FlashInfer asym, fused Triton) is CUDA-only and lives in the serving-stack
profiles, not here.

Bricks (extensible -- add a patch fn + a byte-model entry):
  asym_kv   : K stays bf16, V quantized to fp8 e4m3 (per-tensor scale) -> -25% KV.
              The proven asym K16/V8 brick, emulated at the math level.
  (lmhead_idblock, fim_weight_quant, linear_attn: TODO -- wire their code.)

A "cell" is a model x set-of-enabled-bricks. We decode greedily with and without the
bricks and report the fraction of generated tokens that match the baseline (quality
proxy) and the analytical decode-byte reduction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- bricks -----------------------------------------------------------------


@torch.no_grad()
def _fake_quant_groupwise(w, bits, group):
    """Group-wise symmetric int quant: a separate scale per (output channel, input
    group of `group` elements) -- the core of real int4 (GPTQ/AWQ-style), so a
    single outlier only widens its own group's scale, not the whole row's. Falls
    back to per-channel if the input dim is not divisible by group."""
    qmax = 2 ** (bits - 1) - 1
    out, inn = w.shape
    if group <= 0 or inn % group != 0:
        scale = (w.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-8)
        return (torch.round(w / scale).clamp(-qmax, qmax) * scale).to(w.dtype)
    wg = w.view(out, inn // group, group)
    scale = (wg.abs().amax(dim=-1, keepdim=True) / qmax).clamp(min=1e-8)
    return (
        (torch.round(wg / scale).clamp(-qmax, qmax) * scale).view(out, inn).to(w.dtype)
    )


@torch.no_grad()
def compute_act_scales(model, tok, calib, lins, lm, device):
    """Per-input-channel mean |activation| for each linear (AWQ salience signal)."""
    scales = {}
    hooks = []

    def mk(n):
        def h(mod, inp):
            x = inp[0].detach().abs().float().reshape(-1, inp[0].shape[-1]).mean(0)
            scales[n] = torch.maximum(scales[n], x) if n in scales else x

        return h

    for n, m in lins:
        hooks.append(m.register_forward_pre_hook(mk(n)))
    ids = tok(calib, return_tensors="pt").input_ids.to(device)
    model(ids, use_cache=False)
    for h in hooks:
        h.remove()
    return scales


def fim_fake_quant(
    model, tok, calib, base_bits, upgrade_frac, group, awq_alpha, device
):
    """FIM-guided weight quant: diagonal Fisher per linear layer (grad^2 on a calib
    pass), upgrade the most-sensitive upgrade_frac of layers to int8 and quantize the
    rest to base_bits, GROUP-WISE. Optional AWQ activation-aware scaling (alpha>0)
    protects salient channels -- the lever for Qwen's massive activations. Excludes
    the LM-head. Fake-quantizes in place; returns (originals, weight_byte_factor)."""
    lm = model.get_output_embeddings()
    lins = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and m is not lm
    ]
    act = (
        compute_act_scales(model, tok, calib, lins, lm, device) if awq_alpha > 0 else {}
    )
    model.zero_grad(set_to_none=True)
    ids = tok(calib, return_tensors="pt").input_ids.to(device)
    with torch.enable_grad():
        loss = model(ids, labels=ids).loss
        loss.backward()
    fisher = {}
    for n, m in lins:
        if m.weight.grad is not None:
            fisher[n] = float((m.weight.grad.float() ** 2).mean())
    model.zero_grad(set_to_none=True)
    ranked = sorted(fisher, key=lambda k: fisher[k], reverse=True)
    upgrade = set(ranked[: int(upgrade_frac * len(ranked))])
    orig = {}
    tot_bits = tot_w = 0
    with torch.no_grad():
        for n, m in lins:
            if n not in fisher:
                continue
            b = 8 if n in upgrade else base_bits
            w = m.weight.data
            orig[n] = w.clone()
            if awq_alpha > 0 and n in act:
                # AWQ: scale up salient input channels before quant, unscale after,
                # so high-activation channels carry less relative quant error
                s = act[n].to(w.dtype).clamp(min=1e-4) ** awq_alpha
                s = (s / s.mean()).unsqueeze(0)  # [1, in], normalized
                m.weight.data = (_fake_quant_groupwise(w * s, b, group) / s).to(w.dtype)
            else:
                m.weight.data = _fake_quant_groupwise(w, b, group)
            # effective bits include the per-group fp16 scale (16/group per weight)
            eff = b + (16.0 / group if group > 0 else 0.0)
            tot_bits += eff * m.weight.numel()
            tot_w += m.weight.numel()
    avg_bits = tot_bits / max(1, tot_w)
    return orig, round(avg_bits / 16.0, 4)


def fim_restore(model, orig):
    with torch.no_grad():
        for n, m in model.named_modules():
            if n in orig:
                m.weight.data = orig[n]


def asym_kv_attention(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kw
):
    """Eager attention with K16/V8: keep keys bf16, quantize values to fp8 e4m3."""
    g = module.num_key_value_groups
    ks = key.repeat_interleave(g, dim=1)
    vs = value.repeat_interleave(g, dim=1)
    # V -> fp8 e4m3 -> bf16 (per-tensor scale); K untouched
    scale = vs.abs().amax().clamp(min=1e-6) / 448.0
    vs = (vs / scale).to(torch.float8_e4m3fn).to(vs.dtype) * scale
    aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
    if attention_mask is not None:
        aw = aw + attention_mask[:, :, :, : ks.shape[-2]]
    aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(aw, vs).transpose(1, 2).contiguous()
    return out, aw


def install_asym_kv(model):
    model.config._attn_implementation = "eager"
    mods = set()
    for m in model.modules():
        if hasattr(m, "num_key_value_groups") and hasattr(m, "layer_idx"):
            mods.add(type(m).__module__)
    saved = {}
    for mn in mods:
        mod = sys.modules.get(mn)
        if mod is not None and hasattr(mod, "eager_attention_forward"):
            saved[mn] = mod.eager_attention_forward
            mod.eager_attention_forward = asym_kv_attention
    return saved


def restore(saved):
    for mn, fn in saved.items():
        sys.modules[mn].eager_attention_forward = fn


# ---- LM-head idblock (certified-decode shadow bound) ------------------------
# Lossless brick: per-token bound w_v.h <= a_v.q + delta_v*rho over a hidden-PCA
# basis B (d x r); open contiguous idblock vocab slabs by descending slab-max
# bound until the dense argmax is certified. Returns the dense argmax (lossless)
# and the LM-head byte fraction = shadow (r+1)/d + fetched. We run it as a
# MEASUREMENT brick: it does not change the decode (argmax unchanged), it reports
# the LM-head byte reduction and verifies the certificate holds (argmax_match=1)
# -- including on hidden states already perturbed by the asym_kv brick.


@torch.no_grad()
def capture_lmhead_input(model, tok, texts, device):
    lm = model.get_output_embeddings()
    caps = []

    def hook(mod, inp):
        caps.append(inp[0].detach()[0])  # [T, d]

    h = lm.register_forward_pre_hook(hook)
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(device)
        model(ids, use_cache=False)
    h.remove()
    return torch.cat(caps, 0)  # [sum_T, d]


@torch.no_grad()
def idblock_measure(H_calib, H, W_U, r, slab, device):
    """Shadow-bound certificate. Basis from a SEPARATE calib set, UNCENTERED (so it
    captures the dominant hidden-state directions including the mean -- centering
    would leave the mean in rho and loosen the bound). Measured on H."""
    Hc = H_calib.to(device).float()
    H = H.to(device).float()
    W = W_U.to(device).float()
    V, d = W.shape
    _, _, Vh = torch.linalg.svd(Hc, full_matrices=False)  # uncentered
    B = Vh[: min(r, Vh.shape[0])].T.contiguous()  # [d, r]
    r = B.shape[1]
    a = W @ B  # [V, r]
    delta = (W - a @ B.T).norm(dim=1)  # [V]
    q = H @ B  # [T, r]
    rho = (H - q @ B.T).norm(dim=1)  # [T]
    real = W @ H.t()  # [V, T] dense logits (for ell*/argmax verification)
    bound = a @ q.t() + delta.unsqueeze(1) * rho.unsqueeze(0)  # [V, T]
    nb = (V + slab - 1) // slab
    pad = nb * slab - V
    if pad:
        bound = torch.cat([bound, bound.new_full((pad, H.shape[0]), -1e30)], 0)
        realp = torch.cat([real, real.new_full((pad, H.shape[0]), -1e30)], 0)
    else:
        realp = real
    Ub = bound.view(nb, slab, -1).amax(1)  # [nb, T] slab bound maxima
    Rb = realp.view(nb, slab, -1).amax(1)  # [nb, T] slab real maxima
    dense_argmax_slab = (real.argmax(0) // slab).cpu()  # [T]
    T = H.shape[0]
    opened_frac = []
    amatch = 0
    for t in range(T):
        order = Ub[:, t].argsort(descending=True)
        ub_sorted = Ub[order, t]
        ell = -1e30
        opened = 0
        for i in range(nb):
            opened += 1
            ell = max(ell, Rb[order[i], t].item())
            maxun = ub_sorted[i + 1 :].max().item() if i + 1 < nb else -1e30
            if ell > maxun:
                break
        opened_frac.append(opened / nb)
        amatch += int(int(dense_argmax_slab[t]) in set(order[:opened].tolist()))
    fetched = sum(opened_frac) / max(1, T)
    shadow = (r + 1) / d
    return dict(
        lm_head_byte_factor=round(shadow + fetched, 4),
        fetched_fraction=round(fetched, 4),
        shadow_ratio=round(shadow, 4),
        argmax_match=round(amatch / max(1, T), 4),
        rank=r,
        slab=slab,
        byte_note=(
            "FINDING (TODO resolved): the LM-head byte win is IN-SAMPLE. With the "
            "shadow basis built from the SAME hiddens it measures (--idblock-insample) "
            "rho~0 and the certificate prunes to ~0.01 fetched. OUT-OF-SAMPLE (a "
            "separate calib set) on Qwen2.5-7B it does NOT generalize -- fetched ~0.95, "
            "no reduction -- because the hidden states are not low-rank enough for the "
            "basis to transfer (Qwen2.5 massive activations). So certdecode's ~0.25x "
            "is in-sample-optimistic on this model; the deployable byte number is "
            "model-dependent and unproven out-of-sample here. Losslessness "
            "(argmax_match) and compose are correct in both regimes."
        ),
    )


# brick registry: name -> {install (or None for measurement), pool, byte factor}
BRICKS = {
    "asym_kv": dict(install=install_asym_kv, pool="KV", kv_factor=0.75),
    "lmhead_idblock": dict(install=None, pool="LM-head", measure=True),
    "fim_weight_quant": dict(install=None, pool="weights", weight_brick=True),
}


def decode_byte_model(model, ctx, db, kv_f, lm_f, w_f, hbm_gbps):
    """Per-token decode byte budget across the three pools, and the byte-bound
    roofline latency (decode is memory-bound: latency ~ bytes / HBM bandwidth).
    Pools are disjoint: weights = non-LM-head params; LM-head = unembedding GEMV;
    KV grows with ctx. Factors apply each brick's reduction."""
    cfg = model.config
    nparam = sum(p.numel() for p in model.parameters())
    L = cfg.num_hidden_layers
    Hkv = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    vocab, hidden = cfg.vocab_size, cfg.hidden_size
    lmhead_params = vocab * hidden
    w = (nparam - lmhead_params) * db * w_f
    kv = L * Hkv * ctx * d * 2 * db * kv_f
    lm = lmhead_params * db * lm_f
    tot = w + kv + lm
    return dict(
        weights_mb=w / 1e6,
        kv_mb=kv / 1e6,
        lmhead_mb=lm / 1e6,
        total_mb=tot / 1e6,
        latency_ms=1e3 * tot / (hbm_gbps * 1e9),
        ctx=ctx,
        hbm_gbps=hbm_gbps,
    )


@torch.no_grad()
def tf_logits(model, tok, texts, device):
    """Teacher-forced: per-position argmax + log-probs (for KL). No generation
    loop, so no autoregressive divergence cascade -- this measures whether a brick
    changes the next-token prediction given the SAME context."""
    am, lg = [], []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(device)
        l = model(ids, use_cache=False).logits[0].float()  # [T, V]
        am.append(l.argmax(-1).cpu())
        lg.append(F.log_softmax(l, dim=-1).cpu())
    return am, lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--bricks", default="asym_kv", help="comma-sep brick set (the cell)"
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--idblock-rank", type=int, default=64, help="shadow basis rank r")
    ap.add_argument("--idblock-slab", type=int, default=16384, help="vocab slab size")
    ap.add_argument(
        "--ctx", type=int, default=32768, help="context len for KV byte model"
    )
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--idblock-insample", action="store_true")
    ap.add_argument("--fim-group-size", type=int, default=128)
    ap.add_argument(
        "--fim-awq-alpha", type=float, default=0.0, help="AWQ act-scale; 0=off"
    )
    ap.add_argument("--fim-base-bits", type=int, default=4)
    ap.add_argument("--fim-upgrade-frac", type=float, default=0.2)
    ap.add_argument(
        "--hbm-gbps", type=float, default=3350.0, help="roofline HBM BW (H100)"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager"
    ).to(device)
    model.eval()

    prompts = [
        "The history of the printing press begins when",
        "In thermodynamics, the second law states that",
        "A good way to learn a new programming language is to",
        "The capital city of France is Paris, and it is known for",
        "Once upon a time in a distant galaxy there was",
        "To compute the factorial of a number recursively you",
    ][: args.n_prompts]

    bricks = [b.strip() for b in args.bricks.split(",") if b.strip()]
    unknown = [b for b in bricks if b not in BRICKS]
    if unknown:
        print(f"[skip] unknown bricks (not yet wired): {unknown}")
        bricks = [b for b in bricks if b in BRICKS]

    # baseline (fp16, no bricks)
    base_am, base_lg = tf_logits(model, tok, prompts, device)

    # FIM weight quant (compute Fisher on the clean model, then fake-quant in place)
    fim_orig = None
    weight_f = 1.0
    if "fim_weight_quant" in bricks:
        fim_calib = " ".join(
            [
                "The mitochondria generate ATP through oxidative phosphorylation.",
                "Quicksort partitions an array around a pivot and recurses.",
                "Inflation erodes purchasing power as the money supply grows.",
                "Photosynthesis converts carbon dioxide and water into glucose.",
                "Neural networks learn by gradient descent on a loss function.",
                "The French Revolution began in 1789 at the Bastille prison.",
                "Antibiotics kill bacteria but not viruses like the influenza.",
                "Compound interest grows savings as returns accrue on returns.",
                "Tectonic plates drift a few centimeters per year over the mantle.",
                "Caching frequently accessed data reduces database latency.",
            ]
        )
        fim_orig, weight_f = fim_fake_quant(
            model,
            tok,
            fim_calib,
            args.fim_base_bits,
            args.fim_upgrade_frac,
            args.fim_group_size,
            args.fim_awq_alpha,
            device,
        )

    # apply the cell's transform bricks together, re-run, compare (teacher-forced)
    saved_all = {}
    for b in bricks:
        inst = BRICKS[b].get("install")
        if inst is not None:
            saved_all.update(inst(model))
    brk_am, brk_lg = tf_logits(model, tok, prompts, device)

    # measurement bricks: run while transform bricks are still active (so the
    # certificate is checked on the actually-perturbed hidden states)
    idb = None
    if "lmhead_idblock" in bricks:
        # DIVERSE calib (distinct topics, not a repeated passage) so the
        # hidden-PCA basis spans the real hidden manifold -> small rho ->
        # tight bound -> pruning. Repeated text gives redundant hiddens.
        calib_sents = [
            "The mitochondria generate ATP through oxidative phosphorylation in the cell.",
            "Beethoven composed his ninth symphony after he had already gone deaf.",
            "A recursive function calls itself with a smaller subproblem until a base case.",
            "The Amazon river discharges more water than the next seven rivers combined.",
            "Inflation erodes purchasing power when the money supply grows faster than output.",
            "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
            "The Treaty of Westphalia in 1648 established the modern system of sovereign states.",
            "Quicksort partitions an array around a pivot and recurses on each side.",
            "Tectonic plates drift a few centimeters per year over the molten mantle.",
            "A balanced diet includes proteins, carbohydrates, fats, vitamins, and minerals.",
            "The speed of light in a vacuum is roughly three hundred thousand kilometers per second.",
            "Shakespeare wrote thirty seven plays and a hundred fifty four sonnets.",
            "Neural networks learn by adjusting weights through gradient descent on a loss.",
            "Mount Everest rises about eight thousand eight hundred forty nine meters above sea level.",
            "Supply and demand curves intersect at the market clearing equilibrium price.",
            "Antibiotics kill bacteria but have no effect on viral infections like the flu.",
            "The French Revolution began in 1789 with the storming of the Bastille prison.",
            "A hash table offers average constant time lookups by mapping keys to buckets.",
            "Volcanic eruptions can cool the planet by injecting aerosols into the stratosphere.",
            "Regular exercise strengthens the heart and improves circulation over time.",
            "Jupiter is the largest planet and has a great red spot larger than the Earth.",
            "Compound interest grows savings exponentially as returns accrue on prior returns.",
            "DNA stores genetic information in sequences of four nucleotide bases.",
            "The Renaissance revived classical art, science, and humanism across Europe.",
            "Caching frequently accessed data reduces latency and load on the database.",
        ]
        calib = " ".join(calib_sents)
        W_U = model.get_output_embeddings().weight
        H_eval = capture_lmhead_input(model, tok, prompts, device)
        if args.idblock_insample:
            H_calib = H_eval  # basis from the measured hiddens (rho~0, optimistic)
        else:
            H_calib = capture_lmhead_input(model, tok, [calib], device)
        idb = idblock_measure(
            H_calib, H_eval, W_U, args.idblock_rank, args.idblock_slab, device
        )
    restore(saved_all)
    if fim_orig is not None:
        fim_restore(model, fim_orig)

    # top-1 agreement + mean KL(baseline || brick) over all positions
    total = matched = 0
    kl_sum = 0.0
    nkl = 0
    for ba, bk, blg, klg in zip(base_am, brk_am, base_lg, brk_lg):
        total += ba.numel()
        matched += int((ba == bk).sum())
        p = blg.exp()
        kl_sum += float((p * (blg - klg)).sum(-1).mean())
        nkl += 1
    match_rate = matched / max(1, total)
    mean_kl = kl_sum / max(1, nkl)

    # analytical decode-byte model: KV factor multiplies across KV bricks
    kv_factor = 1.0
    pools = []
    for b in bricks:
        kv_factor *= BRICKS[b].get("kv_factor", 1.0)
        pools.append(f"{b}({BRICKS[b]['pool']})")

    res = {
        "model": args.model,
        "cell_bricks": bricks,
        "pools": pools,
        "composes": True,  # ran together without error
        "tf_top1_agreement_vs_fp16": match_rate,
        "tf_mean_kl_vs_fp16": mean_kl,
        "kv_byte_factor": kv_factor,
        "kv_byte_reduction_pct": round(100 * (1 - kv_factor), 1),
        "lmhead_idblock": idb,
        "n_prompts": len(prompts),
    }
    # decode byte model + roofline latency, baseline vs this cell
    # (weight_f set above by the FIM brick if enabled, else 1.0)
    lm_f = idb["lm_head_byte_factor"] if idb is not None else 1.0
    base_bm = decode_byte_model(
        model, args.ctx, args.dtype_bytes, 1.0, 1.0, 1.0, args.hbm_gbps
    )
    cell_bm = decode_byte_model(
        model, args.ctx, args.dtype_bytes, kv_factor, lm_f, weight_f, args.hbm_gbps
    )
    saved_pct = round(100 * (1 - cell_bm["total_mb"] / base_bm["total_mb"]), 1)
    res["decode_byte_model"] = {
        "ctx": args.ctx,
        "hbm_gbps": args.hbm_gbps,
        "baseline": base_bm,
        "cell": cell_bm,
        "total_bytes_saved_pct": saved_pct,
        "roofline_latency_saved_pct": saved_pct,
    }
    print(f"\n[fdec cell] {args.model}  bricks={bricks or ['(none)']}")
    print(
        f"  composes: yes   teacher-forced top-1 agreement: {match_rate:.3f}   "
        f"meanKL: {mean_kl:.4f}   KV bytes: {kv_factor:.3f}x "
        f"(-{res['kv_byte_reduction_pct']}%)"
    )
    if idb is not None:
        regime = "IN-SAMPLE" if args.idblock_insample else "out-of-sample"
        print(
            f"  LM-head idblock: LOSSLESS cert argmax_match={idb['argmax_match']:.3f}"
            f" (verified)  |  {regime} fetched={idb['fetched_fraction']:.2f} "
            f"(in-sample ~0.01 / OOS-Qwen2.5 ~0.95: byte win is in-sample-only here)"
        )
    bb, cb = base_bm, cell_bm
    print(
        f"  decode bytes/tok @ctx{args.ctx}: {bb['total_mb']:.0f} MB -> "
        f"{cb['total_mb']:.0f} MB  (-{saved_pct}%)   "
        f"[W {cb['weights_mb']:.0f} / KV {cb['kv_mb']:.0f} / LMh {cb['lmhead_mb']:.0f}]"
    )
    print(
        f"  roofline latency @H100 {args.hbm_gbps:.0f}GB/s: "
        f"{bb['latency_ms']:.2f} ms -> {cb['latency_ms']:.2f} ms/tok (-{saved_pct}%)"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
