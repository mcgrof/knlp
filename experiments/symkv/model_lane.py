# SPDX-License-Identifier: GPL-2.0
"""SymKV Phase-1 model lane: does forcing the consensus head-mode ever beat
generic PCA at matched bytes on PREDICTIVE quality?

Method. Patch eager attention so that, per layer and separately for K and V, the
pre-repeat_kv head tensor (b, H, L, D) is projected onto a rank-m head-axis basis
before attention: X_hat = (B Bᵀ) X along the head axis, where B is H x m
orthonormal. B Bᵀ is the rank-m projector, so X_hat = B (Bᵀ X) is exactly the
codec reconstruction -- no extra approximation. In CALIBRATE mode the same hook
instead streams the head second moment C per (layer, K/V). In EVAL mode we install
per-(layer,K/V) projectors for a chosen (method, m) and read teacher-forced
logits, comparing to the uncompressed run by next-token KL and NLL.

Because PCA is the raw-MSE optimum, symkv_raw can only win if predictive KL
decouples from raw KV MSE. We therefore log BOTH raw-KV-MSE and predictive-KL for
every basis, plus u0 energy uᵀCu/trace(C) and the principal angle between u0 and
the PCA-m subspace -- the geometry that would explain any decoupling.

Free local W7900. Falsification-first: the controls (random_sym, mean_only,
grouped_mean) are first-class, and the verdict is written whichever way it falls.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import sys

import torch
import torch.nn.functional as F


# ---- global hook state -----------------------------------------------------
class S:
    mode = "off"            # off | calib | eval
    cov = {}               # (layer, kind) -> HeadCovariance   (kind in {"k","v"})
    proj = {}              # (layer, kind) -> (H x H) projector B Bᵀ  (eval)
    mse = {}               # (layer, kind) -> [sq_err_sum, count]     (eval)
    nkv = None


def _register(mod_name):
    global HeadCovariance
    from symkv import HeadCovariance as _HC
    HeadCovariance = _HC


def hook_factory(orig_repeat_kv):
    def fn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        li = module.layer_idx
        if S.mode == "calib":
            # key/value: (b, H, L, D). Stream head second moment over tokens.
            for kind, T in (("k", key), ("v", value)):
                acc = S.cov.setdefault((li, kind), HeadCovariance(T.shape[1]))
                # X per token is (H, D): move to (b*L, H, D)
                X = T.permute(0, 2, 1, 3).reshape(-1, T.shape[1], T.shape[3])
                acc.update(X)
        elif S.mode == "eval":
            key = _apply_proj(key, li, "k")
            value = _apply_proj(value, li, "v")
        ks = orig_repeat_kv(key, module.num_key_value_groups)
        vs = orig_repeat_kv(value, module.num_key_value_groups)
        aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
        if attention_mask is not None:
            aw = aw + attention_mask[:, :, :, : ks.shape[-2]]
        aw = torch.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(aw, vs).transpose(1, 2).contiguous(), aw
    return fn


def _apply_proj(T, li, kind):
    P = S.proj.get((li, kind))
    if P is None:
        return T
    # T: (b, H, L, D); X_hat = einsum head axis with projector P (H x H)
    Th = torch.einsum("hg,bgld->bhld", P.to(T.dtype), T)
    if S.mse is not None:
        e = S.mse.setdefault((li, kind), [0.0, 0])
        e[0] += float(torch.sum((T.float() - Th.float()) ** 2))
        e[1] += T.numel()
    return Th


# ---- basis geometry diagnostics -------------------------------------------
def u0_energy(C, H):
    from symkv import consensus_mode
    u0 = consensus_mode(H).to(C.dtype)
    return float((u0 @ C @ u0) / torch.diagonal(C).sum())


def principal_angle_u0_pca(C, H, m):
    from symkv import consensus_mode, build_basis
    u0 = consensus_mode(H).to(C.dtype)
    V = build_basis("pca_head", H, m, C=C)            # H x m
    # cos of angle between u0 and its projection onto span(V)
    proj = V @ (V.transpose(-1, -2) @ u0)
    c = float(torch.linalg.norm(proj) / torch.linalg.norm(u0))
    return math.degrees(math.acos(max(0.0, min(1.0, c))))


# ---- driver ----------------------------------------------------------------
def kl_nll(logits_ref, logits_test, target_ids):
    """Mean next-token KL(ref||test) and NLL delta over the scored positions."""
    lr = F.log_softmax(logits_ref.float(), -1)
    lt = F.log_softmax(logits_test.float(), -1)
    kl = (lr.exp() * (lr - lt)).sum(-1).mean().item()
    nll_ref = F.nll_loss(lr, target_ids)
    nll_test = F.nll_loss(lt, target_ids)
    return kl, (nll_test - nll_ref).item()


def run(model_id, n_calib, n_eval, ctxs, modes, methods, corpus_path, device,
        out_path, pkg):
    sys.path.insert(0, pkg)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import importlib
    _register(None)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    mm = importlib.import_module(model.__class__.__module__)
    orig_repeat_kv, orig_eager = mm.repeat_kv, mm.eager_attention_forward
    mm.eager_attention_forward = hook_factory(orig_repeat_kv)
    H = model.config.num_key_value_heads
    corpus = open(corpus_path).read()
    ids0 = tok(corpus).input_ids

    def chunk_text(i, n):
        a = (i * 917) % max(1, len(ids0) - n)
        return ids0[a:a + n]

    results = []
    for ctx in ctxs:
        # ---- calibrate C on n_calib disjoint prompts ----
        S.mode = "calib"; S.cov = {}
        with torch.no_grad():
            for i in range(n_calib):
                ids = torch.tensor([chunk_text(i, ctx)], device=device)
                model(ids, use_cache=False)
        Cs = {key: acc.covariance() for key, acc in S.cov.items()}
        n_layers = model.config.num_hidden_layers

        # ---- reference logits on eval prompts (uncompressed) ----
        S.mode = "off"
        eval_ids = [torch.tensor([chunk_text(1000 + i, ctx)], device=device)
                    for i in range(n_eval)]
        ref_logits = []
        with torch.no_grad():
            for ids in eval_ids:
                ref_logits.append(model(ids, use_cache=False).logits[0, -64:].cpu())

        # geometry per layer (averaged over layers, k and v)
        from symkv import build_basis
        for method in methods:
            for m in modes:
                if method == "full" and m != H:
                    continue
                if method == "mean_only" and m != 1:
                    continue
                seeds = [0, 1, 2, 3, 4] if method == "random_sym" else [0]
                for seed in seeds:
                    S.proj = {}
                    for (li, kind), C in Cs.items():
                        B = build_basis(method, H, m, C=C, seed=seed)
                        S.proj[(li, kind)] = (B @ B.transpose(-1, -2)).to(device)
                    S.mode = "eval"; S.mse = {}
                    kls, nlls = [], []
                    with torch.no_grad():
                        for ids, rl in zip(eval_ids, ref_logits):
                            tl = model(ids, use_cache=False).logits[0, -64:].cpu()
                            tgt = ids[0, -63:].cpu()
                            kl, dnll = kl_nll(rl[:-1], tl[:-1], tgt)
                            kls.append(kl); nlls.append(dnll)
                    raw_mse = (sum(v[0] for v in S.mse.values())
                               / max(1, sum(v[1] for v in S.mse.values())))
                    from symkv import byte_accounting
                    ba = byte_accounting(H, model.config.head_dim if hasattr(model.config, "head_dim")
                                         else Cs[(0, "k")].shape[0], m, n_tokens=ctx)
                    row = {
                        "model": model_id, "ctx": ctx, "method": method, "modes": m,
                        "seed": seed, "kl_mean": sum(kls) / len(kls),
                        "nll_delta_mean": sum(nlls) / len(nlls), "raw_kv_mse": raw_mse,
                        "compression_ratio": ba["compression_ratio"],
                        "u0_energy": (sum(u0_energy(C, H) for C in Cs.values()) / len(Cs)),
                        "angle_u0_pca_deg": (sum(principal_angle_u0_pca(C, H, m) for C in Cs.values())
                                             / len(Cs)) if m < H else 0.0,
                    }
                    results.append(row)
                    print(f"[{model_id.split('/')[-1]}|ctx{ctx}] {method:12} m={m} seed={seed} "
                          f"KL={row['kl_mean']:.4f} dNLL={row['nll_delta_mean']:.4f} "
                          f"rawMSE={raw_mse:.4e} cr={row['compression_ratio']:.2f} "
                          f"u0E={row['u0_energy']:.3f} ang={row['angle_u0_pca_deg']:.1f}", flush=True)
        S.mode = "off"
    mm.eager_attention_forward = orig_eager
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {len(results)} rows -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--ctxs", default="512,1024")
    ap.add_argument("--modes", default="1,2,4,8")
    ap.add_argument("--methods", default="full,mean_only,random_sym,grouped_mean,pca_head,symkv_raw")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pkg", default="/data/knlp", help="dir containing the symkv package")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    run(a.model, a.n_calib, a.n_eval, [int(x) for x in a.ctxs.split(",")],
        [int(x) for x in a.modes.split(",")], a.methods.split(","),
        a.corpus, a.device, a.out, a.pkg)
    print(f"elapsed {time.time()-t0:.1f}s")
