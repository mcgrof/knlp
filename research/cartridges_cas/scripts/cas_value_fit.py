#!/usr/bin/env python3
"""Fit a cartridge's values to the teacher's attention outputs.

Cartridges self-study trains every key and value of the prefix by
backpropagating a distillation loss through the whole frozen reader.
Latent Compaction (arXiv 2610.20582) gets most of its quality from a
cheaper first stage: fit each layer's compact cache locally, so that
attention over it reproduces the attention output the reader computes
over the real context. Attention Matching gives a closed form for one
piece of that: with the keys fixed, the values that best reproduce the
teacher's attention outputs are a least-squares solution. This script
implements that value-only variant for a native cartridge. Keys and the
frozen sink stay as they are; only the trainable values change, and the
artifact keeps the ordinary bias-free cartridge layout.

``capture`` (GPU) runs the frozen reader as the teacher. The Self-Study
conversations were generated with one section of the record in the
system prompt, so each conversation's teacher context is its own row's
``system_prompt``. For every distinct system prompt the reader prefills
that context once. Each selected conversation is then run after it, and
at sampled scored positions the script records the query of every head
before rotary position embedding, the attention output over the system
prompt only, and its log-normalizer. Conversations are split into fit
and validation sets by whole prompt group (the first user message), so
duplicate prompts never straddle the split.

``solve`` (CPU or GPU) loads a cartridge and the captured queries. For
layer l and key-value head h, with the cartridge's keys K (frozen sink
first) and each captured query rotated to the position it has when the
cartridge stands in for the context (cartridge length plus its offset in
the conversation), the attention weights X = softmax(q K^T / sqrt(d))
are fixed. The sink's contribution is subtracted from the target, and
the trainable values solve

    min_V ||X_t V - (Y - x_0 v_sink)||^2 + lam ||V - V0||^2

by least squares on the stacked system (QR, no explicit inverse), for
every query head that shares the key-value head. The ridge pulls toward
the starting values; lam is chosen per layer on the validation queries
from a fixed grid scaled by the mean diagonal of X_t^T X_t. The output
cartridge has the same keys, sink and file layout with the fitted
values. A JSON summary gives, per layer, the validation relative error
of the attention output before and after the fit, the chosen lam, the
value drift, and the gap between the teacher's log-normalizer and the
cartridge's (which a value-only fit cannot change).

Env (capture):
    MODEL       reader (default Qwen/Qwen3-8B)
    PARQUET     Self-Study parquet (messages, system_prompt)
    N_FIT       fit conversations (default 192)
    N_VAL       validation conversations (default 64)
    Q_PER_CONV  scored positions sampled per conversation (default 32)
    SEED        sampling seed (default 0)
    VAL_FRAC    fraction of prompt groups held out (default 0.2)
    OUT_DIR     capture directory (one file per layer, plus meta.json)
    DEVICE      cuda (default)

Env (solve):
    CAPTURE     capture directory
    CART        starting cartridge (its keys and sink are kept)
    OUT_CART    fitted cartridge
    OUT_JSON    summary (default OUT_CART with .json)
    LAMS        comma list of ridge scales (default 1e-4,1e-3,1e-2,1e-1,1)
    ROPE_THETA  rotary base (default from the capture's model config)
    DEVICE      cpu (default) or cuda

Usage:
    python cas_value_fit.py capture
    python cas_value_fit.py solve
    python cas_value_fit.py selftest
"""

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SYSTEM_START = [151644, 8948, 198]  # <|im_start|>system\n
SYSTEM_END = [151645, 198]  # <|im_end|>\n


def rope(x, pos, theta):
    """Rotate x [..., N, d] at integer positions pos [N] with the
    half-split convention Qwen3 uses (rotate_half)."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    ang = pos.to(torch.float64)[:, None] * inv[None, :]
    emb = torch.cat([ang, ang], dim=-1)
    cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


def group_split(convs, val_frac, seed):
    """Split conversation indices by prompt group (first user message)."""
    fit, val = [], []
    for i, c in enumerate(convs):
        user = next(m.content for m in c.messages if m.role == "user")
        h = hashlib.sha256(f"{seed}:{user}".encode()).digest()
        (val if h[0] / 256.0 < val_frac else fit).append(i)
    return fit, val


# --------------------------------------------------------------- capture
def capture():
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    from cartridges.datasets import qwen_messages_to_element, read_conversations

    model_name = os.environ.get("MODEL", "Qwen/Qwen3-8B")
    parquet = os.environ["PARQUET"]
    n_fit = int(os.environ.get("N_FIT", "192"))
    n_val = int(os.environ.get("N_VAL", "64"))
    q_per = int(os.environ.get("Q_PER_CONV", "32"))
    seed = int(os.environ.get("SEED", "0"))
    val_frac = float(os.environ.get("VAL_FRAC", "0.2"))
    out = Path(os.environ["OUT_DIR"])
    dev = os.environ.get("DEVICE", "cuda")
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(dev)
    model.eval()
    cfg = model.config
    L, H, KV, D = (
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
    )
    theta = float(getattr(cfg, "rope_theta", 1e6))
    rep = H // KV

    rows = read_conversations(parquet)
    fit_i, val_i = group_split(rows, val_frac, seed)
    rng = np.random.default_rng(seed)
    fit_i = sorted(rng.choice(fit_i, min(n_fit, len(fit_i)), replace=False))
    val_i = sorted(rng.choice(val_i, min(n_val, len(val_i)), replace=False))
    sel = [(i, "fit") for i in fit_i] + [(i, "val") for i in val_i]
    by_sys = {}
    for i, split in sel:
        by_sys.setdefault(rows[i].system_prompt, []).append((i, split))
    print(
        f"[vfit] {len(rows)} rows; {len(fit_i)} fit + {len(val_i)} val "
        f"conversations over {len(by_sys)} system prompts",
        flush=True,
    )

    # the query of every head after q_norm, before rotary embedding
    qbuf = {}

    def hook(layer):
        def f(_m, _inp, outp):
            qbuf[layer] = outp.detach()  # [1, T, H, D]

        return f

    for li, layer in enumerate(model.model.layers):
        layer.self_attn.q_norm.register_forward_hook(hook(li))

    # attention output before o_proj, to check the position convention
    obuf = {}
    check_layers = sorted({0, L // 2, L - 1})

    def ohook(layer):
        def f(_m, inp):
            obuf[layer] = inp[0].detach()  # [1, T, H * D]

        return f

    for li in check_layers:
        model.model.layers[li].self_attn.o_proj.register_forward_pre_hook(ohook(li))
    checks = []

    store = {
        s: {"q": [[] for _ in range(L)], "y": [[] for _ in range(L)]}
        for s in ("fit", "val")
    }
    store_lse = {s: [[] for _ in range(L)] for s in ("fit", "val")}
    store_t = {"fit": [], "val": []}
    n_tok = {"system": 0, "conv": 0}
    with torch.no_grad():
        for sysp, items in by_sys.items():
            sys_ids = SYSTEM_START + tok.encode(sysp, add_special_tokens=False)
            sys_ids += SYSTEM_END
            Dsys = len(sys_ids)
            n_tok["system"] += Dsys
            cache = DynamicCache()
            model.model(
                input_ids=torch.tensor([sys_ids], device=dev),
                past_key_values=cache,
                use_cache=True,
            )
            kdoc = [cache.layers[l].keys[0, :, :Dsys].float() for l in range(L)]
            vdoc = [cache.layers[l].values[0, :, :Dsys].float() for l in range(L)]
            for i, split in items:
                el = qwen_messages_to_element(rows[i].messages, tokenizer=tok)
                ids = el.input_ids.tolist()
                T = len(ids)
                n_tok["conv"] += T
                # scored positions: the token before each distilled token
                cand = np.unique(el.topk_token_idxs.numpy() - 1)
                cand = cand[(cand >= 0) & (cand < T)]
                pick = np.sort(rng.choice(cand, min(q_per, len(cand)), replace=False))
                model.model(
                    input_ids=torch.tensor([ids], device=dev),
                    position_ids=torch.arange(Dsys, Dsys + T, device=dev)[None],
                    past_key_values=cache,
                    use_cache=True,
                )
                pt = torch.as_tensor(pick, device=dev)
                if not checks:
                    # rebuild the full causal attention output at the picked
                    # positions from the captured queries and the cache
                    for l in check_layers:
                        q = qbuf[l][0, pt].float().transpose(0, 1)
                        qr = rope(q, pt + Dsys, theta)
                        kf = cache.layers[l].keys[0].float()
                        vf = cache.layers[l].values[0].float()
                        kf = kf.repeat_interleave(rep, dim=0)
                        vf = vf.repeat_interleave(rep, dim=0)
                        sc = qr @ kf.transpose(1, 2) / math.sqrt(D)
                        j = torch.arange(kf.shape[1], device=dev)
                        sc = sc.masked_fill(
                            j[None, None] > (pt + Dsys)[None, :, None], -1e30
                        )
                        ref = (torch.softmax(sc, -1) @ vf).transpose(0, 1)
                        got = obuf[l][0, pt].float().view(len(pick), H, D)
                        rel = float((ref - got).norm() / got.norm())
                        checks.append(dict(layer=l, rel_err=rel))
                        print(
                            f"[vfit] position check layer {l}: rel err {rel:.4f}",
                            flush=True,
                        )
                    assert all(c["rel_err"] < 0.05 for c in checks), checks
                cache.crop(Dsys)
                for l in range(L):
                    q = qbuf[l][0, pt].float()  # [n, H, D]
                    qr = rope(q.transpose(0, 1), pt + Dsys, theta)  # [H, n, D]
                    k = kdoc[l].repeat_interleave(rep, dim=0)  # [H, Dsys, D]
                    v = vdoc[l].repeat_interleave(rep, dim=0)
                    s = qr @ k.transpose(1, 2) / math.sqrt(D)  # [H, n, Dsys]
                    lse = torch.logsumexp(s, dim=-1)  # [H, n]
                    y = torch.softmax(s, dim=-1) @ v  # [H, n, D]
                    store[split]["q"][l].append(q.to(torch.bfloat16).cpu())
                    store[split]["y"][l].append(y.transpose(0, 1).cpu())
                    store_lse[split][l].append(lse.transpose(0, 1).cpu())
                store_t[split].append(torch.as_tensor(pick))
            del cache, kdoc, vdoc
            torch.cuda.empty_cache()
            print(
                f"[vfit] system prompt of {Dsys} tokens: {len(items)} "
                f"conversations ({time.time() - t_start:.0f}s)",
                flush=True,
            )

    for l in range(L):
        torch.save(
            {
                s: dict(
                    q=torch.cat(store[s]["q"][l]),
                    y=torch.cat(store[s]["y"][l]),
                    lse=torch.cat(store_lse[s][l]),
                    t=torch.cat(store_t[s]),
                )
                for s in ("fit", "val")
            },
            out / f"layer{l:02d}.pt",
        )
    meta = dict(
        model=model_name,
        parquet=os.path.abspath(parquet),
        rope_theta=theta,
        n_layers=L,
        n_heads=H,
        n_kv_heads=KV,
        head_dim=D,
        n_fit_conv=len(fit_i),
        n_val_conv=len(val_i),
        q_per_conv=q_per,
        n_fit_queries=int(sum(len(x) for x in store_t["fit"])),
        n_val_queries=int(sum(len(x) for x in store_t["val"])),
        system_prompts=len(by_sys),
        tokens=n_tok,
        seed=seed,
        position_check=checks,
        seconds=time.time() - t_start,
        device=torch.cuda.get_device_name() if dev.startswith("cuda") else dev,
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"VFIT_CAPTURE_DONE {out} ({meta['seconds']:.0f}s)", flush=True)


# ----------------------------------------------------------------- solve
def fit_layer(Kc, V0, cap, cart_len, theta, lams, dev):
    """Kc, V0: [KV, S, D] cartridge keys/values, sink first. cap: the
    captured fit/val dicts for this layer. Returns the fitted trainable
    values [KV, S-1, D] and diagnostics."""
    KV, S, D = Kc.shape
    H = cap["fit"]["q"].shape[1]
    rep = H // KV

    def design(split):
        q = cap[split]["q"].float().to(dev)  # [n, H, D]
        t = cap[split]["t"].to(dev)
        qr = rope(q.transpose(0, 1), t + cart_len, theta)  # [H, n, D]
        k = Kc.float().to(dev).repeat_interleave(rep, dim=0)  # [H, S, D]
        s = qr @ k.transpose(1, 2) / math.sqrt(D)  # [H, n, S]
        return (
            torch.softmax(s, dim=-1),
            torch.logsumexp(s, dim=-1),
            cap[split]["y"].float().to(dev).transpose(0, 1),  # [H, n, D]
            cap[split]["lse"].float().to(dev).transpose(0, 1),  # [H, n]
        )

    Xf, _, Yf, _ = design("fit")
    Xv, lse_c, Yv, lse_t = design("val")
    V0 = V0.float().to(dev)
    Vt = torch.empty(KV, S - 1, D, dtype=torch.float64, device=dev)
    err0 = err1 = den = 0.0
    chosen = []
    for h in range(KV):
        hs = slice(h * rep, (h + 1) * rep)
        sink = V0[h, :1]  # [1, D]
        Xt = Xf[hs, :, 1:].reshape(-1, S - 1).double()
        Tt = (Yf[hs] - Xf[hs, :, :1] * sink).reshape(-1, D).double()
        Xvt = Xv[hs, :, 1:].reshape(-1, S - 1).double()
        Tv = (Yv[hs] - Xv[hs, :, :1] * sink).reshape(-1, D).double()
        v0 = V0[h, 1:].double()
        scale = float((Xt * Xt).sum(0).mean())
        best = None
        for lam in lams:
            a = math.sqrt(lam * scale)
            A = torch.cat([Xt, a * torch.eye(S - 1, dtype=Xt.dtype, device=dev)])
            B = torch.cat([Tt, a * v0])
            sol = torch.linalg.lstsq(A.cpu(), B.cpu()).solution.to(dev)
            e = float((Xvt @ sol - Tv).square().sum())
            if best is None or e < best[0]:
                best = (e, lam, sol)
        Vt[h] = best[2]
        chosen.append(best[1])
        err0 += float((Xvt @ v0 - Tv).square().sum())
        err1 += best[0]
        den += float(Tv.square().sum())
    drift = float((Vt - V0[:, 1:].double()).norm() / V0[:, 1:].double().norm())
    if not torch.isfinite(Vt).all():
        raise FloatingPointError("non-finite fitted values")
    return Vt, dict(
        val_rel_err_before=math.sqrt(err0 / den),
        val_rel_err_after=math.sqrt(err1 / den),
        lam=chosen,
        value_drift=drift,
        lse_gap_mean=float((lse_t - lse_c).mean()),
        lse_gap_abs=float((lse_t - lse_c).abs().mean()),
    )


def solve():
    cap_dir = Path(os.environ["CAPTURE"])
    cart = os.environ["CART"]
    out_cart = os.environ["OUT_CART"]
    out_json = os.environ.get("OUT_JSON", str(Path(out_cart).with_suffix(".json")))
    lams = [
        float(x) for x in os.environ.get("LAMS", "1e-4,1e-3,1e-2,1e-1,1").split(",")
    ]
    dev = os.environ.get("DEVICE", "cpu")
    meta = json.loads((cap_dir / "meta.json").read_text())
    theta = float(os.environ.get("ROPE_THETA", meta["rope_theta"]))
    t0 = time.time()

    ck = torch.load(cart, map_location="cpu", weights_only=False)
    fk, fv = ck.get("frozen_keys") or [], ck.get("frozen_values") or []
    assert len(fk) and fk[0].shape[2] == 1, "expects a one-token frozen sink"
    L = len(ck["trainable_keys"])
    cart_len = 1 + ck["trainable_keys"][0].shape[2]
    layers = []
    for l in range(L):
        Kc = torch.cat([fk[l].data, ck["trainable_keys"][l].data], dim=2)[0]
        V0 = torch.cat([fv[l].data, ck["trainable_values"][l].data], dim=2)[0]
        cap = torch.load(cap_dir / f"layer{l:02d}.pt", weights_only=False)
        Vt, info = fit_layer(Kc, V0, cap, cart_len, theta, lams, dev)
        p = ck["trainable_values"][l]
        p.data.copy_(Vt[None].to(p.dtype).cpu())
        info["layer"] = l
        layers.append(info)
        print(
            f"VFIT,{l},{info['val_rel_err_before']:.4f},"
            f"{info['val_rel_err_after']:.4f},{info['value_drift']:.3f},"
            f"{info['lse_gap_mean']:.3f}",
            flush=True,
        )
    torch.save(ck, out_cart)
    summary = dict(
        cart=os.path.abspath(cart),
        out_cart=os.path.abspath(out_cart),
        capture=meta,
        lams=lams,
        cart_len=cart_len,
        layers=layers,
        seconds=time.time() - t0,
    )
    Path(out_json).write_text(json.dumps(summary, indent=1))
    print(f"VFIT_SOLVE_DONE {out_cart} ({summary['seconds']:.0f}s)", flush=True)


# -------------------------------------------------------------- selftest
def selftest():
    """CPU checks: rope matches the reference formula composition rule,
    and the solver recovers known values from noise-free targets."""
    torch.manual_seed(0)
    theta, D = 1e6, 128
    q, k = torch.randn(1, 5, D, dtype=torch.float64), torch.randn(
        1, 5, D, dtype=torch.float64
    )
    p = torch.tensor([3, 10, 100, 1000, 20000])
    # relative property: <rope(q, a), rope(k, b)> depends only on a - b
    a = (rope(q, p, theta) * rope(k, p - 2, theta)).sum(-1)
    b = (rope(q, p + 77, theta) * rope(k, p + 75, theta)).sum(-1)
    assert torch.allclose(a, b, atol=1e-9), "rope is not relative"

    KV, S, H, n = 2, 33, 4, 400
    Kc = torch.randn(KV, S, D) * 0.3
    Vtrue = torch.randn(KV, S, D)
    V0 = Vtrue.clone()
    V0[:, 1:] += torch.randn(KV, S - 1, D)  # wrong start, right sink

    def mk(m, off):
        qq = torch.randn(m, H, D)
        tt = torch.arange(m) + off
        qr = rope(qq.transpose(0, 1), tt + S, theta)
        kk = Kc.repeat_interleave(H // KV, dim=0)
        w = torch.softmax(qr @ kk.transpose(1, 2) / math.sqrt(D), -1)
        y = w @ Vtrue.repeat_interleave(H // KV, dim=0)
        lse = torch.logsumexp(qr @ kk.transpose(1, 2) / math.sqrt(D), -1)
        return dict(
            q=qq.to(torch.bfloat16), t=tt, y=y.transpose(0, 1), lse=lse.transpose(0, 1)
        )

    cap = {"fit": mk(n, 0), "val": mk(100, 1000)}
    Vt, info = fit_layer(Kc, V0, cap, S, theta, [1e-8, 1e-6], "cpu")
    print("selftest", info["val_rel_err_before"], info["val_rel_err_after"])
    assert info["val_rel_err_after"] < 0.05 * info["val_rel_err_before"]
    print("VFIT_SELFTEST PASS")


if __name__ == "__main__":
    {"capture": capture, "solve": solve, "selftest": selftest}[sys.argv[1]]()
