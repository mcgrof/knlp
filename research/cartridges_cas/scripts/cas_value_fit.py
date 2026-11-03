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

``solve`` (CPU or GPU) loads a cartridge and the captured queries. With
METHOD=values (default) it fits the values in closed form as below.
With METHOD=kv it is the distinct local attention fit: Adam on the
trainable keys and values together, per layer, minimizing the relative
error of the attention output over the cartridge plus KV_WLSE times
the squared gap between the cartridge's and the teacher's
log-normalizers, so the cartridge also draws the attention mass of the
context it replaces (no biases are added, so the artifact keeps the
bias-free layout). The best step on the validation queries is kept. For
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
    FIT_MANIFEST  optional cas_freeze_split.py manifest: stop unless the
                fit and validation rows chosen here are exactly its
                fit_rows and fit_check_rows (so a repaired or rewritten
                parquet is captured on the same conversations)
    DEVICE      cuda (default)

Env (solve):
    CAPTURE     capture directory
    CART        starting cartridge (its keys and sink are kept)
    OUT_CART    fitted cartridge
    OUT_JSON    summary (default OUT_CART with .json)
    LAMS        comma list of ridge scales (default 1e-4,1e-3,1e-2,1e-1,1)
    ROPE_THETA  rotary base (default from the capture's model config)
    METHOD      values (closed-form values) or kv (local key-value fit)
    KV_STEPS    Adam steps per layer for METHOD=kv (default 1000)
    KV_LR       Adam rate for METHOD=kv (default 1e-2)
    KV_WLSE     weight of the log-normalizer term (default 0.1)
    DEVICE      cpu (default) or cuda

``filter`` (CPU) drops whole conversations from an existing capture.
The capture stores each conversation's scored positions in a fixed
order with no row labels, so the filter first replays the capture's row
choice and every random position draw from the same parquet and seed,
and stops unless the replayed positions equal the stored positions
exactly for both sets. Only then does it remove the blocks of the
dropped rows and write a capture the solver reads as usual, with the
surviving rows listed in rows.json. The remaining positions are the
ones the original capture drew; nothing is redrawn or refilled.

Env (filter):
    CAPTURE     source capture directory
    OUT_CAPTURE filtered capture directory
    PARQUET     the parquet the capture read (hash must match meta's path)
    DROP_ROWS   comma list of row indices to remove
    VAL_FRAC    as at capture (default 0.2)

Usage:
    python cas_value_fit.py filter
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
    inv = 1.0 / (
        theta ** (torch.arange(0, d, 2, dtype=torch.float64, device=x.device) / d)
    )
    pos = pos.to(x.device)
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
    if os.environ.get("FIT_MANIFEST"):
        man = json.loads(Path(os.environ["FIT_MANIFEST"]).read_text())
        assert [int(i) for i in fit_i] == man["fit_rows"], "fit rows differ"
        assert [int(i) for i in val_i] == man["fit_check_rows"], "check rows differ"
        print("[vfit] fit and check rows match FIT_MANIFEST", flush=True)
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


# ---------------------------------------------------------------- filter
def replay_capture(parquet, n_fit, n_val, q_per, seed, val_frac, tok):
    """The capture's conversation order and position draws, without the
    model: {"fit"|"val": [(row, positions), ...]} in storage order."""
    from cartridges.datasets import qwen_messages_to_element, read_conversations

    rows = read_conversations(parquet)
    fit_i, val_i = group_split(rows, val_frac, seed)
    rng = np.random.default_rng(seed)
    fit_i = sorted(rng.choice(fit_i, min(n_fit, len(fit_i)), replace=False))
    val_i = sorted(rng.choice(val_i, min(n_val, len(val_i)), replace=False))
    sel = [(i, "fit") for i in fit_i] + [(i, "val") for i in val_i]
    by_sys = {}
    for i, split in sel:
        by_sys.setdefault(rows[i].system_prompt, []).append((i, split))
    order = {"fit": [], "val": []}
    for _sysp, items in by_sys.items():
        for i, split in items:
            el = qwen_messages_to_element(rows[i].messages, tokenizer=tok)
            T = len(el.input_ids.tolist())
            cand = np.unique(el.topk_token_idxs.numpy() - 1)
            cand = cand[(cand >= 0) & (cand < T)]
            pick = np.sort(rng.choice(cand, min(q_per, len(cand)), replace=False))
            order[split].append((int(i), pick))
    return order


def filter_capture():
    from transformers import AutoTokenizer

    src = Path(os.environ["CAPTURE"])
    dst = Path(os.environ["OUT_CAPTURE"])
    parquet = os.environ["PARQUET"]
    drop = {int(x) for x in os.environ["DROP_ROWS"].split(",") if x}
    val_frac = float(os.environ.get("VAL_FRAC", "0.2"))
    meta = json.loads((src / "meta.json").read_text())
    assert os.path.abspath(parquet) == meta["parquet"], "not the captured parquet"
    tok = AutoTokenizer.from_pretrained(meta["model"])
    order = replay_capture(
        parquet,
        meta["n_fit_conv"],
        meta["n_val_conv"],
        meta["q_per_conv"],
        meta["seed"],
        val_frac,
        tok,
    )
    first = torch.load(src / "layer00.pt", weights_only=False)
    for s in ("fit", "val"):
        replayed = np.concatenate([p for _, p in order[s]])
        assert np.array_equal(replayed, first[s]["t"].numpy()), f"{s} replay differs"
    found = {i for s in order for i, _ in order[s]} & drop
    assert found == drop, f"rows not in the capture: {sorted(drop - found)}"
    keep = {}
    for s in ("fit", "val"):
        m = np.concatenate([np.full(len(p), i not in drop) for i, p in order[s]])
        keep[s] = torch.as_tensor(m)
    dst.mkdir(parents=True, exist_ok=True)
    for l in range(meta["n_layers"]):
        d = torch.load(src / f"layer{l:02d}.pt", weights_only=False)
        torch.save(
            {s: {k: v[keep[s]] for k, v in d[s].items()} for s in ("fit", "val")},
            dst / f"layer{l:02d}.pt",
        )
    kept = {s: [(i, len(p)) for i, p in order[s] if i not in drop] for s in order}
    meta.update(
        n_fit_conv=len(kept["fit"]),
        n_val_conv=len(kept["val"]),
        n_fit_queries=int(keep["fit"].sum()),
        n_val_queries=int(keep["val"].sum()),
        filtered_from=str(src.resolve()),
        dropped_rows=sorted(drop),
        replay_verified=True,
    )
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    (dst / "rows.json").write_text(json.dumps(kept))
    print(
        f"VFIT_FILTER_DONE {dst} fit {meta['n_fit_conv']} val {meta['n_val_conv']} "
        f"dropped {sorted(drop)}",
        flush=True,
    )


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


def fit_kv_layer(Kc, V0, cap, cart_len, theta, steps, lr, w_lse, dev):
    """Local attention fit of keys and values for one layer. Kc, V0:
    [KV, S, D] with the sink first (kept fixed). Adam on the trainable
    keys and values minimizes, over the fit queries, the relative
    squared error of the attention output over the cartridge plus
    w_lse times the squared log-normalizer gap to the teacher, so the
    cartridge draws the same attention mass as the context it stands
    for. Returns the best keys and values on the validation queries."""
    KV, S, D = Kc.shape
    H = cap["fit"]["q"].shape[1]
    rep = H // KV

    def prep(split):
        q = cap[split]["q"].float().to(dev)
        t = cap[split]["t"].to(dev)
        return (
            rope(q.transpose(0, 1), t + cart_len, theta) / math.sqrt(D),
            cap[split]["y"].float().to(dev).transpose(0, 1),
            cap[split]["lse"].float().to(dev).transpose(0, 1),
        )

    qf, yf, lf = prep("fit")
    qv, yv, lv = prep("val")
    yf_den, yv_den = yf.square().mean(), yv.square().mean()
    ks = Kc[:, :1].float().to(dev)
    vs = V0[:, :1].float().to(dev)
    k0 = Kc[:, 1:].float().to(dev)
    v0 = V0[:, 1:].float().to(dev)
    kt = torch.nn.Parameter(k0.clone())
    vt = torch.nn.Parameter(v0.clone())

    def terms(q, y, l, den):
        k = torch.cat([ks, kt], 1).repeat_interleave(rep, dim=0)
        v = torch.cat([vs, vt], 1).repeat_interleave(rep, dim=0)
        sc = q @ k.transpose(1, 2)
        lse = torch.logsumexp(sc, dim=-1)
        out = torch.softmax(sc, dim=-1) @ v
        return (out - y).square().mean() / den, (lse - l).square().mean()

    with torch.no_grad():
        e0, g0 = terms(qv, yv, lv, yv_den)
    opt = torch.optim.Adam([kt, vt], lr=lr)
    best = (float(e0 + w_lse * g0), 0, k0.clone(), v0.clone(), e0, g0)
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        e, g = terms(qf, yf, lf, yf_den)
        (e + w_lse * g).backward()
        opt.step()
        if step % 25 == 0 or step == steps:
            with torch.no_grad():
                ev, gv = terms(qv, yv, lv, yv_den)
                tot = float(ev + w_lse * gv)
                if not math.isfinite(tot):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                if tot < best[0]:
                    best = (tot, step, kt.detach().clone(), vt.detach().clone(), ev, gv)
    _, bstep, kb, vb, eb, gb = best
    return (
        kb,
        vb,
        dict(
            val_rel_err_before=math.sqrt(float(e0)),
            val_rel_err_after=math.sqrt(float(eb)),
            val_lse_rmse_before=math.sqrt(float(g0)),
            val_lse_rmse_after=math.sqrt(float(gb)),
            best_step=bstep,
            key_drift=float((kb - k0).norm() / k0.norm()),
            value_drift=float((vb - v0).norm() / v0.norm()),
        ),
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
    global METHOD, KV_STEPS, KV_LR, KV_WLSE
    METHOD = os.environ.get("METHOD", "values")
    assert METHOD in ("values", "kv"), METHOD
    KV_STEPS = int(os.environ.get("KV_STEPS", "1000"))
    KV_LR = float(os.environ.get("KV_LR", "1e-2"))
    KV_WLSE = float(os.environ.get("KV_WLSE", "0.1"))
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
        if METHOD == "values":
            Vt, info = fit_layer(Kc, V0, cap, cart_len, theta, lams, dev)
            p = ck["trainable_values"][l]
            p.data.copy_(Vt[None].to(p.dtype).cpu())
            tail = f"{info['value_drift']:.3f},{info['lse_gap_mean']:.3f}"
        else:
            Kt, Vt, info = fit_kv_layer(
                Kc, V0, cap, cart_len, theta, KV_STEPS, KV_LR, KV_WLSE, dev
            )
            for name, val in (("trainable_keys", Kt), ("trainable_values", Vt)):
                p = ck[name][l]
                p.data.copy_(val[None].to(p.dtype).cpu())
            tail = (
                f"{info['key_drift']:.3f},{info['value_drift']:.3f},"
                f"{info['val_lse_rmse_before']:.3f},{info['val_lse_rmse_after']:.3f}"
            )
        info["layer"] = l
        layers.append(info)
        print(
            f"VFIT,{METHOD},{l},{info['val_rel_err_before']:.4f},"
            f"{info['val_rel_err_after']:.4f},{tail}",
            flush=True,
        )
    torch.save(ck, out_cart)
    summary = dict(
        cart=os.path.abspath(cart),
        out_cart=os.path.abspath(out_cart),
        capture=meta,
        method=METHOD,
        lams=lams if METHOD == "values" else None,
        kv=dict(steps=KV_STEPS, lr=KV_LR, w_lse=KV_WLSE) if METHOD == "kv" else None,
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
    # the key-value fit from wrong keys and values lowers both the output
    # error and the normalizer gap
    Kw = Kc.clone()
    Kw[:, 1:] += 0.3 * torch.randn(KV, S - 1, D)
    _, _, kinfo = fit_kv_layer(Kw, V0, cap, S, theta, 300, 2e-2, 0.1, "cpu")
    print("selftest kv", kinfo)
    assert kinfo["val_rel_err_after"] < 0.5 * kinfo["val_rel_err_before"]
    assert kinfo["val_lse_rmse_after"] < kinfo["val_lse_rmse_before"]
    print("VFIT_SELFTEST PASS")


if __name__ == "__main__":
    {
        "capture": capture,
        "filter": filter_capture,
        "solve": solve,
        "selftest": selftest,
    }[sys.argv[1]]()
