"""KV-cache quantization eval: can sub-16-bit symmetric KV keys be enabled on Qwen?

Our K16/V8 finding kept keys at 16 bits because post-RoPE Qwen keys collapse under
FP8. First-round result: symmetric E4M3 per-token keys die (ppl x123); a QK-invariant
Hadamard rotation helps (x10.6) but does not rescue, and the per-channel oracle only
reaches x6.2. That ruled out E4M3+per-token+Hadamard+per-channel, NOT "keys need >8
bits". This round adds the levers that attack the OTHER axes of the failure:

  - softmax-null centering: subtract a fixed per-(layer,kv_head) vector mu from all
    keys. q.(k-mu) = q.k - q.mu, and q.mu is constant across the attention row, so
    softmax (hence the fp16 output) is EXACTLY unchanged, but the centered keys lose
    the large fixed mean/bias component that wastes FP8 range. (cf. SageAttention2
    Q/K mean subtraction.) Qwen2.5 has QKV bias, a prime suspect.
  - INT8 (symmetric / asymmetric-zeropoint) keys: separates "FP8-format problem"
    (3 mantissa bits) from "8-bit problem". INT8 has ~7 uniform bits.
  - per-channel granularity (KIVI-style oracle) and sink-token carve-out (keep the
    first few keys in fp16) to test the attention-sink axis.

FP8 = e4m3 (max 448), INT8 = max 127. Eval = teacher-forced prefill, 4096 tokens of
WikiText-2 test; metrics top-1 / KL(fp16||q) / perplexity ratio vs the fp16-cache
fp16-weight reference. Centering mu here is the ORACLE (mean over the eval sequence);
a deployable version would calibrate mu offline.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fdec_matrix import (  # noqa: E402
    gptq_quantize_model,
    smoothquant_transform,
)
from fdec_quant_eval import (  # noqa: E402
    eval_against_ref,
    load_eval_ids,
    load_gptq_calib,
    load_model,
    logprobs,
)

# config read by the patched attention (HF calls eager_attention_forward as a
# module-level function, so config travels via this dict)
_KV = {
    "kfmt": None,
    "kgran": "tok",
    "center": False,
    "sink": 0,
    "vq": False,
    "had": False,
}

_HAD = {}
_CMU = {}  # calibrated per-layer key mean: layer_idx -> [1, Hkv, 1, D]
_KSCALE = {}  # static calibrated pre-bias residual scale: layer_idx -> scale tensor
_KPROJ_LAYER = {}  # id(k_proj module) -> layer_idx (rebuilt per model load)


def _get_hadamard(n, device, dtype):
    """Normalized Sylvester-Hadamard [n,n] (orthonormal); q@H,k@H preserve q.k exactly
    while spreading axis-aligned outliers (QuaRot-style)."""
    key = (n, device, str(dtype))
    if key not in _HAD:
        H = torch.ones(1, 1)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _HAD[key] = (H[:n, :n] / (n**0.5)).to(device=device, dtype=dtype)
    return _HAD[key]


def _fp8_pertoken(x):
    scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-6) / 448.0
    return (x / scale).to(torch.float8_e4m3fn).to(x.dtype) * scale


def _quant_key(k, fmt, gran):
    """fmt: fp8 (e4m3) | int8 (symmetric) | int8a (asymmetric zero-point).
    gran: tok (per key vector over channels) | chan (per channel over tokens; oracle).
    """
    dim = -1 if gran == "tok" else -2
    if fmt == "fp8":
        scale = k.abs().amax(dim, keepdim=True).clamp(min=1e-6) / 448.0
        return (k / scale).to(torch.float8_e4m3fn).to(k.dtype) * scale
    if fmt == "int8":
        scale = k.abs().amax(dim, keepdim=True).clamp(min=1e-6) / 127.0
        return torch.round(k / scale).clamp(-127, 127) * scale
    if fmt == "int8a":
        mx = k.amax(dim, keepdim=True)
        mn = k.amin(dim, keepdim=True)
        scale = (mx - mn).clamp(min=1e-6) / 255.0
        zp = torch.round(-mn / scale)
        q = (torch.round(k / scale) + zp).clamp(0, 255)
        return (q - zp) * scale
    raise ValueError(fmt)


def _scale_bt(x, fmt, gran):
    """Scale layouts (the deployability axis -- production FP8 KV uses per-tensor /
    per-head scales, not per-token):
      tok    = per (token) over channels          (dim -1)  -- dynamic, custom-kernel
      head   = per (kv_head) over tokens+channels (dims 1,3) -- vLLM-style per-head
      tensor = one scale for the whole key tensor (dims 1,2,3) -- worst static case"""
    dims = {"tok": (-1,), "chan": (1,), "head": (1, 3), "tensor": (1, 2, 3)}[gran]
    a = x.abs().amax(dim=dims, keepdim=True).clamp(min=1e-6)
    return a / (448.0 if fmt == "fp8" else 127.0)


def _apply_bt(x, scale, fmt):
    if fmt == "fp8":
        # clamp to e4m3 max before the cast: with a FROZEN static scale the eval max can
        # exceed the calib max, and e4m3fn maps out-of-range to NaN (no inf). Saturating
        # is what a real static-FP8 path does -- a small clip, not a NaN.
        return (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(
            x.dtype
        ) * scale
    return torch.round(x / scale).clamp(-127, 127) * scale


def _quant_bt(x, fmt, gran):
    return _apply_bt(x, _scale_bt(x, fmt, gran), fmt)


def kproj_hook(module, inp, out):
    """Pre-RoPE key quantization at the k_proj output. With prebias=True, split off the
    bias b_K (kept fp16), quantize only the residual r = XW_K, then add b_K back so RoPE
    runs on (r_q + b_K) -- the ChatGPT-Pro/QKV-bias test: does separating the
    deterministic key-projection bias make low-bit keys easier?"""
    pr = _KV.get("prerope")
    if not pr or _KV.get("_collect"):
        return out
    B, T, KV = out.shape
    D = _KV["head_dim"]
    Hkv = KV // D
    x = out.view(B, T, Hkv, D)
    b = None
    if pr.get("prebias") and getattr(module, "bias", None) is not None:
        b = module.bias.view(1, 1, Hkv, D)
        r = x - b
    else:
        r = x
    li = _KPROJ_LAYER.get(id(module))
    if _KV.get("_kcollect"):
        # calibrate a STATIC scale on the pre-bias residual (max over calib chunks)
        s = _scale_bt(r, pr["fmt"], pr["gran"]).amax(0, keepdim=True)
        _KSCALE[li] = s if li not in _KSCALE else torch.maximum(_KSCALE[li], s)
        return out
    if pr.get("static"):
        rq = _apply_bt(r, _KSCALE[li], pr["fmt"])  # frozen calibrated scale
    else:
        rq = _quant_bt(r, pr["fmt"], pr["gran"])
    x = rq + b if b is not None else rq
    return x.reshape(B, T, KV)


def install_kproj(model):
    import re

    _KPROJ_LAYER.clear()
    hs = []
    for n, m in model.named_modules():
        if n.endswith("k_proj") and hasattr(m, "weight"):
            mt = re.search(r"\.layers\.(\d+)\.", n)
            _KPROJ_LAYER[id(m)] = int(mt.group(1)) if mt else len(_KPROJ_LAYER)
            hs.append(m.register_forward_hook(kproj_hook))
    return hs


def kv_quant_attention(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kw
):
    g = module.num_key_value_groups
    c = _KV
    q, k, v = query, key, value
    if c["had"]:
        Hn = _get_hadamard(k.shape[-1], k.device, k.dtype)
        q = torch.matmul(q, Hn)
        k = torch.matmul(k, Hn)
    if _KV.get("_collect"):
        # record the per-(kv_head,channel) post-RoPE key mean for calibrated centering
        _CMU[module.layer_idx] = k.mean(dim=(0, 2), keepdim=True).detach()
    if c["center"] == "oracle":
        # softmax-null: q.(k-mu) shifts the whole attention row by the constant q.mu,
        # invisible to softmax. ORACLE mu = mean over the EVAL sequence (leaks future
        # tokens into the quant error; diagnostic only).
        k = k - k.mean(dim=-2, keepdim=True)
    elif c["center"] == "calib":
        # DEPLOYABLE: fixed mu calibrated on a disjoint sequence; no eval leakage.
        k = k - _CMU[module.layer_idx]
    if c["kfmt"] is not None and not c.get("prerope") and not _KV.get("_collect"):
        kq = _quant_key(k, c["kfmt"], c["kgran"])
        if c["sink"] > 0:
            kq[..., : c["sink"], :] = k[..., : c["sink"], :]  # keep first keys fp16
        k = kq
    ks = k.repeat_interleave(g, dim=1)
    vs = v.repeat_interleave(g, dim=1)
    if c["vq"]:
        vs = _fp8_pertoken(vs)
    aw = torch.matmul(q, ks.transpose(2, 3)) * scaling
    if attention_mask is not None:
        aw = aw + attention_mask[:, :, :, : ks.shape[-2]]
    aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(aw, vs).transpose(1, 2).contiguous()
    return out, aw


def install_kv(model):
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
            mod.eager_attention_forward = kv_quant_attention
    return saved


def _cfg(
    kfmt=None,
    kgran="tok",
    center="",
    sink=0,
    vq=False,
    had=False,
    wq=None,
    prerope=None,
):
    return dict(
        kfmt=kfmt,
        kgran=kgran,
        center=center,
        sink=sink,
        vq=vq,
        had=had,
        wq=wq,
        prerope=prerope,
    )


def _pr(fmt, gran="tok", prebias=False, static=False):
    return {"fmt": fmt, "gran": gran, "prebias": prebias, "static": static}


# cell -> config. center: "" | "oracle" (eval-seq mean, leaky) | "calib" (deployable)
CELLS = {
    "fp16": _cfg(),
    "k16v8": _cfg(vq=True),  # keys fp16, values fp8 (the deployable bar)
    "k8v8": _cfg(kfmt="fp8", vq=True),  # symmetric e4m3 per-token (collapses)
    "k8v8_pck": _cfg(kfmt="fp8", kgran="chan", vq=True),  # per-channel oracle
    "ki8v8": _cfg(kfmt="int8", vq=True),  # INT8 sym per-token (the real enabler)
    "ki8v8a": _cfg(kfmt="int8a", vq=True),  # INT8 asymmetric (zero-point)
    "ki8v8_pck": _cfg(kfmt="int8", kgran="chan", vq=True),
    # oracle centering (leaky -- diagnostic)
    "k8v8_center": _cfg(kfmt="fp8", center="oracle", vq=True),
    "ki8v8_center": _cfg(kfmt="int8", center="oracle", vq=True),
    # DEPLOYABLE calibrated centering (no eval leakage)
    "k8v8_ccenter": _cfg(kfmt="fp8", center="calib", vq=True),
    "ki8v8_ccenter": _cfg(kfmt="int8", center="calib", vq=True),
    "ki8v8_ccenter_pck": _cfg(kfmt="int8", kgran="chan", center="calib", vq=True),
    "k8v8_sink4": _cfg(kfmt="fp8", sink=4, vq=True),  # keep first 4 keys fp16
    # pre-RoPE / pre-bias residual (the QKV-bias mechanism test)
    "k8v8_prerope": _cfg(prerope=_pr("fp8"), vq=True),  # FP8, post-bias pre-RoPE
    "k8v8_prebias": _cfg(prerope=_pr("fp8", prebias=True), vq=True),  # FP8, pre-bias
    "ki8v8_prerope": _cfg(prerope=_pr("int8"), vq=True),  # INT8 control
    "ki8v8_prebias": _cfg(prerope=_pr("int8", prebias=True), vq=True),
    "ki8v8_prebias_pck": _cfg(prerope=_pr("int8", "chan", prebias=True), vq=True),
    # scale-layout deployability matrix (pre-bias residual; per-head/per-tensor = vLLM)
    "k8v8_prebias_head": _cfg(prerope=_pr("fp8", "head", prebias=True), vq=True),
    "k8v8_prebias_tensor": _cfg(prerope=_pr("fp8", "tensor", prebias=True), vq=True),
    "ki8v8_prebias_head": _cfg(prerope=_pr("int8", "head", prebias=True), vq=True),
    "ki8v8_prebias_tensor": _cfg(prerope=_pr("int8", "tensor", prebias=True), vq=True),
    # STATIC calibrated scales (frozen from disjoint C4) -- the deployable layout
    "k8v8_prebias_head_st": _cfg(
        prerope=_pr("fp8", "head", prebias=True, static=True), vq=True
    ),
    "k8v8_prebias_tensor_st": _cfg(
        prerope=_pr("fp8", "tensor", prebias=True, static=True), vq=True
    ),
    "ki8v8_prebias_head_st": _cfg(
        prerope=_pr("int8", "head", prebias=True, static=True), vq=True
    ),
    "ki8v8_prebias_tensor_st": _cfg(
        prerope=_pr("int8", "tensor", prebias=True, static=True), vq=True
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cells", default="fp16,k16v8,k8v8,k8v8_center,ki8v8")
    ap.add_argument("--eval-tokens", type=int, default=4096)
    ap.add_argument("--gptq-calib-tokens", type=int, default=16384)
    ap.add_argument("--gptq-calib-source", default="c4", choices=["wikitext", "c4"])
    ap.add_argument("--sq-alpha", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = load_eval_ids(tok, args.eval_tokens)
    gold = ids[0, 1:].contiguous()
    inp = ids[:, :-1]
    print(f"[kv-eval] {args.model}  positions={inp.shape[1]}")

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    need_w = any(CELLS[c]["wq"] for c in cells)
    need_calib_mu = any(CELLS[c]["center"] == "calib" for c in cells)
    gptq_calib = (
        load_gptq_calib(tok, args.gptq_calib_tokens, args.gptq_calib_source)
        if need_w
        else None
    )

    def set_cfg(cfg):
        for k in ("kfmt", "kgran", "center", "sink", "vq", "had", "wq", "prerope"):
            _KV[k] = cfg[k]

    from transformers import AutoConfig

    hcfg = AutoConfig.from_pretrained(args.model)
    _KV["head_dim"] = getattr(hcfg, "head_dim", None) or (
        hcfg.hidden_size // hcfg.num_attention_heads
    )

    need_kstatic = any((CELLS[c]["prerope"] or {}).get("static") for c in cells)
    kcalib_ids = (
        tok(load_gptq_calib(tok, args.eval_tokens, "c4"), return_tensors="pt")
        .input_ids[:, : args.eval_tokens]
        .to(device)
        if need_kstatic
        else None
    )

    # fp16 reference
    model = load_model(args.model, device, args.dtype)
    install_kv(model)
    install_kproj(model)
    set_cfg(CELLS["fp16"])
    ref_lp = F.log_softmax(logprobs(model, inp, device), dim=-1).to("cpu").half()
    # calibrate mu on a DISJOINT C4 sequence (no eval leakage) while this model is up
    if need_calib_mu:
        calib_ids = (
            tok(load_gptq_calib(tok, args.eval_tokens, "c4"), return_tensors="pt")
            .input_ids[:, : args.eval_tokens]
            .to(device)
        )
        _CMU.clear()
        _KV["_collect"] = True
        with torch.no_grad():
            model(calib_ids)
        _KV["_collect"] = False
        print(f"[calib] mu collected for {len(_CMU)} layers")
    del model
    torch.cuda.empty_cache()

    rows = []
    for cell in cells:
        cfg = CELLS[cell]
        model = load_model(args.model, device, args.dtype)
        if cfg["wq"] == "sqgptq":
            smoothquant_transform(model, tok, gptq_calib, args.sq_alpha, device)
            gptq_quantize_model(model, tok, gptq_calib, 8, 0.0, 128, device=device)
        install_kv(model)
        install_kproj(model)
        set_cfg(cfg)
        if (cfg["prerope"] or {}).get("static"):
            # freeze the static scale on the disjoint C4 calib for THIS cell's config
            _KSCALE.clear()
            _KV["_kcollect"] = True
            with torch.no_grad():
                model(kcalib_ids)
            _KV["_kcollect"] = False
        lp = logprobs(model, inp, device).to("cpu")
        m = eval_against_ref(lp, ref_lp, gold)
        m["cell"] = cell
        rows.append(m)
        print(
            f"  {cell:<18} top1={m['top1']:.3f} top5={m['top5']:.3f} "
            f"KL={m['meanKL']:.4f} pplq={m['ppl_q']:.3f} pplratio={m['ppl_ratio']:.3f}"
        )
        del model
        torch.cuda.empty_cache()

    print(f"\n[ref] fp16-cache fp16-weight ppl={rows[0]['ppl_ref']:.3f}")
    import json

    print("JSON " + json.dumps({"model": args.model, "rows": rows}))


if __name__ == "__main__":
    main()
