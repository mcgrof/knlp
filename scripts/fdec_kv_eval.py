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
    if c["kfmt"] is not None and not _KV.get("_collect"):
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


def _cfg(kfmt=None, kgran="tok", center="", sink=0, vq=False, had=False, wq=None):
    return dict(kfmt=kfmt, kgran=kgran, center=center, sink=sink, vq=vq, had=had, wq=wq)


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
        for k in ("kfmt", "kgran", "center", "sink", "vq", "had", "wq"):
            _KV[k] = cfg[k]

    # fp16 reference
    model = load_model(args.model, device, args.dtype)
    install_kv(model)
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
        set_cfg(cfg)
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
