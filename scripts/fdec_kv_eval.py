"""KV-cache quantization eval: can symmetric FP8-K/FP8-V be enabled on Qwen?

Our K16/V8 finding kept keys at 16 bits because post-RoPE Qwen keys have outlier
channels that collapse under FP8. This harness measures that directly (fp16 cache vs
K16/V8 vs symmetric K8/V8) and tests two enablers:

  1. weight-side SmoothQuant+GPTQ -- predicted NOT to help, because our SmoothQuant is
     output-preserving (folds into RMSNorm + k_proj columns), so the keys are
     bit-identical and FP8-K quantizes the same;
  2. (phase 2, separate) a QK-invariant per-channel KEY smoothing -- the actual cache
     analog of SmoothQuant: scale outlier key channels down and query channels up so
     q.k is preserved but the keys become FP8-friendly.

FP8 = e4m3, per-token scale. Eval = teacher-forced top-1 / KL / perplexity vs the
fp16-cache fp16-weight reference on wikitext-2 test.
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

# global read by the patched attention (HF calls eager_attention_forward as a
# module-level function, so config travels via this dict)
_KV = {"scheme": "fp16", "kchan": None}


def _fp8_pertoken(x):
    # per-token (per [B,H,T] vector over the head_dim) symmetric e4m3
    scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-6) / 448.0
    return (x / scale).to(torch.float8_e4m3fn).to(x.dtype) * scale


def _fp8_perchannel(x):
    # per-channel (scale over the token dim) -- the KIVI-style key ceiling. Needs all
    # tokens, so it is an oracle/upper-bound for a streaming decode cache, not directly
    # deployable, but it isolates whether 8-bit keys are achievable at all.
    scale = x.abs().amax(-2, keepdim=True).clamp(min=1e-6) / 448.0
    return (x / scale).to(torch.float8_e4m3fn).to(x.dtype) * scale


_HAD = {}


def _get_hadamard(n, device, dtype):
    """Normalized Sylvester-Hadamard [n,n] (n a power of 2), orthonormal so q@H,k@H
    preserve q.k exactly while spreading any axis-aligned outlier across all channels
    -- the parameter-free, calibration-free cache analog of SmoothQuant (QuaRot-style).
    Applied POST-RoPE to both q and k, so the dot product (hence the model) is
    unchanged; only the to-be-quantized key is rotated into a range-uniform basis."""
    key = (n, device, str(dtype))
    if key not in _HAD:
        H = torch.ones(1, 1)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _HAD[key] = (H[:n, :n] / (n**0.5)).to(device=device, dtype=dtype)
    return _HAD[key]


def kv_quant_attention(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kw
):
    g = module.num_key_value_groups
    s = _KV["scheme"]
    q, k, v = query, key, value
    if s == "k8v8_had":
        # rotate head_dim of q and k (orthonormal -> q.k preserved); only k is quantized
        Hn = _get_hadamard(k.shape[-1], k.device, k.dtype)
        q = torch.matmul(q, Hn)
        k = torch.matmul(k, Hn)
    quant_k = s in ("k8v8", "k8v16", "k8v8_had", "k8v8_pck", "k8v8_had_pck")
    quant_v = s in ("k16v8", "k8v8", "k8v8_had", "k8v8_pck", "k8v8_had_pck")
    if s == "k8v8_had_pck":
        Hn = _get_hadamard(k.shape[-1], k.device, k.dtype)
        q = torch.matmul(q, Hn)
        k = torch.matmul(k, Hn)
    if quant_k:
        k = (
            _fp8_perchannel(k)
            if s in ("k8v8_pck", "k8v8_had_pck")
            else _fp8_pertoken(k)
        )
    ks = k.repeat_interleave(g, dim=1)
    vs = v.repeat_interleave(g, dim=1)
    if quant_v:
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


# cell -> (kv scheme, weight transform)
CELLS = {
    "fp16": ("fp16", None),
    "k16v8": ("k16v8", None),
    "k8v16": ("k8v16", None),  # isolate: keys-only FP8 (Codex ablation)
    "k8v8": ("k8v8", None),
    "k8v8_w_sqgptq": ("k8v8", "sqgptq"),
    "k8v8_had": ("k8v8_had", None),  # Hadamard-rotated keys (the cache analog)
    "k8v8_pck": ("k8v8_pck", None),  # per-channel key FP8 (KIVI-style ceiling)
    "k8v8_had_pck": ("k8v8_had_pck", None),  # Hadamard + per-channel keys
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cells", default="fp16,k16v8,k8v8,k8v8_w_sqgptq")
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
    need_w = any(CELLS[c][1] for c in cells)
    gptq_calib = (
        load_gptq_calib(tok, args.gptq_calib_tokens, args.gptq_calib_source)
        if need_w
        else None
    )

    # fp16 reference: fp16 cache + fp16 weights
    model = load_model(args.model, device, args.dtype)
    install_kv(model)
    _KV["scheme"] = "fp16"
    ref_lp = F.log_softmax(logprobs(model, inp, device), dim=-1).to("cpu").half()
    del model
    torch.cuda.empty_cache()

    rows = []
    for cell in cells:
        scheme, wq = CELLS[cell]
        model = load_model(args.model, device, args.dtype)
        if wq == "sqgptq":
            smoothquant_transform(model, tok, gptq_calib, args.sq_alpha, device)
            gptq_quantize_model(model, tok, gptq_calib, 8, 0.0, 128, device=device)
        install_kv(model)
        _KV["scheme"] = scheme
        lp = logprobs(model, inp, device).to("cpu")
        m = eval_against_ref(lp, ref_lp, gold)
        m["cell"] = cell
        rows.append(m)
        print(
            f"  {cell:<16} kv={scheme:<6} w={wq or 'fp16':<6} "
            f"top1={m['top1']:.3f} top5={m['top5']:.3f} KL={m['meanKL']:.4f} "
            f"pplq={m['ppl_q']:.3f} pplratio={m['ppl_ratio']:.3f}"
        )
        del model
        torch.cuda.empty_cache()

    print(f"\n[ref] fp16-cache fp16-weight ppl={rows[0]['ppl_ref']:.3f}")
    import json

    print("JSON " + json.dumps({"model": args.model, "rows": rows}))


if __name__ == "__main__":
    main()
