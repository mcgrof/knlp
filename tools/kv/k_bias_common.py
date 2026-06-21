"""Shared core for the K-projection-bias / FP8-sensitivity audit (Brick: Qwen K-bias hypothesis).

Hypothesis under test: K-cache FP8 quantization failure appears when the K-projection BIAS is
large relative to the dynamic pre-bias K signal and dominates FP8 scale selection -- i.e. bias
MAGNITUDE / outlier dominance, not merely the architectural `attention_bias=True` flag, is the
predictor. This module is the shared contract for the five audit scripts; it discovers structure
at runtime (NEVER assumes Qwen layout) and exposes: model loading, attention discovery (separate
vs fused QKV), K-bias extraction, FP8 fake-quant (the cache-quant operation), and a hook harness
that captures pre-bias / post-bias-pre-RoPE / post-RoPE K activations.

Design choices that matter:
- We do NOT modify the model's bias. We read it, optionally scale a COPY for the alpha causal
  sweep, and for "pre-bias" quantization we subtract the (fp16) bias, quantize the residual, add
  it back -- matching the production-validated KV-PREBIAS recipe.
- abs>448 (raw e4m3 range) is a DIAGNOSTIC only. With scaled FP8, large values are representable
  by changing scale; the real failure is the bias dominating the per-tensor scale and crushing the
  bulk (token-discriminative) residual into a few quant levels. We measure both.
- post-RoPE K is captured via the attention-interface hook (the q,k handed to the attention fn
  are already RoPE'd), the same mechanism used across the fdec harnesses; pre-RoPE post-bias K via
  a forward hook on k_proj; pre-bias K by subtracting the K-bias slice.
"""

import json
import os

import torch
import torch.nn.functional as F

FP8_MAX = 448.0  # e4m3fn max representable magnitude


# ----------------------------------------------------------------------------- loading
def load_model(
    model_id,
    dtype="bfloat16",
    device="cuda:0",
    trust_remote_code=False,
    device_map=None,
    max_memory=None,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    kw = dict(
        dtype=dt,
        trust_remote_code=trust_remote_code,
        # sdpa routes through ALL_ATTENTION_FUNCTIONS so the interface hook sees post-RoPE q,k,v
        # ("eager" bypasses that registry in recent transformers). We don't need attn weights.
        attn_implementation="sdpa",
    )
    if device_map:  # large models that do not fit one GPU (e.g. 72B): shard GPU+CPU
        kw["device_map"] = device_map
        # force CPU (not disk/meta) overflow so weights/biases stay real tensors
        kw["max_memory"] = max_memory or {0: "135GiB", "cpu": "1500GiB"}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if not device_map:
        model = model.to(device)
    model.eval()
    return model, tok


def get_decoder_layers(model):
    # robust across architectures: walk to the ModuleList of decoder blocks
    for path in (
        "model.layers",
        "model.model.layers",
        "transformer.h",
        "model.decoder.layers",
    ):
        obj = model
        ok = True
        for p in path.split("."):
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and hasattr(obj, "__len__"):
            return list(obj)
    raise RuntimeError("could not locate decoder layers")


def find_attn_module(layer):
    for name in ("self_attn", "attn", "attention", "self_attention"):
        if hasattr(layer, name):
            return getattr(layer, name), name
    raise RuntimeError("no attention submodule found in layer")


# ------------------------------------------------------------------- attention discovery
def discover_attention(model):
    """Return per-layer dict describing the K path. Discovers separate vs fused QKV and bias
    presence from the actual modules -- no name-based assumptions."""
    cfg = model.config
    n_q = getattr(cfg, "num_attention_heads")
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_q
    hidden = getattr(cfg, "hidden_size")
    # NB: a config may have head_dim PRESENT but set to None -> getattr default won't fire
    head_dim = getattr(cfg, "head_dim", None) or (hidden // n_q)
    rope_theta = getattr(cfg, "rope_theta", None) or getattr(cfg, "rope_base", None)
    layers = get_decoder_layers(model)
    out = []
    for i, layer in enumerate(layers):
        attn, attn_name = find_attn_module(layer)
        info = dict(
            layer_idx=i,
            attn_module=attn,
            attn_path=attn_name,
            n_q_heads=n_q,
            n_kv_heads=n_kv,
            head_dim=head_dim,
            rope_theta=rope_theta,
            fused=False,
            k_proj=None,
            qkv_proj=None,
            has_q_bias=False,
            has_k_bias=False,
            has_v_bias=False,
            k_bias=None,
            k_slice=None,
        )
        if hasattr(attn, "k_proj"):  # separate q/k/v (Qwen2/Qwen3/Llama/Mistral/Phi-2)
            info["k_proj"] = attn.k_proj
            info["has_q_bias"] = (
                getattr(attn, "q_proj", None) is not None
                and attn.q_proj.bias is not None
            )
            info["has_k_bias"] = attn.k_proj.bias is not None
            info["has_v_bias"] = (
                getattr(attn, "v_proj", None) is not None
                and attn.v_proj.bias is not None
            )
            info["k_bias"] = (
                attn.k_proj.bias.detach().clone()
                if attn.k_proj.bias is not None
                else None
            )
        else:  # fused QKV (Phi-3/Phi-4 style: qkv_proj or Wqkv)
            fused = None
            for fn in ("qkv_proj", "Wqkv", "query_key_value", "c_attn"):
                if hasattr(attn, fn):
                    fused = getattr(attn, fn)
                    break
            if fused is None:
                out.append(info)
                continue
            info["fused"] = True
            info["qkv_proj"] = fused
            q_sz, kv_sz = n_q * head_dim, n_kv * head_dim
            info["k_slice"] = (q_sz, q_sz + kv_sz)
            if fused.bias is not None:
                info["has_q_bias"] = info["has_v_bias"] = info["has_k_bias"] = True
                info["k_bias"] = fused.bias.detach().clone()[q_sz : q_sz + kv_sz]
        out.append(info)
    return out


def k_bias_vector(info, device=None, dtype=torch.float32):
    """K bias as [n_kv_heads*head_dim] (zeros if biasless -- makes pre-bias a no-op)."""
    n = info["n_kv_heads"] * info["head_dim"]
    if info["k_bias"] is None:
        v = torch.zeros(n, dtype=dtype)
    else:
        v = info["k_bias"].to(dtype)
    return v.to(device) if device is not None else v


# ------------------------------------------------------------------------ FP8 fake quant
def fake_quant_fp8(x, layout="per_tensor", unit_scale=False):
    """Symmetric FP8 e4m3 fake-quant. layout in {per_tensor, per_channel, per_head}.
    unit_scale=True ignores the data range (raw representability diagnostic)."""
    orig = x.dtype
    xf = x.float()
    if unit_scale:
        q = xf.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
        return q.to(orig)
    if layout == "per_tensor":
        scale = xf.abs().amax().clamp(min=1e-8) / FP8_MAX
    elif layout in ("per_channel", "per_head"):  # scale over last dim
        scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / FP8_MAX
    else:
        raise ValueError(layout)
    q = (xf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale
    return q.to(orig)


def fake_quant_int8(x, layout="per_tensor"):
    orig = x.dtype
    xf = x.float()
    if layout == "per_tensor":
        scale = xf.abs().amax().clamp(min=1e-8) / 127.0
    else:
        scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = (xf / scale).round().clamp(-127, 127) * scale
    return q.to(orig)


def fake_quant_intN(x, bits, layout="per_tensor"):
    orig = x.dtype
    xf = x.float()
    qmax = 2 ** (bits - 1) - 1
    if layout == "per_tensor":
        scale = xf.abs().amax().clamp(min=1e-8) / qmax
    else:
        scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = (xf / scale).round().clamp(-qmax, qmax) * scale
    return q.to(orig)


# ----------------------------------------------------------- K-cache quant install (hooks)
class KQuantHarness:
    """Installs an attention-interface hook so the cached K/V can be fake-quantized at the point
    the production KV cache sees them (post-RoPE for K). Supports cells:
      bf16, fp8 (normal post-bias/post-RoPE), prebias_fp8, k16v8, fp8_pc (per-channel K), int8.
    The bias used for the prebias/alpha paths is per-(layer) and looked up by module identity.
    """

    def __init__(self, model, infos, cell="bf16", fp8_layout="per_tensor"):
        self.model = model
        self.cell = cell
        self.fp8_layout = fp8_layout
        self.orig = None
        self.handles = []
        self.impl = model.config._attn_implementation
        self.by_mod = {id(info["attn_module"]): info for info in infos}
        self.infos = infos

    def _kproj_hook(self, info):
        # PRE-RoPE faithful pre-bias: quantize (k_proj_out - b_K), add b_K back in fp16, then
        # the model applies RoPE to the corrected K normally. Matches the KV-PREBIAS recipe.
        # Reads the LIVE module bias (mod.bias) so it stays correct if alpha scales the bias.
        layout = self.fp8_layout
        if info["k_bias"] is None:
            return None  # biasless: prebias is a no-op, skip the hook
        k_slice = info["k_slice"] if info["fused"] else None

        def hook(mod, inp, out):
            if mod.bias is None:
                return out
            if k_slice is not None:
                s0, s1 = k_slice
                ks = out[..., s0:s1]
                bb = mod.bias[s0:s1].to(ks.dtype)
                out = out.clone()
                out[..., s0:s1] = fake_quant_fp8(ks - bb, layout=layout) + bb
                return out
            bb = mod.bias.to(out.dtype)
            return fake_quant_fp8(out - bb, layout=layout) + bb

        return hook

    def install(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.orig = ALL_ATTENTION_FUNCTIONS[self.impl]
        orig = self.orig
        h = self
        if self.cell == "prebias_fp8":
            for info in self.infos:
                proj = info["k_proj"] if not info["fused"] else info["qkv_proj"]
                hk = self._kproj_hook(info)
                if proj is not None and hk is not None:
                    self.handles.append(proj.register_forward_hook(hk))

        V8_CELLS = {"fp8", "fp8_pc", "int8", "k16v8", "prebias_fp8"}

        def hook(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
            info = h.by_mod.get(id(module))
            if info is not None and h.cell != "bf16":
                if h.cell == "fp8":
                    k = fake_quant_fp8(k, layout=h.fp8_layout)
                elif h.cell == "fp8_pc":
                    k = fake_quant_fp8(k, layout="per_channel")
                elif h.cell == "int8":
                    k = fake_quant_int8(k, layout=h.fp8_layout)
                elif (
                    h.cell == "kint8"
                ):  # K-only INT8, V native (paper INT8/INT6 classifier)
                    k = fake_quant_int8(k, layout=h.fp8_layout)
                elif h.cell == "kint6":  # K-only INT6, V native
                    k = fake_quant_intN(k, 6, layout=h.fp8_layout)
                # k16v8 + prebias_fp8: K already native / handled pre-RoPE
                if h.cell in V8_CELLS:
                    v = fake_quant_fp8(v, layout=h.fp8_layout)
            return orig(
                module, q, k, v, attention_mask, dropout=dropout, scaling=scaling, **kw
            )

        ALL_ATTENTION_FUNCTIONS[self.impl] = hook
        return self

    def remove(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        if self.orig is not None:
            ALL_ATTENTION_FUNCTIONS[self.impl] = self.orig
            self.orig = None
        for hd in self.handles:
            hd.remove()
        self.handles = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *a):
        self.remove()


# ----------------------------------------------------------------- alpha bias patch (causal)
class AlphaKBiasPatch:
    """Temporarily scale ONLY the K-projection bias by alpha (Q/V untouched), restoring on exit.
    Works for separate k_proj.bias and the K slice of a fused qkv bias."""

    def __init__(self, infos, alpha):
        self.infos = infos
        self.alpha = alpha
        self.saved = []

    def __enter__(self):
        for info in self.infos:
            if not info["has_k_bias"]:
                continue
            if info["k_proj"] is not None and info["k_proj"].bias is not None:
                b = info["k_proj"].bias
                self.saved.append((b, b.detach().clone()))
                b.data.mul_(self.alpha)
            elif info["fused"] and info["qkv_proj"].bias is not None:
                b = info["qkv_proj"].bias
                s0, s1 = info["k_slice"]
                self.saved.append((b, b.detach().clone()))
                b.data[s0:s1].mul_(self.alpha)
        return self

    def __exit__(self, *a):
        for b, orig in self.saved:
            b.data.copy_(orig)
        self.saved = []


# -------------------------------------------------------------------------- stats helpers
def bias_stats(vec):
    """vec: 1D tensor (|values| computed inside). Returns dict of magnitude stats."""
    a = vec.float().abs()
    a_sorted = a.sort().values
    n = a.numel()

    def pct(p):
        return a_sorted[min(n - 1, int(p * n))].item()

    rms = vec.float().pow(2).mean().sqrt().item()
    mx = a.max().item()
    p99 = pct(0.99)
    return dict(
        max_abs=mx,
        p999_abs=pct(0.999),
        p99_abs=p99,
        p95_abs=pct(0.95),
        mean_abs=a.mean().item(),
        rms=rms,
        std=vec.float().std().item(),
        max_over_p99=mx / max(p99, 1e-9),
        max_over_rms=mx / max(rms, 1e-9),
        frac_abs_gt_448=(a > FP8_MAX).float().mean().item(),
    )


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def write_csv(path, rows, fields):
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def calib_prompts(tok, n=32, seq_len=2048):
    """WikiText-103 snippets, tokenized to seq_len. Deterministic."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    text = "\n".join(t for t in ds["text"][:200000] if t and not t.isspace())
    ids = tok(text)["input_ids"]
    chunks = []
    step = seq_len
    for i in range(0, len(ids) - seq_len, step):
        chunks.append(ids[i : i + seq_len])
        if len(chunks) >= n:
            break
    return chunks


# ============================================================================ TIER-2 additions
# Flexible independent-K/V fake-quant for the FP8-variant probe and the Phi/FII diagnosis:
# per-K and per-V {fmt, bits, layout, group}, optional pre-bias K (pre-RoPE residual), and a
# per-layer filter (quantize only listed layers -- for the layer sweep).


def _quant_lastdims(x, fmt, bits, layout, group, unit_scale):
    """x: [..., T, D] (post-RoPE K) or [..., D]. layout over the T/D structure of the last 2 dims.
    fmt in {'fp8','int'}; bits used for int (8/6/4); fp8 is e4m3 (bits ignored)."""
    if fmt is None or bits >= 16:
        return x
    orig = x.dtype
    xf = x.float()
    qmax = FP8_MAX if fmt == "fp8" else (2 ** (bits - 1) - 1)

    def cast(z):
        if fmt == "fp8":
            return z.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
        return z.round().clamp(-qmax, qmax)

    if unit_scale:
        return cast(xf).to(orig)
    # scale dims: reduce over the chosen axes of the last two dims [T, D]
    if layout == "per_tensor":
        scale = xf.abs().amax().clamp(min=1e-8) / qmax
    elif (
        layout == "per_head"
    ):  # one scale per head (assumes head is dim -3): reduce [T,D]
        scale = xf.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8) / qmax
    elif layout == "per_token":  # per (head, token): reduce over D
        scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    elif layout == "per_channel":  # per (head, channel): reduce over T
        scale = xf.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8) / qmax
    elif layout == "per_group":  # per (head, token, channel-group)
        D = xf.shape[-1]
        g = min(group, D)
        ng = (D + g - 1) // g
        pad = ng * g - D
        xp = F.pad(xf, (0, pad))
        xg = xp.reshape(*xp.shape[:-1], ng, g)
        sc = xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        q = cast(xg / sc) * sc
        return q.reshape(*xp.shape)[..., :D].to(orig)
    else:
        raise ValueError(layout)
    return (cast(xf / scale) * scale).to(orig)


def parse_spec(s):
    """'fp8:per_tensor', 'int8:per_token', 'int4:per_group:128', 'bf16' -> dict."""
    if s in (None, "bf16", "none", "fp16"):
        return dict(fmt=None, bits=16, layout="per_tensor", group=128)
    parts = s.split(":")
    head = parts[0]
    layout = parts[1] if len(parts) > 1 else "per_tensor"
    group = int(parts[2]) if len(parts) > 2 else 128
    if head.startswith("fp8"):
        return dict(fmt="fp8", bits=8, layout=layout, group=group)
    if head.startswith("int"):
        return dict(fmt="int", bits=int(head[3:]), layout=layout, group=group)
    raise ValueError(s)


class FlexKVHarness:
    """Independent K/V fake-quant at the cache point. k_spec/v_spec are parse_spec dicts.
    prebias=True: K quantized as the PRE-RoPE residual (k_proj_out - b_K) + b_K (faithful fix),
    in which case k_spec applies to that residual and the post-RoPE K branch is skipped. layers:
    set of layer indices to quantize (None = all)."""

    def __init__(
        self, model, infos, k_spec, v_spec, prebias=False, layers=None, unit_scale=False
    ):
        self.model = model
        self.infos = infos
        self.k = k_spec
        self.v = v_spec
        self.prebias = prebias
        self.layers = layers
        self.unit_scale = unit_scale
        self.impl = model.config._attn_implementation
        self.by_mod = {id(i["attn_module"]): i for i in infos}
        self.orig = None
        self.handles = []

    def _kproj_hook(self, info):
        if info["k_bias"] is None:
            return None
        k_slice = info["k_slice"] if info["fused"] else None
        spec = self.k
        us = self.unit_scale

        def hook(mod, inp, out):
            if mod.bias is None:
                return out
            if k_slice is not None:
                s0, s1 = k_slice
                ks = out[..., s0:s1]
                bb = mod.bias[s0:s1].to(ks.dtype)
                out = out.clone()
                out[..., s0:s1] = (
                    _quant_lastdims(
                        ks - bb,
                        spec["fmt"],
                        spec["bits"],
                        spec["layout"],
                        spec["group"],
                        us,
                    )
                    + bb
                )
                return out
            bb = mod.bias.to(out.dtype)
            return (
                _quant_lastdims(
                    out - bb,
                    spec["fmt"],
                    spec["bits"],
                    spec["layout"],
                    spec["group"],
                    us,
                )
                + bb
            )

        return hook

    def install(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.orig = ALL_ATTENTION_FUNCTIONS[self.impl]
        orig = self.orig
        h = self
        if self.prebias:
            for info in self.infos:
                if self.layers is not None and info["layer_idx"] not in self.layers:
                    continue
                proj = info["k_proj"] if not info["fused"] else info["qkv_proj"]
                hk = self._kproj_hook(info)
                if proj is not None and hk is not None:
                    self.handles.append(proj.register_forward_hook(hk))

        def hook(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
            info = h.by_mod.get(id(module))
            if info is not None and (h.layers is None or info["layer_idx"] in h.layers):
                if not h.prebias and h.k["fmt"] is not None:
                    k = _quant_lastdims(
                        k,
                        h.k["fmt"],
                        h.k["bits"],
                        h.k["layout"],
                        h.k["group"],
                        h.unit_scale,
                    )
                if h.v["fmt"] is not None:
                    v = _quant_lastdims(
                        v,
                        h.v["fmt"],
                        h.v["bits"],
                        h.v["layout"],
                        h.v["group"],
                        h.unit_scale,
                    )
            return orig(
                module, q, k, v, attention_mask, dropout=dropout, scaling=scaling, **kw
            )

        ALL_ATTENTION_FUNCTIONS[self.impl] = hook
        return self

    def remove(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        if self.orig is not None:
            ALL_ATTENTION_FUNCTIONS[self.impl] = self.orig
            self.orig = None
        for hd in self.handles:
            hd.remove()
        self.handles = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *a):
        self.remove()


@torch.no_grad()
def real_ppl(model, tok, device, n=16, seq_len=2048, harness=None):
    """Real next-token PPL (NLL over the dataset's own targets) -- the near-serving metric, NOT a
    vs-BF16 logit diff. If harness is given, it is installed around the forward (quantized cache).
    """
    chunks = calib_prompts(tok, n=n, seq_len=seq_len)
    if harness is not None:
        harness.install()
    tot_nll, tot_tok = 0.0, 0
    try:
        for ids in chunks:
            t = torch.tensor(ids).unsqueeze(0).to(device)
            out = model(t)
            lp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
            tgt = t[0, 1:]
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            tot_nll += nll.sum().item()
            tot_tok += tgt.numel()
    finally:
        if harness is not None:
            harness.remove()
    import math

    return math.exp(tot_nll / max(tot_tok, 1)), tot_tok
