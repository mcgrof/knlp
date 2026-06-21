"""Phase 6: score-preserving Q/K diagonal-gauge equalization.

Attention scores q.k are invariant to a matched inverse diagonal rescale K_i <- D_i K_i,
Q_i <- Q_i / D_i. This lets us change K's quantization difficulty WITHOUT changing the attention
function -- a causal probe (is K-cache fragility a free Q/K-scaling degree of freedom training had
no loss incentive to control?) and a potential training-free repair.

Two variants:
  (A) ACTIVATION probe -- applied POST-RoPE per-channel. RoPE already rotated q,k, so scaling
      channel i of k by D_i and channel i of q by 1/D_i preserves every term q_i*k_i exactly,
      for ANY per-channel D. Exact, simple; used to study K-range vs FP8 error at constant scores.
  (B) WEIGHT-FOLD candidate -- folded PRE-RoPE into the projection weights+biases (deployable, no
      cache-format overhead). RoPE mixes channels within each rotary pair, so D must be SHARED
      within each rotary pair to commute with the rotation; pass-through dims are free. For GQA the
      KV-head's D applies (inverse) to every query head sharing that KV head. Fused QKV is sliced.

GPU-free: the math + folds + invariance are all verifiable on tiny CPU tensors (see the tests).
"""

import torch

from . import common as C
from .common import kbc


# ----------------------------------------------------------------- score helper + D construction
def scores_from_qk(q, k, scaling, num_kv_groups=1):
    """q:[B,Hq,T,D] k:[B,Hkv,T,D] -> pre-softmax scores [B,Hq,T,T] (GQA-expanded)."""
    if num_kv_groups > 1:
        k = k.repeat_interleave(num_kv_groups, dim=1)
    return torch.matmul(q.float(), k.float().transpose(2, 3)) * scaling


def symmetrize_rotary_D(D, rotary_dim):
    """Make D commute with RoPE: within the rotary block, channels i and i+rotary_dim/2 are a
    rotated pair (HF rotate_half), so force D[i]==D[i+rd/2] (geometric mean). Pass-through dims
    (>= rotary_dim) are left free. D: [..., head_dim]."""
    D = D.clone()
    half = rotary_dim // 2
    a = D[..., :half]
    b = D[..., half:rotary_dim]
    g = (a * b).sqrt()
    D[..., :half] = g
    D[..., half:rotary_dim] = g
    return D


def make_D_from_target_amax(k_amax_per_channel, target=None, clamp=(0.25, 4.0)):
    """Diagonal D per K channel that equalizes |K| range toward `target` (default = median amax).
    Clamped to avoid blowing up the Q side (inverse D). k_amax_per_channel: [..., head_dim].
    """
    a = k_amax_per_channel.clamp(min=1e-8)
    t = target if target is not None else a.median()
    D = (t / a).clamp(*clamp)  # D<1 shrinks a hot channel's K (Q grows by 1/D)
    return D


# ------------------------------------------------------------------ (A) activation-level probe
class QKGaugeProbe:
    """Post-RoPE per-channel gauge in the attention-interface hook: K_i*=D, Q_i*=1/D (exact score
    invariance), then optionally fake-quant K. D_by_kvhead: [n_kv_heads, head_dim] (or None=ident).
    quant_k: a parse_spec string ('fp8:per_tensor') or None."""

    def __init__(self, model, infos, D_by_layer=None, quant_k=None):
        self.model = model
        self.infos = infos
        self.D = D_by_layer or {}  # layer_idx -> [n_kv, head_dim]
        self.qk = kbc.parse_spec(quant_k) if quant_k else None
        self.impl = model.config._attn_implementation
        self.by_mod = {id(i["attn_module"]): i for i in infos}
        self.orig = None

    def install(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.orig = ALL_ATTENTION_FUNCTIONS[self.impl]
        orig = self.orig
        h = self

        def hook(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
            info = h.by_mod.get(id(module))
            if info is not None and info["layer_idx"] in h.D:
                D = h.D[info["layer_idx"]].to(k.dtype).to(k.device)  # [n_kv, hd]
                # GQA group count: trust the module, else derive from discovery (some arches
                # don't expose num_key_value_groups on the attention module).
                g = getattr(module, "num_key_value_groups", None) or (
                    info["n_q_heads"] // info["n_kv_heads"]
                )
                k = k * D[None, :, None, :]  # K_i *= D_i  (per kv head/chan)
                qD = D.repeat_interleave(g, dim=0)  # broadcast kv-head D to q heads
                q = q / qD[None, :, None, :]  # Q_i *= 1/D_i  -> scores invariant
                if h.qk is not None:
                    k = kbc._quant_lastdims(
                        k,
                        h.qk["fmt"],
                        h.qk["bits"],
                        h.qk["layout"],
                        h.qk["group"],
                        False,
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

    def __enter__(self):
        return self.install()

    def __exit__(self, *a):
        self.remove()


# --------------------------------------------------------------- (B) weight-folded candidate
@torch.no_grad()
def fold_qk_gauge_weights(info, D, rotary_dim):
    """Fold a per-(kv_head, head_dim) gauge D into the layer's q_proj/k_proj weights+biases IN
    PLACE (deployable, no cache-format change). Enforces RoPE-pair sharing on rotary dims (so it
    commutes with RoPE), broadcasts the KV-head D inverse to every GQA query head, and transforms
    biases too. Handles separate and fused QKV. D: [n_kv_heads, head_dim].

    Returns a restore() closure (saves the touched slices) so callers can undo it.
    """
    n_kv, hd = info["n_kv_heads"], info["head_dim"]
    n_q = info["n_q_heads"]
    g = n_q // n_kv
    Ds = symmetrize_rotary_D(D, rotary_dim)  # [n_kv, hd], RoPE-safe
    k_scale = Ds.reshape(-1)  # [n_kv*hd]
    q_scale = (
        (1.0 / Ds).repeat_interleave(g, dim=0).reshape(-1)
    )  # [n_q*hd], inverse, GQA-broadcast
    saved = []

    def scale_rows(mod, sl, vec):
        # scale OUTPUT rows of a Linear (weight rows + bias) by vec elementwise
        w = mod.weight
        s0, s1 = sl
        saved.append((w, s0, s1, w[s0:s1].detach().clone()))
        w[s0:s1] *= vec.to(w.dtype).to(w.device)[:, None]
        if mod.bias is not None:
            b = mod.bias
            saved.append((b, s0, s1, b[s0:s1].detach().clone()))
            b[s0:s1] *= vec.to(b.dtype).to(b.device)

    if not info["fused"]:
        scale_rows(info["k_proj"], (0, n_kv * hd), k_scale)
        scale_rows(info["attn_module"].q_proj, (0, n_q * hd), q_scale)
    else:
        qkv = info["qkv_proj"]
        q0 = 0
        k0 = n_q * hd
        scale_rows(qkv, (k0, k0 + n_kv * hd), k_scale)
        scale_rows(qkv, (q0, q0 + n_q * hd), q_scale)

    def restore():
        for t, s0, s1, orig in saved:
            t.data[s0:s1] = orig

    return restore
