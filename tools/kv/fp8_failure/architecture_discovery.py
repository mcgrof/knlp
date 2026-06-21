"""Consolidated, GPU-free architecture discovery for the FP8 KV-cache failure atlas.

One place that answers, from the live module graph (never name-guessing), every structural fact a
quant probe needs before it touches a GPU: how many query/kv heads, head_dim, separate vs fused
QKV, which projections carry a bias, and -- the part the tier-1 `discover_attention` does not do --
the partial-RoPE subspace partition (which K channels are rotated vs pass-through). Phi-2 rotates
only the first `rotary_dim` of `head_dim` channels; the pass-through tail is a different
distribution and a prime FP8-failure suspect, so probes must be able to address the two subspaces
separately. This module reuses `k_bias_common.discover_attention` for the K-path facts and adds the
RoPE geometry + boolean channel masks on top.
"""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # tools/kv
import k_bias_common as kbc  # noqa: E402
import torch  # noqa: E402


def rope_geometry(model):
    """(rotary_dim, head_dim). rotary_dim == head_dim => full RoPE (no pass-through tail).

    Resolves rotary_dim from an explicit config field, else partial_rotary_factor*head_dim, else
    falls back to full RoPE. head_dim may be PRESENT-but-None in a config, hence the `or`.
    """
    cfg = model.config
    n_q = getattr(cfg, "num_attention_heads")
    hidden = getattr(cfg, "hidden_size")
    head_dim = getattr(cfg, "head_dim", None) or (hidden // n_q)
    rotary_dim = getattr(cfg, "rotary_dim", None)
    prf = getattr(cfg, "partial_rotary_factor", None)
    if rotary_dim is None and prf is not None:
        rotary_dim = int(head_dim * prf)
    if rotary_dim is None:
        rotary_dim = head_dim
    return int(rotary_dim), int(head_dim)


def subspace_masks(rotary_dim, head_dim, device=None):
    """Boolean [head_dim] masks (rotary, passthrough) over a single head's channels.

    HF applies RoPE to the first `rotary_dim` channels and passes the rest through unchanged. The
    two masks partition the head exactly (their OR is all-ones, their AND is empty)."""
    rot = torch.zeros(head_dim, dtype=torch.bool, device=device)
    rot[:rotary_dim] = True
    return rot, ~rot


def rope_pair_index(rotary_dim, head_dim, device=None):
    """For each channel, the index of its RoPE-rotation partner (HF `rotate_half`: channel i and
    i+rotary_dim/2 form a rotated pair). Pass-through channels are their own partner. Returned as a
    LongTensor[head_dim] -- used to enforce gauge sharing within a pair (see qk_gauge).
    """
    idx = torch.arange(head_dim, device=device)
    half = rotary_dim // 2
    pair = idx.clone()
    pair[:half] = idx[:half] + half
    pair[half:rotary_dim] = idx[half:rotary_dim] - half
    return pair


def discover(model):
    """Full per-layer structural map: tier-1 K-path facts + RoPE geometry + subspace masks.

    Each row is the `k_bias_common.discover_attention` dict augmented with rotary_dim, is_partial,
    and the (rotary, passthrough) boolean head masks. Pure introspection; no forward pass, no GPU.
    """
    infos = kbc.discover_attention(model)
    rotary_dim, head_dim = rope_geometry(model)
    rot_mask, pass_mask = subspace_masks(rotary_dim, head_dim)
    for info in infos:
        info["rotary_dim"] = rotary_dim
        info["is_partial_rope"] = rotary_dim < info["head_dim"]
        info["rotary_mask"] = rot_mask
        info["passthrough_mask"] = pass_mask
    return infos


def summarize(infos):
    """One small dict for the manifest / report header (the things that change the quant story)."""
    if not infos:
        return {}
    a = infos[0]
    return dict(
        n_layers=len(infos),
        n_q_heads=a["n_q_heads"],
        n_kv_heads=a["n_kv_heads"],
        head_dim=a["head_dim"],
        gqa_groups=a["n_q_heads"] // a["n_kv_heads"],
        fused_qkv=a["fused"],
        has_k_bias=any(i["has_k_bias"] for i in infos),
        rotary_dim=a.get("rotary_dim"),
        is_partial_rope=a.get("is_partial_rope", False),
    )
