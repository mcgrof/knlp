# SPDX-License-Identifier: GPL-2.0
"""Paired caches from two frozen models on byte-identical prompts.

Everything a cross-model mapper needs before it can be fitted, and the checks
that say the coordinates are what they are believed to be.

The load-bearing subtlety is RoPE.  A cache stores keys *after* the rotation, so
two models only share a content space once the rotation is undone.  Both models
in the intended pair use the same ``rope_theta``, but their head dimensions
differ, and RoPE frequencies are ``theta**(-2i/d)`` -- so the frequency sets
differ too and each side must be de-rotated with its own ``d``.  The helper
infers ``d`` from the tensor, so this happens correctly as long as each model's
own keys are passed to it, which :func:`verify_rope` exists to confirm rather
than assume.

Values are never rotated and need none of this, which is one reason keys and
values are fitted and judged separately throughout.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.cartridges_cas.scripts.cas_kv_rope import (  # noqa: E402
    derot,
    fast_pair_cosine,
    rerot,
)


@dataclass
class ModelGeometry:
    """Everything about a model's cache that a mapper has to respect."""

    model_id: str
    revision: str
    n_layers: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    hidden: int
    rope_theta: float
    rope_scaling: object
    vocab: int

    @property
    def group(self) -> int:
        return self.n_q_heads // self.n_kv_heads

    @property
    def kv_elems_per_token(self) -> int:
        """Per layer, per tensor (K or V)."""
        return self.n_kv_heads * self.head_dim

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["group"] = self.group
        d["kv_elems_per_token_per_layer"] = self.kv_elems_per_token
        return d


def describe(model, model_id: str, revision: str = "") -> ModelGeometry:
    c = model.config
    head_dim = int(
        getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    )
    return ModelGeometry(
        model_id=model_id,
        revision=revision,
        n_layers=int(c.num_hidden_layers),
        n_q_heads=int(c.num_attention_heads),
        n_kv_heads=int(getattr(c, "num_key_value_heads", c.num_attention_heads)),
        head_dim=head_dim,
        hidden=int(c.hidden_size),
        rope_theta=float(getattr(c, "rope_theta", 0.0) or 0.0),
        rope_scaling=getattr(c, "rope_scaling", None),
        vocab=int(c.vocab_size),
    )


def check_pair(src: ModelGeometry, tgt: ModelGeometry) -> dict:
    """What a mapper between these two has to handle, and what would block it.

    Blockers are returned rather than raised so the audit can report all of
    them at once instead of stopping at the first.
    """
    blockers = []
    if src.vocab != tgt.vocab:
        blockers.append(f"vocab differs: {src.vocab} vs {tgt.vocab}")
    if src.rope_scaling is not None or tgt.rope_scaling is not None:
        blockers.append(
            "a non-default rope scaling is present; the de-rotation helper "
            "implements the default schedule only"
        )
    return {
        "shared_tokenizer_vocab": src.vocab == tgt.vocab,
        "head_dim_changes": src.head_dim != tgt.head_dim,
        "head_dim": [src.head_dim, tgt.head_dim],
        "kv_heads_match": src.n_kv_heads == tgt.n_kv_heads,
        "kv_heads": [src.n_kv_heads, tgt.n_kv_heads],
        "layers": [src.n_layers, tgt.n_layers],
        "layers_align_one_to_one": src.n_layers == tgt.n_layers,
        "rope_theta": [src.rope_theta, tgt.rope_theta],
        "rope_theta_matches": src.rope_theta == tgt.rope_theta,
        "target_cache_bytes_ratio": (
            (tgt.n_layers * tgt.kv_elems_per_token)
            / max(src.n_layers * src.kv_elems_per_token, 1)
        ),
        "blockers": blockers,
    }


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


@dataclass
class Prefill:
    """One model's cache for one prompt, in both key frames."""

    keys_post: list  # [L][1, n_kv, T, D] as stored, post-RoPE
    keys_pre: list  # [L][1, n_kv, T, D] content keys, pre-RoPE
    values: list  # [L][1, n_kv, T, D]
    n_tokens: int
    geometry: ModelGeometry
    info: dict = field(default_factory=dict)


def _kv_heads_from_proj(out: torch.Tensor, n_kv: int, head_dim: int) -> torch.Tensor:
    """``[B, T, n_kv*D]`` from a projection into the canonical ``[B, n_kv, T, D]``."""
    b, t, _ = out.shape
    return out.reshape(b, t, n_kv, head_dim).transpose(1, 2).contiguous()


@torch.no_grad()
def prefill(model, input_ids: torch.Tensor, geom: ModelGeometry) -> Prefill:
    """Run the prompt once and keep both key frames.

    The pre-RoPE keys come from a hook on each layer's key projection, which is
    the only place they exist unrotated.  This pair applies no query/key
    normalisation, so the projection output is the content key; a model that
    normalises keys would need the hook moved past that step, and
    :func:`verify_rope` would catch the mistake if it were not.
    """
    from transformers import DynamicCache

    pre: dict = {}
    hooks = []
    for li, layer in enumerate(model.model.layers):
        hooks.append(
            layer.self_attn.k_proj.register_forward_hook(
                lambda m, i, o, l=li: pre.__setitem__(
                    l, _kv_heads_from_proj(o.detach(), geom.n_kv_heads, geom.head_dim)
                )
            )
        )
    try:
        cache = DynamicCache()
        model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    finally:
        for h in hooks:
            h.remove()

    keys_post = [layer.keys.detach() for layer in cache.layers]
    values = [layer.values.detach() for layer in cache.layers]
    keys_pre = [pre[li] for li in range(len(keys_post))]
    return Prefill(
        keys_post=keys_post,
        keys_pre=keys_pre,
        values=values,
        n_tokens=int(input_ids.shape[1]),
        geometry=geom,
    )


def verify_rope(p: Prefill, positions: Optional[torch.Tensor] = None) -> dict:
    """Confirm the stored keys really are the content keys rotated by position.

    Three checks, because any one alone can pass on a coincidence. The
    de-rotated stored key must match the hooked content key; de-rotating with
    positions shifted by one must *not* match, judged on the fastest-turning
    pairs where a one-slot error cannot hide; and a round trip must return the
    original. Without the middle check a harness that ignored positions
    entirely would look correct.
    """
    T = p.n_tokens
    pos = positions if positions is not None else torch.arange(T)
    theta = p.geometry.rope_theta
    worst_match, worst_shift, worst_rt = 1.0, 0.0, 0.0
    per_layer = []
    for li, (post, pre_k) in enumerate(zip(p.keys_post, p.keys_pre)):
        stored = post[0].float().cpu()
        content = pre_k[0].float().cpu()
        m = float(fast_pair_cosine(derot(stored, pos, theta), content).min())
        sh = float(fast_pair_cosine(derot(stored, pos + 1, theta), content, 8).max())
        rt_t = rerot(derot(stored, pos, theta), pos, theta)
        rt = float((rt_t - stored).norm() / stored.norm().clamp(min=1e-30))
        worst_match = min(worst_match, m)
        worst_shift = max(worst_shift, sh)
        worst_rt = max(worst_rt, rt)
        per_layer.append(
            {"layer": li, "derot_cos": m, "offbyone_cos": sh, "roundtrip_rel": rt}
        )
    passed = worst_match > 0.999 and worst_shift < 0.98 and worst_rt < 1e-3
    return {
        "passed": bool(passed),
        "derot_cos_min": worst_match,
        "offbyone_cos_max": worst_shift,
        "roundtrip_rel_max": worst_rt,
        "per_layer": per_layer,
    }


def to_content_keys(p: Prefill) -> list:
    """De-rotate the stored keys into the frame a mapper can be fitted in."""
    pos = torch.arange(p.n_tokens, device=p.keys_post[0].device)
    return [derot(k.float(), pos, p.geometry.rope_theta) for k in p.keys_post]


def to_stored_keys(
    content: Sequence[torch.Tensor], geom: ModelGeometry, n_tokens: int
) -> list:
    """Re-rotate content keys at the target positions, for injection."""
    pos = torch.arange(n_tokens, device=content[0].device)
    return [rerot(k.float(), pos, geom.rope_theta) for k in content]
