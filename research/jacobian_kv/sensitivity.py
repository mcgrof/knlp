# SPDX-License-Identifier: GPL-2.0
"""Receiver-side sensitivity of a frozen model to errors in its own KV cache.

The object of interest is the second moment

    G = E_x[ J_t(x)' F_t(x) J_t(x) ]

where ``J_t`` maps a target cache block to the target model's future logits and
``F_t`` is an output metric -- the categorical Fisher ``diag(p) - p p'`` to
start.  ``G`` is never materialised.  For an output-space random vector ``z``
with covariance ``F``, one vector-Jacobian product gives ``u = J' z`` and
``E[u u'] = J' F J``, so a handful of probes sketches ``G`` in the only form
that is ever needed: a directional score ``e' G e ~= mean_p (u_p . e)^2``.

Sampling ``z`` needs no matrix square root.  Draw a label ``y ~ Categorical(p)``
and set ``z = onehot(y) - p``; then ``E[z] = 0`` and ``Cov(z) = diag(p) - p p'``
exactly.  The VJP is then just the gradient of the sampled log-likelihood with
respect to the cache block, which is one ordinary backward pass.

The signed mean-Jacobian control is the same machinery with one deliberate
change: the probe directions are held *fixed across contexts* and the resulting
VJPs are averaged before squaring, which builds ``E[J]' F E[J]`` instead.  The
contrast between the two is the whole question -- whether directions that cancel
across contexts still carry squared output effect.

Everything here keeps both models frozen and touches no training state.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import torch

CacheKind = str  # "k" or "v"
BlockKey = tuple  # (layer_index, CacheKind)


# ---------------------------------------------------------------------------
# cache plumbing
# ---------------------------------------------------------------------------


def freeze_model(model) -> None:
    """Drop grad on every parameter so only grafted cache leaves accumulate."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def _layer_tensors(cache, layer_idx: int):
    """Return the ``(keys, values)`` tensors of one cache layer.

    Written against ``transformers`` 4.56 ``DynamicCache.layers[i].keys`` while
    still accepting the older ``key_cache``/``value_cache`` list form, because a
    silent mismatch here would look like a model that has no cache sensitivity.
    """
    layers = getattr(cache, "layers", None)
    if layers is not None:
        layer = layers[layer_idx]
        return layer.keys, layer.values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def _set_layer_tensors(cache, layer_idx: int, keys, values) -> None:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        layers[layer_idx].keys = keys
        layers[layer_idx].values = values
        return
    cache.key_cache[layer_idx] = keys
    cache.value_cache[layer_idx] = values


def n_cache_layers(cache) -> int:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return len(layers)
    return len(cache.key_cache)


@torch.no_grad()
def prefill(model, input_ids: torch.Tensor, attention_mask=None):
    """Run the prompt and return a detached ``DynamicCache``."""
    from transformers import DynamicCache

    cache = DynamicCache()
    model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
    )
    for i in range(n_cache_layers(cache)):
        k, v = _layer_tensors(cache, i)
        _set_layer_tensors(cache, i, k.detach(), v.detach())
    return cache


def clone_cache(cache):
    """Deep copy of a cache's tensors, structure shared."""
    import copy

    out = copy.deepcopy(cache)
    for i in range(n_cache_layers(out)):
        k, v = _layer_tensors(out, i)
        _set_layer_tensors(out, i, k.clone(), v.clone())
    return out


def graft_leaves(
    cache,
    layers: Sequence[int],
    which: Sequence[CacheKind] = ("k", "v"),
) -> dict:
    """Replace the named blocks with grad-requiring leaves, in place.

    ``DynamicLayer.update`` concatenates onto the stored tensor, so a leaf
    planted here stays on the autograd path through the continuation forward.

    Returns a ``{(layer, kind): leaf_tensor}`` map.
    """
    leaves: dict = {}
    for li in layers:
        k, v = _layer_tensors(cache, li)
        nk, nv = k, v
        if "k" in which:
            nk = k.detach().clone().requires_grad_(True)
            leaves[(li, "k")] = nk
        if "v" in which:
            nv = v.detach().clone().requires_grad_(True)
            leaves[(li, "v")] = nv
        _set_layer_tensors(cache, li, nk, nv)
    return leaves


def continuation_logits(model, cache, cont_ids: torch.Tensor, prompt_len: int):
    """Logits for a teacher-forced continuation on top of ``cache``.

    ``cache`` is consumed (extended in place), so pass a clone when the same
    prefill is reused.
    """
    n_new = cont_ids.shape[1]
    total = prompt_len + n_new
    attn = torch.ones(
        cont_ids.shape[0], total, dtype=torch.long, device=cont_ids.device
    )
    pos = torch.arange(prompt_len, total, device=cont_ids.device).unsqueeze(0)
    out = model(
        input_ids=cont_ids,
        attention_mask=attn,
        position_ids=pos,
        past_key_values=cache,
        use_cache=True,
    )
    return out.logits


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


def fisher_probe(logits: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """A zero-mean vector with covariance exactly ``diag(p) - p p'``.

    ``onehot(y) - p`` for ``y ~ Categorical(p)``.  Returned in the logits'
    dtype and shape ``[B, T, V]``.
    """
    probs = torch.softmax(logits.detach().float(), dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    idx = torch.multinomial(flat, num_samples=1, generator=generator).squeeze(-1)
    z = -flat.clone()
    z[torch.arange(flat.shape[0], device=flat.device), idx] += 1.0
    return z.reshape(probs.shape).to(logits.dtype)


def gaussian_probe(logits: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """An isotropic control probe, for checking that the metric matters."""
    return torch.randn(
        logits.shape, generator=generator, device=logits.device, dtype=torch.float32
    ).to(logits.dtype)


def vjp_from_probe(logits: torch.Tensor, leaves: dict, z: torch.Tensor, retain: bool):
    """``u = J' z`` for every grafted leaf, via one backward pass."""
    keys = list(leaves.keys())
    grads = torch.autograd.grad(
        outputs=logits,
        inputs=[leaves[k] for k in keys],
        grad_outputs=z,
        retain_graph=retain,
        allow_unused=True,
    )
    return {
        k: (torch.zeros_like(leaves[k]) if g is None else g.detach())
        for k, g in zip(keys, grads)
    }


@dataclass
class Sketch:
    """A ``[P, D]`` probe matrix per cache block; ``U' U / P`` approximates G."""

    rows: dict = field(default_factory=dict)  # BlockKey -> list of flat [D] tensors
    n_probes: int = 0
    kind: str = "second_moment"

    def add(self, per_block: dict) -> None:
        for key, u in per_block.items():
            self.rows.setdefault(key, []).append(u.reshape(-1).to(torch.float32))

    def stack(self) -> dict:
        return {k: torch.stack(v, dim=0) for k, v in self.rows.items()}

    def score(self, block: BlockKey, error: torch.Tensor) -> float:
        """``e' G e`` estimated directionally as ``mean_p (u_p . e)^2``."""
        U = torch.stack(self.rows[block], dim=0)
        e = error.reshape(-1).to(U.dtype).to(U.device)
        return float((U @ e).pow(2).mean())


def probe_context(
    model,
    cache,
    cont_ids: torch.Tensor,
    prompt_len: int,
    *,
    layers: Sequence[int],
    which: Sequence[CacheKind],
    n_probes: int,
    generator: torch.Generator,
    probe_fn=fisher_probe,
    fixed_probes: Optional[Sequence[torch.Tensor]] = None,
):
    """VJPs for one context.

    Returns ``(per_probe_vjps, logits_detached)``.  ``fixed_probes`` supplies
    caller-held probe directions, which is how the signed mean-Jacobian control
    reuses one direction across contexts.
    """
    work = clone_cache(cache)
    leaves = graft_leaves(work, layers, which)
    with torch.enable_grad():
        logits = continuation_logits(model, work, cont_ids, prompt_len)
        out = []
        for p in range(n_probes):
            z = (
                fixed_probes[p].to(logits.device, logits.dtype)
                if fixed_probes is not None
                else probe_fn(logits, generator)
            )
            out.append(vjp_from_probe(logits, leaves, z, retain=(p < n_probes - 1)))
    return out, logits.detach()


# ---------------------------------------------------------------------------
# the cheap incumbent metrics
# ---------------------------------------------------------------------------


def score_mse(error: torch.Tensor) -> float:
    """Raw cache reconstruction error.  The baseline every metric must beat."""
    return float(error.detach().float().pow(2).mean())


def score_frob(error: torch.Tensor) -> float:
    return float(error.detach().float().pow(2).sum())


def score_wo_metric(error: torch.Tensor, w_o: torch.Tensor, head_dim: int) -> float:
    """Per-head output-projection metric: ``|| W_O^h e_h ||^2`` summed over heads.

    This is the closest cheap precursor already tested in this program, where it
    lost to plain SVD at ranks 32-64 and tied it at 96.  It is myopic by
    construction: it sees one layer's output projection and nothing downstream.

    ``error`` is ``[B, H, T, D]``; ``w_o`` is the layer's ``[hidden, H*D]``
    output projection weight.
    """
    e = error.detach().float()
    b, h, t, d = e.shape
    # [B,H,T,D] -> [B,T,H*D] to match the o_proj input layout
    flat = e.permute(0, 2, 1, 3).reshape(b, t, h * d)
    proj = flat @ w_o.detach().float().transpose(0, 1)
    return float(proj.pow(2).mean())


def score_attention_local(
    error: torch.Tensor,
    attn_probs: torch.Tensor,
    kind: CacheKind,
    power: float = 2.0,
) -> float:
    """CacheBridge-style attention-local sensitivity.

    Weight each cached position by how much attention the receiver actually
    pays to it.  ``attn_probs`` is ``[B, H, Tq, Tkv]``; it is reduced over
    queries to a per-position mass ``a`` and applied as ``sum_t a_t^p ||e_t||^2``.

    The default ``power=2`` is the principled one for values: the attention
    output error is ``sum_t a_t e_t``, so for errors that are uncorrelated
    across positions its expected square is ``sum_t a_t^2 ||e_t||^2``.  Since
    published weightings differ, ``power=1`` is also scored, so a verdict never
    rests on having picked the baseline's exponent favourably.

    This is local to one attention block and carries no downstream information,
    which is exactly the gap the receiver metric claims to close.
    """
    e = error.detach().float()
    a = attn_probs.detach().float().mean(dim=2)  # [B, H, Tkv]
    if a.shape[1] != e.shape[1]:  # GQA: fold query heads into their kv group
        groups = a.shape[1] // e.shape[1]
        a = a.reshape(a.shape[0], e.shape[1], groups, a.shape[-1]).mean(dim=2)
    a = a[..., : e.shape[2]]
    per_pos = e.pow(2).sum(dim=-1)  # [B, H, T]
    return float((a.pow(power) * per_pos).sum())


@contextlib.contextmanager
def no_grad_if(flag: bool):
    if flag:
        with torch.no_grad():
            yield
    else:
        yield
