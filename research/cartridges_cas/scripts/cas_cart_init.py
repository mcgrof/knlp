#!/usr/bin/env python3
"""Cartridge file helpers and path-based KV initializers for the CAS
(Cartridges at Scale) line.

The cartridges library saves a cartridge as a dict with four lists (one
tensor per layer, shape [1, kv_heads, tokens, head_dim]):
``trainable_keys``, ``trainable_values``, ``frozen_keys``,
``frozen_values``. Isolated cartridges carry a one-token frozen attention
sink in the frozen lists. The library's own ``TrainableCache.from_pretrained``
mis-reads the frozen token count (it takes the head axis), and its only
"pretrained" initializer pulls from Weights & Biases, so this module
provides the two things training-from-a-file needs:

``load_cart`` / ``save_cart``
    Read and write that dict as plain bf16 tensors with the frozen count
    recovered from the token axis.
``KVFromCartFile``
    A ``KVCacheFactory`` that initializes training from a cartridge file
    (a saved checkpoint, or an init written by the meta-initialization
    tools), preserving the frozen/trainable split.
``KVFromTextSaved``
    The library's ``KVFromText`` (first-p document tokens) with the option
    to write the step-0 cartridge to disk, so a training run's exact
    starting point is recoverable for delta analysis.

Import with the cartridges tree on ``PYTHONPATH`` and this directory on
``PYTHONPATH`` as well (pydrantic resolves the factory by module name).
"""

from typing import Optional

import torch

from cartridges.cache import AttnConfig, KVCacheFactory, TrainableCache
from cartridges.initialization import KVFromText


def _as_bf16(p):
    return (
        torch.as_tensor(p.data if hasattr(p, "data") else p).detach().to(torch.bfloat16)
    )


def load_cart(path, device="cpu"):
    """Return (keys, values, num_frozen_tokens).

    ``keys``/``values`` are per-layer bf16 tensors [1, H, F+T, D] with the
    frozen tokens (if any) first, i.e. the full init tensors that
    ``TrainableCache`` expects. ``num_frozen_tokens`` is read off the
    token axis of the frozen tensors (0 when none are stored)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    tk = [_as_bf16(p) for p in ck["trainable_keys"]]
    tv = [_as_bf16(p) for p in ck["trainable_values"]]
    fk = ck.get("frozen_keys") or []
    fv = ck.get("frozen_values") or []
    if len(fk):
        fk = [_as_bf16(p) for p in fk]
        fv = [_as_bf16(p) for p in fv]
        assert len(fk) == len(tk) == len(fv) == len(tv)
        nfrozen = fk[0].shape[2]
        keys = [torch.cat([fk[i], tk[i]], dim=2).contiguous() for i in range(len(tk))]
        vals = [torch.cat([fv[i], tv[i]], dim=2).contiguous() for i in range(len(tv))]
    else:
        nfrozen = 0
        keys, vals = tk, tv
    if device != "cpu":
        keys = [k.to(device) for k in keys]
        vals = [v.to(device) for v in vals]
    return keys, vals, nfrozen


def split_cart(keys, values, nfrozen):
    """Inverse of the concat in ``load_cart``: the dict layout the
    library saves and ``load_cart`` reads."""
    return {
        "trainable_keys": [k[:, :, nfrozen:].contiguous() for k in keys],
        "trainable_values": [v[:, :, nfrozen:].contiguous() for v in values],
        "frozen_keys": (
            [k[:, :, :nfrozen].contiguous() for k in keys] if nfrozen else []
        ),
        "frozen_values": (
            [v[:, :, :nfrozen].contiguous() for v in values] if nfrozen else []
        ),
    }


def save_cart(keys, values, nfrozen, path):
    torch.save(split_cart(keys, values, nfrozen), path)


def cart_shape(path):
    """(n_layers, kv_heads, trainable_tokens, head_dim, num_frozen_tokens)."""
    keys, _, nfrozen = load_cart(path)
    n_layers = len(keys)
    _, h, t, d = keys[0].shape
    return n_layers, h, t - nfrozen, d, nfrozen


class KVFromCartFile(KVCacheFactory):
    """Initialize the trainable cache from a cartridge file on disk."""

    class Config(KVCacheFactory.Config):
        path: str

    def initialize_kv_cache(
        self,
        tokenizer=None,
        model=None,
        attn_config: Optional[AttnConfig] = None,
    ) -> TrainableCache:
        device = model.device if model is not None else "cuda"
        keys, values, nfrozen = load_cart(self.config.path, device=device)
        assert len(keys) == attn_config.n_layers, (
            f"{self.config.path}: {len(keys)} layers, model has "
            f"{attn_config.n_layers}"
        )
        assert keys[0].shape[1] == attn_config.n_heads
        assert keys[0].shape[3] == attn_config.head_dim
        print(
            f"[cart-init] {self.config.path}: trainable={keys[0].shape[2] - nfrozen} "
            f"frozen={nfrozen} layers={len(keys)}",
            flush=True,
        )
        return TrainableCache(
            config=attn_config,
            init_keys=keys,
            init_values=values,
            num_frozen_tokens=nfrozen,
        )


class KVFromTextSaved(KVFromText):
    """``KVFromText`` that can also persist the step-0 cartridge."""

    class Config(KVFromText.Config):
        save_path: Optional[str] = None

    def initialize_kv_cache(self, tokenizer, model, attn_config: AttnConfig):
        cache = super().initialize_kv_cache(tokenizer, model, attn_config)
        if self.config.save_path:
            cache.save(self.config.save_path)
            print(
                f"[cart-init] wrote step-0 cartridge to {self.config.save_path}",
                flush=True,
            )
        return cache
