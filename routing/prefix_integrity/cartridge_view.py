# SPDX-License-Identifier: GPL-2.0
"""Read a Cartridge as a prefix object: identity, block geometry, tensors.

A knlp cartridge is `torch.save(list[(k, v)])`, one (K, V) tuple per layer, with
each tensor shaped `(1, n_heads, seq_len, head_dim)` (see
tools/skillsbench_cartridges/train_cartridge.py). A sibling `meta.json` carries
model, dtype, n_layers, budget_tokens and `prefix_token_ids_sha256`, and
`prefix_token_ids.json` holds the token ids.

PIA only needs the geometry (block_size over seq_len) and a stable prefix hash
to build manifests and run the metric/invariant path. Loading the 200 MB+ of
tensors is reserved for the codec and semantic-drift modes, so the metadata path
imports no torch and runs on any CPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Optional

from .datatypes import PrefixIdentity

_DTYPE_BYTES = {
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float32": 4,
    "fp32": 4,
    "float": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
}


def dtype_bytes(dtype: str) -> int:
    return _DTYPE_BYTES.get(str(dtype).lower().replace("torch.", ""), 2)


def _resolve(path: str):
    """Return (cartridge_pt_path, meta_path_or_None, prefix_ids_path_or_None)."""
    if os.path.isdir(path):
        pt = os.path.join(path, "cartridge.pt")
        if not os.path.exists(pt):
            cands = [f for f in os.listdir(path) if f.endswith(".pt")]
            if not cands:
                raise FileNotFoundError(f"no .pt in cartridge dir {path}")
            pt = os.path.join(path, sorted(cands)[0])
        meta = os.path.join(path, "meta.json")
        ids = os.path.join(path, "prefix_token_ids.json")
        return (
            pt,
            (meta if os.path.exists(meta) else None),
            (ids if os.path.exists(ids) else None),
        )
    # bare .pt file: look for siblings by stem
    d = os.path.dirname(path)
    meta = os.path.join(d, "meta.json")
    ids = os.path.join(d, "prefix_token_ids.json")
    return (
        path,
        (meta if os.path.exists(meta) else None),
        (ids if os.path.exists(ids) else None),
    )


def _prefix_hash(meta: Optional[dict], ids_path: Optional[str], pt_path: str) -> str:
    """Stable identity for the logical prefix. Prefer the recorded token-id
    digest; else hash the token-id file; else fall back to a digest of the pt
    filename + size (better than nothing, but not content-true).
    """
    if meta and meta.get("prefix_token_ids_sha256"):
        return meta["prefix_token_ids_sha256"]
    if ids_path and os.path.exists(ids_path):
        with open(ids_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    h = hashlib.sha256()
    h.update(os.path.basename(pt_path).encode())
    try:
        h.update(str(os.path.getsize(pt_path)).encode())
    except OSError:
        pass
    return h.hexdigest()


def _rope_config_hash(meta: Optional[dict]) -> Optional[str]:
    if not meta:
        return None
    keys = ("rope_theta", "rope_scaling", "rope_config", "position_offset")
    sub = {k: meta[k] for k in keys if k in meta}
    if not sub:
        return None
    return hashlib.sha256(json.dumps(sub, sort_keys=True).encode()).hexdigest()[:16]


def load_identity(path: str, block_size: int = 16) -> PrefixIdentity:
    """Build PrefixIdentity from meta.json when present (no torch), else by
    inspecting the saved tensors.
    """
    pt, meta_path, ids_path = _resolve(path)
    meta = None
    if meta_path:
        with open(meta_path) as f:
            meta = json.load(f)

    cartridge_id = os.path.basename(os.path.dirname(pt)) or os.path.basename(pt)
    prefix_hash = _prefix_hash(meta, ids_path, pt)

    if meta and "budget_tokens" in meta and "n_layers" in meta:
        seq_len = int(meta["budget_tokens"])
        n_layers = int(meta["n_layers"])
        dtype = str(meta.get("dtype", "float16"))
        n_heads = meta.get("n_heads")
        head_dim = meta.get("head_dim")
        kv_shape = (
            (n_layers, 1, n_heads, seq_len, head_dim)
            if n_heads
            else (n_layers, 1, None, seq_len, None)
        )
        model_id = str(meta.get("model", "unknown"))
    else:
        # Fall back to reading the tensors.
        import torch  # lazy

        kv = torch.load(pt, map_location="cpu", weights_only=False)
        k0 = kv[0][0]
        n_layers = len(kv)
        _, n_heads, seq_len, head_dim = k0.shape
        dtype = str(k0.dtype).replace("torch.", "")
        kv_shape = (n_layers, 1, int(n_heads), int(seq_len), int(head_dim))
        model_id = str(meta.get("model", "unknown")) if meta else "unknown"

    num_blocks = math.ceil(seq_len / block_size) if seq_len else 0

    return PrefixIdentity(
        model_id=model_id,
        cartridge_id=cartridge_id,
        prefix_hash=prefix_hash,
        block_size=block_size,
        num_blocks=num_blocks,
        dtype=dtype,
        kv_shape=tuple(kv_shape),
        model_revision=(meta.get("model_revision") if meta else None),
        tokenizer_id=(meta.get("tokenizer_id") if meta else None),
        rope_config_hash=_rope_config_hash(meta),
    )


def load_tensors(path: str):
    """Load the full KV: list[(k, v)] per layer. Requires torch."""
    import torch  # lazy

    pt, _, _ = _resolve(path)
    return torch.load(pt, map_location="cpu", weights_only=False)


def kv_bytes(identity: PrefixIdentity) -> int:
    """Total cache bytes implied by the identity (for storage estimates)."""
    elems = 2  # K and V
    for d in identity.kv_shape:
        if d is None:
            return 0
        elems *= int(d)
    return elems * dtype_bytes(identity.dtype)
