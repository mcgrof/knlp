# SPDX-License-Identifier: MIT
"""Bridge between native trained-cartridge files and the prefix artifact.

A native cartridge (the HazyResearch ``cartridges`` library format) is a
dictionary of four per-layer lists, each tensor shaped ``[1, kv_heads,
tokens, head_dim]`` in BF16: ``frozen_keys``/``frozen_values`` hold the
attention-sink rows and ``trainable_keys``/``trainable_values`` the learned
rows. The library installs the sink first, so a cartridge with one frozen
row and ``T`` trainable rows occupies logical positions ``0 .. T`` and the
first reader token sits at position ``T + 1``. Keys are stored after rotary
embedding at those positions.

This module reads that layout into the reader's prefix form, derives the
shared sink from the base model (it is the key/value state of the chat
template's first token at position 0, which no document changes), and
moves generated rows between position frames by removing and re-applying
the model's own rotary embedding in FP32.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch

from . import model as M

Prefix = list[tuple[torch.Tensor, torch.Tensor]]

# ``<|im_start|>``: the first token of every Qwen chat template rendering
# and therefore the token whose position-0 state the native sink holds.
QWEN_IM_START = 151644


@dataclasses.dataclass
class NativeCartridge:
    path: str
    file_digest: str
    layers: int
    kv_heads: int
    head_dim: int
    frozen_rows: int
    trainable_rows: int
    keys: list[torch.Tensor]  # per layer [1, H, F + T, D], sink first
    values: list[torch.Tensor]

    @property
    def rows(self) -> int:
        return self.frozen_rows + self.trainable_rows

    @property
    def positions(self) -> range:
        return range(0, self.rows)

    @property
    def reader_start(self) -> int:
        return self.rows

    def prefix(self, device: torch.device | str) -> Prefix:
        return [
            (k.to(device).contiguous(), v.to(device).contiguous())
            for k, v in zip(self.keys, self.values)
        ]


def _as_tensor(value: Any) -> torch.Tensor:
    tensor = value.data if hasattr(value, "data") else value
    return torch.as_tensor(tensor).detach()


def load_native_cartridge(
    path: Path, dtype: torch.dtype = torch.bfloat16
) -> NativeCartridge:
    """Read a native cartridge file into sink-first per-layer tensors."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("trainable_keys", "trainable_values"):
        if key not in raw:
            raise ValueError(f"{path}: missing {key}")
    trainable_keys = [_as_tensor(t).to(dtype) for t in raw["trainable_keys"]]
    trainable_values = [_as_tensor(t).to(dtype) for t in raw["trainable_values"]]
    frozen_keys = [_as_tensor(t).to(dtype) for t in (raw.get("frozen_keys") or [])]
    frozen_values = [_as_tensor(t).to(dtype) for t in (raw.get("frozen_values") or [])]
    layers = len(trainable_keys)
    if len(trainable_values) != layers:
        raise ValueError("key and value layer counts differ")
    if frozen_keys and (len(frozen_keys) != layers or len(frozen_values) != layers):
        raise ValueError("frozen rows must be present for every layer")
    shape = tuple(trainable_keys[0].shape)
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError(f"unexpected native tensor shape {shape}")
    _, kv_heads, trainable_rows, head_dim = shape
    frozen_rows = int(frozen_keys[0].shape[2]) if frozen_keys else 0
    keys, values = [], []
    for layer in range(layers):
        for tensor in (trainable_keys[layer], trainable_values[layer]):
            if tuple(tensor.shape) != shape:
                raise ValueError("trainable tensors differ in shape across layers")
        if frozen_keys:
            for tensor in (frozen_keys[layer], frozen_values[layer]):
                if tuple(tensor.shape) != (1, kv_heads, frozen_rows, head_dim):
                    raise ValueError("frozen tensors differ in shape across layers")
            keys.append(torch.cat([frozen_keys[layer], trainable_keys[layer]], dim=2))
            values.append(
                torch.cat([frozen_values[layer], trainable_values[layer]], dim=2)
            )
        else:
            keys.append(trainable_keys[layer])
            values.append(trainable_values[layer])
    return NativeCartridge(
        path=str(path),
        file_digest=M.sha256_file(Path(path)),
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        frozen_rows=frozen_rows,
        trainable_rows=trainable_rows,
        keys=[k.contiguous() for k in keys],
        values=[v.contiguous() for v in values],
    )


def save_native_cartridge(
    path: Path, prefix: Prefix, frozen_rows: int, index: int = 0
) -> str:
    """Write reader prefix rows (sink first) in the native file layout so
    the native evaluator can consume a generated cartridge."""
    payload = {
        "trainable_keys": [
            k[index : index + 1, :, frozen_rows:].detach().cpu().contiguous()
            for k, _ in prefix
        ],
        "trainable_values": [
            v[index : index + 1, :, frozen_rows:].detach().cpu().contiguous()
            for _, v in prefix
        ],
        "frozen_keys": (
            [
                k[index : index + 1, :, :frozen_rows].detach().cpu().contiguous()
                for k, _ in prefix
            ]
            if frozen_rows
            else []
        ),
        "frozen_values": (
            [
                v[index : index + 1, :, :frozen_rows].detach().cpu().contiguous()
                for _, v in prefix
            ]
            if frozen_rows
            else []
        ),
    }
    torch.save(payload, path)
    return M.sha256_file(Path(path))


@torch.no_grad()
def sink_rows(reader_model: torch.nn.Module, token_id: int = QWEN_IM_START) -> Prefix:
    """Key/value rows of ``token_id`` at logical position 0 in every layer:
    the native attention sink. Causal attention makes the first position's
    state independent of anything that follows it, so this is the same for
    every document."""
    device = next(reader_model.parameters()).device
    ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
    positions = torch.zeros_like(ids)
    out = reader_model.model(
        input_ids=ids,
        position_ids=positions,
        attention_mask=torch.ones_like(ids),
        use_cache=True,
    )
    cache = out.past_key_values
    layers = M.Geometry.from_config(reader_model.config).layers
    return [
        tuple(t.clone() for t in M.cache_layer(cache, layer)) for layer in range(layers)
    ]


def prefix_equal(a: Prefix, b: Prefix) -> bool:
    return len(a) == len(b) and all(
        torch.equal(ka.cpu(), kb.cpu()) and torch.equal(va.cpu(), vb.cpu())
        for (ka, va), (kb, vb) in zip(a, b)
    )


def prefix_max_abs_diff(a: Prefix, b: Prefix) -> float:
    worst = 0.0
    for (ka, va), (kb, vb) in zip(a, b):
        worst = max(
            worst,
            float((ka.float().cpu() - kb.float().cpu()).abs().max()),
            float((va.float().cpu() - vb.float().cpu()).abs().max()),
        )
    return worst


def rotary_tables(
    reader_model: torch.nn.Module, positions: Sequence[int] | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``cos``/``sin`` tables ``[1, P, D]`` in FP32 from the model's own
    rotary module, so scaling, theta and layout are the pinned model's."""
    rotary = reader_model.model.rotary_emb
    device = next(reader_model.parameters()).device
    pos = torch.as_tensor(positions, dtype=torch.long, device=device).view(1, -1)
    probe = torch.zeros(1, dtype=torch.float32, device=device)
    cos, sin = rotary(probe, pos)
    return cos.float(), sin.float()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rotate_keys(
    keys: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply (or remove, with ``inverse``) the rotary embedding to keys
    shaped ``[B, H, P, D]`` with tables ``[1, P, D]``; computed in FP32."""
    x = keys.float()
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    if inverse:
        s = -s
    return x * c + _rotate_half(x) * s


def rerotate_keys(
    reader_model: torch.nn.Module,
    keys: torch.Tensor,
    from_positions: Sequence[int],
    to_positions: Sequence[int],
) -> torch.Tensor:
    """Move stored keys from the frame of ``from_positions`` to the frame of
    ``to_positions``: ``R(to) R(-from) k`` per row in FP32. The result is
    returned in FP32; callers cast once to the artifact dtype."""
    if keys.shape[2] != len(from_positions) or keys.shape[2] != len(to_positions):
        raise ValueError("one position per key row is required")
    cos_from, sin_from = rotary_tables(reader_model, from_positions)
    cos_to, sin_to = rotary_tables(reader_model, to_positions)
    unrotated = rotate_keys(keys, cos_from, sin_from, inverse=True)
    return rotate_keys(unrotated, cos_to, sin_to)


def generated_to_native_frame(
    reader_model: torch.nn.Module,
    generated: Prefix,
    source_length: int,
    sink: Prefix,
    dtype: torch.dtype,
) -> Prefix:
    """Place generated rows into the native cartridge frame.

    Generated row ``j`` was produced at logical position ``L + j`` right after
    the source; in the native frame it occupies position ``1 + j`` behind the
    shared sink at position 0. Keys are re-rotated in FP32 and cast once;
    values do not depend on position and are copied.
    """
    if not generated:
        raise ValueError("no generated rows")
    rows = int(generated[0][0].shape[2])
    from_positions = list(range(source_length, source_length + rows))
    to_positions = list(range(1, 1 + rows))
    out: Prefix = []
    for layer, ((k, v), (sk, sv)) in enumerate(zip(generated, sink)):
        if sk.shape[2] != 1:
            raise ValueError("the sink must be exactly one row")
        moved = rerotate_keys(reader_model, k, from_positions, to_positions).to(dtype)
        batch = k.shape[0]
        sink_k = sk.to(dtype).expand(batch, -1, -1, -1)
        sink_v = sv.to(dtype).expand(batch, -1, -1, -1)
        out.append(
            (
                torch.cat([sink_k, moved], dim=2).contiguous(),
                torch.cat([sink_v, v.to(dtype)], dim=2).contiguous(),
            )
        )
    return out


def native_frame_positions(rows: int) -> tuple[range, int]:
    """Stored positions and reader start for a native-frame cartridge."""
    return range(0, rows), rows


def digest_native(cart: NativeCartridge) -> str:
    """Digest over the sink-first tensors as the reader would install them."""
    digest = hashlib.sha256()
    for k, v in zip(cart.keys, cart.values):
        for tensor in (k, v):
            digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
