#!/usr/bin/env python3
"""Synthetic overwrite/correction diagnostic for Trellis-style memories.

The task is latest-value retrieval under overwrites:

    SET key value ... SET key new_value ... QUERY key answer

Only the final answer token is supervised. Metrics separate correct latest-value
answers from stale-value interference and other-key confusion.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROLE_FILLER = 0
ROLE_SET = 1
ROLE_SET_KEY = 2
ROLE_SET_VALUE = 3
ROLE_QUERY = 4
ROLE_QUERY_KEY = 5
ROLE_ANSWER = 6
ROLE_NAMES = (
    "filler",
    "set",
    "set_key",
    "set_value",
    "query",
    "query_key",
    "answer",
)


@dataclass(frozen=True)
class Cell:
    seq_len: int
    n_unique: int
    overwrites: int
    latest_gap: int


@dataclass
class BatchMeta:
    query_pos: list[int]
    answers: list[int]
    stale_values: list[set[int]]
    other_values: list[set[int]]
    latest_distance: list[int]
    roles: Any | None = None
    key_ids: Any | None = None


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def build_cells(args: argparse.Namespace) -> list[Cell]:
    cells: list[Cell] = []
    for seq_len in parse_ints(args.contexts):
        for n_unique in parse_ints(args.unique_keys):
            for overwrites in parse_ints(args.overwrites):
                min_tokens = 3 * (n_unique + overwrites) + 3
                if min_tokens > seq_len:
                    continue
                for gap in parse_ints(args.latest_gaps):
                    cells.append(Cell(seq_len, n_unique, overwrites, gap))
    return cells


def json_args(args: argparse.Namespace) -> dict[str, Any]:
    out = vars(args).copy()
    for key, value in list(out.items()):
        if isinstance(value, Path):
            out[key] = str(value)
    return out


def stable_row_seed(row: str) -> int:
    total = 0
    for idx, ch in enumerate(row):
        total += (idx + 1) * ord(ch)
    return total


def row_base_and_layer_mode(row: str) -> tuple[str, str]:
    suffixes = {
        "_layer0": "layer0",
        "_lowerhalf": "lower_half",
        "_upperhalf": "upper_half",
        "_notlayer0": "not_layer0",
    }
    for suffix, layer_mode in suffixes.items():
        if row.endswith(suffix):
            return row[: -len(suffix)], layer_mode
    return row, "all"


def row_base_layer_context(row: str) -> tuple[str, str, str]:
    row_base, layer_mode = row_base_and_layer_mode(row)
    context_suffixes = {
        "_prevctx": "current_prev",
        "_prevonly": "prev",
    }
    for suffix, context_mode in context_suffixes.items():
        if row_base.endswith(suffix):
            return row_base[: -len(suffix)], layer_mode, context_mode
    return row_base, layer_mode, "current"


def vocab_size(args: argparse.Namespace) -> int:
    return args.n_keys + args.n_vals + 2 + args.distractor_vocab


def token_ids(args: argparse.Namespace) -> dict[str, int]:
    return {
        "set": args.n_keys + args.n_vals,
        "query": args.n_keys + args.n_vals + 1,
        "filler0": args.n_keys + args.n_vals + 2,
    }


def _rand_value(rng: random.Random, args: argparse.Namespace, avoid: set[int]) -> int:
    base = args.n_keys
    for _ in range(1024):
        val = base + rng.randrange(args.n_vals)
        if val not in avoid:
            return val
    return base + rng.randrange(args.n_vals)


def _filler(rng: random.Random, args: argparse.Namespace) -> int:
    ids = token_ids(args)
    return ids["filler0"] + rng.randrange(args.distractor_vocab)


def _distribute_fillers(
    rng: random.Random,
    total: int,
    slots: int,
) -> list[int]:
    if slots <= 0:
        return []
    out = [0] * slots
    for _ in range(total):
        out[rng.randrange(slots)] += 1
    return out


def _append_filler(
    seq: list[int],
    roles: list[int],
    key_ids: list[int],
    args: argparse.Namespace,
    rng: random.Random,
) -> None:
    seq.append(_filler(rng, args))
    roles.append(ROLE_FILLER)
    key_ids.append(-1)


def _append_event(
    seq: list[int],
    roles: list[int],
    key_ids: list[int],
    event: tuple[int, int, int],
) -> None:
    set_tok, key_tok, value_tok = event
    seq.extend([set_tok, key_tok, value_tok])
    roles.extend([ROLE_SET, ROLE_SET_KEY, ROLE_SET_VALUE])
    key_ids.extend([-1, key_tok, key_tok])


def make_example(
    cell: Cell,
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[int], dict[str, Any]]:
    ids = token_ids(args)
    keys = rng.sample(range(args.n_keys), cell.n_unique)
    qkey = rng.choice(keys)
    others = [key for key in keys if key != qkey]
    used_values: set[int] = set()
    stale_values: set[int] = set()
    other_values: set[int] = set()

    prefix_events: list[tuple[int, int, int]] = []
    for _ in range(cell.overwrites):
        value = _rand_value(rng, args, used_values)
        used_values.add(value)
        stale_values.add(value)
        prefix_events.append((ids["set"], qkey, value))

    for key in others:
        value = _rand_value(rng, args, used_values)
        used_values.add(value)
        other_values.add(value)
        prefix_events.append((ids["set"], key, value))

    rng.shuffle(prefix_events)
    answer = _rand_value(rng, args, used_values)
    latest_event = (ids["set"], qkey, answer)

    filler_after = max(0, cell.latest_gap - 2)
    min_len = 3 * len(prefix_events) + 3 + filler_after + 3
    if min_len > cell.seq_len:
        filler_after = max(0, cell.seq_len - (3 * len(prefix_events) + 6))
    seq: list[int] = []
    roles: list[int] = []
    key_ids: list[int] = []
    pre_extra = cell.seq_len - (3 * len(prefix_events) + 3 + filler_after + 3)
    fills = _distribute_fillers(rng, max(0, pre_extra), len(prefix_events) + 1)
    for event, n_fill in zip(prefix_events, fills):
        for _ in range(n_fill):
            _append_filler(seq, roles, key_ids, args, rng)
        _append_event(seq, roles, key_ids, event)
    if fills:
        for _ in range(fills[-1]):
            _append_filler(seq, roles, key_ids, args, rng)
    _append_event(seq, roles, key_ids, latest_event)
    latest_value_pos = len(seq) - 1
    for _ in range(filler_after):
        _append_filler(seq, roles, key_ids, args, rng)
    seq.extend([ids["query"], qkey, answer])
    roles.extend([ROLE_QUERY, ROLE_QUERY_KEY, ROLE_ANSWER])
    key_ids.extend([-1, qkey, qkey])
    if len(seq) < cell.seq_len:
        pad_n = cell.seq_len - len(seq)
        pad_seq: list[int] = []
        pad_roles: list[int] = []
        pad_key_ids: list[int] = []
        for _ in range(pad_n):
            _append_filler(pad_seq, pad_roles, pad_key_ids, args, rng)
        seq = pad_seq + seq
        roles = pad_roles + roles
        key_ids = pad_key_ids + key_ids
        latest_value_pos += pad_n
    if len(seq) != cell.seq_len or len(roles) != cell.seq_len:
        raise RuntimeError((len(seq), len(roles), cell))
    query_pos = cell.seq_len - 2
    meta = {
        "query_pos": query_pos,
        "answer": answer,
        "stale_values": stale_values,
        "other_values": other_values,
        "latest_distance": query_pos - latest_value_pos,
        "roles": roles,
        "key_ids": key_ids,
    }
    return seq, meta


def make_batch(cell: Cell, args: argparse.Namespace, rng: random.Random, device):
    import torch

    rows: list[list[int]] = []
    query_pos: list[int] = []
    answers: list[int] = []
    stale_values: list[set[int]] = []
    other_values: list[set[int]] = []
    latest_distance: list[int] = []
    role_rows: list[list[int]] = []
    key_id_rows: list[list[int]] = []
    for _ in range(args.batch):
        seq, meta = make_example(cell, args, rng)
        rows.append(seq)
        query_pos.append(int(meta["query_pos"]))
        answers.append(int(meta["answer"]))
        stale_values.append(set(meta["stale_values"]))
        other_values.append(set(meta["other_values"]))
        latest_distance.append(int(meta["latest_distance"]))
        role_rows.append(list(meta["roles"]))
        key_id_rows.append(list(meta["key_ids"]))
    idx = torch.as_tensor(rows, dtype=torch.long, device=device)
    roles = torch.as_tensor(role_rows, dtype=torch.long, device=device)
    key_ids = torch.as_tensor(key_id_rows, dtype=torch.long, device=device)
    labels = torch.full_like(idx, -100)
    for bi, pos in enumerate(query_pos):
        labels[bi, pos + 1] = idx[bi, pos + 1]
    return idx, labels, BatchMeta(
        query_pos=query_pos,
        answers=answers,
        stale_values=stale_values,
        other_values=other_values,
        latest_distance=latest_distance,
        roles=roles,
        key_ids=key_ids,
    )


def make_cfg(
    args: argparse.Namespace,
    value_readout_act: str,
    value_alpha_mode: str,
    value_alpha_mix: float,
    value_alpha_correction_init: float,
    value_alpha_correction_max: float,
    value_read_query_mode: str,
    value_read_query_gate_init: float,
    value_read_query_gate_max: float,
    update_gate_mode: str,
    update_gate_init: float,
    update_gate_target: str,
    update_gate_layer_mode: str,
    update_gate_context_mode: str,
    update_gate_floor: float,
):
    from trellis_lm.config import TrellisConfig

    return TrellisConfig(
        vocab_size=vocab_size(args),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_head=args.d_head,
        n_slots=args.n_slots,
        max_seq_len=max(parse_ints(args.contexts)),
        dtype=args.dtype,
        activation="silu",
        alpha_mode="linear",
        beta_init=0.99,
        gamma_init=0.005,
        chunk_size=args.chunk_size,
        chunk_refine=0,
        output_path="paper",
        use_short_conv_v=True,
        value_readout_act=value_readout_act,
        trellis_value_alpha_mode=value_alpha_mode,
        trellis_value_alpha_mix=value_alpha_mix,
        trellis_value_alpha_correction_init=value_alpha_correction_init,
        trellis_value_alpha_correction_max=value_alpha_correction_max,
        trellis_value_read_query_mode=value_read_query_mode,
        trellis_value_read_query_gate_init=value_read_query_gate_init,
        trellis_value_read_query_gate_max=value_read_query_gate_max,
        trellis_retention_mode="token_proj",
        trellis_update_stabilizer="layerwise_gamma",
        trellis_layer0_gamma_mult=0.5,
        update_gate_mode=update_gate_mode,
        update_gate_init=update_gate_init,
        trellis_update_gate_target=update_gate_target,
        trellis_update_gate_layer_mode=update_gate_layer_mode,
        trellis_update_gate_context_mode=update_gate_context_mode,
        trellis_update_gate_floor=update_gate_floor,
        residual_update_mix=0.10,
    )


def row_spec(
    row: str,
) -> tuple[str, str, str, float, float, float, str, float, float, str, float, str, float]:
    row, _, _ = row_base_layer_context(row)
    if row == "trellis_none":
        return (
            "trellis",
            "none",
            "shared",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "none",
            0.95,
            "both",
            0.0,
        )
    if row == "trellis_norm_silu":
        return (
            "trellis",
            "norm_silu",
            "shared",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "none",
            0.95,
            "both",
            0.0,
        )
    if row == "trellis_keyed":
        return (
            "trellis",
            "none",
            "key_readout",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "none",
            0.95,
            "both",
            0.0,
        )
    if row == "trellis_keyed_detach":
        return (
            "trellis",
            "none",
            "key_readout_detached",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "none",
            0.95,
            "both",
            0.0,
        )
    if row == "trellis_keyed_norm_silu":
        return (
            "trellis",
            "norm_silu",
            "key_readout",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "none",
            0.95,
            "both",
            0.0,
        )
    if row == "trellis_corr1e3":
        return (
            "trellis",
            "none",
            "shared_plus_key_correction",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "none",
            0.95,
            "both",
            0.0,
        )
    if row == "trellis_corr1e2":
        return (
            "trellis",
            "none",
            "shared_plus_key_correction",
            1.0,
            1e-2,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "none",
            0.95,
            "both",
            0.0,
        )
    if row == "trellis_corr_detach1e3":
        return (
            "trellis",
            "none",
            "shared_plus_key_correction_detached",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "none",
            0.95,
            "both",
            0.0,
        )
    if row == "trellis_corr_norm_silu1e3":
        return (
            "trellis",
            "norm_silu",
            "shared_plus_key_correction",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "none",
            0.95,
            "both",
            0.0,
        )
    if row == "trellis_gate_value095":
        return (
            "trellis",
            "none",
            "shared",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "scalar",
            0.95,
            "value",
            0.0,
        )
    if row == "trellis_gate_value080":
        return (
            "trellis",
            "none",
            "shared",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "trellis_gate_value080_norm_silu":
        return (
            "trellis",
            "norm_silu",
            "shared",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "trellis_gate_value080_prevalpha_corr1e3":
        return (
            "trellis",
            "none",
            "shared_plus_prev_alpha_correction",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "trellis_gate_value080_readalpha_gate005":
        return (
            "trellis",
            "none",
            "shared",
            1.0,
            1e-3,
            0.25,
            "alpha_residual_gate",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "trellis_gate_value080_prevalpha_corr1e2":
        return (
            "trellis",
            "none",
            "shared_plus_prev_alpha_correction",
            1.0,
            1e-2,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "trellis_gate_value080_prevkey_corr1e3":
        return (
            "trellis",
            "none",
            "shared_plus_prev_key_correction",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "trellis_gate_value080_prevkey_corr1e3_readalpha_gate005":
        return (
            "trellis",
            "none",
            "shared_plus_prev_key_correction",
            1.0,
            1e-3,
            0.25,
            "alpha_residual_gate",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "trellis_gate_value080_prevalpha_corr1e3_readalpha_gate005":
        return (
            "trellis",
            "none",
            "shared_plus_prev_alpha_correction",
            1.0,
            1e-3,
            0.25,
            "alpha_residual_gate",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "trellis_gate_value080_prevkey_detach_corr1e3":
        return (
            "trellis",
            "none",
            "shared_plus_prev_key_correction_detached",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "trellis_gate_value080_prevkey_detach_corr1e3_readalpha_gate005":
        return (
            "trellis",
            "none",
            "shared_plus_prev_key_correction_detached",
            1.0,
            1e-3,
            0.25,
            "alpha_residual_gate",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "trellis_gate_value080_localaddr1e3":
        return (
            "trellis",
            "none",
            "shared_plus_local_key_correction",
            1.0,
            1e-3,
            1.0,
            "local_key_address",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "trellis_gate_value080_localaddr1e2":
        return (
            "trellis",
            "none",
            "shared_plus_local_key_correction",
            1.0,
            1e-2,
            1.0,
            "local_key_address",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "trellis_gate_value080_localaddr_detach1e3":
        return (
            "trellis",
            "none",
            "shared_plus_local_key_correction_detached",
            1.0,
            1e-3,
            1.0,
            "local_key_address_detached",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "trellis_gate_value080_localaddr_norm_silu1e3":
        return (
            "trellis",
            "norm_silu",
            "shared_plus_local_key_correction",
            1.0,
            1e-3,
            1.0,
            "local_key_address",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "trellis_gate_value_floor050":
        return (
            "trellis",
            "none",
            "shared",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "scalar",
            0.95,
            "value",
            0.50,
        )
    if row == "trellis_gate_channel_value095":
        return (
            "trellis",
            "none",
            "shared",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "channel",
            0.95,
            "value",
            0.0,
        )
    if row == "trellis_role_oracle_main_update":
        return (
            "trellis_role_main",
            "none",
            "shared",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "trellis_role_oracle_separate_kv_learned_address":
        return (
            "trellis_role_kv_learned",
            "none",
            "shared",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "trellis_role_oracle_separate_kv_oracle_address":
        return (
            "trellis_role_kv_oracle",
            "none",
            "shared",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "scalar",
            0.80,
            "value",
            0.0,
        )
    if row == "gdn_ref":
        return (
            "gated_delta_ref",
            "none",
            "shared",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "none",
            0.95,
            "both",
            0.0,
        )
    if row == "dense":
        return (
            "dense",
            "none",
            "shared",
            1.0,
            1e-3,
            0.25,
            "key_readout",
            0.05,
            0.75,
            "none",
            0.95,
            "both",
            0.0,
        )
    raise ValueError(row)


def as_dtype(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def build_role_oracle_probe_model(cfg, kind: str, args: argparse.Namespace):
    """Probe-only role-oracle binding wrappers.

    These rows are deliberately scoped to this synthetic overwrite harness. They
    do not change the generic Trellis C4/default path. The "main" row only gives
    the Trellis stack role embeddings. The separate-KV rows add a small
    role-gated associative side channel after the Trellis blocks and before the
    final LM head.
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from trellis_lm.model import TrellisLM

    class RoleOracleBindingProbeLM(nn.Module):
        requires_binding_meta = True

        def __init__(self):
            super().__init__()
            self.cfg = cfg
            self.kind = kind
            self.base = TrellisLM(cfg)
            self.role_emb = nn.Embedding(len(ROLE_NAMES), cfg.d_model)
            nn.init.normal_(self.role_emb.weight, std=0.02)
            self.use_separate_kv = kind in (
                "trellis_role_kv_learned",
                "trellis_role_kv_oracle",
            )
            self.use_oracle_address = kind == "trellis_role_kv_oracle"
            if self.use_separate_kv:
                self.addr_dim = cfg.n_slots
                self.key_proj = nn.Linear(cfg.d_model, self.addr_dim, bias=False)
                self.query_proj = nn.Linear(cfg.d_model, self.addr_dim, bias=False)
                self.value_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
                self.binding_out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
                for module in (
                    self.key_proj,
                    self.query_proj,
                    self.value_proj,
                    self.binding_out,
                ):
                    nn.init.normal_(module.weight, std=0.02)
                residual_unit = max(
                    1e-8,
                    min(
                        1.0 - 1e-8,
                        float(args.binding_residual_init)
                        / float(args.binding_residual_max),
                    ),
                )
                self.binding_residual_raw = nn.Parameter(
                    torch.tensor(math.log(residual_unit / (1.0 - residual_unit)))
                )
                self.binding_eta_raw = nn.Parameter(
                    torch.tensor(math.log(math.expm1(float(args.binding_eta_init))))
                )
                oracle = torch.randn(args.n_keys, self.addr_dim)
                oracle = F.normalize(oracle, dim=-1)
                self.register_buffer("oracle_address", oracle)
            else:
                self.addr_dim = 0
                self.key_proj = None
                self.query_proj = None
                self.value_proj = None
                self.binding_out = None
                self.binding_residual_raw = None
                self.binding_eta_raw = None
                self.register_buffer("oracle_address", torch.empty(0))
            self.last_binding_diag: dict[str, Any] | None = None

        def get_num_params(self):
            return sum(p.numel() for p in self.parameters())

        def memory_state_bytes(self, batch_size: int) -> int:
            base = self.base.memory_state_bytes(batch_size)
            if not self.use_separate_kv:
                return base
            elem = 2 if self.cfg.dtype in ("bf16", "fp16") else 4
            return base + batch_size * self.addr_dim * self.cfg.d_model * elem

        def _address(
            self,
            hidden: torch.Tensor,
            key_ids: torch.Tensor,
            role: str,
        ) -> torch.Tensor:
            if self.use_oracle_address:
                safe_ids = key_ids.clamp_min(0)
                return self.oracle_address.to(hidden.dtype)[safe_ids]
            if role == "query":
                if self.query_proj is None:  # pragma: no cover
                    raise RuntimeError("missing query projection")
                return F.normalize(self.query_proj(hidden), dim=-1)
            if self.key_proj is None:  # pragma: no cover
                raise RuntimeError("missing key projection")
            return F.normalize(self.key_proj(hidden), dim=-1)

        @staticmethod
        def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-6)

        def _binding_residual(self, h: torch.Tensor, meta: BatchMeta) -> torch.Tensor:
            if meta.roles is None or meta.key_ids is None:
                raise RuntimeError("role-oracle binding row requires BatchMeta roles")
            if self.value_proj is None or self.binding_out is None:  # pragma: no cover
                raise RuntimeError("separate-KV row missing projections")
            roles = meta.roles
            key_ids = meta.key_ids
            B, T, D = h.shape
            residual = h.new_zeros(B, T, D)
            scale = float(args.binding_residual_max) * torch.sigmoid(
                self.binding_residual_raw
            ).to(h.dtype)
            eta = F.softplus(self.binding_eta_raw).to(h.dtype)
            write_counts = []
            read_counts = []
            align_correct = 0
            align_total = 0
            read_target_cos = []
            residual_norm = []
            update_norm = []
            max_collision = []
            for bi in range(B):
                memory = h.new_zeros(self.addr_dim, D)
                current_addr = None
                addr_by_key: dict[int, torch.Tensor] = {}
                value_by_key: dict[int, torch.Tensor] = {}
                writes = 0
                reads = 0
                event_pos = torch.nonzero(
                    (roles[bi] == ROLE_SET_KEY)
                    | (roles[bi] == ROLE_SET_VALUE)
                    | (roles[bi] == ROLE_QUERY_KEY),
                    as_tuple=False,
                ).flatten()
                for pos_t in event_pos.tolist():
                    role_id = int(roles[bi, pos_t].item())
                    key_id = int(key_ids[bi, pos_t].item())
                    if role_id == ROLE_SET_KEY:
                        current_addr = self._address(
                            h[bi, pos_t],
                            key_ids[bi, pos_t],
                            role="key",
                        )
                        addr_by_key[key_id] = current_addr
                    elif role_id == ROLE_SET_VALUE:
                        if current_addr is None:
                            continue
                        value = self.value_proj(h[bi, pos_t])
                        read_prev = current_addr @ memory
                        delta = eta * current_addr[:, None] * (value - read_prev)[
                            None, :
                        ]
                        memory = memory + delta
                        value_by_key[key_id] = value
                        update_norm.append(float(delta.detach().float().norm().item()))
                        writes += 1
                    elif role_id == ROLE_QUERY_KEY:
                        query_addr = self._address(
                            h[bi, pos_t],
                            key_ids[bi, pos_t],
                            role="query",
                        )
                        read = query_addr @ memory
                        out = scale * self.binding_out(read)
                        residual[bi, pos_t] = out
                        residual_norm.append(float(out.detach().float().norm().item()))
                        reads += 1
                        if addr_by_key:
                            with torch.no_grad():
                                keys = list(addr_by_key)
                                addr_stack = torch.stack(
                                    [addr_by_key[k].detach() for k in keys]
                                )
                                sims = addr_stack.float() @ query_addr.detach().float()
                                pred_key = keys[int(sims.argmax().item())]
                                align_correct += int(pred_key == key_id)
                                align_total += 1
                                if addr_stack.shape[0] > 1:
                                    sim_mat = addr_stack.float() @ addr_stack.float().T
                                    sim_mat = sim_mat - torch.eye(
                                        sim_mat.shape[0],
                                        device=sim_mat.device,
                                        dtype=sim_mat.dtype,
                                    )
                                    max_collision.append(
                                        float(sim_mat.max().detach().item())
                                    )
                        target_value = value_by_key.get(key_id)
                        if target_value is not None:
                            with torch.no_grad():
                                read_target_cos.append(
                                    float(
                                        self._cos(
                                            read.detach().unsqueeze(0),
                                            target_value.detach().unsqueeze(0),
                                        )[0].item()
                                    )
                                )
                write_counts.append(writes)
                read_counts.append(reads)
            self.last_binding_diag = {
                "layer": -1,
                "backend": "role_oracle_binding_probe",
                "binding_kind": self.kind,
                "address_mode": "oracle" if self.use_oracle_address else "learned",
                "residual_scale": float(scale.detach().float().item()),
                "eta": float(eta.detach().float().item()),
                "write_count_mean": sum(write_counts) / max(1, len(write_counts)),
                "read_count_mean": sum(read_counts) / max(1, len(read_counts)),
                "alignment_accuracy": (
                    align_correct / align_total if align_total else None
                ),
                "read_target_cos_mean": (
                    sum(read_target_cos) / len(read_target_cos)
                    if read_target_cos
                    else None
                ),
                "residual_norm_mean": (
                    sum(residual_norm) / len(residual_norm) if residual_norm else 0.0
                ),
                "update_norm_mean": (
                    sum(update_norm) / len(update_norm) if update_norm else 0.0
                ),
                "address_collision_max_mean": (
                    sum(max_collision) / len(max_collision) if max_collision else None
                ),
            }
            return residual

        def forward(
            self,
            idx,
            labels=None,
            training=None,
            batch_meta: BatchMeta | None = None,
        ):
            if training is None:
                training = self.training
            if batch_meta is None or batch_meta.roles is None:
                raise RuntimeError("role-oracle binding row requires BatchMeta")
            x = self.base.wte(idx) + self.role_emb(batch_meta.roles)
            for blk in self.base.blocks:
                x = blk(x, training=training)
            if self.use_separate_kv:
                x = x + self._binding_residual(x, batch_meta)
            else:
                role_counts = {
                    ROLE_NAMES[i]: int((batch_meta.roles == i).sum().item())
                    for i in range(len(ROLE_NAMES))
                }
                self.last_binding_diag = {
                    "layer": -1,
                    "backend": "role_oracle_main_update_probe",
                    "binding_kind": self.kind,
                    "role_counts": role_counts,
                }
            x = self.base.norm_f(x)
            logits = self.base.lm_head(x)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            return logits, loss

    return RoleOracleBindingProbeLM()


def forward_model(model, idx, labels=None, training=None, meta: BatchMeta | None = None):
    if getattr(model, "requires_binding_meta", False):
        return model(idx, labels=labels, training=training, batch_meta=meta)
    return model(idx, labels=labels, training=training)


def accuracy_metrics(preds: list[int], meta: BatchMeta, args: argparse.Namespace) -> dict[str, int]:
    correct = stale = other = non_value = 0
    value_lo = args.n_keys
    value_hi = args.n_keys + args.n_vals
    for pred, answer, stale_values, other_values in zip(
        preds,
        meta.answers,
        meta.stale_values,
        meta.other_values,
    ):
        if pred == answer:
            correct += 1
        elif pred in stale_values:
            stale += 1
        elif pred in other_values:
            other += 1
        elif not (value_lo <= pred < value_hi):
            non_value += 1
    total = len(preds)
    wrong = total - correct
    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "stale_value": stale,
        "other_key_value": other,
        "non_value": non_value,
    }


def add_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + int(value)


def finalize_counts(counts: dict[str, int]) -> dict[str, Any]:
    total = max(1, counts.get("total", 0))
    out: dict[str, Any] = dict(counts)
    out["latest_value_accuracy"] = counts.get("correct", 0) / total
    out["stale_value_error_rate"] = counts.get("stale_value", 0) / total
    out["key_confusion_error_rate"] = counts.get("other_key_value", 0) / total
    out["non_value_error_rate"] = counts.get("non_value", 0) / total
    return out


def collect_trellis_diagnostics(model) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for module in model.modules():
        diag = getattr(module, "last_trellis_diag", None)
        if diag is not None:
            out.append(diag)
    binding_diag = getattr(model, "last_binding_diag", None)
    if binding_diag is not None:
        out.append(binding_diag)
    out.sort(key=lambda item: int(item.get("layer", 0)))
    return out


def append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def update_first_learning_steps(
    first_steps: dict[str, int],
    step: int,
    train_acc: float,
    loss_value: float,
) -> None:
    thresholds = {
        "acc_gt_0": train_acc > 0.0,
        "acc_ge_0_5": train_acc >= 0.50,
        "acc_ge_0_9": train_acc >= 0.90,
        "acc_ge_0_95": train_acc >= 0.95,
        "loss_lt_1": loss_value < 1.0,
    }
    for key, passed in thresholds.items():
        if passed and key not in first_steps:
            first_steps[key] = step


def train_row(row: str, args: argparse.Namespace, cells: list[Cell], device) -> dict[str, Any]:
    import torch
    from trellis_lm.model import build_model

    (
        kind,
        readout,
        value_alpha_mode,
        value_alpha_mix,
        value_alpha_correction_init,
        value_alpha_correction_max,
        value_read_query_mode,
        value_read_query_gate_init,
        value_read_query_gate_max,
        update_gate_mode,
        update_gate_init,
        update_gate_target,
        update_gate_floor,
    ) = row_spec(row)
    row_base, update_gate_layer_mode, update_gate_context_mode = row_base_layer_context(
        row
    )
    cfg = make_cfg(
        args,
        readout,
        value_alpha_mode,
        value_alpha_mix,
        value_alpha_correction_init,
        value_alpha_correction_max,
        value_read_query_mode,
        value_read_query_gate_init,
        value_read_query_gate_max,
        update_gate_mode,
        update_gate_init,
        update_gate_target,
        update_gate_layer_mode,
        update_gate_context_mode,
        update_gate_floor,
    )
    row_meta = {
        "row": row,
        "row_base": row_base,
        "kind": kind,
        "binding_probe_kind": kind if kind.startswith("trellis_role_") else "none",
        "value_readout_act": readout,
        "trellis_value_alpha_mode": value_alpha_mode,
        "trellis_value_alpha_mix": value_alpha_mix,
        "trellis_value_alpha_correction_init": value_alpha_correction_init,
        "trellis_value_alpha_correction_max": value_alpha_correction_max,
        "trellis_value_read_query_mode": value_read_query_mode,
        "trellis_value_read_query_gate_init": value_read_query_gate_init,
        "trellis_value_read_query_gate_max": value_read_query_gate_max,
        "update_gate_mode": update_gate_mode,
        "update_gate_init": update_gate_init,
        "trellis_update_gate_target": update_gate_target,
        "trellis_update_gate_layer_mode": update_gate_layer_mode,
        "trellis_update_gate_context_mode": update_gate_context_mode,
        "trellis_update_gate_floor": update_gate_floor,
        "binding_residual_init": args.binding_residual_init,
        "binding_residual_max": args.binding_residual_max,
        "binding_eta_init": args.binding_eta_init,
    }
    if kind.startswith("trellis_role_"):
        model = build_role_oracle_probe_model(cfg, kind, args).to(device)
    else:
        model = build_model(cfg, kind).to(device)
    # Keep master weights fp32, matching the C4 harness. Trellis intentionally
    # computes beta/gamma paths in fp32 under disabled autocast; casting modules
    # to bf16 makes beta_proj weights bf16 while h.float() is fp32.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(args.seed + 1009 * stable_row_seed(row_base))
    dt = as_dtype(args.dtype)
    hist = []
    first_learning_steps: dict[str, int] = {}
    t0 = time.time()
    ntok = 0
    model.train()
    for step in range(1, args.train_steps + 1):
        cell = rng.choice(cells)
        idx, labels, meta = make_batch(cell, args, rng, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=dt,
            enabled=args.dtype != "fp32",
        ):
            logits, loss = forward_model(
                model,
                idx,
                labels=labels,
                training=True,
                meta=meta,
            )
        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            return {
                **row_meta,
                "status": "diverged",
                "divergence_step": step,
                "divergence_reason": "nonfinite_loss",
                "loss": loss_value,
                "first_learning_steps": first_learning_steps,
                "trellis_diagnostics": (
                    collect_trellis_diagnostics(model) if args.diagnostics else []
                ),
                "history": hist,
                "params": model.get_num_params(),
            }
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        if not math.isfinite(gnorm):
            return {
                **row_meta,
                "status": "diverged",
                "divergence_step": step,
                "divergence_reason": "nonfinite_grad_norm",
                "loss": loss_value,
                "gnorm": gnorm,
                "first_learning_steps": first_learning_steps,
                "trellis_diagnostics": (
                    collect_trellis_diagnostics(model) if args.diagnostics else []
                ),
                "history": hist,
                "params": model.get_num_params(),
            }
        opt.step()
        ntok += idx.numel()
        with torch.no_grad():
            preds = []
            for bi, pos in enumerate(meta.query_pos):
                preds.append(int(logits[bi, pos].argmax(-1).item()))
            counts = accuracy_metrics(preds, meta, args)
            train_acc = counts["correct"] / max(1, counts["total"])
            update_first_learning_steps(
                first_learning_steps,
                step,
                train_acc,
                loss_value,
            )
        diag_due = args.diagnostics and (
            step == 1 or step % (args.diag_every or args.log_every) == 0
        )
        if step == 1 or step % args.log_every == 0 or diag_due:
            layer_diag = collect_trellis_diagnostics(model) if args.diagnostics else []
            entry = {
                "step": step,
                "tokens": ntok,
                "loss": round(loss_value, 6),
                "gnorm": round(gnorm, 6),
                "train_acc": train_acc,
                "tok_s": ntok / max(1e-9, time.time() - t0),
            }
            if layer_diag:
                entry["trellis_diagnostics"] = layer_diag
            hist.append(entry)
            if layer_diag:
                append_jsonl(
                    args.diag_jsonl,
                    {
                        "row": row,
                        "seed": args.seed,
                        "step": step,
                        "tokens": ntok,
                        "loss": loss_value,
                        "gnorm": gnorm,
                        "train_acc": train_acc,
                        "layers": layer_diag,
                    },
                )
            print(
                f"  [{row}] step {step}/{args.train_steps} "
                f"loss {loss_value:.4f} acc {entry['train_acc']:.3f} "
                f"tok/s {entry['tok_s']:.0f}",
                flush=True,
            )
    metrics = eval_row(model, row, args, cells, device)
    correction_scales = []
    for module in model.modules():
        raw = getattr(module, "value_alpha_correction_raw", None)
        if raw is None:
            continue
        max_scale = float(module.cfg.trellis_value_alpha_correction_max)
        scale = max_scale * torch.sigmoid(raw.detach().float())
        correction_scales.append(scale.cpu())
    if correction_scales:
        flat = torch.cat([item.reshape(-1) for item in correction_scales])
        row_meta["trellis_value_alpha_correction_scale"] = {
            "mean": float(flat.mean().item()),
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
        }
    gate_bias_values = []
    gate_weight_norms = []
    gate_bias_effective = []
    for module in model.modules():
        proj = getattr(module, "update_gate_proj", None)
        if proj is None:
            continue
        bias = proj.bias.detach().float()
        gate_bias_values.append(bias.cpu())
        gate_weight_norms.append(proj.weight.detach().float().norm().reshape(1).cpu())
        gate_bias_effective.append(module._update_gate_from_logits(bias).cpu())
    if gate_bias_values:
        bias_flat = torch.cat([item.reshape(-1) for item in gate_bias_values])
        weight_flat = torch.cat(gate_weight_norms)
        effective_flat = torch.cat([item.reshape(-1) for item in gate_bias_effective])
        row_meta["update_gate_bias"] = {
            "mean": float(bias_flat.mean().item()),
            "min": float(bias_flat.min().item()),
            "max": float(bias_flat.max().item()),
        }
        row_meta["update_gate_weight_norm"] = {
            "mean": float(weight_flat.mean().item()),
            "max": float(weight_flat.max().item()),
        }
        row_meta["update_gate_bias_effective"] = {
            "mean": float(effective_flat.mean().item()),
            "min": float(effective_flat.min().item()),
            "max": float(effective_flat.max().item()),
        }
    address_weight_norms = []
    for module in model.modules():
        proj = getattr(module, "value_address_proj", None)
        if proj is None:
            continue
        address_weight_norms.append(
            proj.weight.detach().float().norm().reshape(1).cpu()
        )
    if address_weight_norms:
        flat = torch.cat(address_weight_norms)
        row_meta["value_address_weight_norm"] = {
            "mean": float(flat.mean().item()),
            "max": float(flat.max().item()),
        }
    if getattr(model, "requires_binding_meta", False):
        row_meta["binding_probe_final_diag"] = getattr(
            model,
            "last_binding_diag",
            None,
        )
    metrics.update({
        **row_meta,
        "status": "ok",
        "params": model.get_num_params(),
        "memory_state_bytes_per_seq": model.memory_state_bytes(1),
        "train_tokens": ntok,
        "first_learning_steps": first_learning_steps,
        "history": hist,
    })
    return metrics


def eval_row(model, row: str, args: argparse.Namespace, cells: list[Cell], device):
    import torch

    row_base, _, _ = row_base_layer_context(row)
    rng = random.Random(args.seed + 7919 * stable_row_seed(row_base) + 17)
    model.eval()
    by_cell: list[dict[str, Any]] = []
    by_pressure: dict[str, dict[str, int]] = {}
    by_context: dict[str, dict[str, int]] = {}
    by_overwrite: dict[str, dict[str, int]] = {}
    total_counts: dict[str, int] = {}
    with torch.no_grad():
        for cell in cells:
            cell_counts: dict[str, int] = {}
            distance_sum = 0
            for _ in range(args.eval_batches):
                idx, _, meta = make_batch(cell, args, rng, device)
                logits, _ = forward_model(
                    model,
                    idx,
                    training=False,
                    meta=meta,
                )
                preds = []
                for bi, pos in enumerate(meta.query_pos):
                    preds.append(int(logits[bi, pos].argmax(-1).item()))
                counts = accuracy_metrics(preds, meta, args)
                add_counts(cell_counts, counts)
                add_counts(total_counts, counts)
                pressure = f"{cell.n_unique / max(1, args.n_slots):.2f}"
                by_pressure.setdefault(pressure, {})
                add_counts(by_pressure[pressure], counts)
                ctx = str(cell.seq_len)
                by_context.setdefault(ctx, {})
                add_counts(by_context[ctx], counts)
                ow = str(cell.overwrites)
                by_overwrite.setdefault(ow, {})
                add_counts(by_overwrite[ow], counts)
                distance_sum += sum(meta.latest_distance)
            item = finalize_counts(cell_counts)
            item.update({
                "seq_len": cell.seq_len,
                "n_unique": cell.n_unique,
                "overwrites": cell.overwrites,
                "latest_gap_target": cell.latest_gap,
                "slot_pressure": cell.n_unique / max(1, args.n_slots),
                "latest_distance_mean": (
                    distance_sum / max(1, cell_counts.get("total", 0))
                ),
            })
            by_cell.append(item)
    return {
        "eval": finalize_counts(total_counts),
        "by_cell": by_cell,
        "by_slot_pressure": {
            key: finalize_counts(value) for key, value in sorted(by_pressure.items())
        },
        "by_context": {
            key: finalize_counts(value) for key, value in sorted(by_context.items())
        },
        "by_overwrites": {
            key: finalize_counts(value) for key, value in sorted(by_overwrite.items())
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="trellis_none,trellis_norm_silu,gdn_ref,dense")
    p.add_argument("--contexts", default="512,1024,2048")
    p.add_argument("--unique-keys", default="16,48,96")
    p.add_argument("--overwrites", default="0,1,2,4")
    p.add_argument("--latest-gaps", default="32,128,512")
    p.add_argument("--n-keys", type=int, default=256)
    p.add_argument("--n-vals", type=int, default=256)
    p.add_argument("--distractor-vocab", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-head", type=int, default=64)
    p.add_argument("--n-slots", type=int, default=48)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--diag-every", type=int, default=0)
    p.add_argument("--diag-jsonl", type=Path, default=None)
    p.add_argument("--binding-residual-init", type=float, default=0.10)
    p.add_argument("--binding-residual-max", type=float, default=1.0)
    p.add_argument("--binding-eta-init", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=Path("overwrite_probe_results.json"))
    p.add_argument("--print-cells", action="store_true")
    args = p.parse_args()

    cells = build_cells(args)
    if args.print_cells:
        payload = {
            "args": json_args(args),
            "n_cells": len(cells),
            "cells": [cell.__dict__ for cell in cells],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if not cells:
        raise SystemExit("no viable cells; lower unique keys/overwrites or raise context")

    import torch

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [item.strip() for item in args.rows.split(",") if item.strip()]
    print(f"device={device} rows={rows} cells={len(cells)}", flush=True)
    results = []
    for row in rows:
        print(f"=== {row} ===", flush=True)
        result = train_row(row, args, cells, device)
        results.append(result)
        if result.get("status") == "ok":
            ev = result["eval"]
            print(
                f"  [{row}] eval acc={ev['latest_value_accuracy']:.3f} "
                f"stale={ev['stale_value_error_rate']:.3f} "
                f"keyconf={ev['key_confusion_error_rate']:.3f}",
                flush=True,
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "args": json_args(args),
            "cells": [cell.__dict__ for cell in cells],
            "results": results,
        }, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
