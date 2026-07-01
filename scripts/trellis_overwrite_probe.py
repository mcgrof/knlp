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
    if isinstance(out.get("out"), Path):
        out["out"] = str(out["out"])
    return out


def stable_row_seed(row: str) -> int:
    total = 0
    for idx, ch in enumerate(row):
        total += (idx + 1) * ord(ch)
    return total


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
    pre_extra = cell.seq_len - (3 * len(prefix_events) + 3 + filler_after + 3)
    fills = _distribute_fillers(rng, max(0, pre_extra), len(prefix_events) + 1)
    for event, n_fill in zip(prefix_events, fills):
        seq.extend([_filler(rng, args) for _ in range(n_fill)])
        seq.extend(event)
    if fills:
        seq.extend([_filler(rng, args) for _ in range(fills[-1])])
    seq.extend(latest_event)
    latest_value_pos = len(seq) - 1
    seq.extend([_filler(rng, args) for _ in range(filler_after)])
    seq.extend([ids["query"], qkey, answer])
    if len(seq) < cell.seq_len:
        pad = [_filler(rng, args) for _ in range(cell.seq_len - len(seq))]
        seq = pad + seq
        latest_value_pos += len(pad)
    if len(seq) != cell.seq_len:
        raise RuntimeError((len(seq), cell))
    query_pos = cell.seq_len - 2
    meta = {
        "query_pos": query_pos,
        "answer": answer,
        "stale_values": stale_values,
        "other_values": other_values,
        "latest_distance": query_pos - latest_value_pos,
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
    for _ in range(args.batch):
        seq, meta = make_example(cell, args, rng)
        rows.append(seq)
        query_pos.append(int(meta["query_pos"]))
        answers.append(int(meta["answer"]))
        stale_values.append(set(meta["stale_values"]))
        other_values.append(set(meta["other_values"]))
        latest_distance.append(int(meta["latest_distance"]))
    idx = torch.as_tensor(rows, dtype=torch.long, device=device)
    labels = torch.full_like(idx, -100)
    for bi, pos in enumerate(query_pos):
        labels[bi, pos + 1] = idx[bi, pos + 1]
    return idx, labels, BatchMeta(
        query_pos=query_pos,
        answers=answers,
        stale_values=stale_values,
        other_values=other_values,
        latest_distance=latest_distance,
    )


def make_cfg(
    args: argparse.Namespace,
    value_readout_act: str,
    value_alpha_mode: str,
    value_alpha_mix: float,
    value_alpha_correction_init: float,
    value_alpha_correction_max: float,
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
        trellis_retention_mode="token_proj",
        trellis_update_stabilizer="layerwise_gamma",
        trellis_layer0_gamma_mult=0.5,
        residual_update_mix=0.10,
    )


def row_spec(row: str) -> tuple[str, str, str, float, float, float]:
    if row == "trellis_none":
        return "trellis", "none", "shared", 1.0, 1e-3, 0.25
    if row == "trellis_norm_silu":
        return "trellis", "norm_silu", "shared", 1.0, 1e-3, 0.25
    if row == "trellis_keyed":
        return "trellis", "none", "key_readout", 1.0, 1e-3, 0.25
    if row == "trellis_keyed_detach":
        return "trellis", "none", "key_readout_detached", 1.0, 1e-3, 0.25
    if row == "trellis_keyed_norm_silu":
        return "trellis", "norm_silu", "key_readout", 1.0, 1e-3, 0.25
    if row == "trellis_corr1e3":
        return "trellis", "none", "shared_plus_key_correction", 1.0, 1e-3, 0.25
    if row == "trellis_corr1e2":
        return "trellis", "none", "shared_plus_key_correction", 1.0, 1e-2, 0.25
    if row == "trellis_corr_detach1e3":
        return (
            "trellis",
            "none",
            "shared_plus_key_correction_detached",
            1.0,
            1e-3,
            0.25,
        )
    if row == "trellis_corr_norm_silu1e3":
        return (
            "trellis",
            "norm_silu",
            "shared_plus_key_correction",
            1.0,
            1e-3,
            0.25,
        )
    if row == "gdn_ref":
        return "gated_delta_ref", "none", "shared", 1.0, 1e-3, 0.25
    if row == "dense":
        return "dense", "none", "shared", 1.0, 1e-3, 0.25
    raise ValueError(row)


def as_dtype(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


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
    ) = row_spec(row)
    cfg = make_cfg(
        args,
        readout,
        value_alpha_mode,
        value_alpha_mix,
        value_alpha_correction_init,
        value_alpha_correction_max,
    )
    row_meta = {
        "row": row,
        "kind": kind,
        "value_readout_act": readout,
        "trellis_value_alpha_mode": value_alpha_mode,
        "trellis_value_alpha_mix": value_alpha_mix,
        "trellis_value_alpha_correction_init": value_alpha_correction_init,
        "trellis_value_alpha_correction_max": value_alpha_correction_max,
    }
    model = build_model(cfg, kind).to(device)
    # Keep master weights fp32, matching the C4 harness. Trellis intentionally
    # computes beta/gamma paths in fp32 under disabled autocast; casting modules
    # to bf16 makes beta_proj weights bf16 while h.float() is fp32.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(args.seed + 1009 * stable_row_seed(row))
    dt = as_dtype(args.dtype)
    hist = []
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
            logits, loss = model(idx, labels=labels, training=True)
        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            return {
                **row_meta,
                "status": "diverged",
                "divergence_step": step,
                "divergence_reason": "nonfinite_loss",
                "loss": loss_value,
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
                "history": hist,
                "params": model.get_num_params(),
            }
        opt.step()
        ntok += idx.numel()
        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                preds = []
                for bi, pos in enumerate(meta.query_pos):
                    preds.append(int(logits[bi, pos].argmax(-1).item()))
                counts = accuracy_metrics(preds, meta, args)
            entry = {
                "step": step,
                "tokens": ntok,
                "loss": round(loss_value, 6),
                "gnorm": round(gnorm, 6),
                "train_acc": counts["correct"] / max(1, counts["total"]),
                "tok_s": ntok / max(1e-9, time.time() - t0),
            }
            hist.append(entry)
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
    metrics.update({
        **row_meta,
        "status": "ok",
        "params": model.get_num_params(),
        "memory_state_bytes_per_seq": model.memory_state_bytes(1),
        "train_tokens": ntok,
        "history": hist,
    })
    return metrics


def eval_row(model, row: str, args: argparse.Namespace, cells: list[Cell], device):
    import torch

    rng = random.Random(args.seed + 7919 * stable_row_seed(row) + 17)
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
                logits, _ = model(idx, training=False)
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
