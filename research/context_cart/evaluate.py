# SPDX-License-Identifier: MIT
"""Reader arms, prediction records, and exact-match metrics.

Every arm answers through the same frozen reader, instruction, question
format, greedy decoding, answer cap, and fixed batch schedule. Text arms
(full context, empty, tail text, text summary) prefill their context once per
world and answer each question from a fresh fork of that cache. Memory arms
(untrained gist, learned gist, wrong cartridge) receive 64 key/value rows per
world as the initial cache and process only the instruction and question.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from . import dataset as D
from . import model as M
from . import prompts as P

Prefix = list[tuple[torch.Tensor, torch.Tensor]]


@dataclasses.dataclass
class Item:
    world_id: str
    qid: str
    family: str
    attribute: str
    question: str
    answer: str | None
    stratum: str = "ordinary"


def items_from_worlds(
    worlds: Iterable[dict[str, Any]], key: str = "questions"
) -> list[Item]:
    items = []
    for world in worlds:
        for q in world[key]:
            items.append(
                Item(
                    world_id=world["world_id"],
                    qid=q["qid"],
                    family=q["family"],
                    attribute=q["attribute"],
                    question=q["text"],
                    answer=q.get("answer"),
                    stratum=world.get("stratum", "ordinary"),
                )
            )
    return items


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def split_prompt(
    tokenizer: Any, arm: str, question: str, source_text: str, context: str
) -> tuple[list[int], list[int]]:
    """Context part (prefilled once per world) and question part token ids.

    The split point is the question line; the two parts tokenize exactly as
    the joined prompt does, which is asserted.
    """
    content = P.user_message(arm, question, source_text=source_text, context=context)
    full = P.render_prompt(tokenizer, content)
    marker = "Question: "
    index = full.rfind(marker)
    head, tail = full[:index], full[index:]
    head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    tail_ids = tokenizer(tail, add_special_tokens=False)["input_ids"]
    joined = tokenizer(full, add_special_tokens=False)["input_ids"]
    if list(head_ids) + list(tail_ids) != list(joined):
        raise ValueError("prompt split changes tokenization")
    return list(head_ids), list(tail_ids)


def decode_prediction(
    tokenizer: Any, ids: list[int], eos_ids: tuple[int, ...], pad_id: int
) -> str:
    kept = M.strip_generation(ids, eos_ids, pad_id)
    return tokenizer.decode(kept, skip_special_tokens=True).strip()


@dataclasses.dataclass
class ArmTiming:
    build_seconds: float = 0.0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    answers: int = 0
    worlds: int = 0


def prediction_row(
    item: Item,
    arm: str,
    pred: str,
    ids: list[int],
    prompt_tokens: int,
    cache_rows: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "arm": arm,
        "world_id": item.world_id,
        "qid": item.qid,
        "family": item.family,
        "attribute": item.attribute,
        "stratum": item.stratum,
        "question": item.question,
        "gold": item.answer,
        "pred": pred,
        "pred_ids": ids,
        "prompt_tokens": prompt_tokens,
        "cache_rows": cache_rows,
    }
    if item.answer is not None:
        row["em"] = bool(D.exact_match(pred, item.answer, item.attribute))
    if extra:
        row.update(extra)
    return row


@torch.no_grad()
def encode_worlds(
    writer: M.Writer,
    worlds: list[dict[str, Any]],
    batch: int,
    device: torch.device | str,
) -> tuple[dict[str, Prefix], int, float]:
    """One cartridge per world from the writer; returns per-world single
    prefixes, the source length ``L``, and the build time in seconds."""
    sync = (
        torch.cuda.synchronize
        if torch.device(device).type == "cuda"
        else (lambda: None)
    )
    sync()
    t0 = time.perf_counter()
    prefixes: dict[str, Prefix] = {}
    length = None
    for group in chunks(worlds, batch):
        source = torch.tensor(
            [w["source_ids"] for w in group], dtype=torch.long, device=device
        )
        prefix, length = writer.encode(source)
        for row, world in enumerate(group):
            prefixes[world["world_id"]] = [
                (k[row : row + 1].detach().clone(), v[row : row + 1].detach().clone())
                for k, v in prefix
            ]
    sync()
    return prefixes, int(length), time.perf_counter() - t0


@torch.no_grad()
def answer_memory_arm(
    reader: M.Reader,
    tokenizer: Any,
    arm: str,
    prefixes: dict[str, Prefix],
    items: list[Item],
    first_position: int,
    batch: int,
    max_new_tokens: int,
    mapping: dict[str, str] | None = None,
    keep_logits: bool = False,
) -> tuple[list[dict[str, Any]], ArmTiming, list[torch.Tensor]]:
    """Answer ``items`` with each world's own cartridge, or with the donor
    world's cartridge under ``mapping`` for the wrong-cartridge control."""
    timing = ArmTiming(worlds=len(prefixes))
    rows: list[dict[str, Any]] = []
    logits_out: list[torch.Tensor] = []
    device = reader.device
    for group in chunks(items, batch):
        prompts = [
            P.prompt_ids(tokenizer, P.user_message(arm, item.question))
            for item in group
        ]
        ids, mask = M.left_pad(prompts, reader.pad_id, device)
        sources = [
            mapping[item.world_id] if mapping else item.world_id for item in group
        ]
        prefix = M.stack_prefixes([prefixes[s] for s in sources])
        out = reader.greedy(
            prefix, ids, mask, first_position, max_new_tokens, keep_logits=keep_logits
        )
        timing.prefill_seconds += out["prefill_seconds"]
        timing.decode_seconds += out["decode_seconds"]
        timing.answers += len(group)
        for row, item in enumerate(group):
            gen = out["ids"][row].tolist()
            pred = decode_prediction(tokenizer, gen, reader.eos_ids, reader.pad_id)
            rows.append(
                prediction_row(
                    item,
                    arm,
                    pred,
                    M.strip_generation(gen, reader.eos_ids, reader.pad_id),
                    len(prompts[row]),
                    out["cache_rows"],
                    {"cartridge_world": sources[row]},
                )
            )
            if keep_logits:
                logits_out.append(out["step_logits"][row].cpu())
    return rows, timing, logits_out


@torch.no_grad()
def answer_text_arm(
    reader: M.Reader,
    tokenizer: Any,
    arm: str,
    worlds: list[dict[str, Any]],
    items: list[Item],
    batch: int,
    max_new_tokens: int,
    context_by_world: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], ArmTiming]:
    """Text arms: prefill each world's context once, fork per question."""
    timing = ArmTiming(worlds=len(worlds))
    rows: list[dict[str, Any]] = []
    device = reader.device
    by_world: dict[str, list[Item]] = {}
    for item in items:
        by_world.setdefault(item.world_id, []).append(item)
    for group in chunks(worlds, batch):
        heads, tails = [], []
        for world in group:
            context = (context_by_world or {}).get(world["world_id"], "")
            per_q = []
            head_ids = None
            for item in by_world.get(world["world_id"], []):
                head, tail = split_prompt(
                    tokenizer, arm, item.question, world["source_text"], context
                )
                if head_ids is None:
                    head_ids = head
                elif head != head_ids:
                    raise ValueError(
                        "context part differs between questions of one world"
                    )
                per_q.append((item, tail))
            heads.append(head_ids or [])
            tails.append(per_q)
        sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
        sync()
        t0 = time.perf_counter()
        head_ids, head_mask = M.left_pad(heads, reader.pad_id, device)
        _, cache = reader.forward(None, head_ids, head_mask, 0)
        context_prefix = [
            M.cache_layer(cache, layer) for layer in range(reader.geometry.layers)
        ]
        context_prefix = [(k.clone(), v.clone()) for k, v in context_prefix]
        sync()
        timing.build_seconds += time.perf_counter() - t0
        lengths = head_mask.sum(-1)
        questions = max(len(t) for t in tails)
        for index in range(questions):
            active = [row for row, t in enumerate(tails) if index < len(t)]
            prompts = [tails[row][index][1] for row in active]
            ids, mask = M.left_pad(prompts, reader.pad_id, device)
            prefix = [(k[active], v[active]) for k, v in context_prefix]
            out = reader.greedy(
                prefix,
                ids,
                mask,
                lengths[active],
                max_new_tokens,
                prefix_mask=head_mask[active],
            )
            timing.prefill_seconds += out["prefill_seconds"]
            timing.decode_seconds += out["decode_seconds"]
            timing.answers += len(active)
            for slot, row in enumerate(active):
                item = tails[row][index][0]
                gen = out["ids"][slot].tolist()
                pred = decode_prediction(tokenizer, gen, reader.eos_ids, reader.pad_id)
                rows.append(
                    prediction_row(
                        item,
                        arm,
                        pred,
                        M.strip_generation(gen, reader.eos_ids, reader.pad_id),
                        len(prompts[slot]),
                        out["cache_rows"],
                        {"context_tokens": int(lengths[row].item())},
                    )
                )
    return rows, timing


@torch.no_grad()
def write_summaries(
    reader: M.Reader,
    tokenizer: Any,
    worlds: list[dict[str, Any]],
    batch: int,
    max_new_tokens: int,
) -> tuple[dict[str, str], dict[str, Any], float]:
    """Query-blind text summaries from the frozen reader, one per world."""
    device = reader.device
    summaries: dict[str, str] = {}
    lengths: dict[str, int] = {}
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    sync()
    t0 = time.perf_counter()
    for group in chunks(worlds, batch):
        prompts = [P.summary_prompt_ids(tokenizer, w["source_text"]) for w in group]
        ids, mask = M.left_pad(prompts, reader.pad_id, device)
        out = reader.greedy(None, ids, mask, 0, max_new_tokens)
        for row, world in enumerate(group):
            gen = M.strip_generation(
                out["ids"][row].tolist(), reader.eos_ids, reader.pad_id
            )
            summaries[world["world_id"]] = tokenizer.decode(
                gen, skip_special_tokens=True
            ).strip()
            lengths[world["world_id"]] = len(gen)
    sync()
    seconds = time.perf_counter() - t0
    stats = {
        "worlds": len(worlds),
        "max_new_tokens": max_new_tokens,
        "generated_tokens_mean": sum(lengths.values()) / max(1, len(lengths)),
        "generated_tokens_max": max(lengths.values()) if lengths else 0,
        "generated_tokens_total": sum(lengths.values()),
        "lengths": lengths,
    }
    return summaries, stats, seconds


def tail_text(tokenizer: Any, world: dict[str, Any], tokens: int) -> str:
    return tokenizer.decode(world["source_ids"][-tokens:], skip_special_tokens=False)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact match overall, per family, per stratum, and macro over families."""
    scored = [r for r in rows if "em" in r]
    by_family: dict[str, list[bool]] = {}
    by_stratum: dict[str, list[bool]] = {}
    by_both: dict[str, list[bool]] = {}
    for r in scored:
        by_family.setdefault(r["family"], []).append(r["em"])
        by_stratum.setdefault(r["stratum"], []).append(r["em"])
        by_both.setdefault(f"{r['stratum']}/{r['family']}", []).append(r["em"])
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    families = {f: mean(v) for f, v in sorted(by_family.items())}
    return {
        "n": len(scored),
        "em": mean([r["em"] for r in scored]),
        "macro_em": mean(list(families.values())) if families else float("nan"),
        "by_family": families,
        "by_stratum": {s: mean(v) for s, v in sorted(by_stratum.items())},
        "by_stratum_family": {k: mean(v) for k, v in sorted(by_both.items())},
        "counts_by_family": {f: len(v) for f, v in sorted(by_family.items())},
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def per_world_em(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_world: dict[str, list[bool]] = {}
    for r in rows:
        if "em" in r:
            by_world.setdefault(r["world_id"], []).append(r["em"])
    return {w: sum(v) / len(v) for w, v in by_world.items()}


def paired_bootstrap(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    resamples: int = 2000,
    seed: int = 20261025,
) -> dict[str, Any]:
    """95% interval of mean(world EM a - world EM b), resampling whole worlds."""
    import random

    a, b = per_world_em(rows_a), per_world_em(rows_b)
    worlds = sorted(set(a) & set(b))
    diffs = [a[w] - b[w] for w in worlds]
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(sum(sample) / len(sample))
    means.sort()
    low = means[int(0.025 * resamples)]
    high = means[min(resamples - 1, int(math.ceil(0.975 * resamples)) - 1)]
    return {
        "worlds": len(worlds),
        "mean_difference": sum(diffs) / len(diffs) if diffs else float("nan"),
        "ci95": [low, high],
        "resamples": resamples,
    }
