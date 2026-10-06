#!/usr/bin/env python3
"""Run the frozen CoQA smoke, Q1, and conditional Q2 GPU stages."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import torch

from . import coqa, fit_pair
from .capture import cache_layers, geometry, make_cache
from .run_pair import load_model

ON_CHECKPOINT = None


def log(message):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {message}", flush=True)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def checkpoint():
    if ON_CHECKPOINT is not None:
        ON_CHECKPOINT()


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ids_sha256(ids):
    return hashlib.sha256(
        b"".join(int(token).to_bytes(8, "little", signed=False) for token in ids)
    ).hexdigest()


def cache_length(cache) -> int:
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    pairs = cache_layers(cache)
    return int(pairs[0][0].shape[-2])


def clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


class ModelPair:
    def __init__(self, tokenizer, model14, model32, mappers):
        self.tokenizer = tokenizer
        self.models = {"14B": model14, "32B": model32}
        self.geom = {name: geometry(model) for name, model in self.models.items()}
        self.mappers = mappers
        self.device = next(model14.parameters()).device
        if next(model32.parameters()).device != self.device:
            raise ValueError("both models must be on the same accelerator")

    def prefill(self, model_name: str, token_ids: list[int]):
        if not token_ids:
            raise ValueError("cannot prefill an empty prefix")
        model = self.models[model_name]
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        start = clock(self.device)
        with torch.inference_mode():
            output = model(input_ids=ids, use_cache=True)
        seconds = clock(self.device) - start
        cache = output.past_key_values
        if cache_length(cache) != len(token_ids):
            raise RuntimeError(
                f"{model_name} prefill cache has {cache_length(cache)} tokens, "
                f"expected {len(token_ids)}"
            )
        return cache, seconds

    def consume(self, model_name: str, cache, token_ids: list[int]):
        if not token_ids:
            return cache, 0.0
        before = cache_length(cache)
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        start = clock(self.device)
        with torch.inference_mode():
            output = self.models[model_name](
                input_ids=ids, past_key_values=cache, use_cache=True
            )
        seconds = clock(self.device) - start
        cache = output.past_key_values
        if cache_length(cache) != before + len(token_ids):
            raise RuntimeError(
                f"{model_name} consume cache has {cache_length(cache)} tokens, "
                f"expected {before + len(token_ids)}"
            )
        return cache, seconds

    def map_cache(self, donor: str, receiver: str, cache):
        key = f"{donor}_to_{receiver}"
        mapper = self.mappers[key]
        start = clock(self.device)
        with torch.inference_mode():
            pairs = fit_pair.transfer(
                mapper,
                cache_layers(cache),
                self.geom[donor]["rope_theta"],
                self.geom[receiver]["rope_theta"],
                dtype=next(self.models[receiver].parameters()).dtype,
            )
            mapped = make_cache(pairs)
        seconds = clock(self.device) - start
        if cache_length(mapped) != cache_length(cache):
            raise RuntimeError("mapping changed the cache sequence length")
        return mapped, seconds

    def _decoded(self, generated_ids):
        kwargs = {"clean_up_tokenization_spaces": False}
        raw = self.tokenizer.decode(generated_ids, skip_special_tokens=False, **kwargs)
        visible = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, **kwargs
        )
        answer = visible.split("\n", 1)[0].strip()
        return raw, answer

    def _decode(self, model_name: str, logits, cache, prompt_length: int):
        model = self.models[model_name]
        generated = []
        finite = True
        start = clock(self.device)
        stop_reason = "cap"
        first_token = None
        for step in range(coqa.MAX_NEW_TOKENS):
            scores = logits[0, -1].float()
            if not torch.isfinite(scores).all():
                finite = False
                raise FloatingPointError(
                    f"{model_name} produced nonfinite generation logits"
                )
            token = int(scores.argmax().item())
            if first_token is None:
                first_token = token
            generated.append(token)
            raw, _answer = self._decoded(generated)
            if token == self.tokenizer.eos_token_id:
                stop_reason = "eos"
                break
            if "\n" in raw:
                stop_reason = "newline"
                break
            if step + 1 == coqa.MAX_NEW_TOKENS:
                stop_reason = "cap"
                break
            ids = torch.tensor([[token]], dtype=torch.long, device=self.device)
            with torch.inference_mode():
                output = model(input_ids=ids, past_key_values=cache, use_cache=True)
            logits, cache = output.logits, output.past_key_values
        seconds = clock(self.device) - start
        raw, answer = self._decoded(generated)
        covered = prompt_length + max(0, len(generated) - 1)
        if cache_length(cache) != covered:
            raise RuntimeError(
                f"{model_name} generated cache has {cache_length(cache)} tokens, "
                f"expected {covered}"
            )
        health = coqa.health_events(
            answer,
            raw,
            stop_reason=stop_reason,
            new_tokens=len(generated),
        )
        return {
            "answer": answer,
            "raw_text": raw,
            "token_ids": generated,
            "first_token_id": first_token,
            "stop_reason": stop_reason,
            "new_tokens": len(generated),
            "health_events": health,
            "finite_logits": finite,
            "cache": cache,
            "cache_covered_tokens": covered,
            "generation_seconds": seconds,
        }

    def generate_split(
        self,
        model_name: str,
        prompt_ids: list[int],
        cache=None,
        *,
        cache_cut: int | None = None,
    ):
        """Generate after a cached prefix ending at an arbitrary token cut.

        The historical path caches every prompt token except the final one.
        ``cache_cut`` keeps that behavior as the default while allowing an
        earlier reusable prefix.  Every prompt token after the cut is consumed
        by the receiver in one call before greedy decoding starts.
        """
        if not prompt_ids:
            raise ValueError("cannot generate from an empty prompt")
        if cache_cut is None:
            cache_cut = len(prompt_ids) - 1
        if cache_cut < 1 or cache_cut >= len(prompt_ids):
            raise ValueError(
                f"cache cut {cache_cut} is outside 1..{len(prompt_ids) - 1}"
            )
        prefix = prompt_ids[:cache_cut]
        suffix = prompt_ids[cache_cut:]
        prefill_seconds = 0.0
        if cache is None:
            cache, prefill_seconds = self.prefill(model_name, prefix)
        elif cache_length(cache) != len(prefix):
            raise RuntimeError(
                f"supplied {model_name} cache has {cache_length(cache)}, expected {len(prefix)}"
            )
        ids = torch.tensor([suffix], dtype=torch.long, device=self.device)
        start = clock(self.device)
        with torch.inference_mode():
            output = self.models[model_name](
                input_ids=ids, past_key_values=cache, use_cache=True
            )
        suffix_seconds = clock(self.device) - start
        if cache_length(output.past_key_values) != len(prompt_ids):
            raise RuntimeError(
                f"{model_name} prompt cache has "
                f"{cache_length(output.past_key_values)} tokens, "
                f"expected {len(prompt_ids)}"
            )
        result = self._decode(
            model_name, output.logits, output.past_key_values, len(prompt_ids)
        )
        result["prefill_seconds"] = prefill_seconds
        result["suffix_seconds"] = suffix_seconds
        result["suffix_token_count"] = len(suffix)
        result["cache_cut"] = cache_cut
        result["final_token_seconds"] = suffix_seconds if len(suffix) == 1 else 0.0
        return result

    def generate_direct(self, model_name: str, prompt_ids: list[int]):
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        start = clock(self.device)
        with torch.inference_mode():
            output = self.models[model_name](input_ids=ids, use_cache=True)
        prefill_seconds = clock(self.device) - start
        result = self._decode(
            model_name, output.logits, output.past_key_values, len(prompt_ids)
        )
        result["prefill_seconds"] = prefill_seconds
        result["final_token_seconds"] = 0.0
        return result


def clean_generation(result):
    return {key: value for key, value in result.items() if key != "cache"}


def score_result(result, references):
    scored = coqa.turn_score(references, result["answer"])
    return {**clean_generation(result), **scored}


def load_mapper(path, device):
    mapper = fit_pair.load_mapper(path)
    moved = {
        **mapper,
        "W": mapper["W"].to(device),
        "b": mapper["b"].to(device),
    }
    del mapper
    gc.collect()
    return moved


def load_pair(forward_mapper, reverse_mapper=None):
    from transformers import AutoTokenizer

    source_name = "Qwen/Qwen3-14B"
    source_revision = "40c069824f4251a91eefaf281ebe4c544efd3e18"
    target_name = "Qwen/Qwen3-32B"
    target_revision = "9216db5781bf21249d130ec9da846c4624c16137"
    tokenizer = AutoTokenizer.from_pretrained(target_name, revision=target_revision)
    model14 = load_model(source_name, source_revision, "cuda")
    model32 = load_model(target_name, target_revision, "cuda")
    mappers = {"14B_to_32B": load_mapper(forward_mapper, "cuda")}
    if reverse_mapper:
        mappers["32B_to_14B"] = load_mapper(reverse_mapper, "cuda")
    return ModelPair(tokenizer, model14, model32, mappers)


def q1_one(pair: ModelPair, row: dict):
    prompt_ids = row["token_ids"]
    native = pair.generate_split("32B", prompt_ids)
    source_cache, source_seconds = pair.prefill("14B", prompt_ids[:-1])
    mapped, mapping_seconds = pair.map_cache("14B", "32B", source_cache)
    transfer = pair.generate_split("32B", prompt_ids, mapped)
    native_score = score_result(native, row["references"])
    transfer_score = score_result(transfer, row["references"])
    return {
        "row_id": row["row_id"],
        "conversation_id": row["conversation_id"],
        "domain": row["domain"],
        "turn": row["turn"],
        "token_ids_sha256": row["token_ids_sha256"],
        "prefix_token_count": row["prefix_token_count"],
        "native_answer": native_score["answer"],
        "native_raw_text": native_score["raw_text"],
        "native_token_ids": native_score["token_ids"],
        "native_stop_reason": native_score["stop_reason"],
        "native_health": native_score["health_events"],
        "native_f1": native_score["f1"],
        "native_em": native_score["em"],
        "transfer_answer": transfer_score["answer"],
        "transfer_raw_text": transfer_score["raw_text"],
        "transfer_token_ids": transfer_score["token_ids"],
        "transfer_stop_reason": transfer_score["stop_reason"],
        "transfer_health": transfer_score["health_events"],
        "transfer_f1": transfer_score["f1"],
        "transfer_em": transfer_score["em"],
        "timings_seconds": {
            "native_prefill": native["prefill_seconds"],
            "native_generate": native["generation_seconds"],
            "source_prefill": source_seconds,
            "mapping": mapping_seconds,
            "transfer_generate": transfer["generation_seconds"],
        },
    }


def native_trajectory(pair: ModelPair, story: dict, model_name: str, turns: int):
    rows, ledger, cache, covered = [], None, None, 0
    prior_stop = None
    for turn in range(1, turns + 1):
        if turn == 1:
            ledger = pair.tokenizer.encode(
                coqa.render_prompt(story, 1, []), add_special_tokens=False
            )
            cache, _ = pair.prefill(model_name, ledger[:-1])
            covered = len(ledger) - 1
        else:
            suffix = coqa.next_turn_suffix(
                story["questions"][turn - 1]["input_text"], prior_stop
            )
            suffix_ids = pair.tokenizer.encode(suffix, add_special_tokens=False)
            pending = ledger[covered:] + suffix_ids[:-1]
            cache, _ = pair.consume(model_name, cache, pending)
            ledger.extend(suffix_ids)
            covered = len(ledger) - 1
        result = pair.generate_split(model_name, ledger, cache)
        scored = score_result(result, coqa.answer_references(story, turn))
        rows.append(
            {
                "stream": f"native_{model_name}_only",
                "conversation_id": story["id"],
                "domain": story["source"],
                "turn": turn,
                "model": model_name,
                **clean_generation(scored),
                "f1": scored["f1"],
                "em": scored["em"],
                "prompt_token_ids_sha256": ids_sha256(ledger),
            }
        )
        ledger.extend(result["token_ids"])
        cache = result["cache"]
        covered = result["cache_covered_tokens"]
        prior_stop = result["stop_reason"]
    return rows


def alternating_trajectory(
    pair: ModelPair, story: dict, first_receiver: str, turns: int, schedule: str
):
    donor = "14B" if first_receiver == "32B" else "32B"
    owner = donor
    ledger = pair.tokenizer.encode(
        coqa.render_prompt(story, 1, []), add_special_tokens=False
    )
    owner_cache, _ = pair.prefill(owner, ledger[:-1])
    covered = len(ledger) - 1
    mapping_count = 0
    prior_stop = None
    answers, provenance = [], []

    for turn in range(1, turns + 1):
        if turn > 1:
            suffix = coqa.next_turn_suffix(
                story["questions"][turn - 1]["input_text"], prior_stop
            )
            suffix_ids = pair.tokenizer.encode(suffix, add_special_tokens=False)
            pending = ledger[covered:] + suffix_ids[:-1]
            owner_cache, _ = pair.consume(owner, owner_cache, pending)
            ledger.extend(suffix_ids)
            covered = len(ledger) - 1

        receiver = "32B" if owner == "14B" else "14B"
        if cache_length(owner_cache) != len(ledger) - 1:
            raise RuntimeError("owner cache does not cover the exact handoff prefix")

        shadow = pair.generate_split(receiver, ledger)
        mapped, mapping_seconds = pair.map_cache(owner, receiver, owner_cache)
        mapping_count += 1
        transferred = pair.generate_split(receiver, ledger, mapped)
        refs = coqa.answer_references(story, turn)
        shadow_score = score_result(shadow, refs)
        transfer_score = score_result(transferred, refs)
        answers.extend(
            [
                {
                    "stream": f"{schedule}_native_shadow",
                    "schedule": schedule,
                    "conversation_id": story["id"],
                    "domain": story["source"],
                    "turn": turn,
                    "receiver": receiver,
                    **clean_generation(shadow_score),
                    "f1": shadow_score["f1"],
                    "em": shadow_score["em"],
                },
                {
                    "stream": f"{schedule}_translated",
                    "schedule": schedule,
                    "conversation_id": story["id"],
                    "domain": story["source"],
                    "turn": turn,
                    "donor": owner,
                    "receiver": receiver,
                    "mapping_count": mapping_count,
                    "mapping_seconds": mapping_seconds,
                    **clean_generation(transfer_score),
                    "f1": transfer_score["f1"],
                    "em": transfer_score["em"],
                },
            ]
        )
        provenance.append(
            {
                "schedule": schedule,
                "conversation_id": story["id"],
                "domain": story["source"],
                "turn": turn,
                "donor": owner,
                "receiver": receiver,
                "mapping_count": mapping_count,
                "ledger_tokens_before_answer": len(ledger),
                "cache_tokens_before_final_token": len(ledger) - 1,
                "ledger_sha256_before_answer": ids_sha256(ledger),
                "prefix_sha256": ids_sha256(ledger[:-1]),
                "final_token_id": ledger[-1],
                "generated_token_ids": transferred["token_ids"],
                "generated_stop_reason": transferred["stop_reason"],
                "cache_covered_after_generation": transferred["cache_covered_tokens"],
            }
        )

        ledger.extend(transferred["token_ids"])
        owner_cache = transferred["cache"]
        covered = transferred["cache_covered_tokens"]
        owner = receiver
        prior_stop = transferred["stop_reason"]
    return answers, provenance


def q2_conversation(pair: ModelPair, story: dict, turns: int):
    native14 = native_trajectory(pair, story, "14B", turns)
    native32 = native_trajectory(pair, story, "32B", turns)
    native_index = {(row["model"], row["turn"]): row for row in native14 + native32}
    all_answers = native14 + native32
    all_provenance = []
    analysis = []
    for first_receiver in ("32B", "14B"):
        schedule = f"first_receiver_{first_receiver}"
        answers, provenance = alternating_trajectory(
            pair, story, first_receiver, turns, schedule
        )
        all_answers.extend(answers)
        all_provenance.extend(provenance)
        by_key = {(row["stream"], row["turn"]): row for row in answers}
        for turn in range(1, turns + 1):
            transfer = by_key[(f"{schedule}_translated", turn)]
            shadow = by_key[(f"{schedule}_native_shadow", turn)]
            native = native_index[(transfer["receiver"], turn)]
            analysis.append(
                {
                    "schedule": schedule,
                    "conversation_id": story["id"],
                    "domain": story["source"],
                    "turn": turn,
                    "receiver": transfer["receiver"],
                    "transfer_f1": transfer["f1"],
                    "transfer_health": transfer["health_events"],
                    "shadow_f1": shadow["f1"],
                    "shadow_health": shadow["health_events"],
                    "native_only_f1": native["f1"],
                    "native_only_health": native["health_events"],
                }
            )
    return all_answers, all_provenance, analysis


def resource_receipt(stage, started, answers):
    props = torch.cuda.get_device_properties(0)
    return {
        "schema": "coqa_resource_use_v1",
        "stage": stage,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": props.total_memory / 2**30,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "answers": answers,
        "wall_seconds": time.time() - started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nvidia_driver": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }


def run_smoke(pair: ModelPair, frozen: Path, out: Path, conversations):
    rows = read_jsonl(frozen / "SMOKE_FROZEN_ROWS.jsonl")
    raw_path = out / "SMOKE_RAW.jsonl"
    raw = read_jsonl(raw_path) if raw_path.exists() else []
    completed = {row["row_id"] for row in raw}
    for i, row in enumerate(rows, 1):
        if row["row_id"] in completed:
            continue
        log(f"smoke Q1 mechanics {i}/{len(rows)} {row['row_id']}")
        direct = pair.generate_direct("32B", row["token_ids"])
        split = pair.generate_split("32B", row["token_ids"])
        source, source_s = pair.prefill("14B", row["token_ids"][:-1])
        mapped, map_s = pair.map_cache("14B", "32B", source)
        transfer = pair.generate_split("32B", row["token_ids"], mapped)
        refs = row["references"]
        raw.append(
            {
                "row_id": row["row_id"],
                "conversation_id": row["conversation_id"],
                "domain": row["domain"],
                "turn": row["turn"],
                "direct": score_result(direct, refs),
                "split": score_result(split, refs),
                "transfer": score_result(transfer, refs),
                "source_prefill_seconds": source_s,
                "mapping_seconds": map_s,
            }
        )
        write_jsonl(raw_path, raw)
        checkpoint()

    smoke_stories = [row for row in conversations if "smoke" in row["selected_for"]]
    answer_path = out / "SMOKE_Q2_RAW_ANSWERS.jsonl"
    provenance_path = out / "SMOKE_Q2_TOKEN_PROVENANCE.jsonl"
    analysis_path = out / "SMOKE_Q2_ANALYSIS.jsonl"
    q2_answers = read_jsonl(answer_path) if answer_path.exists() else []
    provenance = read_jsonl(provenance_path) if provenance_path.exists() else []
    q2_analysis = read_jsonl(analysis_path) if analysis_path.exists() else []
    q2_completed, q2_answers, provenance, q2_analysis = (
        coqa.retain_complete_q2_conversations(
            smoke_stories, q2_answers, provenance, q2_analysis, 2
        )
    )
    for i, story in enumerate(smoke_stories, 1):
        if story["id"] in q2_completed:
            continue
        turns = min(2, len(story["questions"]))
        log(f"smoke recursive mechanics {i}/{len(smoke_stories)} {story['id']}")
        answers, prov, analysis = q2_conversation(pair, story, turns)
        q2_answers.extend(answers)
        provenance.extend(prov)
        q2_analysis.extend(analysis)
        write_jsonl(answer_path, q2_answers)
        write_jsonl(provenance_path, provenance)
        write_jsonl(analysis_path, q2_analysis)
        checkpoint()

    direct_f1 = sum(row["direct"]["f1"] for row in raw) / len(raw)
    split_f1 = sum(row["split"]["f1"] for row in raw) / len(raw)
    direct_unhealthy = sum(bool(row["direct"]["health_events"]) for row in raw)
    split_unhealthy = sum(bool(row["split"]["health_events"]) for row in raw)
    material = (
        abs(direct_f1 - split_f1) > 0.03
        or abs(direct_unhealthy - split_unhealthy) / len(raw) > 0.02
    )
    result = {
        "schema": "coqa_smoke_result_v1",
        "q1_style_questions": len(raw),
        "recursive_turn_positions": len(q2_analysis),
        "answers_generated": 3 * len(raw) + len(q2_answers),
        "native_direct": {"mean_f1": direct_f1, "unhealthy": direct_unhealthy},
        "native_split": {"mean_f1": split_f1, "unhealthy": split_unhealthy},
        "native_split_material_discrepancy": material,
        "all_logits_finite": all(
            arm["finite_logits"]
            for row in raw
            for arm in (row["direct"], row["split"], row["transfer"])
        )
        and all(row["finite_logits"] for row in q2_answers),
        "cache_accounting_passed": True,
        "scorer_coverage_passed": all(
            math.isfinite(arm["f1"])
            for row in raw
            for arm in (row["direct"], row["split"], row["transfer"])
        ),
        "scores_used_for_selection": False,
    }
    result["passed"] = (
        result["all_logits_finite"]
        and result["cache_accounting_passed"]
        and result["scorer_coverage_passed"]
        and not material
    )
    write_jsonl(raw_path, raw)
    write_jsonl(answer_path, q2_answers)
    write_jsonl(provenance_path, provenance)
    write_jsonl(analysis_path, q2_analysis)
    checkpoint()
    write_json(out / "SMOKE_RESULT.json", result)
    return result, result["answers_generated"]


def run_q1(pair: ModelPair, frozen: Path, out: Path):
    inputs = read_jsonl(frozen / "Q1_FROZEN_ROWS.jsonl")
    raw_path = out / "Q1_RAW_ANSWERS.jsonl"
    rows = read_jsonl(raw_path) if raw_path.exists() else []
    completed = {row["row_id"] for row in rows}
    for i, row in enumerate(inputs, 1):
        if row["row_id"] in completed:
            continue
        log(f"Q1 {i}/{len(inputs)} {row['row_id']}")
        rows.append(q1_one(pair, row))
        if i % 10 == 0:
            write_jsonl(raw_path, rows)
            checkpoint()
    order = {row["row_id"]: i for i, row in enumerate(inputs)}
    rows.sort(key=lambda row: order[row["row_id"]])
    summary = coqa.summarize_q1(rows)
    write_jsonl(raw_path, rows)
    checkpoint()
    write_json(out / "Q1_RESULT.json", summary)
    lines = [
        "# CoQA Q1 conditional one-hop result",
        "",
        f"Development screen: **{'PROMISING' if summary['screen']['promising'] else 'UNFAVORABLE'}**.",
        "",
        f"Native pooled F1: {100 * summary['pooled_official_turn_weighted']['native_f1']:.2f}.",
        f"Transferred pooled F1: {100 * summary['pooled_official_turn_weighted']['transfer_f1']:.2f}.",
        f"Equal-domain native-minus-transfer gap: {100 * summary['equal_domain']['native_minus_transfer_f1']:.2f} points.",
        f"Observed unhealthy excess: {100 * summary['pooled_official_turn_weighted']['health_excess']:.2f} points.",
        "",
        "This is a development point screen on a disclosed reconstruction, not population noninferiority.",
    ]
    (out / "Q1_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, 2 * len(rows)


def run_q2(pair: ModelPair, out: Path, conversations):
    with open(out / "Q1_RESULT.json", encoding="utf-8") as stream:
        q1 = json.load(stream)
    if not q1["screen"]["promising"]:
        raise RuntimeError(
            "Q2 is not admitted because Q1 did not pass its point screen"
        )
    stories = [row for row in conversations if "q2" in row["selected_for"]]
    answer_path = out / "Q2_RAW_ANSWERS.jsonl"
    provenance_path = out / "Q2_TOKEN_PROVENANCE.jsonl"
    analysis_path = out / "Q2_ANALYSIS.jsonl"
    answers = read_jsonl(answer_path) if answer_path.exists() else []
    provenance = read_jsonl(provenance_path) if provenance_path.exists() else []
    analysis = read_jsonl(analysis_path) if analysis_path.exists() else []
    completed, answers, provenance, analysis = coqa.retain_complete_q2_conversations(
        stories, answers, provenance, analysis, 10
    )
    for i, story in enumerate(stories, 1):
        if story["id"] in completed:
            continue
        turns = min(10, len(story["questions"]))
        log(f"Q2 {i}/{len(stories)} {story['id']} turns={turns}")
        a, p, r = q2_conversation(pair, story, turns)
        answers.extend(a)
        provenance.extend(p)
        analysis.extend(r)
        write_jsonl(answer_path, answers)
        write_jsonl(provenance_path, provenance)
        write_jsonl(analysis_path, analysis)
        checkpoint()
    summary = coqa.summarize_q2(analysis)
    summary["analysis_rows"] = analysis
    write_jsonl(answer_path, answers)
    write_jsonl(provenance_path, provenance)
    write_jsonl(analysis_path, analysis)
    checkpoint()
    write_json(out / "Q2_RESULT.json", summary)
    lines = [
        "# CoQA Q2 repeated-handoff result",
        "",
        f"Development screen: **{'PROMISING' if summary['screen']['promising'] else 'UNFAVORABLE'}**.",
        "",
        "Each starting schedule and receiver direction is gated separately against both its same-input native shadow and matching-turn native-only trajectory.",
        "This reuses Q1 development conversations and is not an independent confirmation.",
    ]
    (out / "Q2_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, len(answers)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "q1", "q2"), required=True)
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--forward-mapper", required=True)
    parser.add_argument("--reverse-mapper")
    args = parser.parse_args(argv)

    if args.stage in ("smoke", "q2") and not args.reverse_mapper:
        parser.error(f"{args.stage} requires --reverse-mapper")
    frozen, out = Path(args.frozen_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(frozen / "FROZEN_CONVERSATIONS.json", encoding="utf-8") as stream:
        conversations = json.load(stream)

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    pair = load_pair(args.forward_mapper, args.reverse_mapper)
    if args.stage == "smoke":
        result, answer_count = run_smoke(pair, frozen, out, conversations)
    elif args.stage == "q1":
        result, answer_count = run_q1(pair, frozen, out)
    else:
        result, answer_count = run_q2(pair, out, conversations)
    resource_use = resource_receipt(args.stage, started, answer_count)
    write_json(out / f"RESOURCE_USE_{args.stage.upper()}.json", resource_use)
    if args.stage == "smoke":
        per_answer = resource_use["wall_seconds"] / max(1, answer_count)
        combined_rate = 4.54 + 4 * 0.0473 + 48 * 0.008
        estimates = {
            "q1_answers": 600,
            "conditional_q2_answers": 1200,
            "seconds_per_smoke_answer_including_model_load": per_answer,
            "q1_hours_with_30pct_contingency": per_answer * 600 * 1.3 / 3600,
            "q2_hours_with_30pct_contingency": per_answer * 1200 * 1.3 / 3600,
        }
        estimates["combined_rate_usd_per_hour"] = combined_rate
        estimates["conditional_total_usd"] = round(
            combined_rate
            * (
                estimates["q1_hours_with_30pct_contingency"]
                + estimates["q2_hours_with_30pct_contingency"]
            ),
            2,
        )
        estimates["note"] = (
            "conservative smoke-derived forecast includes model-load and extra smoke controls "
            "in every per-answer rate; replace with stage receipts after each stage"
        )
        write_json(out / "MEASURED_FORECAST.json", estimates)
    print(
        json.dumps(
            {"stage": args.stage, "result": result, "resource_use": resource_use},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
