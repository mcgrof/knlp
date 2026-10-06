#!/usr/bin/env python3
"""Evaluate retained-span KV-Lingo switching on the frozen CoQA cohorts."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from . import coqa
from .capture import make_cache
from .coqa_runner import cache_length, ids_sha256, write_json, write_jsonl
from .kv_lingo import forward_and_capture_pre_norm_span
from .kv_lingo import CacheGeometry, LinearTranslator
from .kv_lingo_retained import SpanOwnershipLedger, append_translated_span
from .kv_lingo_train import (
    PINS,
    load_models,
    translator_from_artifact,
)


def load_translator(path: Path, device):
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema") == "kv_lingo_stage2_checkpoint_v1":
        sg, tg = value["source_geometry"], value["target_geometry"]
        translator = LinearTranslator(
            CacheGeometry(sg["layers"], sg["kv_heads"], sg["head_dim"]),
            CacheGeometry(tg["layers"], tg["kv_heads"], tg["head_dim"]),
        )
        translator = translator.to(device)
        translator.load_state_dict(value["translator"])
        identity = f"stage2-step{value['global_step']}"
    else:
        translator = translator_from_artifact(value, device)
        identity = value["stage"]
    translator.eval()
    return translator, identity


def clean_decode(tokenizer, token_ids: list[int]):
    kwargs = {"clean_up_tokenization_spaces": False}
    raw = tokenizer.decode(token_ids, skip_special_tokens=False, **kwargs)
    visible = tokenizer.decode(token_ids, skip_special_tokens=True, **kwargs)
    answer = visible.split("\n", 1)[0].strip()
    return raw, answer


def should_stop(tokenizer, generated: list[int], max_tokens: int):
    raw, _answer = clean_decode(tokenizer, generated)
    if generated[-1] == tokenizer.eos_token_id:
        return "eos"
    if "\n" in raw:
        return "newline"
    if len(generated) >= max_tokens:
        return "cap"
    return None


class RetainedPair:
    def __init__(self, tokenizer, models, translators):
        self.tokenizer = tokenizer
        self.models = {"4B": models["4b"], "8B": models["8b"]}
        self.translators = translators
        self.device = next(models["4b"].parameters()).device
        self.reset()

    def reset(self):
        self.caches = {"4B": None, "8B": None}
        self.ledger = SpanOwnershipLedger(("4B", "8B"), 0)
        self.auxiliary = {}
        self.bytes_translated = 0
        self.tokens_translated = 0

    def _append_native(self, model_name: str, token_ids: list[int]):
        if not token_ids:
            raise ValueError("native append cannot be empty")
        if self.ledger.line_covered[model_name] != self.ledger.total_tokens:
            raise RuntimeError("model must catch up before writing native tokens")
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        output, auxiliary = forward_and_capture_pre_norm_span(
            self.models[model_name],
            ids,
            past_key_values=self.caches[model_name],
        )
        span = self.ledger.write(model_name, len(token_ids))
        if auxiliary.tokens != len(token_ids):
            raise RuntimeError("capture does not match appended token count")
        self.auxiliary[span.serial] = auxiliary
        self.caches[model_name] = output.past_key_values
        if cache_length(self.caches[model_name]) != span.end:
            raise RuntimeError("native cache length differs from ownership ledger")
        return output.logits

    def catch_up(self, receiver: str):
        spans = self.ledger.missing(receiver)
        if not spans:
            return {"spans": 0, "tokens": 0, "seconds": 0.0}
        started = time.perf_counter()
        tokens = 0
        for span in spans:
            donor = span.writer
            direction = f"{donor.lower()}-to-{receiver.lower()}"
            translator = self.translators[direction]
            auxiliary = self.auxiliary[span.serial]
            positions = torch.arange(
                span.start, span.end, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            # Stage 2 applies the FP32 translator parameters under BF16
            # autocast.  Retained inference must use the same compute dtype;
            # otherwise BF16 captured spans and FP32 weights cannot be
            # multiplied outside an autocast region.
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                keys, values = translator(auxiliary, self.models[receiver], positions)
                if self.caches[receiver] is None:
                    self.caches[receiver] = make_cache(
                        list(zip(keys, values, strict=True))
                    )
                else:
                    self.caches[receiver] = append_translated_span(
                        self.caches[receiver], keys, values
                    )
            tokens += span.end - span.start
            self.bytes_translated += sum(
                tensor.numel() * tensor.element_size() for tensor in [*keys, *values]
            )
            del auxiliary, keys, values
            self.auxiliary.pop(span.serial)
        self.ledger.translated(receiver, spans)
        if cache_length(self.caches[receiver]) != self.ledger.total_tokens:
            raise RuntimeError("translated receiver cache does not reach the ledger")
        self.tokens_translated += tokens
        torch.cuda.synchronize()
        return {
            "spans": len(spans),
            "tokens": tokens,
            "seconds": time.perf_counter() - started,
        }

    def generate(self, model_name: str, suffix_ids: list[int], max_tokens: int):
        catch_up = self.catch_up(model_name)
        logits = self._append_native(model_name, suffix_ids)
        generated = []
        started = time.perf_counter()
        stop = None
        while stop is None:
            scores = logits[0, -1].float()
            if not torch.isfinite(scores).all():
                raise FloatingPointError(f"{model_name} produced nonfinite logits")
            token = int(scores.argmax())
            generated.append(token)
            stop = should_stop(self.tokenizer, generated, max_tokens)
            # The sampled token is absent from the cache which emitted it.
            # Append it even at a stop so the next receiver can translate it.
            logits = self._append_native(model_name, [token])
        torch.cuda.synchronize()
        raw, answer = clean_decode(self.tokenizer, generated)
        return {
            "answer": answer,
            "raw_text": raw,
            "token_ids": generated,
            "stop_reason": stop,
            "health_events": coqa.health_events(
                answer, raw, stop_reason=stop, new_tokens=len(generated)
            ),
            "generation_seconds": time.perf_counter() - started,
            "catch_up": catch_up,
        }


@torch.inference_mode()
def native_generate(tokenizer, model, token_ids: list[int], max_tokens: int):
    device = next(model.parameters()).device
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    output = model(input_ids=ids, use_cache=True)
    cache = output.past_key_values
    logits = output.logits
    generated = []
    stop = None
    while stop is None:
        scores = logits[0, -1].float()
        if not torch.isfinite(scores).all():
            raise FloatingPointError("native reference produced nonfinite logits")
        token = int(scores.argmax())
        generated.append(token)
        stop = should_stop(tokenizer, generated, max_tokens)
        if stop is None:
            output = model(
                input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
                past_key_values=cache,
                use_cache=True,
            )
            cache, logits = output.past_key_values, output.logits
    raw, answer = clean_decode(tokenizer, generated)
    return {
        "answer": answer,
        "raw_text": raw,
        "token_ids": generated,
        "stop_reason": stop,
        "health_events": coqa.health_events(
            answer, raw, stop_reason=stop, new_tokens=len(generated)
        ),
    }


def score_fields(prefix: str, result: dict, references: list[str]):
    score = coqa.turn_score(references, result["answer"])
    return {
        f"{prefix}_answer": result["answer"],
        f"{prefix}_raw_text": result["raw_text"],
        f"{prefix}_token_ids": result["token_ids"],
        f"{prefix}_stop_reason": result["stop_reason"],
        f"{prefix}_health": result["health_events"],
        f"{prefix}_f1": score["f1"],
        f"{prefix}_em": score["em"],
    }


def suffix_for(story: dict, turn: int, prior_stop: str):
    key = "after_eos" if prior_stop == "eos" else "after_other"
    return list(story["next_turn_suffix_token_ids"][str(turn)][key])


def translated_trajectory(pair: RetainedPair, story: dict, first: str, turns: int):
    pair.reset()
    ledger = []
    rows = []
    prior_stop = None
    for turn in range(1, turns + 1):
        receiver = first if turn % 2 else ("8B" if first == "4B" else "4B")
        suffix = (
            list(story["turn_1_token_ids"])
            if turn == 1
            else suffix_for(story, turn, prior_stop)
        )
        ledger.extend(suffix)
        prompt_hash = ids_sha256(ledger)
        native_shadow = native_generate(
            pair.tokenizer, pair.models[receiver], ledger, coqa.MAX_NEW_TOKENS
        )
        translated = pair.generate(receiver, suffix, coqa.MAX_NEW_TOKENS)
        if translated["token_ids"]:
            ledger.extend(translated["token_ids"])
        references = coqa.answer_references(story, turn)
        rows.append(
            {
                "stream": "translated",
                "conversation_id": story["id"],
                "domain": story["source"],
                "starting_model": first,
                "turn": turn,
                "receiver": receiver,
                "prompt_token_ids_sha256": prompt_hash,
                **score_fields("translated", translated, references),
                **score_fields("same_transcript_native", native_shadow, references),
                "catch_up": translated["catch_up"],
                "tokens_translated_cumulative": pair.tokens_translated,
                "bytes_translated_cumulative": pair.bytes_translated,
                "cache_line_tokens": dict(pair.ledger.line_covered),
            }
        )
        prior_stop = translated["stop_reason"]
    receipt = pair.ledger.receipt()
    receipt["remaining_auxiliary_spans"] = sorted(pair.auxiliary)
    return rows, receipt


def native_alternating(tokenizer, models, story: dict, first: str, turns: int):
    ledger = []
    rows = []
    prior_stop = None
    for turn in range(1, turns + 1):
        receiver = first if turn % 2 else ("8B" if first == "4B" else "4B")
        suffix = (
            list(story["turn_1_token_ids"])
            if turn == 1
            else suffix_for(story, turn, prior_stop)
        )
        ledger.extend(suffix)
        prompt_hash = ids_sha256(ledger)
        result = native_generate(
            tokenizer, models[receiver.lower()], ledger, coqa.MAX_NEW_TOKENS
        )
        ledger.extend(result["token_ids"])
        references = coqa.answer_references(story, turn)
        rows.append(
            {
                "stream": "alternating_native_reprefill",
                "conversation_id": story["id"],
                "domain": story["source"],
                "starting_model": first,
                "turn": turn,
                "receiver": receiver,
                "prompt_token_ids_sha256": prompt_hash,
                **score_fields("native_trajectory", result, references),
            }
        )
        prior_stop = result["stop_reason"]
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=Path, required=True)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--reverse", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B",
        revision=PINS["Qwen/Qwen3-4B"],
        local_files_only=True,
    )
    models = load_models("cuda")
    forward, forward_identity = load_translator(args.forward, "cuda")
    reverse, reverse_identity = load_translator(args.reverse, "cuda")
    pair = RetainedPair(
        tokenizer,
        models,
        {"4b-to-8b": forward, "8b-to-4b": reverse},
    )
    stories = json.loads(args.conversations.read_text())
    if args.limit is not None:
        stories = stories[: args.limit]
    raw_path = args.out / "RAW.jsonl"
    ownership_path = args.out / "OWNERSHIP.jsonl"
    native_path = args.out / "NATIVE_TRAJECTORY.jsonl"
    raw = []
    ownership = []
    native = []
    if raw_path.exists():
        raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    if ownership_path.exists():
        ownership = [
            json.loads(line) for line in ownership_path.read_text().splitlines() if line
        ]
    if native_path.exists():
        native = [
            json.loads(line) for line in native_path.read_text().splitlines() if line
        ]
    completed = {(row["conversation_id"], row["starting_model"]) for row in ownership}
    started = time.time()
    for story in stories:
        for first in ("4B", "8B"):
            if (story["id"], first) in completed:
                continue
            translated, receipt = translated_trajectory(pair, story, first, args.turns)
            native_rows = native_alternating(
                tokenizer, models, story, first, args.turns
            )
            raw.extend(translated)
            native.extend(native_rows)
            ownership.append(
                {
                    "conversation_id": story["id"],
                    "domain": story["source"],
                    "starting_model": first,
                    **receipt,
                }
            )
            write_jsonl(raw_path, raw)
            write_jsonl(native_path, native)
            write_jsonl(ownership_path, ownership)
            logline = f"{story['id']} first={first}: {len(translated)} turns complete"
            print(logline, flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    result = {
        "schema": "kv_lingo_retained_coqa_v1",
        "conversations": len(stories),
        "turns": args.turns,
        "starting_models": ["4B", "8B"],
        "forward": {"path": str(args.forward), "identity": forward_identity},
        "reverse": {"path": str(args.reverse), "identity": reverse_identity},
        "raw_rows": len(raw),
        "native_trajectory_rows": len(native),
        "ownership_rows": len(ownership),
        "wall_seconds_this_invocation": time.time() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    write_json(args.out / "RESULT.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
