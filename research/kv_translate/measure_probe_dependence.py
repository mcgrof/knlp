#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Measure whether document-independent probes actually depend on a KV cache.

This is a pre-training gate for behavioral cache translation.  A frozen
receiver sees the same teacher-forced probe over its correct document cache,
a shape-matched wrong-document cache, and a position-matched neutral cache.
The result says whether unrelated probes carry measurable document signal; it
does not train a translator and low probe KL is never treated as task quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.jacobian_kv.evaluate import (  # noqa: E402
    argmax_flip_rate,
    teacher_forced_kl,
)
from research.kv_translate.pairs import describe, prefill  # noqa: E402
from research.kv_translate.run_a0 import (  # noqa: E402
    continuation_logits,
    wikitext_chunks,
)

DEFAULT_TARGET = "Qwen/Qwen2.5-1.5B"
DEFAULT_TARGET_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"

# Frozen before measurement.  These continuations contain no document text,
# labels, or task answers; their token distributions are the measured channel.
UNRELATED_PROBES = (
    "The days of the week are Monday, Tuesday, Wednesday, Thursday, Friday, "
    "Saturday, and Sunday.",
    "To make a cup of tea, heat fresh water, steep the leaves, remove them, "
    "and serve the drink when it is ready.",
    "A small red kite rose above the quiet park while two friends watched it "
    "turn slowly in the afternoon wind.",
    "Addition combines quantities: two plus three equals five, while five plus "
    "eight equals thirteen.",
)


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_digest():
    return file_sha(os.path.abspath(__file__))


def distribution_distance(reference, candidate):
    kl = teacher_forced_kl(reference, candidate)
    return {
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
        "argmax_flip": argmax_flip_rate(reference, candidate),
        "logit_l2_mean": float(
            (reference.float() - candidate.float()).pow(2).mean(-1).sqrt().mean()
        ),
    }


def percentile(values, probability):
    values = sorted(float(value) for value in values)
    if not values:
        return float("nan")
    return values[max(0, math.ceil(probability * len(values)) - 1)]


def decide(rows, *, absolute_kl_min, repeat_multiplier, argmax_flip_min):
    repeats = [row["kl_mean"] for row in rows if row["comparison"] == "repeat"]
    repeat_p95 = percentile(repeats, 0.95)
    signal_floor = max(float(absolute_kl_min), repeat_multiplier * repeat_p95)
    aggregates = {}
    for probe_type in ("relevant", "unrelated"):
        for comparison in ("wrong_document", "neutral_document"):
            selected = [
                row
                for row in rows
                if row["probe_type"] == probe_type
                and row["comparison"] == comparison
            ]
            aggregates[f"{probe_type}_{comparison}"] = {
                "rows": len(selected),
                "documents": len({row["document_id"] for row in selected}),
                "kl_median": statistics.median(row["kl_mean"] for row in selected),
                "kl_p05": percentile([row["kl_mean"] for row in selected], 0.05),
                "argmax_flip_median": statistics.median(
                    row["argmax_flip"] for row in selected
                ),
                "logit_l2_median": statistics.median(
                    row["logit_l2_mean"] for row in selected
                ),
            }
    unrelated = [
        aggregates["unrelated_wrong_document"],
        aggregates["unrelated_neutral_document"],
    ]
    passed = all(
        aggregate["kl_median"] >= signal_floor
        and aggregate["argmax_flip_median"] >= argmax_flip_min
        for aggregate in unrelated
    )
    relevant_wrong = aggregates["relevant_wrong_document"]["kl_median"]
    unrelated_wrong = aggregates["unrelated_wrong_document"]["kl_median"]
    return {
        "verdict": (
            "UNRELATED_PROBE_SIGNAL_PRESENT"
            if passed
            else "UNRELATED_PROBE_SIGNAL_TOO_WEAK"
        ),
        "passes": passed,
        "repeat_kl_p95": repeat_p95,
        "absolute_kl_min": absolute_kl_min,
        "repeat_multiplier": repeat_multiplier,
        "signal_kl_floor": signal_floor,
        "argmax_flip_min": argmax_flip_min,
        "unrelated_to_relevant_wrong_kl_ratio": (
            unrelated_wrong / relevant_wrong if relevant_wrong > 0 else None
        ),
        "aggregates": aggregates,
        "note": (
            "This gate establishes a probe channel only. It does not establish "
            "cache translation quality or a subliminal-learning result."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--target-revision", default=DEFAULT_TARGET_REVISION)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--cont-len", type=int, default=64)
    parser.add_argument("--documents", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--absolute-kl-min", type=float, default=0.01)
    parser.add_argument("--repeat-multiplier", type=float, default=100.0)
    parser.add_argument("--argmax-flip-min", type=float, default=0.01)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.target, revision=args.target_revision
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.target,
        revision=args.target_revision,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    geometry = describe(model, args.target)
    chunks = wikitext_chunks(
        tokenizer, args.documents, args.ctx, args.cont_len, args.seed
    )

    neutral_ids = tokenizer(
        "\n", return_tensors="pt", add_special_tokens=False
    ).input_ids[0]
    if not len(neutral_ids):
        raise RuntimeError("neutral prefix tokenization is empty")
    neutral_ids = neutral_ids.repeat(math.ceil(args.ctx / len(neutral_ids)))[
        : args.ctx
    ].to(device)
    neutral = prefill(model, neutral_ids.unsqueeze(0), geometry)

    native = []
    relevant = []
    for document_id, ids in chunks:
        ids = ids.to(device)
        native.append(prefill(model, ids[: args.ctx].unsqueeze(0), geometry))
        relevant.append(ids[args.ctx : args.ctx + args.cont_len].unsqueeze(0))
    unrelated = [
        tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(
            device
        )
        for text in UNRELATED_PROBES
    ]
    if any(ids.shape[1] < 2 for ids in unrelated):
        raise RuntimeError("each unrelated probe must contain at least two tokens")

    rows = []
    rows_path = os.path.join(args.out_dir, "rows.jsonl")

    def record(row):
        rows.append(row)
        with open(rows_path, "a") as output:
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())

    for index, ((document_id, _), correct_cache, relevant_ids) in enumerate(
        zip(chunks, native, relevant)
    ):
        wrong_cache = native[(index + 1) % len(native)]
        probes = [("relevant", "document_continuation", relevant_ids)] + [
            ("unrelated", f"fixed_{probe_id}", ids)
            for probe_id, ids in enumerate(unrelated)
        ]
        for probe_type, probe_id, probe_ids in probes:
            correct_logits = continuation_logits(
                model,
                correct_cache.keys_post,
                correct_cache.values,
                probe_ids,
                args.ctx,
            ).float()
            for comparison, cache in (
                ("wrong_document", wrong_cache),
                ("neutral_document", neutral),
            ):
                candidate = continuation_logits(
                    model,
                    cache.keys_post,
                    cache.values,
                    probe_ids,
                    args.ctx,
                ).float()
                record(
                    {
                        "document_id": int(document_id),
                        "wrong_document_id": int(chunks[(index + 1) % len(chunks)][0]),
                        "probe_type": probe_type,
                        "probe_id": probe_id,
                        "probe_tokens": int(probe_ids.shape[1]),
                        "probe_sha256": hashlib.sha256(
                            probe_ids.detach().cpu().numpy().astype("<i8").tobytes()
                        ).hexdigest(),
                        "comparison": comparison,
                        **distribution_distance(correct_logits, candidate),
                    }
                )
                del candidate
            if probe_type == "unrelated" and probe_id == "fixed_0":
                repeated = continuation_logits(
                    model,
                    correct_cache.keys_post,
                    correct_cache.values,
                    probe_ids,
                    args.ctx,
                ).float()
                record(
                    {
                        "document_id": int(document_id),
                        "wrong_document_id": None,
                        "probe_type": probe_type,
                        "probe_id": probe_id,
                        "probe_tokens": int(probe_ids.shape[1]),
                        "probe_sha256": hashlib.sha256(
                            probe_ids.detach().cpu().numpy().astype("<i8").tobytes()
                        ).hexdigest(),
                        "comparison": "repeat",
                        **distribution_distance(correct_logits, repeated),
                    }
                )
                del repeated
            del correct_logits

    decision = decide(
        rows,
        absolute_kl_min=args.absolute_kl_min,
        repeat_multiplier=args.repeat_multiplier,
        argmax_flip_min=args.argmax_flip_min,
    )
    receipt = {
        "schema": "kv_probe_dependence_v1",
        "source_digest": source_digest(),
        "target": args.target,
        "target_revision": args.target_revision,
        "tokenizer_revision": args.target_revision,
        "dtype": args.dtype,
        "device_type": torch.device(device).type,
        "torch": torch.__version__,
        "ctx": args.ctx,
        "continuation_tokens": args.cont_len,
        "documents": args.documents,
        "document_ids": [int(document_id) for document_id, _ in chunks],
        "seed": args.seed,
        "unrelated_probe_text_sha256": [
            hashlib.sha256(text.encode()).hexdigest() for text in UNRELATED_PROBES
        ],
        "neutral_prefix": "newline token sequence repeated to the context length",
        "wrong_document_pairing": "next frozen document, circular",
        "decision": decision,
        "wall_seconds": round(time.time() - started, 3),
    }
    with open(os.path.join(args.out_dir, "RESULT.json"), "w") as output:
        json.dump(receipt, output, indent=2, sort_keys=True)
    print(decision["verdict"], json.dumps(decision["aggregates"], sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
