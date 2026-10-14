"""Prepare and qualify deferred-query cache retrieval without changing old rows."""

import argparse
import copy
import json
import random
from pathlib import Path

import runner
import torch

from tools.kv.prequantized.cache import Observer, make_cache
from tools.kv.prequantized.contract import digest, read, validate_manifest, write_once
from tools.kv.prequantized.deferred import fixture, query_variant
from tools.kv.prequantized.fixtures import WORDS
from tools.kv.prequantized.recipes import FRESH, WRITE_FIRST, recipe
from tools.kv.prequantized.validation import validate_execution_manifest
from tools.kv.prequantized.sensitivity import CRITERION, sensitivity_verdict


def prepare(args):
    from transformers import AutoTokenizer

    source = runner.source_sha()
    old = read(args.old_manifest)
    validate_manifest(old)
    tokenizer = AutoTokenizer.from_pretrained(
        old["model"], revision=old["tokenizer_revision"]
    )
    manifest = copy.deepcopy(old)
    manifest["schema_version"] = 2
    manifest["source_sha"] = source
    manifest["parent_manifest_hash"] = digest(old)
    manifest["parent_source_sha"] = old["source_sha"]
    manifest["matrix"]["retrieval_family"] = "retrieval_v3_deferred_query"
    manifest["matrix"]["quality"]["generation_prompts"] = 0
    manifest["matrix"]["seeds"].update(
        retrieval_development=20261016, retrieval_evaluation=20261017
    )
    manifest["matrix_hash"] = digest(manifest["matrix"])
    manifest["cache_recipe"] = recipe(args.recipe)
    manifest["current_token_semantics"] = manifest["cache_recipe"][
        "current_token_decode"
    ]
    manifest["sensitivity_criterion"] = CRITERION
    samples = [s for s in manifest["samples"] if s["kind"] in ("calibration", "nll")]
    for length in (2048, 8192):
        for index in range(5):
            samples.append(
                fixture(tokenizer, old["model"], length, index, 20261016, True)
            )
        for index in range(100):
            samples.append(fixture(tokenizer, old["model"], length, index, 20261017))
    manifest["samples"] = samples
    manifest["sample_hash"] = digest(samples)
    if runner.source_sha() != source:
        raise ValueError("source changed during preparation")
    validate_execution_manifest(manifest, source)
    write_once(args.manifest, manifest)
    print(
        json.dumps(
            {
                "manifest_hash": digest(manifest),
                "samples": len(samples),
                "recipe": args.recipe,
            }
        ),
        flush=True,
    )


def cache_geometry(cache):
    return {
        "seq_length": int(cache.get_seq_length()),
        "layers": [
            {
                "key_shape": list(layer.keys.shape),
                "value_shape": list(layer.values.shape),
                "cumulative_length": int(
                    getattr(layer, "cumulative_length", layer.keys.shape[-2])
                ),
            }
            for layer in cache.layers
        ],
    }


def cache_hash(cache):
    import hashlib

    result = hashlib.sha256()
    for layer in cache.layers:
        for tensor in (layer.keys, layer.values):
            result.update(
                tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            )
    return result.hexdigest()


@torch.inference_mode()
def sensitivity(args, manifest):
    model, tokenizer = runner.load(manifest)
    prov = runner.provenance(manifest)
    runner.require_gates(args, prov)
    rows = []
    row_path = Path(args.output) / "sensitivity-rows.jsonl"
    if row_path.exists():
        raise ValueError("use a new output directory rather than overwrite controls")
    for sample in (s for s in manifest["samples"] if s["kind"] == "dev"):
        obs = Observer()
        cached = make_cache(model.config, obs)
        runner.forward(model, sample["token_ids"][: sample["catalogue_tokens"]], cached)
        geometry = cache_geometry(cached)
        original_hash = cache_hash(cached)
        indices = [sample["target_index"]]
        for index in (0, len(sample["records"]) // 2, len(sample["records"]) - 1):
            if index not in indices:
                indices.append(index)
            if len(indices) == 3:
                break
        controls = []
        for index in indices:
            variant = query_variant(tokenizer, manifest["model"], sample, index)
            if (
                variant["token_ids"][: variant["catalogue_tokens"]]
                != sample["token_ids"][: sample["catalogue_tokens"]]
            ):
                raise ValueError("question-only intervention changed catalogue tokens")
            controls.append(("query", variant, copy.deepcopy(cached)))
        zero = copy.deepcopy(cached)
        for layer in zero.layers:
            layer.keys.zero_()
            layer.values.zero_()
        if cache_geometry(zero) != geometry:
            raise RuntimeError("zeroing catalogue altered positions or geometry")
        controls.append(("zero", sample, zero))
        rng = random.Random(sample["seed"] + sample["length"] + sample["target_index"])
        for _ in range(1000):
            replacement = " ".join(rng.sample(WORDS, 3))
            if replacement in sample["catalogue"]:
                continue
            changed = query_variant(
                tokenizer,
                manifest["model"],
                sample,
                sample["target_index"],
                replacement,
            )
            if (
                changed["catalogue_tokens"] == sample["catalogue_tokens"]
                and changed["length"] == sample["length"]
            ):
                break
        else:
            raise ValueError("no same-length value intervention")
        changed_cache = make_cache(model.config, Observer())
        runner.forward(
            model, changed["token_ids"][: changed["catalogue_tokens"]], changed_cache
        )
        if cache_geometry(changed_cache) != geometry:
            raise RuntimeError("changed-value intervention altered geometry")
        controls.append(("changed", changed, changed_cache))
        controls.append(
            (
                "concatenated",
                {k: v for k, v in sample.items() if k != "catalogue_tokens"},
                None,
            )
        )
        for condition, variant, cache in controls:
            condition_hash = cache_hash(cache) if cache is not None else None
            result = runner.generate(
                model, tokenizer, variant, manifest, Observer(), catalogue_cache=cache
            )
            row = {
                "sample_id": sample["id"],
                "length": sample["length"],
                "condition": condition,
                "target_key": variant["target_key"],
                "answer": variant["answer"],
                "original_target": variant["target_key"] == sample["target_key"],
                "prompt_hash": variant["prompt_hash"],
                "geometry_before_query": geometry,
                "catalogue_snapshot_hash": original_hash,
                "condition_cache_hash_before_query": condition_hash,
                "result": result,
                "provenance": prov,
            }
            rows.append(row)
            with row_path.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        "sample_id": sample["id"],
                        "condition": condition,
                        "correct": result["correct"],
                    }
                ),
                flush=True,
            )
        if cache_geometry(cached) != geometry or cache_hash(cached) != original_hash:
            raise RuntimeError("query mutated the reusable catalogue snapshot")
    verdict = sensitivity_verdict(rows, manifest["sensitivity_criterion"])
    write_once(
        Path(args.output) / "sensitivity.json",
        {
            **verdict,
            "provenance": prov,
            "criterion": manifest["sensitivity_criterion"],
            "rows_hash": digest(rows),
            "cache_copy_and_position_checks": "passed",
        },
    )
    if verdict["status"] != "passed":
        raise RuntimeError(
            "deferred-query workload failed its frozen sensitivity criterion"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "sensitivity", "drift"))
    parser.add_argument("--old-manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recipe", choices=(FRESH, WRITE_FIRST), default=FRESH)
    args = parser.parse_args()
    runner.threads()
    if args.stage == "prepare":
        prepare(args)
        return
    manifest = read(args.manifest)
    validate_execution_manifest(manifest, runner.source_sha())
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.stage == "drift":
        from tools.kv.prequantized.drift import diagnose

        diagnose(args, manifest, runner)
    else:
        sensitivity(args, manifest)


if __name__ == "__main__":
    main()
