"""Reject mixed, incomplete, or failed evidence before computing a verdict."""

import math

from tools.kv.prequantized.contract import (
    ARMS,
    GATES,
    digest,
    validate_manifest,
    validate_matrix,
)
from tools.kv.prequantized.recipes import manifest_recipe


def validate_execution_manifest(manifest, source):
    validate_manifest(manifest)
    validate_matrix(manifest["matrix"])
    if manifest["source_sha"] != source:
        raise ValueError("execution source differs from the frozen manifest")
    if manifest["matrix_hash"] != digest(manifest["matrix"]):
        raise ValueError("matrix hash mismatch")
    settings = next(
        x for x in manifest["matrix"]["models"] if x["id"] == manifest["model"]
    )
    for key in ("activation_dtype", "max_output_tokens"):
        if settings[key] != manifest[key]:
            raise ValueError(f"model {key} differs from the matrix")
    manifest_recipe(manifest)
    for sample in manifest["samples"]:
        if sample["prompt_hash"] != digest(sample["token_ids"]):
            raise ValueError("sample token hash mismatch")
        if sample["kind"] in ("nll", "calibration"):
            if len(sample["token_ids"]) != sample["length"] + 129:
                raise ValueError("NLL/calibration window length mismatch")
            split = (
                "calibration_article_ids"
                if sample["kind"] == "calibration"
                else "evaluation_article_ids"
            )
            if sample["article_id"] not in manifest[split]:
                raise ValueError("document is outside its declared split")
        elif len(sample["token_ids"]) != sample["length"]:
            raise ValueError("prompt token length mismatch")
        if sample.get("generator_version") == 3:
            if not 0 < sample["catalogue_tokens"] < sample["length"]:
                raise ValueError("invalid deferred-query cache split")
            if sample["query_tokens"] + sample["catalogue_tokens"] != sample["length"]:
                raise ValueError("query/catalogue lengths differ from full prompt")
    for length in (2048, 8192):
        for kind, expected in (("nll", 32), ("retrieval", 100), ("calibration", 16)):
            chosen = [
                s
                for s in manifest["samples"]
                if s["kind"] == kind and s["length"] == length and not s.get("reserve")
            ]
            if len(chosen) != expected:
                raise ValueError(f"incomplete manifest: {kind} at {length}")


def validate_gates(receipts, base):
    if set(receipts) != set(GATES):
        raise ValueError("missing gate receipts")
    for name, item in receipts.items():
        if (
            item.get("gate") != name
            or item.get("status") != "passed"
            or item.get("provenance") != base
        ):
            raise ValueError(f"failed or mismatched gate: {name}")


def validate_scales(data, base, manifest):
    expected = [s["id"] for s in manifest["samples"] if s["kind"] == "calibration"]
    if (
        data["provenance"] != base
        or digest(data["scales"]) != data["scale_hash"]
        or data["sample_ids"] != expected
    ):
        raise ValueError("calibration provenance, samples, or hash mismatch")
    if not data["scales"] or any(
        not math.isfinite(x) or x <= 0 for x in data["scales"].values()
    ):
        raise ValueError("invalid frozen scale")


def row_provenance(base, manifest, arm, scale_hash):
    return {
        **base,
        "arm": arm,
        "scale_hash": scale_hash,
        "measurement_level": "fake_quant_cached_decode",
        "weights_runtime": "packed",
        "kv_storage": "16bit_qdq",
        "activation_dtype": manifest["activation_dtype"],
        "cache_recipe": manifest_recipe(manifest),
    }


def expected_samples(manifest):
    return {
        s["id"]: s
        for s in manifest["samples"]
        if s["kind"] in ("nll", "retrieval", "generation") and not s.get("reserve")
    }


def validate_row(row, sample, prov):
    if row.get("provenance") != prov or row.get("status") != "completed":
        raise ValueError("failed sample or row provenance mismatch")
    for key, value in (
        ("sample_id", sample["id"]),
        ("prompt_hash", sample["prompt_hash"]),
        ("kind", sample["kind"]),
        ("context_tokens", sample["length"]),
    ):
        if row.get(key) != value:
            raise ValueError(f"sample metadata mismatch: {key}")
    receipt = row["cache_receipt"]
    if not receipt["attention_reads"] or any(
        v["nonfinite"] for v in receipt["quantization"].values()
    ):
        raise ValueError("missing attention proof or nonfinite cache")
    if "model" in prov:
        layers = 36 if prov["model"] == "Qwen/Qwen3-8B-AWQ" else 24
        expected_layers = {str(i) for i in range(layers)}
        reads = {str(k): v for k, v in receipt["attention_reads"].items()}
        writes = {str(k): v for k, v in receipt["cache"].items()}
        if set(reads) != expected_layers or set(writes) != expected_layers:
            raise ValueError("cache proof does not cover every model layer")
        if any(
            reads[k] != writes[k]["past_tokens_read"] or writes[k]["calls"] < 1
            for k in reads
        ):
            raise ValueError("attention reads differ from observed cache updates")
        kinds = (
            {"k"}
            if prov["arm"] == "k8_v16"
            else (
                {"v"}
                if prov["arm"] == "k16_v8"
                else {"k", "v"} if prov["arm"] == "k8_v8" else set()
            )
        )
        expected_components = {
            f"{layer}.{kind}" for layer in range(layers) for kind in kinds
        }
        if set(receipt["quantization"]) != expected_components:
            raise ValueError("quantized component inventory differs from the arm")
    if sample["kind"] == "nll":
        losses = row["token_nll"]
        if (
            row["article_id"] != sample["article_id"]
            or row["scored_tokens"] != 128
            or len(losses) != 128
            or not all(math.isfinite(x) for x in losses)
            or not math.isfinite(row["mean_nll"])
        ):
            raise ValueError("invalid NLL row")
        if abs(sum(losses) / 128 - row["mean_nll"]) > 2e-5:
            raise ValueError("NLL mean differs from token losses")
    else:
        for key in ("correct", "malformed", "truncated"):
            if type(row[key]) is not bool:
                raise ValueError("invalid generation outcome")
        if sample["kind"] == "retrieval":
            expected = (
                not row["truncated"]
                and not row["malformed"]
                and row["final"] == sample["answer"]
            )
            if row["correct"] != expected:
                raise ValueError("retrieval score differs from exact answer")


def validate_results(rows, manifest, base, scales, complete=True):
    samples = expected_samples(manifest)
    by_arm = {arm: {} for arm in ARMS}
    for row in rows:
        arm = row["provenance"].get("arm")
        key = row["sample_id"]
        if arm not in by_arm or key not in samples or key in by_arm[arm]:
            raise ValueError("unknown arm/sample or ambiguous duplicate row")
        validate_row(
            row, samples[key], row_provenance(base, manifest, arm, scales["scale_hash"])
        )
        by_arm[arm][key] = row
    if complete and any(set(entries) != set(samples) for entries in by_arm.values()):
        raise ValueError("missing samples; no complete verdict can be computed")
    return by_arm
