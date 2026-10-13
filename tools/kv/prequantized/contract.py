"""Immutable inputs, append-only results, and document-paired statistics."""

import hashlib
import gzip
import json
import math
import re
from pathlib import Path

from tools.kv.fp8_failure.statistics import paired_bootstrap_delta

ARMS = ("k16_v16", "k8_v16", "k16_v8", "k8_v8")
GATES = tuple(f"T{i:02d}" for i in range(6))


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if read(path) != value:
            raise ValueError(f"refusing to replace frozen artifact: {path}")
        return
    if path.suffix == ".gz":
        with path.open("xb") as stream:
            stream.write(gzip.compress(content.encode(), mtime=0))
    else:
        with path.open("x") as stream:
            stream.write(content)


def pinned(revision):
    if not isinstance(revision, str) or not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full immutable 40-character commit")
    return revision


def validate_matrix(matrix):
    if matrix.get("schema_version") != 1:
        raise ValueError("unsupported matrix schema")
    if tuple(arm["id"] for arm in matrix["arms"]) != ARMS:
        raise ValueError("four predeclared arms required")
    q = matrix["quantizer"]
    expected = {
        "format": "fp8_e4m3fn",
        "finite_max": 448,
        "rounding": "nearest_ties_even",
        "scale_granularity": "per_layer_separate_k_v",
        "scale_arithmetic_dtype": "float32",
        "scale_method": "calibration_max_abs_div_448",
        "zero_tensor_scale": 1,
        "write_roundtrip_once": True,
        "quantize_non_kv_activations": False,
        "requantize_weights": False,
        "frozen_before_evaluation": True,
        "calibration_prefixes_per_length": 16,
        "calibration_decode_tokens": 128,
    }
    if any(q.get(key) != value for key, value in expected.items()):
        raise ValueError("matrix changes the implemented numerical recipe")
    if matrix["quality"]["input_tokens"] != [2048, 8192]:
        raise ValueError("the study requires both predeclared lengths")
    quality = {
        "nll_documents_per_length": 32,
        "nll_scored_cached_decode_tokens_per_document": 128,
        "nll_exclude_prefill_prediction": True,
        "retrieval_cases_per_length": 100,
        "retrieval_depth_fractions": [0.1, 0.3, 0.5, 0.7, 0.9],
        "retrieval_min_native_accuracy": 0.8,
        "generation_prompts": 32,
        "bootstrap_resamples": 2000,
        "bootstrap_unit": "document",
        "confidence": 0.95,
        "max_relative_ppl": 1.01,
        "max_retrieval_loss_absolute": 0.05,
    }
    if any(matrix["quality"].get(key) != value for key, value in quality.items()):
        raise ValueError("matrix changes the implemented sampling or decision rule")
    if matrix["seeds"] != {
        "selection": 20261013,
        "calibration": 20261014,
        "bootstrap": 20261015,
    }:
        raise ValueError("matrix changes the implemented seeds")


def validate_manifest(manifest):
    for name in (
        "model_revision",
        "tokenizer_revision",
        "dataset_revision",
        "source_sha",
    ):
        pinned(manifest[name])
    if digest(manifest["samples"]) != manifest["sample_hash"]:
        raise ValueError("sample manifest hash mismatch")
    ids = [row["id"] for row in manifest["samples"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample IDs")
    if set(manifest["calibration_article_ids"]) & set(
        manifest["evaluation_article_ids"]
    ):
        raise ValueError("calibration and evaluation overlap")


def completed_rows(path, provenance):
    rows = {}
    path = Path(path)
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["provenance"] != provenance:
                raise ValueError("resume provenance mismatch")
            if row["sample_id"] in rows:
                raise ValueError("duplicate completed sample")
            rows[row["sample_id"]] = row
    return rows


def append_result(path, row, completed):
    if row["sample_id"] in completed:
        raise ValueError("refusing to overwrite a completed sample")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
    completed[row["sample_id"]] = row


def paired_nll(arm, baseline, seed=20261015):
    if set(arm) != set(baseline) or not arm:
        raise ValueError("NLL requires identical nonempty document IDs")
    keys = sorted(arm)
    # alpha=.10 gives a percentile upper endpoint at .95 (one-sided).
    delta, lower, upper = paired_bootstrap_delta(
        [arm[key] for key in keys],
        [baseline[key] for key in keys],
        n_boot=2000,
        alpha=0.10,
        seed=seed,
    )
    return {
        "documents": len(keys),
        "delta_nll": delta,
        "lower_95_one_sided": lower,
        "upper_95_one_sided": upper,
        "relative_ppl": math.exp(delta),
        "supported_margin": upper <= math.log(1.01),
    }


def paired_retrieval(arm, baseline):
    from scipy.stats import beta

    if set(arm) != set(baseline) or not arm:
        raise ValueError("retrieval requires identical nonempty case IDs")
    n = len(arm)
    losses = sum(bool(baseline[key]) and not arm[key] for key in arm)
    gains = sum(bool(arm[key]) and not baseline[key] for key in arm)
    upper = 1.0 if losses == n else float(beta.ppf(0.95, losses + 1, n - losses))

    def bounds(events):
        lower = 0.0 if events == 0 else float(beta.ppf(0.025, events, n - events + 1))
        upper = 1.0 if events == n else float(beta.ppf(0.975, events + 1, n - events))
        return lower, upper

    # A union bound on the two transition probabilities gives a valid
    # one-sided 95% bound for their difference, without a zero-error bootstrap.
    loss_lower, loss_upper = bounds(losses)
    gain_lower, gain_upper = bounds(gains)
    accuracy = sum(bool(x) for x in baseline.values()) / n
    return {
        "cases": n,
        "baseline_accuracy": accuracy,
        "arm_accuracy": sum(bool(x) for x in arm.values()) / n,
        "correct_to_wrong": losses,
        "wrong_to_correct": gains,
        "accuracy_loss": (losses - gains) / n,
        "loss_upper_95_exact": upper,
        "accuracy_loss_lower_95_one_sided": loss_lower - gain_upper,
        "accuracy_loss_upper_95_one_sided": loss_upper - gain_lower,
        "accuracy_difference_interval_method": "Bonferroni exact transition-probability bounds",
        "supported_margin": accuracy >= 0.8 and upper <= 0.05,
    }
