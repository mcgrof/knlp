# SPDX-License-Identifier: MIT
"""Require complete native quality and GSM8K evidence before deployment timing."""

import argparse
import gzip
import math
import hashlib
import json
from pathlib import Path

from tools.kv.prequantized.contract import paired_nll, paired_retrieval
from .gsm8k import extract, read
from .statistics import paired

ARMS = ("k16_v16", "k16_v8", "k8_v8")


def load(path):
    return json.loads(Path(path).read_text())


def summarize(quality, gsm, task_path, protocol_path):
    task = read(task_path)
    expected = {r["id"]: r for r in task["samples"]}
    rows = {}
    native = {}
    common = None
    hashes = {}
    for arm in ARMS:
        path = Path(quality) / arm / "native-quality.json"
        evidence = load(path)
        p = dict(evidence["provenance"])
        p.pop("arm")
        p["engine_config"] = {
            k: v for k, v in p["engine_config"].items() if k != "kv_cache_dtype"
        }
        if common is None:
            common = p
        if p != common:
            raise ValueError("native arm provenance differs beyond cache dtype")
        if (
            len(evidence["nll_windows"]) != 64
            or len(evidence["retrieval_cases"]) != 200
        ):
            raise ValueError("complete native quality matrix is required")
        inputs = Path(protocol_path).parent
        manifest = json.loads(gzip.decompress((inputs / "qwen.json.gz").read_bytes()))
        wanted = {
            r["id"]: r
            for r in manifest["samples"]
            if r["kind"] in ("nll", "retrieval") and not r.get("reserve")
        }
        wanted_nll = {key for key, r in wanted.items() if r["kind"] == "nll"}
        wanted_ret = set(wanted) - wanted_nll
        if {r["sample_id"] for r in evidence["nll_windows"]} != wanted_nll or {
            r["sample_id"] for r in evidence["retrieval_cases"]
        } != wanted_ret:
            raise ValueError("native manifest case identities incomplete or duplicated")
        directory = Path(quality) / arm
        raw_nll = [
            json.loads(line)
            for line in (directory / "nll.jsonl").read_text().splitlines()
        ]
        if len(raw_nll) != 64 * 129:
            raise ValueError("native raw target coverage incomplete")
        for key in wanted_nll:
            sample = wanted[key]
            group = [r for r in raw_nll if r["sample_id"] == key]
            if (
                len(group) != 129
                or [r["step"] for r in group] != list(range(129))
                or [r["target"] for r in group]
                != sample["token_ids"][sample["length"] : sample["length"] + 129]
            ):
                raise ValueError("native target sequence identity mismatch")
            if any(
                not math.isfinite(r["nll"]) or r["provenance"] != evidence["provenance"]
                for r in group
            ):
                raise ValueError(
                    "native token loss is nonfinite or from another source"
                )
            item = next(r for r in evidence["nll_windows"] if r["sample_id"] == key)
            if (
                abs(sum(r["nll"] for r in group[1:]) / 128 - item["native_mean_nll"])
                > 1e-10
            ):
                raise ValueError("native summary differs from its raw token losses")
        installed = load(directory / "full/installed.json")
        if len(installed["layers"]) != 36 or len(installed["packed"]) != 144:
            raise ValueError("native cache/packed weight coverage incomplete")
        packed_expectation = load(protocol_path)["packed_execution"]
        for module in installed["packed"]:
            if any(
                module[key] != packed_expectation[key]
                for key in ("method", "kernel", "weight_bits")
            ):
                raise ValueError(
                    "packed weight implementation differs from the frozen protocol"
                )
        from .storage import reconcile, verify_attention_reads

        reconcile([t for layer in installed["layers"] for t in layer["cache"]])
        expected_types = (
            {"torch.float16"}
            if arm == "k16_v16"
            else (
                {"torch.float16", "torch.float8_e4m3fn"}
                if arm == "k16_v8"
                else {"torch.float8_e4m3fn"}
            )
        )
        if {
            t["dtype"] for t in load(directory / "full/kernel-read-0.json")
        } != expected_types:
            raise ValueError(
                "attention read physical dtypes differ from the selected mode"
            )
        verify_attention_reads(
            installed["layers"],
            [
                json.loads(line)
                for line in (directory / "full/kernel-read-events.jsonl")
                .read_text()
                .splitlines()
            ],
            expected_types,
        )
        scales = load(inputs / "scales.json")["scales"]
        for index, layer in enumerate(installed["layers"]):
            if layer["k_scale"] != (
                scales[f"{index}.k"] if arm == "k8_v8" else 1.0
            ) or layer["v_scale"] != (
                scales[f"{index}.v"] if arm != "k16_v16" else 1.0
            ):
                raise ValueError("frozen independent K/V scale mapping changed")
        for row in evidence["retrieval_cases"]:
            if (
                row["answer"] != wanted[row["sample_id"]]["answer"]
                or row["correct"]
                != (row["final"] == row["answer"] and not row["truncated"])
                or row["provenance"] != evidence["provenance"]
            ):
                raise ValueError("native retrieval score or provenance changed")
        native[arm] = evidence
        path = Path(gsm) / arm / "gsm8k.jsonl"
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        values = [json.loads(line) for line in raw.splitlines()]
        keyed = {r["id"]: r for r in values}
        if (
            len(values) != 1319
            or len(keyed) != len(values)
            or set(keyed) != set(expected)
        ):
            raise ValueError(
                "GSM8K requires exactly the same complete 1319-question set"
            )
        first_provenance = values[0]["provenance"]
        for key, row in keyed.items():
            if row["provenance"] != first_provenance:
                raise ValueError("mixed GSM8K provenance in an arm")
            for field in (
                "source_sha",
                "runner_files_sha256",
                "backend_sources",
                "model",
                "model_revision",
                "scale_file_sha256",
                "geometry_sha256",
            ):
                if row["provenance"][field] != common[field]:
                    raise ValueError(
                        "GSM8K does not use the qualified native stack: " + field
                    )
            if (
                row["output_tokens"] != len(row["generated_token_ids"])
                or row["input_tokens"] != len(expected[key]["token_ids"])
                or row["output_tokens"] > task["max_output_tokens"]
            ):
                raise ValueError("GSM8K generation token counts or cap changed")
            if row["flexible_correct"] != (
                extract(row["text"], flexible=True) == expected[key]["gold"]
                and not row["truncated"]
                and row["error"] is None
            ):
                raise ValueError("GSM8K flexible-score denominator changed")
            gold = expected[key]["gold"]
            if (
                row["gold"] != gold
                or row["prompt_hash"] != expected[key]["prompt_hash"]
            ):
                raise ValueError("GSM8K gold/prompt identity mismatch")
            strict = extract(row["text"])
            flex = extract(row["text"], flexible=True)
            if row["strict_answer"] != strict or row["flexible_answer"] != flex:
                raise ValueError("GSM8K extraction differs from frozen task")
            if row["correct"] != (
                strict == gold and not row["truncated"] and row["error"] is None
            ):
                raise ValueError("GSM8K correctness or failure denominator changed")
            if (
                row["provenance"]["task_file_sha256"]
                != hashlib.sha256(Path(task_path).read_bytes()).hexdigest()
            ):
                raise ValueError("GSM8K task source changed")
            if (
                row["provenance"]["protocol_sha256"]
                != hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest()
            ):
                raise ValueError("GSM8K execution protocol changed")
        rows[arm] = keyed
    cells = []
    native_pass = {}
    gsm_pass = {}
    gsm_cells = []
    direct = []
    base_native = native["k16_v16"]
    base = {k: v["correct"] for k, v in rows["k16_v16"].items()}
    baseline_useful = sum(base.values()) / 1319 >= task["baseline_accuracy_floor"]
    for arm in ARMS:
        good = True
        for length in (2048, 8192):
            pick_nll = lambda evidence: {
                r["sample_id"]: r["native_mean_nll"]
                for r in evidence["nll_windows"]
                if r["length"] == length
            }
            pick_ret = lambda evidence: {
                r["sample_id"]: r["correct"]
                for r in evidence["retrieval_cases"]
                if r["length"] == length
            }
            if len(pick_nll(native[arm])) != 32 or len(pick_ret(native[arm])) != 100:
                raise ValueError("native quality IDs incomplete")
            for metric, stats in [
                ("nll", paired_nll(pick_nll(native[arm]), pick_nll(base_native))),
                (
                    "retrieval",
                    paired_retrieval(pick_ret(native[arm]), pick_ret(base_native)),
                ),
            ]:
                cells.append({"arm": arm, "length": length, "metric": metric, **stats})
                good &= stats["supported_margin"]
        native_pass[arm] = bool(good)
        stats = paired(
            {k: v["correct"] for k, v in rows[arm].items()},
            base,
            margin=task["loss_margin"],
        )
        gsm_cells.append(
            {
                "arm": arm,
                "extraction": "strict-primary",
                "status": "complete",
                "baseline_useful": baseline_useful,
                "parser_failures": sum(r["parser_failure"] for r in rows[arm].values()),
                "truncations": sum(r["truncated"] for r in rows[arm].values()),
                "errors": sum(r["error"] is not None for r in rows[arm].values()),
                **stats,
            }
        )
        gsm_pass[arm] = stats["supported_margin"] and baseline_useful
        gsm_cells.append(
            {
                "arm": arm,
                "extraction": "flexible-secondary-nominal",
                **paired(
                    {k: v["flexible_correct"] for k, v in rows[arm].items()},
                    {k: v["flexible_correct"] for k, v in rows["k16_v16"].items()},
                    margin=task["loss_margin"],
                ),
            }
        )
    direct.append(
        {
            "arm": "k16_v8",
            "comparator": "k8_v8",
            "exploratory": True,
            **paired(
                {k: v["correct"] for k, v in rows["k16_v8"].items()},
                {k: v["correct"] for k, v in rows["k8_v8"].items()},
                margin=task["loss_margin"],
            ),
        }
    )
    qualified = [arm for arm in ARMS if native_pass[arm] and gsm_pass[arm]]
    if "k16_v16" not in qualified:
        qualified = []
    return {
        "status": "complete",
        "protocol_sha256": hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest(),
        "task_file_sha256": hashlib.sha256(Path(task_path).read_bytes()).hexdigest(),
        "raw_rows_sha256": hashes,
        "native_quality_cells": cells,
        "gsm8k_cells": gsm_cells,
        "direct_gsm8k_comparisons": direct,
        "native_quality_pass": native_pass,
        "gsm8k_pass": gsm_pass,
        "qualified_modes": qualified,
        "note": "Within-engine preservation only. Numerical bridges are separate and never silently relaxed. Flexible extraction and symmetric comparisons are secondary.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("quality", "gsm", "task", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.quality, args.gsm, args.task, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"qualified_modes": result["qualified_modes"], "status": result["status"]}
        )
    )


if __name__ == "__main__":
    main()
