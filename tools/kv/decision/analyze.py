# SPDX-License-Identifier: MIT
"""Replay the two frozen objective families and direct paired contrasts on CPU."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.stats import binomtest
from tools.kv.native.gsm8k import number
from .cache import ARMS
from .statistics import paired_binary, paired_nll

ALPHA = 0.025 / 12


def rows(path, expected):
    data = [json.loads(s) for s in path.read_text().splitlines()]
    result = {r["id"]: r for r in data}
    if len(data) != len(result) or set(result) != set(expected):
        raise ValueError(f"Incomplete or repeated fixed panel: {path}")
    return result


def model_result(inputs, root):
    raw = inputs.read_bytes()
    data = json.loads(gzip.decompress(raw))
    start = json.loads((root / "START.json").read_text())
    done = json.loads((root / "DONE.json").read_text())
    if (
        start["input_sha256"] != hashlib.sha256(raw).hexdigest()
        or not done["native_parity_pass"]
    ):
        raise ValueError("Input identity/parity failed")
    if start["args"]["stage"] != "confirmation":
        raise ValueError("Not confirmation outputs")
    expected = {
        k: {r["id"]: r for r in data["samples"]["confirmation"] if r["kind"] == k}
        for k in ["nll", "gsm8k", "retrieval"]
    }
    panels = {
        a: {k: rows(root / a / (k + ".jsonl"), ids) for k, ids in expected.items()}
        for a in ARMS
    }
    diagnostics = {}
    memory = {}
    for a, kinds in panels.items():
        diagnostics[a] = {}
        memory[a] = {}
        for k, rs in kinds.items():
            for i, r in rs.items():
                item = expected[k][i]
                if k == "nll":
                    if (
                        len(r["nll"]) != 128
                        or not np.isfinite(r["nll"]).all()
                        or abs(np.mean(r["nll"]) - r["mean_nll"]) > 1e-6
                    ):
                        raise ValueError("NLL row invalid")
                else:
                    if k == "gsm8k":
                        matches = re.findall(data["gsm_regex"], r["text"])
                        answer = number(matches[-1]) if matches else None
                        correct = answer == item["gold"]
                        gold = item["gold"]
                    else:
                        # Saved parsed text is bound to retained tokens by the
                        # runtime tokenizer; independent text cleanup matches
                        # the specified Qwen stop/special-token-free decoding.
                        answer = r["text"].strip().strip(" \n\t.\"'`")
                        gold = item["answer"]
                        correct = answer.casefold() == gold.casefold()
                    if (
                        r["answer"] != answer
                        or r["gold"] != gold
                        or r["correct"] != correct
                        or r["error"] is not None
                    ):
                        raise ValueError("Frozen scoring mismatch or task error")
            if k != "nll":
                diagnostics[a][k] = dict(
                    correct=sum(r["correct"] for r in rs.values()),
                    truncated=sum(r["truncated"] for r in rs.values()),
                    parser_failure=sum(r["parser_failure"] for r in rs.values()),
                    errors=sum(r["error"] is not None for r in rs.values()),
                )
        for length in [1024, 8192]:
            audits = [
                r["cache"] for r in kinds["nll"].values() if r["length"] == length
            ]
            persistent = {r["unique_backing_bytes"] for r in audits}
            scales = {r["scale_logical_bytes"] for r in audits}
            if len(persistent) != 1 or len(scales) != 1:
                raise ValueError("Unexpected fixed-shape allocation variation")
            for audit in audits:
                for t in audit["tensors"]:
                    is_quantized = (t["field"] == "keys" and a[0] in "RS") or (
                        t["field"] == "values" and a[0] in "AS"
                    )
                    expected_dtype = (
                        "torch.float32"
                        if t["field"].endswith("scale")
                        else (
                            "torch.float8_e4m3fn"
                            if is_quantized
                            else "torch." + data["dtype"]
                        )
                    )
                    if t["dtype"] != expected_dtype:
                        raise ValueError("Wrong physical tensor dtype")
            memory[a][str(length)] = dict(
                persistent_bytes=persistent.pop(),
                scale_bytes=scales.pop(),
                maximum_reconstructed_native_bytes=max(
                    r["maximum_layer_reconstructed_native_bytes"] for r in audits
                ),
                peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in audits),
                peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in audits),
            )
    for p in ["F", "D"]:
        for length in ["1024", "8192"]:
            if (
                memory["A" + p][length]["persistent_bytes"]
                != memory["R" + p][length]["persistent_bytes"]
            ):
                raise ValueError("A/R measured backing budgets differ")

    def compare(b, c):
        base = panels[b]
        cand = panels[c]
        result = {}
        for k in ["gsm8k", "retrieval"]:
            for length in [None] if k == "gsm8k" else [1024, 8192]:
                ids = [
                    i
                    for i, r in base[k].items()
                    if length is None or r["length"] == length
                ]
                result[k if length is None else k + "-" + str(length)] = paired_binary(
                    {i: base[k][i]["correct"] for i in ids},
                    {i: cand[k][i]["correct"] for i in ids},
                    ALPHA,
                )
        for length in [1024, 8192]:
            ids = sorted(i for i, r in base["nll"].items() if r["length"] == length)
            result["nll-" + str(length)] = paired_nll(
                [base["nll"][i]["mean_nll"] for i in ids],
                [cand["nll"][i]["mean_nll"] for i in ids],
            )
        return result

    candidates = {}
    for a in ARMS[1:]:
        r = compare("N", a)
        common = all(
            r[k]["native_correct"] / r[k]["n"] >= 0.5
            for k in ["gsm8k", "retrieval-1024", "retrieval-8192"]
        ) and all(
            r["retrieval-" + str(l)]["gross_loss_one_sided_bounds"][1] <= 0.05
            and r["nll-" + str(l)]["relative_one_sided_bounds"][1] <= 0.01
            for l in [1024, 8192]
        )
        r["preservation_certificate"] = (
            common and r["gsm8k"]["gross_loss_one_sided_bounds"][1] <= 0.02
        )
        r["aggregate_math_certificate"] = (
            common and r["gsm8k"]["net_one_sided_bounds"][0] >= -0.01
        )
        candidates[a] = r
    direct = []
    for policy in ["F", "D"]:
        for control in ["R", "S"]:
            b = control + policy
            c = "A" + policy
            r = compare(b, c)
            for endpoint, result in r.items():
                if endpoint.startswith("nll"):
                    length = int(endpoint.split("-")[1])
                    ids = sorted(
                        i for i, x in panels[b]["nll"].items() if x["length"] == length
                    )
                    delta = np.array(
                        [
                            panels[c]["nll"][i]["mean_nll"]
                            - panels[b]["nll"][i]["mean_nll"]
                            for i in ids
                        ]
                    )
                    rng = np.random.default_rng(20261020)
                    boot = delta[rng.integers(0, len(ids), (100000, len(ids)))].mean(1)
                    centered = boot - delta.mean()
                    pval = min(
                        1.0,
                        2
                        * min(
                            (np.count_nonzero(centered >= delta.mean()) + 1) / 100001,
                            (np.count_nonzero(centered <= delta.mean()) + 1) / 100001,
                        ),
                    )
                    interval = np.quantile(
                        boot, [0.025, 0.975], method="linear"
                    ).tolist()
                    family_interval = np.quantile(
                        boot, [0.025 / 8, 1 - 0.025 / 8], method="linear"
                    ).tolist()
                    method = "Paired-document centered bootstrap, 100000 seed20261020; approximate"
                else:
                    g, l = result["gains"], result["losses"]
                    pval = float(binomtest(g, g + l, 0.5).pvalue) if g + l else 1.0
                    interval = result["net_interval_95"]
                    family_interval = None
                    method = "Exact two-sided McNemar; descriptive Tango interval"
                direct.append(
                    dict(
                        baseline=b,
                        candidate=c,
                        endpoint=endpoint,
                        result=result,
                        p_two_sided=pval,
                        paired_interval_95=interval,
                        bonferroni8_nll_interval=family_interval,
                        test_method=method,
                    )
                )
    return dict(
        model=data["model"],
        revision=data["revision"],
        source=start["source"],
        input_sha256=start["input_sha256"],
        candidates=candidates,
        direct=direct,
        diagnostics=diagnostics,
        memory=memory,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    results = {}
    for name in ["qwen3", "qwen25"]:
        results[name] = model_result(
            a.study / "inputs" / name / "inputs.json.gz",
            a.study / name / "artifact-mirror/confirmation",
        )
    for endpoint in [
        "gsm8k",
        "retrieval-1024",
        "retrieval-8192",
        "nll-1024",
        "nll-8192",
    ]:
        family = [
            r
            for model in results.values()
            for r in model["direct"]
            if r["endpoint"] == endpoint
        ]
        assert len(family) == 8
        current = 0.0
        for index, r in enumerate(sorted(family, key=lambda x: x["p_two_sided"])):
            current = max(current, min(1.0, (8 - index) * r["p_two_sided"]))
            r["holm8_p"] = current
    a.out.write_text(
        json.dumps(
            dict(
                schema=1,
                alpha=ALPHA,
                certificate_families="12 preservation and 12 aggregate-math; alpha .025 per family",
                models=results,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
