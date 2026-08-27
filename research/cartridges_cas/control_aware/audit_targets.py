#!/usr/bin/env python3
"""Element-level extension of the duplicate-target audit.

Parses every training element the trainer will actually consume (the
post-flatten, post-packing population) with the control-aware row
parser and reports the extended statistics the plan requires:
duplicate-conditioned and clean-row unique-mass means and quantiles,
the all-row added anchor coefficient, first-row and natural-EOT row
counts, one-token answers, and the per-example serialized entry count
distribution.  Doubles as a validation of the row parser: the
duplicate-row fraction and per-position anchor weights must land near
the recorded parquet-level audit (45.39% of rows duplicated, 63.0% of
first rows at mean anchor 0.852, 43.8% of stop rows at 0.858) —
differences reflect packing truncation, not parser drift.

CPU only.  Env: MODEL, DATA_PARQUET, OUT_JSON, MAX_ELEMENTS (0 = all).
"""

import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/ca_audit")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from cartridges.datasets import TrainDataset, DataSource  # noqa: E402
from targets import parse_element  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
DATA_PARQUET = os.environ["DATA_PARQUET"]
OUT_JSON = os.environ.get("OUT_JSON", "audit_targets.json")
MAX_ELEMENTS = int(os.environ.get("MAX_ELEMENTS", "0"))


def q(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(p * len(vals)))]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    dataset = TrainDataset.Config(
        data_sources=[DataSource(path=DATA_PARQUET, type="local")],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ).instantiate(tokenizer=tok, seed=0)
    elements = dataset.elements
    if MAX_ELEMENTS:
        elements = elements[:MAX_ELEMENTS]

    n_rows = n_dup_rows = 0
    dup_mass, clean_mass = [], []
    anchor_coeffs_all_rows = []
    first_rows = first_dup = 0
    eot_rows = eot_dup = 0
    first_anchor_p, eot_anchor_p = [], []
    entry_counts = []
    one_token_answers = 0
    n_natural = 0
    parse_errors = 0

    for el in elements:
        try:
            et = parse_element(
                el.topk_token_idxs, el.topk_token_ids, el.topk_logprobs, eot
            )
        except ValueError:
            parse_errors += 1
            continue
        entry_counts.append(et.n_serialized)
        if len(et.rows) == 1:
            one_token_answers += 1
        if et.eot_row_idx is not None:
            n_natural += 1
        for r in et.rows:
            n_rows += 1
            uids, ulps = r.unique_entries()
            umass = float(torch.tensor(ulps).exp().sum())
            a = r.anchor()
            if a is not None:
                n_dup_rows += 1
                dup_mass.append(umass)
                p = float(torch.tensor(a[1]).exp())
                anchor_coeffs_all_rows.append(p)
            else:
                clean_mass.append(umass)
                anchor_coeffs_all_rows.append(0.0)
            if r.idx == et.first_row_idx:
                first_rows += 1
                if a is not None:
                    first_dup += 1
                    first_anchor_p.append(float(torch.tensor(a[1]).exp()))
            if et.eot_row_idx is not None and r.idx == et.eot_row_idx:
                eot_rows += 1
                if a is not None:
                    eot_dup += 1
                    eot_anchor_p.append(float(torch.tensor(a[1]).exp()))

    report = dict(
        population="training elements (post-flatten, post-packing)",
        model=MODEL,
        parquet=DATA_PARQUET,
        elements=len(elements),
        parse_errors=parse_errors,
        rows=n_rows,
        duplicated_row_fraction=n_dup_rows / max(n_rows, 1),
        unique_mass_mean_duplicated=statistics.fmean(dup_mass) if dup_mass else None,
        unique_mass_mean_clean=statistics.fmean(clean_mass) if clean_mass else None,
        unique_mass_q10_duplicated=q(dup_mass, 0.10),
        unique_mass_q50_duplicated=q(dup_mass, 0.50),
        unique_mass_q90_duplicated=q(dup_mass, 0.90),
        all_row_added_anchor_coefficient_mean=(
            statistics.fmean(anchor_coeffs_all_rows) if anchor_coeffs_all_rows else None
        ),
        first_rows=first_rows,
        first_row_duplicated=first_dup,
        first_row_duplicate_fraction=first_dup / max(first_rows, 1),
        first_row_anchor_mean=(
            statistics.fmean(first_anchor_p) if first_anchor_p else None
        ),
        natural_eot_elements=n_natural,
        eot_rows=eot_rows,
        eot_row_duplicated=eot_dup,
        eot_row_anchor_mean=statistics.fmean(eot_anchor_p) if eot_anchor_p else None,
        one_token_answers=one_token_answers,
        entry_count_mean=statistics.fmean(entry_counts) if entry_counts else None,
        entry_count_q10=q(entry_counts, 0.10),
        entry_count_q50=q(entry_counts, 0.50),
        entry_count_q90=q(entry_counts, 0.90),
        trainer_aggregation=(
            "per-element mean of p*nll over VALID serialized entries "
            "(the legacy continuation trainer's semantics); the packed "
            "upstream trainer used a different batch denominator"
        ),
        reference_parquet_audit=dict(
            duplicated_row_fraction=0.4539,
            first_row_duplicate_fraction=0.630,
            first_row_anchor_mean=0.852,
            eot_row_duplicate_fraction=0.438,
            eot_row_anchor_mean=0.858,
        ),
    )
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print(f"AUDIT_TARGETS_DONE {OUT_JSON}")


if __name__ == "__main__":
    main()
