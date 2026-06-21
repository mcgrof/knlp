"""The artifact contract exists because a per-model loop crash once dropped completed CSVs and the
report mis-classified a model. append_row must be per-model and incremental (header once, survives a
later crash), and aggregate must gather without ever clobbering a per-model file."""

import csv
import os

from tools.kv.fp8_failure import common as C


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_append_row_header_once_and_incremental(tmp_path):
    p = C.model_csv_path(str(tmp_path), "qwen2-7b", "cells.csv")
    fields = ["cell", "ppl"]
    C.append_row(p, {"cell": "bf16", "ppl": 10.0}, fields)
    C.append_row(p, {"cell": "fp8", "ppl": 12.0}, fields)
    with open(p) as f:
        lines = [ln for ln in f.read().splitlines() if ln]
    assert lines[0] == "cell,ppl"  # header exactly once
    assert len(lines) == 3
    rows = _read(p)
    assert [r["cell"] for r in rows] == ["bf16", "fp8"]


def test_two_models_do_not_overwrite_each_other(tmp_path):
    out = str(tmp_path)
    pa = C.model_csv_path(out, "qwen2-7b", "cells.csv")
    pb = C.model_csv_path(out, "llama-3-8b", "cells.csv")
    fields = ["cell", "ppl"]
    C.append_row(pa, {"cell": "bf16", "ppl": 1.0}, fields)
    C.append_row(pb, {"cell": "bf16", "ppl": 2.0}, fields)
    assert pa != pb and os.path.exists(pa) and os.path.exists(pb)
    assert _read(pa)[0]["ppl"] == "1.0"
    assert _read(pb)[0]["ppl"] == "2.0"


def test_aggregate_gathers_without_clobbering(tmp_path):
    out = str(tmp_path)
    fields = ["cell", "ppl"]
    pa = C.model_csv_path(out, "qwen2-7b", "cells.csv")
    pb = C.model_csv_path(out, "llama-3-8b", "cells.csv")
    C.append_row(pa, {"cell": "bf16", "ppl": 1.0}, fields)
    C.append_row(pb, {"cell": "bf16", "ppl": 2.0}, fields)
    agg = C.aggregate(out, "cells.csv", fields)
    assert agg == os.path.join(out, "cells.csv")
    # per-model files still intact after aggregation
    assert os.path.exists(pa) and os.path.exists(pb)
    assert {r["ppl"] for r in _read(agg)} == {"1.0", "2.0"}
    # aggregate is rebuildable: re-running yields the same row count, no duplication of per-model data
    agg2 = C.aggregate(out, "cells.csv", fields)
    assert len(_read(agg2)) == 2


def test_recovery_fraction_math():
    # full recovery, none, partial
    assert C.recovery_fraction(1.0, 0.0, 0.0) == 1.0
    assert C.recovery_fraction(1.0, 1.0, 0.0) == 0.0
    assert abs(C.recovery_fraction(1.0, 0.25, 0.0) - 0.75) < 1e-9
