#!/usr/bin/env python3
"""Stage LongHealth data for offline hosts: fetch (or reuse) the benchmark
JSON and dump per-patient record text files.

Produces exactly what the harness consumes on a host with no outbound
network: RECORDS_DIR/<patient>.txt for truncation init (same record
template as cas_dump_records.py -- keep the two in sync), plus the raw
benchmark JSON for the strict evaluator's LONGHEALTH_JSON option.
Standalone on purpose: no cartridges package, no torch.

Usage:
  stage_longhealth.py --out-dir DIR [--json PATH] [--patients p1,p2]

With --json pointing at an existing benchmark_v5.json the network is
never touched.
"""

import argparse
import json
import urllib.request
from pathlib import Path

DATASET_URL = (
    "https://raw.githubusercontent.com/kbressem/LongHealth/"
    "refs/heads/main/data/benchmark_v5.json"
)
TMPL = (
    "Below is patient {name}'s medical record (ID: {pid}). Born {bd}. "
    "Diagnosis: {dx}. The record consists of {n} notes.\n{notes}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--json", default="", help="existing benchmark_v5.json")
    ap.add_argument("--patients", default="", help="comma list; default all")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.json:
        raw = Path(args.json).read_text()
    else:
        raw = urllib.request.urlopen(DATASET_URL).read().decode()
    (out / "benchmark_v5.json").write_text(raw)
    data = json.loads(raw)

    want = set(p for p in args.patients.split(",") if p)
    records = out / "records"
    records.mkdir(exist_ok=True)
    for pid, row in data.items():
        if want and pid not in want:
            continue
        notes = "\n".join(
            f"<{nid}>\n{txt}\n</{nid}>" for nid, txt in row["texts"].items()
        )
        txt = TMPL.format(
            name=row["name"],
            pid=pid,
            bd=row["birthday"],
            dx=row["diagnosis"],
            n=len(row["texts"]),
            notes=notes,
        )
        (records / f"{pid}.txt").write_text(txt)
        print(f"{pid}: {len(txt)} chars, {len(row['texts'])} notes")
    print(f"STAGED {out}")


if __name__ == "__main__":
    main()
