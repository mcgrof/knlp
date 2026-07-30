#!/usr/bin/env python3
"""Drop unfinished teacher trajectories from a self-study corpus.

The truncation audit showed ~15-19% of teacher answers hit their own 2048-token
generation cap: no closing </think>, no final answer -- unfinished thoughts. The
3072-packing fix stopped the PACKER from truncating stored rows, but those
already-unfinished teacher answers survive in the data, so training still teaches
the cartridge to ramble without concluding (Pro's leading remaining suspect).

This writes a "finished-only" copy of each patient's parquet, keeping a row only
if its assistant answer both contains </think> AND did not hit the generation cap
(answer token length below CAP_GUARD). Same run-dir layout so the trainer glob
finds it unchanged.

Env: PATIENTS, SRC (source self-study root), OUT (finished-only root),
CAP_GUARD (token length treated as "hit the cap", default 2000).
"""

import os
import sys
import glob

os.environ.setdefault("CARTRIDGES_DIR", "/home/mcgrof/cartridges")
sys.path.insert(0, os.environ["CARTRIDGES_DIR"])

import pandas as pd
from transformers import AutoTokenizer

PATIENTS = os.environ.get(
    "PATIENTS", "patient_01 patient_02 patient_03 patient_05 patient_06"
).split()
SRC = os.environ.get("SRC", "/home/mcgrof/cas_out/synth_diverse")
OUT = os.environ.get("OUT", "/home/mcgrof/cas_out/synth_diverse_finished")
CAP_GUARD = int(os.environ.get("CAP_GUARD", "2000"))
MODEL = "Qwen/Qwen3-8B"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def finished(aa):
    if "</think>" not in aa:
        return False
    n = len(tok(aa, add_special_tokens=False)["input_ids"])
    return n < CAP_GUARD  # hit the 2048 gen cap -> unfinished


def main():
    total_kept = total = 0
    for p in PATIENTS:
        pstr = p.replace("patient_", "p")
        g = glob.glob(f"{SRC}/*/synth_*_14bq_8ba_{pstr}_n*/artifact/dataset.parquet")
        if not g:
            print(f"  {p}: no parquet")
            continue
        df = pd.read_parquet(sorted(g)[-1])
        keep = []
        for i, row in df.iterrows():
            aa = next(
                (m["content"] for m in row["messages"] if m.get("role") == "assistant"),
                "",
            )
            if aa and finished(aa):
                keep.append(i)
        kept = df.iloc[keep].reset_index(drop=True)
        n = len(kept)
        d = os.path.join(
            OUT, "run", f"synth_finished_14bq_8ba_{pstr}_n{n}-0", "artifact"
        )
        os.makedirs(d, exist_ok=True)
        kept.to_parquet(os.path.join(d, "dataset.parquet"))
        total_kept += n
        total += len(df)
        print(
            f"  {p}: kept {n}/{len(df)} ({n/len(df):.1%}) finished -> {d}", flush=True
        )
    print(
        f"CAS_FILTER_DONE kept {total_kept}/{total} = {total_kept/max(total,1):.1%} -> {OUT}"
    )


if __name__ == "__main__":
    main()
