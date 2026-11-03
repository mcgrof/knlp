#!/usr/bin/env python3
"""Export self-study rows as items for a source-support judge.

Writes one JSON line per train and development row of a materialized
split: the row index, its split, the hash of its note context (the
system prompt the teacher answered from), the question, and the
teacher's final answer (the text after the closing think tag). An
answer that does not end with the end-of-turn token, or has no closing
think tag, has no final answer to judge and is marked for regeneration
instead. Confirmation rows are never read or written, so their content
stays sealed. The note contexts are written once each, by hash.

Env: PARQUET (the source or repaired master, rows in source order),
MATERIALIZED (cas_split_materialize.py receipt for its split),
OUT_DIR, MODEL (tokenizer, Qwen/Qwen3-8B), PATIENT (label).
"""

import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

EOT, THINK_END = 151645, 151668


def main():
    out = Path(os.environ["OUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    patient = os.environ["PATIENT"]
    tok = AutoTokenizer.from_pretrained(os.environ.get("MODEL", "Qwen/Qwen3-8B"))
    mat = json.loads(Path(os.environ["MATERIALIZED"]).read_text())
    split = {
        i: name for name in ("train", "dev") for i in mat["files"][name]["row_index"]
    }
    t = pq.read_table(os.environ["PARQUET"], columns=["messages", "system_prompt"])
    notes, n = {}, {"judge": 0, "regenerate": 0}
    with open(out / f"items_{patient}.jsonl", "w") as f:
        for i in sorted(split):
            row = t.slice(i, 1).to_pylist()[0]
            sp = row["system_prompt"]
            h = hashlib.sha256(sp.encode()).hexdigest()[:16]
            notes[h] = sp
            ids = list(row["messages"][1]["token_ids"])
            ok = bool(ids) and ids[-1] == EOT and THINK_END in ids
            final = ""
            if ok:
                k = ids.index(THINK_END)
                final = tok.decode(ids[k + 1 : -1], skip_special_tokens=True).strip()
            status = "judge" if ok and final else "regenerate"
            n[status] += 1
            f.write(
                json.dumps(
                    dict(
                        key=f"{patient}:{i}",
                        patient=patient,
                        row=i,
                        split=split[i],
                        note=h,
                        question=row["messages"][0]["content"],
                        answer=final,
                        status=status,
                        answer_tokens=len(ids),
                    )
                )
                + "\n"
            )
    (out / f"notes_{patient}.json").write_text(json.dumps(notes))
    print(f"ITEMS {patient} {json.dumps(n)} notes={len(notes)}", flush=True)


if __name__ == "__main__":
    main()
