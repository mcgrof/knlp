#!/usr/bin/env python3
"""Apply source-support judgments to a patient's self-study rows.

Inputs are the original parquet, the master the judged items came from
(the original or a repaired master, rows in source order), the judged
items, and, for rows that failed, the resampled candidates and their
judgments. For each judged train or development row:

- kept: the master answer passed (SUPPORTED or MINOR_ISSUES, no
  accepted false premise);
- replaced: it failed or had no final answer, and a candidate passed;
  the first SUPPORTED candidate is taken, else the first MINOR_ISSUES
  one, lowest candidate index first;
- excluded: no candidate passed, or the row was never resolved.

Rows that are not judged items (the sealed confirmation rows) keep the
master's content: where the master differs from the original, the row
is carried into the receipt by hash alone, so materializing from this
receipt reproduces the confirmation file the master gave.

Writes OUT_DIR/<stem>_clean.json, a receipt in the cas_teacher_repair
format relative to the ORIGINAL parquet (every row whose content
differs from the original, as status complete with old and new row
hashes), with <stem>_clean.parquet holding those rows, and
OUT_DIR/<stem>_exclude.json listing excluded rows with reasons. A
replaced row's targets are the candidate's own tokens and top-20
log-probabilities, so every training target is still the teacher's.

Env: SOURCE, MASTER, ITEMS, JUDGED, CAND_ITEMS, CAND_JUDGED, CAND_DIR,
OUT_DIR.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

PASS = ("SUPPORTED", "MINOR_ISSUES")


def row_sha(user, ids):
    h = hashlib.sha256(user.encode() + b"\0")
    h.update(np.asarray(ids, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def row_shas(table):
    users, shas = [], []
    for ms in table.column("messages").to_pylist():
        users.append(ms[0]["content"])
        shas.append(row_sha(ms[0]["content"], ms[1]["token_ids"]))
    return users, shas


def jsonl(path):
    out = {}
    if path and os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            out[r["key"]] = r
    return out


def passes(j):
    return j is not None and j.get("verdict") in PASS and not j.get("exclude")


def main():
    out = Path(os.environ["OUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    src = pq.read_table(os.environ["SOURCE"])
    master = pq.read_table(os.environ["MASTER"])
    stem = Path(os.environ["SOURCE"]).stem
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    items = jsonl(os.environ["ITEMS"])
    judged = jsonl(os.environ["JUDGED"])
    cands = jsonl(os.environ.get("CAND_ITEMS"))
    cjudged = jsonl(os.environ.get("CAND_JUDGED"))
    cdir = Path(os.environ.get("CAND_DIR", "."))
    by_row = {}
    for k, c in cands.items():
        by_row.setdefault(c["key"].rsplit(":", 1)[0], []).append(c)
    changed, receipt, exclude = [], [], []
    counts = {"kept": 0, "replaced": 0, "excluded": 0}
    for key, it in sorted(items.items(), key=lambda kv: kv[1]["row"]):
        i = it["row"]
        orig = src.slice(i, 1).to_pylist()[0]
        mrow = master.slice(i, 1).to_pylist()[0]
        user = orig["messages"][0]["content"]
        o_sha = row_sha(user, orig["messages"][1]["token_ids"])
        j = judged.get(key)
        if it["status"] == "judge" and passes(j):
            counts["kept"] += 1
            final = mrow
            how = dict(status="kept", verdict=j["verdict"])
        else:
            pick = None
            cl = sorted(by_row.get(key, []), key=lambda c: c["candidate"])
            for want in PASS:
                for c in cl:
                    cj = cjudged.get(c["key"])
                    if c["status"] == "judge" and passes(cj) and cj["verdict"] == want:
                        pick = (c, cj)
                        break
                if pick:
                    break
            if pick is None:
                counts["excluded"] += 1
                reason = (
                    "no passing candidate"
                    if cl
                    else ("not resampled" if it["status"] == "judge" else "cut")
                )
                exclude.append(
                    dict(
                        row=i,
                        split=it["split"],
                        reason=reason,
                        master_verdict=(j or {}).get("verdict"),
                        candidates=[
                            (cjudged.get(c["key"]) or {}).get("verdict", c["status"])
                            for c in cl
                        ],
                    )
                )
                continue
            c, cj = pick
            z = np.load(cdir / (key.replace(":", "_") + ".npz"))
            k = c["candidate"]
            ids = z[f"ids_{k}"].tolist()
            tids, tlps = z[f"top_ids_{k}"], z[f"top_lps_{k}"]
            n = len(ids)
            a = dict(mrow["messages"][1])
            a["token_ids"] = ids
            a["content"] = tok.decode(ids, skip_special_tokens=True)
            a["top_logprobs"] = dict(
                token_idx=np.repeat(np.arange(n), tids.shape[1]).tolist(),
                token_id=tids.reshape(-1).tolist(),
                logprobs=tlps.reshape(-1).tolist(),
                shape=[n, int(tids.shape[1])],
            )
            final = dict(mrow)
            final["messages"] = [mrow["messages"][0], a]
            counts["replaced"] += 1
            how = dict(
                status="replaced",
                candidate=k,
                seed=c["seed"],
                verdict=cj["verdict"],
                master_verdict=(j or {}).get("verdict", it["status"]),
            )
        f_sha = row_sha(user, final["messages"][1]["token_ids"])
        rec = dict(row=i, split=it["split"], old_row_sha=o_sha, **how)
        if f_sha != o_sha:
            changed.append(final)
            rec.update(
                status_detail=how["status"], status="complete", new_row_sha=f_sha
            )
        receipt.append(rec)
    judged_rows = {it["row"] for it in items.values()}
    m_users, m_shas = row_shas(master)
    s_users, s_shas = row_shas(src)
    assert m_users == s_users, "master and source questions differ"
    carried = 0
    for i, (o_sha, m_sha) in enumerate(zip(s_shas, m_shas)):
        if i in judged_rows or o_sha == m_sha:
            continue
        changed.append(master.slice(i, 1).to_pylist()[0])
        receipt.append(
            dict(
                row=i,
                old_row_sha=o_sha,
                new_row_sha=m_sha,
                status="complete",
                status_detail="carried",
            )
        )
        carried += 1
    counts["carried"] = carried
    if changed:
        pq.write_table(
            pa.Table.from_pylist(changed, schema=src.schema),
            out / f"{stem}_clean.parquet",
        )
    srcsha = hashlib.sha256(open(os.environ["SOURCE"], "rb").read()).hexdigest()
    (out / f"{stem}_clean.json").write_text(
        json.dumps(dict(source_sha256=srcsha, counts=counts, rows=receipt), indent=1)
    )
    (out / f"{stem}_exclude.json").write_text(
        json.dumps(dict(source_sha256=srcsha, rows=exclude), indent=1)
    )
    print(f"ASSEMBLE {stem} {json.dumps(counts)} changed={len(changed)}", flush=True)


if __name__ == "__main__":
    main()
