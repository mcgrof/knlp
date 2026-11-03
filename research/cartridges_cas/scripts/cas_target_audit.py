#!/usr/bin/env python3
"""Audit what a cartridge training run actually supervises.

The Cartridges library packs whole self-study conversations into fixed
length training sequences once, at dataset load, with a fixed seed;
every epoch then sees the same packs in a new order. In "truncate" mode
the conversation that overflows a pack is cut at the pack boundary and
its remainder is never trained, in any epoch. This script rebuilds the
library's element layout and packing on the CPU, without the library,
and reports for each patient and packed length:

- conversations intact, cut by the pack boundary, or dropped whole
  (an overflowing conversation that starts exactly at the boundary);
- teacher-side defects that packing cannot fix: answers that hit the
  teacher's generation cap (no end-of-turn token), answers without a
  closing think tag, malformed rows;
- top-k target checks: positions covered once each, retained
  probability mass, and whether the generated token is in its own
  top-k list (label alignment);
- exact duplicate rows and duplicate prompts;
- optional overlap between self-study prompts and evaluation questions.

Element layout (the library's Qwen converter, token_ids absent for the
user turn so it is retokenized): user = [<|im_start|> user \\n] + u +
[<|im_end|> \\n]; assistant = [<|im_start|> assistant \\n] + n, nothing
appended because it is the last message. Assistant target j sits at
element offset u + 8 + j; the loss reads the logits one position
earlier. Packing follows TrainDataset._prepare_batches.

Env: DATA_DIR (required, per-patient parquets), PATIENTS,
LENGTHS (8192,2048), SEED (42), GBS (128), EVAL_QUESTIONS (optional
JSON list of strings), OUT_DIR (JSON summaries and per-row manifests),
MODEL.
"""

import hashlib
import json
import os
import random
import re

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

DATA_DIR = os.environ["DATA_DIR"]
PATIENTS = os.environ.get("PATIENTS", "01 02 03 05 06").split()
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "8192,2048").split(",")]
SEED = int(os.environ.get("SEED", "42"))
GBS = int(os.environ.get("GBS", "128"))
OUT_DIR = os.environ.get("OUT_DIR", "target_audit")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
EOT, THINK_END = 151645, 151668
USER_START, ASST_START = [151644, 872, 198], [151644, 77091, 198]


def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def audit_rows(rows, tok):
    """Per-row facts independent of packing."""
    out = []
    for i, r in enumerate(rows):
        m = r["messages"]
        roles = [x["role"] for x in m]
        f = {"row": i, "roles": ",".join(roles), "malformed": ""}
        if roles != ["user", "assistant"]:
            f["malformed"] = "roles"
            out.append(f)
            continue
        u, a = m
        if u["token_ids"] is not None:
            uids = list(u["token_ids"])
        else:
            uids = tok.encode(u["content"] or "", add_special_tokens=False)
        aids = a["token_ids"]
        if aids is None or a["top_logprobs"] is None:
            f["malformed"] = "no_assistant_tokens_or_topk"
            out.append(f)
            continue
        aids = np.asarray(aids)
        n = len(aids)
        tl = a["top_logprobs"]
        idx = np.asarray(tl["token_idx"])
        tid = np.asarray(tl["token_id"])
        lp = np.asarray(tl["logprobs"], dtype=np.float64)
        shape = list(tl["shape"]) if tl["shape"] is not None else []
        k = shape[1] if len(shape) == 2 else 0
        f.update(u_len=len(uids), a_len=n, k=k)
        f["elem_len"] = len(USER_START) + len(uids) + 2 + len(ASST_START) + n
        f["tgt_off"] = len(USER_START) + len(uids) + 2 + len(ASST_START)
        f["ends_eot"] = bool(n and aids[-1] == EOT)
        f["think_closed"] = bool((aids == THINK_END).any())
        # top-k structure: k entries for every position 0..n-1, once each
        ok_shape = shape == [n, k] and len(idx) == n * k == len(tid) == len(lp)
        f["topk_shape_ok"] = bool(ok_shape)
        if ok_shape and k:
            pos = idx.reshape(n, k)
            f["topk_pos_ok"] = bool((pos == np.arange(n)[:, None]).all())
            mass = np.exp(lp.reshape(n, k)).sum(1)
            f["mass_mean"] = float(mass.mean())
            f["mass_min"] = float(mass.min())
            f["mass_lt_0p9"] = int((mass < 0.9).sum())
            ids = tid.reshape(n, k)
            hit = (ids == aids[:, None]).any(1)
            f["label_in_topk"] = int(hit.sum())
            f["label_top1"] = int((ids[:, 0] == aids).sum())
        else:
            f["topk_pos_ok"] = False
        h = hashlib.sha256()
        h.update((u["content"] or "").encode())
        h.update(b"\0")
        h.update(aids.astype(np.int64).tobytes())
        f["row_sha"] = h.hexdigest()[:16]
        f["prompt_sha"] = hashlib.sha256(norm(u["content"] or "").encode()).hexdigest()[
            :16
        ]
        f["prompt_norm"] = norm(u["content"] or "")
        meta = r.get("metadata") or {}
        f["note_id"] = meta.get("note_id") if isinstance(meta, dict) else None
        out.append(f)
    return out


def pack(lengths, S, seed):
    """TrainDataset._prepare_batches in truncate mode."""
    order = random.Random(seed).sample(range(len(lengths)), len(lengths))
    batches, cur, n = [], [], 0
    q = list(order)
    i = 0
    while i < len(q):
        e = q[i]
        L = lengths[e]
        if n == 0 and L > S:
            cur.append(e)
            i += 1
            batches.append(cur)
            cur, n = [], 0
        elif n + L > S:
            cur.append(e)  # truncate mode: overflowing element joins and is cut
            i += 1
            batches.append(cur)
            cur, n = [], 0
        else:
            cur.append(e)
            i += 1
            n += L
    if cur:
        batches.append(cur)
    return batches


def audit_packing(facts, S):
    good = [f for f in facts if not f["malformed"]]
    lengths = {f["row"]: f["elem_len"] for f in good}
    keys = [f["row"] for f in good]
    by = {f["row"]: f for f in good}
    batches = pack([lengths[r] for r in keys], S, SEED)
    tot = kept = 0
    intact = cut = dropped = 0
    cut_lost_eot = cut_lost_think = 0
    empty_packs = 0
    per_row = {}
    for b in batches:
        off = 0
        pack_kept = 0
        for j in b:
            f = by[keys[j]]
            n = f["a_len"]
            first = off + f["tgt_off"]
            k_ = max(0, min(n, S - first))
            tot += n
            kept += k_
            pack_kept += k_
            per_row[f["row"]] = k_
            if k_ == n:
                intact += 1
            elif k_ == 0:
                dropped += 1
            else:
                cut += 1
                if f["ends_eot"]:
                    cut_lost_eot += 1
                if f["think_closed"]:
                    cut_lost_think += 1
            off += f["elem_len"]
        if pack_kept == 0:
            empty_packs += 1
    return {
        "packed_len": S,
        "packs": len(batches),
        "steps_per_epoch_at_gbs": len(batches) / GBS,
        "steps_80_epochs": round(80 * len(batches) / GBS),
        "conversations": len(keys),
        "intact": intact,
        "cut_by_pack": cut,
        "dropped_whole": dropped,
        "cut_that_lose_end_of_turn": cut_lost_eot,
        "target_tokens": tot,
        "target_tokens_kept": kept,
        "kept_fraction": kept / tot if tot else 0.0,
        "empty_target_packs": empty_packs,
        "oversize_singletons": sum(1 for r in keys if lengths[r] > S),
    }, per_row


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    os.makedirs(OUT_DIR, exist_ok=True)
    evalq = []
    if os.environ.get("EVAL_QUESTIONS"):
        evalq = [norm(q) for q in json.load(open(os.environ["EVAL_QUESTIONS"]))]
    summary = {}
    for p in PATIENTS:
        path = os.path.join(DATA_DIR, f"patient_{p}.parquet")
        rows = pq.read_table(path, columns=["messages", "metadata"]).to_pylist()
        facts = audit_rows(rows, tok)
        good = [f for f in facts if not f["malformed"]]
        s = {
            "parquet": path,
            "parquet_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "rows": len(rows),
            "malformed": {},
            "teacher_cap_cut": sum(1 for f in good if not f["ends_eot"]),
            "think_not_closed": sum(1 for f in good if not f["think_closed"]),
            "topk_shape_bad": sum(1 for f in good if not f["topk_shape_ok"]),
            "topk_pos_bad": sum(1 for f in good if not f["topk_pos_ok"]),
            "k": sorted({f["k"] for f in good}),
        }
        for f in facts:
            if f["malformed"]:
                s["malformed"][f["malformed"]] = (
                    s["malformed"].get(f["malformed"], 0) + 1
                )
        ok = [f for f in good if f["topk_shape_ok"]]
        npos = sum(f["a_len"] for f in ok)
        s["target_positions"] = npos
        s["mass_mean"] = sum(f["mass_mean"] * f["a_len"] for f in ok) / npos
        s["mass_min"] = min(f["mass_min"] for f in ok)
        s["positions_mass_lt_0p9"] = sum(f["mass_lt_0p9"] for f in ok)
        s["label_in_topk_fraction"] = sum(f["label_in_topk"] for f in ok) / npos
        s["label_top1_fraction"] = sum(f["label_top1"] for f in ok) / npos
        rs = [f["row_sha"] for f in good]
        ps = [f["prompt_sha"] for f in good]
        s["duplicate_rows"] = len(rs) - len(set(rs))
        s["unique_prompts"] = len(set(ps))
        s["duplicate_prompt_rows"] = len(ps) - len(set(ps))
        if evalq:
            pn = {f["prompt_norm"] for f in good}
            s["eval_question_exact_overlap"] = sum(1 for q in evalq if q in pn)
        L = sorted(f["elem_len"] for f in good)
        s["elem_len_p50_p90_max"] = [L[len(L) // 2], L[int(0.9 * (len(L) - 1))], L[-1]]
        s["packing"] = {}
        kept_by_len = {}
        for S in LENGTHS:
            res, per_row = audit_packing(facts, S)
            s["packing"][str(S)] = res
            kept_by_len[S] = per_row
        # teacher cap cuts and pack cuts are separate populations: count
        # conversations that are both, so neither hides the other
        for S in LENGTHS:
            pr = kept_by_len[S]
            s["packing"][str(S)]["cut_or_dropped_and_teacher_cap_cut"] = sum(
                1 for f in good if not f["ends_eot"] and pr[f["row"]] < f["a_len"]
            )
        with open(os.path.join(OUT_DIR, f"rows_patient_{p}.tsv"), "w") as fh:
            cols = (
                [
                    "row",
                    "row_sha",
                    "prompt_sha",
                    "note_id",
                    "u_len",
                    "a_len",
                    "elem_len",
                    "ends_eot",
                    "think_closed",
                    "mass_mean",
                    "label_top1",
                ]
                + [f"kept_{S}" for S in LENGTHS]
                + ["malformed"]
            )
            fh.write("\t".join(cols) + "\n")
            for f in facts:
                vals = [f.get(c, "") for c in cols[:11]]
                vals += [kept_by_len[S].get(f["row"], "") for S in LENGTHS]
                vals.append(f["malformed"])
                fh.write("\t".join(str(v) for v in vals) + "\n")
        summary[f"patient_{p}"] = s
        print(
            f"AUDIT patient_{p} "
            + json.dumps(
                {
                    k: v
                    for k, v in s.items()
                    if k not in ("packing", "parquet", "parquet_sha256")
                }
            ),
            flush=True,
        )
        for S in LENGTHS:
            print(f"  PACK {S} " + json.dumps(s["packing"][str(S)]), flush=True)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(
            {"seed": SEED, "gbs": GBS, "model": MODEL, "patients": summary},
            fh,
            indent=1,
        )


if __name__ == "__main__":
    main()
