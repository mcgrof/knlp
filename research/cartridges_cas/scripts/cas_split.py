#!/usr/bin/env python3
"""Split one document into pieces for separately built cartridges, and
compose the pieces' cartridges back into one loadable bundle.

The question behind it: can a document's cartridge be built as smaller
pieces, each trained on its own section, then loaded together, at the
same total size as one cartridge? This script does the bookkeeping that
experiment needs; training uses ``cas_train_isolated.py`` unchanged.

``prep`` reads a record whose notes are tagged ``<text_i> ... </text_i>``
in chronological order, counts each note's tokens, and cuts the notes
into N contiguous pieces at the note boundaries nearest equal token
shares (a note is never cut). Each piece gets:

- a record file: the record's first line, with its note count replaced
  by the piece's notes, then the piece's notes verbatim;
- a slot budget: the total cartridge length K counts one frozen sink per
  piece, and the remaining K - N trainable slots are shared in
  proportion to the pieces' token counts (largest remainder), so the
  bundle is exactly K slots, the same as the single cartridge;
- a training parquet: the Self-Study rows whose ``metadata.note_id`` is
  one of the piece's notes. Each such row was generated from one note,
  so the rows split cleanly by piece.

With a LongHealth benchmark file it also records, for the first MAX_Q
questions, which pieces hold the answer (``answer_location``), so
scores can be read separately for questions that need one piece and
questions that need several. Everything goes to ``manifest.json``.

``compose`` concatenates piece cartridges in piece order. A piece built
alone stores its keys rotated at positions 0 .. K_i - 1; inside the
bundle it sits after the earlier pieces, so its keys are rotated on by
the sum of the earlier pieces' lengths (rotary rotations add). The first
piece's sink stays the bundle's frozen sink; the later pieces' sinks are
kept in place as ordinary slots, so every slot is accounted for.

Env (prep):
    RECORD     record text with <text_i> tags
    PARQUET    Self-Study parquet with metadata.note_id
    K_TOTAL    total cartridge length of the single cartridge (sink included)
    PIECES     number of pieces (default 2)
    LH_JSON    LongHealth benchmark_v5.json (optional)
    PATIENT    patient id for LH_JSON (e.g. patient_01)
    MAX_Q      panel questions to label (default 20)
    MODEL      tokenizer (default Qwen/Qwen3-8B)
    OUT_DIR    output directory
Env (compose):
    PIECE_CARTS  comma list of piece cartridge files, in piece order
    OUT          bundle file
    THETA        rotary base (default 1e6, Qwen3)

Usage:
    python cas_split.py prep
    python cas_split.py compose
    python cas_split.py selftest
"""

import hashlib
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NOTE_RE = re.compile(r"<(text_\d+)>\n?(.*?)\n?</\1>", re.S)


def parse_record(text):
    """(header line, [(note_id, full tagged note text)]) in record order."""
    header = text.split("\n", 1)[0]
    notes = [(m.group(1), m.group(0)) for m in NOTE_RE.finditer(text)]
    assert notes, "no <text_i> notes found"
    return header, notes


def cut_points(sizes, pieces):
    """Contiguous cut of len(sizes) items into ``pieces`` groups whose
    boundaries sit nearest the equal-share targets; each group non-empty."""
    n = len(sizes)
    assert 1 <= pieces <= n, f"{pieces} pieces from {n} notes"
    total = sum(sizes)
    cum = [0]
    for s in sizes:
        cum.append(cum[-1] + s)
    cuts = [0]
    for j in range(1, pieces):
        target = total * j / pieces
        lo, hi = cuts[-1] + 1, n - (pieces - j)
        best = min(range(lo, hi + 1), key=lambda b: abs(cum[b] - target))
        cuts.append(best)
    cuts.append(n)
    return [(cuts[j], cuts[j + 1]) for j in range(pieces)]


def allocate(weights, slots):
    """Largest-remainder split of ``slots`` in proportion to weights."""
    total = sum(weights)
    raw = [slots * w / total for w in weights]
    base = [int(r) for r in raw]
    rest = slots - sum(base)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:rest]:
        base[i] += 1
    return base


def sha(b):
    return hashlib.sha256(b).hexdigest()


def prep():
    import pandas as pd
    from transformers import AutoTokenizer

    rec_path, pq = os.environ["RECORD"], os.environ["PARQUET"]
    K = int(os.environ["K_TOTAL"])
    N = int(os.environ.get("PIECES", "2"))
    out = os.environ["OUT_DIR"]
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(os.environ.get("MODEL", "Qwen/Qwen3-8B"))
    text = open(rec_path).read()
    header, notes = parse_record(text)
    ntok = [len(tok.encode(t, add_special_tokens=False)) for _, t in notes]
    groups = cut_points(ntok, N)
    ptok = [sum(ntok[a:b]) for a, b in groups]
    trainable = allocate(ptok, K - N)
    assert min(trainable) >= 16, f"a piece gets {min(trainable)} slots; merge it"
    df = pd.read_parquet(pq)
    note_of_row = [m["note_id"] for m in df["metadata"]]
    known = {nid for nid, _ in notes}
    assert set(note_of_row) <= known, set(note_of_row) - known
    pieces = []
    for j, (a, b) in enumerate(groups):
        ids = [nid for nid, _ in notes[a:b]]
        h = re.sub(
            r"The record consists of \d+ notes?\.",
            f"This part of the record holds notes {', '.join(ids)}.",
            header,
        )
        body = h + "\n" + "\n".join(t for _, t in notes[a:b]) + "\n"
        rpath = os.path.join(out, f"piece{j}.txt")
        open(rpath, "w").write(body)
        rows = [i for i, nid in enumerate(note_of_row) if nid in ids]
        ppath = os.path.join(out, f"piece{j}.parquet")
        df.iloc[rows].reset_index(drop=True).to_parquet(ppath)
        pieces.append(
            dict(
                piece=j,
                notes=ids,
                note_tokens=ntok[a:b],
                tokens=ptok[j],
                record_tokens=len(tok.encode(body, add_special_tokens=False)),
                k_total=1 + trainable[j],
                k_trainable=trainable[j],
                rows=len(rows),
                record=rpath,
                record_sha256=sha(body.encode()),
                parquet=ppath,
            )
        )
    man = dict(
        record=os.path.abspath(rec_path),
        record_sha256=sha(text.encode()),
        parquet=os.path.abspath(pq),
        parquet_sha256=sha(open(pq, "rb").read()),
        tokenizer=os.environ.get("MODEL", "Qwen/Qwen3-8B"),
        k_total=K,
        pieces_n=N,
        rule="contiguous notes, cut at the note boundary nearest equal "
        "token shares; one frozen sink per piece; trainable slots by "
        "largest remainder in proportion to note tokens",
        pieces=pieces,
        rows_total=len(df),
    )
    assert sum(p["k_total"] for p in pieces) == K
    if os.environ.get("LH_JSON"):
        pid = os.environ["PATIENT"]
        lh = json.load(open(os.environ["LH_JSON"]))[pid]
        piece_of = {nid: p["piece"] for p in pieces for nid in p["notes"]}
        qs = []
        for q in lh["questions"][: int(os.environ.get("MAX_Q", "20"))]:
            loc = sorted(q.get("answer_location") or {})
            sp = sorted({piece_of[n] for n in loc}) if loc else None
            qs.append(
                dict(
                    qid=f"{pid}_{q['No']}",
                    notes=loc,
                    pieces=sp,
                    kind=(
                        "unknown"
                        if not sp
                        else "single" if len(sp) == 1 else "multiple"
                    ),
                )
            )
        man["questions"] = qs
        man["question_kinds"] = {
            k: sum(1 for q in qs if q["kind"] == k)
            for k in ("single", "multiple", "unknown")
        }
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    for p in pieces:
        print(
            f"SPLIT_PIECE,{p['piece']},{'+'.join(p['notes'])},tokens={p['tokens']},"
            f"K={p['k_total']},rows={p['rows']}",
            flush=True,
        )
    print(f"SPLIT_PREP_DONE {out} {man.get('question_kinds')}", flush=True)


def shift_keys(k, offset, theta):
    """Rotate stored keys [.., P, D] on by ``offset`` positions."""
    from cas_kv_rope import rotate

    if offset == 0:
        return k
    pos = torch.full((k.shape[-2],), offset, device=k.device)
    return rotate(k, pos, theta)


def compose_tensors(parts, theta=1e6):
    """parts: [(keys, values, nfrozen)] per piece, per-layer [1,H,P,D].
    Returns bundle (keys, values, nfrozen of the first piece)."""
    L = len(parts[0][0])
    keys, vals, off = [[] for _ in range(L)], [[] for _ in range(L)], 0
    for pk, pv, _ in parts:
        for i in range(L):
            keys[i].append(shift_keys(pk[i], off, theta))
            vals[i].append(pv[i])
        off += pk[0].shape[2]
    keys = [torch.cat(k, dim=2).contiguous() for k in keys]
    vals = [torch.cat(v, dim=2).contiguous() for v in vals]
    return keys, vals, parts[0][2]


def compose():
    from cas_cart_init import load_cart, save_cart

    paths = os.environ["PIECE_CARTS"].split(",")
    theta = float(os.environ.get("THETA", "1e6"))
    parts = [load_cart(p) for p in paths]
    keys, vals, nf = compose_tensors(parts, theta)
    save_cart(keys, vals, nf, os.environ["OUT"])
    sizes = [p[0][0].shape[2] for p in parts]
    print(f"COMPOSE_DONE {os.environ['OUT']} pieces={sizes} total={sum(sizes)}")


def selftest():
    from cas_kv_rope import derot

    torch.manual_seed(0)
    assert cut_points([10, 10, 10, 10], 2) == [(0, 2), (2, 4)]
    assert cut_points([30, 12], 2) == [(0, 1), (1, 2)]
    assert cut_points([7, 2, 1, 11, 6, 6, 3], 2) == [(0, 4), (4, 7)]
    assert allocate([71, 29], 613) == [435, 178]
    assert sum(allocate([3, 3, 3], 10)) == 10
    # a piece built alone at 0..P-1 and shifted by off must equal the same
    # pre-rotary keys rotated at off..off+P-1
    L, H, D = 2, 2, 8
    raw = [torch.randn(1, H, 5, D) for _ in range(L)]
    from cas_kv_rope import rerot

    a = [rerot(r[:, :, :3]) for r in raw]
    b = [rerot(r[:, :, 3:]) for r in raw]
    va = [torch.randn(1, H, 3, D) for _ in range(L)]
    vb = [torch.randn(1, H, 2, D) for _ in range(L)]
    keys, vals, nf = compose_tensors([(a, va, 1), (b, vb, 1)])
    for i in range(L):
        want = rerot(raw[i])
        assert torch.allclose(keys[i], want, atol=1e-5), (keys[i] - want).abs().max()
        assert torch.equal(vals[i][:, :, :3], va[i]) and torch.equal(
            vals[i][:, :, 3:], vb[i]
        )
        back = derot(keys[i])
        assert torch.allclose(back, raw[i], atol=1e-5)
    assert nf == 1
    print("CAS_SPLIT_SELFTEST PASS")


if __name__ == "__main__":
    {"prep": prep, "compose": compose, "selftest": selftest}[sys.argv[1]]()
