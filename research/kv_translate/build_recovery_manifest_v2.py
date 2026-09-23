# SPDX-License-Identifier: GPL-2.0
"""Version two of the recovery fixture: 64 documents, with cloze provenance.

Three things are wrong with version one and are fixed here.

Its cloze answers were drawn from the whole assembled context, so some of them
land in training filler rather than in the article the row is labelled with.
That is not answer leakage, but it does weaken the thing the experiment
measures: a question about a document should be answerable from that document.
Version two chooses the span inside the article's own retained text and records
where it came from, in article characters and in assembled token offsets, with
the decoder reconstruction beside it so the claim can be checked rather than
trusted.

Its quartile assignment was positional, so extending the same builder to 64
documents would move twelve of the sixteen screened documents into different
quartiles. Assignment is now a function of the document's own index in the
exposed order, which leaves the screened sixteen exactly where they were.

Its random draws came from one stream shared across the build, so a document's
filler depended on how many documents were generated before it. New contexts
are now seeded from the document's identity and the build version. The sixteen
screened contexts are reproduced by replaying version one's stream so their
prompts stay byte-identical, and that is asserted rather than assumed.

Token hashes are little-endian int64 and say so. Version one hashed native
byte order without declaring it, and the older validator expects JSON-encoded
ids; both readings are recorded so neither is silently reinterpreted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.kv_translate.tasks import (
    _code,
    _sentences,
    wikitext_documents,
)  # noqa: E402

VERSION = "v2"
QUARTILES = ((0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.00))
FORBIDDEN_ROLES = ("confirm128", "reserve384")
SCREEN_STRIDE = 4


def tok_hash_le(ids):
    """Little-endian int64, declared. Byte order is part of the contract."""
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def tok_hash_json(ids):
    """What the older validator computes, kept so neither is reinterpreted."""
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def doc_seed(title, version, base):
    """Stable across builds and independent of how many documents precede."""
    h = hashlib.sha256(f"{version}:{base}:{title}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def assemble(tok, body_ids, filler_ids, plant_ids, target_tok, ctx, rng):
    """Build one context, recording which assembled tokens the article owns.

    Provenance is returned as assembled-token ranges so a cloze span can be
    confined to the article and so a reader can check that confinement rather
    than take it on faith.
    """
    need_lead = max(0, target_tok)
    lead = body_ids[:need_lead]
    owned = []
    if lead.shape[0] > 0:
        owned.append((0, int(lead.shape[0]), 0, int(lead.shape[0])))
    if lead.shape[0] < need_lead:
        short = need_lead - lead.shape[0]
        off = int(rng.integers(0, max(1, filler_ids.shape[0] - short - 1)))
        lead = torch.cat([lead, filler_ids[off : off + short]])
    pos = int(lead.shape[0])
    plant_span = (pos, pos + int(plant_ids.shape[0]))
    pos += int(plant_ids.shape[0])
    rest_need = ctx - lead.shape[0] - plant_ids.shape[0]
    tail = body_ids[need_lead : need_lead + rest_need]
    if tail.shape[0] > 0:
        owned.append(
            (pos, pos + int(tail.shape[0]), need_lead, need_lead + int(tail.shape[0]))
        )
    if tail.shape[0] < rest_need:
        short = rest_need - tail.shape[0]
        off = int(rng.integers(0, max(1, filler_ids.shape[0] - short - 1)))
        tail = torch.cat([tail, filler_ids[off : off + short]])
    ids = torch.cat([lead, plant_ids, tail])[:ctx]
    owned = [(a, min(b, ctx), c, d) for a, b, c, d in owned if a < ctx]
    return ids, owned, plant_span


def _is_prose(s: str) -> bool:
    """Reject the corpus's section markup as a cloze answer.

    The sentence splitter treats a run of heading markers as a sentence, so a
    span like "= = = Picturesque Children" can be selected. Asking a model to
    complete heading markup measures nothing about whether it remembers the
    document, and its whitespace does not survive a decode round trip intact.
    """
    w = s.split()
    if not w:
        return False
    marks = sum(1 for t in w if set(t) <= {"="})
    return marks == 0 and "=" not in s


def pick_cloze(tok, ids, owned, plant_span, code, title, rng, margin=8):
    """A stem and answer wholly inside the article's own retained tokens.

    The margin keeps the span off the splice boundaries, where the article
    meets filler or the planted sentence, so the answer is not a sentence that
    only exists because of how the context was assembled.
    """
    for a, b, ca, cb in sorted(owned, key=lambda r: -(r[1] - r[0])):
        lo, hi = a + margin, b - margin
        if hi - lo < 64:
            continue
        if not (hi <= plant_span[0] or lo >= plant_span[1]):
            hi = min(hi, plant_span[0] - margin)
            if hi - lo < 64:
                continue
        text = tok.decode(ids[lo:hi], skip_special_tokens=True)
        cand = [
            s
            for s in _sentences(text)
            if len(s.split()) >= 10
            and code not in s
            and title not in s
            and _is_prose(s)
        ]
        if not cand:
            continue
        s = cand[int(rng.integers(0, len(cand)))]
        w = s.split()
        cut = max(5, len(w) // 2)
        stem, gold = " ".join(w[:cut]), " ".join(w[cut : cut + 5])
        if not gold or gold in stem:
            continue
        # The span must reconstruct from what was recorded, or the record does
        # not identify the answer it claims to.
        if gold not in text:
            continue
        return {
            "stem": stem,
            "gold": gold,
            "stem_words": cut,
            "assembled_token_span": [int(lo), int(hi)],
            "article_token_span": [int(ca), int(cb)],
            "source_article": title,
            "decoder_reconstruction_contains_gold": gold in text,
        }
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--v1", required=True, help="the v1 record, preserved unchanged")
    ap.add_argument("--filler-role", default="train308")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--target-revision", default="main")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--docs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert args.filler_role not in FORBIDDEN_ROLES
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.target, revision=args.target_revision)
    gold = json.load(open(args.gold))
    man = json.load(open(args.manifests))
    v1 = json.load(open(args.v1))
    v1_by_doc = {d["doc"]: d for d in v1["documents"]}

    exposed = [d["doc"] for d in gold["documents"]]
    bodies = dict(wikitext_documents(split="train[:3%]", max_docs=100000))
    fill_titles = [e["doc"] for e in man["roles"][args.filler_role]]
    overlap = sorted(set(fill_titles) & set(exposed))
    if overlap:
        raise SystemExit(f"FIXTURE_BLOCKED: filler overlaps evaluation: {overlap[:3]}")
    filler_ids = tok(
        "\n\n".join(bodies[t] for t in fill_titles if t in bodies),
        return_tensors="pt",
    ).input_ids[0]

    screen_idx = list(range(0, len(exposed), SCREEN_STRIDE))[:16]
    # Version one's shared stream, replayed in its original order, so the
    # screened prompts come back byte-identical.
    v1_rng = np.random.default_rng(args.seed)

    docs_out, problems = [], []
    for j, title in enumerate(exposed[: args.docs]):
        if title not in bodies:
            problems.append(f"{title}: body not recoverable")
            continue
        q = (j // 4) % 4
        q_lo, q_hi = QUARTILES[q]
        code = _code(f"{args.seed}:{title}")
        planted = f" The archive registration code for {title} is {code}. "
        plant_ids = tok(planted, return_tensors="pt").input_ids[0]
        body_ids = tok(bodies[title], return_tensors="pt").input_ids[0]
        target_tok = int(((q_lo + q_hi) / 2.0) * args.ctx)

        in_screen = j in screen_idx
        rng = (
            v1_rng
            if in_screen
            else np.random.default_rng(doc_seed(title, VERSION, args.seed))
        )
        ids, owned, plant_span = assemble(
            tok, body_ids, filler_ids, plant_ids, target_tok, args.ctx, rng
        )
        if ids.shape[0] != args.ctx:
            problems.append(f"{title}: built {ids.shape[0]} tokens")
            continue
        pid = ids.tolist()
        text = tok.decode(ids, skip_special_tokens=True)
        if code not in text:
            problems.append(f"{title}: plant outside the window")
            continue

        # The screened prompts must be the version-one prompts, exactly.
        if in_screen and title in v1_by_doc:
            if pid != v1_by_doc[title]["prompt_ids"]:
                problems.append(f"{title}: screened prompt differs from v1")
                continue

        cz = pick_cloze(tok, ids, owned, plant_span, code, title, rng)
        if cz is None:
            problems.append(f"{title}: no cloze span inside its own article")
            continue

        depth_tok = next(
            (
                k
                for k in range(1, args.ctx + 1)
                if code in tok.decode(ids[:k], skip_special_tokens=True)
            ),
            None,
        )
        depth = None if depth_tok is None else depth_tok / args.ctx
        if depth is None or not (q_lo <= depth < q_hi):
            problems.append(f"{title}: depth {depth} outside quartile {q}")
            continue

        rq = (
            f"\n\nQuestion: What is the archive registration code for "
            f"{title}?\nAnswer with the code only.\nAnswer:"
        )
        cq = (
            "\n\nComplete this sentence from the passage above, exactly as "
            f"written.\n{cz['stem']}"
        )
        fq = "\n\nContinue the passage above in the same style.\n\nContinuation:"
        if code in rq or cz["gold"] in cq:
            problems.append(f"{title}: an answer appears in its own query")
            continue

        docs_out.append(
            {
                "doc": title,
                "dev_index": j,
                "in_screen": in_screen,
                "quartile": q,
                "quartile_rule": "floor(dev_index / 4) % 4",
                "quartile_range": [q_lo, q_hi],
                "plant_depth_tokens": depth_tok,
                "plant_depth_frac": depth,
                "plant_token_span": list(plant_span),
                "article_owned_spans": [list(o) for o in owned],
                "prompt_len": args.ctx,
                "prompt_ids": pid,
                "prompt_ids_sha256_le_i64": tok_hash_le(pid),
                "prompt_ids_sha256_json": tok_hash_json(pid),
                "retrieval_gold": code,
                "retrieval_query": rq,
                "cloze": cz,
                "cloze_gold": cz["gold"],
                "cloze_query": cq,
                "cloze_stem_words": cz["stem_words"],
                "free_query": fq,
            }
        )

    # Deterministic wrong-source partner, and a check that it does not happen
    # to contain the same answer.
    by_doc = {d["doc"]: d for d in docs_out}
    order = [d["doc"] for d in docs_out]
    collisions = []
    for i, d in enumerate(docs_out):
        partner = order[(i + 1) % len(order)]
        d["wrong_source_doc"] = partner
        ptext = " ".join(
            tok.decode(by_doc[partner]["prompt_ids"], skip_special_tokens=True).split()
        )
        own = " ".join(tok.decode(d["prompt_ids"], skip_special_tokens=True).split())
        d["retrieval_gold_in_own_context"] = d["retrieval_gold"] in own
        d["cloze_gold_in_own_context"] = " ".join(d["cloze_gold"].split()) in own
        rg_in_partner = d["retrieval_gold"] in ptext
        cg_in_partner = " ".join(d["cloze_gold"].split()) in ptext
        d["retrieval_gold_in_wrong_source"] = rg_in_partner
        d["cloze_gold_in_wrong_source"] = cg_in_partner
        if rg_in_partner or cg_in_partner:
            collisions.append(
                {
                    "doc": d["doc"],
                    "partner": partner,
                    "retrieval": rg_in_partner,
                    "cloze": cg_in_partner,
                }
            )
        if not d["retrieval_gold_in_own_context"] or not d["cloze_gold_in_own_context"]:
            problems.append(f"{d['doc']}: a gold answer is absent from its own context")

    rec = {
        "contract": "recovery_manifest_v2",
        "version": VERSION,
        "supersedes": {
            "file": os.path.basename(args.v1),
            "joint_sha256": v1.get("joint_sha256"),
            "retained": "v1 is unchanged and remains the v1 record",
        },
        "ctx": args.ctx,
        "docs_requested": args.docs,
        "docs_built": len(docs_out),
        "screen_dev_indices": screen_idx,
        "screen_built": sum(1 for d in docs_out if d["in_screen"]),
        "per_quartile": {
            q: sum(1 for d in docs_out if d["quartile"] == q) for q in range(4)
        },
        "quartile_rule": "floor(dev_index / 4) % 4, so the screened sixteen keep v1's quartiles",
        "seeding": (
            "screened contexts replay v1's shared stream so their prompts stay "
            "byte-identical; new contexts are seeded from document identity and "
            "build version, never from generation order or count"
        ),
        "hash_contract": {
            "token_hash": "sha256 over little-endian int64 token ids",
            "v1_adapter": "v1 hashed native-endian int64 without declaring it; "
            "the JSON-encoded form the older validator expects is "
            "also recorded per document as prompt_ids_sha256_json",
        },
        "cloze_contract": (
            "stem and answer lie wholly inside the article's own retained "
            "tokens, off the splice boundaries and clear of the planted "
            "sentence; the source span is recorded in article and assembled "
            "coordinates"
        ),
        "filler_role": args.filler_role,
        "holdouts_untouched": list(FORBIDDEN_ROLES),
        "pins": {
            "target": args.target,
            "target_revision": args.target_revision,
            "dataset": man["dataset"],
            "dev_split": "train[:3%]",
            "filler_split": man["roles"][args.filler_role][0]["split"],
            "tokenizer": man["tokenizer"],
            "input_gold": os.path.abspath(args.gold),
            "input_manifests": os.path.abspath(args.manifests),
        },
        "wrong_source_collisions": collisions,
        "problems": problems,
        "documents": docs_out,
    }
    rec["prompt_joint_sha256"] = hashlib.sha256(
        "".join(d["prompt_ids_sha256_le_i64"] for d in docs_out).encode()
    ).hexdigest()
    rec["task_joint_sha256"] = hashlib.sha256(
        json.dumps(
            [
                {
                    "doc": d["doc"],
                    "p": d["prompt_ids_sha256_le_i64"],
                    "rq": d["retrieval_query"],
                    "rg": d["retrieval_gold"],
                    "cq": d["cloze_query"],
                    "cg": d["cloze_gold"],
                    "fq": d["free_query"],
                    "ws": d["wrong_source_doc"],
                    "prov": d["cloze"]["assembled_token_span"],
                }
                for d in docs_out
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    json.dump(rec, open(args.out, "w"), indent=2)

    print(f"built {len(docs_out)} of {args.docs}; screen {rec['screen_built']}/16")
    print(f"  per quartile: {rec['per_quartile']}")
    print(f"  wrong-source answer collisions: {len(collisions)}")
    print(f"  prompt joint {rec['prompt_joint_sha256'][:16]}")
    print(f"  task joint   {rec['task_joint_sha256'][:16]}")
    if problems:
        print(f"  problems {len(problems)}: {problems[:4]}")
    ok = len(docs_out) == args.docs and rec["screen_built"] == 16 and not collisions
    if not ok:
        print("FIXTURE_BLOCKED", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
