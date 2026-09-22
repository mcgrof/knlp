# SPDX-License-Identifier: GPL-2.0
"""Training examples that ask for an answer, and end.

The correction has been trained to reproduce the target's continuation of a
document. It is good at that and it drops a character when asked to copy a
code: fourteen of sixteen near misses on the development set are deletions,
and the target itself and a plain suffix prefill never do this. Continuing
prose and emitting an exact token then stopping are different demands, and
only the first has ever been supervised.

These are the second kind. Half the schedule keeps the ordinary continuation
examples; the other half asks a question about the document the cache holds
and supervises the answer and the end of it.

Everything is built from training documents only. The codes are fresh, drawn
from a different seed stream than the evaluation set uses, so a code the
correction has seen cannot be a code it is later asked for. Question wording
is held back from the evaluation templates for the same reason. No
development document, and no code observed failing on one, is used here.

Two question kinds, matching what the evaluation asks:

Planted-code retrieval puts an arbitrary code in the document and asks for it
back. This is the behaviour the deletions appear in.

Span recall asks for a distinctive span the document already contains, so the
answer is longer than one token and the model has to stop in the right place
rather than at a fixed width.

Positions are balanced across the document rather than clustered, because a
correction trained only on facts near the end would learn where to look
instead of how to copy.

Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; CPU only.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from research.kv_translate.tasks import CODE_ALPHABET, _sentences

# Deliberately different from the evaluation's wording. A correction trained
# on the sentence it will be tested with has been taught the sentence.
TRAIN_CODE_Q = (
    "\n\nFrom the passage above, state the archive registration code for "
    "{entity}.\nCode:"
)
TRAIN_SPAN_Q = "\n\nQuote the next few words of the passage after: {stem}\nNext:"
PLANT = " The archive registration code for {entity} is {code}. "


@dataclass
class TrainExample:
    """One supervised example: a prompt, a question, and what to answer."""

    doc_id: str
    kind: str
    prompt_ids: list
    query: str
    answer: str
    meta: dict = field(default_factory=dict)


def train_code(seed: int, title: str, n: int = 8) -> str:
    """A code from a stream the evaluation set never draws from.

    The evaluation hashes ``f"{seed}:{title}"``. This prefixes the namespace
    so that no training code can collide with an evaluation answer even for a
    document that somehow appeared in both -- which the manifests already
    forbid, making this the second of two independent guards rather than the
    only one.
    """
    h = hashlib.sha256(f"train-objective:{seed}:{title}".encode()).digest()
    return "".join(CODE_ALPHABET[b % len(CODE_ALPHABET)] for b in h[:n])


def build_objective_examples(
    docs: Sequence,
    tok,
    ctx: int,
    seed: int = 0,
    forbid_codes: Optional[set] = None,
) -> list:
    """One retrieval and one span example per document, positions balanced.

    ``docs`` is a sequence of ``(title, body)`` for training documents only.
    ``forbid_codes`` is checked rather than trusted: any collision with a
    held-out answer aborts instead of being silently renamed.
    """
    import numpy as np

    rng = np.random.default_rng(seed + 9973)
    forbid = {c.upper() for c in (forbid_codes or set())}
    out = []
    for i, (title, body) in enumerate(docs):
        body_ids = tok(body, return_tensors="pt").input_ids[0]
        if body_ids.shape[0] < ctx:
            continue
        window = tok.decode(body_ids[: int(ctx * 1.15)], skip_special_tokens=True)
        sents = _sentences(window)
        if len(sents) < 6:
            continue

        code = train_code(seed, title)
        if code.upper() in forbid:
            raise SystemExit(
                f"INVALID: training code for {title!r} collides with a "
                "held-out answer; the code namespace is not disjoint"
            )

        # Balanced rather than random: the i-th document plants at a position
        # that walks the range, so the schedule covers depths evenly instead
        # of sampling them and happening to cluster.
        frac = 0.10 + 0.80 * ((i % 8) + 0.5) / 8.0
        at = max(1, min(len(sents) - 1, int(round(frac * len(sents)))))
        text = (
            " ".join(sents[:at])
            + PLANT.format(entity=title, code=code)
            + " ".join(sents[at:])
        )
        ids = tok(text, return_tensors="pt").input_ids[0]
        if ids.shape[0] < ctx:
            continue
        pid = ids[:ctx].tolist()
        prompt = tok.decode(ids[:ctx], skip_special_tokens=True)
        if code not in prompt:
            continue  # the plant fell outside the window

        out.append(
            TrainExample(
                doc_id=title,
                kind="code",
                prompt_ids=pid,
                query=TRAIN_CODE_Q.format(entity=title),
                answer=" " + code,
                meta={"depth_frac": at / max(len(sents), 1), "code": code},
            )
        )

        cand = [
            s
            for s in _sentences(prompt)
            if len(s.split()) >= 10 and code not in s and title not in s
        ]
        if cand:
            s_ = cand[int(rng.integers(0, len(cand)))]
            w = s_.split()
            cut = max(5, len(w) // 2)
            stem, tail = " ".join(w[:cut]), " ".join(w[cut : cut + 5])
            if tail:
                out.append(
                    TrainExample(
                        doc_id=title,
                        kind="span",
                        prompt_ids=pid,
                        query=TRAIN_SPAN_Q.format(stem=stem),
                        answer=" " + tail,
                        meta={"stem_words": cut},
                    )
                )
    return out


def supervised_targets(tok, example, eos_id):
    """Token ids for the question, and the labels for the answer plus its end.

    Returns ``(query_ids, label_ids)`` where the labels are the answer tokens
    followed by end-of-sequence. The first answer token is predicted from the
    query's final position, which is what makes the answer supervised at all:
    shifting the labels by one the other way trains the model to predict the
    second answer token from the first and never to produce the first.

    Nothing after end-of-sequence is supervised. Padding an answer to match a
    token count would be training the model to keep talking after it has
    finished, which is the behaviour the probe exists to correct.
    """
    q = tok(example.query, return_tensors="pt", add_special_tokens=False).input_ids
    a = tok(example.answer, return_tensors="pt", add_special_tokens=False).input_ids
    labels = a[0].tolist() + [eos_id]
    return q, labels


def verify_examples(tok, examples, eos_id, gold_answers=None):
    """Refuse a schedule that cannot teach what it claims to.

    Three ways an example set can be quietly useless: the answer is not
    actually in the prompt the cache will hold, so the model is being asked to
    invent it; the answer tokenises to nothing; or a training code is also a
    held-out answer.
    """
    problems = []
    forbid = {c.upper() for c in (gold_answers or set())}
    for e in examples:
        # Both sides normalised. A span lifted by a sentence splitter can
        # carry a space where the document has a newline, and comparing the
        # two raw is the same one-sided-normalisation mistake that once made
        # a corpus-overlap check unable to fire across a line break.
        prompt = normalise(tok.decode(e.prompt_ids, skip_special_tokens=True))
        ans = normalise(e.answer)
        if ans not in prompt:
            problems.append(f"{e.doc_id}/{e.kind}: answer {ans!r} is not in the prompt")
        _, labels = supervised_targets(tok, e, eos_id)
        if len(labels) < 2:
            problems.append(f"{e.doc_id}/{e.kind}: answer tokenises to nothing")
        if labels[-1] != eos_id:
            problems.append(f"{e.doc_id}/{e.kind}: labels do not end at EOS")
        if e.kind == "code" and e.meta["code"].upper() in forbid:
            problems.append(f"{e.doc_id}: training code is a held-out answer")
    return problems


def normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())
