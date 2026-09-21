# SPDX-License-Identifier: GPL-2.0
"""Document-separated tasks for judging a translated cache by what it can do.

Divergence against a native cache says how different two distributions are.
It does not say whether anything a user asked for still works, and a
translator can look close in divergence while having lost the one fact the
prompt was about. These tasks are the other measurement: each one has a right
answer that can be graded by string comparison, and each one is impossible
without the document, so an arm that scores on them is demonstrably carrying
the document rather than the language.

Separation by document is the point of the construction. Contiguous slices of
one corpus share topic, style and often literal phrasing, so an arm evaluated
on a slice adjacent to the slices it was fitted on is being asked an easier
question than a deployment would ask. Here each evaluation item comes from a
distinct article, and articles used for fitting are excluded by identity
rather than by position.

Three task types, chosen because they fail in different ways:

Planted retrieval attaches an arbitrary code to a named entity at a random
depth of the document. Nothing in the model's weights can supply the code, so
an empty or mismatched cache scores zero by construction and the negative
controls are exact rather than approximate.

Cloze recall elides a distinctive span the document itself contains and asks
for it back. Unlike the planted code this is recoverable from context and
style, so it measures graded degradation where the planted code measures a
cliff.

Free generation has no right answer and is scored for pathology instead:
degenerate repetition, failing to stop, and running to the cap. A translator
that keeps divergence low while producing loops would pass every other
measurement in this lane and be unusable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


@dataclass
class Item:
    """One evaluation item, tied to the document it came from."""

    doc_id: str
    kind: str
    prompt: str
    query: str
    answer: str = ""
    meta: dict = field(default_factory=dict)


def _code(seed: str, n: int = 8) -> str:
    h = hashlib.sha256(seed.encode()).digest()
    return "".join(CODE_ALPHABET[b % len(CODE_ALPHABET)] for b in h[:n])


def _sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 40]


def wikitext_documents(
    min_chars: int = 2000, max_docs: int = 4096, split: str = "train[:3%]"
) -> list:
    """Articles, kept whole, identified by their own title.

    The corpus ships as a token stream with headings in it. Splitting on the
    heading form recovers the article boundaries, which is what makes
    document separation checkable: two items share a document only if they
    share a title.

    The default split is deliberately not the one the maps were fitted on.
    The smaller release of this corpus shares its held-out articles with the
    larger one, so drawing evaluation documents from either held-out split
    returns the very articles the fit already saw -- which the caller's own
    overlap check will reject, leaving nothing to evaluate. The training
    split is disjoint from those articles and large enough to choose from.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    lines = ds["text"]
    docs, title, buf = [], None, []
    head = re.compile(r"^ = ([^=].*?) = $")
    for ln in lines:
        m = head.match(ln.rstrip("\n"))
        if m:
            if title and sum(len(b) for b in buf) >= min_chars:
                docs.append((title, "".join(buf)))
            title, buf = m.group(1).strip(), []
        elif title is not None:
            buf.append(ln)
    if title and sum(len(b) for b in buf) >= min_chars:
        docs.append((title, "".join(buf)))
    return docs[:max_docs]


def build_items(
    docs: Sequence,
    tok,
    ctx: int,
    seed: int = 0,
    kinds: Sequence[str] = ("retrieval", "cloze", "free"),
    exclude: Optional[set] = None,
) -> list:
    """One item per document per kind, all sharing that document's prompt.

    The three kinds share a prompt deliberately: the same cache is then asked
    three different questions, so a difference between kinds is a difference
    in what the question needs rather than in what the cache holds.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    exclude = exclude or set()
    items = []
    for title, body in docs:
        if title in exclude:
            continue
        # Trim to roughly the window first, then plant inside it. Planting at
        # a fraction of the whole article and truncating afterwards drops the
        # fact outside the prompt for any article longer than the window,
        # which silently discarded about nine documents in ten.
        body_ids = tok(body, return_tensors="pt").input_ids[0]
        if body_ids.shape[0] < ctx:
            continue
        window_text = tok.decode(body_ids[: int(ctx * 1.15)], skip_special_tokens=True)
        sents = _sentences(window_text)
        if len(sents) < 6:
            continue
        code = _code(f"{seed}:{title}")
        entity = title
        planted = f" The archive registration code for {entity} is {code}. "
        # Spread across most of what the cache will hold rather than confined
        # to the middle. Confining it to the middle avoids the easier edges,
        # but it also guarantees that a baseline which natively prefills a
        # short suffix can never contain the fact, which would decide that
        # comparison by construction instead of by measurement. The depth is
        # recorded per item so the result can be read against it.
        lo, hi = max(1, int(0.10 * len(sents))), max(2, int(0.92 * len(sents)))
        at = int(rng.integers(lo, hi))
        text = " ".join(sents[:at]) + planted + " ".join(sents[at:])

        ids = tok(text, return_tensors="pt").input_ids[0]
        if ids.shape[0] < ctx:
            continue
        prompt_ids = ids[:ctx]
        prompt = tok.decode(prompt_ids, skip_special_tokens=True)
        if code not in prompt:
            continue  # the plant fell outside the window; drop rather than lie
        # Carried as ids, not as text. Decoding and re-encoding a prompt does
        # not always return the same tokens, and a prompt that came back one
        # token short would silently change the context length every arm is
        # being compared at.
        pid = prompt_ids.tolist()

        if "retrieval" in kinds:
            items.append(
                Item(
                    doc_id=title,
                    kind="retrieval",
                    prompt=prompt,
                    query=(
                        f"\n\nQuestion: What is the archive registration code for "
                        f"{entity}?\nAnswer with the code only.\nAnswer:"
                    ),
                    answer=code,
                    meta={"depth_frac": at / max(len(sents), 1), "prompt_ids": pid},
                )
            )

        if "cloze" in kinds:
            # A span the document contains and the query does not, taken from
            # the part of the prompt the cache actually holds.
            cand = [
                s
                for s in _sentences(prompt)
                if len(s.split()) >= 10 and code not in s and entity not in s
            ]
            if cand:
                s = cand[int(rng.integers(0, len(cand)))]
                words = s.split()
                cut = max(5, len(words) // 2)
                stem, tail = " ".join(words[:cut]), " ".join(words[cut : cut + 5])
                if tail:
                    items.append(
                        Item(
                            doc_id=title,
                            kind="cloze",
                            prompt=prompt,
                            query=(
                                "\n\nComplete this sentence from the passage "
                                f"above, exactly as written.\n{stem}"
                            ),
                            answer=tail,
                            meta={"stem_words": cut, "prompt_ids": pid},
                        )
                    )

        if "free" in kinds:
            items.append(
                Item(
                    doc_id=title,
                    kind="free",
                    prompt=prompt,
                    query=(
                        "\n\nContinue the passage above in the same style."
                        "\n\nContinuation:"
                    ),
                    answer="",
                    meta={"prompt_ids": pid},
                )
            )
    return items


def normalise_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def grade_retrieval(answer: str, produced: str) -> dict:
    """Exact code match, plus whether the model produced any code at all.

    The second number separates a wrong answer from a refusal, which matters
    because a cache that destroys the fact and a cache that destroys the
    ability to answer are different failures.
    """
    found = re.findall(r"\b[A-Z0-9]{6,12}\b", produced.upper())
    return {
        "correct": float(answer.upper() in produced.upper()),
        "emitted_code": float(bool(found)),
        "first_code": found[0] if found else "",
    }


def grade_cloze(answer: str, produced: str) -> dict:
    """Token overlap with the elided span, and exact match on it."""
    want = normalise_text(answer).split()
    got = normalise_text(produced).split()[: max(len(want), 1) * 3]
    if not want:
        return {"exact": 0.0, "overlap": 0.0}
    hit = 0
    pool = list(got)
    for w in want:
        if w in pool:
            pool.remove(w)
            hit += 1
    return {
        "exact": float(normalise_text(produced).startswith(normalise_text(answer))),
        "overlap": hit / len(want),
    }


def grade_free(produced: str, token_ids, eos_ids, max_new: int) -> dict:
    """Pathology counts, not a quality score.

    Repetition is measured as the largest share any single four-gram takes of
    all four-grams produced. Healthy prose sits near zero; a loop drives it
    toward one.
    """
    ids = [t for t in token_ids if t not in eos_ids]
    grams = [tuple(ids[i : i + 4]) for i in range(max(len(ids) - 3, 0))]
    top = 0.0
    if grams:
        counts = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
        top = max(counts.values()) / len(grams)
    stopped = any(t in eos_ids for t in token_ids)
    text = normalise_text(produced)
    nonascii = sum(1 for c in produced if ord(c) > 127) / max(len(produced), 1)
    # A generation with fewer than four content tokens has no four-grams, so
    # the repetition statistic is not merely zero but undefined. Scoring it as
    # healthy would let the worst failure of all -- saying nothing -- earn a
    # perfect pathology score, which is the opposite of what this measures.
    too_short = len(ids) < 4
    degenerate = top > 0.25
    return {
        "repeat_4gram_frac": top,
        "degenerate": float(degenerate),
        "stopped": float(stopped),
        "cap_hit": float(not stopped and len(token_ids) >= max_new),
        "empty": float(len(text) == 0),
        "too_short": float(too_short),
        "nonascii_frac": nonascii,
        # The single number the gate reads. Every way a generation can be
        # unusable has to enter it, or an arm that collapses into silence or
        # into mojibake passes a criterion named for catching exactly that.
        "healthy": float(
            not degenerate and not too_short and len(text) > 0 and nonascii <= 0.10
        ),
    }
