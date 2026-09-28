#!/usr/bin/env python3
"""Binary correctness for short answers, and what counts as a broken answer.

A document question is correct when the model's answer, normalised, equals
the dataset's answer or one of its accepted aliases, normalised the same way.
The normalisation is the one the question answering literature has used since
SQuAD: lower case, punctuation and articles removed, white space collapsed.
Nothing softer is called correct. Token overlap and containment are recorded
beside the score as diagnostics and never decide it.

Generation health is judged apart from correctness. An answer is unhealthy
when it is empty, when it never stops, when it opens a reasoning block and
never closes it, or when it repeats itself. A wrong but well formed answer
is healthy.
"""

from __future__ import annotations

import re
import string
import unicodedata

THINK = re.compile(r"<think>.*?</think>", re.S)
LEAD = re.compile(r"^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[:\-]?\s*", re.I)
ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCT = str.maketrans("", "", string.punctuation)

REPEAT_LINES = 8
REPEAT_SPAN = 24
REPEAT_TIMES = 6


def normalise(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.translate(PUNCT)
    text = ARTICLES.sub(" ", text)
    return " ".join(text.split())


def strip_thinking(output):
    """The text after any reasoning block, or None when one never closed."""
    text = THINK.sub("", output)
    if "<think>" in text:
        return None
    return text


def extract(output):
    """The answer a model gave: its first non-empty line, without a lead-in."""
    text = strip_thinking(output)
    if text is None:
        return None, "unterminated_thinking"
    for line in text.splitlines():
        line = line.strip()
        if line:
            line = LEAD.sub("", line).strip().strip("\"'`*").strip()
            if line:
                return line, "first_line"
    return None, "empty"


def exact_match(prediction, accepted):
    if prediction is None:
        return False
    p = normalise(prediction)
    return bool(p) and any(p == normalise(a) for a in accepted if normalise(a))


def token_f1(prediction, accepted):
    """Best token overlap with any accepted answer. A diagnostic, not a score."""
    if prediction is None:
        return 0.0
    p = normalise(prediction).split()
    best = 0.0
    for a in accepted:
        g = normalise(a).split()
        if not p or not g:
            continue
        common = sum(min(p.count(t), g.count(t)) for t in set(p))
        if common:
            pr, rc = common / len(p), common / len(g)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best


def contains(prediction, accepted):
    if prediction is None:
        return False
    p = f" {normalise(prediction)} "
    return any(f" {normalise(a)} " in p for a in accepted if normalise(a))


def repeats(text):
    """Does the text loop on a line or on a span of characters?"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    run = 1
    for a, b in zip(lines, lines[1:]):
        run = run + 1 if a == b else 1
        if run >= REPEAT_LINES:
            return True
    flat = " ".join(text.split())
    for i in range(0, max(0, len(flat) - REPEAT_SPAN * REPEAT_TIMES) + 1):
        span = flat[i : i + REPEAT_SPAN]
        if flat.startswith(span * REPEAT_TIMES, i):
            return True
    return False


def health(output, *, stopped, new_tokens, max_new_tokens, parsed):
    """The ways one generation can be broken. An empty list is a healthy one.

    ``stopped`` is whether the model emitted its own end of turn. ``parsed``
    is whether the domain's parser found an answer of the expected form.
    """
    events = []
    text = strip_thinking(output)
    if text is None:
        events.append("unterminated_thinking")
        text = output
    if not text.strip():
        events.append("empty")
    if not stopped and new_tokens >= max_new_tokens:
        events.append("unterminated")
    if repeats(text):
        events.append("repetition")
    if not parsed and "empty" not in events:
        events.append("malformed")
    return events


def score_document(output, accepted, *, stopped, new_tokens, max_new_tokens):
    answer, how = extract(output)
    return {
        "correct": exact_match(answer, accepted),
        "answer": answer,
        "parser": how,
        "token_f1": round(token_f1(answer, accepted), 4),
        "contains_accepted": contains(answer, accepted),
        "health_events": health(
            output,
            stopped=stopped,
            new_tokens=new_tokens,
            max_new_tokens=max_new_tokens,
            parsed=answer is not None,
        ),
    }
