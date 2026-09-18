"""Deterministic final-answer extraction and exact numeric grading.

Extraction policy, applied in order, with the rule that fired recorded:

1. ``boxed``: the content of the last ``\\boxed{...}`` (balanced braces).
   The content must normalize to exactly one number; units, currency
   signs, ``\\text{}`` wrappers, thousands separators and a trailing
   percent sign are stripped.  Several distinct numbers inside the box
   (for example ``3, 4``) are *malformed*, not guessed.
2. ``last_number``: otherwise the last number in the response, after
   removing thousands separators.  This matches the upstream GSM8K
   verifier the screening protocol is modelled on, so observed-zero
   buckets are comparable with it.
3. ``none``: no number at all; graded as incorrect.

Numbers compare exactly as rationals (``Fraction``), so ``18``,
``18.0``, ``18.00`` and ``36/2`` are the same answer.  Truncated
responses are graded with the same rules; the caller records the
truncation flag separately.  The grader never returns the gold answer
or anything derived from it except a boolean.
"""

from __future__ import annotations

import re
from decimal import InvalidOperation
from fractions import Fraction

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_NUM_RE = re.compile(_NUM)
_FRAC_RE = re.compile(r"\\[dt]?frac\{\s*(" + _NUM + r")\s*\}\{\s*(" + _NUM + r")\s*\}")
_SLASH_RE = re.compile(r"^(" + _NUM + r")\s*/\s*(" + _NUM + r")$")
_THOUSANDS_RE = re.compile(r"(\d),(?=\d{3}(?!\d))")


def _last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def to_fraction(s: str) -> Fraction | None:
    s = s.strip().rstrip(".")
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def _clean_box(content: str) -> str:
    s = content
    s = re.sub(r"\\text\{[^{}]*\}", " ", s)
    s = re.sub(r"\\(?:mathrm|textbf|mbox)\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\$", " ").replace("$", " ").replace("\\%", " ")
    s = s.replace("%", " ").replace("\\,", "").replace("\\!", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = _THOUSANDS_RE.sub(r"\1", s)
    return s


def normalize_boxed(content: str) -> tuple[Fraction | None, str]:
    """Return (value, status) for the content of a boxed answer."""
    s = _clean_box(content)
    m = _FRAC_RE.search(s)
    if m:
        rest = _FRAC_RE.sub(" ", s)
        if _NUM_RE.search(rest):
            return None, "malformed_multiple"
        num, den = to_fraction(m.group(1)), to_fraction(m.group(2))
        if num is None or den is None or den == 0:
            return None, "malformed_fraction"
        return num / den, "ok"
    sm = _SLASH_RE.match(s.strip())
    if sm:
        num, den = to_fraction(sm.group(1)), to_fraction(sm.group(2))
        if num is None or den is None or den == 0:
            return None, "malformed_fraction"
        return num / den, "ok"
    nums = _NUM_RE.findall(s)
    if not nums:
        return None, "malformed_empty"
    vals = {to_fraction(n) for n in nums}
    vals.discard(None)
    if len(vals) != 1:
        return None, "malformed_multiple"
    return vals.pop(), "ok"


def extract_answer(text: str) -> dict:
    """Extract a final answer.  Returns value (Fraction|None) and rule."""
    boxed = _last_boxed(text)
    if boxed is not None:
        val, status = normalize_boxed(boxed)
        return {"value": val, "rule": "boxed", "status": status, "raw": boxed}
    flat = _THOUSANDS_RE.sub(r"\1", text)
    nums = _NUM_RE.findall(flat)
    if nums:
        val = to_fraction(nums[-1])
        return {
            "value": val,
            "rule": "last_number",
            "status": "ok" if val is not None else "malformed_number",
            "raw": nums[-1],
        }
    return {"value": None, "rule": "none", "status": "no_answer", "raw": None}


def parse_gold(answer_field: str) -> Fraction:
    """Parse the gold value from a GSM8K ``answer`` field (after ####)."""
    tail = answer_field.split("####")[-1]
    tail = _THOUSANDS_RE.sub(r"\1", tail).replace("$", "").strip()
    val = to_fraction(tail)
    if val is None:
        nums = _NUM_RE.findall(tail)
        if len(nums) != 1:
            raise ValueError(f"unparseable gold answer: {tail!r}")
        val = to_fraction(nums[0])
    return val


def answer_key(value: Fraction | None) -> str | None:
    """Canonical string for an extracted value (used for collision counts)."""
    if value is None:
        return None
    return f"{value.numerator}/{value.denominator}"


def grade(text: str, gold: Fraction) -> dict:
    """Grade one completion.  The returned dict carries no gold field."""
    ex = extract_answer(text)
    correct = ex["value"] is not None and ex["value"] == gold
    return {
        "correct": bool(correct),
        "rule": ex["rule"],
        "status": ex["status"],
        "answer": answer_key(ex["value"]),
    }
