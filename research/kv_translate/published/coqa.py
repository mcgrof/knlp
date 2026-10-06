#!/usr/bin/env python3
"""Frozen CoQA protocol helpers for cross-model cache-transfer evaluation.

This module deliberately contains no model-loading code.  It owns the parts of
the protocol that must be inspectable and testable on CPU: conversation
selection, prompt construction, the official CoQA turn score, generation
health, and grouped summaries.  GPU execution lives in ``coqa_runner.py``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import re
import string

import numpy as np

DOMAINS = ("mctest", "gutenberg", "race", "cnn", "wikipedia")
SELECTION_SALT = "knlp-coqa-q1-v1:"
Q1_TURNS = (1, 5, 10)
MAX_NEW_TOKENS = 64
BOOTSTRAP_SEED = 20261001
BOOTSTRAP_REPLICATES = 10_000

INSTRUCTION = (
    "Answer each question using the passage. Give only a short answer.\n"
    "If the passage does not contain the answer, say unknown."
)

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCTUATION = set(string.punctuation)


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def selection_digest(conversation_id: str) -> str:
    return sha256_bytes(f"{SELECTION_SALT}{conversation_id}".encode("utf-8"))


def select_conversations(dataset: dict, per_domain: int = 20):
    """Return the frozen Q1, smoke, and Q2 conversation selections.

    Selection is entirely label/model blind.  Each public domain is sorted by
    SHA256(salt + conversation id); the first ``per_domain`` rows are Q1, the
    next row is smoke, and the first four Q1 rows are the conditional Q2 set.
    """
    by_domain = defaultdict(list)
    for story in dataset["data"]:
        by_domain[story["source"]].append(story)
    extra = sorted(set(by_domain) - set(DOMAINS))
    missing = sorted(set(DOMAINS) - set(by_domain))
    if extra or missing:
        raise ValueError(f"unexpected CoQA domains: extra={extra}, missing={missing}")

    q1, smoke, q2 = [], [], []
    for domain in DOMAINS:
        ordered = sorted(
            by_domain[domain], key=lambda row: (selection_digest(row["id"]), row["id"])
        )
        if len(ordered) < per_domain + 1:
            raise ValueError(f"{domain} has only {len(ordered)} conversations")
        q1.extend(ordered[:per_domain])
        smoke.append(ordered[per_domain])
        q2.extend(ordered[:4])
    return {"q1": q1, "smoke": smoke, "q2": q2}


def answer_references(story: dict, turn: int) -> list[str]:
    """All official references for one one-indexed turn, primary first."""
    i = turn - 1
    refs = [story["answers"][i]["input_text"]]
    for key in sorted(story.get("additional_answers", {}), key=str):
        answers = story["additional_answers"][key]
        if i < len(answers):
            refs.append(answers[i]["input_text"])
    return refs


def primary_answers(story: dict) -> list[str]:
    return [row["input_text"] for row in story["answers"]]


def render_prompt(story: dict, turn: int, past_answers: list[str] | None = None) -> str:
    """Render the exact Q0 prompt through ``turn`` (one indexed).

    Q1 passes the primary references as history.  Q2 maintains token ledgers
    instead of calling this after turn one, because re-tokenizing generated
    history could silently change an already cached prefix.
    """
    questions = story["questions"]
    if turn < 1 or turn > len(questions):
        raise ValueError(f"turn {turn} is outside 1..{len(questions)}")
    past = primary_answers(story) if past_answers is None else past_answers
    if len(past) < turn - 1:
        raise ValueError(f"turn {turn} needs {turn - 1} past answers, got {len(past)}")

    out = [INSTRUCTION, "", f"Passage: {story['story']}", ""]
    for i in range(turn):
        out.append(f"Question: {questions[i]['input_text']}")
        if i + 1 == turn:
            out.append("Answer:")
        else:
            out.append(f"Answer: {past[i]}")
    return "\n".join(out)


def next_turn_suffix(question: str, prior_stop_reason: str) -> str:
    """Text appended after a generated answer in a Q2 token ledger.

    A generated newline is retained in the ledger and already separates the
    next question.  EOS and cap stops do not, so those cases add one newline.
    The returned text is tokenized independently and appended; the existing
    prefix is never re-tokenized.
    """
    lead = "" if prior_stop_reason == "newline" else "\n"
    return f"{lead}Question: {question}\nAnswer:"


def normalize_answer(text: str) -> str:
    """The official CoQA/SQuAD normalization, byte-for-byte in behavior."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def answer_tokens(text: str) -> list[str]:
    return normalize_answer(text).split() if text else []


def exact_match(gold: str, prediction: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(prediction))


def token_f1(gold: str, prediction: str) -> float:
    gold_tokens = answer_tokens(gold)
    pred_tokens = answer_tokens(prediction)
    common = Counter(gold_tokens) & Counter(pred_tokens)
    same = sum(common.values())
    if not gold_tokens or not pred_tokens:
        return float(gold_tokens == pred_tokens)
    if not same:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def turn_score(references: list[str], prediction: str) -> dict[str, float]:
    """Official CoQA multiple-reference score for one turn.

    With multiple references, each annotator is held out in turn and the
    prediction receives its best score against the remaining references; the
    held-out results are averaged.  This is intentionally not a simple max.
    """
    if not references:
        raise ValueError("a CoQA turn needs at least one reference")
    em_sum = f1_sum = 0.0
    for i in range(len(references)):
        gold = (
            references if len(references) == 1 else references[:i] + references[i + 1 :]
        )
        em_sum += max(exact_match(answer, prediction) for answer in gold)
        f1_sum += max(token_f1(answer, prediction) for answer in gold)
    n = float(len(references))
    return {"em": em_sum / n, "f1": f1_sum / n}


def repeated(text: str, span: int = 24, times: int = 6) -> bool:
    flat = " ".join(text.split())
    for i in range(max(0, len(flat) - span * times) + 1):
        chunk = flat[i : i + span]
        if chunk and flat.startswith(chunk * times, i):
            return True
    return False


def health_events(
    prediction: str,
    raw_text: str,
    *,
    stop_reason: str,
    new_tokens: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[str]:
    """Frozen per-question generation-health event definition."""
    events = []
    if not prediction.strip():
        events.append("empty")
    if stop_reason == "cap" and new_tokens >= max_new_tokens:
        events.append("unterminated")
    if "<think>" in raw_text and "</think>" not in raw_text:
        events.append("unterminated_thinking")
    if repeated(raw_text):
        events.append("repetition")
    return events


def _finite_rows(rows: list[dict], fields: tuple[str, ...]):
    for row in rows:
        for field in fields:
            if field not in row or not math.isfinite(float(row[field])):
                raise ValueError(f"row {row.get('row_id')} lacks finite {field}")


def _mean(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _group_rows(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["domain"]][row["conversation_id"]].append(row)
    return grouped


def summarize_q1(rows: list[dict], *, replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    """Summarize paired Q1 rows and apply the prospective point screen."""
    _finite_rows(rows, ("native_f1", "transfer_f1"))
    expected_arms = all("native_health" in r and "transfer_health" in r for r in rows)
    if not expected_arms:
        raise ValueError("every Q1 row needs both health-event lists")

    grouped = _group_rows(rows)
    if set(grouped) != set(DOMAINS):
        raise ValueError(f"Q1 domains are {sorted(grouped)}, expected {list(DOMAINS)}")

    domains = {}
    for domain in DOMAINS:
        vals = [r for conv in grouped[domain].values() for r in conv]
        domains[domain] = {
            "questions": len(vals),
            "conversations": len(grouped[domain]),
            "native_f1": _mean(r["native_f1"] for r in vals),
            "transfer_f1": _mean(r["transfer_f1"] for r in vals),
            "native_minus_transfer_f1": _mean(
                r["native_f1"] - r["transfer_f1"] for r in vals
            ),
            "native_unhealthy": sum(bool(r["native_health"]) for r in vals),
            "transfer_unhealthy": sum(bool(r["transfer_health"]) for r in vals),
        }
        domains[domain]["health_excess"] = (
            domains[domain]["transfer_unhealthy"] - domains[domain]["native_unhealthy"]
        ) / len(vals)

    pooled = {
        "questions": len(rows),
        "native_f1": _mean(r["native_f1"] for r in rows),
        "transfer_f1": _mean(r["transfer_f1"] for r in rows),
        "native_minus_transfer_f1": _mean(
            r["native_f1"] - r["transfer_f1"] for r in rows
        ),
        "native_unhealthy": sum(bool(r["native_health"]) for r in rows),
        "transfer_unhealthy": sum(bool(r["transfer_health"]) for r in rows),
        "answer_text_changed": sum(
            r["native_answer"] != r["transfer_answer"] for r in rows
        ),
        "answer_normalized_changed": sum(
            normalize_answer(r["native_answer"])
            != normalize_answer(r["transfer_answer"])
            for r in rows
        ),
    }
    pooled["health_excess"] = (
        pooled["transfer_unhealthy"] - pooled["native_unhealthy"]
    ) / len(rows)

    equal_domain_gap = _mean(
        domains[domain]["native_minus_transfer_f1"] for domain in DOMAINS
    )
    equal_domain_native = _mean(domains[d]["native_f1"] for d in DOMAINS)
    equal_domain_transfer = _mean(domains[d]["transfer_f1"] for d in DOMAINS)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_equal, boot_domain = [], {domain: [] for domain in DOMAINS}
    conv_ids = {domain: sorted(grouped[domain]) for domain in DOMAINS}
    for _ in range(replicates):
        gaps = []
        for domain in DOMAINS:
            ids = conv_ids[domain]
            picked = rng.choice(ids, size=len(ids), replace=True)
            sampled = [r for cid in picked for r in grouped[domain][str(cid)]]
            gap = _mean(r["native_f1"] - r["transfer_f1"] for r in sampled)
            gaps.append(gap)
            boot_domain[domain].append(gap)
        boot_equal.append(_mean(gaps))

    bootstrap = {
        "seed": BOOTSTRAP_SEED,
        "replicates": replicates,
        "unit": "complete conversations resampled within domain",
        "equal_domain_native_minus_transfer_f1_95_interval": [
            _percentile(boot_equal, 2.5),
            _percentile(boot_equal, 97.5),
        ],
        "by_domain_native_minus_transfer_f1_95_interval": {
            domain: [
                _percentile(boot_domain[domain], 2.5),
                _percentile(boot_domain[domain], 97.5),
            ]
            for domain in DOMAINS
        },
        "interpretation": "descriptive uncertainty, not population noninferiority",
    }

    requirements = {
        "equal_domain_deficit_at_most_0.03": equal_domain_gap <= 0.03,
        "each_domain_deficit_at_most_0.05": all(
            domains[d]["native_minus_transfer_f1"] <= 0.05 for d in DOMAINS
        ),
        "pooled_health_excess_at_most_0.02": pooled["health_excess"] <= 0.02,
    }
    return {
        "schema": "coqa_q1_result_v1",
        "pooled_official_turn_weighted": pooled,
        "equal_domain": {
            "native_f1": equal_domain_native,
            "transfer_f1": equal_domain_transfer,
            "native_minus_transfer_f1": equal_domain_gap,
        },
        "domains": domains,
        "turn_counts": dict(sorted(Counter(r["turn"] for r in rows).items())),
        "bootstrap": bootstrap,
        "screen": {
            "requirements": requirements,
            "promising": all(requirements.values()),
            "meaning": "prospective development point-estimate screen only",
        },
    }


def summarize_q2(rows: list[dict], *, replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    """Apply Q2 screens separately by schedule, receiver, and comparison."""
    _finite_rows(rows, ("transfer_f1", "shadow_f1", "native_only_f1"))
    cells = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for schedule in sorted({r["schedule"] for r in rows}):
        for receiver in sorted(
            {r["receiver"] for r in rows if r["schedule"] == schedule}
        ):
            cell_rows = [
                r
                for r in rows
                if r["schedule"] == schedule and r["receiver"] == receiver
            ]
            key = f"{schedule}:{receiver}"
            cell = {"turn_positions": len(cell_rows), "comparisons": {}}
            for label, f1_field, health_field in (
                ("native_shadow", "shadow_f1", "shadow_health"),
                ("native_only", "native_only_f1", "native_only_health"),
            ):
                by_domain = {}
                for domain in DOMAINS:
                    domain_rows = [r for r in cell_rows if r["domain"] == domain]
                    if not domain_rows:
                        continue
                    by_domain[domain] = {
                        "turn_positions": len(domain_rows),
                        "reference_f1": _mean(r[f1_field] for r in domain_rows),
                        "transfer_f1": _mean(r["transfer_f1"] for r in domain_rows),
                        "reference_unhealthy": sum(
                            bool(r[health_field]) for r in domain_rows
                        ),
                        "transfer_unhealthy": sum(
                            bool(r["transfer_health"]) for r in domain_rows
                        ),
                        "reference_minus_transfer_f1": _mean(
                            r[f1_field] - r["transfer_f1"] for r in domain_rows
                        ),
                        "health_excess": _mean(
                            bool(r["transfer_health"]) - bool(r[health_field])
                            for r in domain_rows
                        ),
                    }
                by_turn = {}
                for turn in sorted({r["turn"] for r in cell_rows}):
                    turn_rows = [r for r in cell_rows if r["turn"] == turn]
                    by_turn[str(turn)] = {
                        "turn_positions": len(turn_rows),
                        "reference_f1": _mean(r[f1_field] for r in turn_rows),
                        "transfer_f1": _mean(r["transfer_f1"] for r in turn_rows),
                        "reference_minus_transfer_f1": _mean(
                            r[f1_field] - r["transfer_f1"] for r in turn_rows
                        ),
                        "reference_unhealthy": sum(
                            bool(r[health_field]) for r in turn_rows
                        ),
                        "transfer_unhealthy": sum(
                            bool(r["transfer_health"]) for r in turn_rows
                        ),
                        "health_excess": _mean(
                            bool(r["transfer_health"]) - bool(r[health_field])
                            for r in turn_rows
                        ),
                    }
                gap = _mean(
                    by_domain[d]["reference_minus_transfer_f1"] for d in by_domain
                )
                health_excess = _mean(by_domain[d]["health_excess"] for d in by_domain)

                grouped = _group_rows(cell_rows)
                boot_gap, boot_health = [], []
                for _ in range(replicates):
                    domain_gaps, domain_health = [], []
                    for domain in sorted(grouped):
                        ids = sorted(grouped[domain])
                        picked = rng.choice(ids, size=len(ids), replace=True)
                        sampled = [
                            r for cid in picked for r in grouped[domain][str(cid)]
                        ]
                        domain_gaps.append(
                            _mean(r[f1_field] - r["transfer_f1"] for r in sampled)
                        )
                        domain_health.append(
                            _mean(
                                bool(r["transfer_health"]) - bool(r[health_field])
                                for r in sampled
                            )
                        )
                    boot_gap.append(_mean(domain_gaps))
                    boot_health.append(_mean(domain_health))
                cell["comparisons"][label] = {
                    "equal_domain_reference_f1": _mean(
                        by_domain[d]["reference_f1"] for d in by_domain
                    ),
                    "equal_domain_transfer_f1": _mean(
                        by_domain[d]["transfer_f1"] for d in by_domain
                    ),
                    "reference_unhealthy": sum(
                        bool(r[health_field]) for r in cell_rows
                    ),
                    "transfer_unhealthy": sum(
                        bool(r["transfer_health"]) for r in cell_rows
                    ),
                    "equal_domain_reference_minus_transfer_f1": gap,
                    "equal_domain_health_excess": health_excess,
                    "by_domain_descriptive": by_domain,
                    "by_turn_descriptive": by_turn,
                    "bootstrap": {
                        "seed": BOOTSTRAP_SEED,
                        "replicates": replicates,
                        "unit": "complete conversations resampled within domain",
                        "reference_minus_transfer_f1_95_interval": [
                            _percentile(boot_gap, 2.5),
                            _percentile(boot_gap, 97.5),
                        ],
                        "health_excess_95_interval": [
                            _percentile(boot_health, 2.5),
                            _percentile(boot_health, 97.5),
                        ],
                        "interpretation": (
                            "descriptive only; a zero-width all-zero health interval "
                            "does not prove a population bound"
                        ),
                    },
                    "passed": gap <= 0.03 and health_excess <= 0.02,
                }
            cells[key] = cell
    return {
        "schema": "coqa_q2_result_v1",
        "cells": cells,
        "screen": {
            "promising": bool(cells)
            and all(
                comp["passed"]
                for cell in cells.values()
                for comp in cell["comparisons"].values()
            ),
            "meaning": "reused-development point-estimate screen; domain results descriptive",
            "bootstrap_replicates": replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    }


def retain_complete_q2_conversations(stories, answers, provenance, analysis, turn_cap):
    """Drop a partially persisted conversation before resuming recursive work."""
    complete = {
        story["id"]
        for story in stories
        if sum(row["conversation_id"] == story["id"] for row in answers)
        == 6 * min(turn_cap, len(story["questions"]))
        and sum(row["conversation_id"] == story["id"] for row in provenance)
        == 2 * min(turn_cap, len(story["questions"]))
        and sum(row["conversation_id"] == story["id"] for row in analysis)
        == 2 * min(turn_cap, len(story["questions"]))
    }
    return (
        complete,
        [row for row in answers if row["conversation_id"] in complete],
        [row for row in provenance if row["conversation_id"] in complete],
        [row for row in analysis if row["conversation_id"] in complete],
    )
