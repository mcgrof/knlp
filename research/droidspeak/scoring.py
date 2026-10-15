"""LongBench-compatible extractive QA scoring."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable


def normalize_answer(text: str) -> str:
    """Apply the standard English QA normalization used by LongBench."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        return "".join(char for char in value if char not in string.punctuation)

    return " ".join(remove_articles(remove_punctuation(text.lower())).split())


def qa_f1(prediction: str, answer: str) -> float:
    """Return token F1 in the [0, 1] range."""

    prediction_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    if not prediction_tokens or not answer_tokens:
        return float(prediction_tokens == answer_tokens)
    overlap = Counter(prediction_tokens) & Counter(answer_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def best_qa_f1(prediction: str, answers: Iterable[str]) -> float:
    """Return the best F1 over the accepted reference answers."""

    answer_list = list(answers)
    if not answer_list:
        raise ValueError("answers must not be empty")
    return max(qa_f1(prediction, answer) for answer in answer_list)
