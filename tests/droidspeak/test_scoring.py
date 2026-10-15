import pytest

from research.droidspeak.scoring import best_qa_f1, normalize_answer, qa_f1


def test_normalize_answer_matches_english_qa_convention() -> None:
    assert normalize_answer("The, QUICK fox!") == "quick fox"


def test_qa_f1_counts_duplicate_tokens() -> None:
    assert qa_f1("red red blue", "red blue blue") == pytest.approx(2 / 3)


def test_best_qa_f1_uses_best_reference() -> None:
    assert best_qa_f1("Seattle", ["Tacoma", "Seattle, Washington"]) == pytest.approx(
        2 / 3
    )
