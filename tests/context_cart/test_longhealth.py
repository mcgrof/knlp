"""Panel summary records the decoder cap the rows were produced with."""

from research.context_cart import longhealth as LH


def _row(run, correct, stop):
    return {
        "run": run,
        "correct": correct,
        "tag_found": True,
        "stop_reason": stop,
        "question_id": f"q{run}",
    }


def test_summarize_records_actual_cap():
    rows = [_row(0, 1, "eos"), _row(1, 0, "cap")]
    default = LH.summarize(rows)
    assert default["decoder"]["max_completion"] == LH.DECODER["max_completion"]
    long = LH.summarize(rows, 8192)
    assert long["decoder"]["max_completion"] == 8192
    assert long["per_run_counts"][1]["cap_hits"] == 1
    assert long["acc_mean"] == 0.5
    assert LH.DECODER["max_completion"] == 2048
