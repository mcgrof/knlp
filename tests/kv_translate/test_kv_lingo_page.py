import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def signed(value: float) -> str:
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.2f}"


def interval_value(value: float) -> str:
    return f"−{abs(value):.2f}" if value < 0 else f"{value:.2f}"


def test_static_page_contains_every_aggregate_comparison():
    html = (ROOT / "docs/kv-lingo.html").read_text(encoding="utf-8")
    data = json.loads(
        (ROOT / "docs/data/kv-lingo-results.json").read_text(encoding="utf-8")
    )
    assert html.count('class="plot-row"') == 32
    assert '<table class="crosswalk">' in html
    assert "<script" not in html

    for pair in data["pairs"]:
        for comparison in pair["comparisons"]:
            deficit = comparison["equal_domain_deficit_f1_points"]
            low, high = comparison["descriptive_bootstrap"][
                "deficit_interval_95_f1_points"
            ]
            overall = (
                f"{signed(deficit)} " f"[{interval_value(low)}, {interval_value(high)}]"
            )
            assert overall in html

            decision = "PASS" if comparison["passed"] else "FAIL"
            worst = (
                f'{comparison["worst_domain_deficit_f1_points"]:.2f} · '
                f'{comparison["worst_domain"]} · {decision}'
            )
            assert worst in html


def test_page_links_source_data_and_primary_paper():
    html = (ROOT / "docs/kv-lingo.html").read_text(encoding="utf-8")
    assert "data/kv-lingo-results.json" in html
    assert "research/kv_translate/published" in html
    assert "https://arxiv.org/abs/2609.32610" in html


def test_page_states_material_paper_differences_and_fairness_boundary():
    html = (ROOT / "docs/kv-lingo.html").read_text(encoding="utf-8")
    required = (
        "What “different Nemotron construction” means",
        "3,487,460 prefix positions",
        "32 exposed CoQA dialogues",
        "256 CoQA dialogues",
        "1 seed vs 3",
        "FAIR FOR THE KNLP GO/NO-GO",
        "NOT AN EXACT PAPER REPLICATION",
        "The negative result is not only a threshold artifact.",
        "The turn-2 F1 is the score of 8B's generated answer 2.",
    )
    for text in required:
        assert text in html
