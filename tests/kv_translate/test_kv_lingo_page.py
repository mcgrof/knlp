from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_static_page_keeps_the_complete_grid_self_contained():
    html = (ROOT / "docs/kv-lingo.html").read_text(encoding="utf-8")
    assert html.count('class="plot-row"') == 32
    assert '<table class="crosswalk">' in html
    assert "<script" not in html
    assert "data/kv-lingo-results.json" not in html
    assert "All 16 comparisons" in html
    assert (
        "No forward/reverse checkpoint pair passes both starts and both references."
        in html
    )


def test_page_links_source_and_primary_paper():
    html = (ROOT / "docs/kv-lingo.html").read_text(encoding="utf-8")
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
