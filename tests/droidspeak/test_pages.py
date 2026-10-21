from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_page(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_droidspeak_page_states_result_and_system_boundary() -> None:
    html = read_page("droidspeak.html")

    required = (
        "4 OF 32 LAYERS RECOMPUTED",
        "50 PROFILE + 50 HELD-OUT RECORDS",
        "Selected DroidSpeak exceeded the receiver-relative floor",
        "0.3149",
        "0.2887",
        "0.2742",
        "Do not read 4.03× as a DroidSpeak system speedup.",
        "No DroidSpeak training checkpoint exists",
        "7,168 cached tokens",
    )
    for text in required:
        assert text in html

    assert "[14,18)" in html
    assert "kv-cache-sharing.html" in html
    assert "<script" not in html
    assert ".json" not in html


def test_comparison_page_keeps_methods_and_results_separate() -> None:
    html = read_page("kv-cache-sharing.html")

    required = (
        "NOT A HEAD-TO-HEAD BENCHMARK",
        "What we learned",
        "Cross-model cache reuse can work",
        "LMCache can move a cache between servers",
        "KV-Lingo used CoQA",
        "DroidSpeak used HotpotQA",
        "0 / 4",
        "4 / 32 layers",
        "+2.68 F1",
        "Modified MoT: what the four-arm attribution isolated",
        "FULL-KEY CONTROL",
        "Could DroidSpeak improve KV-Lingo?",
        "not as a drop-in patch",
        "LMCache is transport, not representation repair",
        "Reproduce the public modified-MoT path",
    )
    for text in required:
        assert text in html

    assert 'href="kv-lingo.html"' in html
    assert 'href="droidspeak.html"' in html
    assert 'href="https://github.com/mcgrof/MoT/' in html
    assert "<script" not in html
    assert 'href="data/' not in html
    assert 'src="data/' not in html


def test_page_navigation_is_bidirectional() -> None:
    droidspeak = read_page("droidspeak.html")
    comparison = read_page("kv-cache-sharing.html")
    kv_lingo = read_page("kv-lingo.html")
    index = read_page("index.html")

    assert 'href="kv-cache-sharing.html"' in droidspeak
    assert 'href="kv-cache-sharing.html"' in kv_lingo
    assert 'href="droidspeak.html"' in kv_lingo
    assert 'href="droidspeak.html"' in comparison
    assert 'href="kv-lingo.html"' in comparison
    assert 'href="kv-cache-sharing.html"' in index
    assert 'href="droidspeak.html"' in index


def test_pages_do_not_expose_private_research_names() -> None:
    pages = (
        read_page("droidspeak.html"),
        read_page("kv-cache-sharing.html"),
        read_page("kv-lingo.html"),
    )
    banned = ("knlp-key-results", "results-archive", "/home/", "/data/")

    for html in pages:
        for text in banned:
            assert text not in html
