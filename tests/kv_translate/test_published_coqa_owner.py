from research.kv_translate.published import coqa, coqa_owner
from research.kv_translate.published import coqa_owner_runner as runner


class DummyCache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self):
        return self.length


class DummyPair:
    def __init__(self):
        self.consumed = []

    def consume(self, model, cache, token_ids):
        self.consumed.append((model, list(token_ids)))
        return DummyCache(cache.length + len(token_ids)), 0.25


def test_owner_ingests_each_withheld_and_generated_token_once():
    pair = DummyPair()
    ledger = [10, 11, 12, 13, 14, 15]
    cache, _seconds = runner._extend_to_late_cut(
        pair, "14B", DummyCache(3), ledger, None
    )
    assert cache.length == 5
    assert pair.consumed == [("14B", [13, 14])]

    cache, returned, _seconds = runner._ingest_complete_ledger(
        pair, "14B", cache, ledger, [20, 21]
    )
    assert returned == [15, 20, 21]
    assert cache.length == len(ledger) == 8

    cache, _seconds = runner._extend_to_late_cut(
        pair, "14B", cache, ledger, [30, 31, 32]
    )
    assert cache.length == len(ledger) - 1 == 10
    assert pair.consumed[-1] == ("14B", [30, 31])


def quality_rows(*, late_transfer=0.79, domain_transfer=None):
    rows = []
    for domain in coqa.DOMAINS:
        for conversation in range(2):
            for turn in range(1, 11):
                translated = 0.79 if turn <= 5 else late_transfer
                if domain_transfer is not None and domain == "wikipedia":
                    translated = domain_transfer
                rows.append(
                    {
                        "row_id": f"{domain}-{conversation}:{turn}",
                        "conversation_id": f"{domain}-{conversation}",
                        "domain": domain,
                        "turn": turn,
                        "translated_f1": translated,
                        "translated_health": [],
                        "same_transcript_native_f1": 0.80,
                        "same_transcript_native_health": [],
                        "native_32b_only_f1": 0.80,
                        "native_32b_only_health": [],
                    }
                )
    return rows


def test_s1_requires_both_native_references_and_late_turn_guard():
    passed = coqa_owner.summarize(quality_rows(), "s1", replicates=20)
    assert passed["next_state"] == "S2"
    assert all(screen["passed"] for screen in passed["screens"].values())

    failed = coqa_owner.summarize(quality_rows(late_transfer=0.72), "s1", replicates=20)
    assert failed["next_state"] == "CLOSE_CONFIGURATION"
    assert all(
        not screen["requirements"]["turns_6_to_10_equal_domain_deficit_at_most_0.05"]
        for screen in failed["screens"].values()
    )


def test_s2_adds_each_domain_gate():
    result = coqa_owner.summarize(
        quality_rows(domain_transfer=0.70), "s2", replicates=20
    )
    assert result["next_state"] == "DEVELOPMENT_ONLY"
    assert all(
        not screen["requirements"]["each_domain_deficit_at_most_0.05"]
        for screen in result["screens"].values()
    )
