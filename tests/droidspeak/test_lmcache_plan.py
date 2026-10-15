from research.droidspeak.lmcache_plan import (
    capability_report,
    legacy_centralized_config,
)


def test_lmcache_claim_boundary_is_fail_closed() -> None:
    report = capability_report()

    assert report["transport_baseline"]["supported"] is True
    assert report["missing_runtime_bridge"]
    assert "does not prove" in report["claim_boundary"]


def test_legacy_config_matches_linked_example() -> None:
    config = legacy_centralized_config()

    assert "chunk_size: 256" in config
    assert 'remote_url: "lm://localhost:65432"' in config
