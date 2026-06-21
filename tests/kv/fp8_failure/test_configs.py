"""The four atlas configs must parse, carry their required keys, and agree with each other (the
serving model set and dataset cross-references must point at things that exist). A broken config is
a wasted GPU run, so validate them GPU-free up front."""

from tools.kv.fp8_failure import configs as CFG


def test_all_configs_load_with_required_keys():
    cfgs = CFG.load_all()
    assert set(cfgs) == {"models", "thresholds", "datasets", "serving"}


def test_model_registry_flattens_and_annotates_tier():
    models = CFG.iter_models(CFG.load("models"))
    assert len(models) >= 10
    assert all("id" in m and "where" in m and "tier" in m for m in models)
    # at least one pod-only and several local
    assert any(m["where"] == "pod" for m in models)
    assert sum(m["where"] == "local" for m in models) >= 5


def test_threshold_bands_are_ordered():
    t = CFG.load("thresholds")["classification"]
    assert t["tolerant"]["nll_increase_pct_max"] < t["catastrophic"]["nll_increase_pct_min"]
    r = CFG.load("thresholds")["recovery"]
    assert 0.0 < r["negligible_max"] < r["substantial_min"] <= 1.0


def test_serving_models_are_subset_of_registry():
    reg_ids = {m["id"] for m in CFG.iter_models(CFG.load("models"))}
    serving_ids = {m["id"] for m in CFG.load("serving")["models"]}
    assert serving_ids <= reg_ids  # no serving model missing from the registry


def test_datasets_smoke_tier_is_tiny():
    d = CFG.load("datasets")
    assert d["smoke"]["holdout"]["n"] <= 8
    assert len(d["seeds"]) >= 3  # paper aggregates need >=3 seeds
