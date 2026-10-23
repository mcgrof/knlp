import pytest

from scripts.check_tracked_artifacts import is_artifact


@pytest.mark.parametrize(
    "path",
    [
        "docs/data/result.json",
        "docs/assets/plot-data.json",
        "docs/data/matrix.csv",
        "tools/kv/native/fixtures/current/fp16/qwen.json.gz",
        "tools/kv/native/fixtures/phi4/manifest.json",
        "results/run/metrics.json",
        "rgsa_v19_results/phase2_correlation.json",
        "kvsplice_trained/model.pt",
    ],
)
def test_captured_artifacts_are_rejected(path):
    assert is_artifact(path)


@pytest.mark.parametrize(
    "path",
    [
        "configs/model.json",
        "research/chores/web/package.json",
        "research/code_reason/schemas/result.schema.json",
        "tools/kv/native/fixtures/README.md",
        "docs/assets/quality.svg",
        "tools/kv/native/runner.py",
    ],
)
def test_source_contracts_and_presentation_assets_remain_valid(path):
    assert not is_artifact(path)
