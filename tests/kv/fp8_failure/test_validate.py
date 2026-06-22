"""The quant-fp8 validate orchestrator must parse a .config the same way the Kconfig pipeline writes
it, map the CONFIG_KNLP_QUANT_FP8_* keys onto typed fields, resolve the model set, and gate on the
enable flag. These are the pure (stdlib-only, no torch, no subprocess) pieces -- the runner self-tests
themselves are exercised by the runners' own --self-test paths, not here."""

from tools.kv.fp8_failure import validate as V


def test_parse_value_coercions():
    assert V._parse_value("y") is True
    assert V._parse_value("n") is False
    assert V._parse_value('"hello world"') == "hello world"
    assert V._parse_value("1024") == 1024
    assert V._parse_value("cuda:0") == "cuda:0"  # bare string falls through


def test_parse_config_keeps_prefix_and_skips_comments(tmp_path):
    cfg = tmp_path / ".config"
    cfg.write_text(
        "# a comment\n"
        "CONFIG_KNLP_QUANT_FP8=y\n"
        'CONFIG_KNLP_QUANT_FP8_PROFILE="validate"\n'
        "CONFIG_KNLP_QUANT_FP8_SEQ_LEN=1024\n"
        "\n"
        "# CONFIG_KNLP_QUANT_FP8_RUN_LONGCTX is not set\n"
    )
    d = V.parse_config(cfg)
    assert d["CONFIG_KNLP_QUANT_FP8"] is True
    assert d["CONFIG_KNLP_QUANT_FP8_PROFILE"] == "validate"
    assert d["CONFIG_KNLP_QUANT_FP8_SEQ_LEN"] == 1024
    # `# ... is not set` lines are comments to this parser (kconfig2py reads them as False, but the
    # orchestrator only ever reads keys it knows, so absence -> dataclass default; that's fine).
    assert "CONFIG_KNLP_QUANT_FP8_RUN_LONGCTX" not in d


def test_from_file_maps_fields_and_enable(tmp_path):
    cfg = tmp_path / ".config"
    cfg.write_text(
        "CONFIG_KNLP_QUANT_FP8=y\n"
        'CONFIG_KNLP_QUANT_FP8_PROFILE="atlas"\n'
        'CONFIG_KNLP_QUANT_FP8_DEVICE="cuda:1"\n'
        "CONFIG_KNLP_QUANT_FP8_SEQ_LEN=512\n"
        'CONFIG_KNLP_QUANT_FP8_SEEDS="0 1 2 3"\n'
        "CONFIG_KNLP_QUANT_FP8_RUN_RECOVERY=n\n"
    )
    c = V.Fp8Config.from_file(cfg)
    assert c.is_enabled() is True
    assert c.profile == "atlas"
    assert c.device == "cuda:1"
    assert c.seq_len == 512 and isinstance(c.seq_len, int)
    assert c.seeds == "0 1 2 3"
    assert c.run_recovery is False
    assert c.run_multiseed is True  # default holds when key absent


def test_disabled_when_flag_missing(tmp_path):
    cfg = tmp_path / ".config"
    cfg.write_text('CONFIG_KNLP_QUANT_FP8_PROFILE="validate"\n')
    c = V.Fp8Config.from_file(cfg)
    assert c.is_enabled() is False
    # defaults still resolve so doctor can describe the would-be run
    assert c.profile == "validate"
    assert c.py().endswith("/python") or "python" in c.py()


def test_model_list_resolution():
    base = dict(raw={"CONFIG_KNLP_QUANT_FP8": True})
    assert V.Fp8Config(models="core", **base).model_list() == V.CORE_MODELS
    assert V.Fp8Config(models="all", **base).model_list() == list(V.SHORT2HF)
    assert V.Fp8Config(models="phi-2 Qwen2.5-7B", **base).model_list() == [
        "phi-2",
        "Qwen2.5-7B",
    ]


def test_core_models_map_to_known_hf_ids():
    # every core short-name must resolve to a real HF id in the registry mirror
    for short in V.CORE_MODELS:
        assert short in V.SHORT2HF and "/" in V.SHORT2HF[short]


def test_selftest_table_covers_every_runner():
    names = {n for n, _ in V.SELFTESTS}
    # the ten atlas runners that ship a --self-test
    for r in [
        "run_smoke",
        "run_failure_classes",
        "run_mechanism",
        "run_v_residual",
        "run_controls",
        "run_multiseed",
        "run_longctx",
        "run_preflight",
        "run_gptj",
        "recovery_pareto",
    ]:
        assert r in names
    # recovery_pareto is the only torch-free one (no --output-dir in its self-test argv)
    rp = dict(V.SELFTESTS)["recovery_pareto"]
    assert rp == ["--self-test"]
