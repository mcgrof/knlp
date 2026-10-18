import json
from pathlib import Path

from research.kv_translate.mot_author.recipe import (
    FROZEN_DIAGNOSTIC_CONTENT_SHA256,
    MODEL_REVISIONS,
    UPSTREAM_SHA,
    build_channel_profile_config,
    build_eval_config,
    build_reproduction_matrix,
    build_train_config,
)
from research.kv_translate.mot_author.runner import _read_kconfig
from research.kv_translate.mot_author.smoke import (
    CHANNELS,
    GRAD_ACCUM_STEPS,
    MICROBATCH_SIZE,
    SMOKE_STEPS,
)
from research.kv_translate.mot_author.data import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    SMOKE_ITEM_COUNT,
    SMOKE_SEED,
    SMOKE_SHUFFLE_BUFFER,
    SMOKE_TOTAL_TOKENS,
    load_smoke_manifest,
    tokenizer_receipt,
)
from research.kv_translate.mot_author.preflight import (
    PREFLIGHT_THRESHOLDS,
    inspect_model_snapshot,
)
from research.kv_translate.mot_author.mechanics import inspect_mechanics_source
from research.kv_translate.mot_author.diagnostic_eval import replay_paired_jsonl
from research.kv_translate.mot_author.recipe import (
    MODIFIED_MOT_COMMITS,
    MODIFIED_MOT_SHA,
)


def test_frozen_gpt2_recipe_uses_paper_training_precision(tmp_path: Path) -> None:
    train = build_train_config(output_path=tmp_path / "run")
    channel = build_channel_profile_config()
    evaluation = build_eval_config(
        output_path=tmp_path / "evaluation",
        checkpoint_dir_path=tmp_path / "run",
    )

    assert len(UPSTREAM_SHA) == 40
    assert train["model_ids"] == "gpt2,gpt2-medium"
    assert train["dtype"] == "bfloat16"
    assert train["max_steps"] == 500
    assert train["batch_size"] * train["grad_accum_steps"] == 16
    assert train["mot_num_translators"] == 2
    assert train["mot_top_k"] == 1
    assert train["num_bottom_full_attn"] == 2
    assert train["attention_backend"] == "dense_selected"
    assert channel["num_train_examples"] == 128
    assert channel["num_val_examples"] == 64
    assert channel["max_steps"] == 128
    assert evaluation["max_examples_per_dataset"] == 100
    assert evaluation["generation_max_new_tokens"] == 64
    assert set(MODEL_REVISIONS) == {"gpt2", "gpt2-medium"}
    assert len(FROZEN_DIAGNOSTIC_CONTENT_SHA256) == 64


def test_reproduction_matrix_keeps_paper_claim_bounded() -> None:
    matrix = build_reproduction_matrix()

    assert "author-code mechanics baseline" in matrix
    assert "exact reproduction target" in matrix
    assert "bfloat16 override" in matrix


def test_kconfig_reader_preserves_strings_and_booleans(tmp_path: Path) -> None:
    config = tmp_path / ".config"
    config.write_text(
        "CONFIG_MOT_AUTHOR_BASELINE=y\n"
        'CONFIG_MOT_AUTHOR_SOURCE="/srv/MoT checkout"\n'
        'CONFIG_MOT_AUTHOR_OUTPUT_DIR="out"\n',
        encoding="utf-8",
    )

    values = _read_kconfig(config)

    assert values["MOT_AUTHOR_BASELINE"] is True
    assert values["MOT_AUTHOR_SOURCE"] == "/srv/MoT checkout"
    assert values["MOT_AUTHOR_OUTPUT_DIR"] == "out"


def test_real_model_preflight_thresholds_are_explicit() -> None:
    assert PREFLIGHT_THRESHOLDS == {
        "native_continuation_max_abs": 1.0e-4,
        "author_replay_logits_max_abs": 1.0e-4,
        "author_replay_logits_min_cosine": 0.99999,
        "author_replay_cache_max_abs": 2.0e-5,
    }


def test_model_snapshot_requires_exact_revision(tmp_path: Path) -> None:
    revision = "a" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    receipt = inspect_model_snapshot(
        snapshot,
        model_name="fixture",
        expected_revision=revision,
    )

    assert receipt["revision_matches"] is True
    assert receipt["snapshot_revision"] == revision
    assert receipt["weight_files"] == [{"name": "model.safetensors", "size_bytes": 7}]


def test_modified_mot_commit_chain_is_explicit() -> None:
    assert len(MODIFIED_MOT_COMMITS) == 5
    assert MODIFIED_MOT_SHA == MODIFIED_MOT_COMMITS[-1]
    assert all(len(commit) == 40 for commit in MODIFIED_MOT_COMMITS)


def test_mechanics_source_rejects_unknown_variant(tmp_path: Path) -> None:
    try:
        inspect_mechanics_source(tmp_path, variant="unknown")
    except RuntimeError as error:
        assert "Not a git checkout" in str(error)
    else:
        raise AssertionError("non-checkout source was accepted")


def test_diagnostic_replay_reports_id_and_score_contracts(tmp_path: Path) -> None:
    rows = [
        {
            "item_id": f"item-{index}",
            "task": "boolq",
            "document_group": f"document-{index}",
            "truncation_status": False,
            "candidate": {"score": float(index), "parse_status": "parsed"},
            "native": {"score": 1.0, "parse_status": "parsed"},
        }
        for index in range(2)
    ]
    source = tmp_path / "records.jsonl"
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = replay_paired_jsonl(source, tmp_path / "replay.json")

    assert result["record_count"] == 2
    assert result["unique_item_id_count"] == 2
    assert result["item_ids_unique"] is True
    assert result["both_arm_score_contract_pass"] is True


def test_training_smoke_recipe_is_bounded_and_frozen() -> None:
    assert SMOKE_STEPS == 2
    assert MICROBATCH_SIZE == 4
    assert GRAD_ACCUM_STEPS == 4
    assert SMOKE_ITEM_COUNT == 32
    assert SMOKE_TOTAL_TOKENS == 128
    assert DATASET_ID == "Skylion007/openwebtext"
    assert len(DATASET_REVISION) == 40
    assert CHANNELS == (
        {"src_layer_idx": 8, "dst_layer_idx": 20},
        {"src_layer_idx": 9, "dst_layer_idx": 21},
        {"src_layer_idx": 10, "dst_layer_idx": 22},
        {"src_layer_idx": 11, "dst_layer_idx": 23},
    )


def test_training_manifest_refuses_a_changed_token_row(tmp_path: Path) -> None:
    snapshot = tmp_path / MODEL_REVISIONS["gpt2-medium"]
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    row = list(range(SMOKE_TOTAL_TOKENS))
    import hashlib
    import json

    row_hash = hashlib.sha256(
        json.dumps(row, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema": "knlp_mot_training_smoke_manifest_v2",
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "streaming": True,
            "shuffle": True,
            "shuffle_seed": SMOKE_SEED,
            "shuffle_buffer": SMOKE_SHUFFLE_BUFFER,
        },
        "tokenizer": tokenizer_receipt(snapshot),
        "item_count": SMOKE_ITEM_COUNT,
        "total_tokens_per_item": SMOKE_TOTAL_TOKENS,
        "items": [
            {"index": index, "token_ids": row, "token_ids_sha256": row_hash}
            for index in range(SMOKE_ITEM_COUNT)
        ],
    }
    manifest["items"][0]["token_ids"][0] = -1
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        load_smoke_manifest(manifest_path=path, target_model=snapshot)
    except RuntimeError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("changed token manifest was accepted")
