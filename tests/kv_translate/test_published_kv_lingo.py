from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from research.kv_translate.published.kv_lingo import (
    CacheGeometry,
    LinearTranslator,
    PreNormSpan,
    capture_pre_norm_span,
    forward_kl,
    geometry,
    span_bytes,
    translator_parameter_count,
)
from research.kv_translate.published.kv_lingo_train import (
    BidirectionalMoments,
    CAPTURE_CUT,
    DISTRIBUTED_ASSIGNMENT,
    DISTRIBUTED_CONTRACT_SCHEMA,
    DISTRIBUTED_EXECUTION_SCHEMA,
    EFFECTIVE_BATCH,
    LEARNING_RATE,
    LOSS_REDUCTION,
    MAP_ARCHITECTURE,
    PINS,
    STAGE2_STEPS,
    WARMUP_STEPS,
    distributed_batch_assignments,
    file_sha256,
    learning_rate,
    validate_new_run_contract,
    validate_resume_contract,
)
from research.kv_translate.published.kv_lingo_retained import (
    SpanOwnershipLedger,
    append_translated_span,
)
from research.kv_translate.published.capture import cache_layers, make_cache


class FakeNorm(nn.Module):
    def forward(self, value):
        return value


class FakeAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.k_norm = FakeNorm()
        with torch.no_grad():
            self.k_proj.weight.copy_(torch.eye(width))
            self.v_proj.weight.copy_(torch.eye(width))


class FakeLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_attn = FakeAttention(width)


class FakeRotary(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, _value, position_ids):
        shape = (*position_ids.shape, self.head_dim)
        return torch.ones(shape), torch.zeros(shape)


class FakeBody(nn.Module):
    def __init__(self, layers, width, head_dim):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(width) for _ in range(layers)])
        self.rotary_emb = FakeRotary(head_dim)
        self.width = width

    def forward(self, input_ids, use_cache=False):
        del use_cache
        value = torch.nn.functional.one_hot(input_ids % self.width, self.width).float()
        for layer in self.layers:
            layer.self_attn.k_proj(value)
            layer.self_attn.v_proj(value)
        return SimpleNamespace()


class FakeModel(nn.Module):
    def __init__(self, layers=2, heads=2, head_dim=3):
        super().__init__()
        width = heads * head_dim
        self.config = SimpleNamespace(
            num_hidden_layers=layers,
            num_key_value_heads=heads,
            num_attention_heads=heads,
            head_dim=head_dim,
            hidden_size=width,
        )
        self.model = FakeBody(layers, width, head_dim)

    def forward(self, input_ids, use_cache=False):
        return self.model(input_ids=input_ids, use_cache=use_cache)


def test_pre_norm_tap_captures_every_layer_and_side():
    model = FakeModel()
    span = capture_pre_norm_span(model, torch.tensor([[0, 1, 2, 3]]))
    assert len(span.keys) == len(span.values) == 2
    assert span.keys[0].shape == (1, 2, 4, 3)
    assert span.tokens == 4
    assert span_bytes(span) == 2 * 2 * 1 * 2 * 4 * 3 * 4


def test_identity_translator_preserves_a_shape_preserving_span():
    model = FakeModel()
    span = capture_pre_norm_span(model, torch.tensor([[0, 1, 2, 3]]))
    translator = LinearTranslator(geometry(model), geometry(model))
    keys, values = translator(span, model, torch.arange(4).unsqueeze(0))
    for actual, expected in zip(keys, span.keys, strict=True):
        assert torch.equal(actual, expected)
    for actual, expected in zip(values, span.values, strict=True):
        assert torch.equal(actual, expected)


def test_translator_parameter_ledger_is_two_dense_maps_per_layer():
    geom = CacheGeometry(layers=36, kv_heads=8, head_dim=128)
    assert translator_parameter_count(geom, geom) == 36 * 2 * 1024 * 1024


def test_forward_kl_is_zero_for_identical_logits_and_rejects_wrong_shape():
    logits = torch.randn(2, 3, 7)
    logprobs = torch.log_softmax(logits, dim=-1)
    assert forward_kl(logprobs, logits).item() == pytest.approx(0.0, abs=2e-7)
    with pytest.raises(ValueError, match="shapes differ"):
        forward_kl(logprobs[:, :2], logits)


def test_stage1_shared_moments_recover_both_exact_linear_directions():
    geom = CacheGeometry(layers=1, kv_heads=1, head_dim=2)
    source = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]]]])
    matrix = torch.tensor([[2.0, 0.5], [0.25, 1.5]])
    target_rows = source.transpose(1, 2).reshape(-1, 2) @ matrix.T
    target = target_rows.reshape(1, 3, 1, 2).transpose(1, 2)
    moments = BidirectionalMoments(geom, "cpu")
    moments.update(
        PreNormSpan(keys=[source], values=[source]),
        PreNormSpan(keys=[target], values=[target]),
    )
    weights, receipts = moments.solve(1e-12)
    assert torch.allclose(weights["4b-to-8b"]["key"][0], matrix, atol=1e-5)
    assert torch.allclose(
        weights["8b-to-4b"]["value"][0], torch.linalg.inv(matrix), atol=1e-5
    )
    assert all(row["rank"] == 2 for rows in receipts.values() for row in rows)


def test_stage2_schedule_is_one_fixed_5000_step_cosine_with_250_step_warmup():
    assert EFFECTIVE_BATCH == 8
    assert WARMUP_STEPS == 250
    assert STAGE2_STEPS == 5000
    assert learning_rate(0) == 0.0
    assert learning_rate(1) == pytest.approx(LEARNING_RATE / WARMUP_STEPS)
    assert learning_rate(WARMUP_STEPS) == pytest.approx(LEARNING_RATE)
    assert 0 < learning_rate(1000) < LEARNING_RATE
    assert learning_rate(STAGE2_STEPS) == pytest.approx(0.0, abs=1e-15)


def test_four_rank_assignment_pairs_long_and_short_within_fixed_batch():
    rows = [
        {
            "index": index,
            "prefix_tokens": index + 1,
            "continuation_tokens": 1,
        }
        for index in range(EFFECTIVE_BATCH)
    ]
    assignments = distributed_batch_assignments(rows, 4)
    assert [[offset for offset, _row in group] for group in assignments] == [
        [0, 7],
        [1, 6],
        [2, 5],
        [3, 4],
    ]
    assert sorted(offset for group in assignments for offset, _row in group) == list(
        range(EFFECTIVE_BATCH)
    )


def test_eight_rank_assignment_preserves_every_row_once():
    rows = [
        {
            "index": index,
            "prefix_tokens": 10 - index,
            "continuation_tokens": index,
        }
        for index in range(EFFECTIVE_BATCH)
    ]
    assignments = distributed_batch_assignments(rows, 8)
    assert all(len(group) == 1 for group in assignments)
    assert sorted(offset for group in assignments for offset, _row in group) == list(
        range(EFFECTIVE_BATCH)
    )


def resume_fixture(tmp_path):
    rows = tmp_path / "train.jsonl"
    tokens = tmp_path / "train.tokens.u32"
    validation_rows = tmp_path / "validation.jsonl"
    validation_tokens = tmp_path / "validation.tokens.u32"
    for path, payload in (
        (rows, b"train rows\n"),
        (tokens, b"train tokens"),
        (validation_rows, b"validation rows\n"),
        (validation_tokens, b"validation tokens"),
    ):
        path.write_bytes(payload)
    stage1 = tmp_path / "stage1.pt"
    torch.save(
        {
            "schema": "kv_lingo_translator_v1",
            "direction": "4b-to-8b",
            "source": {"name": "Qwen/Qwen3-4B", "revision": PINS["Qwen/Qwen3-4B"]},
            "target": {"name": "Qwen/Qwen3-8B", "revision": PINS["Qwen/Qwen3-8B"]},
            "cut": CAPTURE_CUT,
            "provenance": {
                "train_rows_sha256": file_sha256(rows),
                "token_store_sha256": file_sha256(tokens),
            },
        },
        stage1,
    )
    checkpoint = tmp_path / "step1000.pt"
    metrics = [
        {"step": index, "sample_cursor": index * EFFECTIVE_BATCH}
        for index in range(1, 1001)
    ]
    torch.save(
        {
            "schema": "kv_lingo_stage2_checkpoint_v1",
            "direction": "4b-to-8b",
            "global_step": 1000,
            "sample_cursor": 8000,
            "total_schedule_steps": STAGE2_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "effective_batch": EFFECTIVE_BATCH,
            "learning_rate": LEARNING_RATE,
            "stage1_artifact": {"sha256": file_sha256(stage1)},
            "optimizer": {},
            "rng": {},
            "metrics": metrics,
        },
        checkpoint,
    )
    contract = {
        "schema": "kv_lingo_resume_contract_v1",
        "direction": "4b-to-8b",
        "checkpoint_schema": "kv_lingo_stage2_checkpoint_v1",
        "checkpoint_sha256": file_sha256(checkpoint),
        "global_step": 1000,
        "sample_cursor": 8000,
        "stage1_sha256": file_sha256(stage1),
        "train_rows_sha256": file_sha256(rows),
        "train_tokens_sha256": file_sha256(tokens),
        "validation_rows_sha256": file_sha256(validation_rows),
        "validation_tokens_sha256": file_sha256(validation_tokens),
        "model_revisions": PINS,
        "tokenizer_revision": PINS["Qwen/Qwen3-4B"],
        "capture_cut": CAPTURE_CUT,
        "map_architecture": MAP_ARCHITECTURE,
        "loss_reduction": LOSS_REDUCTION,
        "effective_batch": EFFECTIVE_BATCH,
        "warmup_steps": WARMUP_STEPS,
        "total_schedule_steps": STAGE2_STEPS,
        "peak_learning_rate": LEARNING_RATE,
    }
    contract_path = tmp_path / "resume.json"
    contract_path.write_text(__import__("json").dumps(contract))
    args = SimpleNamespace(
        action="train",
        resume=checkpoint,
        resume_contract=contract_path,
        direction="4b-to-8b",
        stage1=stage1,
        rows=rows,
        tokens=tokens,
        validation_rows=validation_rows,
        validation_tokens=validation_tokens,
        max_steps=1250,
    )
    return args, contract


def test_explicit_missing_resume_is_fatal_before_checkpoint_loading(
    tmp_path, monkeypatch
):
    args, _ = resume_fixture(tmp_path)
    args.resume = tmp_path / "absent.pt"
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("loaded artifact"))
    with pytest.raises(FileNotFoundError, match="explicit resume checkpoint is absent"):
        validate_resume_contract(args)


def test_resume_refuses_wrong_direction_and_altered_stream(tmp_path):
    args, _ = resume_fixture(tmp_path)
    args.direction = "8b-to-4b"
    with pytest.raises(ValueError, match="direction mismatch"):
        validate_resume_contract(args)
    args.direction = "4b-to-8b"
    args.rows.write_bytes(b"changed")
    with pytest.raises(ValueError, match="training rows sha256 mismatch"):
        validate_resume_contract(args)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("effective_batch", 7, "effective batch mismatch"),
        ("warmup_steps", 251, "warmup steps mismatch"),
        ("total_schedule_steps", 4999, "total schedule mismatch"),
    ],
)
def test_resume_refuses_changed_batch_or_schedule(tmp_path, field, value, message):
    args, contract = resume_fixture(tmp_path)
    contract[field] = value
    args.resume_contract.write_text(__import__("json").dumps(contract))
    with pytest.raises(ValueError, match=message):
        validate_resume_contract(args)


def test_resume_refuses_wrong_cursor_even_with_matching_sidecar_hash(tmp_path):
    args, contract = resume_fixture(tmp_path)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    checkpoint["sample_cursor"] = 7999
    checkpoint["metrics"][-1]["sample_cursor"] = 7999
    torch.save(checkpoint, args.resume)
    contract["checkpoint_sha256"] = file_sha256(args.resume)
    contract["sample_cursor"] = 7999
    args.resume_contract.write_text(__import__("json").dumps(contract))
    with pytest.raises(ValueError, match="cursor semantics mismatch"):
        validate_resume_contract(args)


def test_valid_exact_resume_contract_passes(tmp_path):
    args, contract = resume_fixture(tmp_path)
    assert validate_resume_contract(args) == contract


def test_distributed_resume_rejects_a_world_size_change(tmp_path, monkeypatch):
    args, contract = resume_fixture(tmp_path)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    checkpoint["distributed"] = {
        "schema": DISTRIBUTED_EXECUTION_SCHEMA,
        "world_size": 4,
        "backend": "nccl",
        "assignment": DISTRIBUTED_ASSIGNMENT,
        "rank_rng": [{} for _ in range(4)],
    }
    torch.save(checkpoint, args.resume)
    args.source_commit = "distributed-test-commit"
    contract["schema"] = DISTRIBUTED_CONTRACT_SCHEMA
    contract["checkpoint_sha256"] = file_sha256(args.resume)
    contract["execution_topology"] = {
        "schema": DISTRIBUTED_EXECUTION_SCHEMA,
        "world_size": 4,
        "backend": "nccl",
        "assignment": DISTRIBUTED_ASSIGNMENT,
        "global_batch": EFFECTIVE_BATCH,
        "source_commit": args.source_commit,
        "implementation_sha256": file_sha256(
            Path(
                __import__(
                    "research.kv_translate.published.kv_lingo_train",
                    fromlist=["__file__"],
                ).__file__
            )
        ),
        "migration_from_world_size": 1,
    }
    args.resume_contract.write_text(__import__("json").dumps(contract))
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(ValueError, match="execution world size mismatch"):
        validate_resume_contract(args)


def test_fresh_stage2_contract_binds_stage1_and_both_splits(tmp_path):
    args, _ = resume_fixture(tmp_path)
    args.resume = None
    contract = validate_new_run_contract(args)
    assert contract["global_step"] == 0
    assert contract["sample_cursor"] == 0
    assert contract["stage1_sha256"] == file_sha256(args.stage1)
    assert contract["validation_tokens_sha256"] == file_sha256(args.validation_tokens)


def test_retained_ledger_translates_only_the_missing_suffix_once():
    ledger = SpanOwnershipLedger(("4B", "8B"), initial_tokens=100)
    first = ledger.write("4B", 7)
    assert ledger.missing("8B") == [first]
    ledger.translated("8B", [first])
    second = ledger.write("8B", 5)
    assert ledger.missing("4B") == [second]
    ledger.translated("4B", [second])
    assert ledger.line_covered == {"4B": 112, "8B": 112}
    with pytest.raises(RuntimeError, match="frontier"):
        ledger.translated("4B", [first])


def test_a_translated_span_is_appended_without_mutating_the_old_cache():
    old_key = torch.zeros(1, 2, 3, 4)
    old_value = torch.ones(1, 2, 3, 4)
    cache = make_cache([(old_key, old_value)])
    new_key = torch.full((1, 2, 2, 4), 2.0)
    new_value = torch.full((1, 2, 2, 4), 3.0)
    appended = append_translated_span(cache, [new_key], [new_value])
    old_pairs = cache_layers(cache)
    new_pairs = cache_layers(appended)
    assert old_pairs[0][0].shape[2] == 3
    assert new_pairs[0][0].shape[2] == 5
    assert torch.equal(new_pairs[0][0][:, :, 3:], new_key)
    assert torch.equal(new_pairs[0][1][:, :, 3:], new_value)
