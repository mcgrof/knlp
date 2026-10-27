# SPDX-License-Identifier: MIT
"""CPU gates for the prefix path on a tiny deterministic Qwen3 model."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from research.context_cart import artifact as A
from research.context_cart import model as M
from tools.reproduce.context_artifacts.artifacts import ArtifactError

FAKE_REVISION = "a" * 40
FAKE_SOURCE_REVISION = "b" * 40
FAKE_CHECKPOINT = "c" * 64


def _identity(model: torch.nn.Module) -> dict:
    geometry = M.Geometry.from_config(model.config)
    return {
        "model": "tiny/qwen3",
        "revision": FAKE_REVISION,
        "dtype": "float32",
        "config_digest": M.config_digest(model.config),
        "weights_digest": M.weights_digest(model),
        "tokenizer_digest": "d" * 64,
        "rope_digest": M.rope_digest(model.config, geometry.head_dim),
        "geometry": geometry.__dict__,
    }


def _random_prefix(model: torch.nn.Module, rows: int, seed: int = 1):
    g = M.Geometry.from_config(model.config)
    gen = torch.Generator().manual_seed(seed)
    prefix = []
    for _ in range(g.layers):
        k = torch.randn(
            1, g.kv_heads, rows, g.head_dim, generator=gen, requires_grad=True
        )
        v = torch.randn(
            1, g.kv_heads, rows, g.head_dim, generator=gen, requires_grad=True
        )
        prefix.append((k, v))
    return prefix


def test_reference_matches_cache_path_values_and_gradients():
    model = M.tiny_base(seed=0)
    reader = M.Reader(model, eos_ids=(1,), pad_id=0)
    rows, length, first = 4, 6, 10 + 4
    prefix = _random_prefix(model, rows)
    ids = torch.randint(2, model.config.vocab_size, (1, length))
    mask = torch.ones(1, length, dtype=torch.long)
    probe = torch.randn(1, length, model.config.hidden_size)

    hidden_cache, cache = reader.forward(prefix, ids, mask, first)
    assert cache.get_seq_length() == rows + length
    (hidden_cache * probe).sum().backward()
    grads_cache = [(k.grad.clone(), v.grad.clone()) for k, v in prefix]
    for k, v in prefix:
        k.grad = None
        v.grad = None

    positions = torch.arange(first, first + length).unsqueeze(0)
    hidden_ref = M.reference_hidden(model, prefix, ids, positions)
    (hidden_ref * probe).sum().backward()
    grads_ref = [(k.grad.clone(), v.grad.clone()) for k, v in prefix]

    torch.testing.assert_close(hidden_cache, hidden_ref, atol=1e-5, rtol=1e-4)
    worst = 0.0
    for (gk, gv), (rk, rv) in zip(grads_cache, grads_ref):
        torch.testing.assert_close(gk, rk, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(gv, rv, atol=1e-5, rtol=1e-4)
        worst = max(worst, (gk - rk).abs().max().item(), (gv - rv).abs().max().item())
        assert gk.abs().sum() > 0 and gv.abs().sum() > 0
    assert worst < 1e-5


def test_positions_and_prefix_change_the_output():
    model = M.tiny_base(seed=0)
    reader = M.Reader(model, eos_ids=(1,), pad_id=0)
    prefix = [(k.detach(), v.detach()) for k, v in _random_prefix(model, 4)]
    ids = torch.randint(2, model.config.vocab_size, (1, 5))
    mask = torch.ones(1, 5, dtype=torch.long)
    base, _ = reader.forward(prefix, ids, mask, 14)
    shifted, _ = reader.forward(prefix, ids, mask, 200)
    other = [(k * 0.5, v) for k, v in prefix]
    changed, _ = reader.forward(other, ids, mask, 14)
    assert (base - shifted).abs().max() > 1e-4
    assert (base - changed).abs().max() > 1e-4


def test_gradients_reach_memory_and_adapters_only():
    pytest.importorskip("peft")
    torch.manual_seed(0)
    writer_base = M.tiny_base(seed=0)
    reader_model = M.tiny_base(seed=0)
    spec = M.ModelSpec(
        name="tiny",
        revision=FAKE_REVISION,
        dtype="float32",
        memory_slots=4,
        lora_rank=2,
        lora_alpha=4,
    )
    writer = M.Writer(writer_base, spec, init_rows=[5, 6, 7, 8])
    reader = M.Reader(reader_model, eos_ids=(1,), pad_id=0)
    reader_before = copy.deepcopy(reader_model.state_dict())
    source = torch.randint(2, 512, (2, 12))
    prefix, length = writer.encode(source)
    assert length == 12 and prefix[0][0].shape == (2, 2, 4, 8)
    prompts = [[9, 10, 11], [9, 10, 11, 12]]
    answers = [[20, 21, 1], [22, 1]]
    ids, mask, labels = M.right_pad_with_labels(
        prompts, answers, pad_id=0, device="cpu"
    )
    loss, count, per_example = reader.answer_loss(
        prefix, ids, mask, labels, first_position=length + 4
    )
    assert count == 5 and per_example.shape == (2,)
    assert torch.isclose(per_example.sum(), loss)
    loss.backward()
    assert writer.memory.grad is not None
    assert (
        torch.isfinite(writer.memory.grad).all() and writer.memory.grad.abs().sum() > 0
    )
    lora_grads = [
        p.grad
        for n, p in writer.peft.named_parameters()
        if "lora_" in n and p.grad is not None
    ]
    assert lora_grads and any(g.abs().sum() > 0 for g in lora_grads)
    frozen = [p for n, p in writer.peft.named_parameters() if "lora_" not in n]
    assert all(p.grad is None for p in frozen)
    assert all(p.grad is None for p in reader_model.parameters())
    trainable = writer.trainable_parameters()
    assert all(p.dtype == torch.float32 for p in trainable)
    # one optimizer step changes the writer and leaves the reader untouched
    opt = torch.optim.AdamW(trainable, lr=1e-2)
    before = writer.memory.detach().clone()
    opt.step()
    assert (writer.memory.detach() - before).abs().sum() > 0
    for key, value in reader_model.state_dict().items():
        assert torch.equal(value, reader_before[key])
    # same source twice gives identical rows; a changed source changes them
    again, _ = writer.encode(source)
    assert torch.equal(again[0][0], writer.encode(source)[0][0][0])
    changed = source.clone()
    changed[0, 3] = (changed[0, 3] + 1) % 512
    # layer-0 rows depend only on the memory embedding; later layers see the source
    assert torch.equal(writer.encode(changed)[0][0][0][0], again[0][0][0])
    assert not torch.equal(writer.encode(changed)[0][1][0][0], again[1][0][0])
    assert torch.equal(writer.encode(changed)[0][1][0][1], again[1][0][1])


def test_greedy_matches_hf_generate_without_prefix():
    model = M.tiny_base(seed=3)
    reader = M.Reader(model, eos_ids=(1,), pad_id=0)
    prompts = [[7, 8, 9, 10], [11, 12], [13, 14, 15, 16, 17]]
    for prompt in prompts:
        ids = torch.tensor([prompt])
        ref = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=6,
            eos_token_id=1,
            pad_token_id=0,
        )[0, len(prompt) :].tolist()
        ours = reader.greedy(None, ids, torch.ones_like(ids), 0, 6)["ids"][0].tolist()
        assert M.strip_generation(ours, (1,), 0) == M.strip_generation(ref, (1,), 0)
    # batched left padding reproduces the batch-one answers
    ids, mask = M.left_pad(prompts, pad_id=0, device="cpu")
    batched = reader.greedy(None, ids, mask, 0, 6)["ids"]
    for row, prompt in enumerate(prompts):
        single = reader.greedy(
            None,
            torch.tensor([prompt]),
            torch.ones(1, len(prompt), dtype=torch.long),
            0,
            6,
        )
        assert M.strip_generation(batched[row].tolist(), (1,), 0) == M.strip_generation(
            single["ids"][0].tolist(), (1,), 0
        )


def test_greedy_with_prefix_batched_matches_single():
    model = M.tiny_base(seed=4)
    reader = M.Reader(model, eos_ids=(1,), pad_id=0)
    singles = [
        [(k.detach(), v.detach()) for k, v in _random_prefix(model, 4, seed=s)]
        for s in (1, 2, 3)
    ]
    prompts = [[7, 8, 9, 10], [11, 12], [13, 14, 15]]
    ids, mask = M.left_pad(prompts, pad_id=0, device="cpu")
    stacked = M.stack_prefixes(singles)
    batched = reader.greedy(stacked, ids, mask, 20, 5)
    for row, prompt in enumerate(prompts):
        single = reader.greedy(
            singles[row],
            torch.tensor([prompt]),
            torch.ones(1, len(prompt), dtype=torch.long),
            20,
            5,
        )
        assert M.strip_generation(
            batched["ids"][row].tolist(), (1,), 0
        ) == M.strip_generation(single["ids"][0].tolist(), (1,), 0)
        torch.testing.assert_close(
            batched["first_logits"][row],
            single["first_logits"][0],
            atol=1e-4,
            rtol=1e-4,
        )


def test_export_reload_bit_exact_and_parity(tmp_path: Path):
    model = M.tiny_base(seed=5)
    reader = M.Reader(model, eos_ids=(1,), pad_id=0)
    identity = _identity(model)
    prefix = [(k.detach(), v.detach()) for k, v in _random_prefix(model, 4)]
    result = A.export_prefix(
        tmp_path / "cart",
        prefix,
        identity,
        first_position=16,
        source_revision=FAKE_SOURCE_REVISION,
        checkpoint_digest=FAKE_CHECKPOINT,
        namespace="test",
        extra={"world_id": "w"},
    )
    assert result.tensor_bytes == 2 * 2 * 4 * 2 * 8 * 4
    loaded = A.load_prefix(tmp_path / "cart", identity, "cpu")
    assert loaded.rows == 4 and loaded.first_position == 16
    for (k, v), (lk, lv) in zip(prefix, loaded.prefix):
        assert torch.equal(k, lk) and torch.equal(v, lv)
    assert M.prefix_digest(prefix) == loaded.tensor_digest
    ids = torch.tensor([[7, 8, 9]])
    mask = torch.ones_like(ids)
    live = reader.greedy(prefix, ids, mask, 16, 5, keep_logits=True)
    restored = reader.greedy(loaded.prefix, ids, mask, 16, 5, keep_logits=True)
    assert live["ids"].tolist() == restored["ids"].tolist()
    assert torch.equal(live["step_logits"], restored["step_logits"])
    # the prefix rows are untouched by cache extension
    assert M.prefix_digest(loaded.prefix) == loaded.tensor_digest
    manifest = json.loads((tmp_path / "cart" / "manifest.json").read_text())
    assert manifest["metadata"]["positions"] == list(range(16, 20))
    assert manifest["representation"] == "prefix_kv"


def test_loader_rejects_identity_geometry_positions_and_corruption(tmp_path: Path):
    model = M.tiny_base(seed=6)
    identity = _identity(model)
    prefix = [(k.detach(), v.detach()) for k, v in _random_prefix(model, 4)]
    A.export_prefix(
        tmp_path / "cart",
        prefix,
        identity,
        first_position=16,
        source_revision=FAKE_SOURCE_REVISION,
        checkpoint_digest=FAKE_CHECKPOINT,
        namespace="test",
        extra={},
    )
    wrong_revision = dict(identity, revision="f" * 40)
    with pytest.raises(ArtifactError):
        A.load_prefix(tmp_path / "cart", wrong_revision, "cpu")
    wrong_geometry = dict(identity, geometry=dict(identity["geometry"], kv_heads=3))
    with pytest.raises(ArtifactError):
        A.load_prefix(tmp_path / "cart", wrong_geometry, "cpu")
    wrong_weights = dict(identity, weights_digest="e" * 64)
    with pytest.raises(ArtifactError):
        A.load_prefix(tmp_path / "cart", wrong_weights, "cpu")
    wrong_dtype = dict(identity, dtype="bfloat16")
    with pytest.raises(ArtifactError):
        A.load_prefix(tmp_path / "cart", wrong_dtype, "cpu")
    # malformed positions in the manifest
    manifest_path = tmp_path / "cart" / "manifest.json"
    original = manifest_path.read_bytes()
    broken = json.loads(original)
    broken["metadata"]["positions"] = [16, 17, 19, 20]
    manifest_path.write_text(json.dumps(broken))
    with pytest.raises(ArtifactError):
        A.load_prefix(tmp_path / "cart", identity, "cpu")
    manifest_path.write_bytes(original)
    A.load_prefix(tmp_path / "cart", identity, "cpu")
    # corrupted payload byte
    payload = tmp_path / "cart" / "payload.safetensors"
    data = bytearray(payload.read_bytes())
    data[-1] ^= 0xFF
    payload.write_bytes(bytes(data))
    with pytest.raises(ArtifactError):
        A.load_prefix(tmp_path / "cart", identity, "cpu")


def test_thd_roundtrip_and_stack():
    model = M.tiny_base(seed=7)
    prefix = [(k.detach(), v.detach()) for k, v in _random_prefix(model, 4)]
    thd = M.to_thd(prefix)
    assert thd["key.0"].shape == (4, 2, 8)
    back = M.from_thd(thd, 2, "cpu")
    for (k, v), (bk, bv) in zip(prefix, back):
        assert torch.equal(k, bk) and torch.equal(v, bv)
    stacked = M.stack_prefixes([prefix, back])
    assert stacked[0][0].shape == (2, 2, 4, 8)


def test_cosine_schedule_endpoints():
    assert M.cosine_lr(0, 1000, 50, 1e-4, 1e-5) == pytest.approx(2e-6)
    assert M.cosine_lr(49, 1000, 50, 1e-4, 1e-5) == pytest.approx(1e-4)
    assert M.cosine_lr(1000, 1000, 50, 1e-4, 1e-5) == pytest.approx(1e-5)


def test_masked_context_prefill_matches_joint_prompt():
    """Text arms prefill contexts of different lengths in one left-padded
    batch, then fork per question. The result must equal the batch-one joint
    prompt (context + question) answered without any prefix."""
    model = M.tiny_base(seed=8)
    reader = M.Reader(model, eos_ids=(1,), pad_id=0)
    heads = [[21, 22, 23, 24, 25, 26], [31, 32], [41, 42, 43, 44]]
    tails = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]
    head_ids, head_mask = M.left_pad(heads, 0, "cpu")
    _, cache = reader.forward(None, head_ids, head_mask, 0)
    context = [M.cache_layer(cache, layer) for layer in range(reader.geometry.layers)]
    tail_ids, tail_mask = M.left_pad(tails, 0, "cpu")
    out = reader.greedy(
        context, tail_ids, tail_mask, head_mask.sum(-1), 5, prefix_mask=head_mask
    )
    assert torch.isfinite(out["first_logits"]).all()
    for row, (head, tail) in enumerate(zip(heads, tails)):
        joint = torch.tensor([head + tail])
        single = reader.greedy(None, joint, torch.ones_like(joint), 0, 5)
        assert M.strip_generation(
            out["ids"][row].tolist(), (1,), 0
        ) == M.strip_generation(single["ids"][0].tolist(), (1,), 0)
        torch.testing.assert_close(
            out["first_logits"][row], single["first_logits"][0], atol=1e-4, rtol=1e-4
        )
