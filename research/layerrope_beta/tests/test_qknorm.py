# SPDX-License-Identifier: MIT
"""Independent attention, cache-order, pairing and resume checks."""

import math
import os
import numpy as np
import pytest
import torch
from research.layerrope_beta.model import Model
from research.layerrope_beta.qknorm.protocol import freeze
from research.layerrope_beta.time_budget.train import train
from research.mo_variants.stage_c_common import sha, write
from research.tinystories_depth.attention import attend, normalize
from research.tinystories_depth.kv_cache import CachePolicy, fakequant, install, parity
from research.tinystories_depth.model import author, mhc
from research.tinystories_depth.train import state_hash

DEVICE = os.environ.get("KNLP_QKNORM_DEVICE", "cpu")
torch.set_num_threads(2)


def test_fp64_explicit_reference_and_gain_gradients():
    torch.manual_seed(77890)
    q, k, v = [
        torch.randn(2, 2, 5, 8, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    ]
    g = torch.tensor(2.3, dtype=torch.float64, requires_grad=True)
    actual = attend(normalize(q), normalize(k), v, g)
    scores = (
        (q / q.norm(dim=-1, keepdim=True))
        @ (k / k.norm(dim=-1, keepdim=True)).transpose(-2, -1)
        * g
    )
    scores = scores.masked_fill(
        torch.ones(5, 5, dtype=torch.bool).triu(1), -float("inf")
    )
    expected = scores.softmax(-1) @ v
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    a = torch.autograd.grad(actual.square().sum(), (q, k, v, g), retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), (q, k, v, g))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-9)
    assert a[-1].abs() > 1e-6
    assert torch.autograd.gradcheck(
        lambda q, k, g: attend(normalize(q), normalize(k), v.detach(), g), (q, k, g)
    )


def test_zero_near_zero_and_fp32_accumulation():
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.tensor([[0.0, 0.0], [1e-20, -1e-20]], dtype=dtype, requires_grad=True)
        y = normalize(x)
        assert y.dtype == dtype and torch.isfinite(y).all()
        y.float().sum().backward()
        assert torch.isfinite(x.grad).all()
    x = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
    assert torch.equal(normalize(x), (x.float() / x.float().norm()).to(x.dtype))


@pytest.mark.parametrize("arm", ("R", "L", "M", "ML"))
def test_pairing_schema_rng_and_noquant(arm):
    torch.manual_seed(77890)
    off = Model(arm, 2, 77890, beta=0.0, vocab=97).to(DEVICE).eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(77890)
    on = Model(arm, 2, 77890, beta=0.0, vocab=97, qknorm=True).to(DEVICE).eval()
    assert torch.equal(torch.get_rng_state(), rng)
    assert not any("qknorm" in n for n in off.state_dict())
    for n, p in off.state_dict().items():
        assert torch.equal(p, on.state_dict()[n]), n
    for b in on.lm.model.layers:
        assert b.self_attn.qknorm_gain.ndim == 0
        assert b.self_attn.qknorm_gain.item() == pytest.approx(
            math.log2(256 * 256 - 256)
        )
    ids = torch.randint(0, 97, (2, 9), device=DEVICE)
    for model in (off, on):
        parity(model, ids, DEVICE)
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        assert not torch.equal(off(ids), on(ids))


def cached_model(model, ids, chunks):
    """Small independent model driver with one tuple per layer, including mHC."""
    past = [None] * model.depth
    pieces, position = [], 0
    with install(model, CachePolicy(3, 2)), torch.autocast(
        device_type=DEVICE, dtype=torch.bfloat16
    ):
        for length in chunks:
            h = model.lm.model.embed_tokens(
                ids[:, position : position + length]
            ).bfloat16()
            if model.routes is not None:
                h = mhc.expand_to_mhc_ref(h, 4)
            for layer, block in enumerate(model.lm.model.layers):
                for site, name in enumerate(("attn", "mlp")):
                    if model.routes is None:
                        inp = h
                    else:
                        pre, post, carry, _ = model.routes[2 * layer + site](h)
                        with torch.autocast(device_type=DEVICE, enabled=False):
                            inp = mhc.mhc_pre_apply_mix_ref(h, pre)
                    if site == 0:
                        inp = block.input_layernorm(inp, block._gain("in_attn"))
                        branch, _, new = block.self_attn(
                            inp,
                            position_ids=torch.arange(
                                position, position + length, device=ids.device
                            )[None],
                            past_key_value=past[layer],
                            use_cache=True,
                        )
                        if past[layer] is not None:
                            for before, after in zip(past[layer], new):
                                assert torch.equal(before, after[:, :, :position])
                        past[layer] = new
                    else:
                        inp = block.post_attention_layernorm(inp, block._gain("in_mlp"))
                        branch = block.mlp(inp)
                    branch = block._block_output(branch, name)
                    if model.routes is None:
                        h = h + branch
                    else:
                        with torch.autocast(device_type=DEVICE, enabled=False):
                            h = mhc.mhc_post_ref(branch, h, post, carry)
            if model.routes is not None:
                h = h.float().mean(-2).bfloat16()
            pieces.append(model.lm.lm_head(model.lm.model.norm(h)))
            position += length
    assert position == ids.shape[1]
    return torch.cat(pieces, dim=1)


@pytest.mark.parametrize("arm", ("R", "L", "M", "ML"))
def test_incremental_and_independent_quantized_oracle(arm):
    model = Model(arm, 2, 77890, beta=0.0, vocab=97, qknorm=True).to(DEVICE).eval()
    attn = model.lm.model.layers[0].self_attn
    hidden = torch.randn(2, 9, 128, device=DEVICE).bfloat16()
    positions = torch.arange(9, device=DEVICE)[None]
    for kb, vb, method in (
        (16, 16, "symmetric"),
        (3, 16, "symmetric"),
        (16, 2, "symmetric"),
        (3, 2, "symmetric"),
        (2, 2, "affine"),
    ):
        with torch.no_grad(), install(
            model, CachePolicy(kb, vb, method)
        ), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            full = attn(hidden, position_ids=positions, use_cache=True)
            for chunks in ((2, 1, 1, 1, 1, 1, 1, 1), (5, 2, 2)):
                past, out, start = None, [], 0
                for length in chunks:
                    y, _, new = attn(
                        hidden[:, start : start + length],
                        position_ids=positions[:, start : start + length],
                        past_key_value=past,
                        use_cache=True,
                    )
                    if past is not None:
                        for before, after in zip(past, new):
                            assert torch.equal(before, after[:, :, :start])
                    past = new
                    out.append(y)
                    start += length
                assert (
                    full[0].float() - torch.cat(out, 1).float()
                ).abs().max() <= freeze()["tolerance"]["attention_bf16_max_abs"]
            q = attn._shape(attn.q_proj(hidden), 9, 2)
            k = attn._shape(attn.k_proj(hidden), 9, 2)
            v = attn._shape(attn.v_proj(hidden), 9, 2)
            q = (q.float() / q.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(
                q.dtype
            )
            k = (k.float() / k.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(
                k.dtype
            )
            cos, sin = attn.rotary_emb(v, seq_len=9)
            q, k = author.apply_rotary_pos_emb(q, k, cos, sin, positions)
            k, v = fakequant(k, kb, method), fakequant(v, vb, method)
            assert torch.equal(k, full[2][0])
            assert torch.equal(v, full[2][1])
            assert torch.equal(fakequant(k[:, :, :3], 16), k[:, :, :3])
            with torch.autocast(device_type=DEVICE, enabled=False):
                scores = (
                    q * attn.qknorm_gain.to(q.dtype)
                ).float() @ k.float().transpose(-2, -1)
                probability = scores.masked_fill(
                    torch.ones(9, 9, device=DEVICE, dtype=torch.bool).triu(1),
                    -float("inf"),
                ).softmax(-1)
                oracle = (probability @ v.float()).to(v.dtype)
            oracle = attn.o_proj(oracle.transpose(1, 2).reshape(2, 9, 128))
            assert (full[0].float() - oracle.float()).abs().max() <= 0.004
    ids = torch.randint(0, 97, (2, 9), device=DEVICE)
    with torch.no_grad():
        full = cached_model(model, ids, (9,))
        for chunks in ((2, 1, 1, 1, 1, 1, 1, 1), (5, 2, 2)):
            other = cached_model(model, ids, chunks)
            assert (full.float() - other.float()).abs().max() <= 0.125


def test_qknorm_resume_update_checkpoint_and_optimizer_group(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for n in ("train", "validation"):
        np.save(
            data / (n + ".npy"),
            np.random.default_rng(77890).integers(0, 97, 513, dtype=np.uint16),
        )
    write(
        data / "TRAINING.json",
        dict(files={n: sha(data / n) for n in ("train.npy", "validation.npy")}),
    )
    job = dict(
        arm="ML",
        seed=77890,
        qknorm=True,
        stage="qualification",
        lr=0.001,
        depth=2,
        vocab=97,
        window=16,
        batch_size=2,
    )
    protocol = freeze()
    protocol["equal_token_updates"] = 2
    kwargs = dict(device=DEVICE, smoke=True, contract=protocol)
    whole = train(data, job, tmp_path / "whole", stop_after_updates=4, **kwargs)
    partial = train(data, job, tmp_path / "partial", stop_after_updates=2, **kwargs)
    c = partial["last_checkpoint"]
    resumed = train(
        data,
        job,
        tmp_path / "resumed",
        resume=c["path"],
        resume_sha=c["sha256"],
        stop_after_updates=4,
        **kwargs
    )
    a = torch.load(
        whole["last_checkpoint"]["path"], map_location="cpu", weights_only=False
    )
    b = torch.load(
        resumed["last_checkpoint"]["path"], map_location="cpu", weights_only=False
    )
    for n in ("model", "optimizer", "rng", "cursor"):
        assert state_hash(a[n]) == state_hash(b[n])
    assert b["training_seconds"] >= partial["training_seconds"]
    assert b["identity"]["job"]["qknorm"]
    assert len(resumed["history"]["update_landmarks"]) == 1
    assert resumed["history"]["update_landmarks"][0]["step"] == 2
    group = b["optimizer"]["param_groups"][-1]
    assert group["name"] == "qknorm_gain" and group["weight_decay"] == 0
