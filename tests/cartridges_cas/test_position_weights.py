"""Per-position loss weights: key stability, mode arithmetic and the
uniform-mode bit identity with the unpatched mean."""

import importlib.util
import json
import pathlib
import types

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "research" / "cartridges_cas" / "scripts" / "cas_position_weights.py"
PATCH = ROOT / "research" / "cartridges_cas" / "patches" / "position_weights.py"


def _load():
    spec = importlib.util.spec_from_file_location("cas_position_weights", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _batch(seed=0, n_pos=64, k=20):
    g = torch.Generator().manual_seed(seed)
    pos = torch.arange(1, n_pos + 1).repeat_interleave(k)
    return types.SimpleNamespace(
        input_ids=torch.randint(0, 1000, (n_pos,), generator=g),
        topk_token_idxs=pos,
        topk_token_ids=torch.randint(0, 1000, (n_pos * k,), generator=g),
        topk_logprobs=torch.log_softmax(torch.randn(n_pos * k, generator=g), 0),
    )


def test_key_is_content_only():
    m = _load()
    a, b = _batch(0), _batch(0)
    assert m.batch_key(a) == m.batch_key(b)
    c = _batch(0)
    c.topk_token_ids = c.topk_token_ids.clone()
    c.topk_token_ids[3] += 1
    assert m.batch_key(c) != m.batch_key(a)
    # the stored logprobs are not part of the key: a mask depends only on
    # which tokens sit where, which is what the hash covers
    d = _batch(0)
    d.topk_logprobs = d.topk_logprobs + 1.0
    assert m.batch_key(d) == m.batch_key(a)


def test_mode_weights_normalize_to_unit_mean():
    m = _load()
    n_info, n_total = 180, 1000
    for mode, floor in (("informative", 0.1), ("rest", 0.0), ("uniform", 0.0)):
        wi, wr = m.mode_weights(mode, floor, n_info, n_total)
        mean = (wi * n_info + wr * (n_total - n_info)) / n_total
        assert mean == pytest.approx(1.0)
    wi, wr = m.mode_weights("informative", 0.1, n_info, n_total)
    assert wi == pytest.approx(10 * wr)
    wi, wr = m.mode_weights("rest", 0.0, n_info, n_total)
    assert wi == 0.0 and wr == pytest.approx(n_total / (n_total - n_info))
    with pytest.raises(ValueError):
        m.mode_weights("nope", 0.1, 1, 2)


def test_uniform_entry_loss_is_bit_identical_and_informative_reweights(tmp_path):
    m = _load()
    b = _batch(1)
    mask = np.zeros(b.topk_token_ids.numel(), dtype=bool)
    mask[: b.topk_token_ids.numel() // 5] = True
    meta = {"n_entries": int(mask.size), "n_info_entries": int(mask.sum())}
    out = tmp_path / "w.npz"
    np.savez_compressed(out, meta=json.dumps(meta), **{f"m_{m.batch_key(b)}": mask})

    ce = torch.rand(mask.size, dtype=torch.bfloat16) * 0.1
    uni = m.load_weights(out, "uniform")
    assert torch.equal(uni.entry_loss(b, ce), ce.mean())

    info = m.load_weights(out, "informative", floor=0.1)
    w = torch.where(
        torch.from_numpy(mask),
        torch.tensor(info.w_info, dtype=ce.dtype),
        torch.tensor(info.w_rest, dtype=ce.dtype),
    )
    assert torch.equal(info.entry_loss(b, ce), (w * ce).mean())
    assert info.w_info > info.w_rest > 0

    rest = m.load_weights(out, "rest")
    assert rest.w_info == 0.0
    assert rest.entry_loss(b, ce).float() == pytest.approx(
        (ce.float()[~torch.from_numpy(mask)] * rest.w_rest).sum() / mask.size, rel=1e-2
    )

    other = _batch(2)
    with pytest.raises(KeyError):
        uni.entry_loss(other, ce)


def test_teacher_argmax_picks_highest_stored_logprob():
    m = _load()
    b = _batch(3, n_pos=8, k=4)
    arg = m.teacher_argmax(b, 16)
    for p in range(8):
        sl = slice(p * 4, (p + 1) * 4)
        best = b.topk_logprobs[sl].argmax()
        assert arg[p] == b.topk_token_ids[sl][best]
    assert (arg[8:] == -1).all()


def test_patch_rewrites_only_the_loss_line(tmp_path):
    src = (
        "import x\n\nclass CacheAndModel(nn.Module):\n    pass\n\n"
        "def train():\n"
        "                    loss = (ce_by_token.mean() / accumulate_grad_steps)\n"
    )
    root = tmp_path / "cartridges" / "cartridges"
    root.mkdir(parents=True)
    (root / "train.py").write_text(src)
    spec = importlib.util.spec_from_file_location("pw_patch", PATCH)
    import sys

    argv = sys.argv
    sys.argv = ["position_weights.py", "--cart-root", str(tmp_path / "cartridges")]
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.main() == 0
        assert mod.main() == 0  # idempotent
    finally:
        sys.argv = argv
    out = (root / "train.py").read_text()
    assert out.count("loss = (_cr_entry_loss(batch, ce_by_token)") == 1
    assert out.count("def _cr_entry_loss") == 1
    assert "ce_by_token.mean() / accumulate" not in out
