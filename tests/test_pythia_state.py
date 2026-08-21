"""Offline tests for the Pythia ZeRO-1 optimizer-state converter.

No network and no real shards: a synthetic checkpoint is built in
tmp_path with the exact dict layout observed in the real
EleutherAI/neox-ckpt-pythia-160m-deduped-v1 optimizer shards (two
Adam param groups — 2-D weights and 1-D params — one sub-partition
per rank per group, flat fp32 buffers, one flat param_shapes
OrderedDict keyed "<pipe>.<name>").
"""

import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fim.fisher_pruning.pythia_state import (  # noqa: E402
    build_layer_offset_map,
    infer_n_layers,
    map_state_to_hf,
    neox_to_hf_name,
    reassemble_zero1,
    stale_adam_scores,
    validate_against_hf,
)

WORLD = 4

# Odd sizes so group 0 (2-D) needs trailing padding while group 1
# (1-D) divides evenly: group 0 has 21 + 15 + 6 = 42 elements ->
# sub-partitions of 11 with 2 padding; group 1 has 5 + 3 = 8 -> 2
# per rank, no padding.
PARAM_SHAPES = OrderedDict(
    [
        ("0.word_embeddings.weight", torch.Size([7, 3])),
        ("2.attention.query_key_value.weight", torch.Size([5, 3])),
        ("2.attention.query_key_value.bias", torch.Size([5])),
        ("2.input_layernorm.weight", torch.Size([3])),
        ("3.mlp.dense_4h_to_h.weight", torch.Size([2, 3])),
    ]
)
GROUP_NAMES = [
    [n for n, s in PARAM_SHAPES.items() if len(s) >= 2],
    [n for n, s in PARAM_SHAPES.items() if len(s) == 1],
]
PAD_SENTINEL = -999.0


def make_reference_state(seed=0):
    g = torch.Generator().manual_seed(seed)
    ref = {"exp_avg": {}, "exp_avg_sq": {}, "master_weights": {}}
    for name, shape in PARAM_SHAPES.items():
        ref["exp_avg"][name] = torch.randn(shape, generator=g)
        ref["exp_avg_sq"][name] = torch.rand(shape, generator=g)
        ref["master_weights"][name] = torch.randn(shape, generator=g)
    return ref


def flatten_group(tensors_by_name, names):
    flat = torch.cat([tensors_by_name[n].reshape(-1) for n in names])
    total = flat.numel()
    sub_len = -(-total // WORLD)  # ceil
    pad = sub_len * WORLD - total
    if pad:
        flat = torch.cat([flat, torch.full((pad,), PAD_SENTINEL)])
    return flat, sub_len


def write_fixture(tmp_path, ref):
    """Write WORLD shards in the observed real-checkpoint layout."""
    flats = {
        kind: [flatten_group(ref[kind], names) for names in GROUP_NAMES]
        for kind in ("exp_avg", "exp_avg_sq", "master_weights")
    }
    for rank in range(WORLD):
        base_state = []
        fp32_groups = []
        for gi in range(len(GROUP_NAMES)):
            sub = {}
            for kind in ("exp_avg", "exp_avg_sq"):
                flat, sub_len = flats[kind][gi]
                sub[kind] = flat[rank * sub_len : (rank + 1) * sub_len].clone()
            base_state.append([sub])
            flat, sub_len = flats["master_weights"][gi]
            fp32_groups.append([flat[rank * sub_len : (rank + 1) * sub_len].clone()])
        shard = {
            "optimizer_state_dict": {
                "loss_scaler": None,
                "dynamic_loss_scale": True,
                "overflow": False,
                "base_optimizer_state": base_state,
                "zero_stage": 1,
                "partition_count": WORLD,
                "num_comm_intervals_per_group": [1] * len(GROUP_NAMES),
                "local_sub_partitions_of_fp32_groups": fp32_groups,
            },
            "param_shapes": OrderedDict(PARAM_SHAPES),
        }
        torch.save(shard, tmp_path / f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")


def test_reassemble_zero1_roundtrip(tmp_path):
    ref = make_reference_state()
    write_fixture(tmp_path, ref)
    state = reassemble_zero1(tmp_path)
    assert list(state["param_shapes"]) == list(PARAM_SHAPES)
    for kind in ("exp_avg", "exp_avg_sq", "master_weights"):
        assert set(state[kind]) == set(PARAM_SHAPES)
        for name in PARAM_SHAPES:
            got = state[kind][name]
            assert got.shape == PARAM_SHAPES[name]
            assert torch.equal(got, ref[kind][name]), (kind, name)
            # The padded tail must never leak into a parameter.
            assert not (got == PAD_SENTINEL).any()


def test_reassemble_rejects_missing_rank(tmp_path):
    ref = make_reference_state()
    write_fixture(tmp_path, ref)
    (tmp_path / "zero_pp_rank_2_mp_rank_00_optim_states.pt").unlink()
    with pytest.raises(FileNotFoundError, match=r"ranks \[2\]"):
        reassemble_zero1(tmp_path)


def test_reassemble_rejects_short_buffer(tmp_path):
    """A flat buffer shorter than the group's parameters must raise."""
    ref = make_reference_state()
    write_fixture(tmp_path, ref)
    # Truncate every rank's group-1 sub-partitions by one element
    # (uniformly, so the equal-length check passes first); the
    # concatenated length then undershoots the parameter sum.
    for rank in range(WORLD):
        p = tmp_path / f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        shard = torch.load(p, weights_only=False)
        osd = shard["optimizer_state_dict"]
        osd["local_sub_partitions_of_fp32_groups"][1][0] = osd[
            "local_sub_partitions_of_fp32_groups"
        ][1][0][:-1]
        for sub in osd["base_optimizer_state"][1]:
            sub["exp_avg"] = sub["exp_avg"][:-1]
            sub["exp_avg_sq"] = sub["exp_avg_sq"][:-1]
        torch.save(shard, p)
    with pytest.raises(ValueError, match="flat buffer"):
        reassemble_zero1(tmp_path)


def test_neox_to_hf_name_mapping():
    offsets = build_layer_offset_map(12)
    cases = {
        "0.word_embeddings.weight": "gpt_neox.embed_in.weight",
        "2.attention.query_key_value.weight": (
            "gpt_neox.layers.0.attention.query_key_value.weight"
        ),
        "2.attention.dense.bias": "gpt_neox.layers.0.attention.dense.bias",
        "13.mlp.dense_4h_to_h.weight": ("gpt_neox.layers.11.mlp.dense_4h_to_h.weight"),
        "13.post_attention_layernorm.bias": (
            "gpt_neox.layers.11.post_attention_layernorm.bias"
        ),
        "15.norm.weight": "gpt_neox.final_layer_norm.weight",
        "15.norm.bias": "gpt_neox.final_layer_norm.bias",
        "16.final_linear.weight": "embed_out.weight",
    }
    for neox, hf in cases.items():
        assert neox_to_hf_name(neox, offsets) == hf
    # Buffers with no HF parameter are dropped, but the dense bias
    # (tested above) survives the "attention.bias" fragment.
    for buf in (
        "2.attention.rotary_emb.inv_freq",
        "2.attention.bias",
        "2.attention.masked_bias",
    ):
        assert neox_to_hf_name(buf, offsets) is None
    with pytest.raises(KeyError):
        neox_to_hf_name("1.anything.weight", offsets)


def test_map_state_to_hf_drops_buffers():
    state = {
        "0.word_embeddings.weight": torch.zeros(2, 2),
        "2.attention.rotary_emb.inv_freq": torch.zeros(2),
        "2.attention.dense.weight": torch.ones(2, 2),
    }
    mapped = map_state_to_hf(state, n_layers=12)
    assert set(mapped) == {
        "gpt_neox.embed_in.weight",
        "gpt_neox.layers.0.attention.dense.weight",
    }


def test_infer_n_layers():
    assert infer_n_layers(PARAM_SHAPES) == -1  # toy shapes stop at pipe 3
    real_like = OrderedDict(
        [
            ("0.word_embeddings.weight", torch.Size([4, 4])),
            ("16.final_linear.weight", torch.Size([4, 4])),
        ]
    )
    assert infer_n_layers(real_like) == 12


def test_validate_against_hf_roundtrip():
    torch.manual_seed(1)
    master = {"gpt_neox.embed_in.weight": torch.randn(6, 4)}
    hf_good = {
        "gpt_neox.embed_in.weight": master["gpt_neox.embed_in.weight"].half().float()
    }
    report = validate_against_hf(master, hf_good)
    assert report["pass"]
    assert report["per_tensor"]["gpt_neox.embed_in.weight"]["ok"]

    hf_bad = {"gpt_neox.embed_in.weight": hf_good["gpt_neox.embed_in.weight"] + 1e-3}
    report = validate_against_hf(master, hf_bad)
    assert not report["pass"]
    assert not report["per_tensor"]["gpt_neox.embed_in.weight"]["ok"]
    assert report["per_tensor"]["gpt_neox.embed_in.weight"]["max_abs_diff"] > 0

    report = validate_against_hf(master, {})
    assert not report["pass"]
    assert report["missing_in_hf"] == ["gpt_neox.embed_in.weight"]


def test_stale_adam_scores_shapes():
    """Score arm wiring against a NeoX-shaped module tree."""
    import torch.nn as nn

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_key_value = nn.Linear(4, 12)
            self.dense = nn.Linear(4, 4)

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense_h_to_4h = nn.Linear(4, 16)
            self.dense_4h_to_h = nn.Linear(16, 4)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()
            self.mlp = Mlp()

    class Neox(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.gpt_neox = Neox()

    model = Model()
    v = {
        f"gpt_neox.layers.0.{leaf}.weight": torch.rand_like(mod.weight)
        for leaf, mod in [
            (
                "attention.query_key_value",
                model.gpt_neox.layers[0].attention.query_key_value,
            ),
            ("attention.dense", model.gpt_neox.layers[0].attention.dense),
            ("mlp.dense_h_to_4h", model.gpt_neox.layers[0].mlp.dense_h_to_4h),
            ("mlp.dense_4h_to_h", model.gpt_neox.layers[0].mlp.dense_4h_to_h),
        ]
    }
    scores = stale_adam_scores(v, model, q=0.25)
    assert set(scores) == {name.rsplit(".weight", 1)[0] for name in v}
    for name, s in scores.items():
        mod_w = dict(model.named_parameters())[name + ".weight"]
        assert s.shape == mod_w.shape
        assert (s >= 0).all()
    with pytest.raises(KeyError, match="no exp_avg_sq"):
        stale_adam_scores({}, model, q=0.25)
