"""Partial-RoPE models split each head into a rotated subspace (first rotary_dim channels) and a
pass-through tail with a different distribution. The masks must partition the head exactly and the
RoPE-pair index must pair channel i with i+rotary_dim/2 (HF rotate_half), leaving pass-through
channels self-paired. Getting this wrong mis-attributes Phi-style failures."""

import torch

from tools.kv.fp8_failure import architecture_discovery as AD


def test_full_rope_has_no_passthrough():
    rot, pas = AD.subspace_masks(rotary_dim=8, head_dim=8)
    assert bool(rot.all()) and int(pas.sum()) == 0


def test_partial_rope_exact_partition():
    rot, pas = AD.subspace_masks(rotary_dim=4, head_dim=8)
    assert int(rot.sum()) == 4 and int(pas.sum()) == 4
    assert bool((rot | pas).all()) and not bool((rot & pas).any())
    assert bool(rot[:4].all()) and bool(pas[4:].all())


def test_rope_pair_index():
    pair = AD.rope_pair_index(rotary_dim=4, head_dim=8)
    # rotary block: i <-> i+2 ; pass-through: self
    assert torch.equal(pair, torch.tensor([2, 3, 0, 1, 4, 5, 6, 7]))
    # pairing is an involution on the rotary block
    assert torch.equal(pair[pair], torch.arange(8))


def test_pair_index_full_rope():
    pair = AD.rope_pair_index(rotary_dim=8, head_dim=8)
    assert torch.equal(pair, torch.tensor([4, 5, 6, 7, 0, 1, 2, 3]))
