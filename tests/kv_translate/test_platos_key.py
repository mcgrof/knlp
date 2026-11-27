import torch

from research.kv_translate.platos_key import (
    Fact,
    fit_rotation,
    neighbor_overlap,
    swap_pairs,
)


def test_scaled_procrustes_recovers_rectangular_map():
    generator = torch.Generator().manual_seed(5)
    source = torch.randn(40, 3, generator=generator)
    axes, _ = torch.linalg.qr(torch.randn(5, 3, generator=generator))
    translation = torch.arange(5)
    target = 2.5 * source @ axes.T + translation
    fit = fit_rotation(source[:30], target[:30])
    torch.testing.assert_close(fit(source[30:]), target[30:], atol=1e-5, rtol=1e-5)
    assert neighbor_overlap(source, source @ axes.T, k=5) == 1.0


def test_pairs_respect_answer_template_subject_and_tokenization():
    facts = [
        Fact(0, "{} is in", "Paris", " France"),
        Fact(1, "{} is in", "Lyon", " France"),
        Fact(2, "{} is in", "Madrid", " Spain"),
        Fact(3, "{} is in", "Rome", " Italy"),
        Fact(4, "{} is in", "London", " England"),
        Fact(5, "{} was founded in", "London", " 47"),
    ]
    pairs = swap_pairs(facts, list(range(6)), 5, lambda fact, answer: fact.id != 2)
    assert pairs == [(0, 3), (1, 4)]
    assert facts[4].split == facts[5].split
