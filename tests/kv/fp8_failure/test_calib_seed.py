"""The multi-seed fix: calib_prompts must draw DISJOINT prompt sets for different seeds (without it
every seed drew identical prompts and the CIs were degenerate), while seed=None stays the back-compat
first-n prefix. Tested on the pure _select_chunks helper (offline, no dataset)."""

import k_bias_common as kbc


def test_select_chunks_seed_none_is_first_n_prefix():
    ids = list(range(1000))
    out = kbc._select_chunks(ids, n=3, seq_len=10, seed=None)
    assert out == [ids[0:10], ids[10:20], ids[20:30]]  # deterministic first-n
    assert kbc._select_chunks(ids, 3, 10) == out  # default == seed=None


def test_select_chunks_seeds_are_disjoint_and_deterministic():
    ids = list(range(1000))
    a0 = kbc._select_chunks(ids, n=4, seq_len=10, seed=0)
    a0b = kbc._select_chunks(ids, n=4, seq_len=10, seed=0)
    a1 = kbc._select_chunks(ids, n=4, seq_len=10, seed=1)
    assert a0 == a0b  # same seed -> same prompts (reproducible)
    assert a0 != a1  # different seeds -> different prompts
    # each chunk is seq_len long and aligned to the grid
    assert all(len(c) == 10 for c in a0)
    starts = {c[0] for c in a0}
    assert all(s % 10 == 0 for s in starts)  # seq_len-aligned offsets
    assert len(starts) == 4  # 4 distinct chunks (no duplicates within a seed)
