"""Deterministic source-construction and parity tests for the
control-aware target transforms.  Pure CPU, no cartridges package, no
GPU: run with `python -m pytest` or directly as a script."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from targets import (  # noqa: E402
    build_target_set,
    calibrate_content_anchors,
    calibrate_scale,
    canonical_row,
    parse_element,
    transform_hash,
)

EOT = 151645


def flat(rows):
    """rows: list of (row_idx, [(id, logprob), ...]) -> flat arrays."""
    idxs, ids, lps = [], [], []
    for ri, entries in rows:
        for tid, lp in entries:
            idxs.append(ri)
            ids.append(tid)
            lps.append(lp)
    return idxs, ids, lps


def greedy_row(ri, chosen, lp_chosen, others):
    """[sampled] + [top-k] layout with the sampled token duplicated as
    top-1 (the greedy-synthesis case)."""
    return (ri, [(chosen, lp_chosen), (chosen, lp_chosen)] + others)


def loss_from(tset, logprob_table, denom):
    """Reference loss: sum p * nll over entries / denom, with the
    student per-(row, token) log-probability looked up in a dense
    table that carries grad."""
    total = None
    for ri, tid, p in zip(tset.row_idxs, tset.token_ids, tset.probs):
        term = -p * logprob_table[ri][tid]
        total = term if total is None else total + term
    return total / denom


def test_greedy_dedup_single_entry():
    et = parse_element(*flat([greedy_row(5, 7, -0.1, [])]), eot_token_id=EOT)
    ts = build_target_set(et, "dedup_legacy_support")
    assert ts.token_ids == [7] and len(ts.token_ids) == 1


def test_nongreedy_no_duplicate():
    et = parse_element(
        *flat([(5, [(3, -1.2), (7, -0.5), (9, -1.0)])]), eot_token_id=EOT
    )
    assert not et.rows[0].is_duplicated
    ts = build_target_set(et, "legacy_grouped_replay")
    assert sorted(ts.token_ids) == [3, 7, 9]


def test_conflicting_duplicate_probs_raise():
    try:
        parse_element(*flat([(5, [(7, -0.1), (7, -0.4)])]), eot_token_id=EOT)
    except ValueError:
        return
    raise AssertionError("conflicting duplicate logprobs must raise")


def test_noncontiguous_row_raises():
    try:
        parse_element([1, 2, 1], [5, 6, 7], [-1.0, -1.0, -1.0], eot_token_id=EOT)
    except ValueError:
        return
    raise AssertionError("non-contiguous row run must raise")


def test_legacy_identity_loss_and_grad():
    rows = [
        greedy_row(3, 7, -0.2, [(8, -2.0)]),
        (4, [(9, -0.7), (11, -1.1)]),
        greedy_row(9, EOT, -0.15, [(2, -2.5)]),
    ]
    et = parse_element(*flat(rows), eot_token_id=EOT)
    torch.manual_seed(0)
    table = {
        ri: (-torch.rand(200000, dtype=torch.float64) - 0.01).requires_grad_()
        for ri in (3, 4, 9)
    }
    denom = et.n_serialized
    raw = loss_from(build_target_set(et, "legacy_raw"), table, denom)
    grouped = loss_from(build_target_set(et, "legacy_grouped_replay"), table, denom)
    assert torch.allclose(raw, grouped, atol=1e-12), (raw, grouped)
    g_raw = torch.autograd.grad(raw, list(table.values()), retain_graph=True)
    g_grp = torch.autograd.grad(grouped, list(table.values()))
    for a, b in zip(g_raw, g_grp):
        assert torch.allclose(a, b, atol=1e-12)
    unique = loss_from(build_target_set(et, "dedup_legacy_support"), table, denom)
    assert float(raw) > float(unique)


def test_control_rows_first_and_eot_once():
    rows = [
        greedy_row(3, 7, -0.2, [(8, -2.0)]),
        greedy_row(4, 9, -0.5, [(1, -3.0)]),
        greedy_row(9, EOT, -0.15, [(2, -2.5)]),
    ]
    et = parse_element(*flat(rows), eot_token_id=EOT)
    assert et.first_row_idx == 3 and et.eot_row_idx == 9
    ts = build_target_set(et, "control_anchor")
    anchors = len(ts.token_ids) - len(
        build_target_set(et, "dedup_legacy_support").token_ids
    )
    assert anchors == 2  # first row + eot row, middle row excluded
    one = parse_element(*flat([greedy_row(3, EOT, -0.1, [])]), eot_token_id=EOT)
    assert one.control_rows() == [3]  # same row is first AND eot: once
    ts1 = build_target_set(one, "control_anchor")
    assert len(ts1.token_ids) == 2  # one unique + exactly one anchor


def test_no_eot_when_capped():
    rows = [greedy_row(3, 7, -0.2, []), greedy_row(9, 12, -0.4, [])]
    et = parse_element(*flat(rows), eot_token_id=EOT)
    assert et.eot_row_idx is None


def test_scale_and_content_calibration():
    rows = [
        greedy_row(3, 7, -0.2, [(8, -2.0)]),
        greedy_row(4, 9, -0.5, [(1, -3.0)]),
        greedy_row(5, 2, -0.9, [(6, -2.2)]),
        greedy_row(9, EOT, -0.15, [(2, -2.5)]),
    ]
    ets = [parse_element(*flat(rows), eot_token_id=EOT)]
    s = calibrate_scale(ets)
    legacy = build_target_set(ets[0], "legacy_raw").coefficient_mass()
    scaled = build_target_set(ets[0], "dedup_scale_matched", scale=s)
    assert abs(scaled.coefficient_mass() - legacy) < 1e-9
    per_rows, cscale, report = calibrate_content_anchors(ets)
    assert report["selected_count"] == report["control_count"] == 2
    content = build_target_set(
        ets[0],
        "content_anchor_matched",
        content_rows=per_rows[0],
        content_scale=cscale,
    )
    control = build_target_set(ets[0], "control_anchor")
    assert abs(content.coefficient_mass() - control.coefficient_mass()) < 1e-9
    for ri in per_rows[0]:
        assert ri not in ets[0].control_rows()


def test_canonical_threshold_reached():
    _, _, ids, lps = canonical_row(
        7, -0.01, [7, 8, 9], [-0.01, -6.0, -7.0], mass_threshold=0.9
    )
    assert ids == [7] and len(lps) == 1


def test_canonical_threshold_never_reached_retains_all():
    lp = float(torch.log(torch.tensor(0.2)))
    _, _, ids, _ = canonical_row(1, lp, [1, 2, 3], [lp, lp, lp], mass_threshold=0.998)
    assert sorted(ids) == [1, 2, 3]


def test_canonical_chosen_outside_topk_unioned_once():
    cid, clp, ids, _ = canonical_row(5, -3.0, [8, 9], [-0.4, -1.6], mass_threshold=1.0)
    assert cid == 5 and clp == -3.0 and ids.count(5) == 1


def test_canonical_mass_over_one_raises():
    try:
        canonical_row(1, -0.001, [1, 2], [-0.001, -0.001])
    except ValueError:
        return
    raise AssertionError("mass above 1+tolerance must raise")


def test_transform_hash_distinct():
    hs = {transform_hash(m) for m in ("legacy_raw", "dedup_legacy_support")}
    assert len(hs) == 2


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"PASS {name}")
    print(f"ALL {len(fns)} TESTS PASSED")
