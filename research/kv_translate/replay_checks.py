# SPDX-License-Identifier: GPL-2.0
"""What a correct sparse replay must do, stated as things a wrong one fails.

Each check here exists because a previous version of the qualification could
have passed while the implementation was wrong in a specific way. They are
written as functions over tensors so they can be exercised on a processor
against a small model, and then run unchanged on the card against the real
pair — the same code, so a pass on one is a statement about the same property
as a pass on the other.

Two ideas run through them. Keys and values are checked separately everywhere,
because the earlier version checked keys and inferred values. And the negative
controls matter more than the positive ones: that a full replay reproduces a
native prefill is necessary and easy to achieve accidentally, whereas a replay
that produces the same answer from three different starting caches has
demonstrated that it really did recompute rather than pass its input through.
"""

from __future__ import annotations

import torch


def _both(keys_a, values_a, keys_b, values_b, fn):
    """Apply a comparison to keys and to values, and keep them apart."""
    return {
        "keys": max(fn(a, b) for a, b in zip(keys_a, keys_b)),
        "values": max(fn(a, b) for a, b in zip(values_a, values_b)),
    }


def rel_frobenius(a, b):
    a32, b32 = a.float(), b.float()
    return float((a32 - b32).norm() / b32.norm().clamp(min=1e-30))


def rel_max(a, b):
    a32, b32 = a.float(), b.float()
    return float((a32 - b32).abs().max() / b32.abs().max().clamp(min=1e-30))


def one_minus_cos(a, b):
    a64, b64 = a.double().reshape(-1), b.double().reshape(-1)
    return float(1.0 - (a64 @ b64) / (a64.norm() * b64.norm()).clamp(min=1e-300))


def per_layer_profile(rep_k, rep_v, nat_k, nat_v):
    """The trend an earlier report described in prose and never recorded.

    Error that grows with depth is what reassociated accumulation through a
    stack looks like. Error that appears at one layer and stays is something
    else. Neither can be told from a maximum over layers, so the whole curve is
    archived and the reader can see which they have.
    """
    out = []
    for li, (rk, rv, nk, nv) in enumerate(zip(rep_k, rep_v, nat_k, nat_v)):
        out.append(
            {
                "layer": li,
                "keys_rel_frobenius": rel_frobenius(rk, nk),
                "values_rel_frobenius": rel_frobenius(rv, nv),
                "keys_one_minus_cos": one_minus_cos(rk, nk),
                "values_one_minus_cos": one_minus_cos(rv, nv),
            }
        )
    return out


def error_quantiles(rep, nat, qs=(0.5, 0.9, 0.99, 0.999, 1.0), cap=2_000_000):
    """Where the elementwise error actually sits, rather than its worst value.

    A maximum over a hundred million numbers describes a tail. These are the
    figures an earlier report quoted from a comment rather than from a
    computation; they are computed here and archived.
    """
    a = rep.float().reshape(-1)
    b = nat.float().reshape(-1)
    scale = b.abs().max().clamp(min=1e-30)
    d = (a - b).abs() / scale
    if d.numel() > cap:
        d = d[:: max(1, d.numel() // cap)]
    d = d.float()
    return {f"q{int(q * 1000)}": float(torch.quantile(d, q)) for q in qs} | {
        "n_sampled": int(d.numel()),
        "n_total": int(rep.numel()),
    }


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------


def check_layer0_exact(rep_k, rep_v, nat_k, nat_v):
    """The first layer must match native exactly, in keys and in values.

    It reads no cache, so its output is a function of the tokens fed, the
    positions given and the slots written. An exact match says those three are
    right and says nothing whatever about attention or about any later layer --
    which is why it is one check here rather than the argument.
    """
    return {
        "keys_exact": bool(torch.equal(rep_k[0], nat_k[0])),
        "values_exact": bool(torch.equal(rep_v[0], nat_v[0])),
        "scope": (
            "necessary input plumbing only: the first layer reads no cache, so "
            "this cannot detect a wrong attention mask or a wrong later layer"
        ),
    }


def check_initial_cache_independence(replay_fn, mapped, zero, wrong, tol):
    """A full replay must reach the same place from any starting cache.

    This is the check that distinguishes recomputing from passing through. If
    every position is recomputed then nothing of the initial cache survives,
    so starting from the translated cache, from zeros, and from a deliberately
    wrong cache must all give the same answer. An implementation that leaked
    its input -- by writing some layers and not others, or by scattering into
    the wrong slots -- fails here and can pass every agreement check above.
    """
    a_k, a_v = replay_fn(mapped)
    b_k, b_v = replay_fn(zero)
    c_k, c_v = replay_fn(wrong)
    mz = _both(a_k, a_v, b_k, b_v, rel_frobenius)
    mw = _both(a_k, a_v, c_k, c_v, rel_frobenius)
    worst = max(mz["keys"], mz["values"], mw["keys"], mw["values"])
    return {
        "mapped_vs_zero": mz,
        "mapped_vs_deterministic_wrong": mw,
        "worst": worst,
        "passed": bool(worst <= tol),
        "limit": tol,
        "why": (
            "a full replay recomputes every position, so no part of the cache "
            "it started from may survive into the result"
        ),
    }


def check_zero_mask_is_a_no_op(replay_fn_mask, keys, values):
    """Repairing nothing must change nothing, and must not copy either."""
    out_k, out_v, info = replay_fn_mask(
        torch.zeros_like(keys[0][0, 0, :, 0], dtype=torch.bool)
    )
    same_obj = out_k is keys and out_v is values
    equal = all(torch.equal(a, b) for a, b in zip(out_k, keys)) and all(
        torch.equal(a, b) for a, b in zip(out_v, values)
    )
    return {
        "returns_the_same_objects": bool(same_obj),
        "tensors_equal": bool(equal),
        "replayed_positions": int(info.get("replayed", -1)),
        "passed": bool(equal and info.get("replayed") == 0),
    }


def check_partial_mask_per_layer(before_k, before_v, after_k, after_v, mask):
    """Unselected positions must be untouched in every layer, keys and values.

    The earlier check folded the layers together with an or, which establishes
    that no unselected position moved somewhere but not that each layer was
    written, and it never looked at values at all. Here each layer is reported
    on its own, in both tensors, in both directions: unselected positions
    identical, selected positions actually changed.
    """
    keep = ~mask
    per_layer = []
    for li in range(len(before_k)):
        per_layer.append(
            {
                "layer": li,
                "unselected_keys_unchanged": bool(
                    torch.equal(after_k[li][:, :, keep, :], before_k[li][:, :, keep, :])
                ),
                "unselected_values_unchanged": bool(
                    torch.equal(after_v[li][:, :, keep, :], before_v[li][:, :, keep, :])
                ),
                "selected_keys_changed": bool(
                    not torch.equal(
                        after_k[li][:, :, mask, :], before_k[li][:, :, mask, :]
                    )
                ),
                "selected_values_changed": bool(
                    not torch.equal(
                        after_v[li][:, :, mask, :], before_v[li][:, :, mask, :]
                    )
                ),
            }
        )
    return {
        "layers": len(per_layer),
        "per_layer": per_layer,
        "all_unselected_preserved": all(
            p["unselected_keys_unchanged"] and p["unselected_values_unchanged"]
            for p in per_layer
        ),
        "all_selected_written": all(
            p["selected_keys_changed"] and p["selected_values_changed"]
            for p in per_layer
        ),
    }


def check_ownership_modes(replay_clone, replay_in_place, keys, values):
    """The path that was timed must be the path that was checked.

    The qualification ran the copying path and every latency figure came from
    the in-place one, so nothing linked them. Two things are required: the two
    modes agree exactly on their output, and the copying mode leaves the
    caller's tensors alone, which is the property the scoring loop depends on
    when it reads one mapped cache under several conditions.
    """
    before_k = [t.clone() for t in keys]
    before_v = [t.clone() for t in values]
    ck, cv = replay_clone()
    inputs_untouched = all(torch.equal(a, b) for a, b in zip(before_k, keys)) and all(
        torch.equal(a, b) for a, b in zip(before_v, values)
    )
    ik, iv = replay_in_place()
    agree_k = all(torch.equal(a, b) for a, b in zip(ck, ik))
    agree_v = all(torch.equal(a, b) for a, b in zip(cv, iv))
    return {
        "clone_left_inputs_unchanged": bool(inputs_untouched),
        "clone_and_in_place_keys_identical": bool(agree_k),
        "clone_and_in_place_values_identical": bool(agree_v),
        "passed": bool(inputs_untouched and agree_k and agree_v),
        "why": (
            "every archived latency figure was produced with in_place enabled "
            "while the qualification ran the copying path; without this the "
            "timed implementation is unqualified"
        ),
    }


def check_native_repeat(prefill_fn):
    """Is a native prefill of the same tokens reproducible on this card?

    An earlier report stated it was, as the reason a small disagreement could
    not be noise. No second prefill was ever run. It is run here, because the
    claim is cheap to test and load-bearing for every agreement figure.
    """
    k1, v1 = prefill_fn()
    k2, v2 = prefill_fn()
    ident = all(torch.equal(a, b) for a, b in zip(k1, k2)) and all(
        torch.equal(a, b) for a, b in zip(v1, v2)
    )
    return {
        "bitwise_identical": bool(ident),
        "keys_rel_frobenius": max(rel_frobenius(a, b) for a, b in zip(k1, k2)),
        "values_rel_frobenius": max(rel_frobenius(a, b) for a, b in zip(v1, v2)),
        "why": (
            "if two native prefills differ, that difference is the floor under "
            "every agreement figure and none of them can be read more finely"
        ),
    }
