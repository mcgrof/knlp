# SPDX-License-Identifier: GPL-2.0
"""Decision arithmetic for the unrelated-probe pre-training gate."""

from research.kv_translate.measure_probe_dependence import decide, percentile


def test_percentile_is_nearest_rank():
    assert percentile([4, 1, 3, 2], 0.5) == 2
    assert percentile([4, 1, 3, 2], 0.95) == 4


def _rows(unrelated_kl=0.02, flip=0.02, repeat=1e-8):
    rows = []
    for document_id in range(4):
        rows.append(
            {
                "document_id": document_id,
                "probe_type": "unrelated",
                "comparison": "repeat",
                "kl_mean": repeat,
                "argmax_flip": 0.0,
                "logit_l2_mean": 0.0,
            }
        )
        for probe_type, kl in (("relevant", 0.2), ("unrelated", unrelated_kl)):
            for comparison in ("wrong_document", "neutral_document"):
                rows.append(
                    {
                        "document_id": document_id,
                        "probe_type": probe_type,
                        "comparison": comparison,
                        "kl_mean": kl,
                        "argmax_flip": flip,
                        "logit_l2_mean": kl,
                    }
                )
    return rows


def test_gate_requires_absolute_kl_and_argmax_signal():
    passed = decide(
        _rows(), absolute_kl_min=0.01, repeat_multiplier=100, argmax_flip_min=0.01
    )
    assert passed["passes"]
    assert passed["verdict"] == "UNRELATED_PROBE_SIGNAL_PRESENT"
    weak_kl = decide(
        _rows(unrelated_kl=0.009),
        absolute_kl_min=0.01,
        repeat_multiplier=100,
        argmax_flip_min=0.01,
    )
    assert not weak_kl["passes"]
    weak_flip = decide(
        _rows(flip=0.009),
        absolute_kl_min=0.01,
        repeat_multiplier=100,
        argmax_flip_min=0.01,
    )
    assert not weak_flip["passes"]


def test_repeat_floor_can_dominate_the_absolute_floor():
    result = decide(
        _rows(unrelated_kl=0.02, repeat=3e-4),
        absolute_kl_min=0.01,
        repeat_multiplier=100,
        argmax_flip_min=0.01,
    )
    assert result["signal_kl_floor"] == 0.03
    assert not result["passes"]
