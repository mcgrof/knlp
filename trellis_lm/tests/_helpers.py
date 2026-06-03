"""Test helpers (importable as trellis_lm.tests._helpers)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trellis_lm.config import TrellisConfig  # noqa: E402


def tiny_cfg(**kw):
    base = dict(
        vocab_size=64, d_model=32, n_layers=2, n_heads=2, d_head=16,
        n_slots=8, max_seq_len=64, dtype="fp32", conv_kernel=4,
    )
    base.update(kw)
    return TrellisConfig(**base)
