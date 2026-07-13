# SPDX-License-Identifier: GPL-2.0
"""SymKV: symmetry-adapted KV-head mode compression (falsification-first R&D)."""

from .config import SymKVConfig, BASIS_METHODS
from .covariance import HeadCovariance
from .basis import consensus_mode, perp_projector, build_basis
from .codec import encode, decode, reconstruct, recon_mse, byte_accounting

__all__ = [
    "SymKVConfig", "BASIS_METHODS", "HeadCovariance",
    "consensus_mode", "perp_projector", "build_basis",
    "encode", "decode", "reconstruct", "recon_mse", "byte_accounting",
]
