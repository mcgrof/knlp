# SPDX-License-Identifier: GPL-2.0
"""Generate qualification fixtures on the CPU, once, and hash them.

Two devices given the same seed do not produce the same random tensor, so a
seed is not a shared input and "seeded identically" is not a comparison. These
are generated once on the host, hashed, and transferred, so every device that
qualifies an operator is answering a question about the same bits.

Env: none; CPU only.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import torch


def fixture_hash(t: torch.Tensor) -> str:
    x = t.detach().to("cpu").contiguous().to(torch.float64)
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()[:32]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=64, help="token positions")
    ap.add_argument("--features", type=int, required=True, help="source width")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    x = torch.randn(args.rows, args.features, generator=g, dtype=torch.float64)
    # A spread of magnitudes, because a fixture drawn from one scale never
    # exercises the rounding behaviour that matters near the ends of a dtype.
    decades = torch.logspace(-3, 1, args.rows, dtype=torch.float64).unsqueeze(1)
    x = (x * decades * args.scale).to(torch.float32)
    h = fixture_hash(x)
    torch.save({"features": x, "sha256": h, "config": vars(args)}, args.out)
    print(json.dumps({"sha256": h, "shape": list(x.shape), "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
