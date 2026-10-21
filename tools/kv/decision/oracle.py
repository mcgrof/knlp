# SPDX-License-Identifier: MIT
"""Independent NumPy codebook oracle for retained native cache operands."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .cache import encode, decode


def oracle(x, scale):
    """Independent E4M3FN codebook, nearest/even; no Torch float8 conversion."""
    codes = np.arange(127, dtype=np.uint8)
    e = (codes // 8).astype(int)
    m = (codes % 8).astype(int)
    values = np.where(e == 0, m / 8 * 2.0 ** (-6), (1 + m / 8) * 2.0 ** (e - 7))
    y = x.float().numpy() / np.asarray(scale, dtype=np.float32)
    z = np.clip(np.abs(y), 0, 448)
    high = np.searchsorted(values, z, side="left").clip(0, 126)
    low = np.maximum(0, high - 1)
    dl = z - values[low]
    dh = values[high] - z
    q = np.where((dl < dh) | ((dl == dh) & (low % 2 == 0)), low, high).astype(np.uint8)
    return q | np.where(np.signbit(y), 128, 0).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", type=Path, required=True)
    a = p.parse_args()
    sc = json.loads((a.calibration / "scales.json").read_text())
    held = torch.load(
        a.calibration / "native-activation-fixtures.pt",
        map_location="cpu",
        weights_only=True,
    )
    rows = []
    for key, x in held.items():
        parts = key.split("/")
        layer = int(parts[1].split("-")[1])
        component = parts[2]
        for policy in ["fixed", "dynamic"]:
            scale = (
                torch.tensor(sc["scales"][layer][component], dtype=torch.float32)
                if policy == "fixed"
                else None
            )
            q, s = encode(x, scale)
            expected = oracle(x, s.numpy())
            same = np.array_equal(q.view(torch.uint8).numpy(), expected)
            reconstructed = decode(q, s, x.dtype)
            rows.append(
                dict(
                    fixture=key,
                    policy=policy,
                    codes_exact=same,
                    shape=list(x.shape),
                    native_dtype=str(x.dtype),
                    finite=bool(torch.isfinite(reconstructed).all()),
                )
            )
    result = dict(
        fixtures=len(held),
        comparisons=len(rows),
        passed=bool(rows) and all(r["codes_exact"] and r["finite"] for r in rows),
        rows=rows,
    )
    (a.calibration / "INDEPENDENT-ORACLE.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    if not result["passed"]:
        raise RuntimeError("Independent activation oracle failed")


if __name__ == "__main__":
    main()
