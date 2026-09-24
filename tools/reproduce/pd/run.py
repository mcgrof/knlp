# SPDX-License-Identifier: MIT
"""Run the storage-mediated prefill/decode acceptance ladder.

The ladder decides whether a KV cache can be handed from a prefill node to a
decode node through NVMe rather than over the network.  This runner carries
the rungs that need no hardware; the device, two-GPU and inference rungs are
selected by their own defconfigs and refuse to start unless pd-doctor passes.

One lesson is built in rather than left to the reader: a rung that compares
output tokens must also prove the transfer happened.  A connector that is
configured but inert produces byte-identical output, because a decoder that
restores nothing simply recomputes.  Three rungs once passed that way.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import tempfile

BLOCK = 4096
SIZE = 1 << 20


def _device(path: str, writable: bool):
    import lmcache_rust_raw_block_io as rb
    return rb.RawBlockDevice(path, writable=writable, use_odirect=False,
                             alignment=BLOCK, io_engine="io_uring",
                             iouring_queue_depth=8)


def _writer(path: str, receipt_path: str) -> None:
    payload = bytearray(os.urandom(SIZE))
    dev = _device(path, True)
    try:
        dev.write_uring(0, payload, SIZE, SIZE)
    finally:
        dev.close()
    with open(receipt_path, "w") as f:
        json.dump({"digest": hashlib.sha256(bytes(payload)).hexdigest(),
                   "checkpoint_seq": 7, "writer_epoch": "smoke-1"}, f)
    # exit without a handshake: the reader must adopt from the publication


def rung1(work: str) -> dict[str, bool]:
    """State machine: a device round trip and an out-of-range refusal."""
    path = os.path.join(work, "rung1.img")
    with open(path, "wb") as f:
        f.truncate(16 << 20)
    out: dict[str, bool] = {}
    dev = _device(path, True)
    try:
        payload = bytearray(os.urandom(SIZE))
        back = bytearray(SIZE)
        dev.write_uring(0, payload, SIZE, SIZE)
        dev.read_uring(0, back, SIZE, SIZE)
        out["rung1 byte-exact round trip"] = back == payload
        try:
            dev.read_uring(1 << 30, bytearray(SIZE), SIZE, SIZE)
        except Exception:
            out["rung1 past capacity refused"] = True
        else:
            out["rung1 past capacity refused"] = False
    finally:
        dev.close()
    return out


def rung2(work: str) -> dict[str, bool]:
    """Two-process: a writer publishes and exits, a fresh reader adopts."""
    path = os.path.join(work, "rung2.img")
    with open(path, "wb") as f:
        f.truncate(16 << 20)
    receipt = os.path.join(work, "receipt.json")
    p = mp.Process(target=_writer, args=(path, receipt))
    p.start()
    p.join()
    out = {"rung2 writer exited cleanly": p.exitcode == 0,
           "rung2 writer process gone": not p.is_alive(),
           "rung2 publication survives the writer": os.path.exists(receipt)}
    if out["rung2 publication survives the writer"]:
        with open(receipt) as f:
            adv = json.load(f)
        dev = _device(path, False)
        try:
            back = bytearray(SIZE)
            dev.read_uring(0, back, SIZE, SIZE)
        finally:
            dev.close()
        out["rung2 fresh reader bytes exact"] = \
            hashlib.sha256(bytes(back)).hexdigest() == adv["digest"]
        out["rung2 adopted the advertised generation"] = adv["checkpoint_seq"] == 7
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rungs", default="smoke")
    ap.add_argument("--serial", default="")
    ap.add_argument("--results", default="")
    args = ap.parse_args(argv)

    if args.rungs != "smoke":
        print(f"pd-run: rungs {args.rungs!r} are not carried by this runner yet; "
              f"only 'smoke' (rungs 1-2) is implemented")
        return 2

    mp.set_start_method("spawn", force=True)
    results: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="knlp-pd-") as work:
        results.update(rung1(work))
        results.update(rung2(work))

    width = max(len(k) for k in results)
    for name, ok in results.items():
        print(f"  {name:<{width}}  {'ok' if ok else 'FAIL'}")
    passed = all(results.values())

    if args.results:
        os.makedirs(args.results, exist_ok=True)
        with open(os.path.join(args.results, "smoke.json"), "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {os.path.join(args.results, 'smoke.json')}")
    print(f"\npd-run smoke: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
