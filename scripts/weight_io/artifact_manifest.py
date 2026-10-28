#!/usr/bin/env python3
"""Write an external artifact manifest for a weight-I/O experiment root.

Each entry names a logical artifact, its location on the producing host,
its size and content hash (a SHA-256 of the file, or of a sorted list of
per-file hashes for a directory), the producing code, owner and
retention, and any verified durable copy. Bulk data stays outside Git;
this manifest is what gets committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def dir_entry(root: Path, rel: str, hash_files: bool) -> dict:
    d = root / rel
    files = sorted(p for p in d.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    ent = {"path": str(d), "n_files": len(files), "bytes": total}
    if hash_files:
        listing = [f"{sha256_file(p)}  {p.relative_to(d)}" for p in files]
        ent["sha256_of_file_list"] = hashlib.sha256(
            "\n".join(listing).encode()
        ).hexdigest()
        ent["file_list_path"] = str(d) + ".sha256.txt"
        Path(ent["file_list_path"]).write_text("\n".join(listing) + "\n")
    return ent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--tools-commit", required=True)
    ap.add_argument(
        "--retention",
        default="until the owning workflow is closed; then cold archive or delete",
    )
    ap.add_argument(
        "--durable",
        action="append",
        default=[],
        help="logical=location of a verified durable copy; may repeat",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    durable = dict(x.split("=", 1) for x in args.durable)
    host = socket.gethostname()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def ent(name, kind, source, **extra):
        e = {
            "logical_name": name,
            "kind": kind,
            "host": host,
            "source": source,
            "owner": args.owner,
            "retention": args.retention,
            "durable_copy": durable.get(name),
            "verified_utc": now,
            **extra,
        }
        return e

    arts = []
    f = root / "stageA/inventory.jsonl"
    arts.append(
        ent(
            "inventory.jsonl",
            "file",
            f"hf_inventory.py headers @ {args.tools_commit}",
            path=str(f),
            bytes=f.stat().st_size,
            sha256=sha256_file(f),
        )
    )
    arts.append(
        ent(
            "stageA",
            "directory",
            f"hf_inventory.py resolve/headers/classify/select @ {args.tools_commit}",
            **dir_entry(root, "stageA", hash_files=False),
        )
    )
    arts.append(
        ent(
            "runs",
            "directory",
            f"stageB_matrix.py / traced_replay.py / gpu_* @ {args.tools_commit}; per-run manifest.json inside",
            **dir_entry(root, "runs", hash_files=True),
        )
    )
    for md in sorted((root / "models").iterdir()):
        if not md.is_dir():
            continue
        for sub in ("full", "compact"):
            if (md / sub).exists():
                arts.append(
                    ent(
                        f"models/{md.name}/{sub}",
                        "directory",
                        (
                            f"hf_inventory.py fetch-full @ {args.tools_commit}; verified against hub LFS SHA-256"
                            if sub == "full"
                            else f"hf_inventory.py fetch @ {args.tools_commit}; per-range SHA-256 in fetch_manifest.jsonl"
                        ),
                        **dir_entry(root, f"models/{md.name}/{sub}", hash_files=False),
                    )
                )
    fm = root / "models/fetch_manifest.jsonl"
    arts.append(
        ent(
            "fetch_manifest.jsonl",
            "file",
            f"hf_inventory.py fetch @ {args.tools_commit}",
            path=str(fm),
            bytes=fm.stat().st_size,
            sha256=sha256_file(fm),
        )
    )
    for name in ("metrics.csv", "metrics_medians.csv", "model_projections.json"):
        p = root / name
        if p.exists():
            arts.append(
                ent(
                    name,
                    "file",
                    f"aggregate_runs.py / model_projections.py @ {args.tools_commit}",
                    path=str(p),
                    bytes=p.stat().st_size,
                    sha256=sha256_file(p),
                )
            )
    doc = {
        "experiment_root": str(root),
        "host": host,
        "generated_utc": now,
        "tools_commit": args.tools_commit,
        "artifacts": arts,
    }
    Path(args.out).write_text(json.dumps(doc, indent=1))
    print(f"wrote {args.out}: {len(arts)} artifacts")


if __name__ == "__main__":
    main()
