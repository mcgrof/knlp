# SPDX-License-Identifier: MIT
"""Restore exact serving traces from shared prompts and explicit arrivals.

The catalog preserves original request IDs and nanosecond offsets. It does
not regenerate arrivals from a random seed or duplicate the prompt corpus.
Restored bytes must match the recorded trace hashes before they are written.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith(".gz") else raw)


def restore(entry, prompts):
    requests = []
    for row in entry["requests"]:
        if set(row) != {"id", "offset_ns", "prompt_index", "max_tokens"}:
            raise ValueError("unexpected compact request fields")
        index = row["prompt_index"]
        if type(index) is not int or not 0 <= index < len(prompts):
            raise ValueError("prompt index is outside the frozen panel")
        requests.append(
            {
                "id": row["id"],
                "offset_ns": row["offset_ns"],
                "prompt_token_ids": prompts[index],
                "max_tokens": row["max_tokens"],
            }
        )
    fields = entry["field_order"]
    if len(fields) != len(set(fields)) or set(fields) != {
        *entry["metadata"],
        "requests",
    }:
        raise ValueError("trace field inventory changed")
    trace = {
        key: requests if key == "requests" else entry["metadata"][key] for key in fields
    }
    raw = (json.dumps(trace, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("restored trace differs from its measured bytes")
    return raw


def restore_bundle(catalog, manifest_path, output=None):
    if catalog["schema"] != 1:
        raise ValueError("unsupported trace catalog schema")
    raw = Path(manifest_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != catalog["manifest_sha256"]:
        raise ValueError("prompt manifest differs from the frozen input")
    manifest = read_json(manifest_path)
    length = catalog["input_tokens"]
    samples = [
        row
        for row in manifest["samples"]
        if row["kind"] == "nll" and row["length"] == length and not row.get("reserve")
    ]
    if length not in (1024, 8192) or len(samples) != 32:
        raise ValueError("expected a frozen 32-document context panel")
    prompts = [row["token_ids"][:length] for row in samples]
    if any(len(row) != length for row in prompts):
        raise ValueError("a frozen prompt is too short")
    names = list(catalog["traces"])
    if not names or any(
        name == "."
        or PurePosixPath(name).is_absolute()
        or ".." in PurePosixPath(name).parts
        or "\\" in name
        or str(PurePosixPath(name)) != name
        for name in names
    ):
        raise ValueError("trace names must be normalized relative paths")
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
    verified = {}
    for name, entry in catalog["traces"].items():
        if entry["metadata"]["manifest_sha256"] != catalog["manifest_sha256"]:
            raise ValueError("trace and catalog name different prompt manifests")
        data = restore(entry, prompts)
        if output is not None:
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
        verified[name] = {"sha256": entry["sha256"], "bytes": len(data)}
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verified = restore_bundle(read_json(args.catalog), args.manifest, args.output)
    print(json.dumps({"verified": verified, "traces": len(verified)}, indent=2))


if __name__ == "__main__":
    main()
