#!/usr/bin/env python3
"""Convert the released FP32 EMO checkpoint to a verified BF16 copy on CPU.

The original directory is never modified. Each output shard holds the same
tensors as its source shard, cast with ``tensor.to(torch.bfloat16)``. After
writing a shard, the script reopens it and checks every tensor bit for bit
against a fresh cast of the source tensor. It writes a manifest with the
source and output hashes, shapes and byte offsets of every tensor, which the
later loading experiments use as their reference.

Usage:
    convert_bf16.py SRC_FP32_DIR DST_BF16_DIR MANIFEST.json
"""

import argparse
import json
import os
import shutil
import struct
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import file_sha256, runtime_identity, tensor_sha256, write_json  # noqa

COPY_FILES = (
    "configuration_emo.py",
    "modeling_emo.py",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "README.md",
)


def read_header(path):
    """Return (header_len, header_dict) of a safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return n, hdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("manifest")
    args = ap.parse_args()

    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    if os.path.exists(dst) and os.listdir(dst):
        sys.exit(f"refusing to write into non-empty {dst}")
    os.makedirs(dst, exist_ok=True)

    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))

    manifest = {
        "source_dir": src,
        "output_dir": dst,
        "runtime": runtime_identity(),
        "cast": "torch.Tensor.to(torch.bfloat16) (round to nearest even)",
        "shards": {},
        "tensors": {},
    }
    t_all = time.time()
    total_out = 0
    for shard in shards:
        t0 = time.time()
        sp, dp = os.path.join(src, shard), os.path.join(dst, shard)
        out = {}
        src_hash = {}
        with safe_open(sp, framework="pt") as f:
            meta = f.metadata()
            for name in f.keys():
                t = f.get_tensor(name)
                if t.dtype != torch.float32:
                    sys.exit(f"{name}: expected float32, got {t.dtype}")
                src_hash[name] = tensor_sha256(t)
                out[name] = t.to(torch.bfloat16).contiguous()
                del t
        save_file(out, dp, metadata=meta or {"format": "pt"})
        del out

        # Verify: reopen and compare with a fresh cast of the source tensor.
        mismatches = []
        with safe_open(sp, framework="pt") as fs, safe_open(dp, framework="pt") as fd:
            for name in fs.keys():
                ref = fs.get_tensor(name).to(torch.bfloat16)
                got = fd.get_tensor(name)
                if got.dtype != torch.bfloat16 or got.shape != ref.shape:
                    mismatches.append(name)
                    continue
                if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                    mismatches.append(name)
                manifest["tensors"][name] = {
                    "shard": shard,
                    "shape": list(got.shape),
                    "dtype": "bfloat16",
                    "src_fp32_sha256": src_hash[name],
                    "bf16_sha256": tensor_sha256(got),
                }
        hlen, hdr = read_header(dp)
        data_start = 8 + hlen
        for name, ent in hdr.items():
            if name == "__metadata__":
                continue
            a, b = ent["data_offsets"]
            manifest["tensors"][name]["file_offset"] = data_start + a
            manifest["tensors"][name]["nbytes"] = b - a
        size = os.path.getsize(dp)
        total_out += size
        manifest["shards"][shard] = {
            "bytes": size,
            "sha256": file_sha256(dp),
            "src_bytes": os.path.getsize(sp),
            "data_start": data_start,
            "verify_mismatches": mismatches,
            "seconds": round(time.time() - t0, 1),
        }
        print(
            f"{shard}: {size/2**30:.3f} GiB, {len(mismatches)} mismatches, "
            f"{time.time()-t0:.1f} s",
            flush=True,
        )
        if mismatches:
            sys.exit(f"verification failed in {shard}: {mismatches[:5]}")

    # Index and config: same weight map, BF16 dtype.
    index["metadata"] = dict(index.get("metadata", {}))
    index["metadata"]["total_size"] = sum(
        t["nbytes"] for t in manifest["tensors"].values()
    )
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    cfg["dtype"] = "bfloat16"
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for fn in COPY_FILES:
        if os.path.exists(os.path.join(src, fn)):
            shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
    manifest["copied_files"] = {
        fn: file_sha256(os.path.join(dst, fn))
        for fn in COPY_FILES + ("config.json", "model.safetensors.index.json")
        if os.path.exists(os.path.join(dst, fn))
    }
    manifest["total_bf16_tensor_bytes"] = index["metadata"]["total_size"]
    manifest["total_bf16_file_bytes"] = total_out
    manifest["seconds"] = round(time.time() - t_all, 1)
    write_json(args.manifest, manifest)
    print(f"done: {total_out/2**30:.3f} GiB in {manifest['seconds']} s")


if __name__ == "__main__":
    main()
