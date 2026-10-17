#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure exact family encoding on independently verified, content-aligned KV."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contract import digest, profile, source_identity, validate_sources
from verify_online_collective_lockstep_gate import (
    _load,
    _load_numerical_family,
    _semantic_group,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--lmcache-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    verified = json.loads(args.verification.read_text())
    if verified.get("status") != "complete" or not verified.get("claims", {}).get(
        "full_collective_lockstep_gate_pass"
    ):
        raise ValueError("independent eight-case verification is required")
    expected_codec = profile()["source_pairs"][verified["source_pair"]]["lmcache"]
    codec_source = source_identity(args.lmcache_source)
    if codec_source["status_porcelain"] or any(
        codec_source[key] != value for key, value in expected_codec.items()
    ):
        raise ValueError("codec source must match the clean dated LMCache pin")
    results = {}
    for index, arm in enumerate(profile()["arms"]):
        path = args.bundle / "arms" / f"{index:02d}-{arm}" / "artifacts/result.json"
        if digest(path) != verified["arm_results"][arm]["sha256"]:
            raise ValueError("capture differs from independent verification")
        results[arm] = _load(path, arm)
        validate_sources(results[arm]["source"], verified["source_pair"])
    forward = results["collective_forward"]
    family = _load_numerical_family(
        args.bundle / "arms/02-collective_forward/artifacts/result.json",
        forward,
        "collective_forward",
    )
    semantic = _semantic_group(forward)
    import torch

    sys.path.insert(0, str(args.lmcache_source))
    from lmcache.v1.distributed.serde.kmesh_family import (
        KMeshFamilyCodecError,
        decode_family,
        encode_family,
    )

    tensors, spans = [], []
    for member in semantic["members"]:
        selected = member["selected_spans"]["spans"]
        if not selected or any(
            left[1] != right[0] for left, right in zip(selected, selected[1:])
        ):
            raise ValueError(
                "the family codec requires one contiguous content-aligned span"
            )
        values = family[member["trace_request_id"]]
        # Preserve every layer and element while moving tokens to the codec axis.
        tensor = (
            torch.stack((values["key"], values["value"]))
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(2, 512, -1)
        )
        tensors.append(tensor)
        spans.append((selected[0][0], selected[-1][1]))
    blob = encode_family(tensors, tuple(spans), 0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = args.output_dir / "family.bin"
    with payload.open("xb") as stream:
        stream.write(blob)
    recovered = decode_family(payload.read_bytes())
    exact = [
        bool(want.view(torch.int16).eq(got.view(torch.int16)).all())
        for want, got in zip(tensors, recovered, strict=True)
    ]
    if not all(exact):
        raise ValueError("family restoration is not bitwise exact")
    damaged = bytearray(blob)
    damaged[-1] ^= 1
    try:
        decode_family(bytes(damaged))
    except KMeshFamilyCodecError:
        corruption_rejected = True
    else:
        raise ValueError("codec accepted a corrupted payload")
    dense = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    result = {
        "schema_version": 1,
        "kind": "offline_family_encoding",
        "members": len(tensors),
        "source_pair": verified["source_pair"],
        "codec_source": expected_codec,
        "verified_summary_sha256": digest(args.verification),
        "encoded_sha256": digest(payload),
        "dense_bytes": dense,
        "encoded_bytes": len(blob),
        "encoded_to_dense": len(blob) / dense,
        "all_members_bitwise_exact": all(exact),
        "corruption_rejected": corruption_rejected,
        "live_gpu_page_sharing": False,
        "performance_claim": False,
    }
    with (args.output_dir / "codec.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
