# SPDX-License-Identifier: MIT
"""Compare saved model tensors with the independently verified serial baseline."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_file(root, binding):
    path = (root / binding["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("tensor path escapes the capture or is absent")
    if path.stat().st_size != binding["nbytes"] or digest(path) != binding["sha256"]:
        raise ValueError("saved tensor bytes differ from the capture manifest")
    return path


def compare(candidate, reference, verification):
    import torch
    from safetensors.torch import load_file

    candidate, reference = Path(candidate), Path(reference)
    result = json.loads((candidate / "result.json").read_text())
    old_result = json.loads((reference / "result.json").read_text())
    verified = json.loads(Path(verification).read_text())
    paired = old_result.get("kind") == "serial_model_reference"
    if paired:
        if (
            verified.get("passed") is not True
            or verified.get("candidate_result_sha256")
            != digest(reference / "result.json")
            or result["source"] != old_result["source"]
            or result["lmcache_config_sha256"] != old_result["lmcache_config_sha256"]
        ):
            # Source paths must also match: run each route from the same frozen
            # checkout, then compare this reference with the historical result.
            raise ValueError("paired reference lacks a passing historical comparison")
    elif (
        verified.get("status") != "complete"
        or not verified.get("claims", {}).get("full_collective_lockstep_gate_pass")
        or digest(reference / "result.json")
        != verified["arm_results"]["serial_oracle"]["sha256"]
    ):
        raise ValueError("reference differs from independent verification")
    if (
        result.get("kind")
        not in ("registered_native_serial_model", "serial_model_reference")
        or result.get("status") != "complete"
    ):
        raise ValueError("candidate is not a completed registered model replay")
    if result["runtime"] != old_result["runtime"]:
        # Capture time, executable/module paths and platform strings are evidence,
        # not scientific settings. Compare the explicit frozen runtime fields.
        pins = json.loads(
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("tokendance/profile.json")
            .read_text()
        )["rtx4090_cuda"]
        for runtime in (result["runtime"], old_result["runtime"]):
            if any(runtime.get(k) != v for k, v in pins["runtime"].items()):
                raise ValueError("model runtime differs from the frozen reference")
    outcomes = []
    manifests = [
        json.loads((p / "post-pic-capture-v1/family-manifest.json").read_text())
        for p in (candidate, reference)
    ]
    if manifests[0]["tensor_contract"] != manifests[1]["tensor_contract"]:
        raise ValueError("model capture geometry changed")
    groups = [x["members"] for x in manifests]
    if any(
        len(g) != 4 or [m["sequence_index"] for m in g] != [1, 2, 3, 4] for g in groups
    ):
        raise ValueError("model family coverage is incomplete")
    masks = []
    for value in (result, old_result):
        masks.append(
            {
                m["trace_request_id"]: (m["selected_spans"], m["retrieval_mask"])
                for m in value["recovery_evidence"]["serial_members"]
            }
        )
    masks_equal = masks[0] == masks[1] and len(masks[0]) == 4
    for root, capture_result in ((candidate, result), (reference, old_result)):
        evidence = capture_result.get(
            "numerical_capture", capture_result.get("numerical_evidence")
        )
        if evidence.get("status") != "complete" or evidence.get(
            "manifest_sha256"
        ) != digest(root / "post-pic-capture-v1/family-manifest.json"):
            raise ValueError("capture manifest differs from the bound model result")

    def selections(value):
        bindings = {
            b["engine_request_id"]: b["trace_request_id"]
            for b in value["recovery_evidence"]["request_id_bindings"]
        }
        rows = value.get("selection_evidence", [])
        selected = {}
        for row in rows:
            key = (bindings[row["request_id"]], row["layer_id"])
            if key in selected:
                raise ValueError("duplicate model selection observation")
            selected[key] = row["selected_indices"]
        if len(selected) != 96 or any(
            set(
                layer
                for trace, layer in selected
                if trace == member["trace_request_id"]
            )
            != set(range(24))
            for member in value["observations"][1:]
        ):
            raise ValueError("actual model selection coverage is incomplete")
        return selected

    selection_equal = selections(result) == selections(old_result) if paired else None
    for actual, expected in zip(*groups, strict=True):
        if any(
            actual[k] != expected[k]
            for k in ("sequence_index", "trace_request_id", "prompt_token_ids_sha256")
        ):
            raise ValueError("model member identity changed")
        row = {"sequence_index": actual["sequence_index"]}
        for field in ("tensor_file", "logits_file"):
            values = [
                load_file(
                    str(bound_file(p / "post-pic-capture-v1", m[field])), device="cpu"
                )
                for p, m in ((candidate, actual), (reference, expected))
            ]
            if values[0].keys() != values[1].keys():
                raise ValueError("captured tensor names changed")
            for key in values[0]:
                a, b = values[0][key], values[1][key]
                if (
                    a.shape != b.shape
                    or a.dtype != b.dtype
                    or not torch.isfinite(a).all()
                    or not torch.isfinite(b).all()
                ):
                    raise ValueError("invalid captured tensor geometry or values")
                exact = torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                close = torch.allclose(a, b, atol=1e-5, rtol=1e-4)
                row[field + ":" + key] = {
                    "bitwise_equal": exact,
                    "max_abs_difference": float((a.float() - b.float()).abs().max()),
                    "passed": exact if field == "tensor_file" else close,
                }
        outcomes.append(row)
    tokens_equal = [o["output_token_ids"] for o in result["observations"]] == [
        o["output_token_ids"] for o in old_result["observations"]
    ]
    passed = (
        tokens_equal
        and masks_equal
        and selection_equal is not False
        and all(
            v["passed"] for r in outcomes for v in r.values() if isinstance(v, dict)
        )
    )
    return {
        "status": "complete",
        "passed": passed,
        "generated_tokens_equal": tokens_equal,
        "retrieval_masks_equal": masks_equal,
        "actual_selection_equal": selection_equal,
        "members": outcomes,
        "candidate_result_sha256": digest(candidate / "result.json"),
        "reference_result_sha256": digest(reference / "result.json"),
        "reference_verification_sha256": digest(verification),
        "policy": "bitwise K/V and identical retrieval masks; logits atol=1e-5, rtol=1e-4; exact generated tokens",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "reference", "verification", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.candidate, args.reference, args.verification)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    if not result["passed"]:
        raise SystemExit("registered model differs from the serial reference")


if __name__ == "__main__":
    main()
