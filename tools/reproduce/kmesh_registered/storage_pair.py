# SPDX-License-Identifier: MIT
"""Run alternating cold-storage pairs with a real model and numerical gates."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    from verify import compare, digest

    parser = argparse.ArgumentParser(description=__doc__)
    names = (
        "vllm-source",
        "lmcache-source",
        "wheel-vllm-package",
        "model-dir",
        "trace",
        "manifest",
    )
    for name in (
        *names,
        "historical-reference",
        "historical-verification",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "storage-profile.json")
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.trials <= 50:
        parser.error("choose one through fifty matched pairs")
    profile = json.loads(args.profile.read_text())
    profile_hash = digest(args.profile)
    helper_hashes = {p.name: digest(p) for p in ROOT.glob("*.py")}
    if profile["storage"]["kind"] != "raw_block_comparison":
        parser.error("storage pair requires its explicit source/runtime profile")
    order = [
        list(("staged", "dma-buf") if i % 2 == 0 else ("dma-buf", "staged"))
        for i in range(args.trials)
    ]
    if args.plan:
        print(json.dumps({"profile": profile, "trial_order": order}, indent=2))
        return
    from storage import require_host_memlock

    require_host_memlock(profile["storage"])
    output = args.output_dir.resolve()
    for source in (ROOT.parents[2], args.lmcache_source, args.vllm_source):
        if output == source.resolve() or output.is_relative_to(source.resolve()):
            parser.error("storage captures must remain outside source checkouts")
    verified = json.loads(args.historical_verification.read_text())
    if (
        verified.get("status") != "complete"
        or verified.get("claims", {}).get("full_collective_lockstep_gate_pass")
        is not True
        or verified["arm_results"]["serial_oracle"]["sha256"]
        != digest(args.historical_reference / "result.json")
    ):
        raise ValueError("historical serial reference is not independently verified")
    output.mkdir(parents=True, exist_ok=False)
    common = [
        sys.executable,
        str(ROOT / "run.py"),
        "--profile",
        str(args.profile.resolve()),
    ]
    for name in names:
        common += ["--" + name, str(getattr(args, name.replace("-", "_")).resolve())]

    def run(relative, route, storage_route="cpu-l1", *, poison_arena=False):
        root = output / relative
        root.parent.mkdir(parents=True, exist_ok=True)
        with root.with_suffix(".log").open("x") as log:
            subprocess.run(
                common
                + [
                    "--route",
                    route,
                    "--storage-route",
                    storage_route,
                    "--output-dir",
                    str(root),
                ]
                + (["--poison-arena"] if poison_arena else []),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return root

    reference = run("reference", "reference")
    historical = compare(
        reference, args.historical_reference, args.historical_verification
    )
    write(reference / "comparison.json", historical)
    if not historical["passed"]:
        raise RuntimeError(
            "same-source reference differs from the historical serial baseline"
        )

    def check_storage(root, route, *, poison_arena=False):
        import storage_verify

        numerical = compare(root, reference, reference / "comparison.json")
        write(root / "comparison.json", numerical)
        if not numerical["passed"]:
            raise RuntimeError("storage/model pair differs from the serial reference")
        result = json.loads((root / "result.json").read_text())
        if (
            result["profile_sha256"] != profile_hash
            or result["helper_sha256"] != helper_hashes
            or result["storage_route"] != route
            or result["poison_arena"] is not poison_arena
        ):
            raise ValueError("a paired model run used different experiment inputs")
        checked = storage_verify.verify(
            route,
            result["storage_evidence"],
            profile["storage"],
            require_poison=poison_arena,
        )
        if result["storage_verification"] != checked:
            raise ValueError("storage verification differs from its actual evidence")
        return result

    qualifications = []
    for route in ("staged", "dma-buf"):
        root = run(f"qualification/{route}", "registered", route, poison_arena=True)
        check_storage(root, route, poison_arena=True)
        qualifications.append(
            {
                "route": route,
                "result_path": str(root.relative_to(output) / "result.json"),
                "result_sha256": digest(root / "result.json"),
                "comparison_sha256": digest(root / "comparison.json"),
                "poisoned_l1_arena": True,
            }
        )

    rows = []
    for trial, routes in enumerate(order):
        for route in routes:
            root = run(f"trial-{trial:02d}/{route}", "registered", route)
            result = check_storage(root, route)
            events = root / "events.jsonl"
            if digest(events) != result["event_ledger_sha256"]:
                raise ValueError("model timing events differ from the bound result")
            captured_events = [
                json.loads(line) for line in events.read_text().splitlines()
            ]
            consumed = [
                event
                for event in captured_events
                if event["action"] == "registered.model.consumed"
            ]
            by_request = {row["request_id"]: row for row in consumed}
            children = result["storage_verification"]["children"]
            if len(by_request) != profile["storage"]["children"] or set(by_request) != {
                row["request_id"] for row in children
            }:
                raise ValueError("actual consuming-model timing coverage differs")
            rows.append(
                {
                    "trial": trial,
                    "route": route,
                    "result_path": str(root.relative_to(output) / "result.json"),
                    "result_sha256": digest(root / "result.json"),
                    "comparison_sha256": digest(root / "comparison.json"),
                    "medians_ns": {
                        **{
                            name: statistics.median(child[name] for child in children)
                            for name in (
                                "restore_ns",
                                "storage_read_ns",
                                "checksum_ns",
                                "cold_prepare_ns",
                            )
                        },
                        "model_consume_ns": statistics.median(
                            row["processing_ns"] for row in consumed
                        ),
                    },
                }
            )
    paired = []
    for trial in range(args.trials):
        pair = {
            row["route"]: row["medians_ns"] for row in rows if row["trial"] == trial
        }
        paired.append(
            {
                "trial": trial,
                "dma_minus_staged_ns": {
                    name: pair["dma-buf"][name] - pair["staged"][name]
                    for name in pair["staged"]
                },
            }
        )
    write(
        output / "paired.json",
        {
            "status": "complete",
            "passed": True,
            "trial_order": order,
            "reference_result_sha256": digest(reference / "result.json"),
            "profile_sha256": profile_hash,
            "helper_sha256": helper_hashes,
            "data_arrival_qualifications": qualifications,
            "rows": rows,
            "paired_differences": paired,
            "limits": "Cold external L1 only; SSD caches may remain warm. Model instrumentation and duplicated reference storage remain active. These samples do not establish production serving speed or total memory savings.",
        },
    )


if __name__ == "__main__":
    main()
