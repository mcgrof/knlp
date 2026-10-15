"""Render and validate the LMCache transport boundary for DroidSpeak."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DOC_URLS = {
    "legacy_share": (
        "https://docs.lmcache.ai/getting_started/quickstart/share_kv_cache.html"
    ),
    "mp": "https://docs.lmcache.ai/mp/index.html",
    "p2p": "https://docs.lmcache.ai/mp/p2p.html",
    "hidden_states": "https://docs.lmcache.ai/non_kv_cache/hidden_states.html",
    "layerwise": "https://docs.lmcache.ai/kv_cache_optimizations/layerwise.html",
    "sdk": "https://docs.lmcache.ai/mp/sdk.html",
}


def capability_report() -> dict[str, Any]:
    """Describe what stock LMCache can and cannot supply to this baseline."""

    return {
        "transport_baseline": {
            "supported": True,
            "scope": "same-model KV sharing between vLLM instances",
            "recommended_mode": "LMCache multiprocess mode",
        },
        "droidspeak_data": {
            "layerwise_kv": {
                "component_exists": True,
                "integration_status": "legacy in-process path only",
            },
            "transition_hidden_state": {
                "component_exists": True,
                "integration_status": (
                    "HiddenStateStore is documented, but the MP KV SDK does not "
                    "document remote arbitrary-layer hidden-state transfer"
                ),
            },
        },
        "missing_runtime_bridge": [
            "model-pair-aware cache namespace and compatibility checks",
            "remote transfer of the selected transition E cache",
            "receiver partial-prefill hook for a contiguous layer interval",
            "assembly of receiver KVs inside the interval with sender KVs outside it",
            "overlap of remote layer loads with receiver recomputation",
        ],
        "claim_boundary": (
            "An LMCache sharing hit proves the transport control only. It does not "
            "prove cross-model DroidSpeak correctness."
        ),
        "docs": DOC_URLS,
    }


def legacy_centralized_config() -> str:
    """Return the exact deprecated centralized example configuration."""

    return "\n".join(
        [
            "chunk_size: 256",
            "local_cpu: true",
            'remote_url: "lm://localhost:65432"',
            'remote_serde: "cachegen"',
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="write the capability report")
    parser.add_argument(
        "--legacy-centralized-yaml",
        type=Path,
        help="write the user-linked deprecated example configuration",
    )
    args = parser.parse_args()
    report = capability_report()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.legacy_centralized_yaml:
        args.legacy_centralized_yaml.parent.mkdir(parents=True, exist_ok=True)
        args.legacy_centralized_yaml.write_text(
            legacy_centralized_config(), encoding="utf-8"
        )
    if not args.json and not args.legacy_centralized_yaml:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
