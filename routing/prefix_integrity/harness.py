# SPDX-License-Identifier: GPL-2.0
"""The harness: replay a candidate against one cartridge and grade it.

For a cartridge C standing in for a long shared prefix, run the candidate
selector across many suffix/query requests that all share C's prefix_hash, and
repeat each request to probe determinism. From that replay it measures block
survival, manifest/artifact stability (determinism vs query-dependence), and
storage geometry, then asks the invariants for a verdict.

This is the selector-mode MVP (MVP 1, 2, 4): pure CPU, no model forward. Codec
semantic drift (MVP 3) is a separate GPU step (semantic_drift.py) whose result
feeds the same invariant gate via kl / top1_agreement.
"""

from __future__ import annotations

from typing import Optional

from . import metrics as M
from .adapters import load_adapter
from .cartridge_view import dtype_bytes, load_identity
from .datatypes import BlockManifest, PrefixIntegrityResult
from .invariants import InvariantInput, Thresholds, evaluate


def _adapter_decl(adapter):
    return (
        getattr(adapter, "policy", "prefix_cache"),
        tuple(getattr(adapter, "cache_key_fields", ("prefix_hash",))),
        bool(getattr(adapter, "has_custom_restore_path", False)),
        bool(getattr(adapter, "shape_preserved", True)),
    )


def _manifest(adapter, request, num_blocks, budget_k, block_size):
    """Get a BlockManifest from an adapter. Token-level / per-head methods
    override manifest() to return partial blocks via from_token_mask; plain
    block selectors fall back to select_blocks -> whole-block manifest.
    """
    fn = getattr(adapter, "manifest", None)
    if callable(fn):
        return fn(request, num_blocks, budget_k, block_size)
    sel = adapter.select_blocks(request, num_blocks, budget_k)
    return BlockManifest.from_selected(num_blocks, sel, block_size=block_size)


def _digest(man: BlockManifest) -> str:
    """Content digest of a manifest: the kept blocks plus which are partial.
    Two manifests that keep the same blocks but differ in partial pattern hash
    differently, so a token-level method that shifts its cut is seen as a new
    stored object.
    """
    import hashlib

    sig = ",".join(f"{b}:{man.status[b].value[0]}" for b in sorted(set(man.selected)))
    return hashlib.sha256(sig.encode()).hexdigest()[:16]


def run_validate(
    cartridge: str,
    adapter_spec: str,
    queries: list,
    budget_k: int,
    block_size: int = 16,
    pins: Optional[str] = None,
    repeats: int = 3,
    adapter_config: Optional[dict] = None,
    thresholds: Optional[Thresholds] = None,
    semantic: Optional[dict] = None,
) -> dict:
    """Run the selector-mode integrity suite. `queries` is a list of dicts with
    at least an 'id' or 'query'. `semantic` optionally carries {kl, top1,
    repairable} from a GPU drift run. Returns a JSON-able dict.
    """
    from .adapters import parse_pins

    identity = load_identity(cartridge, block_size=block_size)
    num_blocks = identity.num_blocks
    adapter = load_adapter(adapter_spec, adapter_config)
    policy, cache_key_fields, custom_restore, shape_preserved = _adapter_decl(adapter)

    pin = parse_pins(pins) or (1, 2, 0)
    anchor_blocks, recent_blocks = pin[0], pin[1]

    if not queries:
        queries = [{"id": "q0", "query": "default"}]

    # Replay: per query, repeat to expose non-determinism; collect across queries.
    per_query_manifests = []
    per_query_selected = []
    per_query_digests = []
    same_query_digest_counts = []
    for q in queries:
        digs = set()
        man0 = None
        for _ in range(max(1, repeats)):
            man = _manifest(adapter, q, num_blocks, budget_k, block_size)
            digs.add(_digest(man))
            if man0 is None:
                man0 = man
        per_query_manifests.append(man0)
        per_query_selected.append(sorted(set(man0.selected)))
        per_query_digests.append(sorted(digs)[0])
        same_query_digest_counts.append(len(digs))

    manifest_stab = M.manifest_stability(per_query_selected)
    artifact_stab = M.artifact_stability(per_query_digests)
    same_query_stab = max(same_query_digest_counts)

    # Representative manifest = first query.
    rep = per_query_manifests[0]
    mm = M.manifest_metrics(
        rep, anchor_blocks=anchor_blocks, recent_blocks=recent_blocks
    )

    # PRE / anchor / partial averaged across queries (query-aware methods vary).
    pres, anchors, partials = [], [], []
    for man in per_query_manifests:
        pres.append(M.prefix_reuse_efficiency(man))
        anchors.append(M.anchor_survival(man, anchor_blocks))
        partials.append(M.partial_block_rate(man))
    pre_mean = sum(pres) / len(pres)
    anchor_mean = sum(anchors) / len(anchors)
    partial_mean = sum(partials) / len(partials)

    # Storage: compression ratio = blocks kept fraction, per query.
    cr_values = [num_blocks / max(1, len(sel)) for sel in per_query_selected]
    cr_stats = M.compression_stats(cr_values)
    read_amp = M.read_amplification(per_query_selected[0])

    inp = InvariantInput(
        policy=policy,
        cache_key_fields=cache_key_fields,
        has_custom_restore_path=custom_restore,
        shape_preserved=shape_preserved,
        reloadable=True,
        artifact_stability_same_query=same_query_stab,
        artifact_stability_across_queries=artifact_stab,
        manifest_stability_across_queries=manifest_stab,
        anchor_survival=anchor_mean,
        partial_block_rate=partial_mean,
        pre=pre_mean,
        cr_cv=cr_stats["cr_cv"],
        read_amplification=read_amp,
        kl=(semantic or {}).get("kl"),
        top1_agreement=(semantic or {}).get("top1"),
        drift_repairable=(semantic or {}).get("repairable", False),
        thresholds=thresholds or Thresholds(),
    )
    verdict = evaluate(inp)

    result = PrefixIntegrityResult(
        status=verdict["status"],
        classification=verdict["classification"],
        danger_score=verdict["danger_score"],
        metrics={
            **mm,
            "pre_mean": round(pre_mean, 4),
            "anchor_survival_mean": round(anchor_mean, 4),
            "manifest_stability": manifest_stab,
            "artifact_stability": artifact_stab,
            "same_query_artifact_count": same_query_stab,
            "cr_mean": round(cr_stats["cr_mean"], 4),
            "cr_cv": round(cr_stats["cr_cv"], 4),
            "n_queries": len(queries),
            "budget_k": budget_k,
            **(
                {
                    "kl": round(semantic["kl"], 4),
                    "top1": round(semantic.get("top1", 0.0), 4),
                }
                if semantic and semantic.get("kl") is not None
                else {}
            ),
        },
        violations=verdict["violations"],
        warnings=verdict["warnings"],
        recommendations=verdict["recommendations"],
    )

    return {
        "identity": {
            "model_id": identity.model_id,
            "cartridge_id": identity.cartridge_id,
            "prefix_hash": identity.prefix_hash,
            "block_size": identity.block_size,
            "num_blocks": identity.num_blocks,
            "dtype": identity.dtype,
            "kv_shape": list(identity.kv_shape),
            "kv_bytes": _kv_bytes(identity),
            "rope_config_hash": identity.rope_config_hash,
        },
        "algorithm": adapter.name,
        "policy": policy,
        "cache_key_fields": list(cache_key_fields),
        "status": result.status,
        "classification": result.classification,
        "danger_score": result.danger_score,
        "metrics": result.metrics,
        "violations": result.violations,
        "warnings": result.warnings,
        "recommendations": result.recommendations,
        "block_survival": M.block_survival_rows(rep),
        "per_query_selected": per_query_selected,
    }


def _kv_bytes(identity) -> int:
    elems = 2
    for d in identity.kv_shape:
        if d is None:
            return 0
        elems *= int(d)
    return elems * dtype_bytes(identity.dtype)
