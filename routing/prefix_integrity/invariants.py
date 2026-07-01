# SPDX-License-Identifier: GPL-2.0
"""Hard gates: turn observed facts about a candidate into PASS/WARN/FAIL, a
plain-language classification, and a danger score.

The central rule this file encodes: an algorithm can preserve next-token
accuracy on one prompt and still be unsafe for prefix sharing, because two
requests with the same prefix no longer agree on what the prefix object is.
Accuracy is necessary but not sufficient; the cache contract is the gate.

Inputs are the observed facts (geometry, determinism, query-dependence, block
survival, optional semantic drift) plus what the algorithm *declares* about
itself (its intended policy and which fields it puts in its cache key). The
verdict is the collision between claim and observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .datatypes import Classification, Status


@dataclass
class Thresholds:
    pre_min: float = 0.70  # WARN if PRE below this
    cr_cv_max: float = 0.30  # WARN if compression-ratio CV above this
    read_amp_max: float = 2.0  # WARN if read ranges per block above this
    kl_max: float = 0.05  # semantic drift tolerance (next-token KL)
    top1_min: float = 0.90  # top-1 agreement tolerance


@dataclass
class InvariantInput:
    """Observed facts + declared intent for one candidate-vs-cartridge run."""

    # declared intent
    policy: str = "prefix_cache"  # prefix_cache | routing | offload_codec
    cache_key_fields: tuple = ("prefix_hash",)
    has_custom_restore_path: bool = False  # custom connector for variable shape

    # observed geometry / reloadability
    shape_preserved: bool = True
    reloadable: bool = True

    # observed determinism (same query + same config, repeated)
    artifact_stability_same_query: int = 1  # distinct digests; 1 == deterministic
    # observed query dependence (different suffixes, same prefix_hash)
    artifact_stability_across_queries: int = 1
    manifest_stability_across_queries: int = 1

    # observed block survival
    anchor_survival: float = 1.0
    partial_block_rate: float = 0.0
    pre: float = 1.0

    # observed storage
    cr_cv: float = 0.0
    read_amplification: float = 1.0

    # optional semantic drift (None => not measured, e.g. CPU-only run)
    kl: Optional[float] = None
    top1_agreement: Optional[float] = None
    drift_repairable: bool = False  # xa25 can repair the drift

    thresholds: Thresholds = field(default_factory=Thresholds)


def _query_dependent(inp: InvariantInput) -> bool:
    return (
        inp.artifact_stability_across_queries > 1
        or inp.manifest_stability_across_queries > 1
    )


def _query_hash_in_key(inp: InvariantInput) -> bool:
    keys = tuple(inp.cache_key_fields)
    return any(k in ("query_hash", "query", "suffix_hash") for k in keys)


def evaluate(inp: InvariantInput) -> dict:
    """Return {status, classification, danger_score, violations, warnings,
    recommendations}. Pure function of the observed/declared facts.
    """
    t = inp.thresholds
    violations: list = []
    warnings: list = []
    recs: list = []

    claims_prefix_safe = inp.policy == "prefix_cache"
    query_dep = _query_dependent(inp)
    has_qhash = _query_hash_in_key(inp)

    # ---- FAIL gates -------------------------------------------------
    if not inp.shape_preserved and not inp.has_custom_restore_path:
        violations.append("shape/layout mismatch with no declared custom restore path")
    if not inp.reloadable:
        violations.append("candidate cannot be reloaded through the connector")
    if inp.artifact_stability_same_query > 1:
        violations.append(
            f"non-deterministic: same prefix_hash+config produced "
            f"{inp.artifact_stability_same_query} distinct artifacts"
        )
    if query_dep and not has_qhash and claims_prefix_safe:
        violations.append(
            "query-dependent content stored under prefix_hash without "
            "query_hash in the cache key (prefix-cache unsafe)"
        )
    if claims_prefix_safe and inp.anchor_survival < 1.0:
        violations.append(
            "anchor/system-prompt block missing or partial under a policy "
            "that claims prefix-cache safety"
        )
    if (
        claims_prefix_safe
        and inp.partial_block_rate > 0.0
        and not inp.has_custom_restore_path
    ):
        violations.append(
            f"partial_block_rate={inp.partial_block_rate:.3f} > 0 on a vanilla "
            "block-based prefix-cache path"
        )

    # ---- WARN gates -------------------------------------------------
    if inp.pre < t.pre_min:
        warnings.append(f"PRE {inp.pre:.3f} below threshold {t.pre_min}")
    if inp.cr_cv > t.cr_cv_max:
        warnings.append(
            f"compression-ratio CV {inp.cr_cv:.3f} above {t.cr_cv_max} "
            "(ratio swings request-to-request)"
        )
    if inp.read_amplification > t.read_amp_max:
        warnings.append(
            f"storage fragmentation: {inp.read_amplification:.2f} read ranges "
            f"per block above {t.read_amp_max}"
        )
    if inp.kl is not None and inp.kl > t.kl_max:
        if inp.drift_repairable:
            warnings.append(
                f"semantic drift KL {inp.kl:.4f} above {t.kl_max} but xa25 "
                "repairs it"
            )
        else:
            violations.append(
                f"semantic drift KL {inp.kl:.4f} above {t.kl_max}, not repairable"
            )
    if inp.top1_agreement is not None and inp.top1_agreement < t.top1_min:
        warnings.append(f"top-1 agreement {inp.top1_agreement:.3f} below {t.top1_min}")
    if query_dep and not has_qhash and not claims_prefix_safe:
        warnings.append(
            "query-aware selector evaluated as if prefix-only reusable; treat "
            "as routing, not prefix cache"
        )

    # ---- status -----------------------------------------------------
    if violations:
        status = Status.FAIL
    elif warnings:
        status = Status.WARN
    else:
        status = Status.PASS

    # ---- classification ---------------------------------------------
    classification = _classify(inp, query_dep, has_qhash, recs)

    danger = _danger_score(inp, violations, warnings, query_dep, has_qhash)

    return {
        "status": status.value,
        "classification": classification.value,
        "danger_score": round(danger, 4),
        "violations": violations,
        "warnings": warnings,
        "recommendations": recs,
    }


def _classify(inp, query_dep, has_qhash, recs) -> Classification:
    """Pick the single most-severe applicable verdict. A method can trip more
    than one condition (SnapKV is both query-dependent and partial-block); the
    verdict is the worst one, since that is the binding constraint on where it
    can deploy. Severity, high to low: DANGEROUS > ROUTING_ONLY >
    {CUSTOM_CONNECTOR, EXTENDED_CACHE_KEY} > SAFE.
    """
    claims_prefix_safe = inp.policy == "prefix_cache"

    # DANGEROUS: the prefix can map to a different or non-reloadable object.
    if (not inp.reloadable) or (
        not inp.shape_preserved and not inp.has_custom_restore_path
    ):
        recs.append("declare a custom restore path or do not store by prefix_hash")
        return Classification.DANGEROUS_FOR_PREFIX_SHARING
    if inp.artifact_stability_same_query > 1:
        recs.append("make artifact generation deterministic (seed/sort)")
        return Classification.DANGEROUS_FOR_PREFIX_SHARING
    # Query-dependent content stored under prefix_hash alone is the fatal case,
    # checked before partial-block so a method that is both is called dangerous.
    if query_dep and not has_qhash and claims_prefix_safe:
        recs.append("add query_hash to the cache key or relabel as routing-only")
        return Classification.DANGEROUS_FOR_PREFIX_SHARING

    # ROUTING_ONLY: query-dependent but honestly labeled as routing, not cache.
    if query_dep and not has_qhash and not claims_prefix_safe:
        recs.append("valid as a query-aware routing prior, not as prefix cache")
        return Classification.ROUTING_ONLY_NOT_PREFIX_CACHE_SAFE

    # SAFE-ONLY tier: reloadable and prefix-key-safe, but needs one accommodation.
    if inp.partial_block_rate > 0.0:
        recs.append("partial blocks need a custom connector or whole-block policy")
        return Classification.SAFE_ONLY_WITH_CUSTOM_CONNECTOR
    if not inp.shape_preserved:  # reloadable only via the declared restore path
        return Classification.SAFE_ONLY_WITH_CUSTOM_CONNECTOR
    if query_dep and has_qhash:
        recs.append("keep query_hash in the cache key; do not key by prefix_hash alone")
        return Classification.SAFE_ONLY_WITH_EXTENDED_CACHE_KEY

    # Deterministic, shape-preserving, query-independent, whole-block.
    return Classification.SAFE_FOR_PREFIX_OFFLOAD


def _danger_score(inp, violations, warnings, query_dep, has_qhash) -> float:
    """0 (safe) .. 1 (dangerous). Hard violations dominate; the query-dependent
    -without-query_hash case is weighted heavily because it is the silent one.
    Semantic drift contributes by magnitude so two shape-preserving codecs that
    both FAIL are still ranked -- a codec that moves the next-token distribution
    far (fp8 on a fragile-key model) scores higher than one that barely moves it
    (near-lossless int8).
    """
    score = 0.0
    score += min(1.0, 0.35 * len(violations))
    if query_dep and not has_qhash and inp.policy == "prefix_cache":
        score += 0.4
    if inp.partial_block_rate > 0 and not inp.has_custom_restore_path:
        score += 0.2
    if not inp.shape_preserved and not inp.has_custom_restore_path:
        score += 0.3
    score += 0.05 * len(warnings)
    score += 0.3 * (1.0 - max(0.0, min(1.0, inp.pre)))
    # Drift severity: KL saturating at ~2 nats, plus the share of queries whose
    # top-1 token flipped. Only contributes when drift was actually measured.
    if inp.kl is not None:
        score += 0.3 * min(1.0, inp.kl / 2.0)
    if inp.top1_agreement is not None:
        score += 0.2 * (1.0 - max(0.0, min(1.0, inp.top1_agreement)))
    return max(0.0, min(1.0, score))
