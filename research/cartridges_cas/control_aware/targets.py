"""Versioned target transforms for control-aware cartridge distillation.

The stored self-study parquet serializes each teacher target row as
[sampled token] + [top-k tokens].  Under greedy synthesis the sampled
token is the top-1 token, so 45% of rows carry it twice with exactly
equal log probability, and the legacy trainer consumes both copies.
The legacy per-example loss therefore decomposes exactly as

    L_legacy = L_unique + L_anchor

where L_unique sums each stored probability once over the unique row
support and L_anchor adds the duplicated sampled-token term on exactly
the duplicated rows.  This module materializes that decomposition as
entry-list transforms so one loss function serves every arm of the
screen: the loss is always sum(p * student_nll) over an entry list,
divided by the ORIGINAL serialized entry count of the example (the
denominator is deliberately never renormalized).

Transforms (TARGET_SCHEMA_VERSION guards the semantics):

    legacy_raw               serialized entries exactly as stored
    dedup_legacy_support     unique support only (drops the anchor term)
    legacy_grouped_replay    unique support + every duplicate anchor,
                             algebraically identical to legacy_raw
    control_anchor           unique support + anchors only on the union
                             of the first-answer-token row and the
                             natural end-of-turn row
    content_anchor_matched   unique support + anchors on non-control
                             duplicated rows, deterministically selected
                             and globally scaled to match the control
                             arm's schedule-wide anchor count and
                             coefficient mass
    dedup_scale_matched      unique support with probabilities scaled by
                             one calibrated scalar matching the legacy
                             schedule-wide coefficient mass

A note on what dedup_scale_matched can and cannot show.  It multiplies
the loss by one global scalar, which AdamW mostly — but, measured, not
entirely — absorbs.  The scalar cancels between the first and second
moments wherever the gradient is large against the optimizer's epsilon,
leaving the update unchanged there; on coordinates whose gradient falls
near epsilon the update is instead close to linear in the scale, and a
trainable KV cache has many such coordinates because a short answer
leaves most cache positions barely involved.

Measured on this screen (scale 1.308, no step clipped): the arm's
displacement from the shared start stays nearly parallel to the
deduplicated arm's — cosine 0.994 at one step, 0.955 at ten — while
running about 3% longer, so the two diverge by 16% of their own
displacement at one step and 31% by ten.  So roughly a tenth of the
loss scale survives the optimizer, as an effective step-size
difference rather than as a different objective.

Read the arm accordingly.  It is a weak control: it can show whether a
few percent of extra step size matters, not whether the full 31% loss
scale does, because the optimizer absorbs most of that before it
reaches the weights.  The legacy-versus-unique comparison is unaffected
either way, because the anchor changes the gradient direction per
coordinate rather than only its magnitude.

Anchors always use each row's own stored probability, never a corpus
mean.  A row that is both first-answer-token and end-of-turn receives
its anchor once.

The canonical_row constructor at the bottom is the forward-looking fix
for the synthesis-side serializer (chosen token kept separately,
dedup before cumulative-mass truncation, retain-all fallback when the
threshold is never reached); it is exercised by tests and intended for
future synthesis, not for rewriting the historical parquet.
"""

import hashlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

TARGET_SCHEMA_VERSION = "control-aware-v1"

MODES = (
    "legacy_raw",
    "dedup_legacy_support",
    "legacy_grouped_replay",
    "control_anchor",
    "content_anchor_matched",
    "dedup_scale_matched",
)


@dataclass
class Row:
    """One target row in serialization order: entry 0 is the sampled
    (chosen) token under the [sampled]+[top-k] layout."""

    idx: int
    ids: List[int]
    logprobs: List[float]

    @property
    def chosen_id(self) -> int:
        return self.ids[0]

    @property
    def chosen_logprob(self) -> float:
        return self.logprobs[0]

    def duplicate_positions(self) -> List[int]:
        """Positions (>0) whose id repeats an earlier position's id."""
        seen = {}
        dups = []
        for j, tid in enumerate(self.ids):
            if tid in seen:
                if abs(self.logprobs[j] - self.logprobs[seen[tid]]) > 1e-6:
                    raise ValueError(
                        f"row {self.idx}: conflicting duplicate logprobs for "
                        f"token {tid}: {self.logprobs[seen[tid]]} vs "
                        f"{self.logprobs[j]}"
                    )
                dups.append(j)
            else:
                seen[tid] = j
        return dups

    @property
    def is_duplicated(self) -> bool:
        return len(self.duplicate_positions()) > 0

    def unique_entries(self) -> Tuple[List[int], List[float]]:
        """First occurrence of each id, in serialization order."""
        seen = set()
        ids, lps = [], []
        for tid, lp in zip(self.ids, self.logprobs):
            if tid in seen:
                continue
            seen.add(tid)
            ids.append(tid)
            lps.append(lp)
        return ids, lps

    def anchor(self) -> Optional[Tuple[int, float]]:
        """The duplicated chosen-token term, if this row is duplicated.
        Returns (token_id, logprob); duplication beyond one extra copy
        is applied per extra copy by the caller (not observed in the
        audited corpus, where every duplicate is exactly one copy)."""
        dups = self.duplicate_positions()
        if not dups:
            return None
        tid = self.ids[dups[0]]
        if tid != self.chosen_id:
            raise ValueError(
                f"row {self.idx}: duplicated token {tid} is not the "
                f"chosen token {self.chosen_id}"
            )
        return tid, self.logprobs[dups[0]]


@dataclass
class ElementTargets:
    """Parsed rows plus element-level control-position facts."""

    rows: List[Row]
    n_serialized: int
    first_row_idx: int
    eot_row_idx: Optional[int]

    def control_rows(self) -> List[int]:
        out = [self.first_row_idx]
        if self.eot_row_idx is not None and self.eot_row_idx != self.first_row_idx:
            out.append(self.eot_row_idx)
        return out


def parse_element(idxs, ids, logprobs, eot_token_id: int) -> ElementTargets:
    """Group flat serialized entries into rows, preserving within-row
    serialization order.  Entries of one row must be contiguous."""
    idxs = [int(x) for x in idxs]
    ids = [int(x) for x in ids]
    logprobs = [float(x) for x in logprobs]
    rows = []
    seen_row_ids = set()
    cur = None
    for i, ri in enumerate(idxs):
        if cur is not None and ri == cur.idx:
            cur.ids.append(ids[i])
            cur.logprobs.append(logprobs[i])
            continue
        if ri in seen_row_ids:
            raise ValueError(f"row {ri} appears in two non-contiguous runs")
        seen_row_ids.add(ri)
        cur = Row(idx=ri, ids=[ids[i]], logprobs=[logprobs[i]])
        rows.append(cur)
    if not rows:
        raise ValueError("element has no target rows")
    for r in rows:
        r.duplicate_positions()  # fail loudly on conflicting duplicates
    first_row = min(r.idx for r in rows)
    last = max(rows, key=lambda r: r.idx)
    eot_row = last.idx if last.chosen_id == eot_token_id else None
    return ElementTargets(
        rows=rows,
        n_serialized=len(idxs),
        first_row_idx=first_row,
        eot_row_idx=eot_row,
    )


@dataclass
class TargetSet:
    """Entry list an arm's loss consumes: loss = sum(p*nll)/denom."""

    row_idxs: List[int]
    token_ids: List[int]
    probs: List[float]
    denom: int

    def tensors(self, device=None):
        t = lambda x, dt: torch.tensor(x, dtype=dt, device=device)
        return (
            t(self.row_idxs, torch.long),
            t(self.token_ids, torch.long),
            t(self.probs, torch.float32),
        )

    def coefficient_mass(self) -> float:
        return sum(self.probs) / self.denom


def _unique_set(et: ElementTargets):
    row_idxs, token_ids, probs = [], [], []
    for r in et.rows:
        uids, ulps = r.unique_entries()
        for tid, lp in zip(uids, ulps):
            row_idxs.append(r.idx)
            token_ids.append(tid)
            probs.append(float(torch.tensor(lp).exp()))
    return row_idxs, token_ids, probs


def _anchors(et: ElementTargets, only_rows: Optional[Sequence[int]] = None):
    """(row, token, p) anchor entries for duplicated rows, optionally
    restricted to a row-index set."""
    allow = set(only_rows) if only_rows is not None else None
    out = []
    for r in et.rows:
        if allow is not None and r.idx not in allow:
            continue
        a = r.anchor()
        if a is None:
            continue
        tid, lp = a
        for _ in r.duplicate_positions():
            out.append((r.idx, tid, float(torch.tensor(lp).exp())))
    return out


def build_target_set(
    et: ElementTargets,
    mode: str,
    scale: float = 1.0,
    content_rows: Optional[Sequence[int]] = None,
    content_scale: float = 1.0,
    denom: Optional[int] = None,
) -> TargetSet:
    """Build the entry list for one arm.  `scale` applies only to
    dedup_scale_matched; `content_rows`/`content_scale` apply only to
    content_anchor_matched and come from calibrate_content_anchors.
    `denom` overrides the serialized entry count with the trainer's
    real denominator (the legacy entry count surviving the position
    and vocabulary validity mask), so calibration weights every
    element exactly as the objective does."""
    assert mode in MODES, mode
    if denom is None:
        denom = et.n_serialized
    if mode == "legacy_raw":
        row_idxs, token_ids, probs = [], [], []
        for r in et.rows:
            for tid, lp in zip(r.ids, r.logprobs):
                row_idxs.append(r.idx)
                token_ids.append(tid)
                probs.append(float(torch.tensor(lp).exp()))
        return TargetSet(row_idxs, token_ids, probs, denom)

    row_idxs, token_ids, probs = _unique_set(et)
    if mode == "dedup_legacy_support":
        return TargetSet(row_idxs, token_ids, probs, denom)
    if mode == "dedup_scale_matched":
        return TargetSet(row_idxs, token_ids, [p * scale for p in probs], denom)
    if mode == "legacy_grouped_replay":
        anchors = _anchors(et)
    elif mode == "control_anchor":
        anchors = _anchors(et, only_rows=et.control_rows())
    elif mode == "content_anchor_matched":
        assert content_rows is not None, "run calibrate_content_anchors first"
        anchors = [
            (ri, tid, p * content_scale)
            for (ri, tid, p) in _anchors(et, only_rows=content_rows)
        ]
    for ri, tid, p in anchors:
        row_idxs.append(ri)
        token_ids.append(tid)
        probs.append(p)
    return TargetSet(row_idxs, token_ids, probs, denom)


def _denoms_for(parsed_elements, denoms):
    if denoms is None:
        return [et.n_serialized for et in parsed_elements]
    assert len(denoms) == len(parsed_elements)
    return list(denoms)


def calibrate_scale(
    parsed_elements: Sequence[ElementTargets],
    denoms: Optional[Sequence[int]] = None,
) -> float:
    """dedup_scale_matched scalar: legacy schedule-wide coefficient mass
    over unique schedule-wide coefficient mass."""
    ds = _denoms_for(parsed_elements, denoms)
    legacy = sum(
        build_target_set(et, "legacy_raw", denom=d).coefficient_mass()
        for et, d in zip(parsed_elements, ds)
    )
    unique = sum(
        build_target_set(et, "dedup_legacy_support", denom=d).coefficient_mass()
        for et, d in zip(parsed_elements, ds)
    )
    return legacy / unique


def calibrate_content_anchors(
    parsed_elements: Sequence[ElementTargets],
    denoms: Optional[Sequence[int]] = None,
):
    """Deterministic content-anchor selection matched to the control
    arm, schedule-wide: per element, select non-control duplicated rows
    in descending stored-probability order (ties by row index) up to
    the element's control-anchor count; then one global scale matches
    the total control-anchor coefficient mass.

    Anchor mass is weighted by 1/denominator, because that is how the
    objective consumes it: each element contributes sum(p*nll)/denom
    and elements are averaged equally, so an element with ten times
    the entries contributes a tenth of the coefficient per anchor.
    Matching raw probability sums instead would leave the two arms
    carrying different anchor mass whenever element sizes differ,
    reintroducing the magnitude confound this arm exists to remove.

    Returns (per_element_rows, content_scale, report)."""
    ds = _denoms_for(parsed_elements, denoms)
    per_element_rows = []
    control_count = 0
    control_mass = 0.0
    selected_mass = 0.0
    selected_count = 0
    for et, d in zip(parsed_elements, ds):
        ctrl = _anchors(et, only_rows=et.control_rows())
        control_count += len(ctrl)
        control_mass += sum(p for _, _, p in ctrl) / d
        ctrl_rows = set(et.control_rows())
        candidates = [
            (r.idx, r.anchor())
            for r in et.rows
            if r.idx not in ctrl_rows and r.is_duplicated
        ]
        candidates.sort(key=lambda c: (-c[1][1], c[0]))
        take = [ri for ri, _ in candidates[: len(ctrl)]]
        sel = _anchors(et, only_rows=take)
        selected_count += len(sel)
        selected_mass += sum(p for _, _, p in sel) / d
        per_element_rows.append(take)
    content_scale = (control_mass / selected_mass) if selected_mass > 0 else 1.0
    report = dict(
        control_count=control_count,
        control_mass=control_mass,
        selected_count=selected_count,
        selected_mass=selected_mass,
        content_scale=content_scale,
    )
    return per_element_rows, content_scale, report


def transform_hash(mode: str, scale: float = 1.0, content_scale: float = 1.0) -> str:
    text = f"{TARGET_SCHEMA_VERSION}|{mode}|{scale:.10g}|{content_scale:.10g}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# forward-looking canonical row constructor (synthesis-side fix)
# ---------------------------------------------------------------------------


def canonical_row(
    chosen_id: int,
    chosen_logprob: float,
    topk_ids: Sequence[int],
    topk_logprobs: Sequence[float],
    mass_threshold: float = 0.998,
    tolerance: float = 1e-4,
):
    """Canonical sparse target row: the chosen token is kept SEPARATELY
    from the distribution; the distribution deduplicates ids before
    cumulative-mass truncation, sorts by probability, retains through
    the first threshold crossing, and retains everything when the
    threshold is never reached.  Returns
    (chosen_id, chosen_logprob, kept_ids, kept_logprobs)."""
    merged = {}
    for tid, lp in zip(topk_ids, topk_logprobs):
        tid = int(tid)
        if tid in merged:
            if abs(merged[tid] - lp) > 1e-6:
                raise ValueError(f"conflicting logprobs for token {tid}")
            continue
        merged[tid] = float(lp)
    if chosen_id not in merged:
        merged[int(chosen_id)] = float(chosen_logprob)
    items = sorted(merged.items(), key=lambda kv: -kv[1])
    probs = torch.tensor([lp for _, lp in items]).exp()
    total = float(probs.sum())
    if total > 1.0 + tolerance:
        raise ValueError(f"unique retained mass {total} exceeds 1+{tolerance}")
    cum = torch.cumsum(probs, 0)
    crossing = (cum >= mass_threshold).nonzero()
    keep = int(crossing[0]) + 1 if crossing.numel() else len(items)
    kept = items[:keep]
    return (
        int(chosen_id),
        float(chosen_logprob),
        [tid for tid, _ in kept],
        [lp for _, lp in kept],
    )
