"""Offline interval profiling from the DroidSpeak paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, order=True)
class RecomputeInterval:
    """A half-open, contiguous receiver-layer interval."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid recompute interval")

    @property
    def layer_count(self) -> int:
        return self.end - self.start

    def contains(self, layer_idx: int) -> bool:
        return self.start <= layer_idx < self.end

    def as_dict(self) -> dict[str, int]:
        return {
            "start": self.start,
            "end": self.end,
            "layer_count": self.layer_count,
        }


@dataclass(frozen=True)
class ProfilePoint:
    """Mean quality and latency for one recomputation interval."""

    interval: RecomputeInterval
    quality: float
    latency_s: float


def enumerate_intervals(
    num_layers: int,
    group_size: int = 2,
    include_full_reuse: bool = True,
) -> list[RecomputeInterval]:
    """Enumerate all contiguous intervals on grouped layer boundaries.

    The paper profiles O(L^2) contiguous intervals. Its evaluation groups two
    adjacent layers to reduce the search cost. A final short group is allowed
    when the layer count is not divisible by the group size.
    """

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    boundaries = list(range(0, num_layers, group_size))
    if boundaries[-1] != num_layers:
        boundaries.append(num_layers)

    intervals: list[RecomputeInterval] = []
    if include_full_reuse:
        intervals.append(RecomputeInterval(0, 0))
    for start_pos, start in enumerate(boundaries[:-1]):
        for end in boundaries[start_pos + 1 :]:
            intervals.append(RecomputeInterval(start, end))
    return intervals


def pareto_frontier(points: Iterable[ProfilePoint]) -> list[ProfilePoint]:
    """Return points not dominated on fewer layers and higher quality."""

    ordered = sorted(
        points,
        key=lambda point: (
            point.interval.layer_count,
            -point.quality,
            point.latency_s,
            point.interval,
        ),
    )
    frontier: list[ProfilePoint] = []
    best_quality = float("-inf")
    for point in ordered:
        if point.quality > best_quality:
            frontier.append(point)
            best_quality = point.quality
    return frontier


def quality_floor(full_prefill_quality: float, max_relative_drop: float) -> float:
    """Compute the paper's receiver-relative quality floor."""

    if full_prefill_quality < 0:
        raise ValueError("full_prefill_quality must be non-negative")
    if not 0 <= max_relative_drop < 1:
        raise ValueError("max_relative_drop must be in [0, 1)")
    return full_prefill_quality * (1.0 - max_relative_drop)


def select_interval(
    points: Iterable[ProfilePoint],
    full_prefill_quality: float,
    max_relative_drop: float = 0.05,
) -> ProfilePoint:
    """Select the fewest-layer frontier point inside the quality budget."""

    frontier = pareto_frontier(points)
    if not frontier:
        raise ValueError("at least one profile point is required")
    floor = quality_floor(full_prefill_quality, max_relative_drop)
    eligible = [point for point in frontier if point.quality >= floor]
    if not eligible:
        raise ValueError("no recomputation interval meets the quality floor")
    return min(
        eligible,
        key=lambda point: (
            point.interval.layer_count,
            point.latency_s,
            point.interval.start,
        ),
    )
