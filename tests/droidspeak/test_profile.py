from research.droidspeak.profile import (
    ProfilePoint,
    RecomputeInterval,
    enumerate_intervals,
    pareto_frontier,
    quality_floor,
    select_interval,
)


def test_two_layer_interval_search_includes_controls() -> None:
    intervals = enumerate_intervals(6, group_size=2)

    assert intervals == [
        RecomputeInterval(0, 0),
        RecomputeInterval(0, 2),
        RecomputeInterval(0, 4),
        RecomputeInterval(0, 6),
        RecomputeInterval(2, 4),
        RecomputeInterval(2, 6),
        RecomputeInterval(4, 6),
    ]


def test_final_short_group_is_profiled() -> None:
    assert RecomputeInterval(4, 5) in enumerate_intervals(5, group_size=2)


def test_pareto_frontier_removes_dominated_points() -> None:
    points = [
        ProfilePoint(RecomputeInterval(0, 0), 0.2, 1.0),
        ProfilePoint(RecomputeInterval(0, 2), 0.3, 2.0),
        ProfilePoint(RecomputeInterval(2, 4), 0.25, 1.5),
        ProfilePoint(RecomputeInterval(0, 4), 0.4, 4.0),
    ]

    assert pareto_frontier(points) == [points[0], points[1], points[3]]


def test_selects_fewest_layer_point_within_relative_budget() -> None:
    points = [
        ProfilePoint(RecomputeInterval(0, 0), 0.7, 1.0),
        ProfilePoint(RecomputeInterval(0, 2), 0.96, 3.0),
        ProfilePoint(RecomputeInterval(0, 4), 1.0, 2.0),
    ]

    selected = select_interval(points, 1.0, max_relative_drop=0.05)

    assert selected == points[1]
    assert quality_floor(1.0, 0.05) == 0.95
