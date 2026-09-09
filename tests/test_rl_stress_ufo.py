"""Adversarial combat-envelope checks for standalone UFO actors."""

from rl.stress_ufo import COMBAT_GOAL_AXES, attach_verdict, combat_corner_goals


def stress_report(*, failures=0, tilt=10.0, velocity=11.0, yaw=0.11):
    goals = combat_corner_goals()
    episodes = [
        {
            "ground_contact": index < failures,
            "out_of_envelope": False,
            "maximum_tilt_deg": tilt,
        }
        for index in range(len(goals))
    ]
    return {
        "goals": goals,
        "seeds": [6000],
        "policies": {
            "checkpoint": {
                "summary": {
                    "mean_velocity_rmse_mps": velocity,
                    "mean_yaw_rate_rmse_radps": yaw,
                },
                "episodes": episodes,
            },
            "reference": {
                "summary": {
                    "mean_velocity_rmse_mps": 10.0,
                    "mean_yaw_rate_rmse_radps": 0.1,
                },
                "episodes": [{} for _ in goals],
            },
        },
    }


def test_combat_corners_cover_every_director_limit():
    goals = combat_corner_goals()
    assert len(goals) == 16
    for index, limits in enumerate(COMBAT_GOAL_AXES):
        assert {goal[index] for goal in goals} == set(limits)


def test_stress_verdict_accepts_bounded_actor():
    report = attach_verdict(stress_report())
    assert report["machine_gate_passed"]
    assert all(report["gates"].values())
    assert report["observed"]["episodes"] == 16


def test_stress_verdict_rejects_failure_and_tracking_excess():
    report = attach_verdict(
        stress_report(failures=1, tilt=20.1, velocity=12.6, yaw=0.151)
    )
    assert not report["machine_gate_passed"]
    assert not report["gates"]["no_terminal_failures"]
    assert not report["gates"]["tilt"]
    assert not report["gates"]["velocity_tracking"]
    assert not report["gates"]["yaw_tracking"]


def test_stress_verdict_rejects_incomplete_corner_coverage():
    source = stress_report()
    source["policies"]["checkpoint"]["episodes"].pop()
    report = attach_verdict(source)
    assert not report["gates"]["coverage"]
