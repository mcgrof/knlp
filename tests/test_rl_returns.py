"""Return handling distinguishes a time limit from a terminal state."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl.ppo import bootstrap_time_limits


class ObservationValue:
    def get_value(self, observations):
        return observations[:, :1]


def test_only_truncations_bootstrap_from_the_final_observation():
    rewards = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    infos = {
        "truncations": np.asarray([False, True, False]),
        "final_observations": (
            None,
            np.asarray([4.0, 99.0], dtype=np.float32),
            np.asarray([8.0, 99.0], dtype=np.float32),
        ),
    }
    adjusted = bootstrap_time_limits(
        rewards, infos, ObservationValue(), torch.device("cpu"), gamma=0.5
    )
    assert adjusted.tolist() == [1.0, 4.0, 3.0]
    assert rewards.tolist() == [1.0, 2.0, 3.0]


def test_missing_final_observation_fails_closed():
    infos = {
        "truncations": np.asarray([True]),
        "final_observations": (None,),
    }
    with pytest.raises(ValueError, match="final observations"):
        bootstrap_time_limits(
            np.asarray([1.0], dtype=np.float32),
            infos,
            ObservationValue(),
            torch.device("cpu"),
            gamma=0.99,
        )
