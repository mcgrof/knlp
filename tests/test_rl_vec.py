"""The vector wrapper preserves Gymnasium end-of-episode semantics."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from rl.vec import SyncVec


class EndingEnv(gym.Env):
    def __init__(self, *, truncate: bool, continuous: bool = False):
        self.truncate = truncate
        self.continuous = continuous
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (1,), np.float32)
        if continuous:
            self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        else:
            self.action_space = gym.spaces.Discrete(2)
        self.last_action = None
        self.value = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.value = 0
        return np.asarray([0.0], dtype=np.float32), {}

    def step(self, action):
        self.last_action = action
        self.value += 1
        observation = np.asarray([float(self.value)], dtype=np.float32)
        return observation, 1.0, not self.truncate, self.truncate, {"terminal": True}


def test_vec_keeps_terminal_observation_and_end_reason():
    vec = SyncVec(
        [
            lambda: EndingEnv(truncate=False),
            lambda: EndingEnv(truncate=True),
        ]
    )
    vec.reset(seed=9)
    observations, rewards, dones, infos = vec.step(np.asarray([0, 1]))
    assert observations.tolist() == [[0.0], [0.0]]
    assert rewards.tolist() == [1.0, 1.0]
    assert dones.tolist() == [True, True]
    assert infos["terminations"].tolist() == [True, False]
    assert infos["truncations"].tolist() == [False, True]
    assert [item.tolist() for item in infos["final_observations"]] == [
        [1.0],
        [1.0],
    ]
    assert infos["final_infos"] == ({"terminal": True}, {"terminal": True})
    vec.close()


def test_vec_passes_box_actions_without_integer_coercion():
    vec = SyncVec([lambda: EndingEnv(truncate=True, continuous=True)])
    vec.reset()
    action = np.asarray([[0.25, -0.75]], dtype=np.float64)
    vec.step(action)
    assert isinstance(vec.envs[0].last_action, np.ndarray)
    assert vec.envs[0].last_action.dtype == np.float32
    assert vec.envs[0].last_action.tolist() == pytest.approx([0.25, -0.75])
    vec.close()
