"""The flight actor samples bounded actions with consistent log probability."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl.continuous import SquashedGaussianAgent, action_bounds


def test_sampled_actions_are_bounded_and_recomputable():
    torch.manual_seed(3)
    agent = SquashedGaussianAgent(4, (-2.0, 0.0), (3.0, 1.0), hidden=16)
    observation = torch.randn(8, 4)
    action, log_prob, entropy, value = agent.get_action_and_value(observation)
    _, repeated_log_prob, _, repeated_value = agent.get_action_and_value(
        observation, action
    )
    assert action.shape == (8, 2)
    assert torch.all(action[:, 0] >= -2.0) and torch.all(action[:, 0] <= 3.0)
    assert torch.all(action[:, 1] >= 0.0) and torch.all(action[:, 1] <= 1.0)
    assert torch.allclose(log_prob, repeated_log_prob, atol=2e-4, rtol=2e-4)
    assert torch.equal(value, repeated_value)
    assert log_prob.shape == entropy.shape == (8,)
    assert value.shape == (8, 1)


def test_deterministic_action_respects_asymmetric_bounds():
    agent = SquashedGaussianAgent(3, (-4.0,), (2.0,), hidden=8)
    action = agent.act_deterministic(torch.zeros(5, 3))
    assert action.shape == (5, 1)
    assert torch.all(action >= -4.0) and torch.all(action <= 2.0)


class BoxLike:
    low = np.asarray([-1.0, -2.0], dtype=np.float64)
    high = np.asarray([1.0, 2.0], dtype=np.float64)


def test_action_bounds_normalizes_dtype_and_rejects_unbounded_spaces():
    low, high = action_bounds(BoxLike())
    assert low.dtype == high.dtype == np.float32
    assert low.tolist() == [-1.0, -2.0]

    class Unbounded:
        low = np.asarray([-np.inf], dtype=np.float32)
        high = np.asarray([np.inf], dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        action_bounds(Unbounded())
