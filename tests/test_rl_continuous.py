"""The flight actor samples bounded actions with consistent log probability."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl.continuous import (
    SquashedGaussianAgent,
    action_bounds,
    load_continuous_state_dict,
)


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


def test_observation_normalization_is_frozen_in_state_dict():
    agent = SquashedGaussianAgent(3, (-1.0,), (1.0,), hidden=8)
    agent.set_observation_normalization((10.0, -2.0, 5.0), (2.0, 4.0, 0.5))
    observation = torch.tensor([[12.0, 2.0, 4.5]])
    expected_input = torch.tensor([[1.0, 1.0, -1.0]])
    expected = torch.tanh(agent.actor_mean(expected_input))
    assert torch.allclose(agent.act_deterministic(observation), expected)

    restored = SquashedGaussianAgent(3, (-1.0,), (1.0,), hidden=8)
    restored.load_state_dict(agent.state_dict())
    assert torch.equal(restored.observation_mean, agent.observation_mean)
    assert torch.equal(restored.observation_scale, agent.observation_scale)


def test_observation_normalization_rejects_bad_shapes_and_scales():
    agent = SquashedGaussianAgent(2, (-1.0,), (1.0,), hidden=8)
    with pytest.raises(ValueError, match="mean"):
        agent.set_observation_normalization((0.0,), (1.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        agent.set_observation_normalization((0.0, 0.0), (1.0, 0.0))


def test_pre_normalizer_checkpoint_loads_with_identity_normalization():
    original = SquashedGaussianAgent(2, (-1.0,), (1.0,), hidden=8)
    legacy = {
        key: value
        for key, value in original.state_dict().items()
        if not key.startswith("observation_")
    }
    restored = SquashedGaussianAgent(2, (-1.0,), (1.0,), hidden=8)
    load_continuous_state_dict(restored, legacy)
    assert restored.observation_mean.tolist() == [0.0, 0.0]
    assert restored.observation_scale.tolist() == [1.0, 1.0]


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
