"""Bounded continuous actor-critic for flight-control environments.

The existing ETR policy remains discrete and checkpoint-compatible.  Flight
controllers use this separate squashed-Gaussian actor, whose sampled and
deterministic actions always respect the environment's finite Box bounds.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def _layer_init(
    layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SquashedGaussianAgent(nn.Module):
    """Actor-critic with a tanh transform into arbitrary finite bounds."""

    _EPSILON = 1e-6

    def __init__(
        self,
        obs_dim: int,
        action_low: Sequence[float],
        action_high: Sequence[float],
        hidden: int = 128,
    ):
        super().__init__()
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        if low.ndim != 1 or high.shape != low.shape or low.numel() == 0:
            raise ValueError("action bounds must be equal, non-empty vectors")
        if not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError("continuous action bounds must be finite")
        if not torch.all(high > low):
            raise ValueError("each upper action bound must exceed its lower bound")
        action_dim = int(low.numel())
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, action_dim), std=0.01),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.register_buffer("action_mid", (high + low) / 2.0)
        self.register_buffer("action_scale", (high - low) / 2.0)

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation)

    def _distribution(self, observation: torch.Tensor) -> Normal:
        mean = self.actor_mean(observation)
        log_std = self.actor_log_std.expand_as(mean).clamp(-20.0, 2.0)
        return Normal(mean, log_std.exp())

    def _to_action(self, unit_action: torch.Tensor) -> torch.Tensor:
        return self.action_mid + self.action_scale * unit_action

    def _to_unit(self, action: torch.Tensor) -> torch.Tensor:
        return ((action - self.action_mid) / self.action_scale).clamp(
            -1.0 + self._EPSILON, 1.0 - self._EPSILON
        )

    def get_action_and_value(
        self,
        observation: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ):
        distribution = self._distribution(observation)
        if action is None:
            raw_action = distribution.rsample()
            unit_action = torch.tanh(raw_action)
            action = self._to_action(unit_action)
        else:
            unit_action = self._to_unit(action)
            raw_action = torch.atanh(unit_action)
        scale_jacobian = self.action_scale * (1.0 - unit_action.square())
        log_prob = distribution.log_prob(raw_action) - torch.log(
            scale_jacobian + self._EPSILON
        )
        # The exact transformed entropy has no simple closed form.  The base
        # Normal entropy is a stable exploration proxy for PPO's entropy term.
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_prob.sum(dim=-1), entropy, self.critic(observation)

    def act_deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        return self._to_action(torch.tanh(self.actor_mean(observation)))


def action_bounds(space) -> tuple[np.ndarray, np.ndarray]:
    """Extract one-dimensional, finite bounds from a Gymnasium Box-like space."""

    low = np.asarray(space.low, dtype=np.float32)
    high = np.asarray(space.high, dtype=np.float32)
    if low.ndim != 1 or high.shape != low.shape or low.size == 0:
        raise ValueError("continuous PPO requires a one-dimensional action space")
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("continuous PPO requires finite action bounds")
    return low, high
