# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout storage for multi-critic PPO.

Extends RolloutStorage with per-critic shapes:
- values:  [T, N, num_critics]
- rewards: [T, N, num_critics]
- returns: [T, N, num_critics]
- advantages: [T, N, 1]  (weighted aggregated for policy update)
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.storage import RolloutStorage


class MultiCriticRolloutStorage(RolloutStorage):
    """Rollout storage supporting multiple critics."""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
        num_critics: int = 4,
    ) -> None:
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        self.num_critics = num_critics

        # Override with multi-critic shapes
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=device)
            self.rewards = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=device)
            # Advantages are weighted-aggregated to [T, N, 1] for the surrogate loss
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        """Add a transition with multi-critic values/rewards."""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! Call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # Multi-critic rewards: [N, num_critics] stored directly (no view(-1, 1))
        self.rewards[self.step].copy_(transition.rewards)

        # RL-specific
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)  # [N, num_critics]
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # RNN hidden states
        self._save_hidden_states(transition.hidden_states)

        self.step += 1
