# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-critic actor-critic for HoST-style standup training.

Each critic independently estimates value for a different reward group
(task, style, regularization, target).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP


class ActorCriticMultiCritic(ActorCritic):
    """Actor-critic with N independent critic networks.

    The actor is shared (single policy). Each critic estimates the value
    for one reward group. evaluate() returns [N, num_critics].
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        num_critics: int = 4,
        **kwargs: dict[str, Any],
    ) -> None:
        # Pop num_critics before passing to parent (parent doesn't know about it)
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        self.num_critics = num_critics

        # Compute critic obs dimension (same as parent)
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            num_critic_obs += obs[obs_group].shape[-1]

        critic_hidden_dims = kwargs.get("critic_hidden_dims", [256, 256, 256])
        activation = kwargs.get("activation", "elu")

        # Replace single critic with N independent critics
        del self.critic
        self.critics = nn.ModuleList([
            MLP(num_critic_obs, 1, critic_hidden_dims, activation)
            for _ in range(num_critics)
        ])
        print(f"Multi-Critic: {num_critics} independent critics, obs_dim={num_critic_obs}")

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Evaluate all critics. Returns [N, num_critics]."""
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        values = torch.cat([c(critic_obs) for c in self.critics], dim=-1)
        return values
