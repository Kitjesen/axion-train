# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-critic PPO for HoST-style standup training.

Key changes from standard PPO:
- Per-critic GAE: each critic computes returns independently
- Weighted advantage aggregation: A = sum(w_i * A_i) / sum(w_i)
- Per-critic value loss: sum of individual critic value losses
- L2C2 value smoothness: penalize ||V_t - V_{t-1}||² for stable critic training
  Ref: HoST (RSS 2025) value_smoothness_coef=0.1
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO


class PPOMultiCritic(PPO):
    """PPO with multi-critic support and L2C2 value smoothness regularization."""

    def __init__(
        self,
        policy,
        storage,
        critic_weights: list[float] | None = None,
        value_smoothness_coef: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(policy, storage, **kwargs)
        self.num_critics = getattr(policy, "num_critics", 1)
        weights = critic_weights or [1.0] * self.num_critics
        self.critic_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.value_smoothness_coef = value_smoothness_coef
        print(f"PPOMultiCritic: {self.num_critics} critics, weights={weights}, "
              f"value_smoothness_coef={value_smoothness_coef}")

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute per-critic returns via GAE, L2C2 smoothness, then weighted-aggregate advantages."""
        st = self.storage
        # Last values: [N, num_critics]
        last_values = self.policy.evaluate(obs).detach()

        # Vectorized GAE across all critics (element-wise, broadcasting handles [T,N,K])
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()  # [N, 1] broadcasts with [N, K]
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]

        # L2C2: Modify return targets to penalize temporal value inconsistency.
        # Equivalent to adding λ||V_t - V_{t-1}||² to the value loss.
        # Correction: ΔR_t = λ(V_{t-1} - 2V_t + V_{t+1})  (second-order finite difference)
        # This pulls value targets toward their temporal neighbors, stabilizing critic training.
        if self.value_smoothness_coef > 0 and st.num_transitions_per_env > 1:
            vals = st.values  # [T, N, K]
            v_prev = torch.cat([vals[:1], vals[:-1]], dim=0)   # boundary: v_prev[0] = v[0]
            v_next = torch.cat([vals[1:], vals[-1:]], dim=0)   # boundary: v_next[T-1] = v[T-1]
            st.returns += self.value_smoothness_coef * (v_prev - 2 * vals + v_next)

        # Per-critic advantages: [T, N, num_critics]
        per_critic_adv = st.returns - st.values

        # Normalize each critic's advantages independently
        for c in range(self.num_critics):
            adv_c = per_critic_adv[:, :, c]
            per_critic_adv[:, :, c] = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-8)

        # Weighted aggregation -> [T, N, 1]
        w = self.critic_weights.view(1, 1, -1)  # [1, 1, num_critics]
        st.advantages = (per_critic_adv * w).sum(dim=-1, keepdim=True) / w.sum()

        # Final normalization of aggregated advantages
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)
