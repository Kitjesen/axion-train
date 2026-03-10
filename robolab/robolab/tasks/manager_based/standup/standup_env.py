# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom standup environment with force curriculum and grouped rewards.

Phase 2: Force curriculum (upward assist force in world frame)
Phase 3: Per-group reward computation for multi-critic PPO
"""

from __future__ import annotations

import logging
import torch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    from .standup_env_cfg import Atom01StandupEnvCfg

logger = logging.getLogger(__name__)


class StandupEnv(ManagerBasedRLEnv):
    """Standup environment with force curriculum and grouped rewards."""

    cfg: Atom01StandupEnvCfg

    def __init__(self, cfg: Atom01StandupEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Force curriculum: per-env assist force
        fc = cfg.force_curriculum
        self._assist_force = torch.full(
            (self.num_envs,), fc.initial_force, device=self.device
        )

        # Action rescaler
        self._action_scale = cfg.action_rescaler.initial_scale

        # 2nd-order action smoothness buffer (a_{t-2})
        # Used by action_smoothness_l2 reward: (a_t - 2*a_{t-1} + a_{t-2})
        self._last_last_action = torch.zeros(
            self.num_envs, self.action_manager.total_action_dim, device=self.device
        )

        # Step counter for periodic logging
        self._step_count = 0

        # Cache reward term configs for grouped reward computation
        self._reward_terms = {}
        rewards_cfg = cfg.rewards
        for attr_name in dir(rewards_cfg):
            if attr_name.startswith("_"):
                continue
            attr = getattr(rewards_cfg, attr_name, None)
            if attr is not None and hasattr(attr, "func") and hasattr(attr, "weight"):
                self._reward_terms[attr_name] = attr

        logger.info(
            f"[StandupEnv] Force curriculum: initial_force={fc.initial_force}N, "
            f"decay={fc.force_decay}N, action_scale={self._action_scale}"
        )

    def _reset_idx(self, env_ids):
        """Reset environments and restore assist force for reset envs."""
        super()._reset_idx(env_ids)
        # Reset assist force to initial value for reset environments
        self._assist_force[env_ids] = self.cfg.force_curriculum.initial_force
        # Reset 2nd-order action smoothness buffer for reset envs
        self._last_last_action[env_ids] = 0.0

    def step(self, action):
        # Update _last_last_action BEFORE super().step() shifts the action history.
        # At this point: action_manager.prev_action = a_{t-2}
        # After super().step(): action_manager.prev_action = a_{t-1}, action = a_t
        # So reward can compute: a_t - 2*a_{t-1} + a_{t-2} correctly.
        self._last_last_action.copy_(self.action_manager.prev_action)

        # Scale actions
        scaled_action = action * self._action_scale

        # Set assist force on robot base (applied during physics)
        self._apply_assist_force()

        # Normal ManagerBasedRLEnv step
        obs, rew, terminated, truncated, info = super().step(scaled_action)

        # Compute per-group rewards for multi-critic
        if hasattr(self.cfg, "reward_groups") and self.cfg.reward_groups:
            info["reward_groups"] = self._compute_group_rewards()

        # Update force curriculum
        self._update_force_curriculum()

        # Periodic diagnostics
        self._step_count += 1
        if self._step_count % 500 == 0:
            robot = self.scene["robot"]
            h = robot.data.root_pos_w[:, 2]
            f = self._assist_force
            logger.info(
                f"[StandupEnv] step={self._step_count} "
                f"height: mean={h.mean():.3f} max={h.max():.3f} "
                f"force: mean={f.mean():.1f} min={f.min():.1f} max={f.max():.1f}"
            )

        return obs, rew, terminated, truncated, info

    def _apply_assist_force(self):
        """Apply upward assist force on base_link.

        Force is NOT gated by orientation — both lying face-up and standing
        have projected_gravity_z ≈ -1, so orientation gating would zero out
        the force exactly when the robot needs it most (lying face-up trying
        to stand). Instead, the force magnitude is controlled purely by the
        curriculum (_update_force_curriculum).
        """
        robot = self.scene["robot"]
        forces = torch.zeros(self.num_envs, robot.num_bodies, 3, device=self.device)
        torques = torch.zeros_like(forces)

        # Apply constant upward force to base_link (body index 0)
        # MUST use is_global=True so force is always world-Z (upward)
        # regardless of robot orientation (default is body-local frame)
        forces[:, 0, 2] = self._assist_force
        robot.set_external_force_and_torque(forces, torques, is_global=True)

    def _compute_group_rewards(self) -> dict[str, torch.Tensor]:
        """Compute per-group rewards for multi-critic training."""
        dt = self.step_dt
        groups = {}
        for group_name, term_names in self.cfg.reward_groups.items():
            total = torch.zeros(self.num_envs, device=self.device)
            for name in term_names:
                term = self._reward_terms.get(name)
                if term is not None and term.weight != 0.0:
                    value = term.func(self, **term.params)
                    total += value * term.weight * dt
            groups[group_name] = total
        return groups

    def _update_force_curriculum(self):
        """Decrease assist force for envs where robot successfully stands."""
        fc = self.cfg.force_curriculum
        robot = self.scene["robot"]
        height = robot.data.root_pos_w[:, 2]
        gravity_z = robot.data.projected_gravity_b[:, 2]

        # Success: standing upright above threshold
        success = (height > fc.success_height) & (gravity_z < fc.success_gravity_z)
        self._assist_force = torch.where(
            success,
            (self._assist_force - fc.force_decay).clamp(min=fc.min_force),
            self._assist_force,
        )
