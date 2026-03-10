# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for AMP environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class modify_reward_weight_linear(ManagerTermBase):
    """Linearly interpolate a reward term's weight between two values.

    Stays at ``weight_initial`` until ``decay_start_steps``, then linearly
    transitions to ``weight_final`` by ``decay_end_steps``, and remains there.

    Args:
        term_name: Name of the reward term to modify.
        weight_initial: Starting weight (used before decay_start_steps).
        weight_final: Ending weight (used after decay_end_steps).
        decay_start_steps: common_step_counter value when decay begins.
        decay_end_steps: common_step_counter value when decay completes.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        weight_initial: float,
        weight_final: float,
        decay_start_steps: int,
        decay_end_steps: int,
    ) -> float:
        step = env.common_step_counter
        if step <= decay_start_steps:
            weight = weight_initial
        elif step >= decay_end_steps:
            weight = weight_final
        else:
            t = (step - decay_start_steps) / (decay_end_steps - decay_start_steps)
            weight = weight_initial + t * (weight_final - weight_initial)

        self._term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, self._term_cfg)
        return weight
