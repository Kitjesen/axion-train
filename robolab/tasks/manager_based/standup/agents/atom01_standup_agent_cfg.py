# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent configuration for ATOM01 standup task.

Phase 2+3: Multi-critic PPO with StandupRunner.
4 independent critics (task, style, regu, target) with weighted advantages.

V10: entropy_coef 0.01 → 0.001，保留 L2C2。
Root cause of V9 failure:
  entropy_loss=62.6 × entropy_coef=0.01 = 0.626  >>  surrogate_loss=0.005
  → entropy gradient dominates actor updates → noise_std 3.80 (target ~0.5)
  → joint_limits=-79, action_smoothness=-20
Fix: entropy_coef=0.001 → entropy gradient (0.0626) ≈ surrogate gradient (0.005)
  V8 proved 0.001 works (noise_std=0.47 at iter 486), just lacked L2C2.
  V10 = V8's entropy + V9's L2C2 + V9's critic_weights.
Resume from V9 model_3600.pt (height 0.66m, multi-critic weights already trained).
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlPpoMultiCriticCfg(RslRlPpoActorCriticCfg):
    """Multi-critic actor-critic config."""
    class_name: str = "ActorCriticMultiCritic"
    num_critics: int = 4


@configclass
class RslRlPpoMultiCriticAlgCfg(RslRlPpoAlgorithmCfg):
    """Multi-critic PPO algorithm config."""
    class_name: str = "PPOMultiCritic"
    # HoST g1_config_ground.py:
    #   reward_group_names = ['task', 'regu', 'style', 'target']  (HoST order)
    #   reward_group_weights = [2.5, 0.1, 1.0, 1.0]              (task=2.5, regu=0.1, style=1.0)
    # Our reward_group_names = ['task', 'style', 'regu', 'target'] (different order)
    # So our weights must be [2.5, 1.0, 0.1, 1.0] to match HoST's effective values.
    critic_weights: list = [2.5, 1.0, 0.1, 1.0]
    # L2C2 value smoothness: stabilizes critic training by penalizing ||V_t - V_{t-1}||²
    # HoST value_smoothness_coef = 0.1. Missing in V7/V8 was the likely root cause of divergence.
    value_smoothness_coef: float = 0.1


@configclass
class RslRlOnPolicyRunnerStandupCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for multi-critic standup task."""

    class_name: str = "StandupRunner"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "atom01_standup"
    wandb_project = "atom01_standup"
    reward_group_names: list = ["task", "style", "regu", "target"]
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    policy = RslRlPpoMultiCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
    )
    algorithm = RslRlPpoMultiCriticAlgCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,  # V10: reduced from 0.01 to prevent entropy dominating actor
                             # V9 failed: entropy_loss=62.6 >> surrogate=0.005 → noise_std=3.80
                             # V8@0.001 had noise_std=0.47; V10 = V8 entropy + V9 L2C2
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.998,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
