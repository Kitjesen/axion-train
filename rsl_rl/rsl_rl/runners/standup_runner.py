# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Runner for multi-critic standup training.

Extends OnPolicyRunner to:
- Construct multi-critic storage
- Pass per-group rewards to the algorithm
"""

from __future__ import annotations

import os
import time
import torch
import warnings
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.multi_critic_rollout_storage import MultiCriticRolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class StandupRunner(OnPolicyRunner):
    """Runner for multi-critic standup training."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.reward_group_names = train_cfg.get("reward_group_names", ["task", "style", "regu", "target"])
        super().__init__(env, train_cfg, log_dir, device)

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        """Construct multi-critic algorithm with appropriate storage."""
        # Resolve configs (same as parent)
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Deprecated normalization
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Create policy
        actor_critic_class = resolve_callable(self.policy_cfg.pop("class_name"))
        actor_critic = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Create storage: multi-critic if policy has multiple critics
        num_critics = getattr(actor_critic, "num_critics", 1)
        if num_critics > 1:
            storage = MultiCriticRolloutStorage(
                "rl", self.env.num_envs, self.cfg["num_steps_per_env"],
                obs, [self.env.num_actions], self.device, num_critics=num_critics,
            )
        else:
            storage = RolloutStorage(
                "rl", self.env.num_envs, self.cfg["num_steps_per_env"],
                obs, [self.env.num_actions], self.device,
            )

        # Create algorithm
        alg_class = resolve_callable(self.alg_cfg.pop("class_name"))
        alg = alg_class(
            actor_critic, storage, device=self.device,
            **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg,
        )
        return alg

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Training loop with multi-critic grouped reward support."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        use_multi_critic = hasattr(self.alg.policy, "num_critics") and self.alg.policy.num_critics > 1

        for it in range(start_it, total_it):
            start = time.time()

            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # Multi-critic: pass grouped rewards to algorithm
                    if use_multi_critic and "reward_groups" in extras:
                        grouped = torch.stack([
                            extras["reward_groups"][g].to(self.device)
                            for g in self.reward_group_names
                        ], dim=-1)  # [N, num_critics]
                        self.alg.process_env_step(obs, grouped, dones, extras)
                    else:
                        self.alg.process_env_step(obs, rewards, dones, extras)

                    # Logger uses total rewards (not grouped)
                    self.logger.process_env_step(rewards, dones, extras, None)

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it, start_it=start_it, total_it=total_it,
                collect_time=collect_time, learn_time=learn_time,
                loss_dict=loss_dict, learning_rate=self.alg.learning_rate,
                action_std=self.alg.policy.action_std, rnd_weight=None,
            )

            if it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
