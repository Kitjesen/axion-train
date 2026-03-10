# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AMP environment config for a single model that can both walk AND get up from the ground.

Design:
- Combined motion data: walking (atom01_lab) + getup (atom01_lab_getup) via symlinks
  in atom01_lab_combined/. Getup motions prefixed with 'getup_'.
- RSI (Reference State Initialization) automatically gives:
    * walking-start episodes (from walking motion frames)
    * lying-start episodes (from getup motion frames)
- AMP discriminator reward shapes behavior based on context:
    * Robot standing → walking style rewarded
    * Robot lying → getup style rewarded
- Task rewards: velocity tracking (walking) + height (standup) + alive
- Termination: NO base_contact — robot can fall and recover.
- Episode length: 20s to allow time for getup + walking.

V4 (resumed from V3 model_12199):
- No curriculum (common_step_counter resets on resume; Phase 3 weights applied from iter 0)
- base_height 2.0→3.0 (slightly stronger standup signal to fix occasional getup failures)
- rising_velocity 3.0→5.0 (stronger direct gradient when lying)
- track_ang_vel_z 1.0→1.5 (better yaw tracking, reduce yaw drift error_vel_yaw 1.57→<1.0)
"""

import os
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import robolab.tasks.manager_based.amp.mdp as mdp
from robolab.tasks.manager_based.amp.atom01_amp_env_cfg import Atom01AmpEnvCfg
from robolab import ROBOLAB_ROOT_DIR


@configclass
class Atom01AmpGetupEnvCfg(Atom01AmpEnvCfg):
    """AMP env for combined walk + getup skill in a single model."""

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------
        # Motion data: walking + getup combined
        # ------------------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "atom01_lab_combined"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            # ---- Walking (weight=1 each) ----
            "127_03": 1,
            "127_04": 1,
            "127_06": 1,
            "B4_-_Stand_to_Walk_backwards_stageii": 1,
            "B9_-__Walk_turn_left_90_stageii": 1,
            "B10_-__Walk_turn_left_45_stageii": 1,
            "B13_-__Walk_turn_right_90_stageii": 1,
            "B14_-__Walk_turn_right_45_t2_stageii": 1,
            "B15_-__Walk_turn_around_stageii": 1,
            "B22_-__side_step_left_stageii": 1,
            "B23_-__side_step_right_stageii": 1,
            "C12_-_run_turn_left_45_stageii": 1,
            "C17_-_run_change_direction_stageii": 1,
            "xuanzhuan": 1,
            # ---- Getup (weight=2: upsampled since harder & fewer) ----
            "getup_crawl_stand": 2,
            "getup_crouchwalk_stand": 2,
            "getup_crouchwalk_stand2": 2,
            "getup_sit_stand": 2,
            "getup_airkick_stand": 2,
            "getup_dance_stand": 2,
            "getup_devishdance_stand": 2,
            "getup_jumpingtwist_stand": 2,
            "getup_kick_stand": 2,
            "getup_punchboxing_stand": 2,
            "getup_punchkarate_stand": 2,
            "getup_run_stand": 2,
            "getup_turntwist_stand": 2,
            "getup_walkbackwards_stand": 2,
            "getup_walksideways_stand": 2,
        }

        # ------------------------------------------------------------------
        # Rewards (V4: Phase 3 values applied from iter 0, no curriculum)
        # ------------------------------------------------------------------
        # Velocity tracking: Phase 3 final values
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5   # V4: 1.0→1.5, reduce yaw drift
        self.rewards.alive.weight = 1.0

        # Height reward: slightly stronger than V3 Phase 3 to fix occasional getup failures
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_exp,
            weight=3.0,  # V4: 2.0→3.0 (V3 Phase 3 was 2.0)
            params={"target_height": 0.72, "sigma": 0.3},
        )

        # Rising velocity: stronger direct gradient when lying (V4: 3.0→5.0)
        self.rewards.rising_velocity = RewTerm(
            func=mdp.rising_velocity,
            weight=5.0,  # V4: 3.0→5.0
            params={"height_threshold": 0.5},
        )

        # Reduce flat orientation penalty when robot might be on ground
        self.rewards.flat_orientation_l2.weight = -0.5  # was -2.0

        # Keep all regularization as in walking AMP
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.joint_vel_l2.weight = -2e-4
        self.rewards.joint_acc_l2.weight = -5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_energy.weight = -1e-4
        self.rewards.joint_torques_l2.weight = -1e-5
        self.rewards.joint_regularization.weight = -1e-3
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_stumble.weight = -0.1
        self.rewards.sound_suppression.weight = -5e-5
        self.rewards.feet_air_time_positive_biped.weight = 0.5
        self.rewards.undesired_contacts.weight = -1.0

        # ------------------------------------------------------------------
        # Termination: allow falling — robot can recover from ground
        # ------------------------------------------------------------------
        self.terminations.base_contact = None    # no fall termination
        self.terminations.bad_orientation = None  # robot starts lying → no orientation gate
        self.terminations.base_height = None      # pelvis is naturally low when lying
        # Only timeout termination remains.

        # ------------------------------------------------------------------
        # Episode length: longer to allow getup + walking
        # ------------------------------------------------------------------
        self.episode_length_s = 20.0  # was 10s for walking only

        # ------------------------------------------------------------------
        # Velocity commands
        # ------------------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 3.0)  # V4: 2.0→3.0
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # ------------------------------------------------------------------
        # Push: strong enough to knock robot over → forces getup practice
        # interval 8-15s; velocity ±2.0 m/s xy + ±3.0 rad/s yaw
        # ------------------------------------------------------------------
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 15.0),
            params={"velocity_range": {
                "x": (-2.0, 2.0),
                "y": (-2.0, 2.0),
                "yaw": (-3.0, 3.0),
            }},
        )

        self.disable_zero_weight_rewards()


@configclass
class Atom01AmpGetupEnvCfg_PLAY(Atom01AmpGetupEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # 给一定速度命令，让站起后能走动
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        # 保留推力：主动把站着的机器人推倒，测试起身能力
