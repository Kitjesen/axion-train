# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AMP environment config for walk + getup + PHUMA combined skill.

Extends Atom01AmpGetupEnvCfg (V4 settings) with PHUMA motion data:

Weight design:
  - 14 lab walking motions (weight=1 each)  →  14 total weight
  - 15 lab getup motions   (weight=2 each)  →  30 total weight
  - 200 PHUMA motions      (weight=0.07 ea) →  14 total weight
                                               ──────────────
                               Total:          58 weight units

Getup still dominates (30/58 = 52%) so the robot definitely learns to
stand up. PHUMA adds 200 diverse walking styles so the discriminator
rewards more natural human-like locomotion alongside the getup skill.

Motion directory: atom01_lab_combined_phuma/
  Created on server with:
    cd /root/autodl-tmp/atom01_train/robolab/data/motions
    mkdir atom01_lab_combined_phuma
    ln -s ../atom01_lab_combined/*.pkl  atom01_lab_combined_phuma/
    ln -s ../atom01_phuma_sample200/*.pkl atom01_lab_combined_phuma/
"""

import os
from isaaclab.utils import configclass

from robolab.tasks.manager_based.amp.atom01_amp_getup_env_cfg import Atom01AmpGetupEnvCfg
from robolab import ROBOLAB_ROOT_DIR


@configclass
class Atom01AmpGetupPhumaEnvCfg(Atom01AmpGetupEnvCfg):
    """Walk + getup + PHUMA: V4 getup skill with PHUMA-enriched walking style."""

    def __post_init__(self):
        super().__post_init__()

        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "atom01_lab_combined_phuma"
        )
        # Explicit weights for lab motions; PHUMA files use default_weight.
        self.motion_data.motion_dataset.motion_data_weights = {
            # ---- Walking (weight=1 each, total=14) ----
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
            # ---- Getup (weight=2: upsampled, total=30) ----
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
        # 200 PHUMA files not listed above each get weight=0.07
        # → 200 × 0.07 = 14 total (equal to lab walk total)
        self.motion_data.motion_dataset.motion_data_default_weight = 0.07


@configclass
class Atom01AmpGetupPhumaEnvCfgV3(Atom01AmpGetupPhumaEnvCfg):
    """V3: same as V2 but push every 25-40s instead of 8-15s.

    Less frequent pushes → robot learns more natural walking gait.
    Still practices getup via RSI (episodes init from lying states).
    Push velocity kept strong (±2m/s) so recovery skill is maintained.

    episode_length_s=50s (was 20s): ensures push events (25-40s interval)
    can actually fire during training. With 20s episodes, pushes never
    triggered since 20s < 25s min interval → no push-recovery training.
    """

    def __post_init__(self):
        super().__post_init__()
        # Reduce push frequency: robot has more time to walk naturally
        self.events.push_robot.interval_range_s = (25.0, 40.0)
        # Extend episode so push events (25-40s) can fire during training
        self.episode_length_s = 50.0


@configclass
class Atom01AmpGetupPhumaEnvCfgV3_PLAY(Atom01AmpGetupPhumaEnvCfgV3):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.debug_vis = False


@configclass
class Atom01AmpGetupPhumaEnvCfg_PLAY(Atom01AmpGetupPhumaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        # Disable velocity arrow overlays for clean video
        self.commands.base_velocity.debug_vis = False


@configclass
class Atom01AmpGetupPhumaEnvCfg_PAPER(Atom01AmpGetupPhumaEnvCfg):
    """Paper-quality recording: 9 robots in 3x3 grid, circular walking, no overlays.

    Circular walking (lin_vel_x=0.8, ang_vel_z=0.8) keeps robots orbiting
    near their start positions (radius ≈ 1m), so they stay in frame the
    entire recording. RSI initializes some robots lying down → shows getup.
    Push events every 8-15s → shows fall recovery mid-episode.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 9
        self.scene.env_spacing = 2.5        # compact 3×3 grid, robots clearly visible
        self.episode_length_s = 30.0        # 30s: getup + circular walk + 1-2 push events
        # Circular walking: robots orbit ~1m radius, stay in camera view
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.8, 0.8)
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.debug_vis = False   # no velocity arrows


@configclass
class Atom01AmpGetupPhumaEnvCfg_SINGLE(Atom01AmpGetupPhumaEnvCfg):
    """Single-robot close-up: RSI from getup-only motions → always starts lying.

    For paper demo: shows complete lie → getup → walk → push → recover cycle.
    With only getup motions as RSI source, robot always initializes on ground.
    Push event at ~10s knocks standing robot down → demonstrates recovery.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 30.0
        # Force RSI to sample only from getup motions → always starts lying
        # Override motion_data_weights: set walking to weight=0, getup to weight=1
        self.motion_data.motion_dataset.motion_data_weights = {
            # Walking: weight=0 → never sampled for RSI
            "127_03": 0, "127_04": 0, "127_06": 0,
            "B4_-_Stand_to_Walk_backwards_stageii": 0,
            "B9_-__Walk_turn_left_90_stageii": 0,
            "B10_-__Walk_turn_left_45_stageii": 0,
            "B13_-__Walk_turn_right_90_stageii": 0,
            "B14_-__Walk_turn_right_45_t2_stageii": 0,
            "B15_-__Walk_turn_around_stageii": 0,
            "B22_-__side_step_left_stageii": 0,
            "B23_-__side_step_right_stageii": 0,
            "C12_-_run_turn_left_45_stageii": 0,
            "C17_-_run_change_direction_stageii": 0,
            "xuanzhuan": 0,
            # Getup: weight=1 each → always sampled from lying states
            "getup_crawl_stand": 1,
            "getup_crouchwalk_stand": 1,
            "getup_crouchwalk_stand2": 1,
            "getup_sit_stand": 1,
            "getup_airkick_stand": 1,
            "getup_dance_stand": 1,
            "getup_devishdance_stand": 1,
            "getup_jumpingtwist_stand": 1,
            "getup_kick_stand": 1,
            "getup_punchboxing_stand": 1,
            "getup_punchkarate_stand": 1,
            "getup_run_stand": 1,
            "getup_turntwist_stand": 1,
            "getup_walkbackwards_stand": 1,
            "getup_walksideways_stand": 1,
        }
        # PHUMA files → weight=0 (not used for RSI in single-robot demo)
        self.motion_data.motion_dataset.motion_data_default_weight = 0.0
        # Circular walking: robot orbits ~1m radius, stays in viewport
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.6, 0.6)
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.debug_vis = False
