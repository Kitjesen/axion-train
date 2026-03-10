# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AMP environment config for PHUMA dataset integration.

Two configs:
  1. Atom01AmpPhumaEnvCfg — pure PHUMA test (200 sampled motions, no atom01_lab).
     Used to validate PHUMA data compatibility with the AMP pipeline before mixing.

  2. Atom01AmpLabPhumaEnvCfg — combined atom01_lab + PHUMA sample.
     Keeps the original 14 atom01_lab walking motions and adds 200 PHUMA motions
     at a lower per-file weight so lab data still dominates the style signal.

Motion data note:
  PHUMA was converted from G1 (29 DOF) → ATOM01 (23 DOF) via joint mapping.
  G1 and ATOM01 have different joint zero positions, so the motions may look
  slightly different from the original lab captures. Leg joints are most
  accurate; arm joints approximate; wrist joints are zeroed.

  atom01_phuma_sample200/ contains 200 randomly sampled pkl files (seed=42)
  from the full 40,061-file conversion.
"""

import os
from isaaclab.utils import configclass

from robolab.tasks.manager_based.amp.atom01_amp_env_cfg import Atom01AmpEnvCfg
from robolab import ROBOLAB_ROOT_DIR


@configclass
class Atom01AmpPhumaEnvCfg(Atom01AmpEnvCfg):
    """AMP env using only PHUMA motion data (200 sampled motions).

    Use this to test if PHUMA data is compatible with the AMP pipeline.
    If training converges, the robot should walk with PHUMA-style movements.
    """

    def __post_init__(self):
        super().__post_init__()

        # Use PHUMA sample: 200 files, all loaded with default weight=1.0.
        # motion_data_weights is empty → all files in the dir use default_weight.
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "atom01_phuma_sample200"
        )
        self.motion_data.motion_dataset.motion_data_weights = {}
        self.motion_data.motion_dataset.motion_data_default_weight = 1.0


@configclass
class Atom01AmpPhumaEnvCfg_PLAY(Atom01AmpPhumaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class Atom01AmpLabPhumaEnvCfg(Atom01AmpEnvCfg):
    """AMP env mixing atom01_lab walking motions + PHUMA sample.

    Weight design:
      - atom01_lab: 14 files × weight=1 each  →  14 total weight units
      - atom01_phuma_sample200: 200 files × default_weight=0.07 each → ~14 units
      → Roughly equal contribution from lab and PHUMA data.

    Requires a combined directory: atom01_lab_phuma200/ with symlinks from both.
    Create on server:
        cd /root/autodl-tmp/atom01_train/robolab/data/motions
        mkdir atom01_lab_phuma200
        ln -s ../atom01_lab/*.pkl atom01_lab_phuma200/
        ln -s ../atom01_phuma_sample200/*.pkl atom01_lab_phuma200/
    """

    def __post_init__(self):
        super().__post_init__()

        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "atom01_lab_phuma200"
        )
        # Explicit high weights for curated lab motions; PHUMA files use default.
        self.motion_data.motion_dataset.motion_data_weights = {
            # ---- atom01_lab walking (weight=1 each) ----
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
        }
        # PHUMA files not listed above get this weight: 14/200 ≈ 0.07
        # so total PHUMA weight ≈ 200 × 0.07 = 14 = same as atom01_lab total
        self.motion_data.motion_dataset.motion_data_default_weight = 0.07


@configclass
class Atom01AmpLabPhumaEnvCfg_PLAY(Atom01AmpLabPhumaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
