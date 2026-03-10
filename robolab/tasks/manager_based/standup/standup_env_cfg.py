# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the HoST-inspired standup task.

Phase 1 MVP: Single-critic PPO, basic rewards, random lying-down initialization.
No force curriculum or action rescaler yet (Phase 2).
No multi-critic PPO yet (Phase 3).

Reference: HoST (RSS 2025) - Humanoid Standing-up Control
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robolab.tasks.manager_based.standup.mdp as mdp
from robolab.assets.robots.roboparty import ATOM01_CFG


# ===========================================================================
# Scene
# ===========================================================================

@configclass
class StandupSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the standup task."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ATOM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ===========================================================================
# MDP: Actions
# ===========================================================================

@configclass
class ActionsCfg:
    """Action specifications: joint position targets."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


# ===========================================================================
# MDP: Observations
# ===========================================================================

@configclass
class ObservationsCfg:
    """Observation groups for policy and critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations (no privileged info)."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations (with privileged info)."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 3
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ===========================================================================
# MDP: Events (Randomization)
# ===========================================================================

@configclass
class EventCfg:
    """Event configuration for standup task.

    Key difference from AMP: robot is initialized lying on the ground.
    """

    # -- startup randomization
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )

    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_.*_link", "right_.*_link"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # -- reset events: spawn robot lying on the ground
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "roll": (-math.pi, math.pi),   # full rotation: face-up, face-down, side
                "pitch": (-0.3, 0.3),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )


# ===========================================================================
# MDP: Rewards
# ===========================================================================

@configclass
class StandupRewardsCfg:
    """Reward terms for standup task.

    Organized into 4 groups matching HoST (g1_config_ground.py):
      task  (critic weight 2.5): core standing objectives
      style (critic weight 1.0): posture quality
      regu  (critic weight 0.1): motion regularization
      target(critic weight 1.0): post-standing fine-grained goals
    """

    # ---- Task group -------------------------------------------------------
    orientation = RewTerm(
        func=mdp.orientation_reward,
        weight=5.0,
        params={"threshold": 0.9},
    )

    base_height = RewTerm(
        func=mdp.base_height_reward,
        weight=15.0,  # V11: 5→15, break crouching local minimum
        params={"target_height": 0.72, "sigma": 0.25},
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.5)

    # ---- Style group ------------------------------------------------------
    # Low angular velocity once somewhat standing (weight=1, gate at 0.45m)
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_reward,
        weight=1.0,
        params={"min_height": 0.45},
    )

    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    # Pulls joints toward default standing pose once upright
    joint_default_pos = RewTerm(
        func=mdp.joint_default_pos_reward,
        weight=2.0,
        params={"sigma": 1.5, "min_height": 0.45},
    )

    # Both feet on ground once standing
    feet_contact = RewTerm(
        func=mdp.feet_contact_reward,
        weight=1.0,
        params={
            "threshold": 1.0,
            "min_height": 0.4,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
        },
    )

    # Joint deviation penalties (HoST style group: binary thresholds)
    # NOTE: SceneEntityCfg with specific joint_names MUST be in params (not default args)
    # so the RewardManager resolves joint_ids correctly.
    thigh_yaw_dev = RewTerm(
        func=mdp.thigh_yaw_deviation,
        weight=-10.0,
        params={
            "threshold_max": 1.4,
            "threshold_any": 0.9,
            "min_height": 0.4,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_thigh_yaw_joint", "right_thigh_yaw_joint"],
            ),
        },
    )

    thigh_roll_dev = RewTerm(
        func=mdp.thigh_roll_deviation,
        weight=-10.0,
        params={
            "threshold_max": 1.4,
            "threshold_any": 0.9,
            "min_height": 0.4,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_thigh_roll_joint", "right_thigh_roll_joint"],
            ),
        },
    )

    arm_roll_dev = RewTerm(
        func=mdp.arm_roll_deviation,
        weight=-2.5,
        params={
            "threshold": 0.02,
            "min_height": 0.4,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_arm_roll_joint", "right_arm_roll_joint"],
            ),
        },
    )

    torso_dev = RewTerm(
        func=mdp.torso_deviation,
        weight=-10.0,
        params={
            "threshold": 1.0,
            "min_height": 0.4,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["torso_joint"]),
        },
    )

    feet_distance = RewTerm(
        func=mdp.feet_distance_penalty,
        weight=-10.0,
        params={"threshold": 0.6, "min_height": 0.3},
    )

    shank_orientation = RewTerm(
        func=mdp.shank_orientation_reward,
        weight=5.0,  # V11: 10→5, reduce dominance of crouching-compatible style rewards
        params={"min_vertical_ratio": 0.8, "margin": 1.0, "min_height": 0.25},
    )

    ground_parallel = RewTerm(
        func=mdp.ground_parallel_reward,
        weight=10.0,  # V11: 20→10, reduce dominance of crouching-compatible style rewards
        params={"variance_threshold": 0.05, "min_height": 0.35},
    )

    # ---- Regularization group --------------------------------------------
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-6)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # 2nd-order smoothness (HoST: regu_smoothness = -0.01)
    action_smoothness = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3)
    joint_energy_penalty = RewTerm(func=mdp.joint_energy, weight=-1e-4)
    # HoST: regu_dof_pos_limits = -100.0 (was -1.0, 100x too weak!)
    joint_pos_limits_penalty = RewTerm(func=mdp.joint_pos_limits, weight=-100.0)

    # ---- Target group (post-standing goals, gated by h > 0.65m) ----------
    # Low angular velocity when fully upright (higher weight than style version)
    target_ang_vel = RewTerm(
        func=mdp.target_ang_vel_xy,
        weight=10.0,
        params={"min_height": 0.65},
    )

    # Stay in place when fully upright
    target_lin_vel = RewTerm(
        func=mdp.target_lin_vel_xy,
        weight=10.0,
        params={"min_height": 0.65},
    )

    # Arms return to default when upright
    target_upper_dof = RewTerm(
        func=mdp.target_upper_dof_pos,
        weight=10.0,
        params={
            "sigma": -0.1,
            "min_height": 0.65,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_arm_pitch_joint", "left_arm_roll_joint", "left_arm_yaw_joint",
                    "left_elbow_pitch_joint", "left_elbow_yaw_joint",
                    "right_arm_pitch_joint", "right_arm_roll_joint", "right_arm_yaw_joint",
                    "right_elbow_pitch_joint", "right_elbow_yaw_joint",
                ],
            ),
        },
    )

    # Maintain target base height precisely when upright
    target_height = RewTerm(
        func=mdp.target_base_height,
        weight=10.0,
        params={"target_height": 0.72, "margin": 0.25, "min_height": 0.65},
    )


# ===========================================================================
# MDP: Terminations
# ===========================================================================

@configclass
class TerminationsCfg:
    """Termination terms for standup task.

    Note: No base_height or bad_orientation termination since robot starts on ground.
    Only time_out is used.
    """
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# ===========================================================================
# MDP: Curriculum (placeholder for Phase 2)
# ===========================================================================

@configclass
class CurriculumCfg:
    """Curriculum terms (placeholder, actual curriculum is in StandupEnv)."""
    pass


# ===========================================================================
# Force Curriculum & Action Rescaler (Phase 2)
# ===========================================================================

@configclass
class ForceCurriculumCfg:
    """Force curriculum: upward assist force that decreases with success."""
    initial_force: float = 120.0   # N (~100% of 12kg * 9.81), near-weightless to bootstrap exploration
    force_decay: float = 5.0       # N decrease per success step (slower decay for stable learning)
    min_force: float = 0.0         # minimum force
    success_height: float = 0.60   # m, robot height threshold for success
    success_gravity_z: float = -0.8  # projected_gravity_z threshold (< means upright)


@configclass
class ActionRescalerCfg:
    """Action rescaler: scale factor for policy actions."""
    initial_scale: float = 0.8     # start with near-full-strength actions
    min_scale: float = 0.3


# ===========================================================================
# Environment Configuration
# ===========================================================================

@configclass
class Atom01StandupEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the ATOM01 standup environment.

    Phase 2: Force curriculum + action rescaler.
    Phase 3: Reward groups for multi-critic PPO.
    """

    # Scene
    scene: StandupSceneCfg = StandupSceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: StandupRewardsCfg = StandupRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Phase 2: Force curriculum & action rescaler
    force_curriculum: ForceCurriculumCfg = ForceCurriculumCfg()
    action_rescaler: ActionRescalerCfg = ActionRescalerCfg()

    # Phase 3+4: Reward groups for multi-critic PPO
    # Maps group name -> list of reward term attribute names in StandupRewardsCfg
    # Critic weights order: [task=2.5, style=1.0, regu=0.1, target=1.0]
    # (matches atom01_standup_agent_cfg.py critic_weights)
    reward_groups: dict = {
        "task": ["orientation", "base_height", "alive"],
        "style": [
            "ang_vel_xy", "flat_orientation", "joint_default_pos", "feet_contact",
            "thigh_yaw_dev", "thigh_roll_dev", "arm_roll_dev", "torso_dev",
            "feet_distance", "shank_orientation", "ground_parallel",
        ],
        "regu": [
            "joint_torques", "joint_acc", "action_rate", "action_smoothness",
            "joint_vel", "joint_energy_penalty", "joint_pos_limits_penalty",
        ],
        "target": ["target_ang_vel", "target_lin_vel", "target_upper_dof", "target_height"],
    }

    # No commands needed for standup task

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0  # 10 seconds to stand up (500 steps)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Override robot init state: start low to the ground for standup
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.20)

        # Update sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class Atom01StandupEnvCfg_PLAY(Atom01StandupEnvCfg):
    """Play configuration with reduced envs and no randomization."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.episode_length_s = 20.0
        self.observations.policy.enable_corruption = False
        # Small force for play to match training level (~15N at iter 6000)
        self.force_curriculum.initial_force = 120.0
        self.action_rescaler.initial_scale = 1.0
