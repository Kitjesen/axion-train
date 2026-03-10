# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the HoST-inspired standup task.

Reference: HoST (RSS 2025) - Humanoid Standing-up Control
https://github.com/OpenRobotLab/HoST
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Helper functions (ported from HoST g1_utils.py)
# ---------------------------------------------------------------------------

def _sigmoid(x: torch.Tensor, value_at_1: float) -> torch.Tensor:
    """Gaussian bell curve. Returns value_at_1 when x = 1.0."""
    scale = np.sqrt(-2 * np.log(value_at_1))
    return torch.exp(-0.5 * (x * scale) ** 2)


def _tolerance(
    x: torch.Tensor,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    value_at_margin: float = 0.1,
) -> torch.Tensor:
    """Smooth tolerance function. Returns 1.0 inside bounds, Gaussian falloff outside.

    Args:
        x: Input tensor.
        bounds: (lower, upper) bounds where output = 1.0.
        margin: Distance from boundary where output = value_at_margin.
        value_at_margin: Output value at distance margin from boundary.
    """
    lower, upper = bounds
    in_bounds = (lower <= x) & (x <= upper)
    if margin == 0:
        return torch.where(in_bounds, torch.ones_like(x), torch.zeros_like(x))
    d = torch.where(x < lower, lower - x, x - upper) / margin
    return torch.where(in_bounds, torch.ones_like(x), _sigmoid(d, value_at_margin))


# ---------------------------------------------------------------------------
# Task rewards (HoST: multiplicative group)
# ---------------------------------------------------------------------------

def orientation_reward(
    env: ManagerBasedRLEnv,
    threshold: float = 0.99,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for body being upright (projected gravity z-component close to -1).

    HoST: _reward_orientation. tolerance(-g_z, [threshold, inf], margin=1, val=0.05)
    When upright: projected_gravity_b[:,2] = -1, so -g_z = +1 >= threshold.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    neg_gz = -asset.data.projected_gravity_b[:, 2]
    return _tolerance(neg_gz, (threshold, float("inf")), margin=1.0, value_at_margin=0.05)


def base_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.72,
    sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for base height approaching target standing height.

    Uses exponential kernel: exp(-(h - h_target)² / sigma²).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    return torch.exp(-torch.square(height - target_height) / (sigma ** 2))


def head_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.65,
    margin: float = 0.8,
    head_body: str = "base_link",
    feet_bodies: list[str] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for head/torso height above feet.

    HoST: _reward_head_height. tolerance(head_h - feet_h, [target, inf], margin, 0.1).
    For ATOM01 we use base_link as proxy for head since there's no keyframe_head body.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Use base (torso) height as proxy
    base_height = asset.data.root_pos_w[:, 2]
    # Approximate feet height as min of ankle body z positions
    feet_body_ids = asset.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"])[0]
    feet_height = asset.data.body_pos_w[:, feet_body_ids, 2].mean(dim=-1)
    height_above_feet = base_height - feet_height
    return _tolerance(height_above_feet, (target_height, float("inf")), margin=margin, value_at_margin=0.1)


# ---------------------------------------------------------------------------
# Regularization rewards (HoST: regu group, additive)
# ---------------------------------------------------------------------------

def joint_torques_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint torques using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint accelerations using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize rate of change of actions."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )


def joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint velocities using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_energy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize mechanical energy: sum(|torque * joint_vel|)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        * torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=-1,
    )


def joint_pos_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions exceeding soft limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


# ---------------------------------------------------------------------------
# Style rewards (HoST: style group, additive)
# ---------------------------------------------------------------------------

def flat_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-flat base orientation (xy components of projected gravity)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def ang_vel_xy_reward(
    env: ManagerBasedRLEnv,
    min_height: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for low angular velocity in xy when standing.

    HoST: _reward_style_ang_vel_xy. exp(-2*(wx²+wy²)) gated by height > min_height.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    standing = (asset.data.root_pos_w[:, 2] > min_height).float()
    ang_vel_sq = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    return torch.exp(-2.0 * ang_vel_sq) * standing


def lin_vel_xy_penalty(
    env: ManagerBasedRLEnv,
    min_height: float = 0.55,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for low horizontal velocity when standing (stay in place).

    HoST: _reward_lin_vel_xy (target group). exp(-5*(vx²+vy²)) gated by height.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    standing = (asset.data.root_pos_w[:, 2] > min_height).float()
    lin_vel_sq = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-5.0 * lin_vel_sq) * standing


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive (not terminated)."""
    return (~env.termination_manager.terminated).float()


def undesired_contacts(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize undesired body contacts (non-feet touching ground)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
        > threshold
    )
    return torch.sum(is_contact, dim=1)


# ---------------------------------------------------------------------------
# Phase 4: Posture quality rewards
# ---------------------------------------------------------------------------

def joint_default_pos_reward(
    env: ManagerBasedRLEnv,
    sigma: float = 1.5,
    min_height: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for joints near default standing pose, gated by height.

    HoST: _reward_joint_default (target group). Pulls joints toward the
    default standing pose once the robot is somewhat upright, fixing
    wide-leg posture and awkward arm positions.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    standing = (asset.data.root_pos_w[:, 2] > min_height).float()
    diff = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.exp(-torch.sum(torch.square(diff), dim=1) / (sigma ** 2)) * standing


def feet_contact_reward(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    min_height: float = 0.4,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for both feet in contact with ground when standing.

    Prevents floating/drifting after standing up. Only active once somewhat upright.
    sensor_cfg should have body_names=["left_ankle_roll_link", "right_ankle_roll_link"].
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    standing = (asset.data.root_pos_w[:, 2] > min_height).float()

    net_contact_forces = contact_sensor.data.net_forces_w_history  # [N, hist, bodies, 3]
    # Max contact force over history per foot: [N, 2]
    feet_force = torch.max(
        torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1
    )[0]
    both_feet = (feet_force > threshold).all(dim=-1).float()  # [N]
    return both_feet * standing


# ---------------------------------------------------------------------------
# Regularization: 2nd-order action smoothness (HoST: regu_smoothness)
# ---------------------------------------------------------------------------

def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """2nd-order action smoothness penalty.

    HoST: regu_smoothness = sum((a_t - 2*a_{t-1} + a_{t-2})^2)
    Requires env._last_last_action buffer provided by StandupEnv.
    """
    if not hasattr(env, "_last_last_action"):
        return torch.zeros(env.num_envs, device=env.device)
    second_order_diff = (
        env.action_manager.action
        - 2.0 * env.action_manager.prev_action
        + env._last_last_action
    )
    return torch.sum(torch.square(second_order_diff), dim=1)


# ---------------------------------------------------------------------------
# Style: joint deviation penalties (HoST: binary threshold approach)
# ---------------------------------------------------------------------------

def thigh_yaw_deviation(
    env: ManagerBasedRLEnv,
    threshold_max: float = 1.4,
    threshold_any: float = 0.9,
    min_height: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_thigh_yaw_joint", "right_thigh_yaw_joint"]
    ),
) -> torch.Tensor:
    """Penalize excessive thigh yaw rotation when upright.

    HoST: hip_yaw_deviation — fires when max(|joint|) > threshold_max
    OR min(|joint|) > threshold_any (both joints are moderately rotated).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    standing = (asset.data.root_pos_w[:, 2] > min_height).float()
    abs_angles = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids])  # [N, 2]
    deviation = (
        (torch.max(abs_angles, dim=-1)[0] > threshold_max)
        | (torch.min(abs_angles, dim=-1)[0] > threshold_any)
    ).float()
    return deviation * standing


def thigh_roll_deviation(
    env: ManagerBasedRLEnv,
    threshold_max: float = 1.4,
    threshold_any: float = 0.9,
    min_height: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_thigh_roll_joint", "right_thigh_roll_joint"]
    ),
) -> torch.Tensor:
    """Penalize excessive thigh roll rotation when upright.

    HoST: hip_roll_deviation — same binary threshold pattern.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    standing = (asset.data.root_pos_w[:, 2] > min_height).float()
    abs_angles = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids])  # [N, 2]
    deviation = (
        (torch.max(abs_angles, dim=-1)[0] > threshold_max)
        | (torch.min(abs_angles, dim=-1)[0] > threshold_any)
    ).float()
    return deviation * standing


def arm_roll_deviation(
    env: ManagerBasedRLEnv,
    threshold: float = 0.02,
    min_height: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_arm_roll_joint", "right_arm_roll_joint"]
    ),
) -> torch.Tensor:
    """Penalize arms rolling into unnatural position.

    HoST: shoulder_roll_deviation — fires when left arm < -threshold or right > threshold.
    ATOM01: arm_roll=0 is neutral; penalize significant asymmetric deviation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    standing = (asset.data.root_pos_w[:, 2] > min_height).float()
    angles = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [N, 2], [left, right]
    deviation = (
        (angles[:, 0] < -threshold) | (angles[:, 1] > threshold)
    ).float()
    return deviation * standing


def torso_deviation(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    min_height: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["torso_joint"]
    ),
) -> torch.Tensor:
    """Penalize large torso rotation when upright.

    HoST: waist_deviation — |waist| > 1.4 threshold.
    ATOM01 has a single torso_joint; use slightly tighter threshold.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    standing = (asset.data.root_pos_w[:, 2] > min_height).float()
    # joint_pos[:, joint_ids] → [N, 1] for single joint; squeeze to [N]
    abs_angle = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids]).view(env.num_envs, -1).max(dim=-1)[0]
    return (abs_angle > threshold).float() * standing


# ---------------------------------------------------------------------------
# Style: body-position based penalties (feet distance, shank, ground parallel)
# ---------------------------------------------------------------------------

def _get_body_ids_cached(env: ManagerBasedRLEnv, asset_name: str, body_names: list) -> list:
    """Cache body indices to avoid repeated find_bodies() calls per step."""
    key = f"_cached_bids_{asset_name}_{'_'.join(body_names)}"
    if not hasattr(env, key):
        asset = env.scene[asset_name]
        # find_bodies returns indices in articulation order; build name->id map
        all_ids, all_names = asset.find_bodies(body_names)
        name_to_id = {n: i for n, i in zip(all_names, all_ids)}
        ordered_ids = [name_to_id[n] for n in body_names]
        setattr(env, key, ordered_ids)
    return getattr(env, key)


def feet_distance_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 0.6,
    min_height: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet being too far apart (> threshold meters).

    HoST: feet_distance — (||left_foot - right_foot||_xy > 0.9).
    ATOM01 is smaller than G1 so use tighter threshold 0.6m.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    standing = (asset.data.root_pos_w[:, 2] > min_height).float()
    ids = _get_body_ids_cached(env, asset_cfg.name,
                               ["left_ankle_roll_link", "right_ankle_roll_link"])
    left_pos = asset.data.body_pos_w[:, ids[0], :2]   # [N, 2]
    right_pos = asset.data.body_pos_w[:, ids[1], :2]  # [N, 2]
    dist = torch.norm(left_pos - right_pos, dim=-1)    # [N]
    return (dist > threshold).float() * standing


def shank_orientation_reward(
    env: ManagerBasedRLEnv,
    min_vertical_ratio: float = 0.8,
    margin: float = 1.0,
    min_height: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for vertical shank orientation (knee directly above ankle).

    HoST: shank_orientation = tolerance((knee-ankle).z / ||knee-ankle||, [0.8, inf])
    * (h > phase1_height). Returns sum over both legs.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    standing = (asset.data.root_pos_w[:, 2] > min_height).float()
    knee_ids = _get_body_ids_cached(env, asset_cfg.name,
                                    ["left_knee_link", "right_knee_link"])
    ankle_ids = _get_body_ids_cached(env, asset_cfg.name,
                                     ["left_ankle_roll_link", "right_ankle_roll_link"])
    knee_pos = asset.data.body_pos_w[:, knee_ids, :]    # [N, 2, 3]
    ankle_pos = asset.data.body_pos_w[:, ankle_ids, :]  # [N, 2, 3]

    # Shank vector: from ankle to knee (should be positive-z when upright)
    shank_vec = knee_pos - ankle_pos                                      # [N, 2, 3]
    norm = torch.norm(shank_vec, dim=-1).clamp(min=1e-6)                  # [N, 2]
    vertical_ratio = shank_vec[:, :, 2] / norm                            # [N, 2]

    reward = (
        _tolerance(vertical_ratio[:, 0], (min_vertical_ratio, float("inf")),
                   margin=margin, value_at_margin=0.1)
        + _tolerance(vertical_ratio[:, 1], (min_vertical_ratio, float("inf")),
                     margin=margin, value_at_margin=0.1)
    )
    return reward * standing


def ground_parallel_reward(
    env: ManagerBasedRLEnv,
    variance_threshold: float = 0.05,
    min_height: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for both ankles at the same height (feet parallel to ground).

    HoST: ground_parallel = (ankle_height_variance < 0.05) * (h > phase2_height).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    standing = (asset.data.root_pos_w[:, 2] > min_height).float()
    ids = _get_body_ids_cached(env, asset_cfg.name,
                               ["left_ankle_roll_link", "right_ankle_roll_link"])
    ankle_heights = torch.stack([
        asset.data.body_pos_w[:, ids[0], 2],
        asset.data.body_pos_w[:, ids[1], 2],
    ], dim=-1)  # [N, 2]
    height_var = ankle_heights.var(dim=-1)  # [N]
    return (height_var < variance_threshold).float() * standing


# ---------------------------------------------------------------------------
# Target group: post-standing rewards (gated by phase3 height ≥ 0.65m)
# ---------------------------------------------------------------------------

def target_ang_vel_xy(
    env: ManagerBasedRLEnv,
    min_height: float = 0.65,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for low angular velocity when fully upright (target group).

    HoST: target_ang_vel_xy = exp(-2*(wx²+wy²)) * (h > phase3_height). Weight=10.
    Separate from style_ang_vel_xy (weight=1, gated at lower height).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    upright = (asset.data.root_pos_w[:, 2] > min_height).float()
    ang_vel_sq = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    return torch.exp(-2.0 * ang_vel_sq) * upright


def target_lin_vel_xy(
    env: ManagerBasedRLEnv,
    min_height: float = 0.65,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for staying in place when fully upright (target group).

    HoST: target_lin_vel_xy = exp(-5*(vx²+vy²)) * (h > phase3_height). Weight=10.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    upright = (asset.data.root_pos_w[:, 2] > min_height).float()
    lin_vel_sq = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-5.0 * lin_vel_sq) * upright


def target_upper_dof_pos(
    env: ManagerBasedRLEnv,
    sigma: float = -0.1,
    min_height: float = 0.65,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=[
            "left_arm_pitch_joint", "left_arm_roll_joint", "left_arm_yaw_joint",
            "left_elbow_pitch_joint", "left_elbow_yaw_joint",
            "right_arm_pitch_joint", "right_arm_roll_joint", "right_arm_yaw_joint",
            "right_elbow_pitch_joint", "right_elbow_yaw_joint",
        ],
    ),
) -> torch.Tensor:
    """Reward for upper body (arms) returning to default pose when fully upright.

    HoST: target_upper_dof_pos = exp(mse * sigma) * (h > phase3_height). Weight=10.
    Pulls arm joints toward default standing pose once robot is upright.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    upright = (asset.data.root_pos_w[:, 2] > min_height).float()
    diff = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    mse = torch.sum(torch.square(diff), dim=-1)
    return torch.exp(mse * sigma) * upright


def target_base_height(
    env: ManagerBasedRLEnv,
    target_height: float = 0.72,
    margin: float = 0.25,
    min_height: float = 0.65,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maintaining target base height when fully upright.

    HoST: target_base_height = tolerance(h, [target, inf], margin=0.25) * (h > phase3).
    Refines the broad base_height_reward to an exact target once upright.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    upright = (asset.data.root_pos_w[:, 2] > min_height).float()
    height = asset.data.root_pos_w[:, 2]
    return _tolerance(height, (target_height, float("inf")), margin=margin,
                      value_at_margin=0.1) * upright
