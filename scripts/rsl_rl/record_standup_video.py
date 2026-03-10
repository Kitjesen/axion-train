#!/usr/bin/env python3
"""
Record standup policy video with multiple robots.
Each robot gets its own TiledCamera view; all views tiled into a 4x4 grid.

Usage:
    python record_standup_video.py \
        --checkpoint /path/to/model_22600.pt \
        --num_envs 16 \
        --video_length 600 \
        --output /root/autodl-tmp/standup_16bots.mp4
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record standup policy video.")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to model checkpoint (.pt file)")
parser.add_argument("--num_envs", type=int, default=16,
                    help="Number of robots to show (default: 16, tiled 4x4)")
parser.add_argument("--video_length", type=int, default=600,
                    help="Steps to record (600 steps = 12s at dt=0.02)")
parser.add_argument("--output", type=str,
                    default="/root/autodl-tmp/standup_16bots.mp4")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ────────────────────────────────────────────────────────
import gymnasium as gym
import numpy as np
import torch
import imageio

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import robolab.tasks  # registers Atom01-Standup-Play
from robolab.tasks.manager_based.standup.standup_env_cfg import Atom01StandupEnvCfg_PLAY
from robolab.tasks.manager_based.standup.agents.atom01_standup_agent_cfg import (
    RslRlOnPolicyRunnerStandupCfg,
)


# ── env config with TiledCamera ───────────────────────────────────────────────

@configclass
class StandupVideoEnvCfg(Atom01StandupEnvCfg_PLAY):
    """PLAY config with per-env TiledCamera for video recording."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = args_cli.num_envs
        self.scene.env_spacing = 3.0
        self.episode_length_s = 14.0  # 700 steps — plenty of time to stand up

        # Per-env camera: positioned 2.5m front-right, 1.5m high, looking at robot
        # Rotation is overridden after env creation via USD look-at.
        self.scene.tiled_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(2.5, -2.5, 1.5),
                rot=(1.0, 0.0, 0.0, 0.0),  # placeholder; overridden below
                convention="world",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 15.0),
            ),
            width=320,
            height=240,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def point_cameras_at(num_envs: int, eye: tuple, target: tuple):
    """Set all per-env cameras to look from `eye` toward `target` (env-local coords)."""
    import omni.usd
    from pxr import UsdGeom, Gf

    stage = omni.usd.get_context().get_stage()
    eye_v = Gf.Vec3d(*eye)
    tgt_v = Gf.Vec3d(*target)
    up_v = Gf.Vec3d(0.0, 0.0, 1.0)

    fwd = (tgt_v - eye_v).GetNormalized()
    right = (fwd ^ up_v).GetNormalized()
    new_up = (right ^ fwd).GetNormalized()

    mat = Gf.Matrix4d()
    mat.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0))
    mat.SetRow(1, Gf.Vec4d(new_up[0], new_up[1], new_up[2], 0))
    mat.SetRow(2, Gf.Vec4d(-fwd[0], -fwd[1], -fwd[2], 0))
    mat.SetRow(3, Gf.Vec4d(eye_v[0], eye_v[1], eye_v[2], 1))

    success = 0
    for i in range(num_envs):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Camera")
        if prim.IsValid():
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(mat)
            success += 1
    print(f"[Camera] Pointed {success}/{num_envs} cameras at {target}")


def tile_frames(frames: np.ndarray, ncols: int = 4) -> np.ndarray:
    """
    frames: [N, H, W, 3]  uint8
    Returns: single image with all N frames in a ncols-wide grid.
    """
    n, h, w, c = frames.shape
    nrows = math.ceil(n / ncols)
    grid = np.zeros((nrows * h, ncols * w, c), dtype=np.uint8)
    for i in range(n):
        r, col = divmod(i, ncols)
        grid[r * h:(r + 1) * h, col * w:(col + 1) * w] = frames[i]
    return grid


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # Build environment
    env_cfg = StandupVideoEnvCfg()
    raw_env = gym.make("Atom01-Standup-Play", cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=True)

    # Point cameras after env prims are created
    point_cameras_at(
        num_envs=args_cli.num_envs,
        eye=(2.5, -2.5, 1.5),
        target=(0.0, 0.0, 0.5),
    )

    # Load policy
    from rsl_rl.runners import StandupRunner, OnPolicyRunner

    agent_cfg = RslRlOnPolicyRunnerStandupCfg()
    device = getattr(args_cli, "device", None) or "cuda:0"
    agent_cfg.device = device

    RunnerCls = StandupRunner if agent_cfg.class_name == "StandupRunner" else OnPolicyRunner
    runner = RunnerCls(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")

    policy = runner.alg.policy
    policy.eval()

    # Camera sensor handle
    camera = raw_env.unwrapped.scene["tiled_camera"]

    # Run inference + capture frames
    obs = env.get_observations()
    frames = []
    print(f"[INFO] Recording {args_cli.video_length} steps "
          f"({args_cli.num_envs} robots, {320}x{240} each → "
          f"{320 * 4}x{240 * math.ceil(args_cli.num_envs / 4)} tiled)...")

    for step in range(args_cli.video_length):
        with torch.inference_mode():
            actions = policy.act_inference(obs)
            obs, _rew, _dones, _extras = env.step(actions)

        rgb = camera.data.output.get("rgb")
        if rgb is not None:
            arr = rgb.cpu().numpy()                          # [N, H, W, C]
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            frames.append(tile_frames(arr[:, :, :, :3], ncols=4))

        if step % 100 == 0:
            print(f"  step {step:4d}/{args_cli.video_length}  "
                  f"frames captured: {len(frames)}")

    env.close()

    # Save video
    if not frames:
        print("[ERROR] No frames captured — check TiledCamera setup.")
        return

    out_dir = os.path.dirname(args_cli.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    imageio.mimwrite(args_cli.output, frames, fps=50)
    h, w = frames[0].shape[:2]
    print(f"\n[DONE] Saved {len(frames)} frames → {args_cli.output}")
    print(f"       Resolution: {w}x{h}  |  Duration: {len(frames)/50:.1f}s")


if __name__ == "__main__":
    main()
    simulation_app.close()
