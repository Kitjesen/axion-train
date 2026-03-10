"""
Convert PHUMA G1 motion data to ATOM01 format for AMP training.

G1 (29 DOF) → ATOM01 (23 DOF) joint mapping:
- Legs: direct mapping (with yaw/roll/pitch reorder)
- Torso: G1 waist_pitch → ATOM01 torso_joint
- Arms: G1 shoulder/elbow → ATOM01 arm/elbow (wrists dropped)
- elbow_yaw: set to 0 (G1 has no direct analog)

Output pkl format (lab_dof_names order, interleaved left/right):
  [0]  left_thigh_yaw    [1]  right_thigh_yaw
  [2]  torso             [3]  left_thigh_roll
  [4]  right_thigh_roll  [5]  left_arm_pitch
  [6]  right_arm_pitch   [7]  left_thigh_pitch
  [8]  right_thigh_pitch [9]  left_arm_roll
  [10] right_arm_roll    [11] left_knee
  [12] right_knee        [13] left_arm_yaw
  [14] right_arm_yaw     [15] left_ankle_pitch
  [16] right_ankle_pitch [17] left_elbow_pitch
  [18] right_elbow_pitch [19] left_ankle_roll
  [20] right_ankle_roll  [21] left_elbow_yaw  (=0)
  [22] right_elbow_yaw   (=0)
"""

import os
import pickle
import numpy as np
import argparse
import mujoco

# ─── G1 joint ordering (PHUMA, 0-indexed) ────────────────────────────────────
# left leg
G1_L_HIP_PITCH   = 0
G1_L_HIP_ROLL    = 1
G1_L_HIP_YAW     = 2
G1_L_KNEE        = 3
G1_L_ANKLE_PITCH = 4
G1_L_ANKLE_ROLL  = 5
# right leg
G1_R_HIP_PITCH   = 6
G1_R_HIP_ROLL    = 7
G1_R_HIP_YAW     = 8
G1_R_KNEE        = 9
G1_R_ANKLE_PITCH = 10
G1_R_ANKLE_ROLL  = 11
# waist
G1_WAIST_YAW     = 12
G1_WAIST_ROLL    = 13
G1_WAIST_PITCH   = 14
# left arm
G1_L_SHO_PITCH   = 15
G1_L_SHO_ROLL    = 16
G1_L_SHO_YAW     = 17
G1_L_ELBOW       = 18
# wrists 19,20,21 → dropped
# right arm
G1_R_SHO_PITCH   = 22
G1_R_SHO_ROLL    = 23
G1_R_SHO_YAW     = 24
G1_R_ELBOW       = 25
# wrists 26,27,28 → dropped

# ─── ATOM01 lab_dof_names order ──────────────────────────────────────────────
# lab_dof_names indices:
LAB_L_THIGH_YAW   = 0
LAB_R_THIGH_YAW   = 1
LAB_TORSO         = 2
LAB_L_THIGH_ROLL  = 3
LAB_R_THIGH_ROLL  = 4
LAB_L_ARM_PITCH   = 5
LAB_R_ARM_PITCH   = 6
LAB_L_THIGH_PITCH = 7
LAB_R_THIGH_PITCH = 8
LAB_L_ARM_ROLL    = 9
LAB_R_ARM_ROLL    = 10
LAB_L_KNEE        = 11
LAB_R_KNEE        = 12
LAB_L_ARM_YAW     = 13
LAB_R_ARM_YAW     = 14
LAB_L_ANKLE_PITCH = 15
LAB_R_ANKLE_PITCH = 16
LAB_L_ELBOW_PITCH = 17
LAB_R_ELBOW_PITCH = 18
LAB_L_ANKLE_ROLL  = 19
LAB_R_ANKLE_ROLL  = 20
LAB_L_ELBOW_YAW   = 21   # set to 0 (no G1 analog)
LAB_R_ELBOW_YAW   = 22   # set to 0 (no G1 analog)

NUM_ATOM01_DOF = 23

# Key body names for FK (must match lab robot model body names)
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link",
    "left_arm_roll_link",
    "right_arm_roll_link",
]

# ATOM01 MJCF path
ATOM01_MJCF = "/root/autodl-tmp/atom01_train/robolab/data/robots/roboparty/atom01/mjcf/atom01.xml"


def g1_to_atom01(g1_dof: np.ndarray) -> np.ndarray:
    """Map G1 29-DOF frame to ATOM01 23-DOF lab ordering.

    Args:
        g1_dof: (T, 29) array of G1 joint positions.
    Returns:
        (T, 23) array in ATOM01 lab_dof_names order.
    """
    T = g1_dof.shape[0]
    atom = np.zeros((T, NUM_ATOM01_DOF), dtype=np.float32)

    # Left leg
    atom[:, LAB_L_THIGH_YAW]   = g1_dof[:, G1_L_HIP_YAW]
    atom[:, LAB_L_THIGH_ROLL]  = g1_dof[:, G1_L_HIP_ROLL]
    atom[:, LAB_L_THIGH_PITCH] = g1_dof[:, G1_L_HIP_PITCH]
    atom[:, LAB_L_KNEE]        = g1_dof[:, G1_L_KNEE]
    atom[:, LAB_L_ANKLE_PITCH] = g1_dof[:, G1_L_ANKLE_PITCH]
    atom[:, LAB_L_ANKLE_ROLL]  = g1_dof[:, G1_L_ANKLE_ROLL]

    # Right leg
    atom[:, LAB_R_THIGH_YAW]   = g1_dof[:, G1_R_HIP_YAW]
    atom[:, LAB_R_THIGH_ROLL]  = g1_dof[:, G1_R_HIP_ROLL]
    atom[:, LAB_R_THIGH_PITCH] = g1_dof[:, G1_R_HIP_PITCH]
    atom[:, LAB_R_KNEE]        = g1_dof[:, G1_R_KNEE]
    atom[:, LAB_R_ANKLE_PITCH] = g1_dof[:, G1_R_ANKLE_PITCH]
    atom[:, LAB_R_ANKLE_ROLL]  = g1_dof[:, G1_R_ANKLE_ROLL]

    # Torso (waist_pitch is closest to torso_joint)
    atom[:, LAB_TORSO] = g1_dof[:, G1_WAIST_PITCH]

    # Left arm
    atom[:, LAB_L_ARM_PITCH]   = g1_dof[:, G1_L_SHO_PITCH]
    atom[:, LAB_L_ARM_ROLL]    = g1_dof[:, G1_L_SHO_ROLL]
    atom[:, LAB_L_ARM_YAW]     = g1_dof[:, G1_L_SHO_YAW]
    atom[:, LAB_L_ELBOW_PITCH] = g1_dof[:, G1_L_ELBOW]
    atom[:, LAB_L_ELBOW_YAW]   = 0.0  # no G1 analog

    # Right arm
    atom[:, LAB_R_ARM_PITCH]   = g1_dof[:, G1_R_SHO_PITCH]
    atom[:, LAB_R_ARM_ROLL]    = g1_dof[:, G1_R_SHO_ROLL]
    atom[:, LAB_R_ARM_YAW]     = g1_dof[:, G1_R_SHO_YAW]
    atom[:, LAB_R_ELBOW_PITCH] = g1_dof[:, G1_R_ELBOW]
    atom[:, LAB_R_ELBOW_YAW]   = 0.0  # no G1 analog

    return atom


def build_mujoco_model():
    """Load ATOM01 MuJoCo model and get key body indices."""
    model = mujoco.MjModel.from_xml_path(ATOM01_MJCF)
    data = mujoco.MjData(model)

    # Find key body indices
    key_body_ids = []
    for name in KEY_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in MJCF. Available bodies: "
                             f"{[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]}")
        key_body_ids.append(bid)

    # Get DOF name mapping: MJCF qpos ordering for actuated joints
    # In ATOM01 MJCF with floating base: qpos[0:7] = root (pos+quat), qpos[7:] = joint angles
    # We need to map lab_dof_names to MJCF qpos indices

    # Get joint qpos address for each joint
    lab_dof_names = [
        "left_thigh_yaw_joint", "right_thigh_yaw_joint", "torso_joint",
        "left_thigh_roll_joint", "right_thigh_roll_joint",
        "left_arm_pitch_joint", "right_arm_pitch_joint",
        "left_thigh_pitch_joint", "right_thigh_pitch_joint",
        "left_arm_roll_joint", "right_arm_roll_joint",
        "left_knee_joint", "right_knee_joint",
        "left_arm_yaw_joint", "right_arm_yaw_joint",
        "left_ankle_pitch_joint", "right_ankle_pitch_joint",
        "left_elbow_pitch_joint", "right_elbow_pitch_joint",
        "left_ankle_roll_joint", "right_ankle_roll_joint",
        "left_elbow_yaw_joint", "right_elbow_yaw_joint",
    ]

    lab_to_qpos = []
    for jname in lab_dof_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise ValueError(f"Joint '{jname}' not found in MJCF")
        qpos_adr = model.jnt_qposadr[jid]
        lab_to_qpos.append(qpos_adr)

    return model, data, key_body_ids, lab_to_qpos


def compute_key_body_pos(
    model, data, key_body_ids, lab_to_qpos,
    root_pos: np.ndarray, root_quat: np.ndarray, dof_pos: np.ndarray
) -> np.ndarray:
    """Compute key body world positions via MuJoCo FK.

    Args:
        root_pos:  (T, 3) root position
        root_quat: (T, 4) root quaternion (x,y,z,w) → MuJoCo uses (w,x,y,z)
        dof_pos:   (T, 23) joint positions in lab_dof_names order
    Returns:
        key_body_pos_w: (T, 6, 3) world positions of key bodies
    """
    T = root_pos.shape[0]
    N = len(key_body_ids)
    key_body_pos_w = np.zeros((T, N, 3), dtype=np.float32)

    for t in range(T):
        # Set root state (MuJoCo qpos: [x,y,z, qw,qx,qy,qz] for free joint)
        data.qpos[0:3] = root_pos[t]
        xyzw = root_quat[t]  # PHUMA format: (x,y,z,w)
        data.qpos[3:7] = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]  # → (w,x,y,z)

        # Set joint positions
        for lab_idx, qpos_adr in enumerate(lab_to_qpos):
            data.qpos[qpos_adr] = dof_pos[t, lab_idx]

        # Forward kinematics
        mujoco.mj_kinematics(model, data)

        # Extract key body positions
        for i, bid in enumerate(key_body_ids):
            key_body_pos_w[t, i] = data.xpos[bid]

    return key_body_pos_w


def convert_file(
    npy_path: str,
    out_dir: str,
    model, data, key_body_ids, lab_to_qpos,
    target_fps: float = 30.0,
) -> str:
    """Convert one PHUMA npy file to ATOM01 pkl format."""
    d = np.load(npy_path, allow_pickle=True).item()

    root_trans = d["root_trans"].astype(np.float32)  # (T, 3)
    root_ori   = d["root_ori"].astype(np.float32)    # (T, 4) xyzw
    g1_dof     = d["dof_pos"].astype(np.float32)     # (T, 29)
    src_fps    = float(d["fps"])

    # Resample to target_fps if needed
    T_src = root_trans.shape[0]
    if abs(src_fps - target_fps) > 0.5:
        T_dst = max(2, int(round(T_src * target_fps / src_fps)))
        idx = np.linspace(0, T_src - 1, T_dst)
        idx0 = np.floor(idx).astype(int)
        idx1 = np.minimum(idx0 + 1, T_src - 1)
        alpha = (idx - idx0).astype(np.float32)

        def lerp(arr):
            return arr[idx0] * (1 - alpha[:, None]) + arr[idx1] * alpha[:, None]

        root_trans = lerp(root_trans)
        root_ori   = lerp(root_ori)
        # Normalize quaternion after lerp
        root_ori  /= np.linalg.norm(root_ori, axis=-1, keepdims=True) + 1e-8
        g1_dof     = lerp(g1_dof)

    # Map G1 → ATOM01 dof
    atom_dof = g1_to_atom01(g1_dof)

    # Compute key body positions via FK
    key_body_pos_w = compute_key_body_pos(
        model, data, key_body_ids, lab_to_qpos,
        root_trans, root_ori, atom_dof
    )

    # Build pkl
    out = {
        "fps":          target_fps,
        "root_pos":     root_trans,
        "root_rot":     root_ori,   # (x,y,z,w) same as existing ATOM01 pkl
        "dof_pos":      atom_dof,
        "loop_mode":    0,           # wrap (most walking clips loop)
        "key_body_pos": key_body_pos_w,
    }

    stem = os.path.splitext(os.path.basename(npy_path))[0]
    out_path = os.path.join(out_dir, stem + ".pkl")
    with open(out_path, "wb") as fp:
        pickle.dump(out, fp)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert PHUMA G1 → ATOM01 pkl")
    parser.add_argument("--input_dir",  required=True,  help="PHUMA G1 category dir")
    parser.add_argument("--output_dir", required=True,  help="Output pkl dir")
    parser.add_argument("--max_files",  type=int, default=0, help="Max files (0=all)")
    parser.add_argument("--fps",        type=float, default=30.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[ATOM01 FK] Loading MuJoCo model...")
    model, data, key_body_ids, lab_to_qpos = build_mujoco_model()
    print(f"  Key bodies: {KEY_BODY_NAMES}")
    print(f"  Model nq={model.nq}, nv={model.nv}")

    # Collect all npy files
    npy_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in sorted(files):
            if f.endswith(".npy"):
                npy_files.append(os.path.join(root, f))

    if args.max_files > 0:
        npy_files = npy_files[:args.max_files]

    print(f"[Convert] {len(npy_files)} files → {args.output_dir}")

    ok, fail = 0, 0
    for i, npy_path in enumerate(npy_files):
        try:
            out = convert_file(npy_path, args.output_dir, model, data, key_body_ids, lab_to_qpos, args.fps)
            ok += 1
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i+1}/{len(npy_files)}] {os.path.basename(out)}")
        except Exception as e:
            fail += 1
            print(f"  [SKIP] {os.path.basename(npy_path)}: {e}")

    print(f"\nDone: {ok} converted, {fail} failed → {args.output_dir}")


if __name__ == "__main__":
    main()
