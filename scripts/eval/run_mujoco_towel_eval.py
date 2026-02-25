#!/usr/bin/env python3
"""
MuJoCo Closed-Loop Evaluation for GROOT G1 Towel Folding
=========================================================

Runs the GROOT policy in a MuJoCo scene with the G1 robot and a deformable towel.
Connects to the GROOT inference server (ZMQ) to get actions from the trained model.

This script:
  1. Loads a MuJoCo scene with G1 + table + towel
  2. At each step, captures ego_view camera image + joint state
  3. Sends observation to GROOT server (or loads model locally)
  4. Applies returned actions to the robot
  5. Records video and logs metrics

Usage:
    # Start GROOT server on Spark (192.168.1.237):
    python gr00t/eval/run_gr00t_server.py \
        --model-path /workspace/checkpoints/groot-g1-gripper-fold-towel-full \
        --embodiment-tag UNITREE_G1 --port 5555

    # Then run this eval from workstation container:
    python run_mujoco_towel_eval.py \
        --scene mujoco_towel_scene/g1_towel_folding.xml \
        --host 192.168.1.237 --port 5555 \
        --n-episodes 5 --max-steps 500

    # Or with local model (no server needed):
    python run_mujoco_towel_eval.py \
        --scene mujoco_towel_scene/g1_towel_folding.xml \
        --model-path /workspace/checkpoints/groot-g1-gripper-fold-towel-full \
        --n-episodes 5

Observation Format:
    Uses the GROOT Policy API nested dict format (same as real robot deployment).
    No --use-sim-policy-wrapper needed on the server.

    obs = {
        "video":    {"ego_view": np.uint8 (B,T,H,W,3)},
        "state":    {"left_leg": (B,T,6), "right_leg": (B,T,6), "waist": (B,T,3),
                     "left_arm": (B,T,7), "right_arm": (B,T,7),
                     "left_hand": (B,T,1), "right_hand": (B,T,1)},
        "language": {"annotation.human.task_description": [[str]]}
    }

Requirements:
    pip install mujoco>=3.2.6 pyzmq numpy opencv-python matplotlib
"""

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

# UNITREE_G1 joint layout (from g1_gripper_unitree.py)
# State: 31 DOF, Action: 23 DOF
STATE_GROUPS = {
    "left_leg":   (0,  6),
    "right_leg":  (6,  12),
    "waist":      (12, 15),
    "left_arm":   (15, 22),
    "right_arm":  (22, 29),
    "left_hand":  (29, 30),
    "right_hand": (30, 31),
}
STATE_DOF = 31

ACTION_GROUPS = {
    "left_arm":            (0,  7,  "RELATIVE"),
    "right_arm":           (7,  14, "RELATIVE"),
    "left_hand":           (14, 15, "ABSOLUTE"),
    "right_hand":          (15, 16, "ABSOLUTE"),
    "waist":               (16, 19, "ABSOLUTE"),
    "base_height_command": (19, 20, "ABSOLUTE"),
    "navigate_command":    (20, 23, "ABSOLUTE"),
}
ACTION_DOF = 23

# ---- Wrist joint index mapping (CRITICAL) ----
# GROOT training data uses: yaw, roll, pitch ordering for wrists
# MuJoCo Menagerie G1 has: roll, pitch, yaw ordering
#
# Training order (per arm): shoulder_pitch, shoulder_roll, shoulder_yaw,
#                           elbow, wrist_yaw, wrist_roll, wrist_pitch
# Menagerie order (per arm): shoulder_pitch, shoulder_roll, shoulder_yaw,
#                             elbow, wrist_roll, wrist_pitch, wrist_yaw
#
# We map Menagerie joints → training state indices, handling the reorder.
# Indices within each arm's 7-DOF block:
#   Training:  [0]=sh_pitch [1]=sh_roll [2]=sh_yaw [3]=elbow [4]=wr_yaw [5]=wr_roll [6]=wr_pitch
#   Menagerie: [0]=sh_pitch [1]=sh_roll [2]=sh_yaw [3]=elbow [4]=wr_roll [5]=wr_pitch [6]=wr_yaw


def build_joint_name_to_state_index(model: mujoco.MjModel) -> dict[str, int]:
    """Map MuJoCo joint names to UNITREE_G1 state vector indices.

    The UNITREE_G1 state layout expects joints in this order:
      left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7),
      left_hand(1), right_hand(1)

    IMPORTANT: Wrist joints are remapped from Menagerie order (roll, pitch, yaw)
    to training order (yaw, roll, pitch). This fixes the known sim-vs-training
    mismatch documented by the team.
    """
    # Maps: (mujoco_joint_name, training_state_index)
    # Wrists are in TRAINING order (yaw, roll, pitch) but MuJoCo names
    # are roll/pitch/yaw — so we map the MuJoCo name to the correct training index.
    expected_joints = [
        # Left leg (state indices 0-5)
        ("left_hip_pitch_joint",     0),
        ("left_hip_roll_joint",      1),
        ("left_hip_yaw_joint",       2),
        ("left_knee_joint",          3),
        ("left_ankle_pitch_joint",   4),
        ("left_ankle_roll_joint",    5),
        # Right leg (state indices 6-11)
        ("right_hip_pitch_joint",    6),
        ("right_hip_roll_joint",     7),
        ("right_hip_yaw_joint",      8),
        ("right_knee_joint",         9),
        ("right_ankle_pitch_joint",  10),
        ("right_ankle_roll_joint",   11),
        # Waist (state indices 12-14)
        ("waist_yaw_joint",          12),
        ("waist_roll_joint",         13),
        ("waist_pitch_joint",        14),
        # Left arm (state indices 15-21)
        ("left_shoulder_pitch_joint",  15),
        ("left_shoulder_roll_joint",   16),
        ("left_shoulder_yaw_joint",    17),
        ("left_elbow_joint",           18),
        # Left wrist: MuJoCo has roll(20), pitch(21), yaw(22) in joint list
        # Training expects: yaw=19, roll=20, pitch=21
        ("left_wrist_yaw_joint",       19),   # Menagerie joint[22] → state[19]
        ("left_wrist_roll_joint",      20),   # Menagerie joint[20] → state[20]
        ("left_wrist_pitch_joint",     21),   # Menagerie joint[21] → state[21]
        # Right arm (state indices 22-28)
        ("right_shoulder_pitch_joint", 22),
        ("right_shoulder_roll_joint",  23),
        ("right_shoulder_yaw_joint",   24),
        ("right_elbow_joint",          25),
        # Right wrist: same remapping
        ("right_wrist_yaw_joint",      26),   # Menagerie joint[29] → state[26]
        ("right_wrist_roll_joint",     27),   # Menagerie joint[27] → state[27]
        ("right_wrist_pitch_joint",    28),   # Menagerie joint[28] → state[28]
        # Hands (state indices 29-30) — gripper joints
        # For the no-hands model: unmapped, defaults to 0.
        # For the with_hands model: we compute an average finger curl below.
    ]

    mapping = {}
    for joint_name, state_idx in expected_joints:
        try:
            mj_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if mj_joint_id >= 0:
                mapping[joint_name] = (mj_joint_id, state_idx)
        except Exception:
            pass

    return mapping


# Finger joint names for with_hands model.
# Used to compute a single gripper value (average curl) for state[29]/state[30]
# and to map a single gripper command to all finger joints.
LEFT_HAND_FINGER_JOINTS = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
]
RIGHT_HAND_FINGER_JOINTS = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]


def _has_hand_joints(model: mujoco.MjModel) -> bool:
    """Check if model has dexterous hand joints."""
    try:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hand_thumb_0_joint")
        return jid >= 0
    except Exception:
        return False


def _get_finger_curl(model: mujoco.MjModel, data: mujoco.MjData,
                     finger_joints: list[str]) -> float:
    """Compute average normalized finger curl (0=open, 1=closed) from finger joints."""
    curls = []
    for jname in finger_joints:
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            qpos_addr = model.jnt_qposadr[jid]
            val = data.qpos[qpos_addr]
            # Normalize to [0, 1] using joint range
            lo = model.jnt_range[jid, 0]
            hi = model.jnt_range[jid, 1]
            if hi > lo:
                curls.append((val - lo) / (hi - lo))
        except Exception:
            pass
    return float(np.mean(curls)) if curls else 0.0


def _apply_gripper_to_fingers(model: mujoco.MjModel, data: mujoco.MjData,
                              gripper_val: float, finger_joints: list[str]):
    """Map a single gripper command (0=open, 1=closed) to all finger joint actuators.

    Linearly interpolates gripper_val across each finger joint's range.
    """
    for jname in finger_joints:
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            lo = model.jnt_range[jid, 0]
            hi = model.jnt_range[jid, 1]
            target = lo + gripper_val * (hi - lo)
            # Find actuator for this joint
            for act_id in range(model.nu):
                if model.actuator_trnid[act_id, 0] == jid:
                    data.ctrl[act_id] = target
                    break
        except Exception:
            pass


def get_state_vector(model: mujoco.MjModel, data: mujoco.MjData,
                     joint_mapping: dict) -> np.ndarray:
    """Extract 31-DOF state vector from MuJoCo simulation."""
    state = np.zeros(STATE_DOF, dtype=np.float32)

    for joint_name, (mj_joint_id, state_idx) in joint_mapping.items():
        qpos_addr = model.jnt_qposadr[mj_joint_id]
        state[state_idx] = data.qpos[qpos_addr]

    # For with_hands model: compute gripper state from finger joints
    if _has_hand_joints(model):
        state[29] = _get_finger_curl(model, data, LEFT_HAND_FINGER_JOINTS)
        state[30] = _get_finger_curl(model, data, RIGHT_HAND_FINGER_JOINTS)

    return state


def apply_actions(model: mujoco.MjModel, data: mujoco.MjData,
                  actions: np.ndarray, joint_mapping: dict,
                  current_state: np.ndarray):
    """Apply 23-DOF action vector to MuJoCo actuators.

    Arms (indices 0-13) are RELATIVE actions (delta from current).
    Everything else is ABSOLUTE.
    """
    if actions.shape[-1] != ACTION_DOF:
        print(f"WARNING: expected {ACTION_DOF} action dims, got {actions.shape[-1]}")
        return

    # Extract action components
    left_arm_delta = actions[0:7]    # RELATIVE
    right_arm_delta = actions[7:14]  # RELATIVE
    left_hand = actions[14]          # ABSOLUTE
    right_hand = actions[15]         # ABSOLUTE
    waist = actions[16:19]           # ABSOLUTE
    # base_height = actions[19]      # Not applied in fixed-base MuJoCo
    # navigate = actions[20:23]      # Not applied in fixed-base MuJoCo

    # Compute target positions for arms (current + delta)
    left_arm_target = current_state[15:22] + left_arm_delta
    right_arm_target = current_state[22:29] + right_arm_delta

    # Build target qpos for actuated joints
    # Map joint names to target positions
    targets = {}

    # Left arm joints — action indices [0-6] are in TRAINING order:
    # [0]=sh_pitch [1]=sh_roll [2]=sh_yaw [3]=elbow [4]=wr_yaw [5]=wr_roll [6]=wr_pitch
    targets["left_shoulder_pitch_joint"] = left_arm_target[0]
    targets["left_shoulder_roll_joint"] = left_arm_target[1]
    targets["left_shoulder_yaw_joint"] = left_arm_target[2]
    targets["left_elbow_joint"] = left_arm_target[3]
    targets["left_wrist_yaw_joint"] = left_arm_target[4]
    targets["left_wrist_roll_joint"] = left_arm_target[5]
    targets["left_wrist_pitch_joint"] = left_arm_target[6]

    # Right arm joints — same ordering
    targets["right_shoulder_pitch_joint"] = right_arm_target[0]
    targets["right_shoulder_roll_joint"] = right_arm_target[1]
    targets["right_shoulder_yaw_joint"] = right_arm_target[2]
    targets["right_elbow_joint"] = right_arm_target[3]
    targets["right_wrist_yaw_joint"] = right_arm_target[4]
    targets["right_wrist_roll_joint"] = right_arm_target[5]
    targets["right_wrist_pitch_joint"] = right_arm_target[6]

    # Waist
    targets["waist_yaw_joint"] = waist[0]
    targets["waist_roll_joint"] = waist[1]
    targets["waist_pitch_joint"] = waist[2]

    # Apply to MuJoCo actuators (position control)
    for jname, target_val in targets.items():
        if jname in joint_mapping:
            mj_joint_id, _ = joint_mapping[jname]
            # Find actuator for this joint
            for act_id in range(model.nu):
                if model.actuator_trnid[act_id, 0] == mj_joint_id:
                    data.ctrl[act_id] = target_val
                    break

    # Grippers — map single gripper value to all finger joints (with_hands model)
    if _has_hand_joints(model):
        _apply_gripper_to_fingers(model, data, left_hand, LEFT_HAND_FINGER_JOINTS)
        _apply_gripper_to_fingers(model, data, right_hand, RIGHT_HAND_FINGER_JOINTS)


def render_ego_view(model: mujoco.MjModel, data: mujoco.MjData,
                    renderer: mujoco.Renderer, camera_name: str = "ego_view",
                    width: int = 224, height: int = 224) -> np.ndarray:
    """Render ego-view camera image for GROOT observation."""
    renderer.update_scene(data, camera=camera_name)
    img = renderer.render()
    # Resize if needed (GROOT expects 224x224 by default)
    if img.shape[0] != height or img.shape[1] != width:
        import cv2
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    return img


def load_towel_scene(scene_path: str, menagerie_dir: str = "/workspace/mujoco_menagerie/unitree_g1") -> mujoco.MjModel:
    """Load the G1 + towel scene, handling keyframe/qpos size conflicts.

    The Menagerie G1 includes a 'stand' keyframe with 36 qpos. When we add
    a flexcomp towel, qpos grows to ~800+, making the keyframe invalid.
    We handle this by loading the scene XML, stripping conflicting keyframes
    from the included G1 model, then compiling.

    Also injects the ego_view camera into the torso_link body to match the
    real G1's Intel RealSense D435 camera position. This is critical for
    GROOT — it needs a head-mounted ego view, not a fixed world camera.
    """
    import re

    # Read our scene XML
    with open(scene_path) as f:
        scene_xml = f.read()

    # Detect which G1 model the scene uses (g1.xml or g1_with_hands.xml)
    g1_filename = "g1.xml"
    if "g1_with_hands.xml" in scene_xml:
        g1_filename = "g1_with_hands.xml"
    g1_path = os.path.join(menagerie_dir, g1_filename)
    with open(g1_path) as f:
        g1_xml = f.read()

    # Strip the keyframe section from G1 XML (it has fixed qpos size)
    g1_xml_no_keyframe = re.sub(
        r'<keyframe>.*?</keyframe>', '', g1_xml, flags=re.DOTALL
    )

    # Inject ego_view camera into torso_link body.
    # The real G1 has an Intel RealSense D435 mounted on the torso:
    #   - Position relative to torso_link: xyz="0.0576 0.0175 0.4299"
    #     (5.8cm forward, 1.75cm right, 43cm up from torso origin)
    #   - Pitch: ~60° downward to center the table/towel in the frame
    #
    # MuJoCo camera convention: looks along -Z in its local frame.
    # xyaxes = (camera_right_x camera_right_y camera_right_z
    #           camera_up_x camera_up_y camera_up_z)
    #
    # For a camera on the torso looking forward-down at pitch θ=60°:
    #   Camera right (X): (0, -1, 0) — torso -Y = robot's right side
    #   Camera up (Y): (sin(60°), 0, cos(60°)) = (0.866, 0, 0.500)
    #   Camera look (-Z): perpendicular, pointing forward and ~60° below horizontal
    ego_camera_xml = (
        '\n            <!-- D435 ego_view camera — matches real G1 head-mounted position -->\n'
        '            <camera name="ego_view" pos="0.0576 0.0175 0.4299"'
        ' xyaxes="0 -1 0 0.866 0 0.500" fovy="60"/>\n'
    )

    # Insert camera inside the torso_link body (after the opening tag or first child)
    # Look for <body name="torso_link"> and insert after the next line
    torso_pattern = r'(<body\s+name="torso_link"[^>]*>)'
    match = re.search(torso_pattern, g1_xml_no_keyframe)
    if match:
        insert_pos = match.end()
        g1_xml_no_keyframe = (
            g1_xml_no_keyframe[:insert_pos]
            + ego_camera_xml
            + g1_xml_no_keyframe[insert_pos:]
        )
        print("  Injected ego_view camera into torso_link body")
    else:
        print("  WARNING: Could not find torso_link body for camera injection")

    # Write a temporary combined model
    import tempfile
    # Write modified G1 XML
    g1_tmp = os.path.join(menagerie_dir, f"_{g1_filename.replace('.xml', '_no_keyframe.xml')}")
    with open(g1_tmp, 'w') as f:
        f.write(g1_xml_no_keyframe)

    # Update scene to reference the no-keyframe version
    scene_xml_fixed = scene_xml.replace(
        f"{menagerie_dir}/{g1_filename}",
        g1_tmp
    )

    # Write scene to temp location in menagerie dir (for mesh resolution)
    scene_tmp = os.path.join(menagerie_dir, "_g1_towel_scene.xml")
    with open(scene_tmp, 'w') as f:
        f.write(scene_xml_fixed)

    try:
        model = mujoco.MjModel.from_xml_path(scene_tmp)
        print(f"  Scene loaded: {model.nq} qpos, {model.nv} dof, "
              f"{model.nu} actuators, {model.nflex} flex bodies")
        return model
    finally:
        # Clean up temp files
        for tmp in [g1_tmp, scene_tmp]:
            if os.path.exists(tmp):
                os.remove(tmp)


def build_groot_observation(image: np.ndarray, state: np.ndarray,
                            language: str) -> dict:
    """Build observation dict in GROOT Policy API nested format.

    Uses the same format as real robot deployment (SO100, etc.) — no
    --use-sim-policy-wrapper needed on the server.

    GROOT expects per-group state arrays:
      video:    {"ego_view": (B,T,H,W,3) uint8}
      state:    {"left_leg": (B,T,6), "right_leg": (B,T,6), "waist": (B,T,3),
                 "left_arm": (B,T,7), "right_arm": (B,T,7),
                 "left_hand": (B,T,1), "right_hand": (B,T,1)}
      language: {"annotation.human.task_description": [["fold the towel"]]}
    """
    B, T = 1, 1

    # Video: (B, T, H, W, 3)
    image_bt = image[np.newaxis, np.newaxis, ...].astype(np.uint8)

    # State: split 31-DOF vector into per-group arrays (B, T, D)
    def _bt(arr):
        return arr.astype(np.float32).reshape(B, T, -1)

    state_dict = {
        "left_leg":   _bt(state[0:6]),
        "right_leg":  _bt(state[6:12]),
        "waist":      _bt(state[12:15]),
        "left_arm":   _bt(state[15:22]),
        "right_arm":  _bt(state[22:29]),
        "left_hand":  _bt(state[29:30]),
        "right_hand": _bt(state[30:31]),
    }

    # Language: (B, T) = [[str]]
    language_dict = {
        "annotation.human.task_description": [[language]],
    }

    return {
        "video": {"ego_view": image_bt},
        "state": state_dict,
        "language": language_dict,
    }


def decode_action_dict(action_dict: dict) -> np.ndarray:
    """Decode GROOT per-group action dict into a flat 23-DOF action array.

    Server returns actions as:
        {"left_arm": (B,T,7), "right_arm": (B,T,7), "left_hand": (B,T,1),
         "right_hand": (B,T,1), "waist": (B,T,3),
         "base_height_command": (B,T,1), "navigate_command": (B,T,3)}

    We concatenate into: (T, 23) matching ACTION_GROUPS order.
    """
    # Order must match ACTION_GROUPS
    group_order = [
        ("left_arm", 7),
        ("right_arm", 7),
        ("left_hand", 1),
        ("right_hand", 1),
        ("waist", 3),
        ("base_height_command", 1),
        ("navigate_command", 3),
    ]

    chunks = []
    horizon = None
    for name, expected_dim in group_order:
        arr = np.array(action_dict[name])  # (B, T, D)
        # Remove batch dim
        if arr.ndim == 3:
            arr = arr[0]  # (T, D)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)  # (1, D)
        if horizon is None:
            horizon = arr.shape[0]
        chunks.append(arr)

    return np.concatenate(chunks, axis=-1)  # (T, 23)


def run_episode(model, data, renderer, policy_client, joint_mapping,
                language, max_steps, action_horizon, render_video=True,
                viewer=None):
    """Run a single evaluation episode."""
    frames = []
    states = []
    actions_log = []

    mujoco.mj_resetData(model, data)
    # Set initial keyframe if available
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    # Orient the robot to face the table (+Y direction).
    # The Menagerie G1's default forward is +X. We rotate 90° around Z.
    # Freejoint qpos layout: [x, y, z, qw, qx, qy, qz]
    # The Menagerie model uses "floating_base_joint" (not "pelvis") for the freejoint.
    for jname in ["floating_base_joint", "pelvis"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0 and model.jnt_type[jid] == 0:  # 0 = free joint
            qpos_addr = model.jnt_qposadr[jid]
            # Orientation: 90° rotation around Z axis
            # quat = (cos(π/4), 0, 0, sin(π/4)) = (0.7071, 0, 0, 0.7071)
            data.qpos[qpos_addr + 3] = 0.7071068  # qw
            data.qpos[qpos_addr + 4] = 0.0        # qx
            data.qpos[qpos_addr + 5] = 0.0        # qy
            data.qpos[qpos_addr + 6] = 0.7071068  # qz
            print(f"  Rotated robot to face +Y via {jname} (qpos[{qpos_addr}:{qpos_addr+7}])")
            break

    mujoco.mj_forward(model, data)

    # Save initial ego_view frame for debugging (what GROOT sees at step 0)
    debug_img = render_ego_view(model, data, renderer)
    try:
        import cv2
        debug_path = os.path.join(
            os.environ.get("EVAL_OUTPUT_DIR", "/tmp/mujoco_towel_eval"),
            "debug_ego_view_step0.png"
        )
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        cv2.imwrite(debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        print(f"  Debug ego_view saved: {debug_path}")
    except Exception:
        pass

    step = 0
    action_buffer = None
    action_idx = 0

    while step < max_steps:
        current_state = get_state_vector(model, data, joint_mapping)
        states.append(current_state.copy())

        # Query policy when action buffer is exhausted
        if action_buffer is None or action_idx >= action_buffer.shape[0]:
            # Render ego view
            image = render_ego_view(model, data, renderer)
            obs = build_groot_observation(image, current_state, language)

            # Get actions from GROOT
            action_dict, info = policy_client.get_action(obs)

            # Decode per-group action dict into flat (T, 23) array
            if isinstance(action_dict, dict) and "left_arm" in action_dict:
                # Nested format (base server, no sim wrapper)
                action_buffer = decode_action_dict(action_dict)
            elif isinstance(action_dict, dict) and "action" in action_dict:
                # Flat format (sim wrapper active)
                action_buffer = np.array(action_dict["action"])
                if action_buffer.ndim == 3:
                    action_buffer = action_buffer[0]
            else:
                action_buffer = np.array(action_dict)
                if action_buffer.ndim == 3:
                    action_buffer = action_buffer[0]

            action_idx = 0

            if step == 0:
                print(f"  Action buffer shape: {action_buffer.shape} "
                      f"(horizon={action_buffer.shape[0]}, dim={action_buffer.shape[-1]})")

        # Apply current action from buffer
        action = action_buffer[min(action_idx, len(action_buffer) - 1)]
        apply_actions(model, data, action, joint_mapping, current_state)
        actions_log.append(action.copy())

        # Step simulation
        mujoco.mj_step(model, data)

        # Sync live viewer
        if viewer is not None:
            viewer.sync()

        # Record frame for video
        if render_video and step % 2 == 0:
            renderer.update_scene(data, camera="overview")
            frame = renderer.render()
            frames.append(frame.copy())

        action_idx += 1
        step += 1

    return {
        "frames": frames,
        "states": np.array(states),
        "actions": np.array(actions_log),
        "steps": step,
    }


def save_video(frames, path, fps=20):
    """Save frames as video."""
    try:
        import cv2
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  Video saved: {path}")
    except ImportError:
        print("  WARNING: opencv-python not installed, skipping video save")


def main():
    parser = argparse.ArgumentParser(description="MuJoCo GROOT G1 Towel Eval")
    parser.add_argument("--scene", type=str, required=True,
                        help="Path to MuJoCo scene XML")
    parser.add_argument("--host", type=str, default="192.168.1.237",
                        help="GROOT server host (default: Spark)")
    parser.add_argument("--port", type=int, default=5555,
                        help="GROOT server port")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local model path (bypasses server)")
    parser.add_argument("--language", type=str, default="fold the towel",
                        help="Task language instruction")
    parser.add_argument("--n-episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--action-horizon", type=int, default=16,
                        help="Steps per action chunk before re-querying")
    parser.add_argument("--output-dir", type=str, default="/tmp/mujoco_towel_eval",
                        help="Output directory for videos and logs")
    parser.add_argument("--render-width", type=int, default=640,
                        help="Render width for video output")
    parser.add_argument("--render-height", type=int, default=480,
                        help="Render height for video output")
    parser.add_argument("--no-video", action="store_true",
                        help="Disable video recording")
    parser.add_argument("--show", action="store_true",
                        help="Show live MuJoCo viewer (requires display/VNC)")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["EVAL_OUTPUT_DIR"] = str(output_dir)

    # Load MuJoCo scene
    print(f"Loading scene: {args.scene}")
    try:
        # Try direct load first (works if scene is in Menagerie dir)
        model = mujoco.MjModel.from_xml_path(args.scene)
        print(f"  Model: {model.nq} qpos, {model.nv} dof, {model.nu} actuators")
        print(f"  Flex bodies: {model.nflex}")
    except Exception as e:
        print(f"  Direct load failed ({e}), trying programmatic scene composition...")
        model = load_towel_scene(args.scene)
    data = mujoco.MjData(model)

    # Build joint mapping
    joint_mapping = build_joint_name_to_state_index(model)
    print(f"  Mapped {len(joint_mapping)}/{STATE_DOF} joints")

    if len(joint_mapping) < 20:
        print("  WARNING: Many joints unmapped. Check joint names in MJCF vs expected names.")
        print("  Expected joint names (first 10):")
        for jname in list(build_joint_name_to_state_index.__code__.co_consts)[:10]:
            if isinstance(jname, str) and "joint" in jname:
                print(f"    {jname}")

    # Create renderer
    renderer = mujoco.Renderer(model, height=args.render_height, width=args.render_width)

    # Connect to GROOT policy
    if args.model_path:
        print(f"Loading local model: {args.model_path}")
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        policy_client = Gr00tPolicy.from_pretrained(
            args.model_path,
            embodiment_tag="UNITREE_G1",
        )
    else:
        print(f"Connecting to GROOT server: {args.host}:{args.port}")
        from gr00t.policy.server_client import PolicyClient
        policy_client = PolicyClient(
            host=args.host,
            port=args.port,
            strict=False,
        )
        if policy_client.ping():
            print("  Server is alive!")
        else:
            print("  WARNING: Server did not respond to ping")

    # Launch live viewer if requested
    viewer = None
    if args.show:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
            print("  Live viewer launched (close window to stop)")
        except Exception as e:
            print(f"  WARNING: Could not launch viewer ({e}). Using headless mode.")

    # Run evaluation episodes
    print(f"\nRunning {args.n_episodes} episodes (max {args.max_steps} steps each)")
    print(f"Language: \"{args.language}\"")
    print()

    all_results = []
    for ep in range(args.n_episodes):
        print(f"Episode {ep + 1}/{args.n_episodes}...")
        t0 = time.time()

        result = run_episode(
            model, data, renderer, policy_client, joint_mapping,
            language=args.language,
            max_steps=args.max_steps,
            action_horizon=args.action_horizon,
            render_video=not args.no_video,
            viewer=viewer,
        )

        elapsed = time.time() - t0
        print(f"  Steps: {result['steps']}, Time: {elapsed:.1f}s")

        # Save video
        if not args.no_video and result["frames"]:
            video_path = output_dir / f"episode_{ep:03d}.mp4"
            save_video(result["frames"], video_path)

        all_results.append(result)

    # Summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Episodes: {args.n_episodes}")
    print(f"Scene: {args.scene}")
    print(f"Language: {args.language}")
    print(f"Output: {args.output_dir}")

    if not args.no_video:
        print(f"\nVideos saved to: {output_dir}/episode_*.mp4")

    renderer.close()
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
