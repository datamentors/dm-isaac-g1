#!/usr/bin/env python3
"""
MuJoCo Closed-Loop Evaluation for GROOT G1 Towel Folding — with WBC
====================================================================

Like run_mujoco_towel_eval.py, but integrates the NVIDIA GR00T
Whole-Body-Control (WBC) ONNX policies for lower body balance/locomotion.

Architecture:
  GROOT server (Spark) → upper body targets (arms, hands, waist, navigate, height)
  WBC ONNX policies    → lower body control (legs + waist) for balance/walking

  The WBC ONNX policies (Balance + Walk) run at 50 Hz (200 Hz sim / 4 decimation).
  GROOT is queried every N WBC steps (controlled by --groot-query-interval).

  WBC controls 15 joints: 6 left leg + 6 right leg + 3 waist
  GROOT controls 14 arm joints + grippers, and provides navigate/height commands

Requirements:
    pip install mujoco>=3.2.6 pyzmq numpy opencv-python onnxruntime pyyaml

Usage:
    # Start GROOT server on Spark with --use-sim-policy-wrapper:
    # (inside inference container)
    python gr00t/eval/run_gr00t_server.py \
        --model-path /workspace/checkpoints/groot-g1-gripper-hospitality-7ds \
        --embodiment-tag UNITREE_G1 --use-sim-policy-wrapper --port 5555

    # Then run this eval from workstation container:
    python run_mujoco_towel_eval_wbc.py \
        --scene mujoco_towel_scene/g1_towel_folding.xml \
        --wbc-dir /workspace/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1 \
        --host 192.168.1.237 --port 5555 \
        --max-steps 1500
"""

import argparse
import os
import re
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


# =============================================================================
# Constants: UNITREE_G1 joint layout
# =============================================================================

# State vector layout (31 DOF)
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

# WBC joint ordering: first 15 actuated joints in the WBC XML
# [0:6]  = left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
# [6:12] = right leg (same order)
# [12:15] = waist (yaw, roll, pitch)
WBC_NUM_ACTIONS = 15

# Arm joints after WBC joints: indices 15-28 in the actuated order
# [15:22] = left arm, [22:29] = right arm
WBC_ARM_START = 15
WBC_ARM_COUNT = 14  # 7 per arm

# Joint names in the WBC G1 model (actuator order)
WBC_JOINT_NAMES = [
    # Left leg (0-5)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # Right leg (6-11)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (12-14)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Left arm (15-21)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Right arm (22-28)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# GROOT arm action order (training order — yaw,roll,pitch for wrists):
#   shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_yaw, wrist_roll, wrist_pitch
# WBC/Menagerie actuator order:
#   shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
# Remap: GROOT[4]=yaw→WBC[6], GROOT[5]=roll→WBC[4], GROOT[6]=pitch→WBC[5]
GROOT_TO_WBC_ARM = [0, 1, 2, 3, 6, 4, 5]  # GROOT index → WBC position within 7-DOF arm


# =============================================================================
# WBC Controller — based on NVIDIA's run_mujoco_gear_wbc.py
# =============================================================================

class WBCController:
    """Whole-Body-Control using ONNX Balance/Walk policies.

    Controls the lower body (15 DOF: legs + waist) of the Unitree G1.
    Based on NVIDIA's GR00T-WholeBodyControl standalone MuJoCo script.
    """

    def __init__(self, wbc_dir: str, device: str = "cpu"):
        import yaml
        import onnxruntime as ort

        self.wbc_dir = wbc_dir

        # Load config
        config_path = os.path.join(wbc_dir, "g1_gear_wbc.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.simulation_dt = cfg["simulation_dt"]        # 0.005
        self.control_decimation = cfg["control_decimation"]  # 4
        self.num_actions = cfg["num_actions"]             # 15
        self.num_obs = cfg["num_obs"]                     # 516
        self.obs_history_len = cfg["obs_history_len"]     # 6
        self.action_scale = cfg["action_scale"]           # 0.25
        self.cmd_scale = np.array(cfg["cmd_scale"])       # [2.0, 2.0, 0.5]
        self.ang_vel_scale = cfg["ang_vel_scale"]         # 0.5
        self.dof_pos_scale = cfg["dof_pos_scale"]         # 1.0
        self.dof_vel_scale = cfg["dof_vel_scale"]         # 0.05

        self.default_angles = np.array(cfg["default_angles"], dtype=np.float32)  # 15
        self.kps = np.array(cfg["kps"], dtype=np.float32)  # 15
        self.kds = np.array(cfg["kds"], dtype=np.float32)  # 15

        self.height_cmd = cfg.get("height_cmd", 0.74)
        self.rpy_cmd = np.array(cfg.get("rpy_cmd", [0.0, 0.0, 0.0]), dtype=np.float32)

        # Load ONNX policies
        balance_path = os.path.join(wbc_dir, "policy", "GR00T-WholeBodyControl-Balance.onnx")
        walk_path = os.path.join(wbc_dir, "policy", "GR00T-WholeBodyControl-Walk.onnx")

        providers = ['CPUExecutionProvider']
        if device != "cpu":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        print(f"  Loading WBC Balance policy: {balance_path}")
        self.balance_session = ort.InferenceSession(balance_path, providers=providers)
        print(f"  Loading WBC Walk policy: {walk_path}")
        self.walk_session = ort.InferenceSession(walk_path, providers=providers)

        # Observation history
        obs_dim = self.num_obs // self.obs_history_len  # 86
        self.obs_dim = obs_dim
        self.obs_history = deque(maxlen=self.obs_history_len)
        for _ in range(self.obs_history_len):
            self.obs_history.append(np.zeros(obs_dim, dtype=np.float32))

        # Last action for observation
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        # Locomotion command (set by GROOT's navigate_command)
        self.loco_cmd = np.zeros(3, dtype=np.float32)  # [vx, vy, vyaw]

        print(f"  WBC initialized: {self.num_actions} actions, obs_dim={obs_dim}, "
              f"history={self.obs_history_len}, total_obs={self.num_obs}")

    def _quat_rotate_inverse(self, q, v):
        """Rotate vector v by inverse of quaternion q. q = [w, x, y, z]."""
        w, x, y, z = q[0], q[1], q[2], q[3]
        # q_conj = [w, -x, -y, -z]
        # rotation: v' = q_conj * v * q
        t = 2.0 * np.cross(np.array([-x, -y, -z]), v)
        return v + w * t + np.cross(np.array([-x, -y, -z]), t)

    def _get_gravity_orientation(self, quat):
        """Project gravity [0,0,-1] into body frame using quaternion."""
        return self._quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

    def compute_observation(self, q_body, dq_body, base_quat, base_ang_vel):
        """Build single-step 86-dim observation for WBC policy.

        Args:
            q_body: (29,) joint positions (all actuated joints, excluding freejoint 7 DOF)
            dq_body: (29,) joint velocities
            base_quat: (4,) pelvis quaternion [w, x, y, z]
            base_ang_vel: (3,) pelvis angular velocity in body frame
        """
        n_joints = 29  # all actuated joints in WBC model

        # Command vector (7D): loco_cmd * cmd_scale + height + rpy
        cmd = np.zeros(7, dtype=np.float32)
        cmd[0:3] = self.loco_cmd * self.cmd_scale
        cmd[3] = self.height_cmd
        cmd[4:7] = self.rpy_cmd

        # Angular velocity scaled
        omega = base_ang_vel.astype(np.float32) * self.ang_vel_scale

        # Gravity orientation in body frame
        gravity = self._get_gravity_orientation(base_quat).astype(np.float32)

        # Joint positions relative to defaults (all 29 joints)
        # WBC defaults only cover 15 joints (legs + waist), pad with zeros for arms
        defaults_full = np.zeros(n_joints, dtype=np.float32)
        defaults_full[:self.num_actions] = self.default_angles
        qj = (q_body.astype(np.float32) - defaults_full) * self.dof_pos_scale

        # Joint velocities
        dqj = dq_body.astype(np.float32) * self.dof_vel_scale

        # Last action (15 lower body)
        obs_single = np.concatenate([
            cmd,          # 7
            omega,        # 3
            gravity,      # 3
            qj,           # 29
            dqj,          # 29
            self.last_action,  # 15
        ]).astype(np.float32)
        # Total: 7 + 3 + 3 + 29 + 29 + 15 = 86 ✓

        self.obs_history.append(obs_single)

        # Concatenate history
        obs_full = np.concatenate(list(self.obs_history))
        return obs_full

    def get_action(self, obs):
        """Run ONNX inference to get lower body joint targets.

        Returns:
            target_q: (15,) target joint positions for legs + waist
        """
        obs_tensor = obs.reshape(1, -1).astype(np.float32)

        # Select policy based on locomotion command magnitude
        if np.linalg.norm(self.loco_cmd) <= 0.05:
            session = self.balance_session
        else:
            session = self.walk_session

        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: obs_tensor})
        action = output[0][0]  # (15,)

        self.last_action = action.copy()

        # Convert to joint position targets
        target_q = action * self.action_scale + self.default_angles
        return target_q

    def pd_control(self, target_q, q_current, dq_current):
        """PD torque control for lower body joints.

        Args:
            target_q: (15,) target positions
            q_current: (15,) current positions
            dq_current: (15,) current velocities
        Returns:
            tau: (15,) torques
        """
        return self.kps * (target_q - q_current) + self.kds * (0.0 - dq_current)


# =============================================================================
# Scene loading — modified for WBC (keeps freejoint, no base lock)
# =============================================================================

def load_towel_scene_wbc(scene_path: str,
                         menagerie_dir: str = "/workspace/mujoco_menagerie/unitree_g1",
                         wbc_timestep: float = 0.005) -> mujoco.MjModel:
    """Load G1 + towel scene configured for WBC (freejoint enabled).

    Unlike the non-WBC version, this KEEPS the freejoint for pelvis so the
    WBC balance controller can stabilize the robot. The timestep is set to
    match WBC's simulation_dt (0.005s = 200 Hz).
    """
    with open(scene_path) as f:
        scene_xml = f.read()

    # Detect G1 model variant
    g1_filename = "g1.xml"
    if "g1_with_hands.xml" in scene_xml:
        g1_filename = "g1_with_hands.xml"
    g1_path = os.path.join(menagerie_dir, g1_filename)
    with open(g1_path) as f:
        g1_xml = f.read()

    # Strip keyframe (qpos size mismatch with flexcomp towel)
    g1_xml_mod = re.sub(r'<keyframe>.*?</keyframe>', '', g1_xml, flags=re.DOTALL)

    # CRITICAL: Convert Menagerie position-servo actuators to direct-torque actuators.
    # Menagerie uses: <general gainprm="500 0 0" biasprm="0 -500 -43" biastype="affine" ...>
    # WBC expects:    <motor gear="1" ...>  (direct torque, gain=1, no bias)
    #
    # We convert by replacing the <actuator> section entirely with torque-mode motors.
    # This matches the WBC G1 model (g1_gear_wbc.xml) actuator configuration.
    actuator_replacement = """<actuator>
    <!-- Direct torque actuators for WBC (converted from Menagerie position-servo) -->
    <motor name="left_hip_pitch" joint="left_hip_pitch_joint" gear="1"/>
    <motor name="left_hip_roll" joint="left_hip_roll_joint" gear="1"/>
    <motor name="left_hip_yaw" joint="left_hip_yaw_joint" gear="1"/>
    <motor name="left_knee" joint="left_knee_joint" gear="1"/>
    <motor name="left_ankle_pitch" joint="left_ankle_pitch_joint" gear="1"/>
    <motor name="left_ankle_roll" joint="left_ankle_roll_joint" gear="1"/>
    <motor name="right_hip_pitch" joint="right_hip_pitch_joint" gear="1"/>
    <motor name="right_hip_roll" joint="right_hip_roll_joint" gear="1"/>
    <motor name="right_hip_yaw" joint="right_hip_yaw_joint" gear="1"/>
    <motor name="right_knee" joint="right_knee_joint" gear="1"/>
    <motor name="right_ankle_pitch" joint="right_ankle_pitch_joint" gear="1"/>
    <motor name="right_ankle_roll" joint="right_ankle_roll_joint" gear="1"/>
    <motor name="waist_yaw" joint="waist_yaw_joint" gear="1"/>
    <motor name="waist_roll" joint="waist_roll_joint" gear="1"/>
    <motor name="waist_pitch" joint="waist_pitch_joint" gear="1"/>
    <motor name="left_shoulder_pitch" joint="left_shoulder_pitch_joint" gear="1"/>
    <motor name="left_shoulder_roll" joint="left_shoulder_roll_joint" gear="1"/>
    <motor name="left_shoulder_yaw" joint="left_shoulder_yaw_joint" gear="1"/>
    <motor name="left_elbow" joint="left_elbow_joint" gear="1"/>
    <motor name="left_wrist_roll" joint="left_wrist_roll_joint" gear="1"/>
    <motor name="left_wrist_pitch" joint="left_wrist_pitch_joint" gear="1"/>
    <motor name="left_wrist_yaw" joint="left_wrist_yaw_joint" gear="1"/>
    <motor name="right_shoulder_pitch" joint="right_shoulder_pitch_joint" gear="1"/>
    <motor name="right_shoulder_roll" joint="right_shoulder_roll_joint" gear="1"/>
    <motor name="right_shoulder_yaw" joint="right_shoulder_yaw_joint" gear="1"/>
    <motor name="right_elbow" joint="right_elbow_joint" gear="1"/>
    <motor name="right_wrist_roll" joint="right_wrist_roll_joint" gear="1"/>
    <motor name="right_wrist_pitch" joint="right_wrist_pitch_joint" gear="1"/>
    <motor name="right_wrist_yaw" joint="right_wrist_yaw_joint" gear="1"/>
  </actuator>"""
    g1_xml_mod = re.sub(
        r'<actuator>.*?</actuator>', actuator_replacement,
        g1_xml_mod, flags=re.DOTALL
    )
    print("  Converted actuators from position-servo to direct-torque mode for WBC")

    # Keep the freejoint! WBC will handle balance.
    # Set initial pelvis height to match WBC default (0.793m)
    # and rotate to face +Y (toward the table)
    g1_xml_mod = re.sub(
        r'(<body\s+name="pelvis"\s+pos=")[^"]*(")',
        r'\g<1>0 0 0.793\2',
        g1_xml_mod
    )
    # Add orientation: face +Y (90° around Z)
    g1_xml_mod = re.sub(
        r'(<body\s+name="pelvis"\s+pos="[^"]*")',
        r'\1 quat="0.7071 0 0 0.7071"',
        g1_xml_mod
    )
    print("  Kept freejoint for WBC balance, set pelvis height=0.793, facing +Y")

    # Inject ego_view camera into torso_link
    ego_camera_xml = (
        '\n            <!-- D435 ego_view camera -->\n'
        '            <camera name="ego_view" pos="0.0576 0.0175 0.4299"'
        ' xyaxes="0 -1 0 0.866 0 0.500" fovy="60"/>\n'
    )
    torso_pattern = r'(<body\s+name="torso_link"[^>]*>)'
    match = re.search(torso_pattern, g1_xml_mod)
    if match:
        insert_pos = match.end()
        g1_xml_mod = g1_xml_mod[:insert_pos] + ego_camera_xml + g1_xml_mod[insert_pos:]
        print("  Injected ego_view camera into torso_link")

    # Override scene timestep to match WBC (0.005s)
    scene_xml_mod = re.sub(
        r'timestep="[^"]*"',
        f'timestep="{wbc_timestep}"',
        scene_xml
    )
    # Also use Newton solver for stability (important for WBC balance)
    if 'solver=' not in scene_xml_mod:
        scene_xml_mod = scene_xml_mod.replace(
            f'timestep="{wbc_timestep}"',
            f'timestep="{wbc_timestep}" solver="Newton"'
        )

    # Write modified G1 XML
    g1_tmp = os.path.join(menagerie_dir, f"_{g1_filename.replace('.xml', '_wbc.xml')}")
    with open(g1_tmp, 'w') as f:
        f.write(g1_xml_mod)

    # Update scene reference
    scene_xml_mod = scene_xml_mod.replace(
        f"{menagerie_dir}/{g1_filename}", g1_tmp
    )

    # Write scene
    scene_tmp = os.path.join(menagerie_dir, "_g1_towel_wbc_scene.xml")
    with open(scene_tmp, 'w') as f:
        f.write(scene_xml_mod)

    try:
        model = mujoco.MjModel.from_xml_path(scene_tmp)
        print(f"  Scene loaded: {model.nq} qpos, {model.nv} dof, "
              f"{model.nu} actuators, {model.nflex} flex bodies")
        print(f"  Timestep: {model.opt.timestep}s ({1/model.opt.timestep:.0f} Hz)")
        return model
    finally:
        for tmp in [g1_tmp, scene_tmp]:
            if os.path.exists(tmp):
                os.remove(tmp)


# =============================================================================
# Joint mapping helpers
# =============================================================================

def build_joint_mapping(model: mujoco.MjModel) -> dict:
    """Build mapping from joint names to MuJoCo IDs and actuator IDs.

    Returns dict: {joint_name: {"joint_id": int, "qpos_addr": int, "qvel_addr": int,
                                 "act_id": int or None}}
    """
    mapping = {}
    for jname in WBC_JOINT_NAMES:
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            entry = {
                "joint_id": jid,
                "qpos_addr": model.jnt_qposadr[jid],
                "qvel_addr": model.jnt_dofadr[jid],
                "act_id": None,
            }
            # Find actuator for this joint
            for aid in range(model.nu):
                if model.actuator_trnid[aid, 0] == jid:
                    entry["act_id"] = aid
                    break
            mapping[jname] = entry
        except Exception:
            pass
    return mapping


def get_body_joint_state(model, data, joint_mapping):
    """Get all 29 body joint positions and velocities.

    Returns:
        q: (29,) joint positions in WBC joint order
        dq: (29,) joint velocities in WBC joint order
    """
    q = np.zeros(29, dtype=np.float32)
    dq = np.zeros(29, dtype=np.float32)
    for i, jname in enumerate(WBC_JOINT_NAMES):
        if jname in joint_mapping:
            m = joint_mapping[jname]
            q[i] = data.qpos[m["qpos_addr"]]
            dq[i] = data.qvel[m["qvel_addr"]]
    return q, dq


def get_base_state(model, data):
    """Get pelvis (floating base) state.

    Returns:
        quat: (4,) quaternion [w, x, y, z]
        ang_vel: (3,) angular velocity in body frame
    """
    # Freejoint: qpos[0:3] = pos, qpos[3:7] = quat [w,x,y,z]
    # Find pelvis body
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        # Fallback: first 7 qpos are the freejoint
        quat = data.qpos[3:7].copy()
        ang_vel = data.qvel[3:6].copy()
    else:
        # Get from body xquat (world frame)
        quat = data.xquat[pelvis_id].copy()  # [w, x, y, z]
        # Angular velocity from the freejoint dof
        # Freejoint: qvel[0:3] = linear vel, qvel[3:6] = angular vel
        ang_vel = data.qvel[3:6].copy()

    return quat.astype(np.float32), ang_vel.astype(np.float32)


def get_groot_state_vector(model, data, joint_mapping):
    """Extract 31-DOF GROOT state vector from simulation.

    Matches the UNITREE_G1 state layout expected by GROOT:
    [left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7),
     left_hand(1), right_hand(1)]

    IMPORTANT: Arm wrists must be remapped from WBC/Menagerie order
    (roll, pitch, yaw) to GROOT training order (yaw, roll, pitch).
    """
    q, _ = get_body_joint_state(model, data, joint_mapping)

    state = np.zeros(STATE_DOF, dtype=np.float32)

    # Left leg: WBC[0:6] → state[0:6]
    state[0:6] = q[0:6]
    # Right leg: WBC[6:12] → state[6:12]
    state[6:12] = q[6:12]
    # Waist: WBC[12:15] → state[12:15]
    state[12:15] = q[12:15]

    # Left arm: WBC[15:22] → state[15:22] with wrist remap
    # WBC order: sh_p, sh_r, sh_y, elbow, wr_roll, wr_pitch, wr_yaw
    # GROOT order: sh_p, sh_r, sh_y, elbow, wr_yaw, wr_roll, wr_pitch
    state[15] = q[15]  # shoulder_pitch
    state[16] = q[16]  # shoulder_roll
    state[17] = q[17]  # shoulder_yaw
    state[18] = q[18]  # elbow
    state[19] = q[21]  # wrist_yaw (WBC[21]) → GROOT state[19]
    state[20] = q[19]  # wrist_roll (WBC[19]) → GROOT state[20]
    state[21] = q[20]  # wrist_pitch (WBC[20]) → GROOT state[21]

    # Right arm: WBC[22:29] → state[22:29] with wrist remap
    state[22] = q[22]  # shoulder_pitch
    state[23] = q[23]  # shoulder_roll
    state[24] = q[24]  # shoulder_yaw
    state[25] = q[25]  # elbow
    state[26] = q[28]  # wrist_yaw (WBC[28]) → GROOT state[26]
    state[27] = q[26]  # wrist_roll (WBC[26]) → GROOT state[27]
    state[28] = q[27]  # wrist_pitch (WBC[27]) → GROOT state[28]

    # Hands: 0 (gripper model has no finger joints in WBC XML)
    state[29] = 0.0
    state[30] = 0.0

    return state


# =============================================================================
# GROOT observation / action helpers
# =============================================================================

def build_groot_observation(image, state, language):
    """Build flat-key observation dict for Gr00tSimPolicyWrapper."""
    B, T = 1, 1

    def _bt(arr):
        return arr.astype(np.float32).reshape(B, T, -1)

    return {
        "video.ego_view": image[np.newaxis, np.newaxis, ...].astype(np.uint8),
        "state.left_leg":   _bt(state[0:6]),
        "state.right_leg":  _bt(state[6:12]),
        "state.waist":      _bt(state[12:15]),
        "state.left_arm":   _bt(state[15:22]),
        "state.right_arm":  _bt(state[22:29]),
        "state.left_hand":  _bt(state[29:30]),
        "state.right_hand": _bt(state[30:31]),
        "annotation.human.task_description": (language,),
    }


def decode_groot_actions(action_dict):
    """Decode flat action dict from server into per-group arrays.

    Returns dict of {name: (T, D)} arrays.
    """
    result = {}
    for key, arr in action_dict.items():
        name = key.replace("action.", "") if key.startswith("action.") else key
        arr = np.array(arr)
        if arr.ndim == 3:
            arr = arr[0]  # Remove batch dim
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        result[name] = arr
    return result


def apply_groot_arms_to_actuators(model, data, action_step, joint_mapping):
    """Apply GROOT arm actions to MuJoCo actuators.

    GROOT arm actions are ABSOLUTE targets in training order (yaw, roll, pitch for wrists).
    We remap to WBC/Menagerie actuator order (roll, pitch, yaw).
    """
    arm_kp = 100.0  # PD gain for arm position control
    arm_kd = 2.0

    for side in ["left", "right"]:
        key = f"{side}_arm"
        if key not in action_step:
            continue
        groot_arm = action_step[key]  # (7,) in GROOT training order

        # GROOT order: sh_p, sh_r, sh_y, elbow, wr_yaw, wr_roll, wr_pitch
        # WBC/actuator names: sh_p, sh_r, sh_y, elbow, wr_roll, wr_pitch, wr_yaw
        wbc_arm_names = [
            f"{side}_shoulder_pitch_joint",
            f"{side}_shoulder_roll_joint",
            f"{side}_shoulder_yaw_joint",
            f"{side}_elbow_joint",
            f"{side}_wrist_roll_joint",   # GROOT[5]
            f"{side}_wrist_pitch_joint",  # GROOT[6]
            f"{side}_wrist_yaw_joint",    # GROOT[4]
        ]
        # Remap: [GROOT idx] → wbc position
        groot_to_act = [0, 1, 2, 3, 5, 6, 4]  # wbc_arm[i] = groot_arm[groot_to_act[i]]

        for i, jname in enumerate(wbc_arm_names):
            if jname not in joint_mapping:
                continue
            m = joint_mapping[jname]
            if m["act_id"] is None:
                continue
            target = float(groot_arm[groot_to_act[i]])
            # Use position control (the actuator is in torque mode, so apply PD)
            q_curr = data.qpos[m["qpos_addr"]]
            dq_curr = data.qvel[m["qvel_addr"]]
            tau = arm_kp * (target - q_curr) + arm_kd * (0.0 - dq_curr)
            data.ctrl[m["act_id"]] = tau


def apply_wbc_torques(model, data, target_q, joint_mapping, wbc_ctrl):
    """Apply WBC lower body torques via PD control.

    Args:
        target_q: (15,) target joint positions from WBC ONNX policy
        joint_mapping: joint name → MuJoCo info dict
        wbc_ctrl: WBCController for PD gains
    """
    # Lower body joint names (first 15 in WBC order)
    lower_body_names = WBC_JOINT_NAMES[:15]

    q_curr = np.zeros(15, dtype=np.float32)
    dq_curr = np.zeros(15, dtype=np.float32)
    for i, jname in enumerate(lower_body_names):
        if jname in joint_mapping:
            m = joint_mapping[jname]
            q_curr[i] = data.qpos[m["qpos_addr"]]
            dq_curr[i] = data.qvel[m["qvel_addr"]]

    # PD torques
    tau = wbc_ctrl.pd_control(target_q, q_curr, dq_curr)

    # Apply to actuators
    for i, jname in enumerate(lower_body_names):
        if jname in joint_mapping:
            m = joint_mapping[jname]
            if m["act_id"] is not None:
                data.ctrl[m["act_id"]] = float(tau[i])


# =============================================================================
# Rendering
# =============================================================================

def render_ego_view(model, data, renderer, camera_name="ego_view",
                    width=224, height=224):
    """Render ego-view camera image for GROOT observation."""
    renderer.update_scene(data, camera=camera_name)
    img = renderer.render()
    if img.shape[0] != height or img.shape[1] != width:
        import cv2
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    return img


def save_video(frames, path, fps=25):
    """Save frames as video."""
    try:
        import cv2
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  Video saved: {path} ({len(frames)} frames)")
    except ImportError:
        print("  WARNING: opencv-python not installed, skipping video save")


# =============================================================================
# Main evaluation loop
# =============================================================================

def run_episode(model, data, renderer, policy_client, joint_mapping,
                wbc_ctrl, language, max_steps, action_horizon,
                groot_query_interval=4, render_video=True):
    """Run one evaluation episode with WBC + GROOT.

    The WBC policy runs at 50 Hz (every control_decimation sim steps).
    GROOT is queried every groot_query_interval WBC steps (default=4 → 12.5 Hz).
    """
    frames = []
    states = []

    mujoco.mj_resetData(model, data)

    # Set initial standing pose for lower body (WBC defaults)
    q_body, _ = get_body_joint_state(model, data, joint_mapping)
    for i, jname in enumerate(WBC_JOINT_NAMES[:15]):
        if jname in joint_mapping:
            m = joint_mapping[jname]
            data.qpos[m["qpos_addr"]] = wbc_ctrl.default_angles[i]

    mujoco.mj_forward(model, data)

    # Save debug ego view
    try:
        debug_img = render_ego_view(model, data, renderer)
        import cv2
        debug_path = os.path.join(
            os.environ.get("EVAL_OUTPUT_DIR", "/tmp/mujoco_towel_eval_wbc"),
            "debug_ego_view_step0.png"
        )
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        cv2.imwrite(debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        print(f"  Debug ego_view saved: {debug_path}")
    except Exception:
        pass

    # Reset WBC state
    wbc_ctrl.loco_cmd = np.zeros(3, dtype=np.float32)
    wbc_ctrl.last_action = np.zeros(wbc_ctrl.num_actions, dtype=np.float32)
    wbc_ctrl.obs_history.clear()
    for _ in range(wbc_ctrl.obs_history_len):
        wbc_ctrl.obs_history.append(np.zeros(wbc_ctrl.obs_dim, dtype=np.float32))

    # GROOT action state
    groot_actions = None  # dict of {name: (T, D)}
    groot_action_idx = 0
    groot_action_horizon_len = 0
    current_groot_step = {}  # Current per-group action for this WBC step

    wbc_step = 0  # Counter for WBC policy steps
    sim_step = 0  # Counter for raw sim steps
    control_decimation = wbc_ctrl.control_decimation  # 4

    # WBC lower body target
    wbc_target_q = wbc_ctrl.default_angles.copy()

    print(f"  WBC: sim_dt={model.opt.timestep}, decimation={control_decimation}, "
          f"policy_rate={1/(model.opt.timestep * control_decimation):.0f} Hz")
    print(f"  GROOT query every {groot_query_interval} WBC steps "
          f"({1/(model.opt.timestep * control_decimation * groot_query_interval):.1f} Hz)")

    while wbc_step < max_steps:
        # --- WBC policy step (every control_decimation sim steps) ---
        # Get body state for WBC observation
        q_body, dq_body = get_body_joint_state(model, data, joint_mapping)
        base_quat, base_ang_vel = get_base_state(model, data)

        # Compute WBC observation and get lower body action
        wbc_obs = wbc_ctrl.compute_observation(q_body, dq_body, base_quat, base_ang_vel)
        wbc_target_q = wbc_ctrl.get_action(wbc_obs)

        # --- GROOT query (every groot_query_interval WBC steps) ---
        if wbc_step % groot_query_interval == 0:
            groot_state = get_groot_state_vector(model, data, joint_mapping)
            states.append(groot_state.copy())

            if groot_actions is None or groot_action_idx >= groot_action_horizon_len:
                # Need new actions from GROOT
                image = render_ego_view(model, data, renderer)
                obs = build_groot_observation(image, groot_state, language)
                action_dict, info = policy_client.get_action(obs)
                groot_actions = decode_groot_actions(action_dict)
                groot_action_idx = 0

                first_key = next(iter(groot_actions))
                groot_action_horizon_len = min(
                    action_horizon, groot_actions[first_key].shape[0]
                )

                if wbc_step == 0:
                    print(f"  GROOT action groups: {list(groot_actions.keys())}")
                    for k, v in groot_actions.items():
                        print(f"    {k}: shape={v.shape}, "
                              f"range=[{v.min():.4f}, {v.max():.4f}]")

            # Get current action step from GROOT
            t = min(groot_action_idx, groot_action_horizon_len - 1)
            current_groot_step = {
                name: arr[t] for name, arr in groot_actions.items()
            }
            groot_action_idx += 1

            # Pass navigate/height commands from GROOT to WBC
            if "navigate_command" in current_groot_step:
                wbc_ctrl.loco_cmd = current_groot_step["navigate_command"].astype(np.float32)
            if "base_height_command" in current_groot_step:
                wbc_ctrl.height_cmd = float(current_groot_step["base_height_command"][0])

        # --- Apply actions for control_decimation sim steps ---
        for _ in range(control_decimation):
            # WBC lower body torques
            apply_wbc_torques(model, data, wbc_target_q, joint_mapping, wbc_ctrl)

            # GROOT upper body (arms)
            if current_groot_step:
                apply_groot_arms_to_actuators(model, data, current_groot_step, joint_mapping)

            # Step simulation
            mujoco.mj_step(model, data)
            sim_step += 1

        # Record video frame (every 4 WBC steps ≈ 12.5 fps at 50Hz WBC)
        if render_video and wbc_step % 4 == 0:
            renderer.update_scene(data, camera="overview")
            frame = renderer.render()
            frames.append(frame.copy())

        wbc_step += 1

    return {
        "frames": frames,
        "states": np.array(states) if states else np.array([]),
        "wbc_steps": wbc_step,
        "sim_steps": sim_step,
    }


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo GROOT G1 Towel Eval with WBC")
    parser.add_argument("--scene", type=str, required=True,
                        help="Path to MuJoCo scene XML")
    parser.add_argument("--wbc-dir", type=str, required=True,
                        help="Path to WBC G1 resources dir (contains g1_gear_wbc.yaml, policy/)")
    parser.add_argument("--host", type=str, default="192.168.1.237",
                        help="GROOT server host (default: Spark)")
    parser.add_argument("--port", type=int, default=5555,
                        help="GROOT server port")
    parser.add_argument("--language", type=str, default="fold the towel",
                        help="Task language instruction")
    parser.add_argument("--n-episodes", type=int, default=1,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=1500,
                        help="Max WBC policy steps per episode (at 50 Hz)")
    parser.add_argument("--action-horizon", type=int, default=20,
                        help="GROOT action horizon (steps per query)")
    parser.add_argument("--groot-query-interval", type=int, default=4,
                        help="Query GROOT every N WBC steps (4=12.5Hz)")
    parser.add_argument("--output-dir", type=str,
                        default="/tmp/mujoco_towel_eval_wbc",
                        help="Output directory")
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["EVAL_OUTPUT_DIR"] = str(output_dir)

    # Load scene with freejoint enabled
    print(f"Loading scene: {args.scene}")
    try:
        model = load_towel_scene_wbc(args.scene)
    except Exception as e:
        print(f"  Scene load failed: {e}")
        raise

    data = mujoco.MjData(model)

    # Build joint mapping
    joint_mapping = build_joint_mapping(model)
    print(f"  Mapped {len(joint_mapping)}/{len(WBC_JOINT_NAMES)} joints")
    if len(joint_mapping) < 20:
        print("  WARNING: Many joints unmapped!")
        for jname in WBC_JOINT_NAMES:
            if jname not in joint_mapping:
                print(f"    MISSING: {jname}")

    # Initialize WBC controller
    print(f"Initializing WBC controller from: {args.wbc_dir}")
    wbc_ctrl = WBCController(args.wbc_dir)

    # Create renderer
    renderer = mujoco.Renderer(model, height=args.render_height,
                                width=args.render_width)

    # Connect to GROOT server
    print(f"Connecting to GROOT server: {args.host}:{args.port}")
    from gr00t.policy.server_client import PolicyClient
    policy_client = PolicyClient(host=args.host, port=args.port)
    if policy_client.ping():
        print("  Server is alive!")
    else:
        print("  WARNING: Server did not respond to ping")

    # Run episodes
    print(f"\nRunning {args.n_episodes} episode(s), "
          f"max {args.max_steps} WBC steps each")
    print(f"Language: \"{args.language}\"\n")

    for ep in range(args.n_episodes):
        print(f"Episode {ep + 1}/{args.n_episodes}...")
        t0 = time.time()

        result = run_episode(
            model, data, renderer, policy_client, joint_mapping, wbc_ctrl,
            language=args.language,
            max_steps=args.max_steps,
            action_horizon=args.action_horizon,
            groot_query_interval=args.groot_query_interval,
            render_video=not args.no_video,
        )

        elapsed = time.time() - t0
        print(f"  WBC steps: {result['wbc_steps']}, "
              f"Sim steps: {result['sim_steps']}, Time: {elapsed:.1f}s")

        if not args.no_video and result["frames"]:
            video_path = output_dir / f"episode_{ep:03d}_wbc.mp4"
            save_video(result["frames"], video_path)

    print("\n" + "=" * 50)
    print("WBC Evaluation Complete")
    print("=" * 50)
    print(f"Output: {output_dir}")
    renderer.close()


if __name__ == "__main__":
    main()
