#!/usr/bin/env python3
"""
MuJoCo Closed-Loop Evaluation for GROOT G1 Towel Folding — with WBC
====================================================================

Integrates the NVIDIA GR00T Whole-Body-Control (WBC) ONNX policies for
lower body balance, with GROOT controlling the upper body (arms + hands).

Architecture:
  GROOT server (Spark) → upper body targets (arms, hands, waist, navigate, height)
  WBC ONNX policies    → lower body control (legs + waist) for balance/walking

  The WBC ONNX policies (Balance + Walk) run at 50 Hz (200 Hz sim / 4 decimation).
  GROOT is queried every N WBC steps (controlled by --groot-query-interval).

  WBC controls 15 joints: 6 left leg + 6 right leg + 3 waist
  GROOT controls 14 arm joints + hands + navigate/height commands

Hand types (--hand-type):
  dex1    — Dex1 prismatic grippers (2 slide joints/hand, injected into g1.xml)
  inspire — Menagerie Inspire dexterous hands (7 revolute joints/hand, g1_with_hands.xml)
  none    — No hands (base g1.xml, hand state sent as training mean)

Requirements:
    pip install mujoco>=3.0.0 pyzmq numpy opencv-python onnxruntime pyyaml

Usage:
    python run_mujoco_towel_eval_wbc.py \
        --scene mujoco_towel_scene/g1_dex1_towel_folding.xml \
        --hand-type dex1 \
        --wbc-dir /workspace/.../robots/g1 \
        --host 192.168.1.237 --port 5555 \
        --max-steps 1500
"""

import argparse
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import mujoco
import mujoco.viewer
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

# Add project root to path so we can import dm_isaac_g1.core
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from dm_isaac_g1.core.robot_configs import (  # noqa: E402
    # Hand types and registry
    DEX1, DEX3, INSPIRE, HAND_TYPES as HAND_TYPE_REGISTRY,
    # Body definitions
    G1_BODY_JOINT_NAMES, G1_BODY_DOF,
    # GROOT layout
    GROOT_STATE_LAYOUT, GROOT_STATE_DOF, GROOT_TO_WBC_ARM_REMAP,
    WBC_JOINT_NAMES, WBC_NUM_ACTIONS, WBC_ARM_START, WBC_ARM_COUNT,
    # Value conversion
    dex1_physical_to_training, dex1_training_to_physical,
)


# =============================================================================
# Constants: UNITREE_G1 joint layout (from centralized robot_configs)
# =============================================================================

STATE_GROUPS = GROOT_STATE_LAYOUT
STATE_DOF = GROOT_STATE_DOF
GROOT_TO_WBC_ARM = GROOT_TO_WBC_ARM_REMAP

# Training-data mean joint state (from checkpoint dataset_statistics.json).
TRAINING_MEAN_STATE = {
    "left_leg":  np.array([-0.43, 0.03, 0.005, 0.64, -0.17, 0.0003], dtype=np.float32),
    "right_leg": np.array([-0.44, -0.05, 0.02, 0.63, -0.18, 0.008], dtype=np.float32),
    "waist":     np.array([0.005, -0.04, 0.24], dtype=np.float32),
    # GROOT order: sh_p, sh_r, sh_y, elbow, wr_yaw, wr_roll, wr_pitch
    "left_arm":  np.array([-0.75, 0.61, 0.08, 0.42, -0.35, 0.40, -0.94], dtype=np.float32),
    "right_arm": np.array([-0.90, -0.54, -0.11, 0.68, 0.30, 0.15, 0.93], dtype=np.float32),
    # Hands in training gripper-range space [0, 5.4]
    "left_hand":  2.9,
    "right_hand": 2.25,
}


# =============================================================================
# Hand type configuration
# =============================================================================

@dataclass
class HandConfig:
    """Configuration for a hand type.

    Each hand type defines how to:
    - Inject hand bodies/joints into g1.xml (or use an existing model)
    - Generate actuator XML for the hand joints
    - Convert between physical joint space and training-data space
    - Read hand state from simulation (for GROOT observation)
    - Apply hand actions from GROOT (to simulation actuators)
    """
    name: str
    g1_filename: str  # Which Menagerie G1 XML to use as base
    joint_names: Dict[str, List[str]]  # {side: [joint_names]}

    def inject_into_xml(self, g1_xml: str) -> str:
        """Inject hand bodies into g1.xml. Override in subclasses."""
        return g1_xml

    def get_actuator_xml(self) -> str:
        """Return actuator XML snippet for these hand joints."""
        return ""

    def physical_to_training(self, physical_value: float) -> float:
        """Convert physical joint value to training-data space."""
        return physical_value

    def training_to_physical(self, training_value: float) -> float:
        """Convert training-data value to physical joint space."""
        return training_value

    def get_hand_state(self, model, data, side: str) -> float:
        """Read hand state from simulation, return in training-data space."""
        return TRAINING_MEAN_STATE[f"{side}_hand"]

    def apply_hand_action(self, model, data, action_value: float, side: str):
        """Apply GROOT hand action to simulation actuators."""
        pass

    def init_hand_joints(self, model, data, side: str, training_value: float):
        """Set initial hand joint positions from training mean."""
        pass

    def has_joints(self, model) -> bool:
        """Check if the model has this hand type's joints."""
        return False


class Dex1HandConfig(HandConfig):
    """Dex1 prismatic parallel-jaw gripper.

    Specs from dm_isaac_g1.core.robot_configs.DEX1:
      - 2 prismatic joints per hand (Joint1_1 primary, Joint2_1 mirrored)
      - Range: [-0.02, 0.0245] meters per finger
      - Attach to wrist_yaw_link at offset [0.0415, 0, 0]
      - Value mapping: physical [-0.02, 0.024] <-> training [5.4, 0.0]
    """

    def __init__(self):
        super().__init__(
            name=DEX1.name,
            g1_filename="g1.xml",
            joint_names=DEX1.joint_names,
        )

    def inject_into_xml(self, g1_xml: str) -> str:
        # Remove Menagerie rubber hand visual meshes (they look like Inspire hands)
        for side in ["left", "right"]:
            g1_xml = re.sub(
                rf'<geom[^>]*mesh="{side}_rubber_hand"[^/]*/>', '', g1_xml
            )
        print("  Removed rubber hand visual meshes (replaced by Dex1 grippers)")

        for side in ["left", "right"]:
            wrist_body_name = f"{side}_wrist_yaw_link"
            pattern = rf'(<body\s+name="{wrist_body_name}"[^>]*>)'
            match = re.search(pattern, g1_xml)
            if match is None:
                print(f"  WARNING: Could not find {wrist_body_name} to inject Dex1")
                continue

            dex1_xml = f"""
        <!-- Dex1 prismatic gripper ({side}) -->
        <body name="{side}_dex1_base" pos="0.0415 0 0">
          <body name="{side}_hand_finger1">
            <joint name="{side}_hand_Joint1_1" type="slide" axis="0 -1 0"
                   range="-0.02 0.0245" damping="0.1"/>
            <geom type="box" size="0.015 0.005 0.04" pos="0 -0.015 0"
                  mass="0.05" rgba="0.2 0.2 0.2 1" condim="3" friction="1.0 0.5 0.1"/>
          </body>
          <body name="{side}_hand_finger2">
            <joint name="{side}_hand_Joint2_1" type="slide" axis="0 1 0"
                   range="-0.02 0.0245" damping="0.1"/>
            <geom type="box" size="0.015 0.005 0.04" pos="0 0.015 0"
                  mass="0.05" rgba="0.2 0.2 0.2 1" condim="3" friction="1.0 0.5 0.1"/>
          </body>
        </body>
"""
            g1_xml = g1_xml[:match.end()] + dex1_xml + g1_xml[match.end():]
            print(f"  Injected Dex1 gripper into {wrist_body_name}")
        return g1_xml

    def get_actuator_xml(self) -> str:
        return """
    <!-- Dex1 gripper actuators (position-controlled prismatic joints) -->
    <position name="left_hand_Joint1_1" joint="left_hand_Joint1_1" kp="800" kv="3.0"/>
    <position name="left_hand_Joint2_1" joint="left_hand_Joint2_1" kp="800" kv="3.0"/>
    <position name="right_hand_Joint1_1" joint="right_hand_Joint1_1" kp="800" kv="3.0"/>
    <position name="right_hand_Joint2_1" joint="right_hand_Joint2_1" kp="800" kv="3.0"/>"""

    def physical_to_training(self, value: float) -> float:
        return dex1_physical_to_training(value)

    def training_to_physical(self, value: float) -> float:
        return dex1_training_to_physical(value)

    def get_hand_state(self, model, data, side: str) -> float:
        jname = self.joint_names[side][0]  # Joint1_1 (primary)
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                return TRAINING_MEAN_STATE[f"{side}_hand"]
            return self.physical_to_training(float(data.qpos[model.jnt_qposadr[jid]]))
        except Exception:
            return TRAINING_MEAN_STATE[f"{side}_hand"]

    def apply_hand_action(self, model, data, action_value: float, side: str):
        physical_pos = self.training_to_physical(action_value)
        for jname in self.joint_names[side]:
            try:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    continue
                for act_id in range(model.nu):
                    if model.actuator_trnid[act_id, 0] == jid:
                        data.ctrl[act_id] = physical_pos
                        break
            except Exception:
                pass

    def init_hand_joints(self, model, data, side: str, training_value: float):
        physical_pos = self.training_to_physical(training_value)
        for jname in self.joint_names[side]:
            try:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid >= 0:
                    data.qpos[model.jnt_qposadr[jid]] = physical_pos
            except Exception:
                pass
        print(f"    {side} hand: training={training_value:.2f} -> physical={physical_pos:.4f} m")

    def has_joints(self, model) -> bool:
        try:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hand_Joint1_1") >= 0
        except Exception:
            return False


class InspireHandConfig(HandConfig):
    """Menagerie Inspire dexterous hands (7 revolute joints per hand).

    Uses g1_with_hands.xml. Maps between GROOT's 1-DOF hand value and 7 finger joints
    by linearly interpolating each finger joint across its range.

    Note: Menagerie g1_with_hands.xml uses Dex3 naming (7 joints/hand), not
    the full Inspire 12-DOF naming. The joint names here match Menagerie's model.
    """

    def __init__(self):
        # Menagerie g1_with_hands uses Dex3-style 7-joint naming
        super().__init__(
            name="inspire",
            g1_filename="g1_with_hands.xml",
            joint_names=DEX3.joint_names,
        )

    def get_actuator_xml(self) -> str:
        lines = "\n    <!-- Inspire hand actuators (position control for finger joints) -->"
        for side in ["left", "right"]:
            for jname in self.joint_names[side]:
                act_name = jname.replace("_joint", "")
                lines += f'\n    <position name="{act_name}" joint="{jname}" kp="10"/>'
        return lines

    def _norm_to_training(self, norm_val: float) -> float:
        """Normalized [0,1] finger curl -> training range ~[0, 4.74]."""
        return norm_val * 4.74

    def _training_to_norm(self, training_val: float) -> float:
        """Training range [0, 5.4] -> normalized [0,1]."""
        return np.clip(training_val / 4.74, 0.0, 1.0)

    def get_hand_state(self, model, data, side: str) -> float:
        vals = []
        for jname in self.joint_names[side]:
            try:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    continue
                lo, hi = model.jnt_range[jid]
                qval = data.qpos[model.jnt_qposadr[jid]]
                if hi - lo > 1e-6:
                    vals.append((qval - lo) / (hi - lo))
            except Exception:
                pass
        if not vals:
            return TRAINING_MEAN_STATE[f"{side}_hand"]
        return self._norm_to_training(float(np.mean(vals)))

    def apply_hand_action(self, model, data, action_value: float, side: str):
        norm_val = self._training_to_norm(action_value)
        for jname in self.joint_names[side]:
            try:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    continue
                lo, hi = model.jnt_range[jid]
                target = lo + norm_val * (hi - lo)
                for act_id in range(model.nu):
                    if model.actuator_trnid[act_id, 0] == jid:
                        data.ctrl[act_id] = target
                        break
            except Exception:
                pass

    def init_hand_joints(self, model, data, side: str, training_value: float):
        norm_val = self._training_to_norm(training_value)
        for jname in self.joint_names[side]:
            try:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid >= 0:
                    lo, hi = model.jnt_range[jid]
                    data.qpos[model.jnt_qposadr[jid]] = lo + norm_val * (hi - lo)
            except Exception:
                pass
        print(f"    {side} hand: training={training_value:.2f} -> norm={norm_val:.3f}")

    def has_joints(self, model) -> bool:
        try:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hand_thumb_0_joint") >= 0
        except Exception:
            return False


class NoHandConfig(HandConfig):
    """No hands — base g1.xml with no hand joints.

    Hand state is sent as training mean (constant). Hand actions are ignored.
    """

    def __init__(self):
        super().__init__(
            name="none",
            g1_filename="g1.xml",
            joint_names={"left": [], "right": []},
        )


# Registry of MuJoCo hand configs (keys match robot_configs.HAND_TYPES)
HAND_TYPES: Dict[str, HandConfig] = {
    "dex1": Dex1HandConfig(),
    "inspire": InspireHandConfig(),
    "none": NoHandConfig(),
}


# =============================================================================
# WBC Controller — based on NVIDIA's run_mujoco_gear_wbc.py
# =============================================================================

class WBCController:
    """Whole-Body-Control using ONNX Balance/Walk policies."""

    def __init__(self, wbc_dir: str, device: str = "cpu"):
        import yaml
        import onnxruntime as ort

        self.wbc_dir = wbc_dir
        config_path = os.path.join(wbc_dir, "g1_gear_wbc.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.simulation_dt = cfg["simulation_dt"]
        self.control_decimation = cfg["control_decimation"]
        self.num_actions = cfg["num_actions"]
        self.num_obs = cfg["num_obs"]
        self.obs_history_len = cfg["obs_history_len"]
        self.action_scale = cfg["action_scale"]
        self.cmd_scale = np.array(cfg["cmd_scale"])
        self.ang_vel_scale = cfg["ang_vel_scale"]
        self.dof_pos_scale = cfg["dof_pos_scale"]
        self.dof_vel_scale = cfg["dof_vel_scale"]

        self.default_angles = np.array(cfg["default_angles"], dtype=np.float32)
        self.kps = np.array(cfg["kps"], dtype=np.float32)
        self.kds = np.array(cfg["kds"], dtype=np.float32)

        self.height_cmd = cfg.get("height_cmd", 0.74)
        self.rpy_cmd = np.array(cfg.get("rpy_cmd", [0.0, 0.0, 0.0]), dtype=np.float32)

        balance_path = os.path.join(wbc_dir, "policy", "GR00T-WholeBodyControl-Balance.onnx")
        walk_path = os.path.join(wbc_dir, "policy", "GR00T-WholeBodyControl-Walk.onnx")

        providers = ['CPUExecutionProvider']
        if device != "cpu":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        print(f"  Loading WBC Balance policy: {balance_path}")
        self.balance_session = ort.InferenceSession(balance_path, providers=providers)
        print(f"  Loading WBC Walk policy: {walk_path}")
        self.walk_session = ort.InferenceSession(walk_path, providers=providers)

        obs_dim = self.num_obs // self.obs_history_len
        self.obs_dim = obs_dim
        self.obs_history = deque(maxlen=self.obs_history_len)
        for _ in range(self.obs_history_len):
            self.obs_history.append(np.zeros(obs_dim, dtype=np.float32))

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.loco_cmd = np.zeros(3, dtype=np.float32)

        print(f"  WBC initialized: {self.num_actions} actions, obs_dim={obs_dim}, "
              f"history={self.obs_history_len}, total_obs={self.num_obs}")

    def _quat_rotate_inverse(self, q, v):
        w, x, y, z = q[0], q[1], q[2], q[3]
        t = 2.0 * np.cross(np.array([-x, -y, -z]), v)
        return v + w * t + np.cross(np.array([-x, -y, -z]), t)

    def _get_gravity_orientation(self, quat):
        return self._quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

    def compute_observation(self, q_body, dq_body, base_quat, base_ang_vel):
        n_joints = 29
        cmd = np.zeros(7, dtype=np.float32)
        cmd[0:3] = self.loco_cmd * self.cmd_scale
        cmd[3] = self.height_cmd
        cmd[4:7] = self.rpy_cmd

        omega = base_ang_vel.astype(np.float32) * self.ang_vel_scale
        gravity = self._get_gravity_orientation(base_quat).astype(np.float32)

        defaults_full = np.zeros(n_joints, dtype=np.float32)
        defaults_full[:self.num_actions] = self.default_angles
        qj = (q_body.astype(np.float32) - defaults_full) * self.dof_pos_scale
        dqj = dq_body.astype(np.float32) * self.dof_vel_scale

        obs_single = np.concatenate([
            cmd, omega, gravity, qj, dqj, self.last_action,
        ]).astype(np.float32)

        self.obs_history.append(obs_single)
        return np.concatenate(list(self.obs_history))

    def get_action(self, obs):
        obs_tensor = obs.reshape(1, -1).astype(np.float32)
        if np.linalg.norm(self.loco_cmd) <= 0.05:
            session = self.balance_session
        else:
            session = self.walk_session

        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: obs_tensor})
        action = output[0][0]
        self.last_action = action.copy()
        return action * self.action_scale + self.default_angles

    def pd_control(self, target_q, q_current, dq_current):
        return self.kps * (target_q - q_current) + self.kds * (0.0 - dq_current)


# =============================================================================
# Scene loading
# =============================================================================

def load_towel_scene_wbc(scene_path: str, hand_config: HandConfig,
                         menagerie_dir: str = "/workspace/mujoco_menagerie/unitree_g1",
                         wbc_timestep: float = 0.005) -> mujoco.MjModel:
    """Load G1 + hands + towel scene configured for WBC."""
    with open(scene_path) as f:
        scene_xml = f.read()

    g1_filename = hand_config.g1_filename
    g1_path = os.path.join(menagerie_dir, g1_filename)
    with open(g1_path) as f:
        g1_xml = f.read()

    # Strip keyframe (qpos size mismatch with flexcomp towel)
    g1_xml_mod = re.sub(r'<keyframe>.*?</keyframe>', '', g1_xml, flags=re.DOTALL)

    # Inject hand bodies (Dex1 injects, Inspire/None are no-ops)
    g1_xml_mod = hand_config.inject_into_xml(g1_xml_mod)

    # Convert actuators to direct-torque for WBC + add hand actuators
    hand_actuator_xml = hand_config.get_actuator_xml()
    actuator_replacement = f"""<actuator>
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
    <motor name="right_wrist_yaw" joint="right_wrist_yaw_joint" gear="1"/>{hand_actuator_xml}
  </actuator>"""
    g1_xml_mod = re.sub(
        r'<actuator>.*?</actuator>', actuator_replacement,
        g1_xml_mod, flags=re.DOTALL
    )
    print(f"  Converted actuators to direct-torque (WBC) + {hand_config.name} hand actuators")

    # Set pelvis height + facing +Y
    g1_xml_mod = re.sub(
        r'(<body\s+name="pelvis"\s+pos=")[^"]*(")',
        r'\g<1>0 0 0.793\2', g1_xml_mod
    )
    g1_xml_mod = re.sub(
        r'(<body\s+name="pelvis"\s+pos="[^"]*")',
        r'\1 quat="0.7071 0 0 0.7071"', g1_xml_mod
    )
    print("  Set pelvis height=0.793, facing +Y")

    # Inject ego_view camera into torso_link
    ego_camera_xml = (
        '\n            <!-- D435 ego_view camera -->\n'
        '            <camera name="ego_view" pos="0.0576 0.0175 0.4299"'
        ' xyaxes="0 -1 0 0.866 0 0.500" fovy="60"/>\n'
    )
    torso_match = re.search(r'(<body\s+name="torso_link"[^>]*>)', g1_xml_mod)
    if torso_match:
        g1_xml_mod = g1_xml_mod[:torso_match.end()] + ego_camera_xml + g1_xml_mod[torso_match.end():]
        print("  Injected ego_view camera into torso_link")

    # Scene: override timestep + solver
    scene_xml_mod = re.sub(r'timestep="[^"]*"', f'timestep="{wbc_timestep}"', scene_xml)
    if 'solver=' not in scene_xml_mod:
        scene_xml_mod = scene_xml_mod.replace(
            f'timestep="{wbc_timestep}"', f'timestep="{wbc_timestep}" solver="Newton"'
        )

    # Write temp files
    g1_tmp = os.path.join(menagerie_dir, f"_g1_{hand_config.name}_wbc.xml")
    with open(g1_tmp, 'w') as f:
        f.write(g1_xml_mod)

    # Update scene include reference — replace any g1*.xml reference
    for gfile in ["g1.xml", "g1_with_hands.xml"]:
        scene_xml_mod = scene_xml_mod.replace(f"{menagerie_dir}/{gfile}", g1_tmp)
    scene_tmp = os.path.join(menagerie_dir, f"_g1_{hand_config.name}_towel_wbc_scene.xml")
    with open(scene_tmp, 'w') as f:
        f.write(scene_xml_mod)

    try:
        model = mujoco.MjModel.from_xml_path(scene_tmp)
        print(f"  Scene loaded: {model.nq} qpos, {model.nv} dof, "
              f"{model.nu} actuators, {model.nflex} flex bodies")
        print(f"  Timestep: {model.opt.timestep}s ({1/model.opt.timestep:.0f} Hz)")

        # Verify hand joints
        if hand_config.name != "none":
            for side in ["left", "right"]:
                for jname in hand_config.joint_names[side]:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                    if jid >= 0:
                        addr = model.jnt_qposadr[jid]
                        lo, hi = model.jnt_range[jid]
                        print(f"    {jname}: jid={jid}, qpos_addr={addr}, range=[{lo:.4f}, {hi:.4f}]")
                    else:
                        print(f"    WARNING: {jname} NOT FOUND in model!")

        return model
    finally:
        for tmp in [g1_tmp, scene_tmp]:
            if os.path.exists(tmp):
                os.remove(tmp)


# =============================================================================
# Joint mapping helpers
# =============================================================================

def build_joint_mapping(model: mujoco.MjModel) -> dict:
    """Build mapping from joint names to MuJoCo IDs and actuator IDs."""
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
            for aid in range(model.nu):
                if model.actuator_trnid[aid, 0] == jid:
                    entry["act_id"] = aid
                    break
            mapping[jname] = entry
        except Exception:
            pass
    return mapping


def get_body_joint_state(model, data, joint_mapping):
    """Get all 29 body joint positions and velocities in WBC order."""
    q = np.zeros(29, dtype=np.float32)
    dq = np.zeros(29, dtype=np.float32)
    for i, jname in enumerate(WBC_JOINT_NAMES):
        if jname in joint_mapping:
            m = joint_mapping[jname]
            q[i] = data.qpos[m["qpos_addr"]]
            dq[i] = data.qvel[m["qvel_addr"]]
    return q, dq


def get_base_state(model, data):
    """Get pelvis quaternion and angular velocity."""
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        quat = data.qpos[3:7].copy()
        ang_vel = data.qvel[3:6].copy()
    else:
        quat = data.xquat[pelvis_id].copy()
        ang_vel = data.qvel[3:6].copy()
    return quat.astype(np.float32), ang_vel.astype(np.float32)


def get_groot_state_vector(model, data, joint_mapping, hand_config: HandConfig):
    """Extract 31-DOF GROOT state vector from simulation.

    Arm wrists are remapped from WBC/Menagerie order (roll, pitch, yaw)
    to GROOT training order (yaw, roll, pitch).
    """
    q, _ = get_body_joint_state(model, data, joint_mapping)
    state = np.zeros(STATE_DOF, dtype=np.float32)

    state[0:6] = q[0:6]     # left leg
    state[6:12] = q[6:12]   # right leg
    state[12:15] = q[12:15] # waist

    # Left arm with wrist remap
    state[15:19] = q[15:19]
    state[19] = q[21]  # wrist_yaw
    state[20] = q[19]  # wrist_roll
    state[21] = q[20]  # wrist_pitch

    # Right arm with wrist remap
    state[22:26] = q[22:26]
    state[26] = q[28]  # wrist_yaw
    state[27] = q[26]  # wrist_roll
    state[28] = q[27]  # wrist_pitch

    # Hands via hand_config
    state[29] = hand_config.get_hand_state(model, data, "left")
    state[30] = hand_config.get_hand_state(model, data, "right")

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
    """Decode flat action dict from server into per-group arrays."""
    result = {}
    for key, arr in action_dict.items():
        name = key.replace("action.", "") if key.startswith("action.") else key
        arr = np.array(arr)
        if arr.ndim == 3:
            arr = arr[0]
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        result[name] = arr
    return result


def apply_groot_upper_body(model, data, action_step, joint_mapping,
                           hand_config: HandConfig,
                           arm_kp: float = 100.0,
                           arm_kd: float = 2.0):
    """Apply GROOT upper body actions (arms + hands) to MuJoCo actuators."""
    for side in ["left", "right"]:
        key = f"{side}_arm"
        if key not in action_step:
            continue
        groot_arm = action_step[key]

        wbc_arm_names = [
            f"{side}_shoulder_pitch_joint",
            f"{side}_shoulder_roll_joint",
            f"{side}_shoulder_yaw_joint",
            f"{side}_elbow_joint",
            f"{side}_wrist_roll_joint",   # GROOT[5]
            f"{side}_wrist_pitch_joint",  # GROOT[6]
            f"{side}_wrist_yaw_joint",    # GROOT[4]
        ]
        groot_to_act = [0, 1, 2, 3, 5, 6, 4]

        for i, jname in enumerate(wbc_arm_names):
            if jname not in joint_mapping:
                continue
            m = joint_mapping[jname]
            if m["act_id"] is None:
                continue
            target = float(groot_arm[groot_to_act[i]])
            q_curr = data.qpos[m["qpos_addr"]]
            dq_curr = data.qvel[m["qvel_addr"]]
            tau = arm_kp * (target - q_curr) + arm_kd * (0.0 - dq_curr)
            data.ctrl[m["act_id"]] = tau

    # Apply hand actions via hand_config
    if hand_config.has_joints(model):
        if "left_hand" in action_step:
            hand_config.apply_hand_action(model, data, float(action_step["left_hand"][0]), "left")
        if "right_hand" in action_step:
            hand_config.apply_hand_action(model, data, float(action_step["right_hand"][0]), "right")


def apply_wbc_torques(model, data, target_q, joint_mapping, wbc_ctrl):
    """Apply WBC lower body torques via PD control."""
    lower_body_names = WBC_JOINT_NAMES[:15]

    q_curr = np.zeros(15, dtype=np.float32)
    dq_curr = np.zeros(15, dtype=np.float32)
    for i, jname in enumerate(lower_body_names):
        if jname in joint_mapping:
            m = joint_mapping[jname]
            q_curr[i] = data.qpos[m["qpos_addr"]]
            dq_curr[i] = data.qvel[m["qvel_addr"]]

    tau = wbc_ctrl.pd_control(target_q, q_curr, dq_curr)

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
                wbc_ctrl, hand_config, language, max_steps, action_horizon,
                groot_query_interval=4, render_video=True,
                arm_kp=100.0, arm_kd=2.0,
                use_groot_waist_cmd=True, waist_action_order="ypr"):
    """Run one evaluation episode with WBC + GROOT."""
    frames = []
    states = []

    mujoco.mj_resetData(model, data)

    # Set initial pose from training data mean state.
    print("  Setting initial pose from training data mean state...")

    # Legs
    for side_label, side_name in [("left_leg", "left"), ("right_leg", "right")]:
        leg_names = [
            f"{side_name}_hip_pitch_joint", f"{side_name}_hip_roll_joint",
            f"{side_name}_hip_yaw_joint", f"{side_name}_knee_joint",
            f"{side_name}_ankle_pitch_joint", f"{side_name}_ankle_roll_joint",
        ]
        for i, jname in enumerate(leg_names):
            if jname in joint_mapping:
                data.qpos[joint_mapping[jname]["qpos_addr"]] = float(TRAINING_MEAN_STATE[side_label][i])

    # Waist
    waist_names = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    for i, jname in enumerate(waist_names):
        if jname in joint_mapping:
            data.qpos[joint_mapping[jname]["qpos_addr"]] = float(TRAINING_MEAN_STATE["waist"][i])

    # Arms (GROOT training order -> Menagerie order)
    for side in ["left", "right"]:
        arm_mean = TRAINING_MEAN_STATE[f"{side}_arm"]
        arm_names_menagerie = [
            f"{side}_shoulder_pitch_joint", f"{side}_shoulder_roll_joint",
            f"{side}_shoulder_yaw_joint", f"{side}_elbow_joint",
            f"{side}_wrist_roll_joint", f"{side}_wrist_pitch_joint",
            f"{side}_wrist_yaw_joint",
        ]
        groot_to_menagerie = [0, 1, 2, 3, 5, 6, 4]
        for i, jname in enumerate(arm_names_menagerie):
            if jname in joint_mapping:
                data.qpos[joint_mapping[jname]["qpos_addr"]] = float(arm_mean[groot_to_menagerie[i]])

    # Hands via hand_config
    if hand_config.has_joints(model):
        for side in ["left", "right"]:
            hand_config.init_hand_joints(
                model, data, side, TRAINING_MEAN_STATE[f"{side}_hand"]
            )

    mujoco.mj_forward(model, data)

    # Log initial state
    init_state = get_groot_state_vector(model, data, joint_mapping, hand_config)
    print(f"  Initial state (31 DOF):")
    for group, (start, end) in STATE_GROUPS.items():
        print(f"    {group:12s}: {init_state[start:end]}")

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
    groot_actions = None
    groot_action_idx = 0
    groot_action_horizon_len = 0
    current_groot_step = {}

    wbc_step = 0
    sim_step = 0
    control_decimation = wbc_ctrl.control_decimation
    wbc_target_q = wbc_ctrl.default_angles.copy()

    print(f"  WBC: sim_dt={model.opt.timestep}, decimation={control_decimation}, "
          f"policy_rate={1/(model.opt.timestep * control_decimation):.0f} Hz")
    print(f"  GROOT query every {groot_query_interval} WBC steps "
          f"({1/(model.opt.timestep * control_decimation * groot_query_interval):.1f} Hz)")

    while wbc_step < max_steps:
        q_body, dq_body = get_body_joint_state(model, data, joint_mapping)
        base_quat, base_ang_vel = get_base_state(model, data)

        wbc_obs = wbc_ctrl.compute_observation(q_body, dq_body, base_quat, base_ang_vel)
        wbc_target_q = wbc_ctrl.get_action(wbc_obs)

        if wbc_step % groot_query_interval == 0:
            groot_state = get_groot_state_vector(model, data, joint_mapping, hand_config)
            states.append(groot_state.copy())

            if groot_actions is None or groot_action_idx >= groot_action_horizon_len:
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

            t = min(groot_action_idx, groot_action_horizon_len - 1)
            current_groot_step = {
                name: arr[t] for name, arr in groot_actions.items()
            }
            groot_action_idx += 1

            if "navigate_command" in current_groot_step:
                wbc_ctrl.loco_cmd = current_groot_step["navigate_command"].astype(np.float32)
            if "base_height_command" in current_groot_step:
                wbc_ctrl.height_cmd = float(current_groot_step["base_height_command"][0])
            if use_groot_waist_cmd and "waist" in current_groot_step:
                waist = current_groot_step["waist"]
                if waist_action_order == "ypr":
                    yaw, pitch, roll = float(waist[0]), float(waist[1]), float(waist[2])
                elif waist_action_order == "yrp":
                    yaw, roll, pitch = float(waist[0]), float(waist[1]), float(waist[2])
                else:
                    raise ValueError(f"Unsupported waist_action_order: {waist_action_order}")
                # WBC expects rpy_cmd in roll, pitch, yaw order.
                wbc_ctrl.rpy_cmd = np.array([roll, pitch, yaw], dtype=np.float32)

        for _ in range(control_decimation):
            apply_wbc_torques(model, data, wbc_target_q, joint_mapping, wbc_ctrl)
            if current_groot_step:
                apply_groot_upper_body(model, data, current_groot_step,
                                       joint_mapping, hand_config,
                                       arm_kp=arm_kp, arm_kd=arm_kd)
            mujoco.mj_step(model, data)
            sim_step += 1

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
    parser.add_argument("--hand-type", type=str, default="dex1",
                        choices=list(HAND_TYPES.keys()),
                        help="Hand type: dex1 (prismatic gripper), "
                             "inspire (dexterous), none (no hands)")
    parser.add_argument("--wbc-dir", type=str, required=True,
                        help="Path to WBC G1 resources dir")
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
    parser.add_argument("--arm-kp", type=float, default=160.0,
                        help="Arm PD proportional gain for GROOT upper-body control")
    parser.add_argument("--arm-kd", type=float, default=4.0,
                        help="Arm PD derivative gain for GROOT upper-body control")
    parser.add_argument("--disable-groot-waist-cmd", action="store_true",
                        help="Ignore action.waist from GROOT when setting WBC rpy_cmd")
    parser.add_argument("--waist-action-order", type=str, default="ypr",
                        choices=["ypr", "yrp"],
                        help="Interpret action.waist as ypr=yaw,pitch,roll (recommended) or yrp")
    parser.add_argument("--output-dir", type=str,
                        default="/tmp/mujoco_towel_eval_wbc",
                        help="Output directory")
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    hand_config = HAND_TYPES[args.hand_type]
    print(f"Hand type: {hand_config.name} (g1 base: {hand_config.g1_filename})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["EVAL_OUTPUT_DIR"] = str(output_dir)

    # Load scene
    print(f"Loading scene: {args.scene}")
    try:
        model = load_towel_scene_wbc(args.scene, hand_config)
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
            hand_config=hand_config,
            language=args.language,
            max_steps=args.max_steps,
            action_horizon=args.action_horizon,
            groot_query_interval=args.groot_query_interval,
            render_video=not args.no_video,
            arm_kp=args.arm_kp,
            arm_kd=args.arm_kd,
            use_groot_waist_cmd=not args.disable_groot_waist_cmd,
            waist_action_order=args.waist_action_order,
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
