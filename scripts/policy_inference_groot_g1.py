# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# Modified for GR00T N1.6 G1 inference with nested dictionary observation format
#
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 Robot Policy Inference with GR00T N1.6

This script runs the Unitree G1 robot in Isaac Sim with GR00T policy inference.
It uses nested dictionary format for observations as required by the GR00T server.

Usage:
    # Ensure GROOT server is running with NEW_EMBODIMENT tag
    # On the DGX Spark:
    #   docker exec -d groot-inference bash -c 'cd /workspace/gr00t && \
    #     python gr00t/eval/run_gr00t_server.py \
    #       --model-path /workspace/checkpoints/groot-g1-inspire-9datasets \
    #       --embodiment-tag NEW_EMBODIMENT --port 5555'

    # Then run this script:
    PYTHONPATH=/workspace/Isaac-GR00T:$PYTHONPATH \
    GR00T_STATS=/path/to/statistics.json \
    /isaac-sim/python.sh scripts/policy_inference_groot_g1.py \
        --server 192.168.1.237:5555 \
        --language "pick up the apple" \
        --enable_cameras
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

# CLI arguments
parser = argparse.ArgumentParser(description="G1 Robot with GR00T N1.6 Policy Inference")
parser.add_argument("--server", type=str, required=True, help="ZMQ server endpoint, e.g. 192.168.1.237:5555")
parser.add_argument("--api_token", type=str, default=None, help="Optional API token for the inference server.")
parser.add_argument("--language", type=str, default="pick up the object", help="Language command for the task.")
parser.add_argument("--video_h", type=int, default=64, help="Camera image height.")
parser.add_argument("--video_w", type=int, default=64, help="Camera image width.")
parser.add_argument("--num_action_steps", type=int, default=30,
                    help="Number of action steps to execute per inference call (default: 30, execute full trajectory).")
parser.add_argument("--action_scale", type=float, default=0.1,
                    help="Scale factor for actions to dampen robot response (default: 0.1).")
parser.add_argument("--max_episode_steps", type=int, default=1000,
                    help="Maximum steps per episode before reset.")
parser.add_argument(
    "--camera_parent",
    type=str,
    default="{ENV_REGEX_NS}/Robot/torso_link",
    help="Camera parent link (default: torso_link for ego view).",
)
parser.add_argument(
    "--camera_pos",
    type=float,
    nargs=3,
    default=(0.3, 0.0, 0.3),
    help="Camera position offset (x y z) in the parent frame.",
)
parser.add_argument(
    "--camera_rot",
    type=float,
    nargs=4,
    default=(0.5, -0.5, 0.5, -0.5),
    help="Camera rotation quaternion (w x y z) in the parent frame.",
)

# Isaac Sim app launcher args
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown = parser.parse_known_args()
if unknown:
    print(f"[WARN] Ignoring unknown args: {unknown}", flush=True)

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""After app launch, import the rest."""
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
    LocomanipulationG1EnvCfg,
)

try:
    from gr00t.policy.server_client import PolicyClient
except Exception as exc:
    raise RuntimeError(
        "Failed to import gr00t. Ensure GR00T is on PYTHONPATH:\n"
        "  PYTHONPATH=/workspace/Isaac-GR00T:$PYTHONPATH"
    ) from exc


# Joint patterns for G1 action space
ACTION_JOINT_PATTERNS = [
    # waist (3 DOF)
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    # left arm (7 DOF)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    # right arm (7 DOF)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    # hands (match all finger joints)
    "left_.*_index_.*",
    "left_.*_middle_.*",
    "left_.*_thumb_.*",
    "right_.*_index_.*",
    "right_.*_middle_.*",
    "right_.*_thumb_.*",
]


@configclass
class G1Gr00tActionsCfg:
    """Action configuration for G1 with GR00T policy."""
    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTION_JOINT_PATTERNS,
        preserve_order=True,
        scale=1.0,
        offset=0.0,
        use_default_offset=False,
    )


def build_flat_observation(
    camera_rgb: np.ndarray,
    joint_pos: np.ndarray,
    group_joint_ids: dict,
    state_dims: dict,
    language_cmd: str,
    video_horizon: int = 1,
    state_horizon: int = 1,
    action_joint_ids: list = None,
) -> dict:
    """
    Build observation in FLAT dictionary format for Gr00tSimPolicyWrapper.

    Supports two formats:
    1. new_embodiment: Uses "observation.state" as concatenated 53-DOF vector
    2. unitree_g1: Uses "state.left_arm", "state.right_arm", etc.

    Args:
        camera_rgb: Camera image (B, H, W, C) uint8
        joint_pos: Joint positions (B, num_joints) float32
        group_joint_ids: Dict mapping group names to joint indices
        state_dims: Dict mapping state keys to expected dimensions
        language_cmd: Language command string
        video_horizon: Temporal horizon for video (usually 1)
        state_horizon: Temporal horizon for state (usually 1)
        action_joint_ids: Joint indices for action space (for new_embodiment)

    Returns:
        Flat observation dictionary
    """
    batch_size = camera_rgb.shape[0]
    observation = {}

    # Video: nested dict format {"video": {"cam_left_high": (B, T, H, W, C)}}
    video_obs = camera_rgb[:, None, ...]  # Add time dimension
    if video_horizon > 1:
        video_obs = np.repeat(video_obs, video_horizon, axis=1)
    observation["video"] = {"cam_left_high": video_obs}

    # Check if using new_embodiment format (observation.state as single key)
    if "observation.state" in state_dims:
        # new_embodiment format: concatenated state vector
        # Use nested dict format {"state": {"observation.state": (B, T, D)}}
        state_dim = state_dims["observation.state"]
        if action_joint_ids is not None and len(action_joint_ids) >= state_dim:
            # Use action joint ordering for state
            vals = joint_pos[:, action_joint_ids[:state_dim]]
        else:
            # Use first N joints
            vals = joint_pos[:, :state_dim]

        vals = vals[:, None, :]  # Add time dimension (B, 1, D)
        if state_horizon > 1:
            vals = np.repeat(vals, state_horizon, axis=1)
        observation["state"] = {"observation.state": vals.astype(np.float32)}
    else:
        # unitree_g1 format: separate body part keys
        for key in ["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]:
            d = state_dims.get(key)
            if d is None:
                continue
            ids = group_joint_ids.get(key, [])
            if len(ids) >= d:
                vals = joint_pos[:, ids[:d]]
            else:
                # Pad if needed
                vals = np.zeros((batch_size, d), dtype=np.float32)
                if len(ids) > 0:
                    vals[:, :len(ids)] = joint_pos[:, ids]

            vals = vals[:, None, :]  # Add time dimension (B, 1, D)
            if state_horizon > 1:
                vals = np.repeat(vals, state_horizon, axis=1)
            observation[f"state.{key}"] = vals.astype(np.float32)

    # Language: nested dict format {"language": {"task": [[cmd], [cmd], ...]}}
    observation["language"] = {"task": [[language_cmd]] * batch_size}

    return observation


def main():
    """Main function."""

    def _resolve_camera_parent(parent: str) -> str:
        if parent.startswith("{ENV_REGEX_NS}") or parent.startswith("/"):
            return parent.rstrip("/")
        return f"{{ENV_REGEX_NS}}/Robot/{parent}".rstrip("/")

    # Setup environment configuration
    env_cfg = LocomanipulationG1EnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.actions = G1Gr00tActionsCfg()
    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    env_cfg.observations.lower_body_policy = None

    # Replace the steering wheel with a sphere (apple-like object)
    # Position it on the table in front of the robot
    env_cfg.scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.5, 0.75),  # On table, in front of robot
            rot=(1, 0, 0, 0),
        ),
        spawn=sim_utils.SphereCfg(
            radius=0.04,  # Apple-sized sphere (~8cm diameter)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),  # ~150g like an apple
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.1, 0.1),  # Red apple color
            ),
        ),
    )

    # Camera configuration
    camera_parent = _resolve_camera_parent(args_cli.camera_parent)
    env_cfg.scene.tiled_camera = TiledCameraCfg(
        prim_path=f"{camera_parent}/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=tuple(args_cli.camera_pos),
            rot=tuple(args_cli.camera_rot),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0)
        ),
        width=args_cli.video_w,
        height=args_cli.video_h,
    )

    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env_action_dim = env.action_manager.total_action_dim
    robot = env.scene["robot"]
    camera = env.scene["tiled_camera"]

    print(f"[INFO] Environment created with action dim: {env_action_dim}", flush=True)

    # Print all joint names for debugging
    all_joint_names = robot.data.joint_names
    print(f"[DEBUG] All joint names ({len(all_joint_names)}):", flush=True)
    for i, name in enumerate(all_joint_names):
        print(f"  [{i}] {name}", flush=True)

    # Resolve joint groups for G1
    group_joint_ids: dict[str, list[int]] = {}
    group_joint_names: dict[str, list[str]] = {}

    def _resolve(names: list[str]) -> tuple[list[int], list[str]]:
        ids, resolved = robot.find_joints(names, preserve_order=True)
        return ids, resolved

    group_joint_ids["left_leg"], group_joint_names["left_leg"] = _resolve([
        "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    ])
    group_joint_ids["right_leg"], group_joint_names["right_leg"] = _resolve([
        "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ])
    group_joint_ids["waist"], group_joint_names["waist"] = _resolve([
        "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"
    ])
    # GROOT unitree_g1 left_arm order (from training data):
    # pitch, roll, yaw, elbow, wrist_yaw, wrist_roll, wrist_pitch
    group_joint_ids["left_arm"], group_joint_names["left_arm"] = _resolve([
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    ])
    group_joint_ids["right_arm"], group_joint_names["right_arm"] = _resolve([
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    ])

    # Debug: Print resolved joint IDs
    print(f"[DEBUG] Group joint IDs:", flush=True)
    for k, v in group_joint_ids.items():
        print(f"  {k}: {v} -> {group_joint_names.get(k, [])}", flush=True)

    # Hands: all finger joints per side
    hand_names = robot.data.joint_names
    left_hand_names = [n for n in hand_names if n.startswith("left_") and any(f in n for f in ["index", "middle", "thumb"])]
    right_hand_names = [n for n in hand_names if n.startswith("right_") and any(f in n for f in ["index", "middle", "thumb"])]
    group_joint_ids["left_hand"], group_joint_names["left_hand"] = _resolve(left_hand_names)
    group_joint_ids["right_hand"], group_joint_names["right_hand"] = _resolve(right_hand_names)

    # Action joint mapping
    action_joint_ids, action_joint_names = robot.find_joints(ACTION_JOINT_PATTERNS, preserve_order=True)
    action_name_to_index = {name: i for i, name in enumerate(action_joint_names)}

    print(f"[DEBUG] Action joint mapping (len={len(action_joint_ids)}):", flush=True)
    for i, (idx, name) in enumerate(zip(action_joint_ids, action_joint_names)):
        print(f"  action[{i}] -> robot[{idx}] = {name}", flush=True)
    print(f"[DEBUG] action_joint_ids type: {type(action_joint_ids)}, content[:5]: {action_joint_ids[:5] if hasattr(action_joint_ids, '__getitem__') else action_joint_ids}", flush=True)

    # Connect to GR00T server
    if ":" not in args_cli.server:
        raise ValueError("--server must be in host:port format, e.g. 192.168.1.237:5555")
    host, port_str = args_cli.server.split(":", 1)

    print(f"[INFO] Connecting to GR00T server at {host}:{port_str}...", flush=True)
    client = PolicyClient(host=host, port=int(port_str), api_token=args_cli.api_token, strict=False)

    # Get modality configuration
    modality_cfg = client.get_modality_config()
    print(f"[INFO] Server modality config received", flush=True)

    video_modality = modality_cfg.get("video")
    state_modality = modality_cfg.get("state")
    video_horizon = len(video_modality.delta_indices) if video_modality else 1
    state_horizon = len(state_modality.delta_indices) if state_modality else 1

    # Load state dimensions from statistics file
    stats_path = os.environ.get("GR00T_STATS")
    if not stats_path:
        raise RuntimeError("GR00T_STATS environment variable must be set to the statistics.json path")

    with open(stats_path, "r") as f:
        stats = json.load(f)

    # Find the embodiment in statistics (try new_embodiment first, then unitree_g1)
    embodiment_key = None
    for key in ["new_embodiment", "unitree_g1", "NEW_EMBODIMENT"]:
        if key in stats:
            embodiment_key = key
            break

    if embodiment_key is None:
        available = ", ".join(stats.keys())
        raise RuntimeError(f"No known embodiment found in statistics. Available: {available}")

    print(f"[INFO] Using embodiment: {embodiment_key}", flush=True)

    state_dims = {k: len(v["min"]) for k, v in stats[embodiment_key]["state"].items()}
    print(f"[INFO] State dimensions: {state_dims}", flush=True)

    # Run inference loop
    print(f"[INFO] Starting inference with language command: '{args_cli.language}'", flush=True)
    print(f"[INFO] Action steps per inference: {args_cli.num_action_steps} (of 30 predicted)", flush=True)
    print(f"[INFO] Action scale: {args_cli.action_scale}", flush=True)
    sys.stdout.flush()

    obs, _ = env.reset()
    client.reset()

    step_count = 0
    action_buffer = None
    action_step_idx = 0
    first_action_logged = False
    trajectory_start_pos = None  # Joint positions when trajectory started (for relative actions)

    with torch.inference_mode():
        while simulation_app.is_running():
            # Get current robot state
            joint_pos = robot.data.joint_pos.detach().cpu().numpy()

            # Check if we need new actions from the server
            if action_buffer is None or action_step_idx >= args_cli.num_action_steps:
                # Get camera image
                rgb = camera.data.output.get("rgb")
                if rgb is None:
                    raise RuntimeError("Camera output not available. Run with --enable_cameras.")
                rgb_np = rgb.detach().cpu().numpy().astype(np.uint8)

                # Build flat observation (for Gr00tSimPolicyWrapper)
                observation = build_flat_observation(
                    camera_rgb=rgb_np,
                    joint_pos=joint_pos,
                    group_joint_ids=group_joint_ids,
                    state_dims=state_dims,
                    language_cmd=args_cli.language,
                    video_horizon=video_horizon,
                    state_horizon=state_horizon,
                    action_joint_ids=action_joint_ids,
                )

                # Get actions from server (returns 30 steps, we use first timestep)
                action_dict, _info = client.get_action(observation)
                if not isinstance(action_dict, dict):
                    raise RuntimeError(f"Unexpected action payload: {action_dict}")

                # Store action buffer and reset index
                action_buffer = action_dict
                action_step_idx = 0
                # Save trajectory starting position for relative actions
                trajectory_start_pos = joint_pos.copy()

                if not first_action_logged:
                    first_action_logged = True
                    print(f"[INFO] First action received. Action keys: {list(action_dict.keys())}", flush=True)
                    for k, v in action_dict.items():
                        if hasattr(v, "shape"):
                            arr = np.asarray(v)
                            print(f"  {k}: shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]", flush=True)
                    if "action.left_arm" in action_dict:
                        la = np.asarray(action_dict["action.left_arm"])
                        print(f"[DEBUG] action.left_arm[0] (first timestep): {la[0, 0, :]}", flush=True)
                    sys.stdout.flush()

            # Build action vector using predicted trajectory
            # For relative actions: target = trajectory_start_pos + delta[t]
            # The delta at timestep t is the CUMULATIVE change from trajectory start
            current_action_vec = joint_pos[:, action_joint_ids].copy()

            def _get_action_at_timestep(key: str, timestep: int) -> np.ndarray | None:
                """Get action value at specific timestep in trajectory."""
                possible_keys = [f"action.{key}", key]
                for k in possible_keys:
                    if k in action_buffer:
                        arr = np.asarray(action_buffer[k])
                        if arr.ndim == 3 and arr.shape[1] > timestep:
                            return arr[:, timestep, :]
                        elif arr.ndim == 2:
                            return arr
                return None

            def _apply_group(key: str, relative: bool):
                """Apply action group to action vector.

                For relative actions: target = trajectory_start_pos + delta[t]
                    The delta at timestep t represents the cumulative change from the
                    starting position of the trajectory.
                For absolute actions: target = action_value directly
                """
                arr = _get_action_at_timestep(key, action_step_idx)
                if arr is None:
                    return
                names = group_joint_names.get(key, [])
                idxs = [action_name_to_index[n] for n in names if n in action_name_to_index]
                if not idxs:
                    return
                d = len(idxs)
                # Handle dimension mismatch
                if arr.shape[1] != d:
                    if arr.shape[1] > d:
                        arr = arr[:, :d]
                    else:
                        arr = np.pad(arr, ((0, 0), (0, d - arr.shape[1])), mode="constant")

                if relative:
                    # Relative: add scaled delta to TRAJECTORY START positions
                    # Apply action_scale to dampen the robot's response
                    ref_ids = group_joint_ids.get(key, [])
                    if len(ref_ids) >= d:
                        start_vals = trajectory_start_pos[:, ref_ids[:d]]
                    else:
                        start_vals = np.zeros((1, d), dtype=np.float32)
                    target = start_vals + arr * args_cli.action_scale
                    current_action_vec[:, idxs] = target
                else:
                    # Absolute: use value directly (hands/waist)
                    current_action_vec[:, idxs] = arr

            # Apply action groups
            _apply_group("waist", relative=False)
            _apply_group("left_arm", relative=True)
            _apply_group("right_arm", relative=True)
            _apply_group("left_hand", relative=False)
            _apply_group("right_hand", relative=False)

            # Debug: print first few steps
            if step_count < 10:
                la_names = group_joint_names.get("left_arm", [])
                la_idxs = [action_name_to_index[n] for n in la_names if n in action_name_to_index]
                print(f"[DEBUG] Step {step_count}: joint_pos[left_arm] = {joint_pos[0, group_joint_ids['left_arm']]}", flush=True)
                arr = _get_action_at_timestep("left_arm", action_step_idx)
                if arr is not None:
                    print(f"[DEBUG] Step {step_count}: action delta[{action_step_idx}] = {arr[0]}", flush=True)
                    print(f"[DEBUG] Step {step_count}: trajectory_start = {trajectory_start_pos[0, group_joint_ids['left_arm']]}", flush=True)
                print(f"[DEBUG] Step {step_count}: target = {current_action_vec[0, la_idxs]}", flush=True)

            # Clip to environment action dimension
            current_action_vec = current_action_vec[:, :env_action_dim]

            # Clip actions to joint limits
            # G1 joints have various limits, use conservative bounds
            current_action_vec = np.clip(current_action_vec, -3.14, 3.14)

            action_tensor = torch.tensor(current_action_vec, device=env.device, dtype=torch.float32)

            # Step environment
            obs, _, terminated, truncated, _ = env.step(action_tensor)

            step_count += 1
            action_step_idx += 1

            # Periodic logging
            if step_count % 100 == 0:
                print(f"[INFO] Step {step_count}", flush=True)

            # Debug: print arm positions periodically
            if step_count % 50 == 0 and step_count <= 300:
                left_arm_ids = group_joint_ids['left_arm']
                print(f"[DEBUG] Step {step_count}: left_arm joints = {joint_pos[0, left_arm_ids]}", flush=True)
                arr = _get_action_at_timestep("left_arm", action_step_idx)
                if arr is not None:
                    print(f"[DEBUG] Step {step_count}: action delta[{action_step_idx}] = {arr[0]}", flush=True)

            if terminated.any() or truncated.any() or step_count >= args_cli.max_episode_steps:
                print(f"[INFO] Episode ended at step {step_count}. Resetting...", flush=True)
                obs, _ = env.reset()
                client.reset()
                step_count = 0
                action_buffer = None
                action_step_idx = 0
                first_action_logged = False
                trajectory_start_pos = None

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
