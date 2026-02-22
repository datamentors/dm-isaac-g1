# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# Modified for GR00T N1.6 G1 inference with nested dictionary observation format
#
# SPDX-License-Identifier: BSD-3-Clause

"""
GR00T Policy Inference with Isaac Sim

Config-driven inference for humanoid robots (G1 + Inspire, G1 + Dex3, etc.)
using GR00T N1.6 VLA models. Supports multiple robots, DOF layouts, camera
configurations, and scenes — all controlled via inference_setups.py.

Setup Configuration:
    Each setup in inference_setups.py defines everything needed for inference:
    - Cameras (single or multi-camera, including wrist cameras)
    - Scene (pick-place, block stacking, etc.)
    - DOF layout (28-DOF Dex3, 53-DOF Inspire, etc.)
    - Language command
    - Object placement

    Run with --list_setups to see all available setups.
    Add new setups in inference_setups.py — no changes needed here.

Usage:
    # Dex3 28-DOF block stacking (4 cameras, scene auto-selected):
    PYTHONPATH=/workspace/Isaac-GR00T:$PYTHONPATH \
    GR00T_STATS=/workspace/checkpoints/groot_g1_dex3_28dof/checkpoint-10000/statistics.json \
    python scripts/policy_inference_groot_g1.py \
        --server 192.168.1.237:5555 \
        --setup dex3_stack \
        --enable_cameras

    # Inspire 53-DOF pick-place (single camera):
    PYTHONPATH=/workspace/Isaac-GR00T:$PYTHONPATH \
    GR00T_STATS=/workspace/checkpoints/groot_g1_inspire_9datasets/processor/statistics.json \
    python scripts/policy_inference_groot_g1.py \
        --server 192.168.1.237:5555 \
        --setup default \
        --scene pickplace_g1_inspire \
        --language "Pick up the red apple and place it on the plate" \
        --enable_cameras
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
_script_dir = Path(__file__).parent.absolute()
_project_root = _script_dir.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

# Add unitree_sim_isaaclab to path for G1 scenes
# Mounted at /workspace/unitree_sim_isaaclab in container
_unitree_sim_path = "/workspace/unitree_sim_isaaclab"
if os.path.exists(_unitree_sim_path) and _unitree_sim_path not in sys.path:
    sys.path.insert(0, _unitree_sim_path)

# Set PROJECT_ROOT so scene configs can resolve USD asset paths (e.g. PackingTable.usd).
# Scene configs use os.environ.get("PROJECT_ROOT") to build asset paths.
if not os.environ.get("PROJECT_ROOT") and os.path.exists(_unitree_sim_path):
    os.environ["PROJECT_ROOT"] = _unitree_sim_path

from isaaclab.app import AppLauncher

# Available scenes for G1 robot
# Uses unitree_sim_isaaclab scenes which have full robot USD with d435_link camera
# See: https://github.com/unitreerobotics/unitree_sim_isaaclab
AVAILABLE_SCENES = {
    # G1 with Inspire hands (from unitree_sim_isaaclab)
    "pickplace_g1_inspire": "tasks.g1_tasks.pick_place_cylinder_g1_29dof_inspire.pickplace_cylinder_g1_29dof_inspire_env_cfg.PickPlaceG129InspireBaseFixEnvCfg",
    "pickplace_redblock_g1_inspire": "tasks.g1_tasks.pick_place_redblock_g1_29dof_inspire.pickplace_redblock_g1_29dof_inspire_joint_env_cfg.PickPlaceG129InspireHandBaseFixEnvCfg",
    "stack_g1_inspire": "tasks.g1_tasks.stack_rgyblock_g1_29dof_inspire.stack_rgyblock_g1_29dof_inspire_joint_env_cfg.StackRgyBlockG129InspireBaseFixEnvCfg",
    # G1 with DEX3 hands (from unitree_sim_isaaclab)
    "pickplace_g1_dex3": "tasks.g1_tasks.pick_place_cylinder_g1_29dof_dex3.pickplace_cylinder_g1_29dof_dex3_joint_env_cfg.PickPlaceG129DEX3JointEnvCfg",
    "stack_g1_dex3": "tasks.g1_tasks.stack_rgyblock_g1_29dof_dex3.stack_rgyblock_g1_29dof_dex3_joint_env_cfg.StackRgyBlockG129DEX3BaseFixEnvCfg",
    # IsaacLab locomotion tasks (simplified robot, no d435_link)
    "locomotion_g1_flat": "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg.G1FlatEnvCfg",
    "locomotion_g1_rough": "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg.G1RoughEnvCfg",
}

# CLI arguments
parser = argparse.ArgumentParser(description="G1 Robot with GR00T N1.6 Policy Inference")
parser.add_argument("--server", type=str, required=True, help="ZMQ server endpoint, e.g. 192.168.1.237:5555")
parser.add_argument("--scene", type=str, default="pickplace_g1_inspire",
                    help=f"Scene/environment to use (can be overridden by setup). Available: {', '.join(AVAILABLE_SCENES.keys())}")
parser.add_argument("--api_token", type=str, default=None, help="Optional API token for the inference server.")
parser.add_argument("--language", type=str, default="Pick up the red apple and place it on the plate", help="Language command for the task.")
parser.add_argument("--video_h", type=int, default=480, help="Camera image height.")
parser.add_argument("--video_w", type=int, default=640, help="Camera image width.")
parser.add_argument("--num_action_steps", type=int, default=30,
                    help="Number of action steps to execute per inference call (default: 30, execute full trajectory).")
parser.add_argument("--action_scale", type=float, default=1.0,
                    help="Scale factor for actions (default: 1.0 for absolute joint targets).")
parser.add_argument("--max_episode_steps", type=int, default=1000,
                    help="Maximum steps per episode before reset.")
parser.add_argument(
    "--setup",
    type=str,
    default="default",
    help=(
        "Inference setup name from inference_setups.py. "
        "Controls camera position/orientation and object placement. "
        "Use --list_setups to see all available options. "
        "Default: 'default' (original d435_link top-down camera)."
    ),
)
parser.add_argument(
    "--list_setups",
    action="store_true",
    help="Print all available inference setups and exit.",
)
parser.add_argument(
    "--debug_dir",
    type=str,
    default="/tmp/groot_debug",
    help="Directory to save debug images and observations.",
)
parser.add_argument(
    "--save_debug_frames",
    action="store_true",
    help="Save camera frames and observation data for debugging.",
)

# Isaac Sim app launcher args
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown = parser.parse_known_args()
if unknown:
    print(f"[WARN] Ignoring unknown args: {unknown}", flush=True)

# Import inference setups (must happen after path setup above, before AppLauncher)
from inference_setups import get_setup, list_setups, SETUPS, CameraSpec  # noqa: E402

# Handle --list_setups early exit (no Isaac Sim needed)
if args_cli.list_setups:
    print(list_setups())
    sys.exit(0)

# Validate setup name before launching Isaac Sim (fail fast)
try:
    _selected_setup = get_setup(args_cli.setup)
    print(f"[INFO] Using inference setup: '{args_cli.setup}' — {_selected_setup.description[:80]}...", flush=True)
except ValueError as e:
    print(f"[ERROR] {e}", flush=True)
    sys.exit(1)

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# Apply Isaac Sim 5.1.0 workarounds for camera/synthetic data stability
# Reference: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/known_issues.html
# ============================================================================
import carb.settings
_settings = carb.settings.get_settings()

# Disable async rendering to prevent frame skipping with Replicator/TiledCamera
_settings.set("/exts/isaacsim.core.throttling/enable_async", False)

# Set DLSS to Quality mode (2) for synthetic data generation
# Required for camera resolutions below 600x600
_settings.set("/rtx/post/dlss/mode", 2)

# Increase subframes for material loading stability
_settings.set("/rtx/replicator/rt_subframes", 4)

print("[INFO] Applied Isaac Sim 5.1.0 camera stability workarounds", flush=True)

"""After app launch, import the rest."""
import numpy as np
import torch

# CRITICAL: Add IsaacLab env_isaaclab paths for pink/pinocchio IK libraries
# This must happen AFTER torch is imported (above) but BEFORE loading scenes that use pink
# The env_isaaclab paths contain pink, pinocchio, hpp-fcl compiled for Python 3.11
import ctypes
_isaaclab_venv = os.environ.get("ISAACLAB_VENV_SITE", "/workspace/IsaacLab/env_isaaclab/lib/python3.11/site-packages")
_isaaclab_cmeel = os.environ.get("ISAACLAB_CMEEL_SITE", f"{_isaaclab_venv}/cmeel.prefix/lib/python3.11/site-packages")
_isaaclab_cmeel_lib = os.environ.get("ISAACLAB_CMEEL_LIB", f"{_isaaclab_venv}/cmeel.prefix/lib")

# Add to Python path (for pink, pinocchio modules)
if _isaaclab_venv not in sys.path:
    sys.path.append(_isaaclab_venv)
if _isaaclab_cmeel not in sys.path:
    sys.path.append(_isaaclab_cmeel)

# Pre-load hpp-fcl shared library to avoid runtime linking issues
try:
    ctypes.CDLL(f"{_isaaclab_cmeel_lib}/libhpp-fcl.so", mode=ctypes.RTLD_GLOBAL)
except Exception as e:
    print(f"[WARN] Could not preload libhpp-fcl.so: {e}", flush=True)

from importlib import import_module

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils


def load_env_cfg_class(scene_name: str):
    """Dynamically load environment config class based on scene name."""
    if scene_name not in AVAILABLE_SCENES:
        raise ValueError(f"Unknown scene: {scene_name}. Available: {list(AVAILABLE_SCENES.keys())}")

    module_path = AVAILABLE_SCENES[scene_name]
    module_name, class_name = module_path.rsplit(".", 1)

    try:
        module = import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Failed to load scene '{scene_name}': {e}")

try:
    from gr00t.policy.server_client import PolicyClient
except Exception as exc:
    raise RuntimeError(
        "Failed to import gr00t. Ensure GR00T is on PYTHONPATH:\n"
        "  PYTHONPATH=/workspace/Isaac-GR00T:$PYTHONPATH"
    ) from exc


# Default joint patterns for G1 action space - DEX3 hands (fallback only)
# Prefer using setup.action_joint_patterns from InferenceSetup config.
# This constant is kept for backward compatibility with legacy setups.
DEX3_ACTION_JOINT_PATTERNS = [
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
    # hands - DEX3 naming only (left_hand_*, right_hand_*)
    "left_hand_.*",
    "right_hand_.*",
]


@configclass
class G1Gr00tActionsCfg:
    """Action configuration for G1 with GR00T policy (DEX3 hands)."""
    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=DEX3_ACTION_JOINT_PATTERNS,
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
    joint_to_53dof_mapping: dict = None,
    extra_camera_rgbs: dict = None,
    primary_camera_name: str = "cam_left_high",
) -> dict:
    """
    Build observation in FLAT dictionary format for Gr00tSimPolicyWrapper.

    Supports two formats:
    1. new_embodiment (28 or 53 DOF): Uses "observation.state" as concatenated vector
    2. unitree_g1: Uses "state.left_arm", "state.right_arm", etc.

    Args:
        camera_rgb: Primary camera image (B, H, W, C) uint8 — keyed by primary_camera_name
        joint_pos: Joint positions (B, num_joints) float32
        group_joint_ids: Dict mapping group names to joint indices
        state_dims: Dict mapping state keys to expected dimensions
        language_cmd: Language command string
        video_horizon: Temporal horizon for video (usually 1)
        state_horizon: Temporal horizon for state (usually 1)
        action_joint_ids: Joint indices for action space (for new_embodiment)
        joint_to_53dof_mapping: Dict mapping DOF indices to robot joint indices (for padding)
        extra_camera_rgbs: Optional dict of additional camera images {cam_name: (B, H, W, C)}
        primary_camera_name: Name for the primary camera in the video dict (default: "cam_left_high")

    Returns:
        Flat observation dictionary
    """
    batch_size = camera_rgb.shape[0]
    observation = {}

    # Video: nested dict format {"video": {"cam_name": (B, T, H, W, C)}}
    video_obs = camera_rgb[:, None, ...]  # Add time dimension
    if video_horizon > 1:
        video_obs = np.repeat(video_obs, video_horizon, axis=1)

    video_dict = {primary_camera_name: video_obs}

    # Add extra cameras if provided (for multi-camera models like Dex3 28-DOF)
    if extra_camera_rgbs:
        for cam_name, cam_rgb in extra_camera_rgbs.items():
            cam_obs = cam_rgb[:, None, ...]
            if video_horizon > 1:
                cam_obs = np.repeat(cam_obs, video_horizon, axis=1)
            video_dict[cam_name] = cam_obs

    observation["video"] = video_dict

    # Check if using new_embodiment format (observation.state as single key)
    if "observation.state" in state_dims:
        # new_embodiment format: concatenated state vector
        # Use nested dict format {"state": {"observation.state": (B, T, D)}}
        state_dim = state_dims["observation.state"]

        # Build state vector with proper joint mapping
        if joint_to_53dof_mapping is not None:
            vals = np.zeros((batch_size, state_dim), dtype=np.float32)
            for dof_idx, robot_joint_idx in joint_to_53dof_mapping.items():
                if robot_joint_idx is not None and robot_joint_idx < joint_pos.shape[1]:
                    vals[:, dof_idx] = joint_pos[:, robot_joint_idx]
        elif action_joint_ids is not None and len(action_joint_ids) >= state_dim:
            # Use action joint ordering for state
            vals = joint_pos[:, action_joint_ids[:state_dim]]
        else:
            # Pad robot joints to expected state_dim with zeros
            robot_joints = joint_pos.shape[1]
            if robot_joints < state_dim:
                vals = np.zeros((batch_size, state_dim), dtype=np.float32)
                vals[:, :robot_joints] = joint_pos
            else:
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
    # Load the selected inference setup (camera + scene layout)
    setup = get_setup(args_cli.setup)
    print(f"[INFO] Inference setup '{args_cli.setup}': {setup.description}", flush=True)

    # Apply setup overrides for scene and language
    scene_name = setup.scene or args_cli.scene
    language_cmd = setup.language or args_cli.language

    # Load environment configuration dynamically
    print(f"[INFO] Loading scene: {scene_name}", flush=True)
    EnvCfgClass = load_env_cfg_class(scene_name)
    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = 1

    # Apply common configuration overrides
    if hasattr(env_cfg, 'curriculum'):
        env_cfg.curriculum = None

    # Action config: use setup.action_joint_patterns if provided, otherwise auto-detect
    if setup.action_joint_patterns:
        # Config-driven: use exact joint patterns from setup
        env_cfg.actions = G1Gr00tActionsCfg()
        env_cfg.actions.joint_pos.joint_names = setup.action_joint_patterns
    elif setup.hand_type == "inspire" or "inspire" in scene_name:
        # Inspire hands: use scene's native action config (it has proper L_*/R_* joint patterns)
        pass
    else:
        # Fallback: DEX3 default patterns
        env_cfg.actions = G1Gr00tActionsCfg()

    # Fix root link (from setup config — True for manipulation, False for locomotion)
    if setup.fix_root_link:
        if hasattr(env_cfg.scene, 'robot') and hasattr(env_cfg.scene.robot, 'spawn'):
            if hasattr(env_cfg.scene.robot.spawn, 'articulation_props'):
                env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True

    # Disable lower body policy (from setup config — True for tabletop manipulation)
    if setup.disable_lower_body:
        if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'lower_body_policy'):
            env_cfg.observations.lower_body_policy = None

    # Scene layout — loaded from inference_setups.py based on --setup argument.
    # Only spawn extra objects (apple + plate) if setup.spawn_objects is True.
    # Scenes like block stacking already have their own objects.
    if setup.spawn_objects:
        print(f"[INFO] Spawning objects — apple: {setup.object_pos}, plate: {setup.plate_pos}", flush=True)

        # Apple (replace the scene's default cylinder object)
        env_cfg.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=setup.object_pos, rot=(1, 0, 0, 0)),
            spawn=sim_utils.SphereCfg(
                radius=0.04,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
            ),
        )

        # Plate (static kinematic object — destination for the apple)
        env_cfg.scene.plate = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            init_state=RigidObjectCfg.InitialStateCfg(pos=setup.plate_pos, rot=(1, 0, 0, 0)),
            spawn=sim_utils.CylinderCfg(
                radius=setup.plate_radius,
                height=0.01,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.85)),
            ),
        )
    else:
        print(f"[INFO] Using scene's native objects (spawn_objects=False)", flush=True)

    # Camera setup — config-driven from InferenceSetup
    # Supports single camera (legacy) and multi-camera (new CameraSpec list)
    primary_cam = setup.get_primary_camera()
    extra_cam_specs = setup.get_extra_cameras()
    duplicate_cam_specs = setup.get_duplicate_cameras()

    def _resolve_camera_prim(cam_spec):
        """Resolve prim path and offset for a CameraSpec."""
        if cam_spec.prim_path:
            return cam_spec.prim_path, cam_spec.pos, cam_spec.rot
        if cam_spec.camera_parent is None:
            return "{ENV_REGEX_NS}/Robot/d435_link/front_cam", (0.0, 0.0, 0.0), (0.5, -0.5, 0.5, -0.5)
        if cam_spec.camera_parent == "__world__":
            return "{ENV_REGEX_NS}/inference_cam_world", cam_spec.pos, cam_spec.rot
        return f"{{ENV_REGEX_NS}}/Robot/{cam_spec.camera_parent}/inference_cam", cam_spec.pos, cam_spec.rot

    # Primary camera (front_camera)
    camera_prim, cam_pos, cam_rot = _resolve_camera_prim(primary_cam)
    print(f"[INFO] Primary camera '{primary_cam.name}': prim={camera_prim}, pos={cam_pos}, rot={cam_rot}", flush=True)

    env_cfg.scene.front_camera = CameraCfg(
        prim_path=camera_prim,
        update_period=0.02,
        height=args_cli.video_h,
        width=args_cli.video_w,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=primary_cam.focal_length,
            focus_distance=400.0,
            horizontal_aperture=primary_cam.horizontal_aperture,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=cam_pos,
            rot=cam_rot,
            convention="ros",
        ),
    )

    # Extra cameras (wrist cameras, etc.) — spawned as additional scene sensors
    extra_camera_names = []
    for i, cam_spec in enumerate(extra_cam_specs):
        attr_name = f"extra_cam_{i}"
        extra_camera_names.append((attr_name, cam_spec.name))
        ecam_prim, ecam_pos, ecam_rot = _resolve_camera_prim(cam_spec)
        print(f"[INFO] Extra camera '{cam_spec.name}': prim={ecam_prim}, pos={ecam_pos}, rot={ecam_rot}", flush=True)
        setattr(env_cfg.scene, attr_name, CameraCfg(
            prim_path=ecam_prim,
            update_period=0.02,
            height=args_cli.video_h,
            width=args_cli.video_w,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=cam_spec.focal_length,
                focus_distance=400.0,
                horizontal_aperture=cam_spec.horizontal_aperture,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=ecam_pos,
                rot=ecam_rot,
                convention="ros",
            ),
        ))

    # Log duplicate cameras (these reuse primary image under a different name)
    for cam_spec in duplicate_cam_specs:
        print(f"[INFO] Duplicate camera '{cam_spec.name}': copies primary '{primary_cam.name}'", flush=True)

    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env_action_dim = env.action_manager.total_action_dim
    robot = env.scene["robot"]
    camera = env.scene["front_camera"]

    # Retrieve extra camera sensors
    extra_cameras = {}
    for attr_name, cam_name in extra_camera_names:
        extra_cameras[cam_name] = env.scene[attr_name]
        print(f"[INFO] Extra camera sensor '{cam_name}' ready", flush=True)

    print(f"[INFO] Environment created with action dim: {env_action_dim}", flush=True)

    # Debug: Print robot and object positions
    robot_root_pos = robot.data.root_pos_w.detach().cpu().numpy()
    print(f"[DEBUG] Robot root position (world): {robot_root_pos[0]}", flush=True)
    if hasattr(env.scene, 'object') and env.scene['object'] is not None:
        obj = env.scene['object']
        obj_pos = obj.data.root_pos_w.detach().cpu().numpy()
        print(f"[DEBUG] Object position (world): {obj_pos[0]}", flush=True)
    # Camera info (from selected setup)
    print(f"[DEBUG] Camera setup: {args_cli.setup}, primary={primary_cam.name}, extras={[c.name for c in extra_cam_specs]}", flush=True)

    # Print camera world pose so we can hardcode it for the world-fixed camera setup.
    # This gives the exact pos/quat of d435_link/front_cam in world frame at step 0,
    # before any robot motion. Use these values in option_b_worldcam in inference_setups.py.
    try:
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        cam_prim_paths = [
            "/World/envs/env_0/Robot/d435_link/front_cam",
            "/World/envs/env_0/Robot/d435_link",
        ]
        for cp in cam_prim_paths:
            prim = stage.GetPrimAtPath(cp)
            if prim.IsValid():
                xform = UsdGeom.Xformable(prim)
                world_xform = xform.ComputeLocalToWorldTransform(0)
                t = world_xform.ExtractTranslation()
                r = world_xform.ExtractRotationQuat()
                ri = r.GetImaginary()
                print(f"[DEBUG] {cp} world pos: ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})", flush=True)
                print(f"[DEBUG] {cp} world rot (w,x,y,z): ({r.GetReal():.4f}, {ri[0]:.4f}, {ri[1]:.4f}, {ri[2]:.4f})", flush=True)
    except Exception as e:
        print(f"[DEBUG] Could not read d435_link world pose: {e}", flush=True)

    # Print all joint names for debugging
    all_joint_names = robot.data.joint_names
    print(f"[DEBUG] All joint names ({len(all_joint_names)}):", flush=True)
    for i, name in enumerate(all_joint_names):
        print(f"  [{i}] {name}", flush=True)

    # Resolve joint groups — config-driven or auto-detect
    group_joint_ids: dict[str, list[int]] = {}
    group_joint_names: dict[str, list[str]] = {}

    def _resolve(names: list[str]) -> tuple[list[int], list[str]]:
        ids, resolved = robot.find_joints(names, preserve_order=True)
        return ids, resolved

    if setup.joint_groups:
        # Config-driven: use exact joint groups from setup
        for group_name, joint_patterns in setup.joint_groups.items():
            group_joint_ids[group_name], group_joint_names[group_name] = _resolve(joint_patterns)
        print(f"[INFO] Joint groups from config: {list(setup.joint_groups.keys())}", flush=True)
    else:
        # Auto-detect: G1-specific defaults (backward compat)
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
        group_joint_ids["left_arm"], group_joint_names["left_arm"] = _resolve([
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
        ])
        group_joint_ids["right_arm"], group_joint_names["right_arm"] = _resolve([
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
        ])

        # Hands: detect type from setup config or robot joint names
        hand_type = setup.hand_type
        if not hand_type:
            # Auto-detect from joint names
            hand_names = robot.data.joint_names
            if any(n.startswith("left_hand_") for n in hand_names):
                hand_type = "dex3"
            elif any(n.startswith("L_") for n in hand_names):
                hand_type = "inspire"

        if hand_type == "dex3":
            hand_names = robot.data.joint_names
            left_hand_dex3 = [n for n in hand_names if n.startswith("left_hand_") and any(f in n for f in ["index", "middle", "thumb"])]
            right_hand_dex3 = [n for n in hand_names if n.startswith("right_hand_") and any(f in n for f in ["index", "middle", "thumb"])]
            group_joint_ids["left_hand"], group_joint_names["left_hand"] = _resolve(left_hand_dex3)
            group_joint_ids["right_hand"], group_joint_names["right_hand"] = _resolve(right_hand_dex3)
        else:
            # Inspire hands — must use EXACT training order
            group_joint_ids["left_hand"], group_joint_names["left_hand"] = _resolve([
                "L_index_proximal_joint", "L_index_intermediate_joint",
                "L_middle_proximal_joint", "L_middle_intermediate_joint",
                "L_pinky_proximal_joint", "L_pinky_intermediate_joint",
                "L_ring_proximal_joint", "L_ring_intermediate_joint",
                "L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint",
                "L_thumb_intermediate_joint", "L_thumb_distal_joint",
            ])
            group_joint_ids["right_hand"], group_joint_names["right_hand"] = _resolve([
                "R_index_proximal_joint", "R_index_intermediate_joint",
                "R_middle_proximal_joint", "R_middle_intermediate_joint",
                "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
                "R_ring_proximal_joint", "R_ring_intermediate_joint",
                "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
                "R_thumb_intermediate_joint", "R_thumb_distal_joint",
            ])

    # Debug: Print resolved joint IDs
    print(f"[DEBUG] Group joint IDs:", flush=True)
    for k, v in group_joint_ids.items():
        print(f"  {k}: {v} -> {group_joint_names.get(k, [])}", flush=True)

    # Build action joint patterns — config-driven or auto-detect
    if setup.action_joint_patterns:
        dynamic_action_patterns = list(setup.action_joint_patterns)
        print(f"[INFO] Action joint patterns from config: {len(dynamic_action_patterns)} patterns", flush=True)
    else:
        # Auto-detect from hand type
        hand_type = setup.hand_type
        if not hand_type:
            hand_names = robot.data.joint_names
            if any(n.startswith("left_hand_") for n in hand_names):
                hand_type = "dex3"
            else:
                hand_type = "inspire"

        dynamic_action_patterns = [
            "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
        ]

        if hand_type == "inspire":
            dynamic_action_patterns.extend([
                "L_index_proximal_joint", "L_index_intermediate_joint",
                "L_middle_proximal_joint", "L_middle_intermediate_joint",
                "L_pinky_proximal_joint", "L_pinky_intermediate_joint",
                "L_ring_proximal_joint", "L_ring_intermediate_joint",
                "L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint",
                "L_thumb_intermediate_joint", "L_thumb_distal_joint",
                "R_index_proximal_joint", "R_index_intermediate_joint",
                "R_middle_proximal_joint", "R_middle_intermediate_joint",
                "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
                "R_ring_proximal_joint", "R_ring_intermediate_joint",
                "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
                "R_thumb_intermediate_joint", "R_thumb_distal_joint",
            ])
        else:
            dynamic_action_patterns.extend(["left_hand_.*", "right_hand_.*"])

    # Action joint mapping
    action_joint_ids, action_joint_names = robot.find_joints(dynamic_action_patterns, preserve_order=True)
    action_name_to_index = {name: i for i, name in enumerate(action_joint_names)}

    # Build reverse mapping: robot joint index → action space index
    # action_joint_ids[action_idx] = robot_joint_idx, so we need the inverse
    robot_joint_to_action_idx = {robot_idx: action_idx for action_idx, robot_idx in enumerate(action_joint_ids)}

    # Build group → action space indices mapping (for _apply_group)
    group_action_ids: dict[str, list[int]] = {}
    for group_name, robot_idxs in group_joint_ids.items():
        action_idxs = []
        for robot_idx in robot_idxs:
            action_idx = robot_joint_to_action_idx.get(robot_idx)
            if action_idx is not None:
                action_idxs.append(action_idx)
        group_action_ids[group_name] = action_idxs

    print(f"[DEBUG] Action joint mapping (len={len(action_joint_ids)}):", flush=True)
    for i, (idx, name) in enumerate(zip(action_joint_ids, action_joint_names)):
        print(f"  action[{i}] -> robot[{idx}] = {name}", flush=True)
    print(f"[DEBUG] Group action IDs (indices into env action space):", flush=True)
    for k, v in group_action_ids.items():
        print(f"  {k}: {v}", flush=True)

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

    # Determine DOF layout — prefer setup config, fallback to statistics.json auto-detect
    total_state_dim = state_dims.get("observation.state", 53)

    if setup.dof_layout:
        # Config-driven DOF layout (from InferenceSetup)
        body_part_ranges = [
            (part_name, start, end)
            for part_name, (start, end) in setup.dof_layout.items()
        ]
        config_total = max(end for _, _, end in body_part_ranges)
        print(f"[INFO] Using config DOF layout: {config_total} DOF ({', '.join(f'{n}:{e-s}' for n, s, e in body_part_ranges)})", flush=True)
    elif total_state_dim == 28:
        # Auto-detect: 28 DOF Dex3 layout (arms + hands only, no legs/waist)
        body_part_ranges = [
            ("left_arm", 0, 7),
            ("right_arm", 7, 14),
            ("left_hand", 14, 21),
            ("right_hand", 21, 28),
        ]
        print(f"[INFO] Auto-detected 28-DOF Dex3 mapping (arms + Dex3 hands)", flush=True)
    else:
        # Auto-detect: 53 DOF Inspire layout (full body)
        body_part_ranges = [
            ("left_leg", 0, 6),
            ("right_leg", 6, 12),
            ("waist", 12, 15),
            ("left_arm", 15, 22),
            ("right_arm", 22, 29),
            ("left_hand", 29, 41),
            ("right_hand", 41, 53),
        ]
        print(f"[INFO] Auto-detected 53-DOF Inspire mapping (full body)", flush=True)

    joint_to_dof_mapping = {}
    for part_name, start_idx, end_idx in body_part_ranges:
        robot_joint_ids = group_joint_ids.get(part_name, [])
        for i, dof_idx in enumerate(range(start_idx, end_idx)):
            if i < len(robot_joint_ids):
                joint_to_dof_mapping[dof_idx] = robot_joint_ids[i]
            else:
                joint_to_dof_mapping[dof_idx] = None

    # Backward compat alias
    joint_to_53dof_mapping = joint_to_dof_mapping

    # Build action DOF ranges from same layout (used for concatenated action format)
    action_dof_ranges = {part_name: (start, end) for part_name, start, end in body_part_ranges}

    mapped_count = sum(1 for v in joint_to_dof_mapping.values() if v is not None)
    missing_count = sum(1 for v in joint_to_dof_mapping.values() if v is None)
    print(f"[INFO] {total_state_dim} DOF mapping: {mapped_count} joints mapped, {missing_count} padded with zeros", flush=True)
    if missing_count > 0:
        missing_indices = [k for k, v in joint_to_dof_mapping.items() if v is None]
        print(f"[INFO] Missing joint indices (padded): {missing_indices}", flush=True)

    # Run inference loop
    print(f"[INFO] Starting inference with language command: '{language_cmd}'", flush=True)
    print(f"[INFO] Action steps per inference: {args_cli.num_action_steps} (of 30 predicted)", flush=True)
    print(f"[INFO] Action scale: {args_cli.action_scale}", flush=True)
    sys.stdout.flush()

    obs, _ = env.reset()
    client.reset()

    # Isaac Sim 5.1.0 stability: render extra frames after reset
    # This allows physics to settle and camera buffers to initialize properly
    print("[INFO] Stabilizing simulation (30 frames)...", flush=True)
    for _ in range(30):
        obs, _, _, _, _ = env.step(torch.zeros_like(env.action_manager.action))

    step_count = 0
    action_buffer = None
    action_step_idx = 0
    first_action_logged = False
    trajectory_start_pos = None  # Joint positions when trajectory started (for relative actions)
    debug_frame_count = 0  # Counter for debug frames

    # Create debug directory if saving frames
    if args_cli.save_debug_frames:
        debug_dir = Path(args_cli.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Saving debug frames to: {debug_dir}", flush=True)

    def save_debug_data(rgb_np, observation, action_dict, frame_idx, step,
                        extra_rgbs=None):
        """Save camera image and observation data for debugging."""
        try:
            import cv2
            _use_cv2 = True
        except ImportError:
            from PIL import Image
            _use_cv2 = False

        def _save_img(img_rgb, path):
            if _use_cv2:
                bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(path), bgr)
            else:
                Image.fromarray(img_rgb).save(path)

        # Primary camera
        _save_img(rgb_np[0], debug_dir / f"frame_{frame_idx:04d}_step_{step:06d}.png")

        # Extra cameras (wrist cameras, etc.)
        if extra_rgbs:
            for cam_name, cam_rgb in extra_rgbs.items():
                _save_img(cam_rgb[0], debug_dir / f"frame_{frame_idx:04d}_step_{step:06d}_{cam_name}.png")

        # Save observation and action data as JSON
        obs_data = {
            "step": step,
            "frame_idx": frame_idx,
            "timestamp": datetime.now().isoformat(),
            "language": language_cmd,
            "camera_shape": list(rgb_np.shape),
            "camera_min_max": [int(rgb_np.min()), int(rgb_np.max())],
        }

        # Log extra camera info
        if extra_rgbs:
            for cam_name, cam_rgb in extra_rgbs.items():
                obs_data[f"cam_{cam_name}_shape"] = list(cam_rgb.shape)
                obs_data[f"cam_{cam_name}_min_max"] = [int(cam_rgb.min()), int(cam_rgb.max())]

        # Add state data
        if "state" in observation:
            for k, v in observation["state"].items():
                arr = np.asarray(v)
                obs_data[f"state_{k}_shape"] = list(arr.shape)
                obs_data[f"state_{k}_sample"] = arr[0, 0, :10].tolist() if arr.ndim >= 3 else arr.flatten()[:10].tolist()

        # Add action data if available
        if action_dict:
            for k, v in action_dict.items():
                arr = np.asarray(v)
                obs_data[f"action_{k}_shape"] = list(arr.shape)
                if arr.ndim >= 3:
                    obs_data[f"action_{k}_step0"] = arr[0, 0, :10].tolist()
                else:
                    obs_data[f"action_{k}_sample"] = arr.flatten()[:10].tolist()

        with open(debug_dir / f"obs_{frame_idx:04d}_step_{step:06d}.json", "w") as f:
            json.dump(obs_data, f, indent=2)

        print(f"[DEBUG] Saved frame {frame_idx} (step {step}) to {debug_dir}", flush=True)

    with torch.inference_mode():
        while simulation_app.is_running():
            # Get current robot state
            joint_pos = robot.data.joint_pos.detach().cpu().numpy()

            # Check if we need new actions from the server
            if action_buffer is None or action_step_idx >= args_cli.num_action_steps:
                # Get primary camera image
                rgb = camera.data.output.get("rgb")
                if rgb is None:
                    raise RuntimeError("Camera output not available. Run with --enable_cameras.")
                rgb_np = rgb.detach().cpu().numpy().astype(np.uint8)

                # Get extra camera images
                extra_camera_rgbs = {}
                for cam_name, cam_sensor in extra_cameras.items():
                    ecam_rgb = cam_sensor.data.output.get("rgb")
                    if ecam_rgb is not None:
                        extra_camera_rgbs[cam_name] = ecam_rgb.detach().cpu().numpy().astype(np.uint8)

                # Add duplicate cameras (copy primary image under different name)
                for dup_cam in duplicate_cam_specs:
                    extra_camera_rgbs[dup_cam.name] = rgb_np.copy()

                # Build flat observation (for Gr00tSimPolicyWrapper)
                observation = build_flat_observation(
                    camera_rgb=rgb_np,
                    joint_pos=joint_pos,
                    group_joint_ids=group_joint_ids,
                    state_dims=state_dims,
                    language_cmd=language_cmd,
                    video_horizon=video_horizon,
                    state_horizon=state_horizon,
                    action_joint_ids=action_joint_ids,
                    joint_to_53dof_mapping=joint_to_53dof_mapping,
                    extra_camera_rgbs=extra_camera_rgbs if extra_camera_rgbs else None,
                    primary_camera_name=primary_cam.name,
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

                # Save debug frames (every inference call, first 10 frames, then every 10th)
                if args_cli.save_debug_frames:
                    if debug_frame_count < 10 or debug_frame_count % 10 == 0:
                        save_debug_data(rgb_np, observation, action_dict, debug_frame_count, step_count,
                                        extra_rgbs=extra_camera_rgbs if extra_camera_rgbs else None)
                    debug_frame_count += 1

            # Build action vector using predicted trajectory
            # Based on embodiment config (rep=ABSOLUTE, type=NON_EEF):
            # Actions are ABSOLUTE joint position targets, not deltas
            # Initialize with ALL robot joints (env_action_dim=53), not just the 41 mapped ones.
            # This ensures the action tensor is the correct size for env.step().
            current_action_vec = joint_pos[:, :env_action_dim].copy()

            def _get_action_at_timestep(key: str, timestep: int) -> np.ndarray | None:
                """Get action value at specific timestep in trajectory.

                Handles two formats:
                1. Split format: action.left_arm, action.right_arm, etc.
                2. Concatenated format: single 'action' key with N DOF
                """
                # First try split format
                possible_keys = [f"action.{key}", key]
                for k in possible_keys:
                    if k in action_buffer:
                        arr = np.asarray(action_buffer[k])
                        if arr.ndim == 3 and arr.shape[1] > timestep:
                            return arr[:, timestep, :]
                        elif arr.ndim == 2:
                            return arr

                # Try concatenated DOF format (action_dof_ranges is dynamic)
                if "action" in action_buffer and key in action_dof_ranges:
                    arr = np.asarray(action_buffer["action"])
                    start, end = action_dof_ranges[key]
                    if arr.ndim == 3 and arr.shape[1] > timestep and arr.shape[2] >= end:
                        return arr[:, timestep, start:end]
                    elif arr.ndim == 2 and arr.shape[1] >= end:
                        return arr[:, start:end]

                return None

            def _apply_group(key: str, relative: bool):
                """Apply action group to action vector.

                Uses group_action_ids (indices into env action space, not robot joint indices).
                For relative actions: target = trajectory_start_pos + delta[t]
                For absolute actions: target = action_value directly
                """
                arr = _get_action_at_timestep(key, action_step_idx)
                if arr is None:
                    return
                # Use group_action_ids (indices into env_action_dim space)
                idxs = group_action_ids.get(key, [])
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
                    ref_ids = group_action_ids.get(key, [])
                    if len(ref_ids) >= d:
                        start_vals = trajectory_start_pos[:, :env_action_dim][:, ref_ids[:d]]
                    else:
                        start_vals = np.zeros((1, d), dtype=np.float32)
                    target = start_vals + arr * args_cli.action_scale
                    current_action_vec[:, idxs] = target
                else:
                    # Absolute: use value directly
                    current_action_vec[:, idxs] = arr

            # Apply action groups — driven by DOF layout
            # Based on embodiment config: rep=ABSOLUTE, type=NON_EEF
            # ALL actions are absolute joint position targets (not deltas)
            for part_name in action_dof_ranges:
                _apply_group(part_name, relative=False)

            # Debug: print first few steps
            if step_count < 10:
                la_action_idxs = group_action_ids.get("left_arm", [])
                la_robot_idxs = group_joint_ids.get("left_arm", [])
                if la_robot_idxs:
                    print(f"[DEBUG] Step {step_count}: current_joint_pos[left_arm] = {joint_pos[0, la_robot_idxs]}", flush=True)
                arr = _get_action_at_timestep("left_arm", action_step_idx)
                if arr is not None:
                    print(f"[DEBUG] Step {step_count}: ABSOLUTE target[{action_step_idx}] = {arr[0]}", flush=True)
                if la_action_idxs:
                    print(f"[DEBUG] Step {step_count}: applied_target = {current_action_vec[0, la_action_idxs]}", flush=True)

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
                left_arm_ids = group_joint_ids.get('left_arm', [])
                if left_arm_ids:
                    print(f"[DEBUG] Step {step_count}: left_arm joints = {joint_pos[0, left_arm_ids]}", flush=True)
                arr = _get_action_at_timestep("left_arm", action_step_idx)
                if arr is not None:
                    print(f"[DEBUG] Step {step_count}: action target[{action_step_idx}] = {arr[0]}", flush=True)

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
