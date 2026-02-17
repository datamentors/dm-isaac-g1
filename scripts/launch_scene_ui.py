# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Simple Scene Launcher for UI-based Debugging

This script launches a pick-and-place scene WITHOUT custom camera configurations
to avoid Isaac Sim 5.1.0 synthetic data bugs. Use this for UI-based debugging.

Available scenes:
    1. pickplace_g1_inspire - G1 with Inspire hands pick-and-place (uses steering wheel)
    2. locomanipulation_g1 - G1 locomanipulation with pick-place
    3. fixed_base_ik_g1 - Fixed base upper body IK

Usage:
    # On workstation, run inside container:
    /isaac-sim/python.sh scripts/launch_scene_ui.py --scene pickplace_g1_inspire

    # View via VNC: 192.168.1.205:5901
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_script_dir = Path(__file__).parent.absolute()
_project_root = _script_dir.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from isaaclab.app import AppLauncher

# Available scenes - these are from unitree_sim_isaaclab
AVAILABLE_SCENES = {
    "pickplace_g1_inspire": "isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_unitree_g1_inspire_hand_env_cfg.PickPlaceG1InspireFTPEnvCfg",
    "locomanipulation_g1": "isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg.LocomanipulationG1EnvCfg",
    "fixed_base_ik_g1": "isaaclab_tasks.manager_based.locomanipulation.pick_place.fixed_base_upper_body_ik_g1_env_cfg.FixedBaseUpperBodyIKG1EnvCfg",
}

parser = argparse.ArgumentParser(description="Launch scene for UI debugging")
parser.add_argument("--scene", type=str, default="pickplace_g1_inspire",
                    choices=list(AVAILABLE_SCENES.keys()),
                    help="Scene to launch")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Force headless=False for UI
args_cli.headless = False
args_cli.enable_cameras = False  # Disable cameras to avoid synthetic data bugs

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-launch imports
import torch
from importlib import import_module
from isaaclab.envs import ManagerBasedRLEnv


def load_env_cfg_class(scene_name: str):
    """Dynamically load environment config class."""
    module_path = AVAILABLE_SCENES[scene_name]
    module_name, class_name = module_path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def main():
    print(f"[INFO] Loading scene: {args_cli.scene}", flush=True)

    EnvCfgClass = load_env_cfg_class(args_cli.scene)
    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Disable curriculum if present
    if hasattr(env_cfg, 'curriculum'):
        env_cfg.curriculum = None

    # Disable any camera sensors to avoid synthetic data bugs
    # The scene's built-in cameras will still be visible in the UI viewport
    if hasattr(env_cfg.scene, 'tiled_camera'):
        env_cfg.scene.tiled_camera = None
    if hasattr(env_cfg.scene, 'front_camera'):
        env_cfg.scene.front_camera = None
    if hasattr(env_cfg.scene, 'wrist_camera_left'):
        env_cfg.scene.wrist_camera_left = None
    if hasattr(env_cfg.scene, 'wrist_camera_right'):
        env_cfg.scene.wrist_camera_right = None

    # Configure sim device
    env_cfg.sim.device = args_cli.device if hasattr(args_cli, 'device') and args_cli.device else "cuda:0"

    print("[INFO] Creating environment...", flush=True)
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print("[INFO] Environment created successfully!", flush=True)
    print(f"[INFO] Robot joint count: {len(env.scene['robot'].data.joint_names)}", flush=True)
    print(f"[INFO] Action dimension: {env.action_manager.total_action_dim}", flush=True)

    # Print robot position
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w.detach().cpu().numpy()
    print(f"[INFO] Robot position: {robot_pos[0]}", flush=True)

    # Print object position if available
    if hasattr(env.scene, 'object') and env.scene.get('object') is not None:
        obj = env.scene['object']
        obj_pos = obj.data.root_pos_w.detach().cpu().numpy()
        print(f"[INFO] Object position: {obj_pos[0]}", flush=True)

    # Reset and run
    print("[INFO] Resetting environment...", flush=True)
    obs, _ = env.reset()

    print("\n" + "="*60, flush=True)
    print("Scene is ready for UI debugging!", flush=True)
    print("Connect via VNC to view and interact with the scene.", flush=True)
    print("Press Ctrl+C to exit.", flush=True)
    print("="*60 + "\n", flush=True)

    # Run simulation loop with zero actions (robot holds position)
    step = 0
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # Zero action - robot holds position
                action = torch.zeros(args_cli.num_envs, env.action_manager.total_action_dim,
                                   device=env.device, dtype=torch.float32)
                obs, _, _, _, _ = env.step(action)

                step += 1
                if step % 500 == 0:
                    print(f"[INFO] Step {step}", flush=True)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user", flush=True)

    env.close()
    print("[INFO] Environment closed", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
