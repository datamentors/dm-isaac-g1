"""Dump robot joint limits from Isaac Sim environment.

Quick diagnostic script to check joint limits, especially for Dex3 hand joints
that may be locked at zero due to mimic joint constraints.

Usage:
    PYTHONPATH=/workspace/unitree_sim_isaaclab:/workspace/dm-isaac-g1/src:$PYTHONPATH \
    conda run --no-capture-output -n unitree_sim_env python scripts/dump_joint_limits.py
"""

import argparse
import os
import sys
from pathlib import Path

# Same path setup as policy_inference_groot_g1.py
_script_dir = Path(__file__).parent.absolute()
_project_root = _script_dir.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))
_unitree_sim_path = "/workspace/unitree_sim_isaaclab"
if os.path.exists(_unitree_sim_path) and _unitree_sim_path not in sys.path:
    sys.path.insert(0, _unitree_sim_path)
if not os.environ.get("PROJECT_ROOT") and os.path.exists(_unitree_sim_path):
    os.environ["PROJECT_ROOT"] = _unitree_sim_path

from isaaclab.app import AppLauncher

AVAILABLE_SCENES = {
    "stack_g1_dex3": "tasks.g1_tasks.stack_rgyblock_g1_29dof_dex3.stack_rgyblock_g1_29dof_dex3_joint_env_cfg.StackRgyBlockG129DEX3BaseFixEnvCfg",
    "pickplace_g1_dex3": "tasks.g1_tasks.pick_place_cylinder_g1_29dof_dex3.pickplace_cylinder_g1_29dof_dex3_joint_env_cfg.PickPlaceG129DEX3JointEnvCfg",
    "stack_g1_inspire": "tasks.g1_tasks.stack_rgyblock_g1_29dof_inspire.stack_rgyblock_g1_29dof_inspire_joint_env_cfg.StackRgyBlockG129InspireBaseFixEnvCfg",
}

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default="stack_g1_dex3",
                    help=f"Scene name. Available: {list(AVAILABLE_SCENES.keys())}")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib
import torch
from isaaclab.envs import ManagerBasedRLEnv

scene_name = args_cli.scene
scene_path = AVAILABLE_SCENES.get(scene_name)
if not scene_path:
    print(f"Unknown scene: {scene_name}")
    sys.exit(1)

module_path, class_name = scene_path.rsplit(".", 1)
mod = importlib.import_module(module_path)
cfg_class = getattr(mod, class_name)

env_cfg = cfg_class()
env_cfg.scene.num_envs = 1

env = ManagerBasedRLEnv(cfg=env_cfg)
robot = env.scene["robot"]

print("\n" + "="*80)
print("JOINT LIMITS DUMP")
print("="*80)

joint_names = robot.data.joint_names
soft_limits = robot.data.soft_joint_pos_limits  # (num_envs, num_joints, 2)

print(f"\nTotal joints: {len(joint_names)}")
print(f"\n{'Idx':<4s} {'Joint Name':<40s} {'Lower':>10s} {'Upper':>10s} {'Range':>10s} {'LOCKED?':>8s}")
print("-" * 82)

for i, name in enumerate(joint_names):
    lo = soft_limits[0, i, 0].item()
    hi = soft_limits[0, i, 1].item()
    rng = hi - lo
    locked = "LOCKED" if abs(rng) < 0.01 else ""
    prefix = ">>>" if "hand" in name else "   "
    print(f"{prefix}{i:<4d} {name:<40s} {lo:+10.4f} {hi:+10.4f} {rng:10.4f} {locked:>8s}")

print("\n" + "="*80)
print("HAND JOINTS SUMMARY")
print("="*80)
for i, name in enumerate(joint_names):
    if "hand" in name:
        lo = soft_limits[0, i, 0].item()
        hi = soft_limits[0, i, 1].item()
        rng = hi - lo
        curr = robot.data.joint_pos[0, i].item()
        print(f"  {name:<40s} range=[{lo:+.3f}, {hi:+.3f}] width={rng:.3f} current={curr:+.4f}")

env.close()
simulation_app.close()
