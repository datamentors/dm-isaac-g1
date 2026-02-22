"""Dump robot joint limits from Isaac Sim environment.

Quick diagnostic script to check joint limits, especially for Dex3 hand joints
that may be locked at zero due to mimic joint constraints.

Usage:
    PYTHONPATH=/workspace/unitree_sim_isaaclab:/workspace/dm-isaac-g1/src:$PYTHONPATH \
    conda run --no-capture-output -n unitree_sim_env python scripts/dump_joint_limits.py
"""

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default="stack_g1_dex3",
                    help="Scene name from AVAILABLE_SCENES")

# IsaacLab CLI args
from omni.isaac.lab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv

sys.path.insert(0, "/workspace/unitree_sim_isaaclab")
from dm_isaac_g1.inference.available_scenes import AVAILABLE_SCENES

scene_name = args_cli.scene
scene_entry = AVAILABLE_SCENES.get(scene_name)
if not scene_entry:
    print(f"Unknown scene: {scene_name}")
    print(f"Available: {list(AVAILABLE_SCENES.keys())}")
    sys.exit(1)

env_cfg = scene_entry["cfg_class"]()
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
    # Highlight hand joints
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
