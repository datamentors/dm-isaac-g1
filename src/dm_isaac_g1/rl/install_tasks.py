#!/usr/bin/env python3
"""
Install dm-isaac-g1 custom RL tasks into a vanilla unitree_rl_lab clone.

This script does two things:
  1. Appends custom reward functions (arm_leg_coordination, lateral_velocity_penalty)
     to unitree_rl_lab's locomotion mdp/rewards.py — needed by MilitaryMarch task.
  2. Copies dm-isaac-g1 task configs into unitree_rl_lab's G1 task directory so they
     are discoverable by the training scripts.

Usage:
    python -m dm_isaac_g1.rl.install_tasks \\
        --unitree-rl-lab /workspace/unitree_rl_lab \\
        --rewards-patch /workspace/dm-isaac-g1/environments/workstation/patches/rewards_custom.py

Called automatically during Docker build. Can also be run manually inside the
container to refresh tasks after a git pull.
"""

import argparse
import shutil
import sys
from pathlib import Path

# Marker used to detect if custom rewards have already been appended.
MARKER = "# === DATAMENTORS CUSTOM REWARDS ==="


def patch_rewards(rewards_path: Path, patch_path: Path) -> None:
    """Append custom reward functions to upstream rewards.py if not already present."""
    rewards_text = rewards_path.read_text()
    if MARKER in rewards_text:
        print(f"  [skip] rewards.py already patched ({rewards_path})")
        return

    patch_text = patch_path.read_text()
    with open(rewards_path, "a") as f:
        f.write(f"\n\n{MARKER}\n")
        f.write(patch_text)
    print(f"  [ok] Appended custom rewards to {rewards_path}")


def copy_task_configs(dm_tasks_dir: Path, unitree_g1_dir: Path) -> None:
    """Copy dm-isaac-g1 task directories into unitree_rl_lab's G1 task directory."""
    if not dm_tasks_dir.is_dir():
        print(f"  [warn] dm-isaac-g1 tasks dir not found: {dm_tasks_dir}")
        return

    for task_dir in sorted(dm_tasks_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("_"):
            continue

        dest = unitree_g1_dir / task_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(task_dir, dest)
        print(f"  [ok] Copied task: {task_dir.name} -> {dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unitree-rl-lab",
        type=Path,
        default=Path("/workspace/unitree_rl_lab"),
        help="Path to unitree_rl_lab clone",
    )
    parser.add_argument(
        "--rewards-patch",
        type=Path,
        default=Path("/workspace/dm-isaac-g1/environments/workstation/patches/rewards_custom.py"),
        help="Path to rewards_custom.py patch file",
    )
    parser.add_argument(
        "--dm-tasks-dir",
        type=Path,
        default=Path("/workspace/dm-isaac-g1/src/dm_isaac_g1/rl/tasks"),
        help="Path to dm-isaac-g1 RL tasks directory",
    )
    args = parser.parse_args()

    rl_lab = args.unitree_rl_lab
    rewards_py = rl_lab / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py"
    g1_tasks_dir = rl_lab / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1"

    # Validate paths
    if not rewards_py.exists():
        print(f"ERROR: rewards.py not found at {rewards_py}", file=sys.stderr)
        sys.exit(1)
    if not g1_tasks_dir.exists():
        print(f"ERROR: G1 tasks dir not found at {g1_tasks_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.rewards_patch.exists():
        print(f"ERROR: Patch file not found at {args.rewards_patch}", file=sys.stderr)
        sys.exit(1)

    print("=== Installing dm-isaac-g1 RL tasks into unitree_rl_lab ===")

    print("\n1. Patching rewards.py with custom functions...")
    patch_rewards(rewards_py, args.rewards_patch)

    print("\n2. Copying task configs...")
    copy_task_configs(args.dm_tasks_dir, g1_tasks_dir)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
