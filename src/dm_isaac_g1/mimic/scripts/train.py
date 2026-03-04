#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""Training script for mimic (motion tracking) policies.

Wraps the standard RSL-RL training loop with wandb integration.
Uses unitree_rl_lab's train.py infrastructure.

Usage:
    python -u src/dm_isaac_g1/mimic/scripts/train.py \
        --task DM-G1-29dof-Mimic-RonaldoCelebration \
        --num_envs 4096 --max_iterations 30000 --headless \
        --logger wandb --log_project_name dm-isaac-g1-mimic
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# -- CLI args ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train a mimic motion-tracking policy.")
parser.add_argument("--task", type=str, required=True, help="Gymnasium task ID")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=30000, help="Max PPO iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--logger", type=str, default="wandb", choices=["wandb", "tensorboard", "neptune"])
parser.add_argument("--log_project_name", type=str, default="dm-isaac-g1-mimic")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- Imports after IsaacLab init -----------------------------------------------
import gymnasium as gym
import torch

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# Register our mimic tasks
import dm_isaac_g1.mimic.tasks  # noqa: F401


def main():
    # Create environment
    env_cfg_entry = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    rsl_rl_cfg_entry = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]

    # Load configs
    module_path, class_name = env_cfg_entry.rsplit(":", 1)
    import importlib
    env_cfg_module = importlib.import_module(module_path)
    env_cfg = getattr(env_cfg_module, class_name)()

    module_path, class_name = rsl_rl_cfg_entry.rsplit(":", 1)
    agent_cfg_module = importlib.import_module(module_path)
    agent_cfg: RslRlOnPolicyRunnerCfg = getattr(agent_cfg_module, class_name)()

    # Override from CLI
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    agent_cfg.experiment_name = args_cli.task

    if args_cli.logger == "wandb":
        agent_cfg.logger = "wandb"
        agent_cfg.wandb_project = args_cli.log_project_name

    # Log directory — use MOTION_NAME env var as run name for WandB identification
    motion_name = os.environ.get("MOTION_NAME", "")
    run_name = f"{motion_name}_seed{args_cli.seed}" if motion_name else f"seed_{args_cli.seed}"
    log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_dir = os.path.join(log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Dump configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    print("[INFO] Logging to:", log_dir)
    print_dict(agent_cfg.to_dict(), nesting=4)

    # Create env + wrapper
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # Resume if requested
    if args_cli.resume:
        runner.load(args_cli.resume)
        print(f"[INFO] Resumed from: {args_cli.resume}")

    # Train
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Export policy
    policy_path = os.path.join(log_dir, "exported")
    os.makedirs(policy_path, exist_ok=True)
    runner.save(os.path.join(log_dir, "model_final.pt"))
    print(f"[INFO] Training complete. Model saved to: {log_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
