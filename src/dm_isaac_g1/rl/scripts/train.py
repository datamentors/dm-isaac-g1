#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""Training script for RL locomotion policies.

Wraps the standard RSL-RL training loop with wandb integration.
Uses unitree_rl_lab's training infrastructure.

Usage:
    python -u src/dm_isaac_g1/rl/scripts/train.py \
        --task DM-G1-29dof-MilitaryMarch \
        --num_envs 4096 --max_iterations 50000 --headless \
        --logger wandb --log_project_name dm-isaac-g1-rl
"""

import argparse
import importlib
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL locomotion policy.")
parser.add_argument("--task", type=str, required=True, help="Gymnasium task ID")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=50000, help="Max PPO iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--logger", type=str, default="wandb", choices=["wandb", "tensorboard", "neptune"])
parser.add_argument("--log_project_name", type=str, default="dm-isaac-g1-rl")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# Register our RL tasks
import dm_isaac_g1.rl.tasks  # noqa: F401

# Also ensure unitree_rl_lab tasks are registered (for tasks that re-use them)
try:
    import unitree_rl_lab.tasks  # noqa: F401
except ImportError:
    pass


def main():
    env_cfg_entry = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    rsl_rl_cfg_entry = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]

    module_path, class_name = env_cfg_entry.rsplit(":", 1)
    env_cfg = getattr(importlib.import_module(module_path), class_name)()

    module_path, class_name = rsl_rl_cfg_entry.rsplit(":", 1)
    agent_cfg: RslRlOnPolicyRunnerCfg = getattr(importlib.import_module(module_path), class_name)()

    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    agent_cfg.experiment_name = args_cli.task

    if args_cli.logger == "wandb":
        agent_cfg.logger = "wandb"
        agent_cfg.wandb_project = args_cli.log_project_name

    log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_dir = os.path.join(log_root, f"seed_{args_cli.seed}")
    os.makedirs(log_dir, exist_ok=True)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    print("[INFO] Logging to:", log_dir)
    print_dict(agent_cfg.to_dict(), nesting=4)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if args_cli.resume:
        runner.load(args_cli.resume)
        print(f"[INFO] Resumed from: {args_cli.resume}")

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    runner.save(os.path.join(log_dir, "model_final.pt"))
    print(f"[INFO] Training complete. Model saved to: {log_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
