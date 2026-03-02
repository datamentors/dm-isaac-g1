#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""Play (inference) script for trained mimic policies.

Loads a trained checkpoint and runs inference with video recording + policy export.

Usage:
    python src/dm_isaac_g1/mimic/scripts/play.py \
        --task DM-G1-29dof-Mimic-RonaldoCelebration \
        --checkpoint logs/rsl_rl/DM-G1-29dof-Mimic-RonaldoCelebration/seed_42/model_final.pt
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained mimic policy.")
parser.add_argument("--task", type=str, required=True, help="Gymnasium task ID")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments for playback")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import importlib

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import dm_isaac_g1.mimic.tasks  # noqa: F401


def main():
    play_cfg_entry = gym.spec(args_cli.task).kwargs["play_env_cfg_entry_point"]
    rsl_rl_cfg_entry = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]

    module_path, class_name = play_cfg_entry.rsplit(":", 1)
    env_cfg = getattr(importlib.import_module(module_path), class_name)()

    module_path, class_name = rsl_rl_cfg_entry.rsplit(":", 1)
    agent_cfg = getattr(importlib.import_module(module_path), class_name)()

    env_cfg.scene.num_envs = args_cli.num_envs

    log_dir = os.path.dirname(args_cli.checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")

    # Export policy
    export_dir = os.path.join(log_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)

    policy = runner.get_inference_policy(device=env.device)
    obs, _ = env.get_observations()

    print("[INFO] Running inference... Press Ctrl+C to stop.")
    while simulation_app.is_running():
        actions = policy(obs)
        obs, _, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
