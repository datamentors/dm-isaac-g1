#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""Play (inference) script for trained RL locomotion policies.

Loads a trained checkpoint, exports ONNX/JIT + deploy.yaml, and runs inference.

Usage:
    python src/dm_isaac_g1/rl/scripts/play.py \
        --task DM-G1-29dof-MilitaryMarch \
        --checkpoint logs/rsl_rl/DM-G1-29dof-MilitaryMarch/seed_42/model_final.pt

    # Export only (no sim playback):
    python src/dm_isaac_g1/rl/scripts/play.py \
        --task DM-G1-29dof-MilitaryMarch \
        --checkpoint logs/rsl_rl/.../model_final.pt --export_only --headless

    # Record video (headless):
    python src/dm_isaac_g1/rl/scripts/play.py \
        --task DM-G1-29dof-MilitaryMarch \
        --checkpoint logs/rsl_rl/.../model_18500.pt --video --video_length 300 --headless
"""

import argparse
import importlib
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained RL locomotion policy.")
parser.add_argument("--task", type=str, required=True, help="Gymnasium task ID")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments for playback")
parser.add_argument("--export_only", action="store_true", help="Export ONNX/JIT and exit without running sim")
parser.add_argument("--video", action="store_true", help="Record video during playback")
parser.add_argument("--video_length", type=int, default=300, help="Length of recorded video (in steps)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab.utils.dict import print_dict
from rsl_rl.runners import OnPolicyRunner

import dm_isaac_g1.rl.tasks  # noqa: F401
try:
    import unitree_rl_lab.tasks  # noqa: F401
except ImportError:
    pass

from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg


def main():
    play_cfg_entry = gym.spec(args_cli.task).kwargs["play_env_cfg_entry_point"]
    rsl_rl_cfg_entry = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]

    module_path, class_name = play_cfg_entry.rsplit(":", 1)
    env_cfg = getattr(importlib.import_module(module_path), class_name)()

    module_path, class_name = rsl_rl_cfg_entry.rsplit(":", 1)
    agent_cfg = getattr(importlib.import_module(module_path), class_name)()

    env_cfg.scene.num_envs = args_cli.num_envs

    log_dir = os.path.dirname(args_cli.checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")

    # Extract policy network and normalizer (matching unitree_rl_lab pattern)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # Export ONNX + JIT
    export_dir = os.path.join(log_dir, "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
    print(f"[INFO] Exported policy to {export_dir}/policy.onnx and policy.pt")

    # Export deployment config (joint maps, PD gains, observation scales, etc.)
    export_deploy_cfg(env.unwrapped, log_dir)
    print(f"[INFO] Exported deploy config to {log_dir}/params/deploy.yaml")

    if args_cli.export_only:
        print("[INFO] Export-only mode, skipping sim playback.")
        env.close()
        return

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs, _ = env.get_observations()

    timestep = 0
    print("[INFO] Running inference... Press Ctrl+C to stop.")
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
