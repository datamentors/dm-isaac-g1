#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""Sim2Sim: Deploy an Isaac Lab trained policy in MuJoCo.

Loads a trained RL/Mimic policy (ONNX or JIT) exported by play.py,
along with the deploy.yaml config, and runs it in a MuJoCo simulation
of the Unitree G1 robot.

Supports:
  - Headless mode (ECS batch rendering with EGL)
  - GUI mode (VNC/display, interactive MuJoCo viewer)
  - Video recording (mp4)
  - Both ONNX and TorchScript (JIT) policy formats
  - Interactive keyboard control (WASD/QE)
  - FSM state machine (Passive/Stand/Walk)
  - Data-driven observations from deploy.yaml
  - Observation history stacking
  - Mimic policies with motion reference files

Usage:
    # Headless with video recording (ECS/cloud):
    python src/dm_isaac_g1/rl/scripts/sim2sim.py \
        --policy exported/policy.onnx \
        --deploy-yaml params/deploy.yaml \
        --headless --video --video-length 10

    # Interactive GUI with keyboard control:
    python src/dm_isaac_g1/rl/scripts/sim2sim.py \
        --policy exported/policy.pt \
        --deploy-yaml params/deploy.yaml \
        --interactive

    # Or use the CLI:
    dm-g1 sim2sim policy.onnx -i
    dm-g1 sim2sim policy.onnx --headless --video

    # Mimic policies:
    dm-g1 sim2sim policy.onnx --motion-file dance.npz --video
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Sim2Sim: Isaac Lab policy -> MuJoCo")
    parser.add_argument("--policy", type=str, required=True,
                        help="Path to exported policy (ONNX or JIT .pt)")
    parser.add_argument("--deploy-yaml", type=str, default=None,
                        help="Path to deploy.yaml (auto-detected if not set)")
    parser.add_argument("--scene", type=str, default=None,
                        help="Path to MuJoCo scene XML (auto-detected if not set)")
    parser.add_argument("--headless", action="store_true",
                        help="Run headless (EGL rendering, no GUI window)")
    parser.add_argument("--video", action="store_true",
                        help="Record video of the simulation")
    parser.add_argument("--video-length", type=float, default=10.0,
                        help="Video length in seconds (default: 10)")
    parser.add_argument("--video-fps", type=int, default=30,
                        help="Video FPS (default: 30)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for videos (default: same as policy dir)")
    parser.add_argument("--sim-duration", type=float, default=60.0,
                        help="Total simulation duration in seconds (GUI mode, default: 60)")
    parser.add_argument("--cmd-vx", type=float, default=0.5,
                        help="Forward velocity command (default: 0.5 m/s)")
    parser.add_argument("--cmd-vy", type=float, default=0.0,
                        help="Lateral velocity command (default: 0.0 m/s)")
    parser.add_argument("--cmd-wz", type=float, default=0.0,
                        help="Yaw rate command (default: 0.0 rad/s)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Enable keyboard control (WASD/QE + FSM keys)")
    parser.add_argument("--motion-file", type=str, default=None,
                        help="Motion NPZ file for mimic policies")
    parser.add_argument("--debug-obs", action="store_true",
                        help="Print observation debug info on first step")
    return parser.parse_args()


def main():
    args = parse_args()

    from dm_isaac_g1.sim2sim.runner import run

    run(
        policy_path=args.policy,
        deploy_yaml=args.deploy_yaml,
        scene=args.scene,
        headless=args.headless,
        video=args.video,
        video_length=args.video_length,
        video_fps=args.video_fps,
        output_dir=args.output_dir,
        sim_duration=args.sim_duration,
        cmd_vx=args.cmd_vx,
        cmd_vy=args.cmd_vy,
        cmd_wz=args.cmd_wz,
        interactive=args.interactive,
        motion_file=args.motion_file,
        debug_obs=args.debug_obs,
    )


if __name__ == "__main__":
    main()
