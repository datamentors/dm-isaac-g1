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

Usage:
    # Headless with video recording (ECS/cloud):
    python src/dm_isaac_g1/rl/scripts/sim2sim.py \
        --policy exported/policy.onnx \
        --deploy-yaml params/deploy.yaml \
        --headless --video --video-length 10

    # Interactive GUI (VNC or local display):
    python src/dm_isaac_g1/rl/scripts/sim2sim.py \
        --policy exported/policy.pt \
        --deploy-yaml params/deploy.yaml

    # With custom MuJoCo scene:
    python src/dm_isaac_g1/rl/scripts/sim2sim.py \
        --policy exported/policy.onnx \
        --deploy-yaml params/deploy.yaml \
        --scene /workspace/unitree_mujoco/unitree_robots/g1/scene.xml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Sim2Sim: Isaac Lab policy → MuJoCo")
    parser.add_argument("--policy", type=str, required=True,
                        help="Path to exported policy (ONNX or JIT .pt)")
    parser.add_argument("--deploy-yaml", type=str, required=True,
                        help="Path to deploy.yaml (from play.py export)")
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
    return parser.parse_args()


def find_mujoco_scene():
    """Auto-detect G1 MuJoCo scene XML."""
    candidates = [
        "/workspace/unitree_mujoco/unitree_robots/g1/scene.xml",
        "/workspace/unitree_model/G1/g1.xml",
        os.path.expanduser("~/unitree_mujoco/unitree_robots/g1/scene.xml"),
    ]
    # Also search common repo locations
    for base in ["/workspace", os.path.expanduser("~"), "."]:
        candidates.append(os.path.join(base, "unitree_rl_lab", "deploy",
                                       "robots", "g1_29dof", "resources", "scene.xml"))

    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_policy(policy_path):
    """Load ONNX or JIT policy."""
    ext = Path(policy_path).suffix.lower()

    if ext == ".onnx":
        import onnxruntime as ort
        session = ort.InferenceSession(policy_path,
                                       providers=["CUDAExecutionProvider",
                                                  "CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_name = session.get_outputs()[0].name

        obs_dim = input_shape[-1]
        print(f"[sim2sim] ONNX policy: input={input_name} shape={input_shape}, "
              f"output={output_name}")

        def infer(obs):
            obs_np = obs.astype(np.float32).reshape(1, -1)
            result = session.run([output_name], {input_name: obs_np})
            return result[0].flatten()

        return infer, obs_dim

    elif ext == ".pt":
        import torch
        model = torch.jit.load(policy_path, map_location="cpu")
        model.eval()

        # Probe input size from first layer
        obs_dim = None
        for name, param in model.named_parameters():
            if "weight" in name:
                obs_dim = param.shape[1]
                break
        if obs_dim is None:
            obs_dim = 47  # fallback for G1 29dof

        print(f"[sim2sim] JIT policy: obs_dim={obs_dim}")

        def infer(obs):
            with torch.inference_mode():
                obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                action = model(obs_t)
                return action.squeeze(0).numpy()

        return infer, obs_dim

    else:
        raise ValueError(f"Unsupported policy format: {ext} (use .onnx or .pt)")


def load_deploy_config(yaml_path):
    """Load deploy.yaml and extract joint config, PD gains, obs/action scaling."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    print(f"[sim2sim] Deploy config keys: {list(cfg.keys())}")
    return cfg


def build_observation(data, cfg, cmd_vel, last_actions, obs_dim):
    """Build the observation vector matching the Isaac Lab training format.

    Standard G1 29-DOF observation (obs_dim ~47-73):
        - base_ang_vel (3) — angular velocity in base frame
        - velocity_commands (3) — [vx, vy, wz]
        - joint_pos (29) — relative to default
        - joint_vel (29) — scaled
        - last_actions (29)
    """
    # Base angular velocity (body frame)
    # MuJoCo stores angular velocity in world frame in qvel[3:6]
    base_ang_vel = data.qvel[3:6].copy()

    # Joint positions relative to default
    n_joints = data.ctrl.shape[0]
    default_pos = np.zeros(n_joints)
    if "default_joint_pos" in cfg:
        dp = cfg["default_joint_pos"]
        default_pos[:len(dp)] = dp[:n_joints]

    # Map joint IDs if available
    joint_ids = list(range(n_joints))
    if "joint_ids_map" in cfg:
        joint_ids = cfg["joint_ids_map"][:n_joints]

    # Get joint positions and velocities from MuJoCo
    # MuJoCo qpos: [7 (root) + n_joints], qvel: [6 (root) + n_joints]
    n_qpos_joints = min(n_joints, data.qpos.shape[0] - 7)
    n_qvel_joints = min(n_joints, data.qvel.shape[0] - 6)

    joint_pos = data.qpos[7:7 + n_qpos_joints].copy()
    joint_vel = data.qvel[6:6 + n_qvel_joints].copy()

    # Apply offsets (relative to default)
    joint_pos_rel = joint_pos - default_pos[:n_qpos_joints]

    # Apply scaling from deploy config
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    ang_vel_scale = 0.25
    cmd_scale = np.array([2.0, 2.0, 0.25])

    if "observations" in cfg:
        for obs_name, obs_cfg in cfg["observations"].items():
            if "scale" in obs_cfg and obs_cfg["scale"]:
                # Could use per-term scales; for now use global defaults
                pass

    # Build observation vector
    obs_parts = [
        base_ang_vel * ang_vel_scale,                    # 3
        cmd_vel * cmd_scale,                             # 3
        joint_pos_rel * dof_pos_scale,                   # n_joints
        joint_vel * dof_vel_scale,                       # n_joints
    ]

    # Add last actions if we have room
    if last_actions is not None:
        obs_parts.append(last_actions[:n_joints])        # n_joints

    obs = np.concatenate(obs_parts).astype(np.float32)

    # Pad or trim to expected obs_dim
    if len(obs) < obs_dim:
        obs = np.pad(obs, (0, obs_dim - len(obs)))
    elif len(obs) > obs_dim:
        obs = obs[:obs_dim]

    return obs


def apply_actions(data, actions, cfg):
    """Apply policy actions to MuJoCo actuators using PD control."""
    n_joints = data.ctrl.shape[0]
    n_act = min(len(actions), n_joints)

    # Get action scaling
    action_scale = 0.25  # default
    action_offset = np.zeros(n_joints)

    if "actions" in cfg:
        for act_name, act_cfg in cfg["actions"].items():
            if "scale" in act_cfg and act_cfg["scale"]:
                scales = act_cfg["scale"]
                if isinstance(scales, list):
                    action_scale = np.array(scales[:n_act])
                elif isinstance(scales, (int, float)):
                    action_scale = float(scales)
            if "offset" in act_cfg and act_cfg["offset"]:
                offsets = act_cfg["offset"]
                if isinstance(offsets, list):
                    action_offset[:len(offsets)] = offsets[:n_joints]

    # Scale and offset actions
    scaled_actions = actions[:n_act] * action_scale
    if isinstance(action_offset, np.ndarray):
        target_pos = scaled_actions + action_offset[:n_act]
    else:
        target_pos = scaled_actions + action_offset

    # Get PD gains from deploy config
    kp = np.ones(n_joints) * 100.0  # default stiffness
    kd = np.ones(n_joints) * 2.0    # default damping

    if "stiffness" in cfg:
        kp_cfg = cfg["stiffness"]
        kp[:len(kp_cfg)] = kp_cfg[:n_joints]
    if "damping" in cfg:
        kd_cfg = cfg["damping"]
        kd[:len(kd_cfg)] = kd_cfg[:n_joints]

    # PD control: torque = kp * (target - current) - kd * vel
    n_qpos = min(n_act, data.qpos.shape[0] - 7)
    n_qvel = min(n_act, data.qvel.shape[0] - 6)

    current_pos = data.qpos[7:7 + n_qpos]
    current_vel = data.qvel[6:6 + n_qvel]

    torque = kp[:n_act] * (target_pos[:n_qpos] - current_pos) - kd[:n_act] * current_vel[:n_qvel]

    # Apply torques (clipped to actuator limits)
    data.ctrl[:n_act] = np.clip(torque,
                                data.model.actuator_ctrlrange[:n_act, 0],
                                data.model.actuator_ctrlrange[:n_act, 1])


def main():
    args = parse_args()

    # Set rendering backend before importing mujoco
    if args.headless:
        os.environ["MUJOCO_GL"] = "egl"

    import mujoco

    # Load policy
    print(f"[sim2sim] Loading policy: {args.policy}")
    infer_fn, obs_dim = load_policy(args.policy)

    # Load deploy config
    print(f"[sim2sim] Loading deploy config: {args.deploy_yaml}")
    deploy_cfg = load_deploy_config(args.deploy_yaml)

    # Find MuJoCo scene
    scene_path = args.scene or find_mujoco_scene()
    if scene_path is None or not os.path.exists(scene_path):
        print("[sim2sim] ERROR: No G1 MuJoCo scene XML found.")
        print("[sim2sim] Please specify --scene or clone unitree_mujoco:")
        print("[sim2sim]   git clone https://github.com/unitreerobotics/unitree_mujoco.git")
        sys.exit(1)

    print(f"[sim2sim] MuJoCo scene: {scene_path}")

    # Load model
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    n_joints = model.nu
    dt = model.opt.timestep
    print(f"[sim2sim] Model: {n_joints} actuators, dt={dt:.4f}s")

    # Determine control decimation from deploy config
    step_dt = deploy_cfg.get("step_dt", 0.02)  # default 50 Hz
    control_decimation = max(1, int(round(step_dt / dt)))
    print(f"[sim2sim] Control: step_dt={step_dt:.4f}s, decimation={control_decimation}")

    # Velocity commands
    cmd_vel = np.array([args.cmd_vx, args.cmd_vy, args.cmd_wz], dtype=np.float32)
    print(f"[sim2sim] Velocity commands: vx={cmd_vel[0]:.2f}, vy={cmd_vel[1]:.2f}, "
          f"wz={cmd_vel[2]:.2f}")

    # Output directory
    output_dir = args.output_dir or str(Path(args.policy).parent)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    last_actions = np.zeros(n_joints, dtype=np.float32)

    if args.headless or args.video:
        # ── Headless / video recording mode ──────────────────────────
        renderer = mujoco.Renderer(model, height=720, width=1280)
        fps = args.video_fps
        video_duration = args.video_length if args.video else args.sim_duration
        total_steps = int(video_duration / step_dt)
        frames_per_step = max(1, int(1.0 / (step_dt * fps)))

        frames = []
        print(f"[sim2sim] Running {video_duration:.1f}s simulation "
              f"({total_steps} control steps)...")

        for step in range(total_steps):
            # Build observation
            obs = build_observation(data, deploy_cfg, cmd_vel, last_actions, obs_dim)

            # Policy inference
            actions = infer_fn(obs)
            last_actions = actions.copy()

            # Apply actions with PD control
            apply_actions(data, actions, deploy_cfg)

            # Step physics (decimation)
            for _ in range(control_decimation):
                mujoco.mj_step(model, data)

            # Capture frame for video (at video FPS rate)
            if args.video and (step % max(1, total_steps // (int(video_duration * fps)))) == 0:
                renderer.update_scene(data)
                frame = renderer.render()
                frames.append(frame.copy())

            # Progress
            elapsed = step * step_dt
            if (step + 1) % (total_steps // 10) == 0:
                print(f"  [{elapsed:.1f}s / {video_duration:.1f}s] "
                      f"step {step + 1}/{total_steps}")

        # Save video
        if args.video and frames:
            try:
                import cv2
                video_path = os.path.join(output_dir, "sim2sim.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = frames[0].shape[:2]
                writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
                for frame in frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()
                print(f"[sim2sim] Video saved: {video_path} ({len(frames)} frames)")
            except ImportError:
                print("[sim2sim] WARNING: opencv not available, skipping video save")
                # Fallback: save frames as images
                frames_dir = os.path.join(output_dir, "sim2sim_frames")
                os.makedirs(frames_dir, exist_ok=True)
                for i, frame in enumerate(frames):
                    from PIL import Image
                    Image.fromarray(frame).save(os.path.join(frames_dir, f"frame_{i:05d}.png"))
                print(f"[sim2sim] Frames saved to: {frames_dir}")

        renderer.close()
        print("[sim2sim] Headless simulation complete.")

    else:
        # ── GUI / interactive mode (VNC or local display) ────────────
        print("[sim2sim] Starting interactive MuJoCo viewer...")
        print("[sim2sim] Press Ctrl+C to stop.")

        viewer = mujoco.viewer.launch_passive(model, data)
        step_count = 0
        sim_time = 0.0

        try:
            while viewer.is_running() and sim_time < args.sim_duration:
                # Build observation
                obs = build_observation(data, deploy_cfg, cmd_vel, last_actions, obs_dim)

                # Policy inference
                actions = infer_fn(obs)
                last_actions = actions.copy()

                # Apply actions
                apply_actions(data, actions, deploy_cfg)

                # Step physics
                for _ in range(control_decimation):
                    mujoco.mj_step(model, data)

                viewer.sync()
                sim_time += step_dt
                step_count += 1

                # Real-time sync
                time.sleep(max(0, step_dt - 0.001))

                if step_count % 500 == 0:
                    print(f"  [sim2sim] t={sim_time:.1f}s, steps={step_count}")

        except KeyboardInterrupt:
            print("\n[sim2sim] Stopped by user.")
        finally:
            viewer.close()

        print(f"[sim2sim] Interactive session ended. "
              f"Ran {step_count} steps ({sim_time:.1f}s)")


if __name__ == "__main__":
    main()
