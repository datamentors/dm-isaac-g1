#!/usr/bin/env python3
"""
G1 Locomotion Policy Inference in Isaac Sim/Isaac Lab

This script runs pre-trained locomotion policies for Unitree G1 in Isaac Sim.
Locomotion policies work in Isaac Sim because they were trained in Isaac Gym
(same PhysX engine).

Usage:
    # From Isaac Lab container on workstation (192.168.1.205)
    python scripts/policy_inference_locomotion.py --task Isaac-Velocity-Flat-G1-Play-v0

    # With custom policy checkpoint
    python scripts/policy_inference_locomotion.py --checkpoint /path/to/model.pt
"""

import argparse
import os
import sys
import torch
import numpy as np

# Isaac Lab imports (will be available inside Isaac Lab container)
try:
    from omni.isaac.lab.app import AppLauncher
except ImportError:
    print("Error: This script must be run inside Isaac Lab environment")
    print("On workstation, run: docker exec -it isaac-lab bash")
    sys.exit(1)

# Parse arguments before AppLauncher
parser = argparse.ArgumentParser(description="G1 Locomotion Inference in Isaac Lab")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-Play-v0",
                    choices=[
                        "Isaac-Velocity-Flat-G1-Play-v0",
                        "Isaac-Velocity-Rough-G1-Play-v0",
                        "Unitree-G1-29dof-Velocity"  # unitree_rl_lab task
                    ],
                    help="Isaac Lab task name")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to policy checkpoint (.pt file)")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel environments")
parser.add_argument("--max_steps", type=int, default=1000,
                    help="Maximum simulation steps")
parser.add_argument("--video", action="store_true",
                    help="Record video of the simulation")
parser.add_argument("--video_dir", type=str, default="./videos",
                    help="Directory to save videos")

# Add AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after AppLauncher
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.utils.math import quat_rotate_inverse, normalize

# Import environment configurations
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, load_cfg_from_registry


def download_default_checkpoint():
    """Download default G1 locomotion checkpoint from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download

        print("Downloading G1 locomotion policy from HuggingFace...")
        checkpoint_path = hf_hub_download(
            repo_id="hardware-pathon-ai/unitree-g1-phase1-locomotion",
            filename="model_5000.pt",
            local_dir="./checkpoints/g1_locomotion"
        )
        return checkpoint_path
    except ImportError:
        print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"Failed to download checkpoint: {e}")
        return None


class LocomotionPolicyRunner:
    """Runs pre-trained locomotion policy in Isaac Lab."""

    def __init__(self, env: ManagerBasedRLEnv, checkpoint_path: str):
        self.env = env
        self.device = env.device

        # Load policy
        self.policy = self._load_policy(checkpoint_path)

        # Observation normalization stats (from training)
        self.obs_mean = None
        self.obs_std = None

        # Previous actions for observation
        self.prev_actions = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim),
            device=self.device
        )

    def _load_policy(self, checkpoint_path: str):
        """Load policy from checkpoint."""
        print(f"Loading policy from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if "model" in checkpoint:
            # rsl_rl format
            policy = checkpoint["model"]
        elif "actor" in checkpoint:
            # Separate actor-critic format
            policy = checkpoint["actor"]
        else:
            # Assume entire checkpoint is the model
            policy = checkpoint

        policy.to(self.device)
        policy.eval()

        # Load normalization stats if available
        if "obs_mean" in checkpoint:
            self.obs_mean = checkpoint["obs_mean"].to(self.device)
            self.obs_std = checkpoint["obs_std"].to(self.device)

        return policy

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations using training stats."""
        if self.obs_mean is not None and self.obs_std is not None:
            return (obs - self.obs_mean) / (self.obs_std + 1e-8)
        return obs

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from policy."""
        with torch.no_grad():
            obs_normalized = self._normalize_obs(obs)
            action = self.policy(obs_normalized)

            # Clamp actions to valid range
            action = torch.clamp(action, -1.0, 1.0)

        return action

    def run(self, max_steps: int = 1000, velocity_command: tuple = (0.5, 0.0, 0.0)):
        """
        Run the locomotion policy.

        Args:
            max_steps: Maximum number of simulation steps
            velocity_command: (vx, vy, yaw_rate) velocity command
        """
        print(f"\n{'='*60}")
        print("G1 Locomotion Inference")
        print(f"{'='*60}")
        print(f"Task: {args.task}")
        print(f"Velocity command: vx={velocity_command[0]:.2f}, vy={velocity_command[1]:.2f}, yaw={velocity_command[2]:.2f}")
        print(f"Max steps: {max_steps}")
        print(f"{'='*60}\n")

        # Reset environment
        obs, info = self.env.reset()

        step = 0
        episode_reward = 0.0

        try:
            while step < max_steps and simulation_app.is_running():
                # Set velocity command (if environment supports it)
                if hasattr(self.env, "command_manager"):
                    self.env.command_manager.set_command(
                        "base_velocity",
                        torch.tensor([velocity_command], device=self.device)
                    )

                # Build observation tensor
                obs_tensor = self._build_observation_tensor(obs)

                # Get action from policy
                action = self.get_action(obs_tensor)

                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)

                # Update previous actions
                self.prev_actions = action.clone()

                episode_reward += reward.mean().item()
                step += 1

                # Log every 100 steps
                if step % 100 == 0:
                    print(f"Step {step}: reward={reward.mean().item():.4f}, cumulative={episode_reward:.4f}")

                # Handle resets
                if terminated.any() or truncated.any():
                    print(f"Episode ended at step {step}. Resetting...")
                    obs, info = self.env.reset()
                    episode_reward = 0.0

        except KeyboardInterrupt:
            print("\nStopping inference...")

        print(f"\nCompleted {step} steps with total reward: {episode_reward:.4f}")

    def _build_observation_tensor(self, obs_dict) -> torch.Tensor:
        """Build flat observation tensor from observation dictionary."""
        # Standard locomotion observation structure:
        # [base_ang_vel (3), projected_gravity (3), joint_pos (29), joint_vel (29), prev_actions (29)]

        if isinstance(obs_dict, torch.Tensor):
            return obs_dict

        obs_parts = []

        # Base angular velocity
        if "base_ang_vel" in obs_dict:
            obs_parts.append(obs_dict["base_ang_vel"])
        elif "root_ang_vel" in obs_dict:
            obs_parts.append(obs_dict["root_ang_vel"])

        # Projected gravity
        if "projected_gravity" in obs_dict:
            obs_parts.append(obs_dict["projected_gravity"])
        elif "proj_gravity" in obs_dict:
            obs_parts.append(obs_dict["proj_gravity"])

        # Joint positions
        if "joint_pos" in obs_dict:
            obs_parts.append(obs_dict["joint_pos"])
        elif "dof_pos" in obs_dict:
            obs_parts.append(obs_dict["dof_pos"])

        # Joint velocities
        if "joint_vel" in obs_dict:
            obs_parts.append(obs_dict["joint_vel"])
        elif "dof_vel" in obs_dict:
            obs_parts.append(obs_dict["dof_vel"])

        # Previous actions
        obs_parts.append(self.prev_actions)

        # Concatenate all parts
        if obs_parts:
            return torch.cat(obs_parts, dim=-1)
        else:
            # Fallback: return policy observation directly
            return obs_dict.get("policy", obs_dict)


def main():
    """Main entry point."""

    # Determine checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Try to get default checkpoint
        checkpoint_path = get_checkpoint_path(args.task) if hasattr(sys.modules[__name__], 'get_checkpoint_path') else None

        if checkpoint_path is None:
            checkpoint_path = download_default_checkpoint()

        if checkpoint_path is None:
            print("No checkpoint specified and failed to download default.")
            print("Please provide --checkpoint path or install huggingface_hub")
            simulation_app.close()
            return

    # Load environment configuration
    print(f"Loading environment: {args.task}")

    try:
        env_cfg = load_cfg_from_registry(args.task, "env_cfg")
        env_cfg.scene.num_envs = args.num_envs

        # Create environment
        env = ManagerBasedRLEnv(cfg=env_cfg)

    except Exception as e:
        print(f"Failed to create environment: {e}")
        print("\nTrying alternative environment setup...")

        # Alternative: Use direct gym registration
        import gymnasium as gym
        env = gym.make(args.task, num_envs=args.num_envs)

    # Create policy runner
    runner = LocomotionPolicyRunner(env, checkpoint_path)

    # Define velocity commands to test
    velocity_commands = [
        (0.5, 0.0, 0.0),   # Forward
        (0.0, 0.0, 0.5),   # Turn left
        (-0.3, 0.0, 0.0),  # Backward
        (0.5, 0.2, 0.0),   # Forward + strafe
    ]

    # Run with first command
    runner.run(max_steps=args.max_steps, velocity_command=velocity_commands[0])

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
