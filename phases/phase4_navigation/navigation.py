#!/usr/bin/env python3
"""
Phase 4: Navigation + Inference + RL
Combines navigation, GROOT inference, and RL for autonomous behavior.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Navigation + Inference + RL")
    parser.add_argument(
        "--map-size",
        type=float,
        nargs=2,
        default=[20.0, 20.0],
        help="Navigation map size (x, y) in meters",
    )
    parser.add_argument(
        "--num-waypoints",
        type=int,
        default=5,
        help="Number of waypoints to navigate",
    )
    parser.add_argument(
        "--groot-host",
        type=str,
        default=None,
        help="GROOT server host",
    )
    parser.add_argument(
        "--rl-checkpoint",
        type=str,
        default=None,
        help="Path to RL policy checkpoint",
    )
    return parser.parse_args()


class NavigationController:
    """High-level navigation controller for G1."""

    def __init__(
        self,
        map_size: tuple = (20.0, 20.0),
        device: str = "cuda",
    ):
        self.map_size = map_size
        self.device = device

        # Navigation state
        self.current_position = np.array([0.0, 0.0])
        self.current_heading = 0.0
        self.waypoints = []
        self.current_waypoint_idx = 0

    def set_waypoints(self, waypoints: list) -> None:
        """Set navigation waypoints."""
        self.waypoints = waypoints
        self.current_waypoint_idx = 0

    def get_current_target(self) -> Optional[np.ndarray]:
        """Get current navigation target."""
        if self.current_waypoint_idx >= len(self.waypoints):
            return None
        return np.array(self.waypoints[self.current_waypoint_idx])

    def update_position(self, position: np.ndarray, heading: float) -> None:
        """Update current position from robot state."""
        self.current_position = position[:2]
        self.current_heading = heading

    def get_velocity_command(self) -> np.ndarray:
        """Compute velocity command towards current waypoint."""
        target = self.get_current_target()
        if target is None:
            return np.zeros(3)  # Stop

        # Compute direction to target
        direction = target - self.current_position
        distance = np.linalg.norm(direction)

        # Check if waypoint reached
        if distance < 0.3:  # 30cm threshold
            self.current_waypoint_idx += 1
            print(f"Waypoint {self.current_waypoint_idx} reached!")
            return self.get_velocity_command()

        # Compute velocity command
        direction_normalized = direction / (distance + 1e-6)

        # Desired heading
        desired_heading = np.arctan2(direction_normalized[1], direction_normalized[0])
        heading_error = desired_heading - self.current_heading

        # Wrap heading error
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi

        # Velocity command: [vx, vy, angular_vel]
        max_linear_vel = 0.5  # m/s
        max_angular_vel = 1.0  # rad/s

        linear_vel = min(max_linear_vel, distance)
        angular_vel = np.clip(heading_error * 2.0, -max_angular_vel, max_angular_vel)

        return np.array([
            linear_vel * np.cos(self.current_heading),
            linear_vel * np.sin(self.current_heading),
            angular_vel,
        ])

    def is_navigation_complete(self) -> bool:
        """Check if all waypoints have been reached."""
        return self.current_waypoint_idx >= len(self.waypoints)


class AutonomousG1:
    """Autonomous G1 controller combining navigation, inference, and RL."""

    def __init__(
        self,
        groot_client=None,
        rl_policy=None,
        device: str = "cuda",
    ):
        self.groot_client = groot_client
        self.rl_policy = rl_policy
        self.device = device

        self.navigation = NavigationController()

        # Mode: "navigate", "manipulate", "idle"
        self.mode = "idle"

    def set_mode(self, mode: str) -> None:
        """Set operating mode."""
        self.mode = mode
        print(f"Mode set to: {mode}")

    def get_action(
        self,
        observation: dict,
        task: Optional[str] = None,
    ) -> np.ndarray:
        """Get action based on current mode and observation."""

        if self.mode == "navigate":
            # Use RL policy for locomotion with navigation commands
            velocity_cmd = self.navigation.get_velocity_command()

            if self.rl_policy is not None:
                obs_tensor = {
                    k: torch.tensor(v, device=self.device).unsqueeze(0)
                    for k, v in observation.items()
                }
                # Add velocity command to observation
                action, _, _ = self.rl_policy.get_action(obs_tensor, deterministic=True)
                return action.cpu().numpy()[0]
            else:
                # Fallback: return zeros
                return np.zeros(17)

        elif self.mode == "manipulate":
            # Use GROOT inference for manipulation
            if self.groot_client is not None:
                obs_array = np.concatenate([
                    observation.get("joint_pos", np.zeros(29)),
                    observation.get("joint_vel", np.zeros(29)),
                ])
                return self.groot_client.get_action(obs_array, task_description=task)
            else:
                return np.zeros(17)

        else:  # idle
            return np.zeros(17)


def main():
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    print("=" * 60)
    print("Phase 4: Navigation + Inference + RL")
    print("=" * 60)

    # Create navigation controller
    nav = NavigationController(map_size=tuple(args.map_size))

    # Generate random waypoints
    waypoints = []
    for _ in range(args.num_waypoints):
        x = np.random.uniform(-args.map_size[0] / 2, args.map_size[0] / 2)
        y = np.random.uniform(-args.map_size[1] / 2, args.map_size[1] / 2)
        waypoints.append([x, y])

    nav.set_waypoints(waypoints)
    print(f"\nGenerated {args.num_waypoints} waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"  {i+1}: ({wp[0]:.2f}, {wp[1]:.2f})")

    # Create autonomous controller
    autonomous = AutonomousG1()
    autonomous.navigation = nav

    print("\nPhase 4 setup complete!")
    print("Full implementation requires:")
    print("  1. Isaac Lab environment with navigation map")
    print("  2. GROOT inference integration")
    print("  3. RL policy from Phase 2/3")
    print("  4. Object detection for manipulation targets")


if __name__ == "__main__":
    main()
