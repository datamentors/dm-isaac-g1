#!/usr/bin/env python3
"""
Phase 1: Run G1 Robot with GROOT Inference
This script runs the G1 robot in Isaac Sim using GROOT policy inference.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phases.phase1_inference.groot_client import GrootClient


def parse_args():
    parser = argparse.ArgumentParser(description="Run G1 with GROOT inference")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no visualization)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="Number of simulation steps",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="stand",
        choices=["stand", "walk", "reach", "wave"],
        help="Task to perform",
    )
    parser.add_argument(
        "--groot-host",
        type=str,
        default=None,
        help="GROOT server host (overrides env)",
    )
    parser.add_argument(
        "--groot-port",
        type=int,
        default=None,
        help="GROOT server port (overrides env)",
    )
    return parser.parse_args()


def main():
    # Load environment variables
    load_dotenv(PROJECT_ROOT / ".env")

    args = parse_args()

    print("=" * 60)
    print("Phase 1: G1 Robot with GROOT Inference")
    print("=" * 60)

    # Initialize GROOT client
    print("\nConnecting to GROOT server...")
    groot = GrootClient(
        host=args.groot_host,
        port=args.groot_port,
    )

    # Check GROOT server health
    if groot.health_check():
        print(f"GROOT server healthy at {groot.base_url}")
    else:
        print(f"Warning: Cannot connect to GROOT server at {groot.base_url}")
        print("Running in demonstration mode without inference...")
        groot = None

    # Initialize Isaac Sim
    print("\nInitializing Isaac Sim...")
    try:
        from omni.isaac.kit import SimulationApp

        simulation_app = SimulationApp({
            "headless": args.headless,
            "width": 1280,
            "height": 720,
        })

        from phases.phase1_inference.g1_scene import G1Scene

        # Create scene
        scene = G1Scene(
            config_path=str(PROJECT_ROOT / "configs" / "scene.yaml"),
        )
        scene.setup()
        print("Scene created successfully!")

    except ImportError as e:
        print(f"Isaac Sim not available: {e}")
        print("\nTo run this script, ensure:")
        print("1. You are on the Blackwell workstation")
        print("2. Isaac Sim is installed and configured")
        print("3. Run via: ~/.local/share/ov/pkg/isaac_sim-*/python.sh run_inference.py")
        return

    # Task-specific setup
    task_descriptions = {
        "stand": "Stand still and maintain balance",
        "walk": "Walk forward slowly",
        "reach": "Reach forward with right arm",
        "wave": "Wave right arm side to side",
    }

    task_desc = task_descriptions.get(args.task, "stand")
    print(f"\nTask: {args.task} - {task_desc}")

    # Run simulation loop
    print(f"\nRunning simulation for {args.num_steps} steps...")
    step_count = 0

    try:
        while step_count < args.num_steps:
            # Get robot state
            state = scene.get_robot_state()

            if state:
                # Build observation
                joint_pos = state.get("joint_positions", np.zeros(29))
                joint_vel = state.get("joint_velocities", np.zeros(29))
                observation = np.concatenate([joint_pos, joint_vel])

                # Get action from GROOT
                if groot is not None:
                    try:
                        action = groot.get_action(
                            observation=observation,
                            task_description=task_desc,
                        )
                        scene.set_robot_action(action.tolist())
                    except Exception as e:
                        print(f"Inference error: {e}")
                else:
                    # Demo mode: simple standing pose
                    default_action = np.zeros(29)
                    scene.set_robot_action(default_action.tolist())

            # Step simulation
            scene.step()
            step_count += 1

            # Progress update
            if step_count % 100 == 0:
                print(f"Step {step_count}/{args.num_steps}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")

    finally:
        # Cleanup
        print("\nCleaning up...")
        scene.close()
        if groot is not None:
            groot.close()
        simulation_app.close()

    print("Done!")


if __name__ == "__main__":
    main()
