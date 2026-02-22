#!/usr/bin/env python3
"""Test script to verify connection to GROOT inference server.

This script tests the ZeroMQ connection to the GROOT server running
on the DGX Spark server (192.168.1.237:5555).

Configured for the fine-tuned Dex3 28-DOF model with 4 cameras:
  - cam_left_high, cam_right_high, cam_left_wrist, cam_right_wrist
  - 28 DOF: left arm(7) + right arm(7) + left Dex3(7) + right Dex3(7)

Usage:
    python scripts/test_groot_connection.py
"""

import sys
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from dm_isaac_g1.core.config import load_config
from dm_isaac_g1.inference.client import GrootClient

# Dex3 28-DOF camera and joint layout
CAMERA_KEYS = ["cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"]
IMAGE_SIZE = (480, 640)  # H, W matching training data
NUM_DOF = 28


def main():
    """Test GROOT server connection."""
    print("=" * 60)
    print("GROOT Inference Server Connection Test")
    print("Model: Dex3 28-DOF (4 cameras)")
    print("=" * 60)

    # Load config
    config = load_config()
    print(f"\nServer: {config.groot_server_host}:{config.groot_server_port}")

    # Create client
    client = GrootClient(config=config)

    # Health check
    print("\n1. Health Check...")
    try:
        is_healthy = client.health_check()
        if is_healthy:
            print("   OK - Server is responding")
        else:
            print("   FAIL - Server not responding")
            print("\n   Make sure the GROOT server is running on Spark:")
            print("   ssh nvidia@192.168.1.237")
            print("   docker start groot-server")
            sys.exit(1)
    except Exception as e:
        print(f"   FAIL - Connection error: {e}")
        sys.exit(1)

    # Test inference with 4 cameras + 28 DOF
    print("\n2. Test Inference (28 DOF, 4 cameras)...")
    try:
        # Create dummy 28 DOF observation (G1+Dex3)
        # Left arm(7) + Right arm(7) + Left Dex3(7) + Right Dex3(7) = 28
        observation = np.zeros(NUM_DOF, dtype=np.float32)
        observation[0] = 0.2   # left shoulder pitch
        observation[7] = 0.2   # right shoulder pitch

        # Create 4 camera images (480x640 matching training)
        images = {
            key: np.random.randint(0, 255, (*IMAGE_SIZE, 3), dtype=np.uint8)
            for key in CAMERA_KEYS
        }

        print(f"   Observation: {observation.shape} ({NUM_DOF} DOF)")
        print(f"   Cameras: {len(images)} x {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")

        t0 = time.time()
        action = client.get_action(
            observation=observation,
            images=images,
            task="Stack the blocks on the table",
            action_horizon=16,
            execute_steps=1,
        )
        latency = time.time() - t0

        print(f"   OK - Got action prediction in {latency:.2f}s")
        print(f"   Action shape: {action.shape if hasattr(action, 'shape') else type(action)}")

        if isinstance(action, np.ndarray):
            print(f"   Action range: [{action.min():.4f}, {action.max():.4f}]")
            print(f"   Action mean: {action.mean():.4f}")

            # Show Dex3 28-DOF action breakdown
            print(f"\n   Action breakdown ({NUM_DOF} DOF):")
            print(f"   - Left arm  (0-6):   {np.array2string(action[:7], precision=4)}")
            print(f"   - Right arm (7-13):  {np.array2string(action[7:14], precision=4)}")
            print(f"   - Left Dex3 (14-20): {np.array2string(action[14:21], precision=4)}")
            print(f"   - Right Dex3(21-27): {np.array2string(action[21:28], precision=4)}")

    except Exception as e:
        print(f"   FAIL - Inference error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test with full trajectory
    print("\n3. Test Full Trajectory (16-step horizon)...")
    try:
        t0 = time.time()
        result = client.get_action(
            observation=observation,
            images=images,
            task="Pick up the red block from the table",
            action_horizon=16,
            execute_steps=8,
            return_full_trajectory=True,
        )
        latency = time.time() - t0

        print(f"   OK - Got trajectory in {latency:.2f}s")
        if isinstance(result, dict):
            act = result["action"]
            traj = result["trajectory"]
            print(f"   - Execute action: shape={act.shape}")
            print(f"   - Full trajectory: shape={traj.shape}")
            print(f"   - Horizon: {result.get('action_horizon', 'N/A')}")
        else:
            print(f"   - Result type: {type(result)}")

    except Exception as e:
        print(f"   FAIL - Trajectory error: {e}")
        import traceback
        traceback.print_exc()

    # Latency test
    print("\n4. Latency Test (3 consecutive inferences)...")
    try:
        latencies = []
        for i in range(3):
            obs = np.random.randn(NUM_DOF).astype(np.float32) * 0.1
            imgs = {
                key: np.random.randint(0, 255, (*IMAGE_SIZE, 3), dtype=np.uint8)
                for key in CAMERA_KEYS
            }
            t0 = time.time()
            client.get_action(observation=obs, images=imgs, task="Stack blocks")
            dt = time.time() - t0
            latencies.append(dt)
            print(f"   Inference {i+1}: {dt:.2f}s")

        avg = sum(latencies) / len(latencies)
        print(f"   Average latency: {avg:.2f}s")

    except Exception as e:
        print(f"   FAIL - Latency test error: {e}")

    # Clean up
    client.close()

    print("\n" + "=" * 60)
    print("Connection test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
