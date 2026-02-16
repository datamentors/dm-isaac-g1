#!/usr/bin/env python3
"""Test script to verify connection to GROOT inference server.

This script tests the ZeroMQ connection to the GROOT server running
on the DGX Spark server (192.168.1.237:5555).

Usage:
    python scripts/test_groot_connection.py

Expected output:
    - Server health check status
    - Test inference with dummy observation
    - Action prediction shape and values
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from dm_isaac_g1.core.config import load_config
from dm_isaac_g1.inference.client import GrootClient


def main():
    """Test GROOT server connection."""
    print("=" * 60)
    print("GROOT Inference Server Connection Test")
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
            print("   ✓ Server is responding")
        else:
            print("   ✗ Server not responding")
            print("\n   Make sure the GROOT server is running on Spark:")
            print("   ssh nvidia@192.168.1.237")
            print("   docker exec -it groot-server bash")
            print("   cd /workspace/gr00t && python gr00t/eval/run_gr00t_server.py \\")
            print("       --model-path /workspace/gr00t/checkpoints/groot-g1-inspire-9datasets \\")
            print("       --embodiment-tag new_embodiment --port 5555")
            sys.exit(1)
    except Exception as e:
        print(f"   ✗ Connection error: {e}")
        sys.exit(1)

    # Test inference
    print("\n2. Test Inference...")
    try:
        # Create dummy 53 DOF observation (G1+Inspire)
        # Legs(12) + Waist(3) + Arms(14) + Hands(24) = 53
        observation = np.zeros(53, dtype=np.float32)

        # Set some reasonable initial pose
        # Waist slightly forward
        observation[14] = 0.1  # waist_pitch

        # Arms in natural position
        observation[15] = 0.2  # left_shoulder_pitch
        observation[22] = 0.2  # right_shoulder_pitch

        # Create dummy image (GROOT requires video input)
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        print(f"   Input observation shape: {observation.shape}")
        print(f"   Input image shape: {image.shape}")

        # Get action with receding horizon (predict 16, execute 1)
        action = client.get_action(
            observation=observation,
            image=image,
            task="Stand ready for manipulation task",
            action_horizon=16,
            execute_steps=1,
        )

        print(f"   ✓ Got action prediction!")
        print(f"   Action shape: {action.shape if hasattr(action, 'shape') else type(action)}")

        if isinstance(action, np.ndarray):
            print(f"   Action range: [{action.min():.4f}, {action.max():.4f}]")
            print(f"   Action mean: {action.mean():.4f}")

            # Show action breakdown
            print("\n   Action breakdown (53 DOF):")
            print(f"   - Left leg  (0-5):   {action[0:6]}")
            print(f"   - Right leg (6-11):  {action[6:12]}")
            print(f"   - Waist    (12-14):  {action[12:15]}")
            print(f"   - Left arm (15-21):  {action[15:22]}")
            print(f"   - Right arm(22-28):  {action[22:29]}")
            print(f"   - Left hand(29-40):  {action[29:41]}")
            print(f"   - Right hand(41-52): {action[41:53]}")

    except Exception as e:
        print(f"   ✗ Inference error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test with full trajectory
    print("\n3. Test Full Trajectory...")
    try:
        result = client.get_action(
            observation=observation,
            image=image,
            task="Pick up the red block from the table",
            action_horizon=16,
            execute_steps=8,
            return_full_trajectory=True,
        )

        print(f"   ✓ Got trajectory prediction!")
        if isinstance(result, dict):
            print(f"   - Action shape: {result['action'].shape if hasattr(result['action'], 'shape') else type(result['action'])}")
            print(f"   - Trajectory shape: {result['trajectory'].shape if hasattr(result['trajectory'], 'shape') else type(result['trajectory'])}")
            print(f"   - Horizon: {result.get('action_horizon', 'N/A')}")
        else:
            print(f"   - Result type: {type(result)}")

    except Exception as e:
        print(f"   ✗ Trajectory error: {e}")
        import traceback
        traceback.print_exc()

    # Clean up
    client.close()

    print("\n" + "=" * 60)
    print("Connection test completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Set up Isaac Sim environment on workstation")
    print("2. Run: dm-g1 infer benchmark --env Isaac-Stack-RgyBlock-G129-Inspire-Joint")
    print("3. Or test with: python scripts/test_isaac_sim.py")


if __name__ == "__main__":
    main()
