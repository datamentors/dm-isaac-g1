#!/usr/bin/env python3
"""Test script for Isaac Sim integration with GROOT inference.

This script runs inference testing in Isaac Sim environments that
match the training data used for the fine-tuned model.

Training Data Match:
- G1_Dex3_BlockStacking → Isaac-Stack-RgyBlock-G129-Inspire-Joint
- G1_Fold_Towel, G1_Clean_Table, etc. → Isaac-PickPlace-* environments

Usage:
    # Run on workstation with Isaac Sim installed
    python scripts/test_isaac_sim.py --env stack_blocks --episodes 5

    # Or via CLI
    dm-g1 infer benchmark --env Isaac-Stack-RgyBlock-G129-Inspire-Joint --episodes 5
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Test GROOT inference with Isaac Sim")
    parser.add_argument(
        "--env",
        type=str,
        default="stack_blocks",
        choices=["stack_blocks", "pick_redblock", "pick_cylinder", "move_cylinder"],
        help="Isaac Sim environment to test",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=16,
        help="GROOT action prediction horizon (8-16 recommended)",
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=1,
        help="Steps to execute before re-planning (1 for receding horizon)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (requires display)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record episode videos",
    )
    parser.add_argument(
        "--groot-host",
        type=str,
        default=None,
        help="GROOT server host (default: from config)",
    )
    parser.add_argument(
        "--groot-port",
        type=int,
        default=None,
        help="GROOT server port (default: from config)",
    )

    args = parser.parse_args()

    # Map env names to IsaacEnv values
    env_map = {
        "stack_blocks": "STACK_BLOCKS",
        "pick_redblock": "PICK_REDBLOCK",
        "pick_cylinder": "PICK_CYLINDER",
        "move_cylinder": "MOVE_CYLINDER",
    }

    # Task descriptions matching training data
    task_map = {
        "stack_blocks": "Stack the colored blocks on top of each other",
        "pick_redblock": "Pick up the red block and place it in the target area",
        "pick_cylinder": "Pick up the cylinder and place it in the target area",
        "move_cylinder": "Move the cylinder to the target position",
    }

    print("=" * 60)
    print("Isaac Sim + GROOT Inference Test")
    print("=" * 60)
    print(f"\nEnvironment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Action horizon: {args.action_horizon}")
    print(f"Execute steps: {args.execute_steps}")
    print(f"Control strategy: {'Receding Horizon (MPC)' if args.execute_steps == 1 else 'Partial Execution'}")

    # Import Isaac Sim runner
    try:
        from dm_isaac_g1.inference.isaac_runner import IsaacSimRunner, IsaacEnv

        # Get environment enum
        env_enum = getattr(IsaacEnv, env_map[args.env])
        task = task_map[args.env]

        print(f"\nTask: {task}")
        print(f"Full env name: {env_enum.value}")

        # Create runner
        runner = IsaacSimRunner()

        # Setup with GROOT server
        print("\nConnecting to GROOT server...")
        success = runner.setup(
            env=env_enum,
            groot_host=args.groot_host,
            groot_port=args.groot_port,
        )

        if not success:
            print("Failed to connect to GROOT server!")
            print("\nMake sure the server is running on Spark:")
            print("  ssh nvidia@192.168.1.237")
            print("  docker exec -it groot-server bash")
            print("  cd /workspace/gr00t && python gr00t/eval/run_gr00t_server.py ...")
            sys.exit(1)

        # Run benchmark
        print(f"\nRunning {args.episodes} episodes...")
        results = runner.run_benchmark(
            env=env_enum,
            num_episodes=args.episodes,
            task=task,
        )

        # Print results
        print("\n" + "=" * 60)
        print("Benchmark Results")
        print("=" * 60)
        print(f"Environment: {results['environment']}")
        print(f"Episodes: {results['num_episodes']}")
        print(f"Success Rate: {results['success_rate']*100:.1f}%")
        print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean Steps: {results['mean_steps']:.1f}")

        print("\nPer-episode results:")
        for i, ep in enumerate(results["results"]):
            status = "✓" if ep["success"] else "✗"
            print(f"  Episode {i+1}: {status} reward={ep['reward']:.2f}, steps={ep['steps']}")

        # Cleanup
        runner.close()

    except ImportError as e:
        print(f"\nError: Could not import Isaac Sim runner: {e}")
        print("\nThis script must be run on a machine with Isaac Sim installed.")
        print("Run on the workstation (192.168.1.205) with Isaac Lab environment.")
        print("\nAlternatively, use the remote runner:")
        print("  dm-g1 infer benchmark --env Isaac-Stack-RgyBlock-G129-Inspire-Joint")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
