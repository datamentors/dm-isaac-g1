#!/usr/bin/env python3
"""
Test GROOT UNITREE_G1 inference without Isaac Sim.

Sends a synthetic observation to the GROOT server in UNITREE_G1 format
(flat dot-separated keys) and prints the action response. This verifies
the observation format is correct before running full Isaac Sim inference.

Usage:
    # From workstation (dm-workstation container):
    PYTHONPATH=/workspace/Isaac-GR00T:$PYTHONPATH \
    python scripts/test_groot_unitree_g1.py \
        --server 192.168.1.237:5555 \
        --language "fold the towel"

    # With statistics file (prints state mean for reference):
    GR00T_STATS=/workspace/checkpoints/groot-g1-gripper-hospitality-7ds/processor/statistics.json \
    PYTHONPATH=/workspace/Isaac-GR00T:$PYTHONPATH \
    python scripts/test_groot_unitree_g1.py \
        --server 192.168.1.237:5555
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Test GROOT UNITREE_G1 observation format")
    parser.add_argument("--server", type=str, required=True, help="host:port, e.g. 192.168.1.237:5555")
    parser.add_argument("--language", type=str, default="fold the towel", help="Language command")
    parser.add_argument("--num_requests", type=int, default=3, help="Number of inference requests to send")
    args = parser.parse_args()

    # Parse server address
    if ":" not in args.server:
        raise ValueError("--server must be host:port format")
    host, port_str = args.server.split(":", 1)
    port = int(port_str)

    # Import PolicyClient
    try:
        from gr00t.policy.server_client import PolicyClient
    except ImportError:
        print("[ERROR] Cannot import gr00t. Set PYTHONPATH=/workspace/Isaac-GR00T:$PYTHONPATH")
        sys.exit(1)

    # Connect
    print(f"[INFO] Connecting to GROOT server at {host}:{port}...")
    client = PolicyClient(host=host, port=port, strict=False)

    # Get modality config
    try:
        modality_cfg = client.get_modality_config()
        print(f"[INFO] Server modality config keys: {list(modality_cfg.keys())}")
        for mod_name, mod_cfg in modality_cfg.items():
            print(f"  {mod_name}: keys={mod_cfg.modality_keys}, "
                  f"delta_indices={mod_cfg.delta_indices}")
    except Exception as e:
        print(f"[WARN] Could not get modality config: {e}")
        modality_cfg = {}

    # Load statistics if available (for reference)
    stats_path = os.environ.get("GR00T_STATS")
    if stats_path and os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        for emb_key in ["unitree_g1", "new_embodiment", "NEW_EMBODIMENT"]:
            if emb_key in stats:
                print(f"\n[INFO] Statistics embodiment: {emb_key}")
                state_stats = stats[emb_key].get("state", {})
                for state_key, state_info in state_stats.items():
                    if "mean" in state_info:
                        mean = np.array(state_info["mean"])
                        print(f"  state.{state_key}: dim={len(mean)}, "
                              f"mean_range=[{mean.min():.3f}, {mean.max():.3f}]")
                break
        print()

    # Build synthetic UNITREE_G1 observation
    # State: 31 DOF — left_leg(6) + right_leg(6) + waist(3) + left_arm(7) + right_arm(7) + left_hand(1) + right_hand(1)
    state_layout = {
        "left_leg":   6,
        "right_leg":  6,
        "waist":      3,
        "left_arm":   7,
        "right_arm":  7,
        "left_hand":  1,
        "right_hand": 1,
    }

    # Use training mean from stats if available, otherwise zeros
    state_values = {}
    if stats_path and os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        for emb_key in ["unitree_g1", "new_embodiment"]:
            if emb_key in stats:
                for part_name, dof in state_layout.items():
                    part_stats = stats[emb_key]["state"].get(part_name)
                    if part_stats and "mean" in part_stats:
                        state_values[part_name] = np.array(part_stats["mean"][:dof], dtype=np.float32)
                    else:
                        state_values[part_name] = np.zeros(dof, dtype=np.float32)
                break

    if not state_values:
        for part_name, dof in state_layout.items():
            state_values[part_name] = np.zeros(dof, dtype=np.float32)

    # Build observation dict — UNITREE_G1 flat format
    # Camera: synthetic 480x640 gray image
    ego_view = np.full((1, 1, 480, 640, 3), 128, dtype=np.uint8)

    observation = {
        "video.ego_view": ego_view,
        "annotation.human.task_description": (args.language,),
    }

    # Add state parts as flat keys
    for part_name, vals in state_values.items():
        observation[f"state.{part_name}"] = vals.reshape(1, 1, -1).astype(np.float32)

    # Print observation summary
    print("[INFO] Observation dict:")
    for k, v in observation.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min()}, {v.max()}]")
        else:
            print(f"  {k}: {v}")

    # Send requests
    print(f"\n[INFO] Sending {args.num_requests} inference requests...")
    client.reset()

    for i in range(args.num_requests):
        t0 = time.time()
        try:
            action_dict, info = client.get_action(observation)
            dt = time.time() - t0
            print(f"\n[INFO] Request {i+1}/{args.num_requests} — {dt:.3f}s")
            print(f"  Action keys: {list(action_dict.keys())}")
            for k, v in action_dict.items():
                arr = np.asarray(v)
                print(f"  {k}: shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]")
                if arr.ndim >= 3 and arr.shape[1] > 0:
                    # Print first timestep values
                    first_step = arr[0, 0, :]
                    print(f"    step[0]: {first_step}")
        except Exception as e:
            dt = time.time() - t0
            print(f"\n[ERROR] Request {i+1}/{args.num_requests} failed ({dt:.3f}s): {e}")
            import traceback
            traceback.print_exc()

    print("\n[INFO] Test complete.")


if __name__ == "__main__":
    main()
