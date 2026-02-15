#!/usr/bin/env python3
"""
G1 Manipulation Policy Inference in MuJoCo

This script runs GROOT manipulation policies in MuJoCo (their native training environment).
This works because GROOT manipulation policies (PnPAppleToPlate, etc.) were trained on MuJoCo.

Usage:
    # Run PnPAppleToPlate demo
    python scripts/policy_inference_mujoco.py \
        --server-host 192.168.1.237 \
        --server-port 5555 \
        --task pnp_apple

    # Run with custom model
    python scripts/policy_inference_mujoco.py \
        --server-host 192.168.1.237 \
        --model nvidia/GR00T-N1.6-G1-PnPAppleToPlate
"""

import argparse
import os
import sys
import time
import numpy as np

# Check MuJoCo installation
try:
    import mujoco
    import mujoco.viewer
    print(f"MuJoCo version: {mujoco.__version__}")
except ImportError:
    print("Error: MuJoCo not installed")
    print("Install with: pip install mujoco mujoco-py gymnasium[mujoco]")
    sys.exit(1)

# Check GROOT client
try:
    from gr00t.policy.server_client import PolicyClient
except ImportError:
    print("Warning: gr00t package not found. Using ZMQ client directly.")
    PolicyClient = None

import zmq


def parse_args():
    parser = argparse.ArgumentParser(description="G1 Manipulation Inference in MuJoCo")
    parser.add_argument("--server-host", type=str, default="192.168.1.237",
                        help="GROOT server host IP")
    parser.add_argument("--server-port", type=int, default=5555,
                        help="GROOT server port")
    parser.add_argument("--task", type=str, default="pnp_apple",
                        choices=["pnp_apple", "manipulation", "demo"],
                        help="Task to run")
    parser.add_argument("--model", type=str, default="nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
                        help="Model name on GROOT server")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum simulation steps")
    parser.add_argument("--render", action="store_true", default=True,
                        help="Render visualization")
    parser.add_argument("--g1-model", type=str, default=None,
                        help="Path to G1 MJCF/XML model")
    return parser.parse_args()


class ZMQPolicyClient:
    """Simple ZMQ client for GROOT server when gr00t package not available."""

    def __init__(self, host: str, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"Connected to GROOT server at {host}:{port}")

    def get_action(self, observation: dict) -> dict:
        """Send observation to server and get action."""
        import msgpack
        import msgpack_numpy as m
        m.patch()

        # Serialize observation
        packed = msgpack.packb(observation, use_bin_type=True)
        self.socket.send(packed)

        # Receive action
        response = self.socket.recv()
        action = msgpack.unpackb(response, raw=False)
        return action

    def close(self):
        self.socket.close()
        self.context.term()


class G1MuJoCoEnv:
    """G1 robot environment in MuJoCo for manipulation tasks."""

    # G1 arm joint names (matching GROOT observation format)
    LEFT_ARM_JOINTS = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
    ]

    RIGHT_ARM_JOINTS = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
    ]

    WAIST_JOINTS = [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ]

    # Hand joints (simplified - 6 per hand for GROOT)
    LEFT_HAND_JOINTS = [
        "left_hand_index", "left_hand_middle",
        "left_hand_ring", "left_hand_pinky",
        "left_hand_thumb_0", "left_hand_thumb_1",
    ]

    RIGHT_HAND_JOINTS = [
        "right_hand_index", "right_hand_middle",
        "right_hand_ring", "right_hand_pinky",
        "right_hand_thumb_0", "right_hand_thumb_1",
    ]

    def __init__(self, model_path: str = None):
        """Initialize G1 MuJoCo environment."""
        self.model_path = model_path or self._find_g1_model()
        print(f"Loading G1 model from: {self.model_path}")

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Create renderer
        self.renderer = mujoco.Renderer(self.model, height=256, width=256)

        # Joint indices cache
        self._cache_joint_indices()

        # Simulation parameters
        self.dt = self.model.opt.timestep
        self.control_decimation = 10  # Control every N physics steps

    def _find_g1_model(self) -> str:
        """Find G1 MJCF model."""
        # Check common locations
        possible_paths = [
            # Local unitree_mujoco clone
            os.path.expanduser("~/unitree_mujoco/g1/scene.xml"),
            os.path.expanduser("~/unitree_mujoco/g1/g1.xml"),
            # LeRobot G1 model
            os.path.expanduser("~/.cache/huggingface/hub/models--lerobot--unitree-g1-mujoco/snapshots/*/scene.xml"),
            # Workspace
            "/workspace/unitree_mujoco/g1/scene.xml",
            "./models/g1/scene.xml",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Download from HuggingFace
        return self._download_g1_model()

    def _download_g1_model(self) -> str:
        """Download G1 model from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download

            print("Downloading G1 MuJoCo model from HuggingFace...")
            model_path = hf_hub_download(
                repo_id="lerobot/unitree-g1-mujoco",
                filename="scene.xml",
                local_dir="./models/g1_mujoco"
            )
            return model_path
        except Exception as e:
            raise RuntimeError(f"Failed to find or download G1 model: {e}")

    def _cache_joint_indices(self):
        """Cache joint indices for fast lookup."""
        self.joint_indices = {}

        for name in self.model.names.decode().split('\x00'):
            if name:
                try:
                    idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if idx >= 0:
                        self.joint_indices[name] = idx
                except Exception:
                    pass

    def get_joint_positions(self, joint_names: list) -> np.ndarray:
        """Get positions for specified joints."""
        positions = []
        for name in joint_names:
            if name in self.joint_indices:
                idx = self.joint_indices[name]
                qpos_addr = self.model.jnt_qposadr[idx]
                positions.append(self.data.qpos[qpos_addr])
            else:
                positions.append(0.0)
        return np.array(positions, dtype=np.float32)

    def set_joint_positions(self, joint_names: list, positions: np.ndarray):
        """Set positions for specified joints."""
        for name, pos in zip(joint_names, positions):
            if name in self.joint_indices:
                idx = self.joint_indices[name]
                qpos_addr = self.model.jnt_qposadr[idx]
                self.data.qpos[qpos_addr] = pos

    def get_camera_image(self, camera_name: str = "ego_view") -> np.ndarray:
        """Get RGB image from camera."""
        self.renderer.update_scene(self.data, camera=camera_name)
        img = self.renderer.render()
        return img

    def get_observation(self) -> dict:
        """
        Build observation in GROOT SimPolicyWrapper format.

        Returns:
            dict: Observation with keys:
                - video.ego_view: RGB image from head camera
                - state.left_arm: 7 joint positions
                - state.right_arm: 7 joint positions
                - state.left_hand: 6 hand joint positions
                - state.right_hand: 6 hand joint positions
                - state.waist: 3 torso joint positions
        """
        # Get camera image (256x256 RGB)
        try:
            ego_image = self.get_camera_image("head_camera")
        except Exception:
            # Fallback: use renderer with default camera
            self.renderer.update_scene(self.data)
            ego_image = self.renderer.render()

        # Get joint positions
        left_arm = self.get_joint_positions(self.LEFT_ARM_JOINTS)
        right_arm = self.get_joint_positions(self.RIGHT_ARM_JOINTS)
        left_hand = self.get_joint_positions(self.LEFT_HAND_JOINTS)
        right_hand = self.get_joint_positions(self.RIGHT_HAND_JOINTS)
        waist = self.get_joint_positions(self.WAIST_JOINTS)

        return {
            "video.ego_view": ego_image,
            "state.left_arm": left_arm,
            "state.right_arm": right_arm,
            "state.left_hand": left_hand,
            "state.right_hand": right_hand,
            "state.waist": waist,
        }

    def apply_action(self, action: dict):
        """
        Apply action from GROOT policy.

        Args:
            action: Dictionary with action trajectory (30 timesteps)
        """
        # GROOT outputs trajectory: (30, num_joints)
        if "action" in action:
            action_trajectory = action["action"]
        else:
            action_trajectory = action

        # Get current positions
        current_left_arm = self.get_joint_positions(self.LEFT_ARM_JOINTS)
        current_right_arm = self.get_joint_positions(self.RIGHT_ARM_JOINTS)
        current_left_hand = self.get_joint_positions(self.LEFT_HAND_JOINTS)
        current_right_hand = self.get_joint_positions(self.RIGHT_HAND_JOINTS)
        current_waist = self.get_joint_positions(self.WAIST_JOINTS)

        # Extract action for current timestep (first timestep of trajectory)
        if isinstance(action_trajectory, np.ndarray) and len(action_trajectory.shape) > 1:
            action_step = action_trajectory[0]  # First timestep
        else:
            action_step = action_trajectory

        # Parse action into joint groups (assuming GROOT action format)
        # Actions are deltas relative to current state
        idx = 0

        # Left arm (7 DOF)
        if idx + 7 <= len(action_step):
            left_arm_delta = action_step[idx:idx+7]
            new_left_arm = current_left_arm + left_arm_delta
            self.set_joint_positions(self.LEFT_ARM_JOINTS, new_left_arm)
            idx += 7

        # Right arm (7 DOF)
        if idx + 7 <= len(action_step):
            right_arm_delta = action_step[idx:idx+7]
            new_right_arm = current_right_arm + right_arm_delta
            self.set_joint_positions(self.RIGHT_ARM_JOINTS, new_right_arm)
            idx += 7

        # Left hand (6 DOF)
        if idx + 6 <= len(action_step):
            left_hand_delta = action_step[idx:idx+6]
            new_left_hand = current_left_hand + left_hand_delta
            self.set_joint_positions(self.LEFT_HAND_JOINTS, new_left_hand)
            idx += 6

        # Right hand (6 DOF)
        if idx + 6 <= len(action_step):
            right_hand_delta = action_step[idx:idx+6]
            new_right_hand = current_right_hand + right_hand_delta
            self.set_joint_positions(self.RIGHT_HAND_JOINTS, new_right_hand)
            idx += 6

        # Waist (3 DOF)
        if idx + 3 <= len(action_step):
            waist_delta = action_step[idx:idx+3]
            new_waist = current_waist + waist_delta
            self.set_joint_positions(self.WAIST_JOINTS, new_waist)

    def step(self):
        """Step the simulation forward."""
        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)

    def reset(self):
        """Reset the simulation."""
        mujoco.mj_resetData(self.model, self.data)
        # Set to standing pose
        self._set_initial_pose()

    def _set_initial_pose(self):
        """Set robot to initial standing pose."""
        # Typical standing pose for G1
        initial_positions = {
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_elbow_pitch_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_elbow_pitch_joint": 0.0,
        }

        for name, pos in initial_positions.items():
            if name in self.joint_indices:
                idx = self.joint_indices[name]
                qpos_addr = self.model.jnt_qposadr[idx]
                self.data.qpos[qpos_addr] = pos

    def close(self):
        """Clean up resources."""
        self.renderer.close()


def run_manipulation_inference(args):
    """Run GROOT manipulation inference in MuJoCo."""

    print(f"\n{'='*60}")
    print("G1 Manipulation Inference in MuJoCo")
    print(f"{'='*60}")
    print(f"Server: {args.server_host}:{args.server_port}")
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"{'='*60}\n")

    # Create MuJoCo environment
    env = G1MuJoCoEnv(model_path=args.g1_model)

    # Connect to GROOT server
    if PolicyClient is not None:
        client = PolicyClient(host=args.server_host, port=args.server_port)
    else:
        client = ZMQPolicyClient(host=args.server_host, port=args.server_port)

    # Reset environment
    env.reset()

    # Create viewer if rendering
    viewer = None
    if args.render:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    try:
        for step in range(args.max_steps):
            # Get observation
            obs = env.get_observation()

            # Get action from GROOT server
            action = client.get_action(obs)

            # Apply action
            env.apply_action(action)

            # Step simulation
            env.step()

            # Update viewer
            if viewer is not None:
                viewer.sync()

            # Log progress
            if step % 50 == 0:
                print(f"Step {step}/{args.max_steps}")

            # Control rate
            time.sleep(env.dt * env.control_decimation)

    except KeyboardInterrupt:
        print("\nStopping inference...")

    finally:
        # Cleanup
        client.close()
        env.close()
        if viewer is not None:
            viewer.close()

    print(f"Completed {step} steps")


def main():
    args = parse_args()
    run_manipulation_inference(args)


if __name__ == "__main__":
    main()
