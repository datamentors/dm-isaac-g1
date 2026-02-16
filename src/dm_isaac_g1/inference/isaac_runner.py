"""Isaac Sim environment runner for testing GROOT inference.

Provides integration with Unitree's Isaac Sim environments for
testing the fine-tuned GROOT model in simulation.

GR00T N1.6 Action Chunking Strategy
------------------------------------
GR00T 1.6 uses Flow Matching with a 32-layer DiT to predict action chunks.
Key parameters:

- action_horizon: Number of steps to predict (8-16 recommended, configurable)
- execute_steps: How many of the predicted steps to execute before re-planning

Control Strategies:
1. Receding Horizon (MPC-style, recommended):
   - action_horizon=16, execute_steps=1
   - Re-plan every step for maximum robustness
   - Best for dynamic environments and disturbance rejection

2. Partial Execution:
   - action_horizon=16, execute_steps=8
   - Execute half, then re-plan
   - Balance between efficiency and robustness

3. Open-Loop (fast but less robust):
   - action_horizon=8, execute_steps=8
   - Execute full trajectory before re-planning
   - Use only in controlled, predictable environments

WARNING: Single-step prediction (action_horizon=1) causes jittering and
significantly degrades performance. Always use action_horizon >= 8.

Supported environments from unitree_sim_isaaclab:
- Isaac-PickPlace-Cylinder-G129-Inspire-Joint
- Isaac-PickPlace-RedBlock-G129-Inspire-Joint
- Isaac-Stack-RgyBlock-G129-Inspire-Joint
- Isaac-Move-Cylinder-G129-Inspire-Wholebody
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any

import numpy as np

from dm_isaac_g1.core.config import Config, load_config
from dm_isaac_g1.core.remote import WorkstationConnection
from dm_isaac_g1.inference.client import GrootClient


class IsaacEnv(Enum):
    """Available Isaac Sim environments for G1 + Inspire."""

    PICK_CYLINDER = "Isaac-PickPlace-Cylinder-G129-Inspire-Joint"
    PICK_REDBLOCK = "Isaac-PickPlace-RedBlock-G129-Inspire-Joint"
    STACK_BLOCKS = "Isaac-Stack-RgyBlock-G129-Inspire-Joint"
    MOVE_CYLINDER = "Isaac-Move-Cylinder-G129-Inspire-Wholebody"

    # Additional environments from unitree_sim_isaaclab
    G1_LOCOMOTION = "Isaac-Velocity-Flat-G1-v0"
    G1_ROUGH = "Isaac-Velocity-Rough-G1-v0"


@dataclass
class EpisodeResult:
    """Result of running an episode."""

    success: bool
    total_reward: float
    steps: int
    final_observation: Optional[np.ndarray] = None
    metrics: Optional[Dict[str, Any]] = None


class IsaacSimRunner:
    """Runner for testing GROOT inference in Isaac Sim environments.

    Connects to the workstation's Isaac Sim container to run
    simulated episodes using the GROOT model for control.

    Example:
        ```python
        runner = IsaacSimRunner()
        runner.setup(IsaacEnv.PICK_REDBLOCK)
        result = runner.run_episode(task="Pick up the red block")
        print(f"Success: {result.success}, Reward: {result.total_reward}")
        ```
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize Isaac Sim runner.

        Args:
            config: Configuration instance.
        """
        self.config = config or load_config()
        self.workstation = WorkstationConnection(config=self.config)
        self.groot_client: Optional[GrootClient] = None
        self.current_env: Optional[IsaacEnv] = None
        self._session_id: Optional[str] = None

    def setup(
        self,
        env: IsaacEnv = IsaacEnv.PICK_REDBLOCK,
        groot_host: Optional[str] = None,
        groot_port: Optional[int] = None,
    ) -> bool:
        """Setup the environment and GROOT client.

        Args:
            env: Isaac environment to use.
            groot_host: GROOT server host.
            groot_port: GROOT server port.

        Returns:
            True if setup successful.
        """
        self.current_env = env

        # Connect to GROOT server
        self.groot_client = GrootClient(
            host=groot_host,
            port=groot_port,
            config=self.config,
        )

        if not self.groot_client.health_check():
            print(f"Warning: GROOT server not responding at {self.groot_client.base_url}")
            return False

        print(f"Connected to GROOT server at {self.groot_client.base_url}")
        print(f"Environment: {env.value}")
        return True

    def run_episode(
        self,
        task: str = "Complete the manipulation task",
        max_steps: int = 500,
        action_horizon: int = 16,
        execute_steps: int = 1,
        render: bool = False,
        record: bool = False,
    ) -> EpisodeResult:
        """Run a single episode using GROOT for control.

        This executes the simulation on the workstation and uses
        the GROOT server for action inference.

        Args:
            task: Task description for language conditioning.
            max_steps: Maximum steps per episode.
            action_horizon: Number of steps GROOT predicts (8-16 recommended).
            execute_steps: Steps to execute before re-planning (1=receding horizon).
            render: Whether to render the simulation.
            record: Whether to record video.

        Returns:
            EpisodeResult with success status and metrics.
        """
        if self.groot_client is None or self.current_env is None:
            raise RuntimeError("Must call setup() before run_episode()")

        # Generate session ID for this episode
        import uuid
        self._session_id = str(uuid.uuid4())[:8]

        # Build the episode runner script
        script = self._generate_episode_script(
            env=self.current_env,
            task=task,
            max_steps=max_steps,
            action_horizon=action_horizon,
            execute_steps=execute_steps,
            render=render,
            record=record,
        )

        # Write script to container
        script_path = f"/tmp/episode_{self._session_id}.py"
        self.workstation.docker_exec(
            f"cat > {script_path} << 'SCRIPT_EOF'\n{script}\nSCRIPT_EOF",
            activate_env=False,
        )

        # Run the episode
        try:
            stdout, stderr, code = self.workstation.docker_exec(
                f"python {script_path}",
                timeout=max_steps * 2,  # Allow enough time
            )

            # Parse results from stdout
            return self._parse_episode_result(stdout)

        except Exception as e:
            print(f"Episode failed: {e}")
            return EpisodeResult(success=False, total_reward=0.0, steps=0)

    def _generate_episode_script(
        self,
        env: IsaacEnv,
        task: str,
        max_steps: int,
        action_horizon: int,
        execute_steps: int,
        render: bool,
        record: bool,
    ) -> str:
        """Generate Python script to run episode on workstation.

        Args:
            env: Environment to use.
            task: Task description.
            max_steps: Maximum steps.
            action_horizon: Steps GROOT predicts.
            execute_steps: Steps to execute before re-planning.
            render: Whether to render.
            record: Whether to record.

        Returns:
            Python script as string.
        """
        return f'''#!/usr/bin/env python3
"""Auto-generated episode runner for GROOT inference testing.

Action Chunking Configuration:
- action_horizon: {action_horizon} (steps predicted by GROOT)
- execute_steps: {execute_steps} (steps executed before re-planning)
"""
import io
import json
import numpy as np
import zmq
import msgpack

# Configuration
GROOT_HOST = "{self.config.groot_server_host}"
GROOT_PORT = {self.config.groot_server_port}
ENV_NAME = "{env.value}"
TASK = "{task}"
MAX_STEPS = {max_steps}
ACTION_HORIZON = {action_horizon}
EXECUTE_STEPS = {execute_steps}

# MsgSerializer for ZeroMQ communication
class MsgSerializer:
    @staticmethod
    def encode_ndarray(arr):
        output = io.BytesIO()
        np.save(output, arr, allow_pickle=False)
        return {{"__ndarray_class__": True, "as_npy": output.getvalue()}}

    @staticmethod
    def decode_ndarray(obj):
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)

    @staticmethod
    def encode_custom(obj):
        if isinstance(obj, np.ndarray):
            return MsgSerializer.encode_ndarray(obj)
        return obj

    @staticmethod
    def decode_custom(obj):
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            return MsgSerializer.decode_ndarray(obj)
        return obj

    @staticmethod
    def to_bytes(data):
        return msgpack.packb(data, default=MsgSerializer.encode_custom)

    @staticmethod
    def from_bytes(data):
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom)

# Global ZMQ client
_zmq_context = zmq.Context()
_zmq_socket = None

def get_action(observation, image=None):
    """Get action trajectory from GROOT server via ZeroMQ."""
    global _zmq_socket

    if _zmq_socket is None:
        _zmq_socket = _zmq_context.socket(zmq.REQ)
        _zmq_socket.setsockopt(zmq.RCVTIMEO, 60000)
        _zmq_socket.setsockopt(zmq.SNDTIMEO, 30000)
        _zmq_socket.setsockopt(zmq.LINGER, 0)
        _zmq_socket.connect(f"tcp://{{GROOT_HOST}}:{{GROOT_PORT}}")

    # Build observation dict for new_embodiment config
    # Expects: video['cam_left_high'], state['observation.state'], language['task']
    obs_dict = {{
        "state": {{}},
        "language": {{"task": [[TASK]]}},
    }}

    # 53 DOF observation as single state vector
    if observation.ndim == 1:
        observation = observation.reshape(1, 1, -1)
    obs_dict["state"]["observation.state"] = observation.astype(np.float32)

    # Image is required for GROOT VLA model
    if image is not None:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if image.ndim == 3:
            image = image.reshape(1, 1, *image.shape)
        obs_dict["video"] = {{"cam_left_high": image}}
    else:
        # Create dummy image if not provided (model requires video input)
        dummy_image = np.zeros((1, 1, 256, 256, 3), dtype=np.uint8)
        obs_dict["video"] = {{"cam_left_high": dummy_image}}

    request = {{
        "endpoint": "get_action",
        "data": {{"observation": obs_dict, "options": None}},
    }}

    _zmq_socket.send(MsgSerializer.to_bytes(request))
    response = MsgSerializer.from_bytes(_zmq_socket.recv())

    if isinstance(response, (list, tuple)):
        response = response[0]

    # Extract action from new_embodiment response format
    # Response: {{'action': array of shape (batch, horizon, dof)}}
    if isinstance(response, dict) and "action" in response:
        full_action = response["action"]
        if isinstance(full_action, np.ndarray):
            if full_action.ndim == 3:
                full_action = full_action[0]  # Remove batch dim
            return full_action[:EXECUTE_STEPS] if full_action.ndim > 1 else full_action

    # Fallback for other response formats
    return response

def run_episode():
    """Run episode and return results."""
    # Import Isaac Sim (must be done after env setup)
    try:
        from omni.isaac.lab.app import AppLauncher
        app_launcher = AppLauncher(headless={not render})
        simulation_app = app_launcher.app

        import omni.isaac.lab_tasks
        import gymnasium as gym

        # Create environment
        env = gym.make(ENV_NAME, render_mode="rgb_array" if {render} else None)
        obs, info = env.reset()

        total_reward = 0.0
        success = False
        step = 0

        while step < MAX_STEPS:
            # Get observation state (assuming it's in obs dict)
            if isinstance(obs, dict):
                state = obs.get("policy", obs.get("observation", np.zeros(53)))
            else:
                state = obs

            # Get image if available
            image = None
            if isinstance(obs, dict) and "image" in obs:
                image = obs["image"]

            # Get action trajectory from GROOT
            actions = get_action(state, image)

            # Handle single action vs action sequence
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)

            # Execute each action in the returned sequence
            for action in actions:
                if step >= MAX_STEPS:
                    break

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1

                if terminated:
                    success = info.get("success", reward > 0)
                    break

            if terminated:
                break

        env.close()
        simulation_app.close()

        # Output results as JSON
        result = {{
            "success": bool(success),
            "total_reward": float(total_reward),
            "steps": step,
            "action_horizon": ACTION_HORIZON,
            "execute_steps": EXECUTE_STEPS,
        }}
        print("RESULT:" + json.dumps(result))

    except ImportError as e:
        print(f"RESULT:{{\\"success\\": false, \\"error\\": \\"{{e}}\\", \\"steps\\": 0, \\"total_reward\\": 0}}")

if __name__ == "__main__":
    run_episode()
'''

    def _parse_episode_result(self, stdout: str) -> EpisodeResult:
        """Parse episode result from script output.

        Args:
            stdout: Script output.

        Returns:
            EpisodeResult parsed from output.
        """
        import json

        for line in stdout.split("\n"):
            if line.startswith("RESULT:"):
                data = json.loads(line[7:])
                return EpisodeResult(
                    success=data.get("success", False),
                    total_reward=data.get("total_reward", 0.0),
                    steps=data.get("steps", 0),
                    metrics=data,
                )

        return EpisodeResult(success=False, total_reward=0.0, steps=0)

    def run_benchmark(
        self,
        env: IsaacEnv,
        num_episodes: int = 10,
        task: str = "Complete the manipulation task",
    ) -> Dict[str, Any]:
        """Run multiple episodes and compute statistics.

        Args:
            env: Environment to benchmark.
            num_episodes: Number of episodes to run.
            task: Task description.

        Returns:
            Dictionary with benchmark results.
        """
        self.setup(env)

        results = []
        for i in range(num_episodes):
            print(f"Episode {i+1}/{num_episodes}...")
            result = self.run_episode(task=task)
            results.append(result)
            print(f"  Success: {result.success}, Reward: {result.total_reward:.2f}")

        # Compute statistics
        successes = [r.success for r in results]
        rewards = [r.total_reward for r in results]
        steps = [r.steps for r in results]

        return {
            "environment": env.value,
            "num_episodes": num_episodes,
            "success_rate": sum(successes) / len(successes),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_steps": np.mean(steps),
            "results": [
                {"success": r.success, "reward": r.total_reward, "steps": r.steps}
                for r in results
            ],
        }

    def list_available_envs(self) -> List[str]:
        """List available Isaac Sim environments.

        Returns:
            List of environment names.
        """
        return [env.value for env in IsaacEnv]

    def close(self):
        """Clean up resources."""
        if self.groot_client:
            self.groot_client.close()
