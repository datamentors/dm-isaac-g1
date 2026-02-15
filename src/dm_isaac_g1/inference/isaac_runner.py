"""Isaac Sim environment runner for testing GROOT inference.

Provides integration with Unitree's Isaac Sim environments for
testing the fine-tuned GROOT model in simulation.

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
        render: bool = False,
        record: bool = False,
    ) -> EpisodeResult:
        """Run a single episode using GROOT for control.

        This executes the simulation on the workstation and uses
        the GROOT server for action inference.

        Args:
            task: Task description for language conditioning.
            max_steps: Maximum steps per episode.
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
        render: bool,
        record: bool,
    ) -> str:
        """Generate Python script to run episode on workstation.

        Args:
            env: Environment to use.
            task: Task description.
            max_steps: Maximum steps.
            render: Whether to render.
            record: Whether to record.

        Returns:
            Python script as string.
        """
        return f'''#!/usr/bin/env python3
"""Auto-generated episode runner for GROOT inference testing."""
import json
import numpy as np
import requests

# Configuration
GROOT_HOST = "{self.config.groot_server_host}"
GROOT_PORT = {self.config.groot_server_port}
ENV_NAME = "{env.value}"
TASK = "{task}"
MAX_STEPS = {max_steps}

def get_action(observation, image=None):
    """Get action from GROOT server."""
    payload = {{"observation": observation.tolist()}}
    if image is not None:
        import base64
        payload["image"] = base64.b64encode(image.tobytes()).decode()
        payload["image_shape"] = list(image.shape)
    payload["task"] = TASK

    response = requests.post(
        f"http://{{GROOT_HOST}}:{{GROOT_PORT}}/inference",
        json=payload,
        timeout=5.0,
    )
    response.raise_for_status()
    return np.array(response.json()["action"])

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

        for step in range(MAX_STEPS):
            # Get observation state (assuming it's in obs dict)
            if isinstance(obs, dict):
                state = obs.get("policy", obs.get("observation", np.zeros(53)))
            else:
                state = obs

            # Get image if available
            image = None
            if isinstance(obs, dict) and "image" in obs:
                image = obs["image"]

            # Get action from GROOT
            action = get_action(state, image)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated:
                success = info.get("success", reward > 0)
                break

        env.close()
        simulation_app.close()

        # Output results as JSON
        result = {{
            "success": bool(success),
            "total_reward": float(total_reward),
            "steps": step + 1,
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
