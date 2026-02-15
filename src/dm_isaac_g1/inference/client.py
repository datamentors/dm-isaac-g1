"""GROOT Inference Client for communicating with the GROOT server.

Supports both synchronous and asynchronous communication with the
GROOT inference server running on the Spark server (192.168.1.237).

GR00T N1.6 Action Chunking:
---------------------------
GR00T 1.6 uses Flow Matching with a 32-layer Diffusion Transformer (DiT) to
predict action chunks (typically 8-16 steps). Key characteristics:

- Action horizon is configurable at inference time (8-16 steps recommended)
- Single-step inference (action_horizon=1) causes jittering and poor performance
- Actions are state-relative for smoother, more accurate motion
- The flow matching approach reconstructs continuous actions from noise

For closed-loop control, use receding horizon (MPC-style):
1. Get observation from environment
2. Request N-step trajectory from GROOT
3. Execute first M steps (M <= N, typically M=1 or M=N//2)
4. Get new observation
5. Repeat until task complete
"""

import base64
from typing import Optional, Union

import httpx
import numpy as np

from dm_isaac_g1.core.config import Config


class GrootClient:
    """Synchronous client for GROOT inference.

    Connects to the GROOT server to get action predictions from
    the fine-tuned model.

    Example:
        ```python
        client = GrootClient(host="192.168.1.237", port=5555)
        if client.health_check():
            action = client.get_action(
                observation=robot_state,
                image=camera_image,
                task="Fold the towel"
            )
        ```
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 30.0,
        config: Optional[Config] = None,
    ):
        """Initialize the GROOT client.

        Args:
            host: GROOT server host. Defaults to config or 192.168.1.237.
            port: GROOT server port. Defaults to config or 5555.
            timeout: Request timeout in seconds.
            config: Optional Config instance for settings.
        """
        if config is None:
            from dm_isaac_g1.core.config import load_config
            config = load_config()

        self.host = host or config.groot_server_host
        self.port = port or config.groot_server_port
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def health_check(self) -> bool:
        """Check if the GROOT server is healthy.

        Returns:
            True if server is responsive, False otherwise.
        """
        try:
            response = self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except httpx.RequestError:
            return False

    def get_action(
        self,
        observation: np.ndarray,
        image: Optional[np.ndarray] = None,
        task: Optional[str] = None,
        action_horizon: int = 16,
        execute_steps: int = 1,
        return_full_trajectory: bool = False,
    ) -> Union[np.ndarray, dict]:
        """Get action from the GROOT model.

        GR00T 1.6 predicts action chunks (typically 8-16 steps). This method
        allows configuring the action horizon at runtime for optimal performance.

        Args:
            observation: Robot state observation (53 DOF for G1+Inspire).
            image: Optional RGB image from robot camera (H, W, 3).
            task: Optional task description for language-conditioned control.
            action_horizon: Number of steps to predict (8-16 recommended).
                           Single-step (1) causes jittering - avoid for production.
            execute_steps: Number of steps to return for execution (1 to action_horizon).
                          For receding horizon control, use 1. For open-loop, use full horizon.
            return_full_trajectory: If True, return full predicted trajectory.

        Returns:
            Action array (53 DOF) or dict with trajectory if requested.
            If execute_steps > 1, returns array of shape (execute_steps, 53).

        Raises:
            httpx.HTTPError: If request fails.
            ValueError: If execute_steps > action_horizon.
        """
        if execute_steps > action_horizon:
            raise ValueError(
                f"execute_steps ({execute_steps}) cannot exceed action_horizon ({action_horizon})"
            )

        payload = {
            "observation": observation.tolist(),
            "action_horizon": action_horizon,
            "execute_steps": execute_steps,
        }

        if image is not None:
            # Encode image as base64 for transmission
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            payload["image"] = base64.b64encode(image.tobytes()).decode("utf-8")
            payload["image_shape"] = list(image.shape)

        if task:
            payload["task"] = task

        if return_full_trajectory:
            payload["return_trajectory"] = True

        response = self._client.post(
            f"{self.base_url}/inference",
            json=payload,
        )
        response.raise_for_status()

        result = response.json()

        if return_full_trajectory:
            return {
                "action": np.array(result["action"]),
                "trajectory": np.array(result.get("trajectory", [])),
                "action_horizon": result.get("action_horizon", action_horizon),
            }

        return np.array(result["action"])

    def get_policy_info(self) -> dict:
        """Get information about the loaded policy.

        Returns:
            Dictionary with model info (name, embodiment, DOF, etc.).
        """
        response = self._client.get(f"{self.base_url}/policy/info")
        response.raise_for_status()
        return response.json()

    def load_model(self, model_path: str) -> dict:
        """Request server to load a different model.

        Args:
            model_path: Path or HuggingFace repo ID for model.

        Returns:
            Response from server.
        """
        response = self._client.post(
            f"{self.base_url}/model/load",
            json={"model_path": model_path},
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class GrootClientAsync:
    """Asynchronous client for GROOT inference.

    Use this for high-frequency control loops or when integrating
    with async frameworks.

    Example:
        ```python
        async with GrootClientAsync() as client:
            if await client.health_check():
                action = await client.get_action(observation, image)
        ```
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 30.0,
        config: Optional[Config] = None,
    ):
        """Initialize async GROOT client.

        Args:
            host: GROOT server host.
            port: GROOT server port.
            timeout: Request timeout in seconds.
            config: Optional Config instance.
        """
        if config is None:
            from dm_isaac_g1.core.config import load_config
            config = load_config()

        self.host = host or config.groot_server_host
        self.port = port or config.groot_server_port
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def health_check(self) -> bool:
        """Check if the GROOT server is healthy."""
        try:
            response = await self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except httpx.RequestError:
            return False

    async def get_action(
        self,
        observation: np.ndarray,
        image: Optional[np.ndarray] = None,
        task: Optional[str] = None,
        action_horizon: int = 16,
        execute_steps: int = 1,
    ) -> np.ndarray:
        """Get action from the GROOT model asynchronously.

        Args:
            observation: Robot state observation.
            image: Optional RGB image.
            task: Optional task description.
            action_horizon: Number of steps to predict (8-16 recommended).
            execute_steps: Number of steps to return for execution.

        Returns:
            Action array. Shape (53,) if execute_steps=1, else (execute_steps, 53).
        """
        if execute_steps > action_horizon:
            raise ValueError(
                f"execute_steps ({execute_steps}) cannot exceed action_horizon ({action_horizon})"
            )

        payload = {
            "observation": observation.tolist(),
            "action_horizon": action_horizon,
            "execute_steps": execute_steps,
        }

        if image is not None:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            payload["image"] = base64.b64encode(image.tobytes()).decode("utf-8")
            payload["image_shape"] = list(image.shape)

        if task:
            payload["task"] = task

        response = await self._client.post(
            f"{self.base_url}/inference",
            json=payload,
        )
        response.raise_for_status()

        result = response.json()
        return np.array(result["action"])

    async def close(self):
        """Close the async HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
