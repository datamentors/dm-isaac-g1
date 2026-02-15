"""GROOT Inference Client for communicating with the GROOT server.

Supports both synchronous and asynchronous communication with the
GROOT inference server running on the Spark server (192.168.1.237).
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
        return_full_trajectory: bool = False,
    ) -> Union[np.ndarray, dict]:
        """Get action from the GROOT model.

        Args:
            observation: Robot state observation (53 DOF for G1+Inspire).
            image: Optional RGB image from robot camera (H, W, 3).
            task: Optional task description for language-conditioned control.
            return_full_trajectory: If True, return full action trajectory.

        Returns:
            Action array (53 DOF) or dict with trajectory if requested.

        Raises:
            httpx.HTTPError: If request fails.
        """
        payload = {
            "observation": observation.tolist(),
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
    ) -> np.ndarray:
        """Get action from the GROOT model asynchronously.

        Args:
            observation: Robot state observation.
            image: Optional RGB image.
            task: Optional task description.

        Returns:
            Action array.
        """
        payload = {
            "observation": observation.tolist(),
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
