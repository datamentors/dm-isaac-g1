"""
GROOT Inference Client
Connects to the GROOT server running in dm-groot-inference
"""

import os
from typing import Optional

import httpx
import numpy as np


class GrootClient:
    """Client for communicating with the GROOT inference server."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 30.0,
    ):
        """Initialize the GROOT client.

        Args:
            host: GROOT server host (defaults to env GROOT_SERVER_HOST)
            port: GROOT server port (defaults to env GROOT_SERVER_PORT)
            timeout: Request timeout in seconds
        """
        self.host = host or os.getenv("GROOT_SERVER_HOST", "localhost")
        self.port = port or int(os.getenv("GROOT_SERVER_PORT", "5555"))
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def health_check(self) -> bool:
        """Check if the GROOT server is healthy."""
        try:
            response = self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except httpx.RequestError:
            return False

    def get_action(
        self,
        observation: np.ndarray,
        image: Optional[np.ndarray] = None,
        task_description: Optional[str] = None,
    ) -> np.ndarray:
        """Get action from the GROOT model.

        Args:
            observation: Robot state observation (joint positions, velocities, etc.)
            image: Optional RGB image from robot camera
            task_description: Optional task description for language-conditioned control

        Returns:
            Action array (joint position targets)
        """
        payload = {
            "observation": observation.tolist(),
        }

        if image is not None:
            # Encode image as base64 or send as separate endpoint
            payload["image"] = image.tolist()

        if task_description:
            payload["task"] = task_description

        response = self._client.post(
            f"{self.base_url}/inference",
            json=payload,
        )
        response.raise_for_status()

        result = response.json()
        return np.array(result["action"])

    def get_policy_info(self) -> dict:
        """Get information about the loaded policy."""
        response = self._client.get(f"{self.base_url}/policy/info")
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
    """Async client for communicating with the GROOT inference server."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 30.0,
    ):
        self.host = host or os.getenv("GROOT_SERVER_HOST", "localhost")
        self.port = port or int(os.getenv("GROOT_SERVER_PORT", "5555"))
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
        task_description: Optional[str] = None,
    ) -> np.ndarray:
        """Get action from the GROOT model asynchronously."""
        payload = {
            "observation": observation.tolist(),
        }

        if image is not None:
            payload["image"] = image.tolist()

        if task_description:
            payload["task"] = task_description

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
