"""GROOT Inference Client for communicating with the GROOT server.

Supports both synchronous and asynchronous communication with the
GROOT inference server running on the Spark server (192.168.1.237).

Current Model: Dex3 28-DOF with 4 cameras
  - Cameras: cam_left_high, cam_right_high, cam_left_wrist, cam_right_wrist
  - State: 28 DOF (left arm 7 + right arm 7 + left Dex3 7 + right Dex3 7)
  - Action: 28 DOF, 16-step prediction horizon

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

Protocol:
---------
The GROOT server uses ZeroMQ REQ/REP pattern with msgpack serialization.
The message format is:
- Request: {'endpoint': 'get_action', 'data': {'observation': ..., 'options': ...}}
- Response: {'action': array of shape (batch, horizon, dof)}
"""

import io
from typing import Optional, Union

import msgpack
import numpy as np
import zmq

from dm_isaac_g1.core.config import Config


class MsgSerializer:
    """Serializer compatible with GR00T server's msgpack protocol."""

    @staticmethod
    def encode_ndarray(arr: np.ndarray) -> dict:
        """Encode numpy array to bytes using npy format."""
        output = io.BytesIO()
        np.save(output, arr, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": output.getvalue()}

    @staticmethod
    def decode_ndarray(obj: dict) -> np.ndarray:
        """Decode numpy array from bytes."""
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)

    @staticmethod
    def encode_custom(obj):
        """Custom encoder for msgpack."""
        if isinstance(obj, np.ndarray):
            return MsgSerializer.encode_ndarray(obj)
        return obj

    @staticmethod
    def decode_custom(obj):
        """Custom decoder for msgpack."""
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            return MsgSerializer.decode_ndarray(obj)
        return obj

    @staticmethod
    def to_bytes(data) -> bytes:
        """Serialize data to msgpack bytes."""
        return msgpack.packb(data, default=MsgSerializer.encode_custom)

    @staticmethod
    def from_bytes(data: bytes):
        """Deserialize data from msgpack bytes."""
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom)


class GrootClient:
    """Synchronous ZeroMQ client for GROOT inference.

    Connects to the GROOT server using ZeroMQ REQ/REP pattern
    to get action predictions from the fine-tuned model.

    Example:
        ```python
        client = GrootClient(host="192.168.1.237", port=5555)
        if client.health_check():
            images = {
                "cam_left_high": left_high_img,
                "cam_right_high": right_high_img,
                "cam_left_wrist": left_wrist_img,
                "cam_right_wrist": right_wrist_img,
            }
            action = client.get_action(
                observation=robot_state,  # 28 DOF for Dex3
                images=images,
                task="Stack the blocks"
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
        self.timeout_ms = int(timeout * 1000)
        self._context = zmq.Context()
        self._socket: Optional[zmq.Socket] = None

    def _connect(self):
        """Establish ZeroMQ connection."""
        if self._socket is None:
            self._socket = self._context.socket(zmq.REQ)
            self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.connect(f"tcp://{self.host}:{self.port}")

    def health_check(self) -> bool:
        """Check if the GROOT server is healthy.

        Returns:
            True if server is responsive, False otherwise.
        """
        try:
            # Use a separate socket with short timeout for health check
            test_socket = self._context.socket(zmq.REQ)
            test_socket.setsockopt(zmq.RCVTIMEO, 3000)  # 3 second timeout
            test_socket.setsockopt(zmq.SNDTIMEO, 3000)
            test_socket.setsockopt(zmq.LINGER, 0)
            test_socket.connect(f"tcp://{self.host}:{self.port}")

            # Send ping request
            request = {"endpoint": "ping"}
            test_socket.send(MsgSerializer.to_bytes(request))
            test_socket.recv()  # Wait for any response
            test_socket.close()
            return True
        except zmq.ZMQError:
            return False

    def get_action(
        self,
        observation: np.ndarray,
        images: Optional[dict[str, np.ndarray]] = None,
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
            observation: Robot state observation (28 DOF for G1+Dex3).
            images: Dict mapping camera names to RGB images (H, W, 3).
                    Expected keys: cam_left_high, cam_right_high,
                    cam_left_wrist, cam_right_wrist.
            image: Deprecated. Single RGB image sent as cam_left_high only.
                   Use `images` dict instead for multi-camera models.
            task: Optional task description for language-conditioned control.
            action_horizon: Number of steps to predict (8-16 recommended).
                           Single-step (1) causes jittering - avoid for production.
            execute_steps: Number of steps to return for execution (1 to action_horizon).
                          For receding horizon control, use 1. For open-loop, use full horizon.
            return_full_trajectory: If True, return full predicted trajectory.

        Returns:
            Action array (28 DOF) or dict with trajectory if requested.
            If execute_steps > 1, returns array of shape (execute_steps, 28).

        Raises:
            zmq.ZMQError: If request fails.
            ValueError: If execute_steps > action_horizon.
        """
        if execute_steps > action_horizon:
            raise ValueError(
                f"execute_steps ({execute_steps}) cannot exceed action_horizon ({action_horizon})"
            )

        self._connect()

        # Build observation dict for GROOT server
        # The new_embodiment config expects:
        # - video: {camera_name: image_array} for each camera
        # - state: {'observation.state': state_vector}
        # - language: {'task': [[task_string]]}
        obs_dict = {
            "state": {},
            "language": {"task": [[task]]} if task else {"task": [["Complete the task"]]},
        }

        # Prepare observation as single state vector
        # Dex3 28-DOF: left arm(7) + right arm(7) + left Dex3(7) + right Dex3(7)
        if observation.ndim == 1:
            observation = observation.reshape(1, 1, -1)
        elif observation.ndim == 2:
            observation = observation.reshape(1, observation.shape[0], observation.shape[1])

        obs_dict["state"]["observation.state"] = observation.astype(np.float32)

        # Add camera images (required for GROOT VLA model)
        # Prefer `images` dict; fall back to legacy single `image` param
        if images is not None:
            video_dict = {}
            for cam_name, img in images.items():
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                if img.ndim == 3:
                    img = img.reshape(1, 1, *img.shape)
                video_dict[cam_name] = img
            obs_dict["video"] = video_dict
        elif image is not None:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            if image.ndim == 3:
                image = image.reshape(1, 1, *image.shape)
            obs_dict["video"] = {"cam_left_high": image}

        # Build request
        request = {
            "endpoint": "get_action",
            "data": {
                "observation": obs_dict,
                "options": None,
            },
        }

        # Send request and get response
        self._socket.send(MsgSerializer.to_bytes(request))
        response = MsgSerializer.from_bytes(self._socket.recv())

        # Handle error response
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GROOT server error: {response['error']}")

        # Extract actions from response
        # new_embodiment response format: {'action': array of shape (batch, horizon, dof)}
        if isinstance(response, (list, tuple)):
            response = response[0]

        # Extract the action tensor
        if isinstance(response, dict) and "action" in response:
            full_action = response["action"]
            if isinstance(full_action, np.ndarray):
                # Remove batch dimension: (1, horizon, dof) -> (horizon, dof)
                if full_action.ndim == 3:
                    full_action = full_action[0]

                # Return requested number of steps
                if execute_steps == 1:
                    action = full_action[0] if full_action.ndim > 1 else full_action
                else:
                    action = full_action[:execute_steps]

                if return_full_trajectory:
                    return {
                        "action": action,
                        "trajectory": full_action,
                        "action_horizon": full_action.shape[0] if full_action.ndim > 1 else 1,
                    }

                return action

        # Fallback for other response formats (e.g., gr1 embodiment with body parts)
        action_parts = []
        for part in ["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]:
            if part in response:
                arr = response[part]
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 3:
                        arr = arr[0]  # Remove batch dim
                    action_parts.append(arr)

        if action_parts:
            full_action = np.concatenate(action_parts, axis=-1)
            if execute_steps == 1:
                return full_action[0] if full_action.ndim > 1 else full_action
            return full_action[:execute_steps]

        # Return raw response if nothing else works
        return response

    def get_policy_info(self) -> dict:
        """Get information about the loaded policy.

        Returns:
            Dictionary with model info (name, embodiment, DOF, etc.).
        """
        self._connect()
        request = {"endpoint": "get_policy_info"}
        self._socket.send(MsgSerializer.to_bytes(request))
        return MsgSerializer.from_bytes(self._socket.recv())

    def close(self):
        """Close the ZeroMQ client."""
        if self._socket:
            self._socket.close()
            self._socket = None
        self._context.term()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class GrootClientAsync:
    """Asynchronous client for GROOT inference.

    Use this for high-frequency control loops or when integrating
    with async frameworks. Uses ZeroMQ with asyncio integration.

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
        self.timeout_ms = int(timeout * 1000)
        # Use sync client internally since zmq async requires zmq.asyncio
        self._sync_client = GrootClient(
            host=self.host,
            port=self.port,
            timeout=timeout,
            config=config,
        )

    async def health_check(self) -> bool:
        """Check if the GROOT server is healthy."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_client.health_check)

    async def get_action(
        self,
        observation: np.ndarray,
        images: Optional[dict[str, np.ndarray]] = None,
        image: Optional[np.ndarray] = None,
        task: Optional[str] = None,
        action_horizon: int = 16,
        execute_steps: int = 1,
    ) -> np.ndarray:
        """Get action from the GROOT model asynchronously.

        Args:
            observation: Robot state observation (28 DOF for Dex3).
            images: Dict mapping camera names to RGB images.
            image: Deprecated. Single RGB image. Use `images` instead.
            task: Optional task description.
            action_horizon: Number of steps to predict (8-16 recommended).
            execute_steps: Number of steps to return for execution.

        Returns:
            Action array. Shape (28,) if execute_steps=1, else (execute_steps, 28).
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_client.get_action(
                observation=observation,
                images=images,
                image=image,
                task=task,
                action_horizon=action_horizon,
                execute_steps=execute_steps,
            ),
        )

    async def close(self):
        """Close the async client."""
        self._sync_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
