"""GROOT Server Manager for deploying models on Spark server.

Manages the GROOT inference server running on the Spark server (192.168.1.237),
including starting, stopping, and switching models.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dm_isaac_g1.core.config import Config, load_config
from dm_isaac_g1.core.remote import WorkstationConnection


@dataclass
class ServerStatus:
    """Status of the GROOT inference server."""

    running: bool
    model_path: Optional[str] = None
    port: int = 5555
    pid: Optional[int] = None
    gpu_memory_mb: Optional[int] = None


class GrootServerManager:
    """Manager for GROOT inference server on Spark.

    Controls the GROOT server lifecycle on the Spark server (192.168.1.237).
    The server runs the fine-tuned model and accepts inference requests.

    Example:
        ```python
        manager = GrootServerManager()
        manager.start(model_path="datamentorshf/groot-g1-inspire-9datasets")
        status = manager.status()
        print(f"Server running: {status.running}")
        manager.stop()
        ```
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize server manager.

        Args:
            config: Configuration instance. Loads from .env if not provided.
        """
        self.config = config or load_config()
        self._spark_host = self.config.groot_server_host
        self._spark_user = self.config.spark_user
        self._spark_password = self.config.spark_password

    def _ssh_exec(self, command: str, timeout: int = 60) -> str:
        """Execute command on Spark server via SSH.

        Args:
            command: Command to execute.
            timeout: Timeout in seconds.

        Returns:
            Command output.
        """
        import subprocess

        ssh_cmd = [
            "sshpass",
            "-p",
            self._spark_password,
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            f"{self._spark_user}@{self._spark_host}",
            command,
        ]

        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return result.stdout

    def status(self) -> ServerStatus:
        """Get current server status.

        Returns:
            ServerStatus with running state and model info.
        """
        try:
            # Check if server process is running
            output = self._ssh_exec("pgrep -f 'run_gr00t_server' || echo 'not_running'")

            if "not_running" in output:
                return ServerStatus(running=False)

            pid = int(output.strip().split()[0]) if output.strip() else None

            # Get GPU memory usage
            gpu_output = self._ssh_exec(
                "nvidia-smi --query-compute-apps=pid,used_memory "
                "--format=csv,noheader,nounits 2>/dev/null || echo ''"
            )

            gpu_memory = None
            if gpu_output.strip() and pid:
                for line in gpu_output.strip().split("\n"):
                    parts = line.split(",")
                    if len(parts) >= 2 and str(pid) in parts[0]:
                        gpu_memory = int(parts[1].strip())
                        break

            return ServerStatus(
                running=True,
                pid=pid,
                port=self.config.groot_server_port,
                gpu_memory_mb=gpu_memory,
            )

        except Exception as e:
            print(f"Error checking server status: {e}")
            return ServerStatus(running=False)

    def start(
        self,
        model_path: str = "datamentorshf/groot-g1-inspire-9datasets",
        port: int = 5555,
        embodiment_tag: str = "NEW_EMBODIMENT",
        background: bool = True,
    ) -> bool:
        """Start the GROOT inference server.

        Args:
            model_path: HuggingFace model repo or local path.
            port: Port to run server on.
            embodiment_tag: Embodiment tag for the model.
            background: Run in background with nohup.

        Returns:
            True if server started successfully.
        """
        # Check if already running
        status = self.status()
        if status.running:
            print(f"Server already running (PID: {status.pid})")
            return True

        # Build start command
        cmd = (
            f"cd /workspace/Isaac-GR00T && "
            f"source /opt/conda/etc/profile.d/conda.sh && "
            f"conda activate grootenv && "
            f"python gr00t/eval/run_gr00t_server.py "
            f"--model-path {model_path} "
            f"--embodiment-tag {embodiment_tag} "
            f"--port {port}"
        )

        if background:
            cmd = f"nohup {cmd} > /tmp/groot_server.log 2>&1 &"

        try:
            self._ssh_exec(cmd, timeout=30)
            print(f"Server starting with model: {model_path}")
            return True
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False

    def stop(self) -> bool:
        """Stop the GROOT inference server.

        Returns:
            True if server stopped successfully.
        """
        try:
            self._ssh_exec("pkill -f 'run_gr00t_server' || true")
            print("Server stopped")
            return True
        except Exception as e:
            print(f"Error stopping server: {e}")
            return False

    def restart(
        self,
        model_path: Optional[str] = None,
        port: int = 5555,
    ) -> bool:
        """Restart the server, optionally with a new model.

        Args:
            model_path: New model to load (uses existing if None).
            port: Port to run on.

        Returns:
            True if restart successful.
        """
        self.stop()

        import time
        time.sleep(2)  # Wait for process to fully terminate

        model = model_path or "datamentorshf/groot-g1-inspire-9datasets"
        return self.start(model_path=model, port=port)

    def get_logs(self, lines: int = 50) -> str:
        """Get recent server logs.

        Args:
            lines: Number of lines to retrieve.

        Returns:
            Log output.
        """
        return self._ssh_exec(f"tail -{lines} /tmp/groot_server.log 2>/dev/null || echo 'No logs'")

    def deploy_model(
        self,
        checkpoint_path: str,
        model_name: str = "groot-g1-custom",
    ) -> bool:
        """Deploy a new model checkpoint to the server.

        Copies checkpoint from workstation to Spark server and starts serving.

        Args:
            checkpoint_path: Path to checkpoint on workstation.
            model_name: Name for the deployed model.

        Returns:
            True if deployment successful.
        """
        # Copy from workstation to Spark
        scp_cmd = (
            f"sshpass -p '{self.config.workstation_password}' "
            f"scp -r -o StrictHostKeyChecking=no "
            f"{self.config.workstation_user}@{self.config.workstation_host}:{checkpoint_path} "
            f"/workspace/models/{model_name}"
        )

        try:
            self._ssh_exec(scp_cmd, timeout=600)
            print(f"Model copied to /workspace/models/{model_name}")

            # Start server with new model
            return self.start(model_path=f"/workspace/models/{model_name}")

        except Exception as e:
            print(f"Deployment failed: {e}")
            return False
