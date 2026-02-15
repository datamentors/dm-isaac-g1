"""Remote execution utilities for workstation and container access."""

import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

from dm_isaac_g1.core.config import Config


@dataclass
class WorkstationConnection:
    """SSH connection to the Blackwell workstation.

    Provides methods for executing commands on the workstation
    and inside the isaac-sim Docker container.
    """

    config: Config

    def execute(
        self,
        command: str,
        timeout: int = 120,
        check: bool = True,
    ) -> Tuple[str, str, int]:
        """Execute a command on the workstation via SSH.

        Args:
            command: Command to execute.
            timeout: Timeout in seconds.
            check: If True, raise exception on non-zero exit code.

        Returns:
            Tuple of (stdout, stderr, return_code).
        """
        ssh_cmd = [
            "sshpass",
            "-p",
            self.config.workstation_password,
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            f"ConnectTimeout={min(timeout, 30)}",
            f"{self.config.workstation_user}@{self.config.workstation_host}",
            command,
        ]

        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command failed with code {result.returncode}: {result.stderr}"
            )

        return result.stdout, result.stderr, result.returncode

    def docker_exec(
        self,
        command: str,
        container: Optional[str] = None,
        activate_env: bool = True,
        timeout: int = 120,
        check: bool = True,
    ) -> Tuple[str, str, int]:
        """Execute a command inside the Docker container.

        Args:
            command: Command to execute.
            container: Container name (default: config.container_name).
            activate_env: If True, activate grootenv before command.
            timeout: Timeout in seconds.
            check: If True, raise exception on non-zero exit code.

        Returns:
            Tuple of (stdout, stderr, return_code).
        """
        container = container or self.config.container_name

        if activate_env:
            full_cmd = (
                f"source /opt/conda/etc/profile.d/conda.sh && "
                f"conda activate grootenv && {command}"
            )
        else:
            full_cmd = command

        docker_cmd = f"docker exec {container} bash -c '{full_cmd}'"

        return self.execute(docker_cmd, timeout=timeout, check=check)

    def copy_to_container(
        self,
        local_path: str,
        container_path: str,
        container: Optional[str] = None,
    ) -> None:
        """Copy a file to the container.

        Args:
            local_path: Path on workstation.
            container_path: Destination path in container.
            container: Container name.
        """
        container = container or self.config.container_name
        cmd = f"docker cp {local_path} {container}:{container_path}"
        self.execute(cmd)

    def copy_from_container(
        self,
        container_path: str,
        local_path: str,
        container: Optional[str] = None,
    ) -> None:
        """Copy a file from the container.

        Args:
            container_path: Path in container.
            local_path: Destination path on workstation.
            container: Container name.
        """
        container = container or self.config.container_name
        cmd = f"docker cp {container}:{container_path} {local_path}"
        self.execute(cmd)

    def sync_repo(self, repo_path: str = "/home/datamentors/dm-isaac-g1") -> str:
        """Pull latest code from git repository.

        Args:
            repo_path: Path to repository on workstation.

        Returns:
            Git pull output.
        """
        stdout, _, _ = self.execute(f"cd {repo_path} && git pull origin main")
        return stdout

    def get_training_progress(self, log_file: str) -> Optional[dict]:
        """Get current training progress from log file.

        Args:
            log_file: Path to training log file.

        Returns:
            Dictionary with loss, step count, etc. or None if not available.
        """
        try:
            stdout, _, _ = self.docker_exec(
                f"grep -E '^\\{{' {log_file} | tail -1",
                activate_env=False,
                check=False,
            )
            if stdout.strip():
                import json

                return json.loads(stdout.strip())
        except Exception:
            pass
        return None

    def check_gpu_status(self) -> str:
        """Check GPU status on workstation.

        Returns:
            nvidia-smi output.
        """
        stdout, _, _ = self.execute("nvidia-smi")
        return stdout
