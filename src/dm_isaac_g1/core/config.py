"""Configuration management for dm-isaac-g1."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration for dm-isaac-g1.

    Loads settings from environment variables and .env file.
    """

    # Workstation settings
    workstation_host: str = "192.168.1.205"
    workstation_user: str = "datamentors"
    workstation_password: str = ""
    workstation_ssh_port: int = 22

    # GROOT inference server
    groot_server_host: str = "192.168.1.237"
    groot_server_port: int = 5555
    spark_user: str = "nvidia"
    spark_password: str = ""

    # HuggingFace
    hf_token: str = ""

    # Model configuration
    groot_model_path: str = "nvidia/GR00T-N1.6-3B"
    finetuned_model_path: str = "datamentorshf/groot-g1-inspire-9datasets"

    # Training configuration
    num_envs: int = 4096
    seed: int = 42

    # Paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    checkpoints_dir: Path = field(default_factory=lambda: Path.cwd() / "checkpoints")
    datasets_dir: Path = field(default_factory=lambda: Path.cwd() / "datasets")

    # Docker
    container_name: str = "isaac-sim"
    workspace_path: str = "/workspace"

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "Config":
        """Load configuration from environment variables.

        Args:
            env_file: Path to .env file. If None, searches in current directory.

        Returns:
            Config instance with values from environment.
        """
        if env_file is None:
            # Search for .env in current directory and parents
            current = Path.cwd()
            for parent in [current] + list(current.parents):
                candidate = parent / ".env"
                if candidate.exists():
                    env_file = candidate
                    break

        if env_file and env_file.exists():
            load_dotenv(env_file)

        return cls(
            workstation_host=os.getenv("WORKSTATION_HOST", "192.168.1.205"),
            workstation_user=os.getenv("WORKSTATION_USER", "datamentors"),
            workstation_password=os.getenv("WORKSTATION_PASSWORD", ""),
            workstation_ssh_port=int(os.getenv("WORKSTATION_SSH_PORT", "22")),
            groot_server_host=os.getenv("GROOT_SERVER_HOST", "192.168.1.237"),
            groot_server_port=int(os.getenv("GROOT_SERVER_PORT", "5555")),
            spark_user=os.getenv("SPARK_USER", "nvidia"),
            spark_password=os.getenv("SPARK_PASSWORD", ""),
            hf_token=os.getenv("HF_TOKEN", ""),
            groot_model_path=os.getenv("GROOT_MODEL_PATH", "nvidia/GR00T-N1.6-3B"),
            num_envs=int(os.getenv("NUM_ENVS", "4096")),
            seed=int(os.getenv("SEED", "42")),
            project_root=Path(env_file).parent if env_file else Path.cwd(),
        )


def load_config(env_file: Optional[Path] = None) -> Config:
    """Load configuration from environment.

    Args:
        env_file: Optional path to .env file.

    Returns:
        Config instance.
    """
    return Config.from_env(env_file)
