"""Command-line interface for dm-isaac-g1."""

from pathlib import Path
from typing import Optional

import click


@click.group()
@click.version_option()
def main():
    """DM-ISAAC-G1: G1 Robot Training Suite.

    Fine-tuning, Inference, Imitation Learning, and Reinforcement Learning
    for the Unitree G1 EDU 2 robot with UNITREE_G1 Gripper Hands.
    """
    pass


# =============================================================================
# Data Commands
# =============================================================================


@main.group()
def data():
    """Dataset management commands."""
    pass


@data.command()
@click.argument("repo_id")
@click.option("--output", "-o", type=click.Path(), default="./datasets", help="Output directory")
@click.option("--lfs/--no-lfs", default=True, help="Use git LFS for download")
def download(repo_id: str, output: str, lfs: bool):
    """Download a dataset from HuggingFace.

    Example: dm-g1 data download unitreerobotics/G1_Fold_Towel
    """
    from dm_isaac_g1.data.download import download_dataset

    output_path = Path(output)
    click.echo(f"Downloading {repo_id} to {output_path}...")

    try:
        path = download_dataset(repo_id, output_path, use_lfs=lfs)
        click.echo(f"Downloaded to: {path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@data.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--hand-type", "-t", type=click.Choice(["gripper", "dex3", "trifinger"]))
@click.option("--dry-run", is_flag=True, help="Analyze only, don't convert")
def convert(input_path: str, output_path: str, hand_type: Optional[str], dry_run: bool):
    """Convert dataset to 53 DOF Inspire format.

    Example: dm-g1 data convert ./G1_Fold_Towel ./G1_Fold_Towel_Inspire
    """
    from dm_isaac_g1.data.convert import convert_to_inspire

    success = convert_to_inspire(
        Path(input_path),
        Path(output_path),
        hand_type=hand_type,
        dry_run=dry_run,
    )

    if not success:
        raise SystemExit(1)


@data.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def validate(dataset_path: str):
    """Validate dataset structure and contents.

    Example: dm-g1 data validate ./G1_Fold_Towel_Inspire
    """
    from dm_isaac_g1.data.validate import validate_dataset

    result = validate_dataset(Path(dataset_path))

    click.echo(f"\nValidation Result: {'VALID' if result.valid else 'INVALID'}")

    if result.errors:
        click.echo("\nErrors:")
        for err in result.errors:
            click.echo(f"  - {err}")

    if result.warnings:
        click.echo("\nWarnings:")
        for warn in result.warnings:
            click.echo(f"  - {warn}")

    click.echo("\nInfo:")
    for key, value in result.info.items():
        click.echo(f"  {key}: {value}")

    if not result.valid:
        raise SystemExit(1)


@data.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def stats(dataset_path: str):
    """Compute normalization statistics for dataset.

    Example: dm-g1 data stats ./G1_Fold_Towel_Inspire
    """
    from dm_isaac_g1.data.stats import compute_stats

    compute_stats(Path(dataset_path))
    click.echo("Statistics computed and saved to meta/stats.json")


# =============================================================================
# Inference Commands
# =============================================================================


@main.group()
def infer():
    """Inference commands."""
    pass


@infer.command()
@click.option("--host", default=None, help="GROOT server host")
@click.option("--port", default=None, type=int, help="GROOT server port")
def status(host: Optional[str], port: Optional[int]):
    """Check GROOT server status.

    Example: dm-g1 infer status
    """
    from dm_isaac_g1.inference.client import GrootClient

    client = GrootClient(host=host, port=port)

    if client.health_check():
        click.echo(f"GROOT server is running at {client.base_url}")
        try:
            info = client.get_policy_info()
            click.echo(f"Model: {info.get('model_path', 'unknown')}")
        except Exception:
            pass
    else:
        click.echo(f"GROOT server not responding at {client.base_url}")
        raise SystemExit(1)


@infer.command()
@click.option("--model", default="datamentorshf/groot-g1-gripper-hospitality-7ds", help="Model to load")
@click.option("--port", default=5555, type=int, help="Server port")
def serve(model: str, port: int):
    """Start GROOT inference server on Spark.

    Example: dm-g1 infer serve --model datamentorshf/groot-g1-gripper-hospitality-7ds
    """
    from dm_isaac_g1.inference.server import GrootServerManager

    manager = GrootServerManager()

    click.echo(f"Starting GROOT server with model: {model}")
    if manager.start(model_path=model, port=port):
        click.echo(f"Server started on port {port}")
    else:
        click.echo("Failed to start server", err=True)
        raise SystemExit(1)


@infer.command()
def stop():
    """Stop GROOT inference server.

    Example: dm-g1 infer stop
    """
    from dm_isaac_g1.inference.server import GrootServerManager

    manager = GrootServerManager()
    manager.stop()
    click.echo("Server stopped")


@infer.command()
@click.option("--env", default="Isaac-PickPlace-RedBlock-G129-Inspire-Joint", help="Environment")
@click.option("--episodes", default=5, type=int, help="Number of episodes")
@click.option("--task", default="Complete the manipulation task", help="Task description")
def benchmark(env: str, episodes: int, task: str):
    """Run inference benchmark in Isaac Sim.

    Example: dm-g1 infer benchmark --env Isaac-PickPlace-RedBlock-G129-Inspire-Joint
    """
    from dm_isaac_g1.inference.isaac_runner import IsaacSimRunner, IsaacEnv

    runner = IsaacSimRunner()

    # Find matching env
    try:
        isaac_env = IsaacEnv(env)
    except ValueError:
        click.echo(f"Unknown environment: {env}")
        click.echo("Available environments:")
        for e in IsaacEnv:
            click.echo(f"  - {e.value}")
        raise SystemExit(1)

    click.echo(f"Running benchmark: {env}")
    click.echo(f"Episodes: {episodes}")

    results = runner.run_benchmark(isaac_env, num_episodes=episodes, task=task)

    click.echo(f"\nResults:")
    click.echo(f"  Success Rate: {results['success_rate']*100:.1f}%")
    click.echo(f"  Mean Reward: {results['mean_reward']:.2f}")
    click.echo(f"  Mean Steps: {results['mean_steps']:.0f}")


# =============================================================================
# Remote Commands
# =============================================================================


@main.group()
def remote():
    """Remote workstation commands."""
    pass


@remote.command()
def connect():
    """Connect to workstation via SSH.

    Example: dm-g1 remote connect
    """
    import subprocess

    from dm_isaac_g1.core.config import load_config

    config = load_config()

    click.echo(f"Connecting to {config.workstation_user}@{config.workstation_host}...")

    subprocess.run([
        "sshpass",
        "-p",
        config.workstation_password,
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        f"{config.workstation_user}@{config.workstation_host}",
    ])


@remote.command()
@click.argument("command")
def exec(command: str):
    """Execute command on workstation.

    Example: dm-g1 remote exec "nvidia-smi"
    """
    from dm_isaac_g1.core.remote import WorkstationConnection
    from dm_isaac_g1.core.config import load_config

    config = load_config()
    conn = WorkstationConnection(config=config)

    stdout, stderr, code = conn.execute(command, check=False)
    click.echo(stdout)
    if stderr:
        click.echo(stderr, err=True)

    if code != 0:
        raise SystemExit(code)


@remote.command()
def sync():
    """Sync repository to workstation via git pull.

    Example: dm-g1 remote sync
    """
    from dm_isaac_g1.core.remote import WorkstationConnection
    from dm_isaac_g1.core.config import load_config

    config = load_config()
    conn = WorkstationConnection(config=config)

    click.echo("Syncing repository to workstation...")
    output = conn.sync_repo()
    click.echo(output)
    click.echo("Sync complete!")


@remote.command()
def gpu():
    """Check GPU status on workstation.

    Example: dm-g1 remote gpu
    """
    from dm_isaac_g1.core.remote import WorkstationConnection
    from dm_isaac_g1.core.config import load_config

    config = load_config()
    conn = WorkstationConnection(config=config)

    output = conn.check_gpu_status()
    click.echo(output)


if __name__ == "__main__":
    main()
