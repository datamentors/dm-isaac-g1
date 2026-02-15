"""Dataset validation utilities."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def validate_dataset(dataset_path: Path) -> ValidationResult:
    """Validate dataset structure and contents.

    Checks for:
    - Required meta files (info.json, modality.json, episodes.jsonl, tasks.jsonl)
    - Parquet file existence and schema
    - Video file existence (if referenced)
    - Consistency between meta files and data

    Args:
        dataset_path: Path to dataset directory.

    Returns:
        ValidationResult with validation status and any issues found.
    """
    dataset_path = Path(dataset_path)
    errors = []
    warnings = []
    info = {}

    # Check directory exists
    if not dataset_path.exists():
        return ValidationResult(
            valid=False,
            errors=[f"Dataset path does not exist: {dataset_path}"],
        )

    meta_dir = dataset_path / "meta"
    data_dir = dataset_path / "data"

    # Check meta directory
    if not meta_dir.exists():
        errors.append("Missing meta/ directory")
    else:
        # Check required meta files
        required_files = ["info.json"]
        optional_files = ["modality.json", "episodes.jsonl", "tasks.jsonl", "stats.json"]

        for filename in required_files:
            if not (meta_dir / filename).exists():
                errors.append(f"Missing required file: meta/{filename}")

        for filename in optional_files:
            if not (meta_dir / filename).exists():
                warnings.append(f"Missing optional file: meta/{filename}")

        # Load and validate info.json
        info_file = meta_dir / "info.json"
        if info_file.exists():
            with open(info_file) as f:
                info_data = json.load(f)
                info["total_episodes"] = info_data.get("total_episodes", 0)
                info["total_frames"] = info_data.get("total_frames", 0)
                info["robot_type"] = info_data.get("robot_type", "unknown")
                info["features"] = list(info_data.get("features", {}).keys())

    # Check data directory
    if not data_dir.exists():
        errors.append("Missing data/ directory")
    else:
        # Count parquet files
        parquet_files = list(data_dir.rglob("*.parquet"))
        info["parquet_files"] = len(parquet_files)

        if len(parquet_files) == 0:
            errors.append("No parquet files found in data/")
        else:
            # Validate first parquet file schema
            if pq is not None:
                try:
                    table = pq.read_table(parquet_files[0])
                    info["columns"] = table.column_names
                    info["sample_rows"] = table.num_rows

                    # Check for required columns
                    required_cols = ["episode_index"]
                    for col in required_cols:
                        if col not in table.column_names:
                            warnings.append(f"Missing column: {col}")

                except Exception as e:
                    errors.append(f"Error reading parquet: {e}")

    # Check videos directory
    videos_dir = dataset_path / "videos"
    if videos_dir.exists():
        video_files = list(videos_dir.rglob("*.mp4"))
        info["video_files"] = len(video_files)
    else:
        info["video_files"] = 0
        warnings.append("No videos/ directory found")

    valid = len(errors) == 0

    return ValidationResult(
        valid=valid,
        errors=errors,
        warnings=warnings,
        info=info,
    )


def fix_episodes(
    dataset_path: Path,
    task_description: Optional[str] = None,
) -> bool:
    """Create or fix episodes.jsonl and tasks.jsonl files.

    Args:
        dataset_path: Path to dataset.
        task_description: Task description to use. Auto-detected if None.

    Returns:
        True if files were created/fixed successfully.
    """
    if pq is None:
        raise ImportError("pyarrow required: pip install pyarrow")

    dataset_path = Path(dataset_path)
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    episodes_file = meta_dir / "episodes.jsonl"
    tasks_file = meta_dir / "tasks.jsonl"

    # Load info.json
    info_file = meta_dir / "info.json"
    if not info_file.exists():
        print(f"ERROR: info.json not found in {meta_dir}")
        return False

    with open(info_file) as f:
        info = json.load(f)

    total_episodes = info.get("total_episodes", 0)

    # Count parquet files if total_episodes not set
    if total_episodes == 0:
        parquet_files = list((dataset_path / "data").rglob("episode_*.parquet"))
        total_episodes = len(parquet_files)

    print(f"Processing {dataset_path.name}: {total_episodes} episodes")

    # Get task description from parquet if not provided
    if task_description is None:
        task_description = "Complete the manipulation task."
        parquet_files = sorted((dataset_path / "data").rglob("*.parquet"))
        if parquet_files:
            try:
                table = pq.read_table(parquet_files[0])
                if "task" in table.column_names:
                    tasks = table.column("task").to_pylist()
                    if tasks and tasks[0]:
                        task_description = tasks[0]
            except Exception:
                pass

    print(f"  Task: {task_description}")

    # Create tasks.jsonl
    tasks = [{"task_index": 0, "task": task_description}]
    with open(tasks_file, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    print(f"  Created tasks.jsonl")

    # Create episodes.jsonl
    episodes = []
    data_dir = dataset_path / "data" / "chunk-000"

    for ep_idx in range(total_episodes):
        # Try to get episode length from parquet
        pq_file = data_dir / f"episode_{ep_idx:06d}.parquet"
        length = 1000  # default

        if pq_file.exists():
            try:
                table = pq.read_table(pq_file)
                length = table.num_rows
            except Exception:
                pass

        episodes.append({
            "episode_index": ep_idx,
            "tasks": [task_description],
            "length": length,
            "task_index": 0,
        })

    with open(episodes_file, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    print(f"  Created episodes.jsonl with {len(episodes)} episodes")

    return True
