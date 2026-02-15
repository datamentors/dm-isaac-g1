#!/usr/bin/env python3
"""
Combine multiple Inspire-format datasets into a single unified dataset for GROOT training.

This script takes multiple converted Inspire datasets and combines them into a single
dataset suitable for multi-task training with GROOT N1.6.

Usage:
    python combine_inspire_datasets.py --input-dir /workspace/datasets_inspire \
                                       --output /workspace/datasets/G1_Inspire_Combined

Features:
    - Combines all parquet files with proper episode indexing
    - Merges task descriptions from all sources
    - Handles video files (copies to combined dataset)
    - Generates unified metadata (info.json, modality.json)
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any
import glob

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow required. Install with: pip install pyarrow")
    exit(1)


def find_inspire_datasets(input_dir: Path) -> List[Path]:
    """Find all Inspire-format datasets in the input directory."""
    datasets = []

    for item in input_dir.iterdir():
        if item.is_dir():
            # Check if it has info.json (valid dataset)
            info_file = item / "meta" / "info.json"
            if info_file.exists():
                datasets.append(item)
            else:
                # Check subdirectories
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        sub_info = subitem / "meta" / "info.json"
                        if sub_info.exists():
                            datasets.append(subitem)

    return sorted(datasets)


def load_dataset_info(dataset_path: Path) -> Dict[str, Any]:
    """Load info.json from a dataset."""
    info_file = dataset_path / "meta" / "info.json"
    if info_file.exists():
        with open(info_file) as f:
            return json.load(f)
    return {}


def load_tasks(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load tasks from tasks.jsonl."""
    tasks_file = dataset_path / "meta" / "tasks.jsonl"
    tasks = []
    if tasks_file.exists():
        with open(tasks_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
    return tasks


def load_episodes(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load episodes from episodes.jsonl."""
    episodes_file = dataset_path / "meta" / "episodes.jsonl"
    episodes = []
    if episodes_file.exists():
        with open(episodes_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
    return episodes


def combine_datasets(
    input_dir: str,
    output_path: str,
    dry_run: bool = False
) -> bool:
    """
    Combine all Inspire datasets into a single unified dataset.

    Args:
        input_dir: Directory containing converted Inspire datasets
        output_path: Path for combined output dataset
        dry_run: If True, only analyze without combining

    Returns:
        True if successful
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    print(f"\n{'='*60}")
    print("Combining Inspire Datasets")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output path: {output_path}")

    # Find all datasets
    print("\n[1/5] Finding datasets...")
    datasets = find_inspire_datasets(input_dir)

    if not datasets:
        print("ERROR: No valid datasets found")
        return False

    print(f"  Found {len(datasets)} datasets:")
    for ds in datasets:
        info = load_dataset_info(ds)
        frames = info.get("total_frames", "?")
        episodes = info.get("total_episodes", "?")
        print(f"    - {ds.name}: {episodes} episodes, {frames} frames")

    if dry_run:
        print("\n[DRY RUN] Would combine these datasets")
        return True

    # Create output directory
    print("\n[2/5] Setting up output directory...")
    output_path.mkdir(parents=True, exist_ok=True)
    data_dir = output_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Combine parquet files
    print("\n[3/5] Combining parquet files...")

    all_tables = []
    episode_offset = 0
    total_frames = 0
    all_tasks = []
    all_episodes = []
    task_offset = 0

    for ds_idx, dataset in enumerate(datasets):
        print(f"  Processing {dataset.name}...")

        # Load tasks and add offset
        tasks = load_tasks(dataset)
        for task in tasks:
            task["task_index"] = task.get("task_index", 0) + task_offset
            all_tasks.append(task)

        # Load episodes and add offset
        episodes = load_episodes(dataset)
        for ep in episodes:
            ep["episode_index"] = ep.get("episode_index", 0) + episode_offset
            ep["task_index"] = ep.get("task_index", 0) + task_offset
            all_episodes.append(ep)

        # Find parquet files
        parquet_files = list((dataset / "data").rglob("*.parquet"))
        if not parquet_files:
            parquet_files = list(dataset.rglob("*.parquet"))

        for pq_file in parquet_files:
            table = pq.read_table(pq_file)

            # Update episode_index with offset
            if "episode_index" in table.column_names:
                episode_col = table.column("episode_index").to_pylist()
                episode_col = [e + episode_offset for e in episode_col]
                table = table.drop("episode_index")
                table = table.append_column("episode_index", pa.array(episode_col))

            # Update task_index if present
            if "task_index" in table.column_names:
                task_col = table.column("task_index").to_pylist()
                task_col = [t + task_offset for t in task_col]
                table = table.drop("task_index")
                table = table.append_column("task_index", pa.array(task_col))

            all_tables.append(table)
            total_frames += table.num_rows

        # Update offsets
        info = load_dataset_info(dataset)
        episode_offset += info.get("total_episodes", len(episodes))
        task_offset += len(tasks)

    # Concatenate all tables
    print(f"  Concatenating {len(all_tables)} parquet files...")

    if all_tables:
        # Ensure all tables have the same schema by aligning columns
        all_columns = set()
        for table in all_tables:
            all_columns.update(table.column_names)

        aligned_tables = []
        for table in all_tables:
            missing = all_columns - set(table.column_names)
            for col in missing:
                # Add missing column with None values
                null_array = pa.array([None] * table.num_rows)
                table = table.append_column(col, null_array)
            aligned_tables.append(table)

        combined = pa.concat_tables(aligned_tables, promote_options="default")

        # Write combined parquet
        output_parquet = data_dir / "train-00000-of-00001.parquet"
        pq.write_table(combined, output_parquet)
        print(f"  Wrote {combined.num_rows} rows to {output_parquet.name}")

    # Copy video files
    print("\n[4/5] Handling video files...")
    video_dir = output_path / "videos"
    video_dir.mkdir(exist_ok=True)

    video_count = 0
    for dataset in datasets:
        src_videos = dataset / "videos"
        if src_videos.exists():
            for video_file in src_videos.iterdir():
                if video_file.suffix in [".mp4", ".avi", ".mkv"]:
                    dst = video_dir / f"{dataset.name}_{video_file.name}"
                    shutil.copy2(video_file, dst)
                    video_count += 1

    print(f"  Copied {video_count} video files")

    # Generate metadata
    print("\n[5/5] Generating metadata...")

    meta_dir = output_path / "meta"
    meta_dir.mkdir(exist_ok=True)

    # info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "unitree_g1_inspire",
        "total_episodes": episode_offset,
        "total_frames": total_frames,
        "fps": 30,
        "splits": {"train": f"0:{episode_offset}"},
        "source_datasets": [ds.name for ds in datasets],
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [53],
                "description": "G1 body (29 DOF) + Inspire hands (24 DOF)"
            },
            "action": {
                "dtype": "float32",
                "shape": [53],
                "description": "Action vector (53 DOF)"
            },
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float64", "shape": [1]},
            "task": {"dtype": "string", "shape": [1]},
        }
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # modality.json
    modality = {
        "state": {
            "left_leg": list(range(0, 6)),
            "right_leg": list(range(6, 12)),
            "waist": list(range(12, 15)),
            "left_arm": list(range(15, 22)),
            "right_arm": list(range(22, 29)),
            "left_inspire_hand": list(range(29, 41)),
            "right_inspire_hand": list(range(41, 53)),
        },
        "action": {
            "left_leg": list(range(0, 6)),
            "right_leg": list(range(6, 12)),
            "waist": list(range(12, 15)),
            "left_arm": list(range(15, 22)),
            "right_arm": list(range(22, 29)),
            "left_inspire_hand": list(range(29, 41)),
            "right_inspire_hand": list(range(41, 53)),
        },
    }

    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    # tasks.jsonl
    if all_tasks:
        with open(meta_dir / "tasks.jsonl", "w") as f:
            for task in all_tasks:
                f.write(json.dumps(task) + "\n")

    # episodes.jsonl
    if all_episodes:
        with open(meta_dir / "episodes.jsonl", "w") as f:
            for ep in all_episodes:
                f.write(json.dumps(ep) + "\n")

    print(f"\n{'='*60}")
    print("Combination Complete!")
    print(f"{'='*60}")
    print(f"  Total datasets combined: {len(datasets)}")
    print(f"  Total episodes: {episode_offset}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total tasks: {len(all_tasks)}")
    print(f"  Output path: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple Inspire-format datasets into one",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Combine all datasets in the inspire directory
    python combine_inspire_datasets.py --input-dir /workspace/datasets_inspire \\
                                       --output /workspace/datasets/G1_Inspire_Combined

    # Dry run to see what would be combined
    python combine_inspire_datasets.py --input-dir /workspace/datasets_inspire \\
                                       --output /workspace/datasets/G1_Inspire_Combined --dry-run
        """
    )

    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Directory containing converted Inspire datasets"
    )

    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for combined dataset"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze without combining"
    )

    args = parser.parse_args()

    success = combine_datasets(
        input_dir=args.input_dir,
        output_path=args.output,
        dry_run=args.dry_run
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
