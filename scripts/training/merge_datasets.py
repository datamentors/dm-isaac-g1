"""Merge multiple GR00T UNITREE_G1 datasets into one combined dataset.

Combines converted hospitality datasets into a single training dataset
by renumbering episodes and consolidating metadata.

Usage:
    python scripts/training/merge_datasets.py \\
        --input-dir /workspace/datasets/groot \\
        --output /workspace/datasets/groot_merged \\
        --datasets G1_Fold_Towel G1_Clean_Table G1_Wipe_Table \\
                   G1_Prepare_Fruit G1_Pour_Medicine G1_Organize_Tools G1_Pack_PingPong
"""

import argparse
import json
import shutil
from pathlib import Path


CHUNK_SIZE = 1000
VIDEO_KEY = "observation.images.ego_view"


def merge_datasets(input_dir: Path, output_dir: Path, dataset_names: list[str]):
    """Merge multiple GR00T datasets into one."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    (output_dir / "meta").mkdir()
    (output_dir / "data").mkdir()

    global_ep = 0
    total_frames = 0
    all_tasks = {}
    all_episodes = []
    all_episodes_stats = []

    for ds_name in dataset_names:
        ds_path = input_dir / ds_name
        info_path = ds_path / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)

        n_eps = info["total_episodes"]
        print(f"\n=== {ds_name}: {n_eps} episodes ===")

        # Read task descriptions
        tasks_path = ds_path / "meta" / "tasks.jsonl"
        ds_tasks = {}
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    t = json.loads(line)
                    ds_tasks[t["task_index"]] = t["task"]

        # Read episodes info
        episodes_path = ds_path / "meta" / "episodes.jsonl"
        ds_episodes = []
        if episodes_path.exists():
            with open(episodes_path) as f:
                for line in f:
                    ds_episodes.append(json.loads(line))

        # Read episodes stats
        estats_path = ds_path / "meta" / "episodes_stats.jsonl"
        ds_estats = []
        if estats_path.exists():
            with open(estats_path) as f:
                for line in f:
                    ds_estats.append(json.loads(line))

        for local_ep in range(n_eps):
            chunk_src = f"chunk-{local_ep // CHUNK_SIZE:03d}"
            chunk_dst = f"chunk-{global_ep // CHUNK_SIZE:03d}"

            # Copy parquet
            src_pq = ds_path / "data" / chunk_src / f"episode_{local_ep:06d}.parquet"
            dst_pq_dir = output_dir / "data" / chunk_dst
            dst_pq_dir.mkdir(parents=True, exist_ok=True)
            dst_pq = dst_pq_dir / f"episode_{global_ep:06d}.parquet"
            shutil.copy2(src_pq, dst_pq)

            # Copy video
            src_vid_dir = ds_path / "videos" / chunk_src / VIDEO_KEY
            dst_vid_dir = output_dir / "videos" / chunk_dst / VIDEO_KEY
            dst_vid_dir.mkdir(parents=True, exist_ok=True)
            src_vid = src_vid_dir / f"episode_{local_ep:06d}.mp4"
            dst_vid = dst_vid_dir / f"episode_{global_ep:06d}.mp4"
            if src_vid.exists():
                shutil.copy2(src_vid, dst_vid)

            # Remap episode info
            if local_ep < len(ds_episodes):
                ep_info = ds_episodes[local_ep].copy()
                task_desc = ds_tasks.get(
                    ep_info.get("task_index", 0),
                    ds_name.replace("G1_", "").replace("_", " "),
                )
                if task_desc not in all_tasks.values():
                    task_idx = len(all_tasks)
                    all_tasks[task_idx] = task_desc
                else:
                    task_idx = [k for k, v in all_tasks.items() if v == task_desc][0]
                ep_info["episode_index"] = global_ep
                ep_info["task_index"] = task_idx
                all_episodes.append(ep_info)

            if local_ep < len(ds_estats):
                estat = ds_estats[local_ep].copy()
                estat["episode_index"] = global_ep
                all_episodes_stats.append(estat)

            # Track frames
            if local_ep < len(ds_episodes):
                total_frames += ds_episodes[local_ep].get("length", 0)

            global_ep += 1

            if global_ep % 200 == 0:
                print(f"  Processed {global_ep} episodes...")

    print(f"\nTotal episodes: {global_ep}")
    print(f"Total frames: {total_frames}")

    # Copy modality.json from first dataset
    shutil.copy2(
        input_dir / dataset_names[0] / "meta" / "modality.json",
        output_dir / "meta" / "modality.json",
    )

    # Write info.json
    n_chunks = (global_ep + CHUNK_SIZE - 1) // CHUNK_SIZE
    info_merged = {
        "codebase_version": "v2.1",
        "robot_type": "unitree_g1",
        "total_episodes": global_ep,
        "total_frames": total_frames,
        "total_tasks": len(all_tasks),
        "total_videos": global_ep,
        "total_chunks": n_chunks,
        "chunks_size": CHUNK_SIZE,
        "fps": 30,
        "splits": {"train": f"0:{global_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [31]},
            "action": {"dtype": "float32", "shape": [23]},
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "video_info": {
                    "video.fps": 30.0,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "task": {"dtype": "string"},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
        },
    }
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info_merged, f, indent=2)

    # Write tasks.jsonl
    with open(output_dir / "meta" / "tasks.jsonl", "w") as f:
        for idx, task in sorted(all_tasks.items()):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    # Write episodes.jsonl
    with open(output_dir / "meta" / "episodes.jsonl", "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep) + "\n")

    # Write episodes_stats.jsonl
    with open(output_dir / "meta" / "episodes_stats.jsonl", "w") as f:
        for es in all_episodes_stats:
            f.write(json.dumps(es) + "\n")

    print(f"\nMerge complete at {output_dir}")
    print(f"Tasks: {all_tasks}")


def main():
    parser = argparse.ArgumentParser(description="Merge GR00T datasets")
    parser.add_argument(
        "--input-dir",
        default="/workspace/datasets/groot",
        help="Directory containing individual converted datasets",
    )
    parser.add_argument(
        "--output",
        default="/workspace/datasets/groot_merged",
        help="Output merged dataset path",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "G1_Fold_Towel",
            "G1_Clean_Table",
            "G1_Wipe_Table",
            "G1_Prepare_Fruit",
            "G1_Pour_Medicine",
            "G1_Organize_Tools",
            "G1_Pack_PingPong",
        ],
        help="Dataset names to merge",
    )
    args = parser.parse_args()

    merge_datasets(Path(args.input_dir), Path(args.output), args.datasets)


if __name__ == "__main__":
    main()
