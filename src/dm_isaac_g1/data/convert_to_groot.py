"""Convert hospitality datasets to GR00T UNITREE_G1 format.

Takes per-body-part LeRobot v2 datasets (from unitreerobotics) and produces
GR00T-compatible datasets with:
  - Single `observation.state` column (flat float32 array)
  - Single `action` column (flat float32 array)
  - `meta/modality.json` mapping indices to UNITREE_G1 body part names
  - Video symlinked/copied for one ego-view camera

State layout (31 DOF):
  left_leg:   6 DOF (indices  0- 5) — from observation.body[0:6]
  right_leg:  6 DOF (indices  6-11) — from observation.body[6:12]
  waist:      3 DOF (indices 12-14) — from observation.body[12:15]
  left_arm:   7 DOF (indices 15-21) — from observation.body[15:22] or observation.left_arm
  right_arm:  7 DOF (indices 22-28) — from observation.body[22:29] or observation.right_arm
  left_hand:  1 DOF (index   29)    — from observation.left_gripper
  right_hand: 1 DOF (index   30)    — from observation.right_gripper

Action layout (23 DOF):
  left_arm:            7 DOF (indices  0- 6) — from action.left_arm (RELATIVE)
  right_arm:           7 DOF (indices  7-13) — from action.right_arm (RELATIVE)
  left_hand:           1 DOF (index   14)    — from action.left_gripper (ABSOLUTE)
  right_hand:          1 DOF (index   15)    — from action.right_gripper (ABSOLUTE)
  waist:               3 DOF (indices 16-18) — from action.body[3:6] (Yaw,Pitch,Roll) (ABSOLUTE)
  base_height_command: 1 DOF (index   19)    — from action.body[6] (Height) (ABSOLUTE)
  navigate_command:    3 DOF (indices 20-22) — from action.body[0:3] (VX,VY,AngZ) (ABSOLUTE)

Usage:
    python -m dm_isaac_g1.data.convert_to_groot \\
        --input /workspace/datasets/hospitality/G1_Fold_Towel \\
        --output /workspace/datasets/groot/G1_Fold_Towel \\
        --ego-camera cam_left_high
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


# State index layout
STATE_LAYOUT = {
    "left_leg":   {"start": 0,  "end": 6},
    "right_leg":  {"start": 6,  "end": 12},
    "waist":      {"start": 12, "end": 15},
    "left_arm":   {"start": 15, "end": 22},
    "right_arm":  {"start": 22, "end": 29},
    "left_hand":  {"start": 29, "end": 30},
    "right_hand": {"start": 30, "end": 31},
}
TOTAL_STATE_DOF = 31

# Action index layout
ACTION_LAYOUT = {
    "left_arm":            {"start": 0,  "end": 7},
    "right_arm":           {"start": 7,  "end": 14},
    "left_hand":           {"start": 14, "end": 15},
    "right_hand":          {"start": 15, "end": 16},
    "waist":               {"start": 16, "end": 19},
    "base_height_command": {"start": 19, "end": 20},
    "navigate_command":    {"start": 20, "end": 23},
}
TOTAL_ACTION_DOF = 23


def _to_np(val, expected_len: int) -> np.ndarray:
    """Convert value to numpy array, padding/truncating to expected length."""
    if val is None:
        return np.zeros(expected_len, dtype=np.float32)
    arr = np.asarray(val, dtype=np.float32).flatten()
    if len(arr) < expected_len:
        arr = np.pad(arr, (0, expected_len - len(arr)))
    return arr[:expected_len]


def process_row(row: dict) -> tuple:
    """Convert a single row from hospitality format to UNITREE_G1 format.

    Returns:
        (observation_state, action) as numpy arrays.
    """
    # --- Build observation.state (31 DOF) ---
    body = _to_np(row.get("observation.body"), 29)

    left_gripper = _to_np(row.get("observation.left_gripper"), 1)
    right_gripper = _to_np(row.get("observation.right_gripper"), 1)

    # body layout: [left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)]
    obs_state = np.concatenate([
        body,           # 29 DOF (legs + waist + arms)
        left_gripper,   # 1 DOF
        right_gripper,  # 1 DOF
    ])  # Total: 31

    # --- Build action (23 DOF) ---
    left_arm_action = _to_np(row.get("action.left_arm"), 7)
    right_arm_action = _to_np(row.get("action.right_arm"), 7)
    left_grip_action = _to_np(row.get("action.left_gripper"), 1)
    right_grip_action = _to_np(row.get("action.right_gripper"), 1)
    body_action = _to_np(row.get("action.body"), 7)

    # action.body layout: [VX, VY, AngZ, Yaw, Pitch, Roll, Height]
    waist_action = body_action[3:6]       # Yaw, Pitch, Roll
    height_action = body_action[6:7]      # Height
    navigate_action = body_action[0:3]    # VX, VY, AngZ

    action = np.concatenate([
        left_arm_action,    # 7 DOF (RELATIVE)
        right_arm_action,   # 7 DOF (RELATIVE)
        left_grip_action,   # 1 DOF (ABSOLUTE)
        right_grip_action,  # 1 DOF (ABSOLUTE)
        waist_action,       # 3 DOF (ABSOLUTE)
        height_action,      # 1 DOF (ABSOLUTE)
        navigate_action,    # 3 DOF (ABSOLUTE)
    ])  # Total: 23

    return obs_state, action


def convert_parquet_file(
    input_file: Path,
    output_file: Path,
    ego_camera: str,
) -> int:
    """Convert a single parquet file."""
    table = pq.read_table(input_file)
    cols = table.column_names
    num_rows = table.num_rows

    data = {col: table.column(col).to_pylist() for col in cols}

    obs_states = []
    actions = []

    for i in range(num_rows):
        row = {col: data[col][i] for col in data}
        obs_state, action = process_row(row)
        obs_states.append(obs_state.tolist())
        actions.append(action.tolist())

    # Build output columns
    output_data = {
        "observation.state": obs_states,
        "action": actions,
    }

    # Passthrough metadata columns
    passthrough_cols = [
        "frame_index", "episode_index", "timestamp",
        "index", "task_index",
    ]
    for col in passthrough_cols:
        if col in data:
            output_data[col] = data[col]

    # Keep only the ego camera image reference, renamed to ego_view
    ego_source_key = f"observation.images.{ego_camera}"
    ego_output_key = "observation.images.ego_view"
    if ego_source_key in data:
        output_data[ego_output_key] = data[ego_source_key]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_table = pa.table(output_data)
    pq.write_table(output_table, output_file)

    return num_rows


def generate_modality_json(output_path: Path, ego_camera: str):
    """Generate modality.json for UNITREE_G1 embodiment."""
    modality = {
        "state": {
            name: {
                "start": info["start"],
                "end": info["end"],
                "original_key": "observation.state",
            }
            for name, info in STATE_LAYOUT.items()
        },
        "action": {
            name: {
                "start": info["start"],
                "end": info["end"],
                "original_key": "action",
            }
            for name, info in ACTION_LAYOUT.items()
        },
        "video": {
            "ego_view": {
                "original_key": "observation.images.ego_view",
            },
        },
        "annotation": {
            "human.task_description": {
                "original_key": "task_index",
            },
        },
    }

    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    print(f"  Wrote modality.json")


def generate_info_json(
    output_path: Path,
    source_info: dict,
    total_frames: int,
    num_episodes: int,
    ego_camera: str,
):
    """Generate info.json for converted dataset."""
    ego_source_key = f"observation.images.{ego_camera}"
    ego_video_key = "observation.images.ego_view"
    source_features = source_info.get("features", {})
    video_info = source_features.get(ego_source_key, {})

    new_info = {
        "codebase_version": "v2.1",
        "robot_type": "unitree_g1",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": source_info.get("total_tasks", 1),
        "total_videos": num_episodes,
        "total_chunks": source_info.get("total_chunks", 1),
        "chunks_size": source_info.get("chunks_size", 1000),
        "fps": source_info.get("fps", 30),
        "splits": source_info.get("splits", {"train": f"0:{num_episodes}"}),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [TOTAL_STATE_DOF],
                "names": [
                    "kLeftHipPitch", "kLeftHipRoll", "kLeftHipYaw",
                    "kLeftKnee", "kLeftAnklePitch", "kLeftAnkleRoll",
                    "kRightHipPitch", "kRightHipRoll", "kRightHipYaw",
                    "kRightKnee", "kRightAnklePitch", "kRightAnkleRoll",
                    "kWaistYaw", "kWaistRoll", "kWaistPitch",
                    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw",
                    "kLeftElbow", "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
                    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw",
                    "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
                    "kLeftGripper",
                    "kRightGripper",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": [TOTAL_ACTION_DOF],
                "names": [
                    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw",
                    "kLeftElbow", "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
                    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw",
                    "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
                    "kLeftGripper",
                    "kRightGripper",
                    "kWaistYaw", "kWaistPitch", "kWaistRoll",
                    "kHeight",
                    "kVX", "kVY", "kAngZ",
                ],
            },
            ego_video_key: video_info,
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }

    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "info.json", "w") as f:
        json.dump(new_info, f, indent=2)

    print(f"  Wrote info.json")


def convert_dataset(
    input_path: Path,
    output_path: Path,
    ego_camera: str = "cam_left_high",
    dry_run: bool = False,
) -> bool:
    """Convert hospitality dataset to GR00T UNITREE_G1 format.

    Args:
        input_path: Path to input hospitality dataset.
        output_path: Path for output GR00T-formatted dataset.
        ego_camera: Camera to use as ego_view (default: cam_left_high).
        dry_run: If True, analyze only.

    Returns:
        True if successful.
    """
    if pq is None:
        raise ImportError("pyarrow required: pip install pyarrow")

    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"\n{'='*60}")
    print("Converting to GR00T UNITREE_G1 format")
    print(f"{'='*60}")
    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Ego camera: {ego_camera}")

    # Load source info
    info_path = input_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"ERROR: {info_path} not found")
        return False

    with open(info_path) as f:
        source_info = json.load(f)

    print(f"  Robot type: {source_info.get('robot_type', 'unknown')}")
    print(f"  Episodes: {source_info.get('total_episodes', '?')}")
    print(f"  FPS: {source_info.get('fps', '?')}")

    # Verify ego camera exists in features
    ego_key = f"observation.images.{ego_camera}"
    features = source_info.get("features", {})
    if ego_key not in features:
        available = [k for k in features if "observation.images" in k]
        print(f"ERROR: Camera '{ego_camera}' not found. Available: {available}")
        return False

    if dry_run:
        print(f"\n[DRY RUN] Would convert {source_info.get('total_episodes', '?')} episodes")
        print(f"  State: {TOTAL_STATE_DOF} DOF, Action: {TOTAL_ACTION_DOF} DOF")
        return True

    output_path.mkdir(parents=True, exist_ok=True)

    # Process parquet files
    data_dir = input_path / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))

    if not parquet_files:
        print("ERROR: No parquet files found")
        return False

    print(f"\n[Processing {len(parquet_files)} parquet files...]")
    total_frames = 0
    for i, pq_file in enumerate(parquet_files):
        rel_path = pq_file.relative_to(input_path)
        out_file = output_path / rel_path

        if (i + 1) % 50 == 0 or i == 0 or i == len(parquet_files) - 1:
            print(f"  [{i+1}/{len(parquet_files)}] {pq_file.name}")

        frames = convert_parquet_file(pq_file, out_file, ego_camera)
        total_frames += frames

    # Generate metadata
    print("\n[Generating metadata...]")
    generate_modality_json(output_path, ego_camera)
    generate_info_json(output_path, source_info, total_frames, len(parquet_files), ego_camera)

    # Copy support files
    print("\n[Copying support files...]")
    meta_in = input_path / "meta"
    meta_out = output_path / "meta"
    for filename in ["episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"]:
        src = meta_in / filename
        if src.exists():
            shutil.copy2(src, meta_out / filename)
            print(f"  Copied {filename}")

    # Copy/symlink videos for the ego camera only
    ego_video_dir = input_path / "videos"
    if ego_video_dir.exists():
        print(f"\n[Copying ego-view videos ({ego_camera})...]")
        # Find the ego camera video subdirectories
        for chunk_dir in sorted(ego_video_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            src_cam_dir = chunk_dir / ego_camera
            if not src_cam_dir.exists():
                # Try with full key
                src_cam_dir = chunk_dir / ego_key
                if not src_cam_dir.exists():
                    continue

            dst_cam_dir = output_path / "videos" / chunk_dir.name / "observation.images.ego_view"
            dst_cam_dir.mkdir(parents=True, exist_ok=True)

            for mp4 in sorted(src_cam_dir.glob("*.mp4")):
                dst = dst_cam_dir / mp4.name
                if not dst.exists():
                    shutil.copy2(mp4, dst)

            print(f"  Copied {chunk_dir.name}/{ego_camera}")

    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"  Episodes: {len(parquet_files)}")
    print(f"  Frames: {total_frames}")
    print(f"  State DOF: {TOTAL_STATE_DOF}")
    print(f"  Action DOF: {TOTAL_ACTION_DOF}")
    print(f"{'='*60}")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert hospitality dataset to GR00T UNITREE_G1 format"
    )
    parser.add_argument("--input", required=True, help="Input dataset path")
    parser.add_argument("--output", required=True, help="Output dataset path")
    parser.add_argument(
        "--ego-camera", default="cam_left_high",
        help="Camera to use as ego_view (default: cam_left_high)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Analyze only")
    args = parser.parse_args()

    success = convert_dataset(
        Path(args.input), Path(args.output),
        ego_camera=args.ego_camera,
        dry_run=args.dry_run,
    )
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
