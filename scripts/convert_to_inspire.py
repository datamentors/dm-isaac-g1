#!/usr/bin/env python3
"""
Convert G1 robot datasets to unified Inspire Hand format (53 DOF).

Supports multiple dataset formats:
1. Hospitality (G1_ALL): observation.body + observation.left/right_gripper
2. Dex3: observation.state (28 DOF) = arms + Dex3 hands (upper body only)
3. Teleop: Various formats

Target: G1 Body (29 DOF) + Inspire Hands (24 DOF) = 53 DOF total
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow is required. Install with: pip install pyarrow")
    exit(1)


# =============================================================================
# Constants
# =============================================================================

BODY_DOF = 29
INSPIRE_HAND_DOF = 12
TOTAL_DOF = 53


# =============================================================================
# Hand Mapping Functions
# =============================================================================

def gripper_to_inspire(gripper_value: float) -> List[float]:
    """Convert gripper (1 DOF) to Inspire (12 DOF)."""
    inspire = [0.0] * 12
    inspire[0] = gripper_value   # index_proximal
    inspire[2] = gripper_value   # middle_proximal
    inspire[4] = gripper_value   # pinky_proximal
    inspire[6] = gripper_value   # ring_proximal
    inspire[9] = gripper_value   # thumb_proximal_pitch
    return inspire


def dex3_to_inspire(dex3: List[float]) -> List[float]:
    """
    Convert Dex3 hand (7 DOF) to Inspire (12 DOF).

    Dex3 order: [Thumb0, Thumb1, Thumb2, Middle0, Middle1, Index0, Index1]
    Inspire order: [index_prox, index_inter, middle_prox, middle_inter,
                   pinky_prox, pinky_inter, ring_prox, ring_inter,
                   thumb_yaw, thumb_pitch, thumb_inter, thumb_dist]
    """
    if len(dex3) != 7:
        # Pad or truncate
        dex3 = list(dex3)[:7] + [0.0] * max(0, 7 - len(dex3))

    inspire = [0.0] * 12
    # Index: Dex3 indices 5,6 -> Inspire 0,1
    inspire[0] = dex3[5]   # Index0 -> index_proximal
    inspire[1] = dex3[6]   # Index1 -> index_intermediate
    # Middle: Dex3 indices 3,4 -> Inspire 2,3
    inspire[2] = dex3[3]   # Middle0 -> middle_proximal
    inspire[3] = dex3[4]   # Middle1 -> middle_intermediate
    # Thumb: Dex3 indices 0,1,2 -> Inspire 9,10,11
    inspire[9] = dex3[0]   # Thumb0 -> thumb_proximal_pitch
    inspire[10] = dex3[1]  # Thumb1 -> thumb_intermediate
    inspire[11] = dex3[2]  # Thumb2 -> thumb_distal
    # Ring and pinky stay zero
    return inspire


def trifinger_to_inspire(trifinger: List[float]) -> List[float]:
    """Convert Tri-finger (7 DOF) to Inspire (12 DOF)."""
    if len(trifinger) != 7:
        trifinger = list(trifinger)[:7] + [0.0] * max(0, 7 - len(trifinger))

    inspire = [0.0] * 12
    inspire[0] = trifinger[3]   # index_proximal
    inspire[1] = trifinger[4]   # index_intermediate
    inspire[2] = trifinger[5]   # middle_proximal
    inspire[3] = trifinger[6]   # middle_intermediate
    inspire[9] = trifinger[0]   # thumb_proximal_pitch
    inspire[10] = trifinger[1]  # thumb_intermediate
    inspire[11] = trifinger[2]  # thumb_distal
    return inspire


def to_list(val) -> List[float]:
    """Convert various types to float list."""
    if val is None:
        return []
    if hasattr(val, 'tolist'):
        val = val.tolist()
    if isinstance(val, (int, float)):
        return [float(val)]
    return [float(x) for x in val]


# =============================================================================
# Format Detection
# =============================================================================

def detect_format(info: Dict) -> Dict[str, Any]:
    """Detect dataset format from info.json."""
    features = info.get("features", {})
    feature_keys = list(features.keys())

    result = {
        "format": "unknown",
        "hand_type": "gripper",
        "has_body": False,
        "has_separate_parts": False,
        "has_observation_state": False,
        "state_shape": None,
        "joint_names": [],
    }

    # Check for observation.state (combined format)
    if "observation.state" in features:
        result["has_observation_state"] = True
        result["state_shape"] = features["observation.state"].get("shape", [])
        names = features["observation.state"].get("names", [])
        if names and isinstance(names[0], list):
            names = names[0]
        result["joint_names"] = names

        # Check format based on shape
        if result["state_shape"]:
            state_size = result["state_shape"][0]

            # 43 DOF = full body (29) + tri-finger hands (14)
            if state_size == 43:
                if any("hand" in str(n).lower() for n in names):
                    result["format"] = "teleop_trifinger"
                    result["hand_type"] = "trifinger"

            # 28 DOF = arms (14) + Dex3 hands (14) - upper body only
            elif state_size == 28:
                if any("Hand" in str(n) for n in names):
                    result["format"] = "dex3_combined"
                    result["hand_type"] = "dex3"

    # Check for separate body parts (Hospitality format)
    if "observation.body" in features:
        result["has_body"] = True
        result["format"] = "hospitality"
        result["has_separate_parts"] = True

    # Detect hand type from feature names
    for key in feature_keys:
        if "gripper" in key.lower():
            result["hand_type"] = "gripper"
            if result["format"] == "unknown":
                result["format"] = "hospitality"

    return result


# =============================================================================
# Data Processing
# =============================================================================

def process_hospitality_row(row: Dict, hand_type: str) -> tuple:
    """Process a row from hospitality format (observation.body + grippers)."""
    # Get body state (29 DOF)
    body = to_list(row.get("observation.body", []))
    if len(body) < 29:
        body = body + [0.0] * (29 - len(body))
    elif len(body) > 29:
        body = body[:29]

    # Get grippers
    left_grip = to_list(row.get("observation.left_gripper", [0.0]))
    right_grip = to_list(row.get("observation.right_gripper", [0.0]))

    # Convert to Inspire
    left_inspire = gripper_to_inspire(left_grip[0] if left_grip else 0.0)
    right_inspire = gripper_to_inspire(right_grip[0] if right_grip else 0.0)

    obs_state = body + left_inspire + right_inspire

    # Actions
    left_arm_action = to_list(row.get("action.left_arm", []))
    right_arm_action = to_list(row.get("action.right_arm", []))
    left_grip_action = to_list(row.get("action.left_gripper", [0.0]))
    right_grip_action = to_list(row.get("action.right_gripper", [0.0]))

    # Build action body
    action_body = list(body)  # Start with observation body
    if left_arm_action:
        action_body[15:22] = left_arm_action[:7] + [0.0] * max(0, 7 - len(left_arm_action))
    if right_arm_action:
        action_body[22:29] = right_arm_action[:7] + [0.0] * max(0, 7 - len(right_arm_action))

    left_inspire_action = gripper_to_inspire(left_grip_action[0] if left_grip_action else 0.0)
    right_inspire_action = gripper_to_inspire(right_grip_action[0] if right_grip_action else 0.0)

    action = action_body + left_inspire_action + right_inspire_action

    return obs_state, action


def process_dex3_combined_row(row: Dict) -> tuple:
    """
    Process a row from Dex3 combined format (observation.state with 28 DOF).

    Layout: [left_arm(7), right_arm(7), left_dex3(7), right_dex3(7)]
    Target: [legs(12), waist(3), left_arm(7), right_arm(7), left_inspire(12), right_inspire(12)]
    """
    obs_state_28 = to_list(row.get("observation.state", [0.0] * 28))
    action_28 = to_list(row.get("action", [0.0] * 28))

    # Pad to 28 if needed
    obs_state_28 = obs_state_28[:28] + [0.0] * max(0, 28 - len(obs_state_28))
    action_28 = action_28[:28] + [0.0] * max(0, 28 - len(action_28))

    # Parse 28 DOF
    left_arm = obs_state_28[0:7]
    right_arm = obs_state_28[7:14]
    left_dex3 = obs_state_28[14:21]
    right_dex3 = obs_state_28[21:28]

    left_arm_action = action_28[0:7]
    right_arm_action = action_28[7:14]
    left_dex3_action = action_28[14:21]
    right_dex3_action = action_28[21:28]

    # Build 53 DOF body: legs(12) + waist(3) + arms(14)
    # Legs and waist are zero (upper body only in Dex3 data)
    body_29 = [0.0] * 15 + left_arm + right_arm  # 15 zeros + 7 + 7 = 29

    # Convert hands
    left_inspire = dex3_to_inspire(left_dex3)
    right_inspire = dex3_to_inspire(right_dex3)

    obs_state = body_29 + left_inspire + right_inspire

    # Action body
    action_body = [0.0] * 15 + left_arm_action + right_arm_action
    left_inspire_action = dex3_to_inspire(left_dex3_action)
    right_inspire_action = dex3_to_inspire(right_dex3_action)

    action = action_body + left_inspire_action + right_inspire_action

    return obs_state, action


def process_teleop_trifinger_row(row: Dict) -> tuple:
    """
    Process a row from Teleop tri-finger format (observation.state with 43 DOF).

    Layout: [legs(12), waist(3), left_arm(7), left_hand(7), right_arm(7), right_hand(7)]
    Target: [legs(12), waist(3), left_arm(7), right_arm(7), left_inspire(12), right_inspire(12)]
    """
    obs_state_43 = to_list(row.get("observation.state", [0.0] * 43))
    action_43 = to_list(row.get("action", [0.0] * 43))

    # Pad to 43 if needed
    obs_state_43 = obs_state_43[:43] + [0.0] * max(0, 43 - len(obs_state_43))
    action_43 = action_43[:43] + [0.0] * max(0, 43 - len(action_43))

    # Parse 43 DOF
    legs = obs_state_43[0:12]
    waist = obs_state_43[12:15]
    left_arm = obs_state_43[15:22]
    left_trifinger = obs_state_43[22:29]
    right_arm = obs_state_43[29:36]
    right_trifinger = obs_state_43[36:43]

    # Build 29 DOF body: legs(12) + waist(3) + left_arm(7) + right_arm(7)
    body_29 = legs + waist + left_arm + right_arm

    # Convert hands from tri-finger to Inspire
    left_inspire = trifinger_to_inspire(left_trifinger)
    right_inspire = trifinger_to_inspire(right_trifinger)

    obs_state = body_29 + left_inspire + right_inspire

    # Same for actions
    legs_action = action_43[0:12]
    waist_action = action_43[12:15]
    left_arm_action = action_43[15:22]
    left_trifinger_action = action_43[22:29]
    right_arm_action = action_43[29:36]
    right_trifinger_action = action_43[36:43]

    action_body = legs_action + waist_action + left_arm_action + right_arm_action
    left_inspire_action = trifinger_to_inspire(left_trifinger_action)
    right_inspire_action = trifinger_to_inspire(right_trifinger_action)

    action = action_body + left_inspire_action + right_inspire_action

    return obs_state, action


def process_parquet_file(
    input_file: Path,
    output_file: Path,
    format_info: Dict,
    info: Dict
) -> int:
    """Process a single parquet file."""
    table = pq.read_table(input_file)
    cols = table.column_names
    num_rows = table.num_rows
    data = {col: table.column(col).to_pylist() for col in cols}

    output_data = {
        "observation.state": [],
        "action": [],
    }

    # Passthrough columns
    passthrough = ["frame_index", "episode_index", "timestamp", "task", "index", "task_index"]
    for col in cols:
        if any(p in col.lower() for p in passthrough):
            output_data[col] = data[col]
        elif "observation.images" in col:
            output_data[col] = data[col]

    fmt = format_info["format"]

    for i in range(num_rows):
        row = {col: data[col][i] for col in data}

        if fmt == "hospitality":
            obs_state, action = process_hospitality_row(row, format_info["hand_type"])
        elif fmt == "dex3_combined":
            obs_state, action = process_dex3_combined_row(row)
        elif fmt == "teleop_trifinger":
            obs_state, action = process_teleop_trifinger_row(row)
        else:
            # Fallback: try to extract what we can
            state = to_list(row.get("observation.state", []))
            act = to_list(row.get("action", []))
            # Pad to 53
            obs_state = state[:53] + [0.0] * max(0, 53 - len(state))
            action = act[:53] + [0.0] * max(0, 53 - len(act))

        output_data["observation.state"].append(obs_state)
        output_data["action"].append(action)

    # Write output
    output_table = pa.table(output_data)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output_table, output_file)

    return num_rows


def generate_metadata(output_path: Path, info: Dict, total_frames: int, num_episodes: int, format_info: Dict):
    """Generate metadata files."""
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Get video features
    video_features = {k: v for k, v in info.get("features", {}).items() if "observation.images" in k}

    new_info = {
        "codebase_version": "v2.1",
        "robot_type": "unitree_g1_inspire",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "fps": info.get("fps", 30),
        "splits": {"train": f"0:{num_episodes}"},
        "data_path": info.get("data_path", "data/chunk-000/episode_{episode_index:06d}.parquet"),
        "video_path": info.get("video_path", "videos/{video_key}/episode_{episode_index:06d}.mp4"),
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [53],
                "description": "G1 body (29 DOF) + Inspire hands (24 DOF)"
            },
            "action": {
                "dtype": "float32",
                "shape": [53],
                "description": "Action: body (29) + hands (24)"
            },
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            **video_features
        },
        "conversion_info": {
            "source_format": format_info["format"],
            "source_hand_type": format_info["hand_type"],
            "target_hand_type": "inspire",
            "total_dof": 53,
        }
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(new_info, f, indent=2)

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

    if video_features:
        modality["video"] = list(video_features.keys())[0]

    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)


def copy_support_files(input_path: Path, output_path: Path):
    """Copy support files."""
    meta_in = input_path / "meta"
    meta_out = output_path / "meta"
    meta_out.mkdir(parents=True, exist_ok=True)

    for filename in ["episodes.jsonl", "tasks.jsonl"]:
        src = meta_in / filename
        if src.exists():
            shutil.copy2(src, meta_out / filename)
            print(f"  Copied {filename}")

    video_dir = input_path / "videos"
    if video_dir.exists():
        output_video = output_path / "videos"
        if output_video.exists():
            shutil.rmtree(output_video)
        print(f"  Copying videos (may take a while)...")
        shutil.copytree(video_dir, output_video)
        print(f"  Copied videos")


def convert_dataset(
    input_path: str,
    output_path: str,
    hand_type: Optional[str] = None,
    dry_run: bool = False
) -> bool:
    """Convert dataset to 53 DOF Inspire format."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"\n{'='*60}")
    print("Converting to Inspire format (53 DOF)")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    info_path = input_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"ERROR: info.json not found")
        return False

    with open(info_path) as f:
        info = json.load(f)

    format_info = detect_format(info)
    if hand_type:
        format_info["hand_type"] = hand_type

    print(f"\n[INFO]")
    print(f"  Robot type: {info.get('robot_type', 'unknown')}")
    print(f"  Episodes: {info.get('total_episodes', '?')}")
    print(f"  Frames: {info.get('total_frames', '?')}")
    print(f"  Detected format: {format_info['format']}")
    print(f"  Hand type: {format_info['hand_type']}")
    if format_info.get('state_shape'):
        print(f"  State shape: {format_info['state_shape']}")

    if dry_run:
        print(f"\n[DRY RUN] Would convert to 53 DOF")
        return True

    output_path.mkdir(parents=True, exist_ok=True)

    # Find parquet files
    data_dir = input_path / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))

    if not parquet_files:
        print("ERROR: No parquet files found")
        return False

    print(f"\n[Processing {len(parquet_files)} files...]")

    total_frames = 0
    for i, pq_file in enumerate(parquet_files):
        rel_path = pq_file.relative_to(input_path)
        out_file = output_path / rel_path

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(parquet_files)}] {pq_file.name}")

        try:
            frames = process_parquet_file(pq_file, out_file, format_info, info)
            total_frames += frames
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    num_episodes = len(parquet_files)

    print("\n[Generating metadata...]")
    generate_metadata(output_path, info, total_frames, num_episodes, format_info)

    print("\n[Copying support files...]")
    copy_support_files(input_path, output_path)

    print(f"\n{'='*60}")
    print("Complete!")
    print(f"{'='*60}")
    print(f"  Frames: {total_frames}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Output: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Convert G1 datasets to 53 DOF Inspire format")
    parser.add_argument("--input", "-i", required=True, help="Input dataset")
    parser.add_argument("--output", "-o", required=True, help="Output dataset")
    parser.add_argument("--hand-type", "-t", choices=["gripper", "dex3", "trifinger"], help="Override hand type")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only")
    args = parser.parse_args()

    success = convert_dataset(args.input, args.output, args.hand_type, args.dry_run)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
