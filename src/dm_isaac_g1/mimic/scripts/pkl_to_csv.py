#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""Convert video2robot pickle output to mimic-compatible CSV.

video2robot (https://github.com/AIM-Intelligence/video2robot) produces a pickle
with keys: fps, robot_type, num_frames, root_pos (N,3), root_rot (N,4), dof_pos (N,29).

The mimic pipeline expects a CSV with 36 columns per frame (no header):
  Cols 0-2:   root position (x, y, z)
  Cols 3-6:   root quaternion (qx, qy, qz, qw) — xyzw convention
  Cols 7-35:  29 joint angles (generalized coordinates)

Usage:
    python pkl_to_csv.py --input retarget/unitree_g1.pkl --output ronaldo_celebration.csv
"""

import argparse
import pickle

import numpy as np


def convert_pkl_to_csv(input_path: str, output_path: str, target_fps: int | None = None) -> None:
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    fps = float(data["fps"])
    root_pos = np.array(data["root_pos"])       # (N, 3)
    root_rot = np.array(data["root_rot"])        # (N, 4) — xyzw quaternion
    dof_pos = np.array(data["dof_pos"])          # (N, 29)

    num_frames = root_pos.shape[0]
    print(f"Loaded: {num_frames} frames at {fps} fps, robot_type={data.get('robot_type', 'unknown')}")

    # Optional FPS resampling via linear interpolation
    if target_fps is not None and target_fps != fps:
        duration = (num_frames - 1) / fps
        old_times = np.linspace(0, duration, num_frames)
        new_num = int(duration * target_fps) + 1
        new_times = np.linspace(0, duration, new_num)

        root_pos = np.array([np.interp(new_times, old_times, root_pos[:, i]) for i in range(3)]).T
        root_rot = np.array([np.interp(new_times, old_times, root_rot[:, i]) for i in range(4)]).T
        # Renormalize quaternions after interpolation
        root_rot = root_rot / np.linalg.norm(root_rot, axis=-1, keepdims=True)
        dof_pos = np.array([np.interp(new_times, old_times, dof_pos[:, i]) for i in range(29)]).T

        num_frames = new_num
        print(f"Resampled: {num_frames} frames at {target_fps} fps")

    # Assemble: [root_pos(3), root_rot(4), dof_pos(29)] = 36 columns
    csv_data = np.concatenate([root_pos, root_rot, dof_pos], axis=-1)
    assert csv_data.shape[1] == 36, f"Expected 36 columns, got {csv_data.shape[1]}"

    np.savetxt(output_path, csv_data, delimiter=",", fmt="%.8f")
    print(f"Saved: {output_path} ({num_frames} frames, 36 columns)")


def main():
    parser = argparse.ArgumentParser(description="Convert video2robot pkl to mimic CSV")
    parser.add_argument("--input", "-i", required=True, help="Path to video2robot pickle file")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument("--target_fps", type=int, default=None, help="Resample to target FPS (default: keep original)")
    args = parser.parse_args()

    convert_pkl_to_csv(args.input, args.output, args.target_fps)


if __name__ == "__main__":
    main()
