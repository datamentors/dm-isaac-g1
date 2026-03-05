"""Motion CSV loader for mimic sim2sim.

Matches the C++ MotionLoader_ in State_Mimic.h:
- Loads CSV with format: [root_pos(3), root_quat_xyzw(4), joint_pos(N)] per row
- Computes joint velocities as finite differences
- Interpolates between frames
- Provides joint_pos(), joint_vel(), root_quaternion() at any time
"""

import csv

import numpy as np


class MotionLoader:
    """Load and interpolate motion reference data from CSV."""

    def __init__(self, motion_file, fps=30.0):
        self.dt = 1.0 / fps
        self._load(motion_file)
        self.update(0.0)

    def _load(self, motion_file):
        """Load CSV motion file."""
        path = str(motion_file)

        if path.endswith(".npz"):
            self._load_npz(path)
            return

        # CSV format: root_pos(3), root_quat_xyzw(4), joint_pos(N)
        rows = []
        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                rows.append([float(v) for v in row])

        if not rows:
            raise ValueError(f"Empty motion file: {path}")

        data = np.array(rows, dtype=np.float32)
        self.num_frames = len(data)
        self.duration = self.num_frames * self.dt

        self.root_positions = data[:, :3]
        # CSV has xyzw quaternion, convert to wxyz for internal use
        self.root_quaternions_wxyz = np.column_stack([
            data[:, 6], data[:, 3], data[:, 4], data[:, 5]
        ])
        self.dof_positions = data[:, 7:]
        self.dof_velocities = self._compute_derivative(self.dof_positions)

    def _load_npz(self, path):
        """Fallback: load NPZ motion file."""
        npz = np.load(path)
        # Try common key names
        motion_data = None
        for key in ["motion_command", "motion", "data", "commands"]:
            if key in npz:
                motion_data = npz[key]
                break
        if motion_data is None:
            keys = list(npz.keys())
            if keys:
                motion_data = npz[keys[0]]
        if motion_data is None:
            raise ValueError(f"No motion data found in {path}")

        self.num_frames = len(motion_data)
        self.duration = self.num_frames * self.dt
        # NPZ: assume raw motion_command format [joint_pos, joint_vel]
        n_dof = motion_data.shape[1] // 2 if motion_data.ndim == 2 else 29
        self.root_positions = np.zeros((self.num_frames, 3), dtype=np.float32)
        self.root_quaternions_wxyz = np.tile([1, 0, 0, 0], (self.num_frames, 1)).astype(np.float32)
        if motion_data.ndim == 2 and motion_data.shape[1] >= 2 * n_dof:
            self.dof_positions = motion_data[:, :n_dof].astype(np.float32)
            self.dof_velocities = motion_data[:, n_dof:2*n_dof].astype(np.float32)
        else:
            self.dof_positions = motion_data.astype(np.float32)
            self.dof_velocities = self._compute_derivative(self.dof_positions)

    def _compute_derivative(self, data):
        """Finite difference velocity (matches C++ _comupte_raw_derivative)."""
        vel = np.zeros_like(data)
        if len(data) > 1:
            vel[:-1] = (data[1:] - data[:-1]) / self.dt
            vel[-1] = vel[-2]
        return vel

    def update(self, time):
        """Set current time for interpolation."""
        phase = np.clip(time / max(self.duration, 1e-6), 0.0, 1.0)
        self._index_0 = int(round(phase * (self.num_frames - 1)))
        self._index_1 = min(self._index_0 + 1, self.num_frames - 1)
        self._blend = round((time - self._index_0 * self.dt) / self.dt * 1e5) / 1e5

    def joint_pos(self):
        """Interpolated joint positions at current time."""
        return (self.dof_positions[self._index_0] * (1 - self._blend) +
                self.dof_positions[self._index_1] * self._blend)

    def joint_vel(self):
        """Interpolated joint velocities at current time."""
        return (self.dof_velocities[self._index_0] * (1 - self._blend) +
                self.dof_velocities[self._index_1] * self._blend)

    def root_position(self):
        """Interpolated root position at current time."""
        return (self.root_positions[self._index_0] * (1 - self._blend) +
                self.root_positions[self._index_1] * self._blend)

    def root_quaternion_wxyz(self):
        """Interpolated root quaternion (wxyz) at current time via slerp."""
        q0 = self.root_quaternions_wxyz[self._index_0]
        q1 = self.root_quaternions_wxyz[self._index_1]
        return _slerp(q0, q1, self._blend)


def _slerp(q0, q1, t):
    """Spherical linear interpolation between quaternions (wxyz format)."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1
