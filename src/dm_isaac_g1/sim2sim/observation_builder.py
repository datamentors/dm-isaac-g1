"""Data-driven observation builder for sim2sim.

Reads the observation spec from deploy.yaml and constructs the observation
vector by dispatching to per-term builder functions. Supports history
stacking and per-term scaling.

Observation terms:
    base_ang_vel        - Angular velocity in body frame (3)
    projected_gravity   - Gravity vector in body frame (3)
    velocity_commands   - [vx, vy, wz] commands (3)
    joint_pos_rel       - Joint positions relative to default (N)
    joint_vel_rel       - Joint velocities (N)
    last_action         - Previous policy output (N)
    gait_phase          - Sinusoidal gait clock signal (2)
    motion_command      - Motion reference for mimic policies (58)
    motion_anchor_ori_b - Anchor body orientation for mimic (6)
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SimState:
    """Mutable state for sim2sim simulation loop."""

    cmd_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    last_actions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    obs_history: deque = field(default_factory=deque)
    global_phase: float = 0.0
    step_dt: float = 0.02
    step_count: int = 0
    sim_time: float = 0.0
    fsm_state: str = "stand"
    n_joints: int = 29
    motion_data: Optional[np.ndarray] = None
    motion_loader: object = None  # MotionLoader instance for mimic
    init_quat: Optional[np.ndarray] = None  # init_quat for anchor_ori correction
    deploy_cfg: Optional[dict] = None
    max_history: int = 1
    _term_histories: Optional[dict] = None


def _quat_rotate_inverse(quat_wxyz, vec):
    """Rotate a vector by the inverse of a quaternion (wxyz format)."""
    w, x, y, z = quat_wxyz
    # Compute rotation matrix transpose (inverse rotation)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y + w * z)
    r02 = 2 * (x * z - w * y)
    r10 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z + w * x)
    r20 = 2 * (x * z + w * y)
    r21 = 2 * (y * z - w * x)
    r22 = 1 - 2 * (x * x + y * y)
    # R^T * vec  (inverse rotation = transpose for rotation matrices)
    return np.array([
        r00 * vec[0] + r10 * vec[1] + r20 * vec[2],
        r01 * vec[0] + r11 * vec[1] + r21 * vec[2],
        r02 * vec[0] + r12 * vec[1] + r22 * vec[2],
    ])


def _quat_multiply(q1, q2):
    """Multiply two quaternions (wxyz format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_conjugate(q):
    """Conjugate (inverse for unit quaternion) in wxyz format."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_from_axis_angle(axis, angle):
    """Create quaternion (wxyz) from axis and angle."""
    half = angle * 0.5
    s = np.sin(half)
    return np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s])


def _quat_chain_zxy(root_quat_wxyz, j_z, j_x, j_y):
    """Compute torso quaternion: root * Rz(j_z) * Rx(j_x) * Ry(j_y).

    Matches C++ torso_quat_w: root * AngleAxis(j12, Z) * AngleAxis(j13, X) * AngleAxis(j14, Y)
    """
    q = root_quat_wxyz.copy().astype(np.float64)
    q = _quat_multiply(q, _quat_from_axis_angle(np.array([0, 0, 1.0]), j_z))
    q = _quat_multiply(q, _quat_from_axis_angle(np.array([1, 0, 0.0]), j_x))
    q = _quat_multiply(q, _quat_from_axis_angle(np.array([0, 1, 0.0]), j_y))
    return q / np.linalg.norm(q)


def _quat_to_rotation_matrix(q_wxyz):
    """Convert quaternion (wxyz) to 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def _yaw_quaternion(q_wxyz):
    """Extract yaw-only quaternion (rotation around Z axis) from wxyz quaternion."""
    w, x, y, z = q_wxyz
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return _quat_from_axis_angle(np.array([0, 0, 1.0]), yaw)


# ── Per-term builder functions ──────────────────────────────────────────


def build_base_ang_vel(data, obs_cfg, state):
    """Angular velocity in body frame (3-dim)."""
    # MuJoCo qvel[3:6] is angular velocity in world frame.
    # Rotate into body frame using inverse of body quaternion.
    quat = data.qpos[3:7]  # MuJoCo: [w, x, y, z]
    ang_vel_world = data.qvel[3:6]
    return _quat_rotate_inverse(quat, ang_vel_world)


def build_projected_gravity(data, obs_cfg, state):
    """Gravity vector projected into body frame (3-dim).

    In world frame, gravity points down: [0, 0, -1] (normalized).
    We rotate this into the body frame.
    """
    quat = data.qpos[3:7]
    gravity_world = np.array([0.0, 0.0, -1.0])
    return _quat_rotate_inverse(quat, gravity_world)


def build_velocity_commands(data, obs_cfg, state):
    """Velocity commands [vx, vy, wz] (3-dim)."""
    return state.cmd_vel.copy()


def build_joint_pos_rel(data, obs_cfg, state):
    """Joint positions relative to default (N-dim)."""
    cfg = state.deploy_cfg
    n = state.n_joints
    default_pos = np.zeros(n)
    if "default_joint_pos" in cfg:
        dp = cfg["default_joint_pos"]
        default_pos[:min(len(dp), n)] = dp[:n]

    n_qpos = min(n, data.qpos.shape[0] - 7)
    raw_pos = data.qpos[7:7 + n_qpos]

    # Apply joint_ids_map: reorder from MuJoCo order to SDK/training order
    joint_ids = cfg.get("joint_ids_map")
    if joint_ids and len(joint_ids) >= n_qpos:
        # joint_ids_map maps training-order index -> MuJoCo index
        # We need: for each training-order slot i, read MuJoCo joint joint_ids[i]
        reordered = np.zeros(n)
        for i in range(min(n, len(joint_ids))):
            mj_idx = joint_ids[i]
            if mj_idx < n_qpos:
                reordered[i] = raw_pos[mj_idx] - default_pos[i]
        return reordered
    else:
        return raw_pos[:n] - default_pos[:n_qpos]


def build_joint_vel_rel(data, obs_cfg, state):
    """Joint velocities (N-dim)."""
    cfg = state.deploy_cfg
    n = state.n_joints
    n_qvel = min(n, data.qvel.shape[0] - 6)
    raw_vel = data.qvel[6:6 + n_qvel]

    joint_ids = cfg.get("joint_ids_map")
    if joint_ids and len(joint_ids) >= n_qvel:
        reordered = np.zeros(n)
        for i in range(min(n, len(joint_ids))):
            mj_idx = joint_ids[i]
            if mj_idx < n_qvel:
                reordered[i] = raw_vel[mj_idx]
        return reordered
    else:
        result = np.zeros(n)
        result[:n_qvel] = raw_vel
        return result


def build_last_action(data, obs_cfg, state):
    """Last policy actions (N-dim)."""
    n = state.n_joints
    if len(state.last_actions) >= n:
        return state.last_actions[:n].copy()
    result = np.zeros(n)
    result[:len(state.last_actions)] = state.last_actions
    return result


def build_gait_phase(data, obs_cfg, state):
    """Sinusoidal gait phase clock signal (2-dim).

    Generates [sin(phase), cos(phase)] where phase accumulates
    at rate 1/period each control step.
    """
    period = obs_cfg.get("params", {}).get("period", 0.8)
    delta_phase = state.step_dt / period
    state.global_phase = (state.global_phase + delta_phase) % 1.0
    phase = state.global_phase * 2.0 * np.pi
    return np.array([np.sin(phase), np.cos(phase)])


def build_motion_command(data, obs_cfg, state):
    """Motion reference command for mimic policies (58-dim for G1 29-DOF).

    Matches C++ State_Mimic.cpp: motion_command = [joint_pos_reordered, joint_vel_reordered]
    Both pos and vel are remapped from motion-file order to training order via joint_ids_map.
    """
    n = state.n_joints
    dim = len(obs_cfg.get("scale", []))
    if dim == 0:
        dim = 2 * n  # G1 29-DOF: 58

    loader = state.motion_loader
    if loader is not None:
        # Update loader to current sim time
        loader.update(state.sim_time)
        pos_dfs = loader.joint_pos()
        vel_dfs = loader.joint_vel()

        # Reorder via joint_ids_map (matches C++ ids remapping)
        cfg = state.deploy_cfg or {}
        joint_ids = cfg.get("joint_ids_map")
        if joint_ids and len(joint_ids) <= len(pos_dfs):
            pos_bfs = np.zeros(n, dtype=np.float32)
            vel_bfs = np.zeros(n, dtype=np.float32)
            for i in range(min(n, len(joint_ids))):
                mj_idx = joint_ids[i]
                if mj_idx < len(pos_dfs):
                    pos_bfs[i] = pos_dfs[mj_idx]
                if mj_idx < len(vel_dfs):
                    vel_bfs[i] = vel_dfs[mj_idx]
        else:
            pos_bfs = pos_dfs[:n].copy()
            vel_bfs = vel_dfs[:n].copy()

        result = np.concatenate([pos_bfs, vel_bfs])
        if len(result) >= dim:
            return result[:dim]
        padded = np.zeros(dim)
        padded[:len(result)] = result
        return padded

    # Fallback: raw motion_data array (legacy NPZ path)
    if state.motion_data is not None:
        idx = min(state.step_count, len(state.motion_data) - 1)
        ref = state.motion_data[idx]
        if len(ref) >= dim:
            return ref[:dim].copy()
        result = np.zeros(dim)
        result[:len(ref)] = ref
        return result

    return np.zeros(dim)


def build_motion_anchor_ori_b(data, obs_cfg, state):
    """Anchor body orientation relative to motion reference (6-dim).

    Matches C++ State_Mimic.cpp:
    - Computes torso quaternion via kinematic chain: root * Rz(motor12) * Rx(motor13) * Ry(motor14)
    - Computes anchor quaternion from motion reference the same way
    - Returns relative rotation as 6-dim (first two columns of rotation matrix transposed)
    """
    dim = len(obs_cfg.get("scale", []))
    if dim == 0:
        dim = 6

    # Robot torso quaternion via kinematic chain (joints 12,13,14 = waist ZXY)
    root_quat = data.qpos[3:7]  # wxyz
    n_qpos = data.qpos.shape[0] - 7
    if n_qpos > 14:
        j12 = data.qpos[7 + 12]  # waist yaw (Z axis)
        j13 = data.qpos[7 + 13]  # waist roll (X axis)
        j14 = data.qpos[7 + 14]  # waist pitch (Y axis)
    else:
        j12, j13, j14 = 0.0, 0.0, 0.0

    real_torso = _quat_chain_zxy(root_quat, j12, j13, j14)

    # Anchor (reference) torso quaternion from motion loader
    loader = state.motion_loader
    if loader is not None:
        ref_root_quat = loader.root_quaternion_wxyz()
        ref_joint_pos = loader.joint_pos()
        rj12 = ref_joint_pos[12] if len(ref_joint_pos) > 14 else 0.0
        rj13 = ref_joint_pos[13] if len(ref_joint_pos) > 14 else 0.0
        rj14 = ref_joint_pos[14] if len(ref_joint_pos) > 14 else 0.0
        ref_torso = _quat_chain_zxy(ref_root_quat, rj12, rj13, rj14)

        # rot_ = (init_quat * ref_torso)^-1 * real_torso
        init_quat = state.init_quat if state.init_quat is not None else np.array([1, 0, 0, 0], dtype=np.float64)
        combined = _quat_multiply(init_quat, ref_torso)
        rot_q = _quat_multiply(_quat_conjugate(combined), real_torso)
    else:
        # No motion reference: just use identity-relative orientation
        rot_q = real_torso

    # Convert quaternion to rotation matrix, take transpose, extract first 2 columns as 6-dim
    rot = _quat_to_rotation_matrix(rot_q).T
    result = np.array([rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1], rot[2, 0], rot[2, 1]])
    return result[:dim]


# ── Term registry ──────────────────────────────────────────────────────

OBS_TERM_BUILDERS = {
    "base_ang_vel": build_base_ang_vel,
    "projected_gravity": build_projected_gravity,
    "velocity_commands": build_velocity_commands,
    "joint_pos_rel": build_joint_pos_rel,
    "joint_vel_rel": build_joint_vel_rel,
    "last_action": build_last_action,
    "gait_phase": build_gait_phase,
    "motion_command": build_motion_command,
    "motion_anchor_ori_b": build_motion_anchor_ori_b,
}


# ── Main observation builder ──────────────────────────────────────────


def build_observation(data, deploy_cfg, state, obs_dim, debug=False):
    """Build the full observation vector from deploy.yaml spec.

    Iterates observation terms in deploy_cfg["observations"] order,
    applies per-term scaling, and stacks history frames if needed.

    Args:
        data: MuJoCo MjData
        deploy_cfg: dict from deploy.yaml
        state: SimState with mutable sim state
        obs_dim: expected observation dimension from policy
        debug: if True, print per-term shapes

    Returns:
        np.ndarray of shape (obs_dim,), dtype float32
    """
    obs_spec = deploy_cfg.get("observations", {})

    if not obs_spec:
        # Fallback: no observation spec, use legacy builder
        return _build_observation_legacy(data, deploy_cfg, state, obs_dim)

    obs_parts = []
    for obs_name, obs_cfg in obs_spec.items():
        builder = OBS_TERM_BUILDERS.get(obs_name)
        if builder is None:
            print(f"[sim2sim] WARNING: Unknown observation term '{obs_name}', "
                  f"filling with zeros (dim={len(obs_cfg.get('scale', []))})")
            dim = len(obs_cfg.get("scale", []))
            obs_parts.append(np.zeros(dim))
            continue

        raw = builder(data, obs_cfg, state)
        scale = np.array(obs_cfg.get("scale", [1.0] * len(raw)))
        clip_val = obs_cfg.get("clip")
        scale_first = obs_spec.get("scale_first", False)

        # Handle scalar scale
        if len(scale) == 1 and len(raw) > 1:
            scale = np.full(len(raw), scale[0])
        elif len(scale) < len(raw):
            scale = np.pad(scale, (0, len(raw) - len(scale)), constant_values=1.0)

        # Apply scale and clip in correct order (matches C++ manager_term_cfg.h)
        result = raw.copy()
        if scale_first:
            result = result * scale[:len(result)]
            if clip_val is not None and len(clip_val) == 2:
                result = np.clip(result, clip_val[0], clip_val[1])
        else:
            if clip_val is not None and len(clip_val) == 2:
                result = np.clip(result, clip_val[0], clip_val[1])
            result = result * scale[:len(result)]

        if debug and state.step_count == 0:
            print(f"  obs.{obs_name}: shape={raw.shape}, "
                  f"range=[{raw.min():.3f}, {raw.max():.3f}]")

        obs_parts.append(result)

    # History stacking — per-term buffers (matches C++ ObservationTermCfg)
    # Two modes:
    #   use_gym_history=True:  interleave by timestep [all_t0, all_t1, ..., all_tN]
    #   use_gym_history=False: per-term concat [term0_all_history, term1_all_history, ...]
    # Default is False (matching C++ observation_manager.h)
    use_gym_history = obs_spec.get("use_gym_history", False)

    # Initialize per-term history buffers on first call
    if not hasattr(state, '_term_histories') or state._term_histories is None:
        state._term_histories = {}

    term_names = [n for n in obs_spec.keys() if n not in ("use_gym_history", "scale_first")]
    for i, obs_name in enumerate(term_names):
        obs_cfg = obs_spec[obs_name]
        hist_len = obs_cfg.get("history_length", 1)
        if obs_name not in state._term_histories:
            state._term_histories[obs_name] = deque(maxlen=hist_len)
        buf = state._term_histories[obs_name]
        buf.append(obs_parts[i].copy())
        # Pad with zeros if not enough history yet
        while len(buf) < hist_len:
            buf.appendleft(np.zeros_like(obs_parts[i]))

    max_history = max(
        (obs_spec[n].get("history_length", 1) for n in term_names),
        default=1,
    )
    state.max_history = max_history

    if use_gym_history and max_history > 1:
        # Gym-style: interleave by timestep
        # [all_terms_at_t0, all_terms_at_t1, ..., all_terms_at_tN]
        obs_list = []
        for h in range(max_history):
            for obs_name in term_names:
                buf = state._term_histories[obs_name]
                obs_list.append(buf[h])
        obs = np.concatenate(obs_list)
    else:
        # Default: per-term history concatenation
        # [term0_h0..hN, term1_h0..hN, ...]
        obs_list = []
        for obs_name in term_names:
            buf = state._term_histories[obs_name]
            for frame in buf:
                obs_list.append(frame)
        obs = np.concatenate(obs_list)

    obs = obs.astype(np.float32)

    # Validate dimension
    if len(obs) != obs_dim:
        if state.step_count == 0:
            single_frame = sum(len(p) for p in obs_parts)
            print(f"[sim2sim] WARNING: Observation dim mismatch: built {len(obs)}, "
                  f"policy expects {obs_dim}. "
                  f"Frame={single_frame}, history={max_history}, "
                  f"expected={single_frame}*{max_history}={single_frame*max_history}")
        # Pad or trim
        if len(obs) < obs_dim:
            obs = np.pad(obs, (0, obs_dim - len(obs)))
        else:
            obs = obs[:obs_dim]

    return obs


def _build_observation_legacy(data, deploy_cfg, state, obs_dim):
    """Legacy observation builder for configs without observation spec."""
    n = state.n_joints
    base_ang_vel = build_base_ang_vel(data, {}, state)
    cmd = state.cmd_vel.copy()

    default_pos = np.zeros(n)
    if "default_joint_pos" in deploy_cfg:
        dp = deploy_cfg["default_joint_pos"]
        default_pos[:min(len(dp), n)] = dp[:n]

    n_qpos = min(n, data.qpos.shape[0] - 7)
    n_qvel = min(n, data.qvel.shape[0] - 6)
    joint_pos_rel = data.qpos[7:7 + n_qpos] - default_pos[:n_qpos]
    joint_vel = data.qvel[6:6 + n_qvel]

    obs_parts = [
        base_ang_vel * 0.25,
        cmd * np.array([2.0, 2.0, 0.25]),
        joint_pos_rel,
        joint_vel * 0.05,
    ]
    if len(state.last_actions) > 0:
        obs_parts.append(state.last_actions[:n])

    obs = np.concatenate(obs_parts).astype(np.float32)
    if len(obs) < obs_dim:
        obs = np.pad(obs, (0, obs_dim - len(obs)))
    elif len(obs) > obs_dim:
        obs = obs[:obs_dim]
    return obs
