"""Main sim2sim runner: loads policy, builds MuJoCo sim, runs control loop.

Supports:
  - Headless mode (EGL) with video recording
  - GUI mode with optional keyboard control and FSM
  - Both RL (velocity) and Mimic (motion reference) policies
  - Data-driven observations from deploy.yaml
"""

import os
import sys
import time

import numpy as np

from dm_isaac_g1.sim2sim.deploy_config import (
    auto_detect_deploy_yaml,
    find_mujoco_scene,
    get_default_g1_29dof_deploy_cfg,
    load_deploy_config,
)
from dm_isaac_g1.sim2sim.observation_builder import SimState, build_observation, _quat_rotate_inverse
from dm_isaac_g1.sim2sim.policy_loader import load_policy


def _bad_orientation(data, threshold=1.0):
    """Check if robot has fallen (projected gravity Z component too low).

    Matches C++ isaaclab::mdp::bad_orientation: checks if the Z component of
    gravity projected into body frame exceeds threshold (i.e., robot is tilted).
    For threshold=1.0, triggers when robot is horizontal or inverted.
    """
    quat = data.qpos[3:7]  # wxyz
    gravity_body = _quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
    # If gravity Z in body frame is positive (robot upside down) or very tilted
    return gravity_body[2] > -0.1  # roughly > 84 degrees tilt


def _compute_target_pos(actions, deploy_cfg, n_joints):
    """Convert policy actions to target joint positions using action config."""
    n_act = min(len(actions), n_joints)
    action_scale = 0.25
    action_offset = np.zeros(n_joints)

    if "actions" in deploy_cfg:
        for act_cfg in deploy_cfg["actions"].values():
            if "scale" in act_cfg and act_cfg["scale"]:
                s = act_cfg["scale"]
                if isinstance(s, list):
                    action_scale = np.array(s[:n_act])
                elif isinstance(s, (int, float)):
                    action_scale = float(s)
            if "offset" in act_cfg and act_cfg["offset"]:
                o = act_cfg["offset"]
                if isinstance(o, list):
                    action_offset[:min(len(o), n_joints)] = o[:n_joints]

    scaled = actions[:n_act] * action_scale
    if isinstance(action_offset, np.ndarray):
        target = scaled + action_offset[:n_act]
    else:
        target = scaled + action_offset
    return target


def _apply_torque(data, torque, n_joints):
    """Apply torques to MuJoCo actuators (clipped to actuator limits)."""
    n_act = min(len(torque), n_joints, data.ctrl.shape[0])
    data.ctrl[:n_act] = np.clip(
        torque[:n_act],
        data.model.actuator_ctrlrange[:n_act, 0],
        data.model.actuator_ctrlrange[:n_act, 1],
    )


def run(
    policy_path,
    deploy_yaml=None,
    scene=None,
    headless=False,
    video=False,
    video_length=10.0,
    video_fps=30,
    output_dir=None,
    sim_duration=60.0,
    cmd_vx=0.5,
    cmd_vy=0.0,
    cmd_wz=0.0,
    interactive=False,
    motion_file=None,
    debug_obs=False,
):
    """Run sim2sim: load policy, simulate in MuJoCo, optionally record video.

    Args:
        policy_path: path to .onnx or .pt policy
        deploy_yaml: path to deploy.yaml (auto-detected if None)
        scene: path to MuJoCo scene XML (auto-detected if None)
        headless: use EGL rendering (no GUI)
        video: record video
        video_length: video duration in seconds
        video_fps: video frames per second
        output_dir: output directory for videos
        sim_duration: GUI mode sim duration
        cmd_vx, cmd_vy, cmd_wz: initial velocity commands
        interactive: enable keyboard control in GUI mode
        motion_file: path to motion NPZ for mimic policies
        debug_obs: print observation debug info
    """
    # Set rendering backend before importing mujoco
    if headless:
        os.environ["MUJOCO_GL"] = "egl"

    import mujoco

    # Load policy
    print(f"[sim2sim] Loading policy: {policy_path}")
    infer_fn, obs_dim = load_policy(policy_path)

    # Load deploy config
    if deploy_yaml is None:
        deploy_yaml = auto_detect_deploy_yaml(policy_path)

    if deploy_yaml:
        print(f"[sim2sim] Deploy config: {deploy_yaml}")
        deploy_cfg = load_deploy_config(deploy_yaml)
    else:
        print("[sim2sim] No deploy.yaml found, using G1 29-DOF defaults")
        deploy_cfg = get_default_g1_29dof_deploy_cfg()

    # Find scene
    scene_path = scene or find_mujoco_scene()
    if not scene_path or not os.path.exists(scene_path):
        print("[sim2sim] ERROR: No G1 MuJoCo scene XML found.")
        print("[sim2sim] Specify --scene or clone unitree_mujoco:")
        print("[sim2sim]   git clone https://github.com/unitreerobotics/unitree_mujoco.git")
        sys.exit(1)

    print(f"[sim2sim] Scene: {scene_path}")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    n_joints = model.nu
    dt = model.opt.timestep
    print(f"[sim2sim] Model: {n_joints} actuators, dt={dt:.4f}s")

    # Control timing
    step_dt = deploy_cfg.get("step_dt", 0.02)
    control_decimation = max(1, int(round(step_dt / dt)))
    print(f"[sim2sim] Control: step_dt={step_dt:.4f}s, decimation={control_decimation}")

    # Output
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(policy_path))
    os.makedirs(output_dir, exist_ok=True)

    # Initialize state
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    state = SimState(
        cmd_vel=np.array([cmd_vx, cmd_vy, cmd_wz], dtype=np.float32),
        last_actions=np.zeros(n_joints, dtype=np.float32),
        step_dt=step_dt,
        n_joints=n_joints,
        deploy_cfg=deploy_cfg,
    )

    # Load motion data for mimic
    if motion_file:
        from dm_isaac_g1.sim2sim.motion_loader import MotionLoader
        fps = deploy_cfg.get("fps", 30.0)
        try:
            state.motion_loader = MotionLoader(motion_file, fps=fps)
            print(f"[sim2sim] Loaded motion: {motion_file} "
                  f"({state.motion_loader.num_frames} frames, "
                  f"{state.motion_loader.duration:.2f}s, "
                  f"{state.motion_loader.dof_positions.shape[1]} DOF)")
        except Exception as e:
            print(f"[sim2sim] WARNING: Failed to load motion file: {e}")
            print(f"[sim2sim] Trying NPZ fallback...")
            motion_npz = np.load(motion_file)
            for key in ["motion_command", "motion", "data", "commands"]:
                if key in motion_npz:
                    state.motion_data = motion_npz[key]
                    print(f"[sim2sim] Loaded motion data: {key} shape={state.motion_data.shape}")
                    break

    # Print obs info
    obs_spec = deploy_cfg.get("observations", {})
    if obs_spec:
        frame_dim = sum(len(cfg.get("scale", [])) for cfg in obs_spec.values())
        max_hist = max((cfg.get("history_length", 1) for cfg in obs_spec.values()), default=1)
        print(f"[sim2sim] Observations: {len(obs_spec)} terms, "
              f"frame_dim={frame_dim}, history={max_hist}, "
              f"total={frame_dim * max_hist}, policy_expects={obs_dim}")

    # Velocity ranges for info
    cmd_ranges = deploy_cfg.get("commands", {}).get("base_velocity", {}).get("ranges", {})
    if cmd_ranges:
        print(f"[sim2sim] Velocity ranges: "
              f"vx={cmd_ranges.get('lin_vel_x')}, "
              f"vy={cmd_ranges.get('lin_vel_y')}, "
              f"wz={cmd_ranges.get('ang_vel_z')}")

    print(f"[sim2sim] Initial cmd_vel: vx={cmd_vx:.2f}, vy={cmd_vy:.2f}, wz={cmd_wz:.2f}")

    # Import FSM
    from dm_isaac_g1.sim2sim.fsm import RobotFSM

    fsm = RobotFSM(deploy_cfg, n_joints)

    if headless or video:
        _run_headless(
            model, data, infer_fn, deploy_cfg, state, fsm,
            obs_dim=obs_dim,
            video=video,
            video_length=video_length if video else sim_duration,
            video_fps=video_fps,
            output_dir=output_dir,
            step_dt=step_dt,
            control_decimation=control_decimation,
            n_joints=n_joints,
            debug_obs=debug_obs,
        )
    else:
        _run_gui(
            model, data, infer_fn, deploy_cfg, state, fsm,
            obs_dim=obs_dim,
            sim_duration=sim_duration,
            step_dt=step_dt,
            control_decimation=control_decimation,
            n_joints=n_joints,
            interactive=interactive,
            cmd_ranges=cmd_ranges,
            debug_obs=debug_obs,
        )


def _run_headless(
    model, data, infer_fn, deploy_cfg, state, fsm,
    obs_dim, video, video_length, video_fps, output_dir,
    step_dt, control_decimation, n_joints, debug_obs,
):
    """Headless simulation with optional video recording."""
    import mujoco

    renderer = mujoco.Renderer(model, height=720, width=1280)
    total_steps = int(video_length / step_dt)
    target_frames = int(video_length * video_fps)
    frame_interval = max(1, total_steps // target_frames)

    frames = []
    print(f"[sim2sim] Running {video_length:.1f}s ({total_steps} steps)...")

    # Initialize mimic init_quat (matches C++ State_Mimic::enter)
    if state.motion_loader is not None:
        from dm_isaac_g1.sim2sim.observation_builder import _yaw_quaternion, _quat_multiply, _quat_to_rotation_matrix
        state.motion_loader.update(0.0)
        ref_yaw = _yaw_quaternion(state.motion_loader.root_quaternion_wxyz())
        robot_yaw = _yaw_quaternion(data.qpos[3:7])
        ref_rot = _quat_to_rotation_matrix(ref_yaw)
        robot_rot = _quat_to_rotation_matrix(robot_yaw)
        init_rot = robot_rot @ ref_rot.T
        # Convert rotation matrix back to quaternion for init_quat
        from dm_isaac_g1.sim2sim.observation_builder import _quat_from_axis_angle
        # Use simplified: init_quat = robot_yaw * ref_yaw^-1
        from dm_isaac_g1.sim2sim.observation_builder import _quat_conjugate
        state.init_quat = _quat_multiply(robot_yaw, _quat_conjugate(ref_yaw))

    # Start in WALK mode for headless
    fsm.transition_to("walk", data, 0.0)

    for step in range(total_steps):
        state.step_count = step
        state.sim_time = step * step_dt

        # Build observation and infer
        obs = build_observation(data, deploy_cfg, state, obs_dim, debug=debug_obs and step == 0)
        actions = infer_fn(obs)
        state.last_actions = actions.copy()

        # Compute target positions and torques
        target_pos = _compute_target_pos(actions, deploy_cfg, n_joints)
        torque = fsm.compute_torque(data, target_pos, state.sim_time)
        _apply_torque(data, torque, n_joints)

        # Step physics
        for _ in range(control_decimation):
            mujoco.mj_step(model, data)

        # Capture frame
        if video and (step % frame_interval) == 0:
            renderer.update_scene(data)
            frame = renderer.render()
            frames.append(frame.copy())

        # Check bad orientation (matches C++ bad_orientation threshold=1.0)
        if _bad_orientation(data):
            print(f"[sim2sim] Bad orientation detected at t={step * step_dt:.2f}s, stopping.")
            break

        # Progress
        progress_interval = max(1, total_steps // 10)
        if (step + 1) % progress_interval == 0:
            elapsed = step * step_dt
            print(f"  [{elapsed:.1f}s / {video_length:.1f}s] step {step + 1}/{total_steps}")

    # Save video
    if video and frames:
        try:
            import cv2
            video_path = os.path.join(output_dir, "sim2sim.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(video_path, fourcc, video_fps, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"[sim2sim] Video saved: {video_path} ({len(frames)} frames)")
        except ImportError:
            print("[sim2sim] WARNING: opencv not available, saving frames as images")
            frames_dir = os.path.join(output_dir, "sim2sim_frames")
            os.makedirs(frames_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                from PIL import Image
                Image.fromarray(frame).save(os.path.join(frames_dir, f"frame_{i:05d}.png"))
            print(f"[sim2sim] Frames saved to: {frames_dir}")

    renderer.close()
    print("[sim2sim] Headless simulation complete.")


def _run_gui(
    model, data, infer_fn, deploy_cfg, state, fsm,
    obs_dim, sim_duration, step_dt, control_decimation, n_joints,
    interactive, cmd_ranges, debug_obs,
):
    """Interactive GUI simulation with optional keyboard control."""
    import mujoco
    import mujoco.viewer

    kb = None
    if interactive:
        from dm_isaac_g1.sim2sim.keyboard_controller import KeyboardController
        max_vx = 1.0
        max_vy = 0.5
        max_wz = 0.5
        if cmd_ranges:
            vx_range = cmd_ranges.get("lin_vel_x", [-1.0, 1.0])
            vy_range = cmd_ranges.get("lin_vel_y", [-0.5, 0.5])
            wz_range = cmd_ranges.get("ang_vel_z", [-0.5, 0.5])
            if vx_range:
                max_vx = max(abs(vx_range[0]), abs(vx_range[1]))
            if vy_range:
                max_vy = max(abs(vy_range[0]), abs(vy_range[1]))
            if wz_range:
                max_wz = max(abs(wz_range[0]), abs(wz_range[1]))

        kb = KeyboardController(vel_step=0.1, max_vx=max_vx, max_vy=max_vy, max_wz=max_wz)
        kb.cmd_vel[:] = state.cmd_vel

    def key_callback(keycode):
        if kb is not None:
            kb.process_key(keycode)

    if interactive:
        print("[sim2sim] Interactive mode: WASD=move, QE=turn, Space=stop, 1/2/3=FSM, R=reset")

    print("[sim2sim] Starting MuJoCo viewer... Press Ctrl+C to stop.")

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

    # Start in STAND, user presses 3 to WALK (or auto-walk if non-interactive)
    if not interactive:
        fsm.transition_to("walk", data, 0.0)

    step_count = 0
    sim_time = 0.0

    try:
        while viewer.is_running() and sim_time < sim_duration:
            state.step_count = step_count
            state.sim_time = sim_time

            # Handle keyboard input
            if kb is not None:
                state.cmd_vel[:] = kb.get_cmd_vel()

                # FSM transitions
                fsm_req = kb.consume_fsm_transition()
                if fsm_req:
                    fsm.transition_to(fsm_req, data, sim_time)

                # Reset
                if kb.consume_reset():
                    import mujoco as mj
                    mj.mj_resetData(model, data)
                    mj.mj_forward(model, data)
                    state.last_actions[:] = 0
                    state.obs_history.clear()
                    state.global_phase = 0.0
                    fsm.transition_to("stand", data, sim_time)
                    print("[sim2sim] Robot reset.")

            # Control
            if fsm.policy_active:
                obs = build_observation(
                    data, deploy_cfg, state, obs_dim,
                    debug=debug_obs and step_count == 0,
                )
                actions = infer_fn(obs)
                state.last_actions = actions.copy()
                target_pos = _compute_target_pos(actions, deploy_cfg, n_joints)
            else:
                target_pos = None

            torque = fsm.compute_torque(data, target_pos, sim_time)
            _apply_torque(data, torque, n_joints)

            # Step physics
            for _ in range(control_decimation):
                mujoco.mj_step(model, data)

            viewer.sync()
            sim_time += step_dt
            step_count += 1

            # Real-time sync
            time.sleep(max(0, step_dt - 0.001))

            # Status
            if kb is not None and step_count % 50 == 0:
                print(f"\r{kb.status_line(fsm.state.value, sim_time, step_count)}",
                      end="", flush=True)
            elif step_count % 500 == 0:
                print(f"  [sim2sim] t={sim_time:.1f}s, steps={step_count}")

    except KeyboardInterrupt:
        print("\n[sim2sim] Stopped by user.")
    finally:
        viewer.close()

    print(f"\n[sim2sim] Session ended: {step_count} steps ({sim_time:.1f}s)")
