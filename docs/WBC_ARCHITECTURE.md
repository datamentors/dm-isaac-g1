# GR00T Whole-Body Control (WBC) Architecture Reference

Reference documentation for the NVIDIA GR00T-WholeBodyControl pipeline, based on analysis of the [official repo](https://github.com/NVlabs/GR00T-WholeBodyControl).

## System Overview

The WBC system runs as **3 independent processes** communicating via ROS2 topics (ByteMultiArray + msgpack) and DDS topics (Unitree SDK2).

```
┌──────────────────────────────────────────────────────────┐
│  BRIDGE (external integration)                            │
│  ├─ Subscribe G1Env/env_state_act  ◄── FROM WBC CTRL     │
│  ├─ Read camera stream (ZMQ PUB/SUB)                      │
│  ├─ Call GR00T policy server (ZMQ REQ/REP, torch serial.) │
│  ├─ Map model output → WBC goal                          │
│  └─ Publish ControlPolicy/upper_body_pose + navigate_cmd  │
└──────────────────────┬────────────────────────────────────┘
                       │ (ROS2 ByteMultiArray, msgpack)
┌──────────────────────▼────────────────────────────────────┐
│  WBC Control Loop (run_g1_control_loop.py) @ 50 Hz        │
│  ├─ Subscribe ControlPolicy/upper_body_pose               │
│  ├─ G1DecoupledWholeBodyPolicy                            │
│  │   ├─ InterpolationPolicy (upper: arms+waist+hands)     │
│  │   └─ G1GearWbcPolicy (lower: legs via ONNX, 2 models) │
│  ├─ G1Env.queue_action()  ──► SIM (DDS: rt/lowcmd,       │
│  │   ├─ BodyCommandSender      rt/dex3/{l,r}/cmd)        │
│  │   └─ HandCommandSender                                 │
│  ├─ G1Env.observe()  ◄──── SIM (DDS: rt/lowstate,        │
│  │                              rt/dex3/{l,r}/state,      │
│  │                              rt/secondary_imu,         │
│  │                              rt/odostate [sim only])   │
│  └─ Publish G1Env/env_state_act  ──► TO BRIDGE            │
└───────────────────────────────────────────────────────────┘

GR00T Server (separate process, ZMQ REP on port 5555)
  └─ Endpoints: set_observation, get_action, get_modality_config
```

**Key insight**: The sim ↔ WBC controller communication is **bidirectional via DDS**. The bridge ONLY talks to the WBC controller via ROS2 topics — never directly to the sim.

## Entry Points

| Script | Purpose | Location |
|--------|---------|----------|
| `run_g1_control_loop.py` | Main WBC control loop (50 Hz) | `decoupled_wbc/control/main/teleop/` |
| `run_teleop_policy_loop.py` | Teleop input → publishes upper_body_pose | same |
| `run_sim_loop.py` | Standalone MuJoCo sim process | same |
| `run_g1_data_exporter.py` | Subscribes env_state_act → writes LeRobot datasets | same |
| `run_navigation_policy_loop.py` | Keyboard WASD → publishes navigate_cmd | same |
| `run_sync_sim_data_collection.py` | All-in-one synchronous sim data collection | same |

## Communication Topics

### ROS2 Topics (ByteMultiArray, msgpack-encoded dicts)

| Topic | Direction | Content |
|-------|-----------|---------|
| `ControlPolicy/upper_body_pose` | Bridge/Teleop → WBC | `target_upper_body_pose[17]`, `target_time`, `navigate_cmd[3]`, `base_height_command`, `wrist_pose[14]` |
| `G1Env/env_state_act` | WBC → Bridge/Exporter | `q[43]`, `dq[43]`, `wrist_pose[14]`, `action[43]`, `navigate_command[3]`, `floating_base_pose[7]`, timestamps |
| `ControlPolicy/lower_body_policy_status` | WBC → Monitor | `use_policy_action: bool` |
| `ControlPolicy/joint_safety_status` | WBC → Monitor | `joint_safety_ok: bool` |
| `WBCPolicy/robot_config` | WBC (service) → Exporter | Config dict via ROS2 Trigger service |

### DDS Topics (Unitree SDK2, direct struct serialization)

| Topic | Direction | IDL Type | Content |
|-------|-----------|----------|---------|
| `rt/lowstate` | Sim → WBC | `LowState_hg` | 29 motor states (q/dq/ddq/tau), pelvis IMU |
| `rt/lowcmd` | WBC → Sim | `LowCmd_hg` | 29 motor commands (q/dq/tau/kp/kd) + CRC |
| `rt/dex3/left/state` | Sim → WBC | `HandState_` | 7 hand motor states |
| `rt/dex3/right/state` | Sim → WBC | `HandState_` | 7 hand motor states |
| `rt/dex3/left/cmd` | WBC → Sim | `HandCmd_` | 7 hand motor commands |
| `rt/dex3/right/cmd` | WBC → Sim | `HandCmd_` | 7 hand motor commands |
| `rt/secondary_imu` | Sim → WBC | `IMUState_` | Torso IMU quaternion + gyroscope |
| `rt/odostate` | Sim → WBC | `OdoState_` | Floating base pos/vel (sim only) |

### ZMQ (model inference)

| Pattern | Port | Serialization | Endpoints |
|---------|------|--------------|-----------|
| REQ/REP | 5555 | `torch.save`/`torch.load` | `set_observation`, `get_action`, `get_modality_config`, `ping`, `kill` |
| PUB/SUB | configurable | msgpack | Camera image streaming (`SensorServer`/`SensorClient`) |

## WBC Control Loop — Step-by-Step (50 Hz)

```python
while ros_manager.ok():
    # 1. [sim only] Advance physics (4 substeps at 200Hz)
    env.step_simulator()

    # 2. Read robot state from DDS
    obs = env.observe()
    #   G1Body: rt/lowstate → body_q[29], body_dq[29], floating_base_pose[7]
    #   G1ThreeFingerHand: rt/dex3/*/state → hand_q[7], hand_dq[7]
    #   RobotModel.get_configuration_from_actuated_joints() → whole_q[43]
    wbc_policy.set_observation(obs)

    # 3. Read goal from bridge/teleop
    upper_body_cmd = subscriber.get_msg()  # "ControlPolicy/upper_body_pose"
    if upper_body_cmd:
        wbc_policy.set_goal(upper_body_cmd)

    # 4. Compute action
    wbc_action = wbc_policy.get_action(time=t_now)  # → {"q": np.array(43)}

    # 5. Send commands to robot/sim via DDS
    env.queue_action(wbc_action)
    #   JointSafetyMonitor → startup ramp, velocity/position checks
    #   BodyCommandSender → rt/lowcmd
    #   HandCommandSender → rt/dex3/*/cmd

    # 6. Publish state for bridge/exporter
    data_exp_pub.publish(msg)  # → "G1Env/env_state_act"

    rate.sleep()  # maintain 50 Hz
```

## G1DecoupledWholeBodyPolicy — Action Assembly

```
get_action(time):
  1. Safety timeout: if teleop_mode && >1s since last goal → stop

  2. Upper body (InterpolationPolicy):
     scipy interp1d → target_upper_body_pose[17] (3 waist + 14 arms)
     + base_height_command (0.74m default)
     + navigate_cmd[3]

  3. FK: compute torso RPY relative to waist

  4. Lower body (G1GearWbcPolicy):
     86-dim obs × 6 history = 516-dim → ONNX inference
     policy_1 (||cmd||<0.05 → balance) or policy_2 (walk)
     → 15-dim action (12 legs + 3 waist)

  5. Assemble q[43]:
     q[upper_body_indices] = interpolated upper body (arms+waist+hands)
     q[lower_body_indices] = RL lower body (legs+waist)
     # lower body overwrites waist when both claim it

  return {"q": q}  # 43-DOF
```

## Joint Group Hierarchy (G1 + Dex3 = 43 DOF)

```
body (29 DOF)
  ├── lower_body (15)
  │     ├── legs (12)
  │     │     ├── left_leg (6): hip_pitch/roll/yaw, knee, ankle_pitch/roll
  │     │     └── right_leg (6): same
  │     └── waist (3): yaw, roll, pitch
  └── upper_body_no_hands (14)
        └── arms (14)
              ├── left_arm (7): shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
              └── right_arm (7): same

upper_body (28 DOF)
  ├── upper_body_no_hands (14)
  └── hands (14)
        ├── left_hand (7): thumb_0/1/2, index_0/1, middle_0/1
        └── right_hand (7): same
```

Waist placement is configurable:
- `LOWER_BODY` (default): waist in lower_body only
- `UPPER_BODY`: waist moves to upper_body_no_hands (IK controls it)
- `LOWER_AND_UPPER_BODY`: waist in both groups

## Three Joint Orderings

1. **Pinocchio DOF order**: URDF joint order, used for kinematics
2. **GR00T actuated order**: `supplemental_info.body_actuated_joints` + hand joints — DDS wire format
3. **RoboCasa/Robosuite order**: internal Robosuite `_ref_joint_indexes`

Key conversion functions:
- `RobotModel.get_configuration_from_actuated_joints()`: actuated → Pinocchio
- `RobotModel.get_body_actuated_joints()`: Pinocchio → actuated
- `Gr00tObsActionConverter.robocasa_to_gr00t_actuated_order()`: RoboCasa → actuated

## Sim Layer

Both MuJoCo and RoboCasa expose **identical DDS interfaces** via `UnitreeSdk2Bridge`:

**MuJoCo**: `mujoco.mj_step()` → `PublishLowState()` → DDS topics → WBC reads them
**RoboCasa**: `Robosuite.step()` → `Gr00tObsActionConverter` reorder → `PublishLowState()` → same DDS topics

Gym envs registered at import time: `gr00tlocomanip_g1_sim/{TaskName}_G1_gear_wbc`

## Design Insight: Data-Driven DOF Chain

The entire obs/action shape chain is **driven by `RobotSupplementalInfo`**:
- `G1SupplementalInfo` defines joint names, counts, limits, groups
- `RobotModel` builds index maps from it
- `G1DecoupledWholeBodyPolicy` uses group indices only — no hand-specific logic
- `SyncEnv`/`Gr00tObsActionConverter` auto-derive DOF counts

**Changing hand type = changing supplemental info + URDF + DDS interface. Everything else auto-adapts.**

## Dex3 vs Dex1 Implications

| | Dex3 (standard) | Dex1 (our hospitality model) |
|---|---|---|
| DOF per hand | 7 (revolute) | 2 (prismatic) |
| Total DOF | 43 (29 body + 14 hand) | 33 (29 body + 4 hand) |
| DDS topics | `rt/dex3/{l,r}/{cmd,state}` | `rt/dex1/{l,r}/{cmd,state}` |
| URDF | `g1_29dof_with_hand.urdf` | Needs new Dex1 URDF |
| Hand class | `G1ThreeFingerHand` | Needs `G1Dex1Hand` |
| Model stats | `state.left_hand: 7 dims` | `state.left_hand: 1 dim` |

## PnP Eval Results (Standard WBC + Dex3)

NVIDIA pretrained PnP model evaluated on workstation:
- **1/3 success (33%)** — within NVIDIA benchmark of 58% ± 15%
- Videos saved to `dataset_review/official_eval_pnp/`
