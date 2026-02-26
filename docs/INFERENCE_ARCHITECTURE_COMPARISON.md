# Inference Architecture Comparison: Direct Control vs WBC Pipeline

Comparison of how our G1 robot runs inference in two different architectures:
1. **Direct Control** — our original approach (dm-isaac-g1)
2. **WBC Pipeline** — NVIDIA's GR00T Whole-Body Control (GR00T-WholeBodyControl)

Both use the same GROOT model server for policy inference, but differ fundamentally in how observations reach the model and how model outputs become robot joint commands.

---

## 1. Direct Control Architecture (Original)

Simple 2-process setup: one simulation, one GROOT model server.

```
┌─────────────────────────────────────────────────┐
│  Simulation Process (Isaac Sim or MuJoCo)        │
│                                                   │
│  env.reset()                                      │
│  while not done:                                  │
│    ┌──────────────────────────────────┐           │
│    │ 1. Read joint positions (31 DOF) │           │
│    │ 2. Render ego camera (480×640)   │           │
│    │ 3. Build observation dict        │──► ZMQ ──►│── GROOT Server (Spark)
│    │ 4. client.get_action(obs)        │◄── ZMQ ◄──│   port 5555
│    │ 5. Extract action[t] from buffer │           │
│    │ 6. Apply to joints directly      │           │
│    │ 7. env.step(action)              │           │
│    └──────────────────────────────────┘           │
└─────────────────────────────────────────────────┘
```

### Observation Format

The simulation reads joint positions directly and packages them into the GROOT observation format:

```
observation = {
    "video": {
        "ego_view": (1, 1, 480, 640, 3) uint8
    },
    "state": {
        "left_leg":   (1, 1, 6)  float32    ─┐
        "right_leg":  (1, 1, 6)  float32     │ 31 DOF total
        "waist":      (1, 1, 3)  float32     │ (read from sim joint positions)
        "left_arm":   (1, 1, 7)  float32     │
        "right_arm":  (1, 1, 7)  float32     │
        "left_hand":  (1, 1, 1)  float32     │ gripper (normalized 0-1)
        "right_hand": (1, 1, 1)  float32    ─┘
    },
    "annotation": {
        "human.task_description": [["fold the towel"]]
    }
}
```

### Action Format

GROOT returns a 30-step action trajectory. The eval script executes some number of steps before re-querying:

```
action_dict = {
    "action.waist":      (1, 30, 3)   ABSOLUTE
    "action.left_arm":   (1, 30, 7)   RELATIVE (delta from trajectory start)
    "action.right_arm":  (1, 30, 7)   RELATIVE (delta from trajectory start)
    "action.left_hand":  (1, 30, 1)   ABSOLUTE
    "action.right_hand": (1, 30, 1)   ABSOLUTE
    "action.base_height_command": (1, 30, 1)  ABSOLUTE (ignored in fixed-base)
    "action.navigate_command":    (1, 30, 3)  ABSOLUTE (ignored in fixed-base)
}
```

**Key: arms use relative deltas** — `target = start_pos + delta * action_scale`.

### Action Application

```python
for t in range(num_action_steps):
    # Extract action at timestep t
    for group in ["waist", "left_arm", "right_arm", "left_hand", "right_hand"]:
        if group in ("left_arm", "right_arm"):
            target = trajectory_start_pos + action[t] * action_scale  # relative
        else:
            target = action[t]  # absolute

    # Apply directly to joint targets via PD controller
    for _ in range(control_decimation):
        env.step(target_joint_positions)
```

### Key Characteristics

| Property | Value |
|----------|-------|
| Processes | 2 (sim + GROOT server) |
| Communication | ZMQ REQ/REP only |
| Robot base | **Fixed** (no walking) |
| Balance controller | **None** |
| Lower body | Observed only, NOT controlled |
| Hand type | 1 DOF gripper (Dex1) |
| State dims | 31 DOF (legs observed, not controlled) |
| Action dims | 23 DOF (waist + arms + hands + nav + height) |
| Action horizon | 30 steps, execute K before re-query |
| Camera | 1 ego-view |
| Frequency | Sim-rate / control_decimation |
| Action type | Mixed (relative arms, absolute rest) |

### Limitations

1. **Fixed base only** — robot cannot walk, must be placed at the task
2. **No balance** — legs are passive; if base isn't fixed, robot falls over
3. **No locomotion** — `navigate_command` and `base_height_command` are in the action output but have no effect without a controller to execute them
4. **Tight coupling** — observation building and action application are embedded in the eval script, making it hard to swap sim backends

---

## 2. WBC Pipeline Architecture (NVIDIA GR00T-WholeBodyControl)

3-process architecture with decoupled control. The key innovation: **the GROOT model only controls the upper body, while a separate RL-trained policy controls the legs for balance and locomotion**.

```
┌──────────────────────────────────────────────────────────┐
│  BRIDGE (external integration)                            │
│  ├─ Subscribe env_state_act           ◄── FROM WBC CTRL  │
│  ├─ Read camera stream (ZMQ PUB/SUB)                      │
│  ├─ Call GROOT server (ZMQ REQ/REP)   ──► GROOT (Spark)   │
│  ├─ Map model output → upper body goal                    │
│  └─ Publish upper_body_pose + navigate_cmd                │
└──────────────────────┬────────────────────────────────────┘
                       │ ROS2 topics (msgpack)
┌──────────────────────▼────────────────────────────────────┐
│  WBC Control Loop (50 Hz)                                  │
│  ├─ Read robot state from DDS (rt/lowstate, rt/dex*/*)     │
│  ├─ G1DecoupledWholeBodyPolicy:                            │
│  │   ├─ InterpolationPolicy (upper body: arms+waist+hands) │
│  │   └─ G1GearWbcPolicy (lower body: legs via ONNX RL)    │
│  ├─ Send joint commands via DDS (rt/lowcmd, rt/dex*/*)     │
│  └─ Publish env_state_act for bridge                       │
└──────────────────────┬────────────────────────────────────┘
                       │ DDS topics (Unitree SDK2)
┌──────────────────────▼────────────────────────────────────┐
│  Simulation (MuJoCo or RoboCasa)                           │
│  ├─ Physics step                                           │
│  ├─ Publish sensor data → DDS                              │
│  └─ Read joint commands ← DDS                              │
└───────────────────────────────────────────────────────────┘
```

### How the 3 Processes Communicate

```
GROOT Server ◄──ZMQ──► Bridge ──ROS2──► WBC Control Loop ◄──DDS──► Simulation
   (Spark)              (gets obs,        (50 Hz motor         (physics +
                         sends goal)       control)              sensors)
```

The bridge **never talks to the sim directly**. It only reads published state from the WBC loop and publishes goal commands back.

### Observation Flow (Sim → WBC → Bridge → GROOT)

```
Step 1: Sim publishes raw sensor data via DDS
  rt/lowstate      → 29 body motor states (q, dq, ddq, tau) + pelvis IMU
  rt/dex3/*/state  → 7 hand motor states per hand (or rt/dex1/* for Dex1: 2 per hand)
  rt/secondary_imu → torso IMU
  rt/odostate      → floating base position/velocity (sim only)

Step 2: WBC control loop reads DDS and assembles whole-body state
  body_q[29] + hand_q[7+7] → whole_q[43]  (via RobotModel joint reordering)
  + FK for wrist poses

Step 3: WBC publishes assembled state via ROS2
  G1Env/env_state_act → {q[43], dq[43], wrist_pose[14], action[43], ...}

Step 4: Bridge reads state and builds GROOT observation
  observation = {
      "video": {"ego_view": camera_image},
      "state": {
          "left_leg": q[0:6], "right_leg": q[6:12], "waist": q[12:15],
          "left_arm": q[15:22], "right_arm": q[22:29],
          "left_hand": q[29:36], "right_hand": q[36:43]
      },
      "annotation": {"human.task_description": [["pick up the apple"]]}
  }

Step 5: Bridge sends observation to GROOT server via ZMQ
```

### Action Flow (GROOT → Bridge → WBC → Sim)

```
Step 1: GROOT returns action trajectory
  action_dict = {waist[3], left_arm[7], right_arm[7], left_hand[N], right_hand[N],
                 base_height[1], navigate_cmd[3]}

Step 2: Bridge extracts and publishes goal via ROS2
  ControlPolicy/upper_body_pose = {
      target_upper_body_pose[17]: [waist(3) + left_arm(7) + right_arm(7)],
      target_time: when to reach this pose,
      navigate_cmd[3]: [vx, vy, omega],
      base_height_command: 0.74m default
  }

Step 3: WBC policy computes full joint command (THIS IS THE KEY DIFFERENCE)
  ┌─ InterpolationPolicy (upper body):
  │    scipy interp1d smoothly interpolates to target pose
  │    → arms[14] + waist[3] + hands[N]
  │
  └─ G1GearWbcPolicy (lower body, ONNX RL):
       86-dim obs (cmd, height, rpy, omega, gravity, q, dq, last_action)
       × 6 history frames = 516-dim input
       → ONNX inference → 15-dim action (12 legs + 3 waist)
       Uses policy_1 (balance, ||cmd||<0.05) or policy_2 (walk)

  Assembly: q[43] = upper_body[arms+hands] + lower_body[legs+waist]
  (lower body overwrites waist when both claim it)

Step 4: WBC sends joint targets via DDS
  rt/lowcmd         → 29 body motor targets (q, dq, tau, kp, kd)
  rt/dex3/*/cmd     → 7 hand motor targets per hand (or rt/dex1/*: 2 per hand)

Step 5: Sim reads DDS commands and steps physics
  PD control: tau = tau_ff + kp*(q_cmd - q) + kd*(dq_cmd - dq)
```

### Key Characteristics

| Property | Value |
|----------|-------|
| Processes | 3 (sim + WBC control loop + GROOT server) + bridge |
| Communication | DDS (sim↔WBC) + ROS2 (WBC↔bridge) + ZMQ (bridge↔GROOT) |
| Robot base | **Free-floating** (walks, balances) |
| Balance controller | **RL-trained ONNX policy** (2 models: balance + walk) |
| Lower body | Actively controlled by RL policy at 50 Hz |
| Hand type | Configurable: Dex3 (7 DOF) or Dex1 (2 DOF) |
| State dims | 43 DOF with Dex3, 33 DOF with Dex1 |
| Action dims | 43 DOF with Dex3, 33 DOF with Dex1 |
| Control frequency | 50 Hz (with 4 physics substeps at 200 Hz) |
| Camera | Ego-view + optional wrist cameras |
| Action type | Absolute targets (interpolated) |

---

## 3. Side-by-Side Comparison

| Aspect | Direct Control | WBC Pipeline |
|--------|---------------|--------------|
| **Processes** | 2 (sim + GROOT) | 3+ (sim + WBC + GROOT + bridge) |
| **Robot stability** | Fixed base required | Self-balancing (RL) |
| **Walking** | No | Yes (velocity commands) |
| **GROOT controls** | Arms + waist + hands | Arms + waist + hands (same) |
| **Leg control** | None (passive) | RL policy (ONNX, 50 Hz) |
| **Action application** | Direct to joints | Interpolated + safety-monitored |
| **Action space** | Mixed (relative arms) | Absolute targets |
| **Safety** | Clip to ±3.14 | Joint limits + velocity monitoring + startup ramp |
| **Sim coupling** | Tight (in eval script) | Loose (DDS interface) |
| **Real robot** | Not supported | Same code, same DDS interface |
| **Sim backends** | Isaac Sim or MuJoCo | MuJoCo or RoboCasa (same DDS) |

### What GROOT Actually Controls (Same in Both)

In both architectures, the GROOT model predicts the **same thing**: upper body joint targets.

```
GROOT output (23 DOF):
  waist:         3 DOF (yaw, roll, pitch)
  left_arm:      7 DOF (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
  right_arm:     7 DOF (same)
  left_hand:     N DOF (1 for Dex1, 7 for Dex3)
  right_hand:    N DOF (1 for Dex1, 7 for Dex3)
  base_height:   1 DOF
  navigate_cmd:  3 DOF (vx, vy, omega)
```

The difference is what happens AFTER GROOT produces its output:

- **Direct Control**: joint targets go straight to the sim PD controller. No balance, no interpolation, no safety checks.
- **WBC Pipeline**: joint targets become a "goal" that the WBC policy interpolates toward while simultaneously running an RL balance controller for the legs.

### Why WBC Matters for Real Deployment

```
Direct Control:                    WBC Pipeline:
  GROOT → joints                     GROOT → goal
  (no balance)                       WBC interpolates smoothly
  (no safety)                        RL balances legs
  (fixed base only)                  Safety monitors all joints
                                     Same code runs on real robot
```

---

## 4. Our Dex1 Modifications to the WBC Pipeline

Our hospitality model was trained with `UNITREE_G1` embodiment on Dex1 data (1 DOF per hand), but NVIDIA's standard WBC pipeline uses Dex3 (7 DOF per hand). We parameterized the pipeline to support both.

### What Changed

| Layer | Change | Files |
|-------|--------|-------|
| Robot model | `hand_type="dex1"` param, Dex1 URDF + meshes | `g1_supplemental_info.py`, `g1.py`, `g1_29dof_with_dex1.urdf` |
| DDS interface | Dynamic topics `rt/dex1/*/cmd,state`, dof=2 | `command_sender.py`, `state_processor.py` |
| Hand env | `G1Dex1Hand` class (2 prismatic DOF) | `g1_hand.py`, `g1_env.py` |
| RoboCasa sim | Dex1 gripper MJCF + `G1Dex1` robot variant | `g1_dex1_hands.py`, `g1_robot.py`, MJCF XMLs |

### What Did NOT Change (Auto-Adapts)

- `G1DecoupledWholeBodyPolicy` — uses group indices from robot model, no hand-specific logic
- `SyncEnv` — derives DOF counts from `RobotModel`
- `Gr00tObsActionConverter` — reads joint groups from `G1SupplementalInfo`
- ONNX lower body policy — only controls legs+waist, hand-agnostic

### DOF Comparison

```
                    Dex3 (standard)    Dex1 (our model)
Body:               29 DOF             29 DOF
Left hand:           7 DOF              2 DOF
Right hand:          7 DOF              2 DOF
Total:              43 DOF             33 DOF

DDS hand topics:    rt/dex3/*/         rt/dex1/*/
URDF:               g1_29dof_with_hand g1_29dof_with_dex1
Gym env:            *_G1_gear_wbc      *_G1Dex1_gear_wbc
```

---

## 5. Running Inference

### Direct Control (Old)

```bash
# Terminal 1: GROOT server on Spark
python scripts/eval/run_groot_server.py \
    --model-path /path/to/checkpoint \
    --embodiment-tag UNITREE_G1

# Terminal 2: Eval on workstation
python scripts/eval/run_mujoco_towel_eval.py \
    --groot-host 192.168.1.237 \
    --groot-port 5555 \
    --num-episodes 10
```

### WBC Pipeline (New, with Dex1)

```bash
# Terminal 1: GROOT server on Spark
python gr00t/eval/run_gr00t_server.py \
    --model-path /path/to/checkpoint \
    --embodiment-tag UNITREE_G1 \
    --use-sim-policy-wrapper

# Terminal 2: WBC eval on workstation
$WBC_PY gr00t/eval/rollout_policy.py \
    --n_episodes 5 \
    --env_name gr00tlocomanip_g1_dex1_sim/LMPnPAppleToPlateDC_G1Dex1_gear_wbc \
    --policy_client_host 192.168.1.237 \
    --policy_client_port 5555 \
    --n_action_steps 20
```
