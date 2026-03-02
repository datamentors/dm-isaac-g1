# Inference Architecture Comparison: Direct Control vs WBC Pipeline

Comparison of how our G1 robot runs inference in two different architectures:
1. **Direct Control** — our original approach (dm-isaac-g1)
2. **WBC Pipeline** — NVIDIA's GR00T Whole-Body Control (GR00T-WholeBodyControl)

Both use the same GROOT model server for policy inference, but differ fundamentally in how observations reach the model and how model outputs become robot joint commands.

---

## 1. Direct Control Architecture (Original)

Simple 2-process setup: one simulation, one GROOT model server.

```mermaid
flowchart LR
    subgraph SIM["Simulation Process (Isaac Sim / MuJoCo)"]
        direction TB
        S1["1. Read joint positions (31 DOF)"]
        S2["2. Render ego camera (480x640)"]
        S3["3. Build observation dict"]
        S4["4. client.get_action(obs)"]
        S5["5. Extract action[t] from buffer"]
        S6["6. Apply to joints directly"]
        S7["7. env.step(action)"]
        S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
        S7 -.->|"loop"| S1
    end

    subgraph GROOT["GROOT Server (Spark:5555)"]
        MODEL["GROOT Policy Model"]
    end

    S3 -- "obs via ZMQ REQ" --> MODEL
    MODEL -- "actions via ZMQ REP" --> S4

    style SIM fill:#4a90d9,stroke:#2c5f8a,color:#fff
    style GROOT fill:#e07b39,stroke:#b35c1e,color:#fff
    style MODEL fill:#c96830,stroke:#a04f18,color:#fff
    style S1 fill:#6ba3e0,stroke:#4a7fb8,color:#fff
    style S2 fill:#6ba3e0,stroke:#4a7fb8,color:#fff
    style S3 fill:#6ba3e0,stroke:#4a7fb8,color:#fff
    style S4 fill:#6ba3e0,stroke:#4a7fb8,color:#fff
    style S5 fill:#6ba3e0,stroke:#4a7fb8,color:#fff
    style S6 fill:#6ba3e0,stroke:#4a7fb8,color:#fff
    style S7 fill:#6ba3e0,stroke:#4a7fb8,color:#fff
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

```mermaid
flowchart TB
    subgraph GROOT["GROOT Server (Spark:5555)"]
        GMODEL["GROOT Policy Model"]
    end

    subgraph BRIDGE["Bridge (External Integration)"]
        B1["Subscribe env_state_act"]
        B2["Read camera stream"]
        B3["Call GROOT server"]
        B4["Map output to upper body goal"]
        B5["Publish upper_body_pose + navigate_cmd"]
    end

    subgraph WBC["WBC Control Loop (50 Hz)"]
        W1["Read robot state from DDS"]
        subgraph POLICY["G1DecoupledWholeBodyPolicy"]
            INTERP["InterpolationPolicy\n(upper body: arms+waist+hands)"]
            RLPOL["G1GearWbcPolicy\n(lower body: legs via ONNX RL)"]
        end
        W2["Send joint commands via DDS"]
        W3["Publish env_state_act for bridge"]
    end

    subgraph SIM["Simulation (MuJoCo / RoboCasa)"]
        P1["Physics step"]
        P2["Publish sensor data via DDS"]
        P3["Read joint commands via DDS"]
    end

    B3 -- "obs via ZMQ REQ" --> GMODEL
    GMODEL -- "actions via ZMQ REP" --> B3

    B5 -- "ROS2 topics\n(msgpack)" --> W1
    W3 -- "ROS2 topics\n(msgpack)" --> B1

    W2 -- "DDS topics\n(Unitree SDK2)" --> P3
    P2 -- "DDS topics\n(Unitree SDK2)" --> W1

    style GROOT fill:#e07b39,stroke:#b35c1e,color:#fff
    style GMODEL fill:#c96830,stroke:#a04f18,color:#fff
    style BRIDGE fill:#7b68ee,stroke:#5a4cbf,color:#fff
    style B1 fill:#9384f2,stroke:#7060d4,color:#fff
    style B2 fill:#9384f2,stroke:#7060d4,color:#fff
    style B3 fill:#9384f2,stroke:#7060d4,color:#fff
    style B4 fill:#9384f2,stroke:#7060d4,color:#fff
    style B5 fill:#9384f2,stroke:#7060d4,color:#fff
    style WBC fill:#2ecc71,stroke:#1e9650,color:#fff
    style POLICY fill:#27ae60,stroke:#1a7a42,color:#fff
    style INTERP fill:#3dd87f,stroke:#2bb365,color:#fff
    style RLPOL fill:#3dd87f,stroke:#2bb365,color:#fff
    style W1 fill:#3dd87f,stroke:#2bb365,color:#fff
    style W2 fill:#3dd87f,stroke:#2bb365,color:#fff
    style W3 fill:#3dd87f,stroke:#2bb365,color:#fff
    style SIM fill:#4a90d9,stroke:#2c5f8a,color:#fff
    style P1 fill:#6ba3e0,stroke:#4a7fb8,color:#fff
    style P2 fill:#6ba3e0,stroke:#4a7fb8,color:#fff
    style P3 fill:#6ba3e0,stroke:#4a7fb8,color:#fff
```

### How the 3 Processes Communicate

```mermaid
flowchart LR
    GROOT["GROOT Server\n(Spark)"]
    BRIDGE["Bridge\n(gets obs, sends goal)"]
    WBC["WBC Control Loop\n(50 Hz motor control)"]
    SIM["Simulation\n(physics + sensors)"]

    GROOT <-- "ZMQ\nREQ/REP" --> BRIDGE
    BRIDGE -- "ROS2\ntopics" --> WBC
    WBC -- "ROS2\ntopics" --> BRIDGE
    WBC <-- "DDS\ntopics" --> SIM

    style GROOT fill:#e07b39,stroke:#b35c1e,color:#fff
    style BRIDGE fill:#7b68ee,stroke:#5a4cbf,color:#fff
    style WBC fill:#2ecc71,stroke:#1e9650,color:#fff
    style SIM fill:#4a90d9,stroke:#2c5f8a,color:#fff
```

The bridge **never talks to the sim directly**. It only reads published state from the WBC loop and publishes goal commands back.

### Observation Flow (Sim → WBC → Bridge → GROOT)

```mermaid
flowchart LR
    subgraph SIM["Simulation"]
        S1["Publish raw sensor data\nrt/lowstate (29 body motors + IMU)\nrt/dex3/*/state (7 hand motors/hand)\nrt/secondary_imu (torso IMU)\nrt/odostate (floating base)"]
    end

    subgraph WBC["WBC Control Loop"]
        W1["Read DDS & assemble state\nbody_q[29] + hand_q[7+7] = whole_q[43]\n+ FK for wrist poses"]
        W2["Publish assembled state\nG1Env/env_state_act\n{q[43], dq[43], wrist_pose[14], ...}"]
        W1 --> W2
    end

    subgraph BRIDGE["Bridge"]
        B1["Build GROOT observation\nvideo: ego_view (480x640)\nstate: legs/waist/arms/hands\nannotation: task description"]
        B2["Send observation"]
        B1 --> B2
    end

    subgraph GROOT["GROOT Server"]
        G1["Receive observation\n& run inference"]
    end

    S1 -- "DDS" --> W1
    W2 -- "ROS2" --> B1
    B2 -- "ZMQ REQ" --> G1

    style SIM fill:#4a90d9,stroke:#2c5f8a,color:#fff
    style S1 fill:#6ba3e0,stroke:#4a7fb8,color:#fff
    style WBC fill:#2ecc71,stroke:#1e9650,color:#fff
    style W1 fill:#3dd87f,stroke:#2bb365,color:#fff
    style W2 fill:#3dd87f,stroke:#2bb365,color:#fff
    style BRIDGE fill:#7b68ee,stroke:#5a4cbf,color:#fff
    style B1 fill:#9384f2,stroke:#7060d4,color:#fff
    style B2 fill:#9384f2,stroke:#7060d4,color:#fff
    style GROOT fill:#e07b39,stroke:#b35c1e,color:#fff
    style G1 fill:#c96830,stroke:#a04f18,color:#fff
```

### Action Flow (GROOT → Bridge → WBC → Sim)

```mermaid
flowchart LR
    subgraph GROOT["GROOT Server"]
        G1["Return action trajectory\nwaist[3], arms[7+7], hands[N+N]\nbase_height[1], navigate_cmd[3]"]
    end

    subgraph BRIDGE["Bridge"]
        B1["Extract & publish goal\nupper_body_pose[17]\ntarget_time, navigate_cmd[3]\nbase_height_command: 0.74m"]
    end

    subgraph WBC["WBC Control Loop"]
        direction TB
        subgraph UPPER["InterpolationPolicy (Upper Body)"]
            U1["scipy interp1d smooth interpolation\n-> arms[14] + waist[3] + hands[N]"]
        end
        subgraph LOWER["G1GearWbcPolicy (Lower Body, ONNX RL)"]
            L1["86-dim obs x 6 history = 516-dim input\n-> 15-dim action (12 legs + 3 waist)\npolicy_1 (balance) / policy_2 (walk)"]
        end
        ASM["Assembly: q[43] = upper + lower\n(lower body overwrites waist)"]
        U1 --> ASM
        L1 --> ASM
        W1["Send joint targets via DDS\nrt/lowcmd (29 body motors)\nrt/dex*/*/cmd (hand motors)"]
        ASM --> W1
    end

    subgraph SIM["Simulation"]
        S1["Read DDS commands & step physics\nPD: tau = tau_ff + kp*(q_cmd-q) + kd*(dq_cmd-dq)"]
    end

    G1 -- "ZMQ REP" --> B1
    B1 -- "ROS2" --> UPPER
    B1 -- "ROS2" --> LOWER
    W1 -- "DDS" --> S1

    style GROOT fill:#e07b39,stroke:#b35c1e,color:#fff
    style G1 fill:#c96830,stroke:#a04f18,color:#fff
    style BRIDGE fill:#7b68ee,stroke:#5a4cbf,color:#fff
    style B1 fill:#9384f2,stroke:#7060d4,color:#fff
    style WBC fill:#2ecc71,stroke:#1e9650,color:#fff
    style UPPER fill:#27ae60,stroke:#1a7a42,color:#fff
    style U1 fill:#3dd87f,stroke:#2bb365,color:#fff
    style LOWER fill:#e74c3c,stroke:#c0392b,color:#fff
    style L1 fill:#f25c4e,stroke:#d9453a,color:#fff
    style ASM fill:#3dd87f,stroke:#2bb365,color:#fff
    style W1 fill:#3dd87f,stroke:#2bb365,color:#fff
    style SIM fill:#4a90d9,stroke:#2c5f8a,color:#fff
    style S1 fill:#6ba3e0,stroke:#4a7fb8,color:#fff
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

```mermaid
flowchart LR
    subgraph GROOT_OUT["GROOT Output (23+ DOF)"]
        direction TB
        WAIST["waist: 3 DOF\n(yaw, roll, pitch)"]
        LA["left_arm: 7 DOF\n(shoulder p/r/y, elbow, wrist r/p/y)"]
        RA["right_arm: 7 DOF\n(same)"]
        LH["left_hand: N DOF\n(1 for Dex1, 7 for Dex3)"]
        RH["right_hand: N DOF\n(1 for Dex1, 7 for Dex3)"]
        BH["base_height: 1 DOF"]
        NAV["navigate_cmd: 3 DOF\n(vx, vy, omega)"]
    end

    style GROOT_OUT fill:#e07b39,stroke:#b35c1e,color:#fff
    style WAIST fill:#f5a623,stroke:#d48b0f,color:#fff
    style LA fill:#c96830,stroke:#a04f18,color:#fff
    style RA fill:#c96830,stroke:#a04f18,color:#fff
    style LH fill:#d4944d,stroke:#b37530,color:#fff
    style RH fill:#d4944d,stroke:#b37530,color:#fff
    style BH fill:#8b6914,stroke:#6b5010,color:#fff
    style NAV fill:#8b6914,stroke:#6b5010,color:#fff
```

The difference is what happens AFTER GROOT produces its output:

- **Direct Control**: joint targets go straight to the sim PD controller. No balance, no interpolation, no safety checks.
- **WBC Pipeline**: joint targets become a "goal" that the WBC policy interpolates toward while simultaneously running an RL balance controller for the legs.

### Why WBC Matters for Real Deployment

```mermaid
flowchart TB
    subgraph DC["Direct Control"]
        direction TB
        DC_GROOT["GROOT"] --> DC_JOINTS["Joints (direct)"]
        DC_N1["No balance"]
        DC_N2["No safety"]
        DC_N3["Fixed base only"]
    end

    subgraph WBCP["WBC Pipeline"]
        direction TB
        WBC_GROOT["GROOT"] --> WBC_GOAL["Goal"]
        WBC_GOAL --> WBC_INTERP["WBC interpolates smoothly"]
        WBC_GOAL --> WBC_RL["RL balances legs"]
        WBC_INTERP --> WBC_SAFE["Safety monitors all joints"]
        WBC_RL --> WBC_SAFE
        WBC_REAL["Same code runs on real robot"]
    end

    style DC fill:#e74c3c,stroke:#c0392b,color:#fff
    style DC_GROOT fill:#e07b39,stroke:#b35c1e,color:#fff
    style DC_JOINTS fill:#f25c4e,stroke:#d9453a,color:#fff
    style DC_N1 fill:#f25c4e,stroke:#d9453a,color:#fff
    style DC_N2 fill:#f25c4e,stroke:#d9453a,color:#fff
    style DC_N3 fill:#f25c4e,stroke:#d9453a,color:#fff
    style WBCP fill:#2ecc71,stroke:#1e9650,color:#fff
    style WBC_GROOT fill:#e07b39,stroke:#b35c1e,color:#fff
    style WBC_GOAL fill:#3dd87f,stroke:#2bb365,color:#fff
    style WBC_INTERP fill:#3dd87f,stroke:#2bb365,color:#fff
    style WBC_RL fill:#3dd87f,stroke:#2bb365,color:#fff
    style WBC_SAFE fill:#3dd87f,stroke:#2bb365,color:#fff
    style WBC_REAL fill:#3dd87f,stroke:#2bb365,color:#fff
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

| | Dex3 (standard) | Dex1 (our model) |
|---|---|---|
| **Body** | 29 DOF | 29 DOF |
| **Left hand** | 7 DOF | 2 DOF |
| **Right hand** | 7 DOF | 2 DOF |
| **Total** | **43 DOF** | **33 DOF** |
| **DDS hand topics** | `rt/dex3/*/` | `rt/dex1/*/` |
| **URDF** | `g1_29dof_with_hand` | `g1_29dof_with_dex1` |
| **Gym env** | `*_G1_gear_wbc` | `*_G1Dex1_gear_wbc` |

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
