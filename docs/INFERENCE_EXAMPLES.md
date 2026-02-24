# GROOT Inference Examples

Concrete code examples showing how to call the GROOT inference server, how observations are constructed, and how camera/state keys are mapped between Isaac Sim and the model.

## Table of Contents

- [Key Concept: Two Embodiment Formats](#key-concept-two-embodiment-formats)
- [Camera Key Mapping: `cam_left_high` vs `ego_view`](#camera-key-mapping-cam_left_high-vs-ego_view)
- [Example 1: UNITREE_G1 Gripper (31 DOF, `ego_view`)](#example-1-unitree_g1-gripper-31-dof-ego_view)
- [Example 2: new_embodiment Dex3 (28 DOF, `cam_left_high`)](#example-2-new_embodiment-dex3-28-dof-cam_left_high)
- [Example 3: new_embodiment Inspire (53 DOF, `cam_left_high`)](#example-3-new_embodiment-inspire-53-dof-cam_left_high)
- [Example 4: Multi-Camera Dex3 (4 cameras)](#example-4-multi-camera-dex3-4-cameras)
- [State Vector Construction](#state-vector-construction)
- [Action Handling](#action-handling)
- [Client Comparison: PolicyClient vs GrootClient](#client-comparison-policyclient-vs-grootclient)
- [Common Mistakes](#common-mistakes)
- [File Reference](#file-reference)

---

## Key Concept: Two Embodiment Formats

The GROOT server accepts observations in different formats depending on the **embodiment tag** used when the model was fine-tuned:

| Embodiment Tag | Camera Key | State Key | DOF | Client |
|----------------|-----------|-----------|-----|--------|
| `UNITREE_G1` (gripper) | `ego_view` | `observation.state` (31 DOF) | 31 state / 23 action | `PolicyClient` |
| `new_embodiment` (Dex3 28 DOF) | `cam_left_high` + extras | `observation.state` (28 DOF) | 28 state / 28 action | `PolicyClient` |
| `new_embodiment` (Inspire 53 DOF) | `cam_left_high` | `observation.state` (53 DOF) | 53 state / 53 action | `PolicyClient` |

The camera key name must **exactly match** what the model was trained with. Using the wrong key results in the model receiving no visual input, producing random/poor actions.

---

## Camera Key Mapping: `cam_left_high` vs `ego_view`

### How it works during training data conversion

When converting LeRobot datasets to GR00T format for the UNITREE_G1 embodiment, the camera is **renamed** from `cam_left_high` to `ego_view`:

```python
# From src/dm_isaac_g1/data/convert_to_groot.py (lines 165-169)
ego_source_key = f"observation.images.{ego_camera}"   # "observation.images.cam_left_high"
ego_output_key = "observation.images.ego_view"          # renamed for UNITREE_G1
if ego_source_key in data:
    output_data[ego_output_key] = data[ego_source_key]
```

This means:
- **UNITREE_G1 models** were trained with the key `ego_view` (even though the physical camera is the same d435_link head camera called `cam_left_high`)
- **new_embodiment models** (Dex3, Inspire) were trained with `cam_left_high` directly

### How it works during inference

The `PolicyClient` from `gr00t.policy.server_client` uses a `Gr00tSimPolicyWrapper` on the server side that handles key mapping automatically. Your observation dict just needs to use the correct key for the embodiment.

---

## Example 1: UNITREE_G1 Gripper (31 DOF, `ego_view`)

Use this format when running against the `groot-g1-gripper-hospitality-7ds` model (or any model fine-tuned with `UNITREE_G1` embodiment tag).

```python
import numpy as np
from gr00t.policy.server_client import PolicyClient

# Connect to GROOT server on Spark
client = PolicyClient(host="192.168.1.237", port=5555, strict=False)

# --- Build observation ---

# Camera: single ego-view image from d435_link head camera
# Shape: (batch=1, time=1, H=480, W=640, channels=3), dtype uint8
camera_image = get_camera_rgb()  # your camera capture function
ego_view = camera_image.reshape(1, 1, 480, 640, 3).astype(np.uint8)

# State: 31 DOF flat vector
# Layout: left_leg(6) + right_leg(6) + waist(3) + left_arm(7) + right_arm(7) + left_gripper(1) + right_gripper(1)
joint_positions = get_robot_joints()  # your joint state function
state = joint_positions.reshape(1, 1, 31).astype(np.float32)

# Language: task description matching training data
task = "fold the towel"

# Assemble observation dict
# IMPORTANT: Use "ego_view" as the camera key, NOT "cam_left_high"
observation = {
    "video.ego_view": ego_view,                            # (1, 1, 480, 640, 3) uint8
    "state.observation.state": state,                      # (1, 1, 31) float32
    "annotation.human.task_description": [task],           # list of strings
}

# --- Get action ---
action_dict, info = client.get_action(observation)
# action_dict["action"] shape: (1, 30, 23) — batch=1, horizon=30, action_dof=23

# Extract first action step
action = action_dict["action"][0, 0, :]  # (23,) — single timestep

# Action layout (23 DOF):
# [0:3]   waist (ABSOLUTE)
# [3:10]  left arm (RELATIVE — delta from trajectory start)
# [10:17] right arm (RELATIVE)
# [17]    left gripper (ABSOLUTE)
# [18]    right gripper (ABSOLUTE)
# [19]    base height (ABSOLUTE)
# [20:23] navigation vx, vy, angular_z (ABSOLUTE)
```

### State vector layout (31 DOF)

```
Index  Component        DOF  Notes
─────  ─────────────    ───  ─────
0-5    Left Leg          6   hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll
6-11   Right Leg         6   (same order)
12-14  Waist             3   yaw, pitch, roll
15-21  Left Arm          7   shoulder_pitch/roll/yaw, elbow, wrist_yaw/roll/pitch
22-28  Right Arm         7   (same order)
29     Left Gripper      1   open/close
30     Right Gripper     1   open/close
```

---

## Example 2: new_embodiment Dex3 (28 DOF, `cam_left_high`)

Use this format when running against the `groot-g1-dex3-28dof` model (fine-tuned with `new_embodiment` / `NEW_EMBODIMENT` tag).

```python
import numpy as np
from gr00t.policy.server_client import PolicyClient

client = PolicyClient(host="192.168.1.237", port=5555, strict=False)

# --- Build observation ---

# Camera: use "cam_left_high" (NOT "ego_view")
camera_image = get_camera_rgb()
cam_left_high = camera_image.reshape(1, 1, 480, 640, 3).astype(np.uint8)

# State: 28 DOF — arms + Dex3 hands only (no legs, no waist)
# Layout: left_arm(7) + right_arm(7) + left_dex3(7) + right_dex3(7)
joint_positions = get_arm_and_hand_joints()
state = joint_positions.reshape(1, 1, 28).astype(np.float32)

# Observation uses NESTED dict format (not flat dot-separated keys)
observation = {
    "video": {
        "cam_left_high": cam_left_high,           # (1, 1, 480, 640, 3) uint8
    },
    "state": {
        "observation.state": state,                # (1, 1, 28) float32
    },
    "language": {
        "task": [["Stack the blocks"]],            # nested list: [[task_per_batch]]
    },
}

# --- Get action ---
action_dict, info = client.get_action(observation)
# action_dict["action"] shape: (1, 30, 28)

action = action_dict["action"][0, 0, :]  # (28,) ABSOLUTE joint targets

# Action layout (28 DOF):
# [0:7]   left arm  — shoulder_pitch/roll/yaw, elbow, wrist_yaw/roll/pitch
# [7:14]  right arm — (same order)
# [14:21] left Dex3 hand — index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2
# [21:28] right Dex3 hand — (same order)
```

---

## Example 3: new_embodiment Inspire (53 DOF, `cam_left_high`)

For models fine-tuned with 53-DOF Inspire hands:

```python
import numpy as np
from gr00t.policy.server_client import PolicyClient

client = PolicyClient(host="192.168.1.237", port=5555, strict=False)

# Camera: "cam_left_high"
cam = get_camera_rgb().reshape(1, 1, 480, 640, 3).astype(np.uint8)

# State: 53 DOF full body
# Layout: left_leg(6) + right_leg(6) + waist(3) + left_arm(7) + right_arm(7)
#         + left_inspire_hand(12) + right_inspire_hand(12)
state = get_full_body_joints().reshape(1, 1, 53).astype(np.float32)

observation = {
    "video": {"cam_left_high": cam},
    "state": {"observation.state": state},
    "language": {"task": [["Pick up the red apple and place it on the plate"]]},
}

action_dict, info = client.get_action(observation)
# action_dict["action"] shape: (1, 30, 53) — ABSOLUTE joint targets
```

---

## Example 4: Multi-Camera Dex3 (4 cameras)

The Dex3 block-stacking model uses 4 cameras. This is the full production observation format used by `policy_inference_groot_g1.py`.

```python
import numpy as np
from gr00t.policy.server_client import PolicyClient

client = PolicyClient(host="192.168.1.237", port=5555, strict=False)

# --- Capture all 4 camera images ---
# Each camera: (1, 1, 480, 640, 3) uint8
cam_left_high  = get_head_camera().reshape(1, 1, 480, 640, 3).astype(np.uint8)
cam_right_high = cam_left_high.copy()  # duplicate of head camera (training used same image)
cam_left_wrist = get_left_wrist_camera().reshape(1, 1, 480, 640, 3).astype(np.uint8)
cam_right_wrist = get_right_wrist_camera().reshape(1, 1, 480, 640, 3).astype(np.uint8)

# --- Build state ---
state = get_arm_and_hand_joints().reshape(1, 1, 28).astype(np.float32)

# --- Assemble observation ---
observation = {
    "video": {
        "cam_left_high": cam_left_high,
        "cam_right_high": cam_right_high,    # duplicate of primary (head camera)
        "cam_left_wrist": cam_left_wrist,
        "cam_right_wrist": cam_right_wrist,
    },
    "state": {
        "observation.state": state,
    },
    "language": {
        "task": [["Stack the blocks"]],
    },
}

action_dict, info = client.get_action(observation)
```

### Camera mounting points

| Camera | Robot Link | Description |
|--------|-----------|-------------|
| `cam_left_high` | `d435_link` (head) | Primary head camera, forward-facing |
| `cam_right_high` | `d435_link` (head) | Duplicate of primary (same image, different key) |
| `cam_left_wrist` | `left_hand_camera_base_link` | Left wrist camera |
| `cam_right_wrist` | `right_hand_camera_base_link` | Right wrist camera |

See `scripts/inference_setups.py` `dex3_stack` setup (line 390) for the exact positions and rotations.

---

## State Vector Construction

### Mapping Isaac Sim joints to training DOF layout

Isaac Sim robots have joints in articulation order (e.g., 37 joints for G1+Dex3). The training data uses a different, smaller DOF vector. The mapping is built from the DOF layout:

```python
# From scripts/policy_inference_groot_g1.py, build_flat_observation() lines 348-373

# DOF layout for 28-DOF Dex3 model:
body_part_ranges = [
    ("left_arm",  0,  7),   # DOF indices 0-6 → left arm joints
    ("right_arm", 7,  14),  # DOF indices 7-13 → right arm joints
    ("left_hand", 14, 21),  # DOF indices 14-20 → left Dex3 hand joints
    ("right_hand", 21, 28), # DOF indices 21-27 → right Dex3 hand joints
]

# Build mapping: training DOF index → Isaac Sim robot joint index
joint_to_dof_mapping = {}
for part_name, start_idx, end_idx in body_part_ranges:
    robot_joint_ids = group_joint_ids[part_name]  # resolved from robot articulation
    for i, dof_idx in enumerate(range(start_idx, end_idx)):
        joint_to_dof_mapping[dof_idx] = robot_joint_ids[i]

# Construct state vector
state_dim = 28  # from statistics.json
vals = np.zeros((batch_size, state_dim), dtype=np.float32)
for dof_idx, robot_joint_idx in joint_to_dof_mapping.items():
    if robot_joint_idx is not None:
        vals[:, dof_idx] = joint_pos[:, robot_joint_idx]

# Apply sign negation for joints with flipped sim-vs-real conventions
if negate_state_mask is not None:
    vals = vals * negate_state_mask[:state_dim]

# Add time dimension: (B, D) → (B, 1, D)
vals = vals[:, None, :]
observation["state"] = {"observation.state": vals.astype(np.float32)}
```

### Sign negation

Some sim URDF joints use the opposite sign convention from the real robot training data. The `negate_action_joints` field in `InferenceSetup` lists which joints to negate. Negation is applied to **both** the state observation (so the model sees correct signs) and the action output (so the sim joint moves in the correct direction).

```python
# Joints with flipped sign for Dex3 (from inference_setups.py dex3_stack):
negate_action_joints = [
    "left_hand_middle_0_joint",   # training: +0.91, sim range: [-1.49, -0.08]
    "left_hand_thumb_2_joint",    # training: -0.55, sim range: [+0.09, +1.66]
    "right_hand_index_0_joint",   # training: -0.54, sim range: [+0.08, +1.49]
    "right_hand_middle_0_joint",  # training: -0.88, sim range: [+0.08, +1.49]
    "right_hand_thumb_2_joint",   # training: +0.43, sim range: [-1.66, -0.09]
]
```

---

## Action Handling

### UNITREE_G1: Mixed absolute + relative

```python
action = action_dict["action"][0]  # (30, 23) — 30 timesteps, 23 DOF

# Waist (indices 0-2): ABSOLUTE targets — use directly
waist_target = action[step, 0:3]

# Arms (indices 3-16): RELATIVE deltas — add to trajectory start position
left_arm_target = trajectory_start_pos[3:10] + action[step, 3:10]
right_arm_target = trajectory_start_pos[10:17] + action[step, 10:17]

# Grippers (indices 17-18): ABSOLUTE targets
left_gripper = action[step, 17]
right_gripper = action[step, 18]
```

### new_embodiment (Dex3/Inspire): All absolute

```python
action = action_dict["action"][0]  # (30, 28) or (30, 53)

# ALL joints are ABSOLUTE position targets — use directly
joint_targets = action[step, :]  # apply directly to robot
```

### Full trajectory execution with action chunking

```python
# Get 30-step trajectory from server
action_dict, info = client.get_action(observation)
action_trajectory = action_dict["action"][0]  # (30, DOF)

# Execute N steps, then re-query with fresh observation
num_action_steps = 30  # execute full trajectory before re-planning

for step_idx in range(num_action_steps):
    joint_target = action_trajectory[step_idx]

    # Apply sign negation for sim
    if negate_action_mask is not None:
        joint_target = joint_target * negate_action_mask

    # Apply to sim (with optional control decimation)
    for _ in range(control_decimation):
        obs = env.step(joint_target)

# After executing all steps, get new observation and re-query server
```

---

## Client Comparison: PolicyClient vs GrootClient

| Feature | `PolicyClient` (NVIDIA) | `GrootClient` (custom) |
|---------|------------------------|----------------------|
| Source | `gr00t.policy.server_client` | `dm_isaac_g1.inference.client` |
| Used by | `policy_inference_groot_g1.py` (production) | `test_groot_connection.py`, standalone |
| Protocol | ZeroMQ via `Gr00tSimPolicyWrapper` | ZeroMQ (direct) |
| Observation format | Flat or nested dict (wrapper handles mapping) | Nested dict only |
| Camera key handling | Wrapper maps keys automatically | Sends keys as-is |
| Server-side wrapper | `--use-sim-policy-wrapper` on server | No wrapper needed |
| `client.reset()` | Yes (resets internal state) | No reset method |
| `get_modality_config()` | Yes | No |

### When to use which

- **Production inference in Isaac Sim**: Use `PolicyClient` — it works with the `Gr00tSimPolicyWrapper` on the server, handles observation normalization, and provides `get_modality_config()`.
- **Quick connection tests**: Use `GrootClient` — simpler API, fewer dependencies.
- **Real robot deployment**: Use `PolicyClient` — it's the standard NVIDIA interface.

---

## Common Mistakes

### 1. Wrong camera key for the embodiment

```python
# WRONG: Using cam_left_high with UNITREE_G1 model
observation = {"video.cam_left_high": image}  # Model expects "ego_view"!

# CORRECT: Use ego_view for UNITREE_G1
observation = {"video.ego_view": image}

# CORRECT: Use cam_left_high for new_embodiment (Dex3/Inspire)
observation = {"video": {"cam_left_high": image}}
```

### 2. Wrong observation dict format

```python
# WRONG: Flat keys with new_embodiment model
observation = {
    "video.cam_left_high": image,              # flat format
    "state.observation.state": state,           # flat format
}

# CORRECT: Nested dict for new_embodiment
observation = {
    "video": {"cam_left_high": image},          # nested
    "state": {"observation.state": state},      # nested
    "language": {"task": [["Stack blocks"]]},   # nested
}

# NOTE: UNITREE_G1 with PolicyClient uses the FLAT format:
observation = {
    "video.ego_view": image,
    "state.observation.state": state,
    "annotation.human.task_description": ["fold the towel"],
}
```

### 3. Missing time dimension

```python
# WRONG: Image shape (1, 480, 640, 3) — missing time dim
image = camera_rgb.reshape(1, 480, 640, 3)

# CORRECT: Image shape (1, 1, 480, 640, 3) — batch + time
image = camera_rgb.reshape(1, 1, 480, 640, 3)

# WRONG: State shape (1, 28) — missing time dim
state = joints.reshape(1, 28)

# CORRECT: State shape (1, 1, 28)
state = joints.reshape(1, 1, 28)
```

### 4. Forgetting sign negation

```python
# WRONG: Sending model's predicted action directly to sim
env.step(action_from_model)

# CORRECT: Apply sign negation for joints with flipped conventions
action_for_sim = action_from_model * negate_action_mask
env.step(action_for_sim)

# Also negate the STATE before sending to model:
state_for_model = sim_joint_positions * negate_state_mask
```

### 5. Using action_horizon=1

```python
# WRONG: Single-step prediction causes jittering
action = client.get_action(obs, action_horizon=1)

# CORRECT: Use 8-16 step horizon with receding horizon control
action = client.get_action(obs, action_horizon=16, execute_steps=1)
```

---

## File Reference

| File | Description |
|------|-------------|
| [`scripts/policy_inference_groot_g1.py`](../scripts/policy_inference_groot_g1.py) | Production inference loop with Isaac Sim (uses `PolicyClient`) |
| [`scripts/inference_setups.py`](../scripts/inference_setups.py) | Camera, scene, and DOF configuration for each setup |
| [`scripts/test_groot_connection.py`](../scripts/test_groot_connection.py) | Quick connection test using `GrootClient` |
| [`src/dm_isaac_g1/inference/client.py`](../src/dm_isaac_g1/inference/client.py) | Custom ZeroMQ client (`GrootClient` + `GrootClientAsync`) |
| [`src/dm_isaac_g1/inference/server.py`](../src/dm_isaac_g1/inference/server.py) | GROOT server lifecycle management |
| [`src/dm_isaac_g1/inference/isaac_runner.py`](../src/dm_isaac_g1/inference/isaac_runner.py) | Isaac Sim episode runner for testing |
| [`src/dm_isaac_g1/data/convert_to_groot.py`](../src/dm_isaac_g1/data/convert_to_groot.py) | Training data conversion (`cam_left_high` → `ego_view`) |
| [`docs/INFERENCE_GUIDE.md`](INFERENCE_GUIDE.md) | Server architecture, UNITREE_G1 observation/action format |
| [`docs/SIMULATION_INFERENCE_GUIDE.md`](SIMULATION_INFERENCE_GUIDE.md) | MuJoCo / Isaac Sim / Real Robot evaluation paths |
| [`docs/INFERENCE_SETUP.md`](INFERENCE_SETUP.md) | Spark server setup, Docker, dependencies |
| [`docs/INFERENCE_DEBUGGING.md`](INFERENCE_DEBUGGING.md) | Debugging absolute vs delta actions |

---

## Related Docs

- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) — Server architecture and UNITREE_G1 format details
- [SIMULATION_INFERENCE_GUIDE.md](SIMULATION_INFERENCE_GUIDE.md) — Full evaluation pipeline (MuJoCo, Isaac Sim, Real Robot)
- [INFERENCE_SETUP.md](INFERENCE_SETUP.md) — Spark server setup and Docker configuration
- [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) — How to fine-tune your own model
