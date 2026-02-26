# MuJoCo Evaluation Guide

Custom MuJoCo scenes for evaluating GROOT G1 policies, starting with towel folding.

---

## Architecture

```
┌──────────────────────────┐    ZMQ (5556)     ┌──────────────────────────────┐
│  MuJoCo Simulation       │◄────────────────►│  GROOT Inference Server       │
│  (run_mujoco_towel_eval) │                   │  --use-sim-policy-wrapper     │
│                          │                   │                               │
│  G1 (Menagerie 43 act.)  │                   │  Gr00tSimPolicyWrapper        │
│  + Table + Towel (flex)  │                   │  converts flat keys ↔ nested  │
│  + ego_view camera       │                   │                               │
│  + Fixed base (no WBC)   │                   │  groot-g1-gripper-            │
│                          │                   │  fold-towel-full              │
└──────────────────────────┘                   └──────────────────────────────┘
         │                                               │
         ▼                                               ▼
  Observations (flat keys)                         Actions (flat keys, ALL ABSOLUTE)
  video.ego_view (224x224 RGB)                     action.left_arm  (T,7)
  state.left_arm (7), state.right_arm (7)          action.right_arm (T,7)
  state.waist (3), state.left/right_leg (6)        action.left_hand (T,1)
  state.left/right_hand (1)                        action.right_hand(T,1)
  annotation.human.task_description                action.waist     (T,3)
                                                   action.base_height_command (T,1)
                                                   action.navigate_command    (T,3)
```

Both the simulation and the server run inside the `dm-workstation` container on the Blackwell workstation (192.168.1.205).

**Key design choices:**
- Server uses `--use-sim-policy-wrapper` which wraps the model with `Gr00tSimPolicyWrapper`. This handles flat key ↔ nested dict conversion and is the official NVIDIA approach for sim eval.
- Robot base is **locked** (freejoint removed, pelvis fixed at 90° facing table). Without `gr00t_wbc` whole-body controller, the robot has no balance — fixing the base lets us test arm behavior.
- `ego_view` camera is **injected into torso_link body** at runtime, matching the real G1's Intel RealSense D435 head-mount position.
- All action values from the server are **ABSOLUTE joint targets**. The model internally predicts relative deltas for arms, but `StateActionProcessor.unapply_action()` converts them to absolute positions server-side before returning to the client.

---

## Quick Start

```bash
# SSH into workstation
sshpass -p datamentors ssh datamentors@192.168.1.205
docker exec -it dm-workstation bash

# --- Terminal 1: Start GROOT server with sim wrapper ---
cd /workspace/Isaac-GR00T
python3 -m gr00t.eval.run_gr00t_server \
    --model-path /workspace/checkpoints/groot-g1-gripper-fold-towel-full \
    --embodiment-tag UNITREE_G1 \
    --port 5556 \
    --use-sim-policy-wrapper

# --- Terminal 2: Run MuJoCo eval ---
cd /workspace/dm-isaac-g1/scripts/eval
MUJOCO_GL=egl python3 run_mujoco_towel_eval.py \
    --scene mujoco_towel_scene/g1_gripper_towel_folding.xml \
    --host localhost --port 5556 \
    --n-episodes 5 --max-steps 500 \
    --action-horizon 20 \
    --language "fold the towel"
```

Videos are saved to `/tmp/mujoco_towel_eval/episode_*.mp4`.

**Note:** If `AutoProcessor.from_pretrained` fails with "Unrecognized processing class", copy the processor config to the checkpoint root:
```bash
cp /workspace/checkpoints/groot-g1-gripper-fold-towel-full/processor/processor_config.json \
   /workspace/checkpoints/groot-g1-gripper-fold-towel-full/
cp /workspace/checkpoints/groot-g1-gripper-fold-towel-full/processor/statistics.json \
   /workspace/checkpoints/groot-g1-gripper-fold-towel-full/
cp /workspace/checkpoints/groot-g1-gripper-fold-towel-full/processor/embodiment_id.json \
   /workspace/checkpoints/groot-g1-gripper-fold-towel-full/
```
This is a known issue with `transformers>=4.51` where the custom processor registration changed.

---

## File Map

```
scripts/eval/
├── setup_eval_workstation.sh              # One-time setup (WBC, datasets, deps)
├── run_mujoco_towel_eval.py               # Closed-loop MuJoCo eval script
└── mujoco_towel_scene/
    ├── g1_towel_folding.xml               # Scene: G1 (no hands) + table + towel
    ├── g1_gripper_towel_folding.xml       # Scene: G1 (with_hands) + table + towel
    └── setup_scene.sh                     # Clone Menagerie + verify scene
```

| File | Purpose |
|------|---------|
| `setup_eval_workstation.sh` | Installs MuJoCo 3.2.6, clones Menagerie, downloads datasets, runs sanity checks |
| `run_mujoco_towel_eval.py` | Main eval loop: loads scene, connects to GROOT via PolicyClient, runs episodes, records video |
| `g1_towel_folding.xml` | MuJoCo MJCF scene with G1 29-DOF (no grippers), table, and flexcomp cloth towel |
| `g1_gripper_towel_folding.xml` | MuJoCo MJCF scene with G1 43-DOF (with Inspire hands), table, and towel |
| `setup_scene.sh` | Clones MuJoCo Menagerie, creates symlinks, verifies scene loads |

---

## Towel Scene Details

### Scene Composition

The scene (`g1_towel_folding.xml`) combines:

1. **Unitree G1 robot** from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) — either `g1.xml` (29 act) or `g1_with_hands.xml` (43 act, Inspire fingers)
2. **Table** — box geom at (0, 0.55, 0.74), sized 0.9m x 0.7m
3. **Deformable towel** — MuJoCo 3.0+ `flexcomp` with `mujoco.elasticity.shell` plugin
4. **Cameras** — `ego_view` (injected into torso_link at load time, matching D435 head-mount) and `overview` (fixed world camera for debug video)

### Towel (Cloth) Configuration

```xml
<flexcomp name="towel" type="grid" count="16 16 1" spacing="0.02 0.02 0.02"
          dim="2" mass="0.15" radius="0.005">
  <contact condim="3" selfcollide="auto" friction="0.8 0.3 0.1"/>
  <edge equality="true" damping="0.002"/>
  <plugin plugin="mujoco.elasticity.shell">
    <config key="poisson" value="0.3"/>
    <config key="young" value="5e3"/>
    <config key="thickness" value="0.002"/>
  </plugin>
</flexcomp>
```

| Parameter | Value | Effect |
|-----------|-------|--------|
| `count` | 16 16 1 | 16x16 grid = 256 nodes |
| `spacing` | 0.02 | 2 cm between nodes = ~30 cm x 30 cm towel |
| `mass` | 0.15 | 150 grams total |
| `young` | 5e3 | Young's modulus (lower = floppier) |
| `poisson` | 0.3 | Poisson ratio |
| `thickness` | 0.002 | 2 mm cloth thickness |

Tune `young` (1e2 to 1e5) to control stiffness. Lower values make the towel drape more realistically.

### Scene Composition at Load Time

The `load_towel_scene()` function performs several XML manipulations before compilation:

1. **Strip keyframes** — The Menagerie G1 `stand` keyframe has 36 qpos, but flexcomp towel increases qpos to ~811. Keyframe is removed to avoid size conflict.
2. **Lock the base** — Removes `<freejoint name="floating_base_joint"/>` so the pelvis is fixed to world. Without `gr00t_wbc`, the freejoint robot has no balance controller and falls.
3. **Rotate to face table** — Sets `quat="0.7071 0 0 0.7071"` on the pelvis body (90° around Z, default G1 faces +X, table is at +Y).
4. **Inject ego_view camera** — Adds a camera into the `torso_link` body matching the real G1's Intel RealSense D435 position: `pos="0.0576 0.0175 0.4299"` with 60° pitch downward.
5. **Write temp XML** — Compiles from a temporary file in the Menagerie directory (for mesh path resolution), then cleans up.

---

## Wrist Joint Ordering (CRITICAL)

The GROOT training data and MuJoCo use **different wrist joint ordering**. This affects both state observations sent to the model and actions applied to the robot.

### The Mismatch

| Index (per arm, after elbow) | Training (expected) | MuJoCo Menagerie | Isaac Sim (was also wrong) |
|:---:|:---:|:---:|:---:|
| 4 | `wrist_yaw` | `wrist_roll` | `wrist_roll` |
| 5 | `wrist_roll` | `wrist_pitch` | `wrist_pitch` |
| 6 | `wrist_pitch` | `wrist_yaw` | `wrist_yaw` |

### The Fix

`run_mujoco_towel_eval.py` handles this in `build_joint_name_to_state_index()`:

```python
# Left wrist: MuJoCo has roll(20), pitch(21), yaw(22) in joint list
# Training expects: yaw=19, roll=20, pitch=21
("left_wrist_yaw_joint",   19),  # Menagerie joint[22] -> state[19]
("left_wrist_roll_joint",  20),  # Menagerie joint[20] -> state[20]
("left_wrist_pitch_joint", 21),  # Menagerie joint[21] -> state[21]
```

The same remapping is applied for right wrist and in `apply_actions()` for action output.

### Verification

```python
# Run on workstation to verify mapping:
docker exec dm-workstation python3 -c "
import sys; sys.path.insert(0, '/workspace/dm-isaac-g1/scripts/eval')
from run_mujoco_towel_eval import load_towel_scene, build_joint_name_to_state_index
import mujoco
m = load_towel_scene('/workspace/dm-isaac-g1/scripts/eval/mujoco_towel_scene/g1_towel_folding.xml')
mapping = build_joint_name_to_state_index(m)
for name in ['left_wrist_yaw_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint']:
    mj_id, st_idx = mapping[name]
    print(f'{name}: mujoco_joint={mj_id} -> state[{st_idx}]')
"
# Expected output:
#   left_wrist_yaw_joint: mujoco_joint=22 -> state[19]   (yaw first)
#   left_wrist_roll_joint: mujoco_joint=20 -> state[20]
#   left_wrist_pitch_joint: mujoco_joint=21 -> state[21]
```

---

## UNITREE_G1 DOF Layout Reference

### State (31 DOF)

| Group | Indices | DOF | Joints |
|-------|---------|-----|--------|
| left_leg | 0-5 | 6 | hip pitch/roll/yaw, knee, ankle pitch/roll |
| right_leg | 6-11 | 6 | same |
| waist | 12-14 | 3 | yaw, roll, pitch |
| left_arm | 15-21 | 7 | shoulder pitch/roll/yaw, elbow, wrist **yaw/roll/pitch** |
| right_arm | 22-28 | 7 | same |
| left_hand | 29 | 1 | gripper (0=open, 1=closed) |
| right_hand | 30 | 1 | gripper |

### Action (23 DOF)

| Group | Indices | DOF | Type |
|-------|---------|-----|------|
| left_arm | 0-6 | 7 | RELATIVE (delta from current) |
| right_arm | 7-13 | 7 | RELATIVE |
| left_hand | 14 | 1 | ABSOLUTE |
| right_hand | 15 | 1 | ABSOLUTE |
| waist | 16-18 | 3 | ABSOLUTE |
| base_height | 19 | 1 | ABSOLUTE (not used in fixed-base MuJoCo) |
| navigate | 20-22 | 3 | ABSOLUTE (not used in fixed-base MuJoCo) |

### Gripper Note

The `g1_gripper_towel_folding.xml` scene uses `g1_with_hands.xml` (43 actuators: 29 body + 14 finger). The 7-DOF Inspire fingers per hand are mapped to a single gripper value:
- **State**: `_get_finger_curl()` computes average normalized finger curl (0=open, 1=closed)
- **Action**: `_apply_gripper_to_fingers()` maps the single gripper command to all 7 finger joints via linear interpolation across each joint's range

The `g1_towel_folding.xml` scene uses `g1.xml` (29 actuators, no grippers). State indices 29-30 default to 0.

---

## Evaluation Modes

### 1. Open-Loop Eval (Offline, No Server)

Compares model predictions against ground truth from a dataset. No simulation needed.

```bash
cd /workspace/Isaac-GR00T
python3 gr00t/eval/open_loop_eval.py \
    --model-path /workspace/checkpoints/groot-g1-gripper-fold-towel-full \
    --dataset-path /workspace/datasets/groot/G1_Fold_Towel \
    --embodiment-tag UNITREE_G1 \
    --steps 300 \
    --traj_ids 0 1 2
```

Can also use the X-Embodiment dataset already on the workstation:

```bash
python3 gr00t/eval/open_loop_eval.py \
    --model-path /workspace/checkpoints/groot-g1-gripper-fold-towel-full \
    --dataset-path /workspace/Isaac-GR00T/datasets/gr00t_x_embodiment/unitree_g1.LMPnPAppleToPlateDC \
    --embodiment-tag UNITREE_G1 \
    --steps 200 --traj_ids 0 1 2
```

Output: MSE/MAE metrics + trajectory comparison plots at `/tmp/open_loop_eval/traj_*.jpeg`.

### 2. MuJoCo Closed-Loop Eval (Custom Towel Scene)

Full interaction with physics simulation. Requires GROOT server with `--use-sim-policy-wrapper`.

```bash
# Server (terminal 1):
cd /workspace/Isaac-GR00T
python3 -m gr00t.eval.run_gr00t_server \
    --model-path /workspace/checkpoints/groot-g1-gripper-fold-towel-full \
    --embodiment-tag UNITREE_G1 --port 5556 --use-sim-policy-wrapper

# Eval (terminal 2):
cd /workspace/dm-isaac-g1/scripts/eval
MUJOCO_GL=egl python3 run_mujoco_towel_eval.py \
    --scene mujoco_towel_scene/g1_gripper_towel_folding.xml \
    --host localhost --port 5556 \
    --n-episodes 5 --max-steps 500 --action-horizon 20
```

### 3. MuJoCo WBC Eval (NVIDIA PnP Apple to Plate)

Uses the official GR00T-WholeBodyControl loco-manipulation environment with full balance control. Requires WBC venv setup (separate from main env due to conflicting deps).

```bash
# Setup (one-time) — creates a separate uv venv with robosuite + gr00t_wbc:
cd /workspace/Isaac-GR00T
git submodule update --init external_dependencies/GR00T-WholeBodyControl
bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh

# Terminal 1: Server (uses main env, NOT WBC venv):
cd /workspace/Isaac-GR00T
python3 -m gr00t.eval.run_gr00t_server \
    --model-path /workspace/checkpoints/groot-g1-gripper-fold-towel-full \
    --embodiment-tag UNITREE_G1 --port 5556 --use-sim-policy-wrapper

# Terminal 2: Rollout (uses WBC venv):
source gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/activate

python3 gr00t/eval/rollout_policy.py \
    --n_episodes 10 --max_episode_steps 1440 \
    --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \
    --policy_client_host localhost --policy_client_port 5556 \
    --n_action_steps 20 --n_envs 1
```

**Note:** The WBC pipeline provides proper balance via `WholeBodyControlWrapper`. The `base_height_command` and `navigate_command` actions are consumed by WBC, while our fixed-base MuJoCo eval ignores them.

---

## `run_mujoco_towel_eval.py` Reference

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--scene` | (required) | Path to MuJoCo scene XML |
| `--host` | `localhost` | GROOT server host |
| `--port` | `5556` | GROOT server port (must use `--use-sim-policy-wrapper`) |
| `--language` | `"fold the towel"` | Task language instruction |
| `--n-episodes` | `5` | Number of evaluation episodes |
| `--max-steps` | `500` | Max steps per episode |
| `--action-horizon` | `20` | Steps to execute per 30-step action chunk (NVIDIA uses 20) |
| `--output-dir` | `/tmp/mujoco_towel_eval` | Output directory for videos |
| `--render-width` | `640` | Render width for output video |
| `--render-height` | `480` | Render height for output video |
| `--no-video` | `false` | Disable video recording |

### Key Functions

| Function | Purpose |
|----------|---------|
| `load_towel_scene()` | Loads scene XML, strips keyframe, locks base, rotates pelvis, injects ego_view camera |
| `build_joint_name_to_state_index()` | Maps MuJoCo joint names to UNITREE_G1 state indices with wrist remapping |
| `get_state_vector()` | Extracts 31-DOF state vector from MuJoCo `data.qpos` |
| `build_groot_observation()` | Builds flat-key observation dict for `Gr00tSimPolicyWrapper` |
| `decode_action_dict()` | Decodes flat action keys (`action.left_arm`, etc.) into per-group arrays |
| `apply_actions()` | Applies per-group action dict to MuJoCo actuators (all ABSOLUTE targets from server) |
| `render_ego_view()` | Renders ego_view camera at 224x224 for GROOT input |
| `run_episode()` | Runs a single evaluation episode with action chunking |
| `_get_finger_curl()` | Computes average normalized finger curl for gripper state |
| `_apply_gripper_to_fingers()` | Maps single gripper command to all Inspire finger joints |

---

## Creating Custom Scenes

### Requirements

- **MuJoCo 3.0+** for `flexcomp` deformable bodies (we use 3.2.6)
- **MuJoCo Menagerie** for the G1 MJCF model
- The `mujoco.elasticity.shell` plugin for cloth simulation

### Template

To create a new custom scene (e.g., table wiping, fruit preparation):

1. Copy `g1_towel_folding.xml` as a starting template
2. Replace the towel `<flexcomp>` with your objects:
   - Rigid objects: standard `<body><geom .../></body>`
   - Deformable objects: `<flexcomp>` with appropriate plugin
   - Articulated objects: `<body>` with `<joint>` children
3. Adjust camera positions to match the task viewpoint
4. Update `--language` when running the eval

### Object Examples

**Rigid object (apple):**
```xml
<body name="apple" pos="0 0.55 0.80">
  <joint type="free"/>
  <geom type="sphere" size="0.04" mass="0.15" rgba="0.8 0.1 0.1 1"
        condim="3" friction="0.6"/>
</body>
```

**Plate:**
```xml
<body name="plate" pos="0.15 0.55 0.76">
  <geom type="cylinder" size="0.12 0.01" mass="0.3" rgba="0.9 0.9 0.9 1"/>
</body>
```

**Soft sponge (3D deformable):**
```xml
<extension>
  <plugin plugin="mujoco.elasticity.solid"/>
</extension>
<body name="sponge" pos="0 0.55 0.80">
  <flexcomp name="sponge" type="grid" count="6 4 3" spacing="0.015 0.015 0.015"
            dim="3" mass="0.05" radius="0.008">
    <plugin plugin="mujoco.elasticity.solid">
      <config key="poisson" value="0.3"/>
      <config key="young" value="1e3"/>
    </plugin>
  </flexcomp>
</body>
```

---

## Dependencies Installed on Workstation

The following were installed in the `dm-workstation` container (system Python 3.13) in addition to the base Isaac Sim + GROOT packages:

| Package | Version | Purpose |
|---------|---------|---------|
| `mujoco` | 3.2.6 | MuJoCo physics engine + flexcomp cloth |
| `torchvision` | 0.25.0 | Required by GROOT video utils |
| `pandas` | 3.0.1 | Required by LeRobotEpisodeLoader |
| `matplotlib` | 3.10.8 | Trajectory plotting for open_loop_eval |
| `av` | 15.0.0 | Video decoding for LeRobot datasets |
| `datasets` | 4.6.0 | HuggingFace datasets loading |
| `diffusers` | 0.36.0 | Diffusion action head |
| `peft` | 0.18.1 | Parameter-efficient fine-tuning |
| `scipy` | 1.17.1 | Scientific computing |
| `opencv-python-headless` | 4.13.0 | Image/video processing |

These are currently installed manually. They will be baked into the Dockerfile on the next image build (see below).

---

## Dockerfile & Environment Updates (Next Build)

The following changes will be included in the next `docker compose build` to make MuJoCo eval available out of the box, removing the need for manual `pip install` after container creation.

### Dockerfile Changes

Three additions to `environments/workstation/Dockerfile`:

**1. MuJoCo + Menagerie installation** (after the IsaacLab section):

```dockerfile
# ==========================================
# MuJoCo 3.x (for custom scene evaluation)
# ==========================================
# Required for:
#   - Custom G1 scenes with deformable objects (flexcomp towels, sponges)
#   - GR00T-WholeBodyControl loco-manipulation eval
#   - Standalone policy validation without Isaac Sim
RUN pip install mujoco==3.2.6

# Clone MuJoCo Menagerie (curated MJCF robot models including G1)
RUN git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git \
    /workspace/mujoco_menagerie

# Set EGL rendering for headless MuJoCo
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
```

**2. PYTHONPATH update** (add MuJoCo eval scripts):

```dockerfile
ENV PYTHONPATH="/workspace/dm-isaac-g1/scripts/eval:...existing paths...:${PYTHONPATH}"
```

**3. Volume mount for Menagerie persistence** in `docker-compose.yml`:

```yaml
volumes:
  # ... existing mounts ...
  # MuJoCo Menagerie (persistent, avoid re-cloning)
  - mujoco_menagerie:/workspace/mujoco_menagerie
```

### requirements-groot.txt Addition

```
# MuJoCo evaluation
mujoco==3.2.6
opencv-python-headless>=4.8.0
```

---

## Available Datasets

| Dataset | Location | Episodes | Use |
|---------|----------|----------|-----|
| X-Embodiment PnP | `/workspace/Isaac-GR00T/datasets/gr00t_x_embodiment/unitree_g1.LMPnPAppleToPlateDC` | 103 | Open-loop eval (on workstation) |
| G1_Fold_Towel | `unitreerobotics/G1_Fold_Towel` on HuggingFace | 200 | Download for open-loop eval |
| G1_Clean_Table | `unitreerobotics/G1_Clean_Table` on HuggingFace | 200 | Download for open-loop eval |

### Available Checkpoints

| Checkpoint | Location | Tasks |
|------------|----------|-------|
| `groot-g1-gripper-fold-towel-full` | `/workspace/checkpoints/groot-g1-gripper-fold-towel-full` | Fold towel (200 eps, 10k steps) |
| `groot-g1-gripper-hospitality-7ds` | Download from HuggingFace | All 7 hospitality tasks (1400 eps) |

---

## Troubleshooting

### Scene fails to load: "invalid qpos size"

The Menagerie G1 keyframe has 36 qpos but flexcomp objects add more. Use `load_towel_scene()` from `run_mujoco_towel_eval.py` which strips the keyframe automatically.

### Scene fails to load: "No such file: pelvis.STL"

Mesh resolution issue. The G1 model expects meshes relative to its own directory. Either:
- Symlink the scene into `/workspace/mujoco_menagerie/unitree_g1/`
- Use `load_towel_scene()` which handles this

### "edge stiffness only available for dim=1"

Remove `stiffness` from `<edge>` in the flexcomp. For 2D cloth, stiffness is controlled by the `mujoco.elasticity.shell` plugin (`young` parameter).

### Robot falls through floor

Set initial qpos with the free joint z-position at ~0.79 (standing height). The `stand` keyframe from Menagerie uses `qpos[2] = 0.79`.

### Wrist motions look wrong

Verify the wrist remapping is active. Check that `left_wrist_yaw_joint` maps to state index 19, not 22. Run the verification script in the "Wrist Joint Ordering" section above.

---

## Related Documentation

- [SIMULATION_INFERENCE_GUIDE.md](SIMULATION_INFERENCE_GUIDE.md) — Overview of all 3 eval methods (MuJoCo, Isaac Sim, Real Robot)
- [INFERENCE_EXAMPLES.md](INFERENCE_EXAMPLES.md) — GROOT observation/action format examples
- [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) — Training new models
- [TEAM_MODEL_SUMMARY.md](TEAM_MODEL_SUMMARY.md) — All deployed models and datasets
