# UNITREE_G1 Gripper Inference — Changes Summary

## Overview

Added UNITREE_G1 gripper (Dex1) support to the Isaac Sim inference pipeline. The GROOT server on Spark serves the `groot-g1-gripper-fold-towel-full` checkpoint (trained on 7 hospitality datasets: fold towel, clean table, wipe table, etc.) using the `UNITREE_G1` embodiment tag.

**Commits:** `856a017` → `e760206` (5 commits on `main`)

---

## Files Changed

### 1. `scripts/test_groot_unitree_g1.py` (NEW)

Standalone test script to validate GROOT server communication **without Isaac Sim**. Sends synthetic UNITREE_G1 observations and prints the action response.

**What it does:**
- Connects to the GROOT server via ZMQ
- Sends a synthetic observation in nested dict format (the format the server expects)
- Prints action shapes, ranges, and first-timestep values
- Optionally loads `statistics.json` to use training-mean joint positions

**Usage:**
```bash
# From dm-workstation container:
PYTHONPATH=/workspace/Isaac-GR00T:$PYTHONPATH \
python scripts/test_groot_unitree_g1.py \
    --server 192.168.1.237:5555 \
    --language "fold the towel"

# Via Tailscale (when not on office LAN):
PYTHONPATH=/workspace/Isaac-GR00T:$PYTHONPATH \
python scripts/test_groot_unitree_g1.py \
    --server 100.69.71.21:5555

# With statistics file for realistic joint positions:
GR00T_STATS=/workspace/checkpoints/groot-g1-gripper-fold-towel-full/processor/statistics.json \
PYTHONPATH=/workspace/Isaac-GR00T:$PYTHONPATH \
python scripts/test_groot_unitree_g1.py \
    --server 192.168.1.237:5555
```

**Expected output (success):**
```
[INFO] Server modality config keys: ['video', 'state', 'action', 'language']
  video: keys=['ego_view'], delta_indices=[0]
  state: keys=['left_leg', 'right_leg', 'waist', 'left_arm', 'right_arm', 'left_hand', 'right_hand'], delta_indices=[0]
  action: keys=['left_arm', 'right_arm', 'left_hand', 'right_hand', 'waist', 'base_height_command', 'navigate_command'], delta_indices=[0..29]

[INFO] Request 1/3 — 0.120s
  left_arm: shape=(1, 30, 7), range=[-0.9197, 0.5136]
  right_arm: shape=(1, 30, 7), range=[-0.9197, 0.5136]
  left_hand: shape=(1, 30, 1), range=[0.0000, 1.0000]
  right_hand: shape=(1, 30, 1), range=[0.0000, 1.0000]
  waist: shape=(1, 30, 3)
  base_height_command: shape=(1, 30, 1)
  navigate_command: shape=(1, 30, 3)
```

---

### 2. `scripts/inference_setups.py` (MODIFIED)

Added two new inference setups for UNITREE_G1 gripper:

| Setup | Scene | Description |
|-------|-------|-------------|
| `gripper_unitree` | `pickplace_g1_dex1` | Fold towel / hospitality tasks with Dex1 gripper hands |
| `gripper_redblock` | `pickplace_redblock_g1_dex1` | Red block pick-and-place variant |

**Key config details:**
- **Camera:** 1x `ego_view` (d435_link head camera) — NOT `cam_left_high`
- **State:** 31 DOF — left_leg(6) + right_leg(6) + waist(3) + left_arm(7) + right_arm(7) + left_hand(1) + right_hand(1)
- **Action:** 23 DOF — arms(14) RELATIVE deltas, grippers(2)+waist(3)+base_height(1)+nav(3) ABSOLUTE (per `embodiment_configs.py`)
- **Scene:** Uses `pickplace_g1_dex1` (Dex1 robot USD with actual gripper joints: `left_hand_Joint1_1`, `right_hand_Joint1_1`)
- **hand_type:** `"gripper"` — triggers UNITREE_G1 observation/action format

---

### 3. `scripts/policy_inference_groot_g1.py` (MODIFIED)

Major changes to support the UNITREE_G1 embodiment alongside existing Dex3/Inspire:

#### a) Dex1 scenes added to `AVAILABLE_SCENES`
```python
"pickplace_g1_dex1": "tasks.g1_tasks.pick_place_cylinder_g1_29dof_dex1...PickPlaceG129DEX1BaseFixEnvCfg"
"pickplace_redblock_g1_dex1": "tasks.g1_tasks.pick_place_redblock_g1_29dof_dex1...PickPlaceG129DEX1BaseFixEnvCfg"
```

#### b) Nested dict observation format for ALL embodiments
The GROOT server runs **without** `--use-sim-policy-wrapper`, so it expects nested dicts:
```python
# Video
{"video": {"ego_view": (B, T, H, W, 3)}}

# State (UNITREE_G1 — split body parts)
{"state": {"left_arm": (B, T, 7), "right_arm": (B, T, 7), ...}}

# State (new_embodiment — concatenated)
{"state": {"observation.state": (B, T, 28_or_53)}}

# Language
{"language": {"annotation.human.task_description": [["fold the towel"]]}}
```

#### c) Mixed RELATIVE/ABSOLUTE action handling
UNITREE_G1 actions use mixed representation as defined in the **authoritative source**:
`Isaac-GR00T/gr00t/configs/data/embodiment_configs.py` → `MODALITY_CONFIGS["unitree_g1"]["action"]`

| Action Group | `ActionRepresentation` | Meaning |
|---|---|---|
| `left_arm` (7 DOF) | **RELATIVE** | Deltas from trajectory start joint position |
| `right_arm` (7 DOF) | **RELATIVE** | Deltas from trajectory start joint position |
| `left_hand` (1 DOF) | ABSOLUTE | Direct gripper target (binary-like) |
| `right_hand` (1 DOF) | ABSOLUTE | Direct gripper target (binary-like) |
| `waist` (3 DOF) | ABSOLUTE | Direct joint target |
| `base_height_command` (1 DOF) | ABSOLUTE | Direct height target |
| `navigate_command` (3 DOF) | ABSOLUTE | Direct navigation target |

**How RELATIVE arms work in the inference script:**
```
target_joint_pos = trajectory_start_pos + model_delta[t] * action_scale
```
Where `trajectory_start_pos` is captured once per inference call (every 30 steps).

#### d) DDS bypass for standalone inference
The unitree_sim_isaaclab scenes (dex1, dex3, inspire) have observation terms that use DDS (hardware communication middleware). These hang in pure inference mode without a physical G1. The script now replaces scene observations/rewards/terminations with a minimal config:
```python
env_cfg.observations = _MinimalObsCfg()  # No DDS
env_cfg.rewards = None
env_cfg.terminations = None
env_cfg.events = None
```

#### e) Training mean from split body-part stats
UNITREE_G1 statistics.json stores per-body-part means (not a single concatenated vector). The script concatenates them in the correct order for initial pose setup.

**Important note on `statistics.json` sections:**
- `state.<part>` — raw joint position statistics (ABSOLUTE positions)
- `action.<part>` — raw action statistics used for normalization (values are in ABSOLUTE space for arms too)
- `relative_action.<part>` — relative delta statistics (tiny values ~0.005 rad for arms)

The `action.left_arm` means look similar to `state.left_arm` because the raw training data stores absolute positions. The GROOT model internally handles the absolute→relative conversion during training and relative→absolute during inference. The server returns **relative deltas** for arms after unnormalization.

---

## Server Details

**GROOT server on Spark:**
- Model: `groot-g1-gripper-fold-towel-full` (trained on 7 Unitree hospitality datasets)
- Embodiment tag: `UNITREE_G1`
- Port: 5555
- Command: `python gr00t/eval/run_gr00t_server.py --model-path /workspace/checkpoints/groot-g1-gripper-hospitality-7ds --embodiment-tag UNITREE_G1 --port 5555`
- **No** `--use-sim-policy-wrapper` flag (expects nested dicts, not flat keys)

**Network access:**
- Office LAN: `192.168.1.237:5555`
- Tailscale: `100.69.71.21:5555` (node: `spark-7112-1`)

---

## How to Run Isaac Sim Inference

```bash
# From dm-workstation container:
export DISPLAY=:1
export PROJECT_ROOT=/workspace/unitree_sim_isaaclab
export GR00T_STATS=/workspace/checkpoints/groot-g1-gripper-fold-towel-full/processor/statistics.json
export PYTHONPATH=/workspace/dm-isaac-g1/src:/workspace/Isaac-GR00T:$PYTHONPATH

cd /workspace/dm-isaac-g1
conda run --no-capture-output -n unitree_sim_env python scripts/policy_inference_groot_g1.py \
    --server 192.168.1.237:5555 \
    --setup gripper_unitree \
    --num_action_steps 30 \
    --enable_cameras \
    --save_debug_frames
```

**Key flags:**
- `--setup gripper_unitree` — selects the UNITREE_G1 gripper config
- `--enable_cameras` — required for camera image capture
- `--save_debug_frames` — saves camera frames + observation JSON to `/tmp/groot_debug/`
- `--control_decimation N` — hold each action for N sim steps to match training frequency

**Control frequency alignment:**
The Dex1 scene runs at dt=0.005 with decimation=2 → **100Hz control rate**.
UNITREE_G1 training data was collected at **30Hz**.
Use `--control_decimation 3` to get 100Hz/3 ≈ 33Hz (closest integer match to 30Hz).
Without this flag (default 1), each action step runs at 100Hz which is 3.3x faster than training.

**VNC:** Connect to `vnc://192.168.1.205:5901` (password: `datament`) to view the simulation.

---

## Architecture Note

The current `env.step()` approach drives the sim robot directly with GROOT predictions. This is useful for **visualization** but is **not a validated sim2real path** — the sim dynamics don't match the real G1's low-level controller.

For **real hardware deployment**, the correct architecture is:
```
Camera (G1) → GROOT inference → DDS publish → G1 low-level controller
```
The DDS integration (publishing GROOT actions to the G1's subscription topics) is a next step.

---

## Official GROOT Evaluation Scripts

Isaac-GR00T provides two evaluation tools (at `/workspace/Isaac-GR00T/gr00t/eval/`):

### 1. Open-Loop Eval — `open_loop_eval.py`

Evaluates model predictions against recorded trajectories from a LeRobot dataset.
Generates per-joint plots comparing ground truth vs predicted actions, plus MSE/MAE metrics.

```bash
# Against the GROOT server (no local GPU needed for inference):
cd /workspace/Isaac-GR00T
python gr00t/eval/open_loop_eval.py \
    --host 192.168.1.237 \
    --port 5555 \
    --dataset_path /path/to/unitree_g1_dataset \
    --embodiment_tag UNITREE_G1 \
    --traj_ids 0 1 2 3 4 \
    --steps 200 \
    --action_horizon 30 \
    --save_plot_path /tmp/open_loop_eval/

# Or with local model checkpoint (requires GPU):
python gr00t/eval/open_loop_eval.py \
    --model_path /workspace/checkpoints/groot-g1-gripper-fold-towel-full \
    --embodiment_tag UNITREE_G1 \
    --dataset_path /path/to/unitree_g1_dataset \
    --traj_ids 0 1 2 \
    --action_horizon 30
```

**Output:** JPEG plots at `--save_plot_path` (default `/tmp/open_loop_eval/traj_{id}.jpeg`).
Plots show state joints, ground truth actions, and predicted actions over time per DOF.

### 2. Rollout Eval — `rollout_policy.py`

Runs live policy rollouts in parallel gymnasium environments. **Generates MP4 videos** with success rate metrics.

```bash
# Terminal 1 — Start server WITH --use-sim-policy-wrapper:
python gr00t/eval/run_gr00t_server.py \
    --model_path /workspace/checkpoints/groot-g1-gripper-fold-towel-full \
    --embodiment_tag UNITREE_G1 \
    --use_sim_policy_wrapper

# Terminal 2 — Run rollout evaluation:
python gr00t/eval/rollout_policy.py \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \
    --n_episodes 10 \
    --max_episode_steps 1440 \
    --n_action_steps 20 \
    --n_envs 5
```

**Output:** MP4 videos at `/tmp/sim_eval_videos_{model_name}_ac{n_action_steps}_{uuid}/`
Each video shows multi-camera views with task description overlay.
Filename contains success indicator: `{uuid}_s1.mp4` (success) or `{uuid}_s0.mp4` (failure).

**Important:** Rollout eval requires `gr00t_wbc` package (Whole-Body Control) for UNITREE_G1.
One-time setup:
```bash
apt-get install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
```

### Running eval on the 7-dataset hospitality model

For our `groot-g1-gripper-fold-towel-full` checkpoint:

1. **Open-loop eval** is the quickest — needs a training dataset in LeRobot format.
   Point `--dataset_path` to one of the 7 hospitality datasets and `--model_path` to the checkpoint.

2. **Rollout eval** requires the WBC sim environment. The official benchmark task is
   `gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc` (expected ~58% success rate ±15%).

---

## Quick Reference

| What | Where |
|------|-------|
| Test GROOT server (no sim) | `scripts/test_groot_unitree_g1.py` |
| Isaac Sim inference | `scripts/policy_inference_groot_g1.py --setup gripper_unitree` |
| Setup configs | `scripts/inference_setups.py` (see `gripper_unitree`, `gripper_redblock`) |
| GROOT open-loop eval | `/workspace/Isaac-GR00T/gr00t/eval/open_loop_eval.py` |
| GROOT rollout eval (videos) | `/workspace/Isaac-GR00T/gr00t/eval/rollout_policy.py` |
| Embodiment config (authoritative) | `/workspace/Isaac-GR00T/gr00t/configs/data/embodiment_configs.py` |
| Training config reference | `src/dm_isaac_g1/finetuning/configs/g1_gripper_unitree.py` |
| Statistics file | `/workspace/checkpoints/groot-g1-gripper-fold-towel-full/processor/statistics.json` |
| Debug frames | `/tmp/groot_debug/` (when `--save_debug_frames` is used) |
| Inference log | `/tmp/inference_gripper.log` (when launched via VNC command) |
