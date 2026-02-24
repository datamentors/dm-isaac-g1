# Simulation & Inference Evaluation Guide

Evaluating fine-tuned GR00T N1.6 models for the Unitree G1 (UNITREE_G1 gripper embodiment) in simulation and on real hardware.

## Overview

There are three ways to evaluate a fine-tuned GR00T model:

| Method | Simulator | Robot | Use Case | Complexity |
|--------|-----------|-------|----------|------------|
| **MuJoCo (WholeBodyControl)** | MuJoCo 3.x | Unitree G1 gripper | Loco-manipulation eval, quick validation | Low (out-of-the-box) |
| **Isaac Sim (Isaac Lab)** | Isaac Sim 5.x | Unitree G1 gripper | Full physics sim, RL, domain randomization | Medium (needs env adaptation) |
| **Real Robot** | — | Unitree G1 EDU 2 | Production deployment | High |

All three connect to the same **GROOT inference server** via ZeroMQ (port 5555).

```
┌──────────────────────┐     ZMQ (port 5555)     ┌──────────────────────┐
│   Simulation Client  │◄───────────────────────►│  GROOT Inference     │
│  (MuJoCo / Isaac Sim │                         │  Server (Spark)      │
│   / Real Robot)      │                         │  192.168.1.237       │
└──────────────────────┘                         └──────────────────────┘
        │                                                │
        ▼                                                ▼
   Observations                                    Fine-tuned
   (ego_view + 31 DOF state)                       GR00T Model
        │                                                │
        └──────────► Actions (23 DOF) ◄──────────────────┘
                  (30-step trajectory)
```

---

## Option 1: MuJoCo Evaluation (Recommended Starting Point)

The Isaac-GR00T repo ships with a **ready-to-use MuJoCo evaluation** for the Unitree G1 via the GR00T-WholeBodyControl example.

### Available Scene

| Environment | Task | Baseline Success Rate |
|-------------|------|----------------------|
| `gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc` | Navigate to apple, pick up, place on plate | ~58% (+/-15%) |

This is the **only pre-built G1 MuJoCo scene** out of the box. It uses:
- **G1_29dof** gripper model (matches UNITREE_G1)
- **Decoupled WBC**: RL-trained lower body (locomotion) + IK upper body (manipulation)
- **500 Hz** simulation with 2ms timestep

### Setup

```bash
# Inside dm-workstation container on workstation (192.168.1.205)
cd /workspace/Isaac-GR00T

# One-time setup: install WholeBodyControl dependencies
bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
```

### Running Evaluation

**Terminal 1 — Start GROOT inference server:**

```bash
# On Spark (192.168.1.237) — server is already running
# Or start locally on workstation:
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /workspace/checkpoints/groot-g1-gripper-hospitality-7ds \
    --embodiment-tag UNITREE_G1 \
    --port 5555 \
    --use-sim-policy-wrapper
```

The `--use-sim-policy-wrapper` flag is **required** for simulation — it enables the `Gr00tSimPolicyWrapper` which handles sim-specific action/observation transformations.

**Terminal 2 — Run evaluation rollout:**

```bash
uv run python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --max_episode_steps 1440 \
    --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \
    --n_action_steps 20 \
    --n_envs 5
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--n_episodes` | 10 | Number of evaluation episodes |
| `--max_episode_steps` | 1440 | Max steps per episode |
| `--n_action_steps` | 20 | Steps to execute from each action chunk before re-querying |
| `--n_envs` | 5 | Parallel environments |

### Validate Pipeline First (ReplayPolicy)

Before testing your model, validate the observation/action pipeline using ground-truth dataset actions:

```bash
# Replays recorded actions without model inference
uv run python gr00t/eval/run_gr00t_server.py \
    --dataset-path /workspace/datasets/groot/G1_Fold_Towel \
    --embodiment-tag UNITREE_G1 \
    --execution-horizon 8
```

### NVIDIA Pre-trained Checkpoint

NVIDIA provides `nvidia/GR00T-N1.6-G1-PnPAppleToPlate` on HuggingFace for this scene. However, there is a [known issue (#485)](https://github.com/NVIDIA/Isaac-GR00T/issues/485) where it fails to load. Use our fine-tuned checkpoints instead.

---

## Option 2: Isaac Sim / Isaac Lab Evaluation

Isaac Sim provides full physics simulation with photo-realistic rendering, domain randomization, and GPU-accelerated environments. The **Unitree official Isaac Lab repo** provides ready-to-use G1 scenes.

### Unitree Isaac Lab Scenes (Out-of-the-Box)

The [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) repo is the **primary source** for G1 manipulation scenes. It's already mounted on the workstation at `/workspace/unitree_sim_isaaclab`.

**G1 Gripper Manipulation Tasks** (directly usable with our UNITREE_G1 models):

| Scene | Task | Hand | Ready? |
|-------|------|------|--------|
| `pick_place_cylinder_g1_29dof_gripper` | Pick cylinder, place at target | Gripper (1 DOF) | Yes |
| `pick_place_redblock_g1_29dof_gripper` | Pick red block, place at target | Gripper (1 DOF) | Yes |
| `wholebody_g1_29dof_gripper` | Mobile manipulation (walk + pick) | Gripper (1 DOF) | Yes |

**G1 Dex3 Manipulation Tasks** (3-finger hands, different embodiment):

| Scene | Task | Hand |
|-------|------|------|
| `pick_place_cylinder_g1_29dof_dex3` | Pick cylinder with dexterous hand | Dex3 (3-finger) |
| `pick_place_redblock_g1_29dof_dex3` | Pick red block with dexterous hand | Dex3 (3-finger) |
| `rgb_block_stacking_g1_29dof_dex3` | Stack colored blocks | Dex3 (3-finger) |

**G1 Dex1 Manipulation Tasks** (single dexterous hand):

| Scene | Task | Hand |
|-------|------|------|
| `pick_place_cylinder_g1_29dof_dex1` | Pick cylinder | Dex1 |
| `pick_place_redblock_g1_29dof_dex1` | Pick red block | Dex1 |

#### Setup (unitree_sim_isaaclab)

```bash
# On workstation (192.168.1.205), inside dm-workstation container
cd /workspace/unitree_sim_isaaclab

# Fetch G1 USD assets (one-time)
bash fetch_assets.sh

# Install (see: https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/main/doc/isaacsim5.0_install.md)
conda run --no-capture-output -n unitree_sim_env pip install -e .
```

#### Running a G1 Gripper Scene

```bash
# Pick-and-place with G1 gripper
conda run --no-capture-output -n unitree_sim_env python \
    unitree_sim_isaaclab/tasks/pick_place_cylinder_g1_29dof_gripper/train.py \
    --num_envs 4096

# Wholebody mobile manipulation with G1 gripper
conda run --no-capture-output -n unitree_sim_env python \
    unitree_sim_isaaclab/tasks/wholebody_g1_29dof_gripper/train.py \
    --num_envs 4096
```

### NVIDIA Isaac Lab Built-in G1 Environments

Isaac Lab also ships with built-in G1 environments (from [Isaac Lab envs](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)):

| Environment | Task | Hand Type |
|-------------|------|-----------|
| `Isaac-PickPlace-G1-InspireFTP-Abs-v0` | Pick and place object to basket | Inspire fingertip |
| `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0` | Pick and place (fixed base, IK control) | Three-finger |
| `Isaac-PickPlace-Locomanipulation-G1-Abs-v0` | Full loco-manipulation pick & place | Three-finger |
| `Isaac-Velocity-Flat-G1-v0` | Flat terrain locomotion | — |
| `Isaac-Velocity-Rough-G1-v0` | Rough terrain locomotion | — |

### G1 USD Assets

| Variant | Source | USD Path / Notes |
|---------|--------|------------------|
| G1 Base | Isaac Sim built-in | `Robots/Unitree/G1/g1.usd` |
| G1 with Hand (29-DOF) | Isaac Sim built-in | `Robots/Unitree/G1/G1_with_hand/g1_29dof_with_hand_rev_1_0.usd` |
| G1-29dof-gripper | unitree_sim_isaaclab | Via `fetch_assets.sh` |
| G1-29dof-dex1 | unitree_sim_isaaclab | Via `fetch_assets.sh` |
| G1-29dof-dex3 | unitree_sim_isaaclab | Via `fetch_assets.sh` |
| G1-29dof-inspire | unitree_sim_isaaclab | Via `fetch_assets.sh` |

### Closed-Loop Evaluation with GROOT in Isaac Sim

The GROOT inference server uses the same ZeroMQ protocol regardless of the simulation backend. To connect Isaac Lab to the GROOT server:

```python
from gr00t.policy.server_client import PolicyClient

# Connect to GROOT server (Spark or local)
client = PolicyClient(host="192.168.1.237", port=5555, strict=False)

# Inside your Isaac Lab environment step loop:
observation = {
    "video.ego_view": camera_image,           # (B, T, H, W, 3) uint8
    "state.observation.state": joint_state,   # (B, T, 31) float32
    "annotation.human.task_description": ["pick up the apple"],
}

action_dict, info = client.get_action(observation)
# Apply action_dict to robot in Isaac Sim
```

**Isaac Lab-Arena** ([docs](https://isaac-sim.github.io/IsaacLab-Arena/main/index.html)) provides a modular evaluation framework for composing custom eval tasks:

```bash
python isaaclab_arena/evaluation/policy_runner.py \
    --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
    --policy_config_yaml_path <config.yaml> \
    --num_steps 2000 \
    --enable_cameras \
    <task_name> --embodiment g1
```

### Current Status: G1 + GROOT in Isaac Sim

| Capability | Available? | Source | Notes |
|------------|-----------|--------|-------|
| G1 gripper pick-and-place | Yes | unitree_sim_isaaclab | Cylinder + red block scenes |
| G1 gripper wholebody manipulation | Yes | unitree_sim_isaaclab | Mobile pick-and-place |
| G1 built-in Isaac Lab envs | Yes | Isaac Lab | Inspire/3-finger hand variants |
| G1 locomotion | Yes | Isaac Lab | Flat + rough terrain |
| G1 + GROOT closed-loop (MuJoCo) | Yes | Isaac-GR00T WBC | Out-of-the-box |
| G1 + GROOT closed-loop (Isaac Sim) | Partial | Isaac Lab-Arena | Needs embodiment swap from GR1 examples |
| Complex manipulation (towel folding) | No | — | Requires custom environment |

**Key gap**: NVIDIA's Isaac Sim + GROOT closed-loop examples currently use the **GR1** robot (via [IsaacLabEvalTasks](https://github.com/isaac-sim/IsaacLabEvalTasks)). Adapting these for G1 requires swapping the embodiment configuration. The [Isaac Lab-Arena](https://isaac-sim.github.io/IsaacLab-Arena/main/index.html) modular builder makes this possible by composing Scene + Embodiment (G1) + Task. The [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) scenes are the best starting point for G1-native Isaac Sim work.

---

## Option 3: Real Robot Deployment

Connect the physical Unitree G1 EDU 2 to the GROOT inference server running on Spark.

### Architecture

```
┌──────────────┐     ZMQ      ┌──────────────┐
│  Unitree G1  │◄────────────►│  Spark Server │
│  EDU 2       │  Port 5555   │  groot-server │
│  (on-site)   │              │  192.168.1.237│
└──────────────┘              └──────────────┘
```

### Connection

```python
from gr00t.policy.server_client import PolicyClient

client = PolicyClient(host="192.168.1.237", port=5555, strict=False)

# Robot control loop
while not done:
    # Capture from robot
    observation = {
        "video.ego_view": robot.get_ego_camera(),
        "state.observation.state": robot.get_joint_state(),  # 31 DOF
        "annotation.human.task_description": ["fold the towel"],
    }

    # Get action from GROOT
    action_dict, info = client.get_action(observation)

    # Apply to robot (23 DOF action)
    robot.set_joint_targets(action_dict["action"])

    # Reset at episode boundaries
    # client.reset()
```

### Currently Deployed

| | |
|---|---|
| **Server** | Spark (192.168.1.237), `groot-server` container |
| **Model** | `groot-g1-gripper-hospitality-7ds` |
| **Port** | 5555 (ZMQ) |
| **Embodiment** | `UNITREE_G1` |
| **Tasks trained** | Fold towel, clean table, wipe table, prepare fruit, pour medicine, organize tools, pack ping pong |

---

## Evaluation Checklist

### Before Running Any Evaluation

- [ ] GROOT inference server running on Spark (`docker ps | grep groot-server`)
- [ ] Correct model loaded (check `ps aux | grep run_gr00t_server`)
- [ ] Network connectivity to Spark port 5555
- [ ] For simulation: `--use-sim-policy-wrapper` flag enabled on server
- [ ] Correct embodiment tag: `UNITREE_G1`

### Open-Loop Evaluation (Quick Validation)

Compare model predictions against ground-truth actions from the training dataset:

```bash
cd /workspace/Isaac-GR00T

uv run python gr00t/eval/open_loop_eval.py \
    --model-path /workspace/checkpoints/groot-g1-gripper-hospitality-7ds \
    --dataset-path /workspace/datasets/groot/G1_Fold_Towel \
    --embodiment-tag UNITREE_G1
```

### Closed-Loop Evaluation

Full interaction with simulation or real robot (see Options 1-3 above).

---

## Simulation Comparison

| Feature | MuJoCo (WBC) | Isaac Sim (Isaac Lab) |
|---------|--------------|----------------------|
| **Setup time** | ~30 min (run setup script) | ~2 hours (Isaac Sim install + env) |
| **G1 scenes out-of-the-box** | 1 (PnP Apple to Plate) | 5+ (pick-place, locomotion) |
| **GROOT integration** | Native (rollout_policy.py) | Needs adapter (Isaac Lab-Arena) |
| **Physics fidelity** | Good | Excellent (GPU PhysX) |
| **Rendering** | Basic | Photo-realistic (RTX) |
| **Domain randomization** | Limited | Extensive |
| **GPU requirement** | CPU or GPU | GPU required (NVIDIA) |
| **Parallel envs** | Yes (n_envs) | Yes (GPU-accelerated) |
| **Custom scenes** | Requires MuJoCo XML | USD-based, visual editor |
| **Best for** | Quick model validation | Full sim-to-real pipeline |

### Recommended Approach

1. **Start with MuJoCo WBC** — validate model outputs with the PnP Apple to Plate scene
2. **Open-loop eval** — compare predictions vs ground truth on training data
3. **Isaac Sim** — for full closed-loop evaluation with domain randomization
4. **Real robot** — final validation on physical hardware

---

## Complete Resource Map

All GitHub repos, NVIDIA resources, Unitree resources, and HuggingFace repos relevant to simulation and inference.

### GitHub Repositories

#### NVIDIA

| Repository | URL | Purpose | Used For |
|-----------|-----|---------|----------|
| **Isaac-GR00T** | [github.com/NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | GR00T N1.6 fine-tuning + evaluation framework | Training, inference server, open/closed-loop eval |
| **GR00T-WholeBodyControl** | [github.com/NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) | Decoupled WBC controller (RL lower body + IK upper body) | MuJoCo G1 loco-manipulation eval |
| **IsaacLabEvalTasks** | [github.com/isaac-sim/IsaacLabEvalTasks](https://github.com/isaac-sim/IsaacLabEvalTasks) | Isaac Sim eval benchmarks (GR1-T2) | Reference for adapting to G1 |
| **Isaac Lab-Arena** | [github.com/isaac-sim/IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) | Modular scene/embodiment/task composition | Custom G1 eval environments |
| **Isaac Lab** | [github.com/isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | Core Isaac Lab framework (v2.2.0 on workstation) | RL training, simulation environments |

#### Unitree

| Repository | URL | Purpose | Used For |
|-----------|-----|---------|----------|
| **unitree_sim_isaaclab** | [github.com/unitreerobotics/unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) | G1 Isaac Lab scenes + USD assets (gripper, dex1, dex3, inspire) | **Primary sim scenes for G1 gripper** |
| **unitree_rl_lab** | [github.com/unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | RL training for G1 locomotion | Locomotion policy training |
| **unitree_sdk2_python** | [github.com/unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) | Python SDK for physical G1 EDU 2 | Real robot deployment |
| **unitree_IL_lerobot** | [github.com/unitreerobotics/unitree_IL_lerobot](https://github.com/unitreerobotics/unitree_IL_lerobot) | MuJoCo-based imitation learning | Reference implementation |

#### Datamentors

| Repository | URL | Purpose |
|-----------|-----|---------|
| **dm-isaac-g1** | [github.com/datamentors/dm-isaac-g1](https://github.com/datamentors/dm-isaac-g1) | Main project: data pipeline, training configs, inference client, CLI |

### HuggingFace Resources

#### Models — NVIDIA (Base)

| Model | HuggingFace | Purpose |
|-------|-------------|---------|
| GR00T N1.6-3B | [nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) | Base model for all fine-tuning |
| G1 PnP Apple to Plate | [nvidia/GR00T-N1.6-G1-PnPAppleToPlate](https://huggingface.co/nvidia/GR00T-N1.6-G1-PnPAppleToPlate) | Pre-trained G1 eval checkpoint ([issue #485](https://github.com/NVIDIA/Isaac-GR00T/issues/485)) |

#### Models — Datamentors (Fine-tuned)

See [TEAM_MODEL_SUMMARY.md](TEAM_MODEL_SUMMARY.md) for the full model catalog (7 models).

#### Datasets — Unitree Hospitality (Training Data)

| Dataset | HuggingFace | Episodes | Hand |
|---------|-------------|----------|------|
| G1_Fold_Towel | [unitreerobotics/G1_Fold_Towel](https://huggingface.co/datasets/unitreerobotics/G1_Fold_Towel) | 200 | Gripper |
| G1_Clean_Table | [unitreerobotics/G1_Clean_Table](https://huggingface.co/datasets/unitreerobotics/G1_Clean_Table) | 200 | Gripper |
| G1_Wipe_Table | [unitreerobotics/G1_Wipe_Table](https://huggingface.co/datasets/unitreerobotics/G1_Wipe_Table) | 200 | Gripper |
| G1_Prepare_Fruit | [unitreerobotics/G1_Prepare_Fruit](https://huggingface.co/datasets/unitreerobotics/G1_Prepare_Fruit) | 200 | Gripper |
| G1_Pour_Medicine | [unitreerobotics/G1_Pour_Medicine](https://huggingface.co/datasets/unitreerobotics/G1_Pour_Medicine) | 200 | Gripper |
| G1_Organize_Tools | [unitreerobotics/G1_Organize_Tools](https://huggingface.co/datasets/unitreerobotics/G1_Organize_Tools) | 200 | Gripper |
| G1_Pack_PingPong | [unitreerobotics/G1_Pack_PingPong](https://huggingface.co/datasets/unitreerobotics/G1_Pack_PingPong) | 200 | Gripper |

#### Datasets — NVIDIA (Simulation + Teleop)

| Dataset | HuggingFace | Type |
|---------|-------------|------|
| X-Embodiment Sim | [nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) | Simulated (includes `unitree_g1.LMPnPAppleToPlateDC`, 103 eps) |
| G1 Teleop | [nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1) | Real robot teleop (`g1-pick-apple`, 311 eps) |

#### Datasets — Unitree Extended (Dex3/Dex1 — different hand types)

| Dataset | HuggingFace | Hand |
|---------|-------------|------|
| G1_Dex3_BlockStacking | [unitreerobotics/G1_Dex3_BlockStacking](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_BlockStacking) | Dex3 |
| G1_Dex3_GraspSquare | [unitreerobotics/G1_Dex3_GraspSquare](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_GraspSquare) | Dex3 |
| G1_Dex3_ObjectPlacement | [unitreerobotics/G1_Dex3_ObjectPlacement](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ObjectPlacement) | Dex3 |
| G1_Dex3_PickApple | [unitreerobotics/G1_Dex3_PickApple](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_PickApple) | Dex3 |
| G1_Dex3_Pouring | [unitreerobotics/G1_Dex3_Pouring](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_Pouring) | Dex3 |
| G1_Dex3_ToastedBread | [unitreerobotics/G1_Dex3_ToastedBread](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread) | Dex3 |
| G1_Dex1_MountCamera | [unitreerobotics/G1_Dex1_MountCamera](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_MountCamera) | Dex1 |

### Isaac-GR00T Evaluation Scripts

| Script | Path (in Isaac-GR00T repo) | Purpose |
|--------|---------------------------|---------|
| GROOT server | `gr00t/eval/run_gr00t_server.py` | Inference server (ZMQ, port 5555) |
| Rollout client | `gr00t/eval/rollout_policy.py` | Closed-loop MuJoCo evaluation |
| Open-loop eval | `gr00t/eval/open_loop_eval.py` | Offline prediction vs ground-truth comparison |
| WBC setup | `gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh` | One-time MuJoCo WBC environment setup |
| Readiness check | `scripts/eval/check_sim_eval_ready.py` | Verify all simulation dependencies |

### NVIDIA Documentation & Blogs

| Resource | URL |
|----------|-----|
| Isaac Lab Environments | [isaac-sim.github.io/IsaacLab/main/source/overview/environments.html](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html) |
| Isaac Lab-Arena Docs | [isaac-sim.github.io/IsaacLab-Arena](https://isaac-sim.github.io/IsaacLab-Arena/main/index.html) |
| GR00T N1.6 Sim-to-Real Workflow | [developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow](https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow) |
| Isaac Lab-Arena + LeRobot | [huggingface.co/blog/nvidia/generalist-robotpolicy-eval-isaaclab-arena-lerobot](https://huggingface.co/blog/nvidia/generalist-robotpolicy-eval-isaaclab-arena-lerobot) |
| GR00T Policy Guide | [github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md) |
| GR00T N1.6 Model Card | [huggingface.co/nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) |

### Known Upstream Issues

| Issue | Repository | Impact |
|-------|-----------|--------|
| [#485](https://github.com/NVIDIA/Isaac-GR00T/issues/485) | Isaac-GR00T | `nvidia/GR00T-N1.6-G1-PnPAppleToPlate` fails to load (missing `model_name` field) |
| [#342](https://github.com/NVIDIA/Isaac-GR00T/issues/342) | Isaac-GR00T | Torchcodec video backend set as default (causes failures on some systems) |
| [#508](https://github.com/NVIDIA/Isaac-GR00T/issues/508) | Isaac-GR00T | PyAV bulk video decoder fallback needed |
| [#268](https://github.com/NVIDIA/Isaac-GR00T/issues/268) | Isaac-GR00T | Isaac Sim + GROOT integration docs not yet available |
| [#4037](https://github.com/isaac-sim/IsaacLab/issues/4037) | IsaacLab | G1 USD asset mismatch between IsaacLab default and Unitree official |

---

## Related Documentation

- [TEAM_MODEL_SUMMARY.md](TEAM_MODEL_SUMMARY.md) — Models, datasets & infrastructure overview
- [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) — Train your own model
- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) — Inference server details
- [INFERENCE_SETUP.md](INFERENCE_SETUP.md) — Spark server setup
