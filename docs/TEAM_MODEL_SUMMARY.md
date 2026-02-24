# GR00T N1.6 — Models, Inference & Datasets Summary

**Date**: February 24, 2026
**Team**: Datamentors Robotics & AI
**Repository**: `dm-isaac-g1`

---

## 1. HuggingFace Models (datamentorshf org)

All models are fine-tuned from **NVIDIA GR00T N1.6-3B** (`nvidia/GR00T-N1.6-3B`) for the **Unitree G1 EDU 2** robot.

### Production Models (UNITREE_G1 Gripper Embodiment)

These models use the official pre-registered `UNITREE_G1` embodiment tag in Isaac-GR00T:
- **31 DOF state** (legs, waist, arms, grippers)
- **23 DOF action** (arms RELATIVE, grippers/waist/nav ABSOLUTE)
- **1 camera** (ego_view)
- **30-step action horizon**

| # | Model | HuggingFace | Datasets | Training Steps | Loss | Visibility | Status |
|---|-------|-------------|----------|---------------|------|------------|--------|
| 1 | **Hospitality 7-Dataset** | [groot-g1-gripper-hospitality-7ds](https://huggingface.co/datamentorshf/groot-g1-gripper-hospitality-7ds) | All 7 hospitality (1,400 episodes) | 10,000 | 0.055 | Public | **Deployed on Spark** |
| 2 | Fold Towel (Full) | [groot-g1-gripper-fold-towel-full](https://huggingface.co/datamentorshf/groot-g1-gripper-fold-towel-full) | G1_Fold_Towel (200 episodes) | 10,000 (6k + 4k resume) | — | Private | Uploaded |
| 3 | Fold Towel (Partial) | [groot-g1-gripper-fold-towel](https://huggingface.co/datamentorshf/groot-g1-gripper-fold-towel) | G1_Fold_Towel (200 episodes) | 6,000 | 0.029 | Public | Uploaded |

### Legacy Models (Deprecated — not recommended for production)

These used experimental embodiment configurations and are kept for reference only.

| # | Model | HuggingFace | Embodiment | Datasets | Steps | Visibility |
|---|-------|-------------|------------|----------|-------|------------|
| 4 | Inspire 9-Dataset | [groot-g1-inspire-9datasets](https://huggingface.co/datamentorshf/groot-g1-inspire-9datasets) | `NEW_EMBODIMENT` (53 DOF, Inspire hands) | 9 mixed datasets with joint remapping | ~10,000 | Private |
| 5 | Loco-Manipulation | [groot-g1-loco-manip](https://huggingface.co/datamentorshf/groot-g1-loco-manip) | `UNITREE_G1` | LMPnPAppleToPlateDC sim data (103 eps) | 5,000 | Private |
| 6 | Teleop | [groot-g1-teleop](https://huggingface.co/datamentorshf/groot-g1-teleop) | `NEW_EMBODIMENT` (Tri-finger) | g1-pick-apple real teleop (311 eps) | 4,000 | Private |
| 7 | Dex3 28-DOF | [groot-g1-dex3-28dof](https://huggingface.co/datamentorshf/groot-g1-dex3-28dof) | `NEW_EMBODIMENT` (Dex3 hands) | Dex3 sim data | — | Private |

**Total: 7 models** on HuggingFace (3 production, 4 legacy)

---

## 2. Live Inference Server (Spark)

| | |
|---|---|
| **Server** | DGX Spark — `192.168.1.237` |
| **Container** | `groot-server` (healthy, running) |
| **GPU** | NVIDIA GB10 (unified memory architecture) |
| **Port** | `5555` (ZMQ protocol) |
| **Model loaded** | `groot-g1-gripper-hospitality-7ds` |
| **Embodiment** | `UNITREE_G1` |
| **Action horizon** | 30 steps |

### How to connect

```python
from gr00t.policy.server_client import PolicyClient

client = PolicyClient(host="192.168.1.237", port=5555, strict=False)
config = client.get_modality_config()
action_dict, info = client.get_action(observation)
```

### How to switch models

```bash
# SSH to Spark
ssh nvidia@192.168.1.237

# Download a different model
docker exec groot-server bash -c "
    huggingface-cli download datamentorshf/groot-g1-gripper-fold-towel-full \
        --local-dir /workspace/checkpoints/groot-g1-gripper-fold-towel-full
"

# Restart server with new model
docker restart groot-server
# Or manually specify a different model path in docker-compose / .env
```

### Server health check

```bash
# From any machine on the network
ssh nvidia@192.168.1.237 "docker exec groot-server bash -c 'curl -s http://localhost:5555/health'"

# Check container
ssh nvidia@192.168.1.237 "docker ps | grep groot"
```

---

## 3. Observation & Action Format (UNITREE_G1)

### State Vector (31 DOF — input to model)

| Index | Component | DOF |
|-------|-----------|-----|
| 0–5 | Left Leg | 6 |
| 6–11 | Right Leg | 6 |
| 12–14 | Waist | 3 |
| 15–21 | Left Arm | 7 |
| 22–28 | Right Arm | 7 |
| 29 | Left Gripper | 1 |
| 30 | Right Gripper | 1 |

### Action Vector (23 DOF — output from model)

| Index | Component | DOF | Representation |
|-------|-----------|-----|----------------|
| 0–2 | Waist | 3 | ABSOLUTE |
| 3–9 | Left Arm | 7 | RELATIVE |
| 10–16 | Right Arm | 7 | RELATIVE |
| 17 | Left Gripper | 1 | ABSOLUTE |
| 18 | Right Gripper | 1 | ABSOLUTE |
| 19 | Base Height | 1 | ABSOLUTE |
| 20–22 | Navigate (VX, VY, AngZ) | 3 | ABSOLUTE |

**Camera**: 1 ego-view image (`observation.images.ego_view`)
**Language**: Task description string (`annotation.human.task_description`)

---

## 4. Training Datasets

### Source: Unitree Hospitality Datasets (HuggingFace)

All 7 datasets are from the `unitreerobotics` org on HuggingFace (LeRobot v2 format). Each has **200 episodes** with **1 DOF gripper hands**.

| Dataset | HuggingFace | Episodes | Frames | Task |
|---------|-------------|----------|--------|------|
| G1_Fold_Towel | [unitreerobotics/G1_Fold_Towel](https://huggingface.co/datasets/unitreerobotics/G1_Fold_Towel) | 200 | 310,000 | Fold a towel |
| G1_Clean_Table | [unitreerobotics/G1_Clean_Table](https://huggingface.co/datasets/unitreerobotics/G1_Clean_Table) | 200 | 196,000 | Clean table surface |
| G1_Wipe_Table | [unitreerobotics/G1_Wipe_Table](https://huggingface.co/datasets/unitreerobotics/G1_Wipe_Table) | 200 | 264,000 | Wipe table |
| G1_Prepare_Fruit | [unitreerobotics/G1_Prepare_Fruit](https://huggingface.co/datasets/unitreerobotics/G1_Prepare_Fruit) | 200 | 123,000 | Prepare fruit |
| G1_Pour_Medicine | [unitreerobotics/G1_Pour_Medicine](https://huggingface.co/datasets/unitreerobotics/G1_Pour_Medicine) | 200 | 158,000 | Pour medicine |
| G1_Organize_Tools | [unitreerobotics/G1_Organize_Tools](https://huggingface.co/datasets/unitreerobotics/G1_Organize_Tools) | 200 | 182,000 | Organize tools |
| G1_Pack_PingPong | [unitreerobotics/G1_Pack_PingPong](https://huggingface.co/datasets/unitreerobotics/G1_Pack_PingPong) | 200 | 160,000 | Pack ping pong balls |
| **Total (merged)** | — | **1,400** | **1,280,000** | All 7 tasks |

### Workstation Dataset Locations

| Location | Contents | Status |
|----------|----------|--------|
| `/workspace/datasets/hospitality/` | Raw downloaded datasets | **Currently empty** — re-download from HuggingFace if needed |
| `/workspace/datasets/groot/` | Converted GR00T format datasets | **Currently empty** — re-convert from raw if needed |
| `/workspace/checkpoints/` | Training output checkpoints | Has `fold-towel-full` (18.4 GB) |

**Note**: Datasets were cleaned after training to free disk space. They can be easily re-downloaded from HuggingFace and re-converted using the `dm-isaac-g1` pipeline. See [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) for full reproduction steps.

**Workstation disk**: 1.1 TB free of 1.4 TB (19% used)

### How to re-download and convert

```bash
# SSH to workstation, enter container
ssh datamentors@192.168.1.205
docker exec -it dm-workstation bash

# Download a dataset
huggingface-cli download unitreerobotics/G1_Fold_Towel --repo-type dataset \
    --local-dir /workspace/datasets/hospitality/G1_Fold_Towel

# Convert to GR00T format
python -m dm_isaac_g1.data.convert_to_groot \
    --input /workspace/datasets/hospitality/G1_Fold_Towel \
    --output /workspace/datasets/groot/G1_Fold_Towel \
    --ego-camera cam_left_high
```

---

## 5. Infrastructure

| Server | IP | User | GPU | Role | Disk |
|--------|-----|------|-----|------|------|
| **Blackwell Workstation** | 192.168.1.205 | datamentors | RTX PRO 6000 (98 GB VRAM) | Training, data pipeline, Isaac Sim | 1.4 TB (1.1 TB free) |
| **DGX Spark** | 192.168.1.237 | nvidia | GB10 | Inference server | 2.3 TB free |

| Container | Server | Purpose |
|-----------|--------|---------|
| `dm-workstation` | Workstation | Training environment (Isaac-GR00T, conda `unitree_sim_env`) |
| `groot-server` | Spark | Inference server (GROOT model serving on port 5555) |

---

## 6. Quick Reference — Reproduction Commands

### Train the hospitality 7-dataset model (from scratch)

```bash
# Inside dm-workstation container on workstation
cd /workspace/Isaac-GR00T

conda run --no-capture-output -n unitree_sim_env \
    torchrun --nproc_per_node=1 gr00t/experiment/launch_finetune.py \
    --dataset-path /workspace/datasets/groot/merged_hospitality \
    --embodiment_tag UNITREE_G1 \
    --num-steps 10000 \
    --batch-size 32 \
    --output-dir /workspace/checkpoints/groot-g1-gripper-hospitality-7ds
```

### Deploy model to Spark

```bash
# Upload to HuggingFace
huggingface-cli upload datamentorshf/groot-g1-gripper-hospitality-7ds \
    /workspace/checkpoints/groot-g1-gripper-hospitality-7ds . \
    --repo-type model

# Download on Spark
ssh nvidia@192.168.1.237
docker exec groot-server bash -c "
    huggingface-cli download datamentorshf/groot-g1-gripper-hospitality-7ds \
        --local-dir /workspace/checkpoints/groot-g1-gripper-hospitality-7ds
"

# Restart inference server
docker restart groot-server
```

---

## 7. Simulation & Evaluation Options

There are two simulation environments available for testing fine-tuned models, plus real robot deployment. See [SIMULATION_INFERENCE_GUIDE.md](SIMULATION_INFERENCE_GUIDE.md) for full setup instructions.

### MuJoCo (Quick Validation)

Out-of-the-box via the [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) example:

| Scene | Task | Baseline |
|-------|------|----------|
| `LMPnPAppleToPlateDC_G1_gear_wbc` | Navigate + pick apple + place on plate | ~58% success |

```bash
# Server (Spark — already running)
# Client (workstation):
uv run python gr00t/eval/rollout_policy.py \
    --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \
    --n_episodes 10 --n_action_steps 20 --n_envs 5
```

### Isaac Sim / Isaac Lab (Full Physics Sim)

The [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) repo provides ready-to-use G1 gripper scenes (mounted at `/workspace/unitree_sim_isaaclab` on the workstation):

| Scene | Task | Hand |
|-------|------|------|
| `pick_place_cylinder_g1_29dof_gripper` | Pick cylinder, place at target | Gripper |
| `pick_place_redblock_g1_29dof_gripper` | Pick red block, place at target | Gripper |
| `wholebody_g1_29dof_gripper` | Mobile manipulation | Gripper |

Additional G1 scenes with dex3/dex1 hands also available. Isaac Lab built-in environments include locomotion (`Isaac-Velocity-Flat-G1-v0`, `Isaac-Velocity-Rough-G1-v0`) and pick-place with Inspire/3-finger hands.

### Real Robot

Connect the physical Unitree G1 EDU 2 directly to the Spark inference server on port 5555 via ZMQ.

---

## 8. Key Repositories & Resources

### GitHub Repositories

| Repository | Owner | Purpose |
|-----------|-------|---------|
| [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | NVIDIA | Fine-tuning + evaluation framework (core) |
| [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) | NVlabs | MuJoCo WBC controller for G1 |
| [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) | Unitree | Isaac Lab G1 scenes + USD assets |
| [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | Unitree | G1 locomotion RL training |
| [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) | Unitree | Python SDK for physical G1 |
| [unitree_IL_lerobot](https://github.com/unitreerobotics/unitree_IL_lerobot) | Unitree | MuJoCo imitation learning reference |
| [IsaacLabEvalTasks](https://github.com/isaac-sim/IsaacLabEvalTasks) | NVIDIA | Isaac Sim eval tasks (GR1, adaptable to G1) |
| [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) | NVIDIA | Modular scene/embodiment/task composer |
| [dm-isaac-g1](https://github.com/datamentors/dm-isaac-g1) | Datamentors | Main project repo |

### HuggingFace Resources

| Resource | Owner | Type | Purpose |
|----------|-------|------|---------|
| [GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) | NVIDIA | Model | Base VLA model for fine-tuning |
| [datamentorshf](https://huggingface.co/datamentorshf) | Datamentors | Org | All 7 fine-tuned models (see Section 1) |
| [unitreerobotics](https://huggingface.co/unitreerobotics) | Unitree | Org | 7 hospitality datasets + dex3/dex1 datasets |
| [PhysicalAI-GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) | NVIDIA | Dataset | Simulated G1 PnP data (103 eps) |
| [PhysicalAI-GR00T-Teleop-G1](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1) | NVIDIA | Dataset | Real G1 teleop data (311 eps) |

---

## 9. Next Steps

- **MuJoCo evaluation**: Validate models using the WBC PnP Apple to Plate scene
- **Isaac Sim scenes**: Test G1 gripper pick-place scenes from unitree_sim_isaaclab
- **G1 + GROOT closed-loop in Isaac Sim**: Adapt IsaacLabEvalTasks / Isaac Lab-Arena for G1
- **Real robot evaluation**: Connect G1 EDU 2 to Spark inference server
- **Task-specific models**: Train specialist models on individual datasets for comparison
- **Additional datasets**: Collect custom demonstration data for new tasks

---

*For simulation setup details, see [SIMULATION_INFERENCE_GUIDE.md](SIMULATION_INFERENCE_GUIDE.md).*
*For fine-tuning reproduction steps, see [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md).*
*For inference server details, see [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md).*
