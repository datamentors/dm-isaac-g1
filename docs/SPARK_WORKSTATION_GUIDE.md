# DGX Spark Fleet — Workstation Guide

**Last updated**: 2026-03-04

Turn each DGX Spark into a full training/sim/eval workstation — not just an inference server. Same capabilities as the Blackwell workstation, adapted for ARM64 Grace Hopper architecture.

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│  DGX Spark Fleet (ARM64 Grace Hopper, GB10 GPU 128GB unified memory)     │
│                                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐  ┌──────────┐ │
│  │ Spark 1          │  │ Spark 2          │  │ Spark 3    │  │ Spark 4  │ │
│  │ 192.168.1.???    │  │ 192.168.1.???    │  │ 192.168.1.???│ │ ...    │ │
│  │                  │  │                  │  │            │  │          │ │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │  │ ...        │  │ ...      │ │
│  │ │dm-spark-ws   │ │  │ │dm-spark-ws   │ │  │            │  │          │ │
│  │ │              │ │  │ │              │ │  │            │  │          │ │
│  │ │ GR00T FT     │ │  │ │ MuJoCo Eval  │ │  │            │  │          │ │
│  │ │ RL Training  │ │  │ │ GROOT Server │ │  │            │  │          │ │
│  │ │ Mimic Train  │ │  │ │ WBC Eval     │ │  │            │  │          │ │
│  │ │ MuJoCo Eval  │ │  │ │ Video2Robot  │ │  │            │  │          │ │
│  │ │ GROOT Server │ │  │ │              │ │  │            │  │          │ │
│  │ └──────────────┘ │  │ └──────────────┘ │  │            │  │          │ │
│  │  :5555 (ZMQ)     │  │  :5555 (ZMQ)     │  │            │  │          │ │
│  │  :8000 (HTTP)    │  │  :8000 (HTTP)    │  │            │  │          │ │
│  └─────────────────┘  └─────────────────┘  └────────────┘  └──────────┘ │
│                                                                           │
│  Image: dm-spark-workstation:latest (ARM64)                               │
│  Base:  nvcr.io/nvidia/pytorch:25.04-py3                                  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Hardware: DGX Spark

| Spec | Value |
|------|-------|
| **Architecture** | ARM64 (aarch64) Grace Hopper Superchip |
| **GPU** | GB10 — 128 GB unified memory |
| **Compute** | sm_121 |
| **CPU** | ARM Grace (10 cores) |
| **RAM** | Unified with GPU (128 GB total) |
| **Storage** | 256 GB NVMe |
| **OS** | Ubuntu 22.04 (JetPack / DGX OS) |

### Key Differences from Workstation (Blackwell)

| | Blackwell Workstation | DGX Spark |
|---|---|---|
| **Arch** | x86_64 | ARM64 (aarch64) |
| **GPU** | RTX PRO 6000 (98 GB VRAM) | GB10 (128 GB unified) |
| **CUDA** | 12.8 (driver 590.48) | 13.0 (NVIDIA container) |
| **PyTorch** | 2.7.0+cu128 (pip) | NVIDIA container pre-built |
| **flash-attn** | 2.8.3 (compiled) | Not available on ARM64 |
| **torchcodec** | 0.4.0+cu128 | Not available on ARM64 |
| **Isaac Sim** | 5.0.0 (pip) | Not available on ARM64 |
| **IsaacLab** | v2.2.0 | Not available on ARM64 |
| **Base image** | `nvidia/cuda:12.8.0-devel-ubuntu22.04` | `nvcr.io/nvidia/pytorch:25.04-py3` |

### What Works / Doesn't Work on Spark

| Capability | Status | Notes |
|-----------|--------|-------|
| **GR00T Inference** | Works | Running today, proven |
| **GR00T Fine-tuning (1-GPU)** | Works | DeepSpeed + bf16, standard attention fallback |
| **MuJoCo Eval** | Works | `mujoco` has ARM64 wheels, EGL rendering |
| **WBC RoboCasa Eval** | Works | MuJoCo-based, no Isaac Sim needed |
| **RL Training (MuJoCo-based)** | Works | RSL-RL + MuJoCo (not Isaac Lab) |
| **Video2Robot Pipeline** | Works | PromptHMR + GMR, no GPU rendering needed |
| **Mimic Training (MuJoCo)** | Works | MuJoCo motion tracking |
| **Isaac Sim** | Not available | No ARM64 build — x86_64 only |
| **Isaac Lab** | Not available | Depends on Isaac Sim |
| **Isaac Lab RL** | Not available | Use MuJoCo-based RL instead |
| **VNC Desktop** | Optional | Can install XFCE + TurboVNC if needed |

---

## Spark Inventory

> **TODO**: Fill in with actual IPs and roles once hardware is inventoried.

| Spark # | IP | Hostname | Current Role | Target Role |
|---------|-----|----------|-------------|-------------|
| 1 | 192.168.1.237 | spark-01 | GROOT Inference | Inference + Eval |
| 2 | 192.168.1.??? | spark-02 | (unused) | Training |
| 3 | 192.168.1.??? | spark-03 | (unused) | Training |
| 4 | 192.168.1.??? | spark-04 | (unused) | Training / Eval |

---

## Prerequisites

1. **SSH access** to each Spark:
   ```bash
   sshpass -p "datamentors" ssh -o StrictHostKeyChecking=no nvidia@<spark-ip>
   ```
   User: `nvidia`, password: `datamentors`

2. **Docker + NVIDIA Container Toolkit** pre-installed (part of DGX OS)

3. **Network**: All Sparks on same LAN (192.168.1.0/24) with workstation

---

## Quick Reference

```bash
# On your Mac, SSH into a Spark
sshpass -p "datamentors" ssh -o StrictHostKeyChecking=no nvidia@192.168.1.237

# On the Spark host:
cd ~/dm-isaac-g1/environments/spark

# Build the full workstation image (first time, ~30 min)
docker compose -f docker-compose.spark.yml build

# Start the workstation container
docker compose -f docker-compose.spark.yml up -d workstation

# Shell into the container
docker exec -it dm-spark-workstation bash

# Check GPU
docker exec dm-spark-workstation python -c "import torch; print(torch.cuda.get_device_name())"
```

---

## One-Time Setup (per Spark)

### 1. Clone repos on the Spark host

```bash
ssh nvidia@<spark-ip>

# Create workspace directories
sudo mkdir -p /workspace/{checkpoints,datasets,datasets_inspire}
sudo chown -R nvidia:nvidia /workspace

# Clone dm-isaac-g1
cd ~
git clone https://github.com/datamentors/dm-isaac-g1.git

# Clone Isaac-GR00T (for GR00T framework)
cd /workspace
git clone https://github.com/NVIDIA/Isaac-GR00T.git
# Apply patches (see agent.md "Isaac-GR00T Upstream Patches")

# Clone WBC (for eval)
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git GR00T-WholeBodyControl-dex1

# Clone MuJoCo models
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

### 2. Build the workstation image

```bash
cd ~/dm-isaac-g1/environments/spark
docker compose -f docker-compose.spark.yml build workstation
```

This builds `dm-spark-workstation:latest` (~25 GB, ~30 min first time).

### 3. Start the container

```bash
docker compose -f docker-compose.spark.yml up -d workstation
```

### 4. Post-build setup (inside container)

```bash
docker exec -it dm-spark-workstation bash

# Install dm-isaac-g1 as editable
pip install -e /workspace/dm-isaac-g1

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}, GPU: {torch.cuda.get_device_name()}')"
python -c "import mujoco; print(f'MuJoCo: {mujoco.__version__}')"
python -c "import gr00t; print('GR00T: OK')"
```

---

## Use Cases

### 1. GR00T Inference Server

Same as current Spark setup. Run the GROOT server for eval clients.

```bash
docker exec -it dm-spark-workstation bash

cd /workspace/Isaac-GR00T
python gr00t/eval/run_gr00t_server.py \
    --model-path /workspace/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment-tag UNITREE_G1 \
    --port 5555 \
    --use-sim-policy-wrapper
```

### 2. GR00T Fine-tuning (1-GPU)

Train on Spark's GB10 (128 GB unified memory — more VRAM than workstation's 98 GB).

```bash
docker exec -it dm-spark-workstation bash

cd /workspace/Isaac-GR00T

# Download dataset from HuggingFace (or sync from workstation)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('datamentorshf/unitree_g1.LMPnPAppleToPlateDC',
                  local_dir='/workspace/datasets/gr00t_x_embodiment/unitree_g1.LMPnPAppleToPlateDC',
                  repo_type='dataset')
"

# Fine-tune (single GPU, no flash-attn — uses standard attention)
python -u gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path /workspace/datasets/gr00t_x_embodiment/unitree_g1.LMPnPAppleToPlateDC \
    --embodiment_tag UNITREE_G1 \
    --num_gpus 1 \
    --output_dir /workspace/checkpoints/groot-ft-spark \
    --max_steps 10000 \
    --use_wandb \
    --global_batch_size 128 \
    --gradient_accumulation_steps 8
```

**Note**: No flash-attn on ARM64 — model uses standard attention (slower, but functionally identical). The 128 GB unified memory compensates for the extra VRAM usage.

### 3. MuJoCo Closed-Loop Eval

Run MuJoCo evaluation scenes. Can connect to GROOT server on the same Spark or another Spark.

```bash
docker exec -it dm-spark-workstation bash

cd /workspace/dm-isaac-g1

# Self-contained: run eval + server on same Spark
# Terminal 1: Start GROOT server
cd /workspace/Isaac-GR00T
python gr00t/eval/run_gr00t_server.py \
    --model-path /workspace/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment-tag UNITREE_G1 --port 5555 --use-sim-policy-wrapper &

# Terminal 2: Run MuJoCo eval (EGL rendering, no display needed)
cd /workspace/dm-isaac-g1
MUJOCO_GL=egl python scripts/eval/run_mujoco_towel_eval.py \
    --policy_client_host localhost \
    --policy_client_port 5555 \
    --n_episodes 5

# Or connect to another Spark running the server
MUJOCO_GL=egl python scripts/eval/run_mujoco_towel_eval.py \
    --policy_client_host 192.168.1.237 \
    --policy_client_port 5555 \
    --n_episodes 5
```

### 4. WBC RoboCasa Eval (38 Dex1 Environments)

Full WBC pipeline evaluation — all MuJoCo-based, no Isaac Sim needed.

```bash
docker exec -it dm-spark-workstation bash

cd /workspace/Isaac-GR00T

# Run official eval (requires GROOT server running)
MUJOCO_GL=egl python gr00t/eval/rollout_policy.py \
    --n_episodes 5 --max_episode_steps 720 \
    --env_name gr00tlocomanip_g1_dex1_sim/LMPnPAppleToPlateDC_G1Dex1_gear_wbc \
    --policy_client_host 192.168.1.237 --policy_client_port 5555 \
    --n_action_steps 20 --n_envs 1
```

### 5. Video2Robot Pipeline

Convert human video → robot motion data (PromptHMR + GMR).

```bash
docker exec -it dm-spark-workstation bash

# Step 1: Extract poses (PromptHMR)
cd /workspace/video2robot
python scripts/extract_pose.py \
    --video /workspace/data/my_video.mp4 \
    --output /workspace/data/poses/

# Step 2: Retarget to G1 (GMR)
python scripts/retarget.py \
    --input /workspace/data/poses/ \
    --robot unitree_g1 \
    --output /workspace/data/retarget/

# Step 3: Convert PKL → CSV (for mimic training)
cd /workspace/dm-isaac-g1
python src/dm_isaac_g1/mimic/scripts/pkl_to_csv.py \
    --input /workspace/data/retarget/results.pkl \
    --output /workspace/data/my_motion.csv
```

**Note**: CSV → NPZ conversion requires Isaac Sim (headless), which is x86_64 only. Run that step on the workstation or ECS.

### 6. Mimic Training (MuJoCo-based)

Train motion tracking policies using MuJoCo (not Isaac Lab).

```bash
docker exec -it dm-spark-workstation bash

cd /workspace/dm-isaac-g1

# Train mimic policy (MuJoCo backend, headless)
python -u src/dm_isaac_g1/mimic/scripts/train.py \
    --task DM-G1-29dof-Mimic-RonaldoCelebration \
    --num_envs 2048 \
    --max_iterations 30000 \
    --headless \
    --sim_backend mujoco
```

### 7. Parallel Fleet Operations

Run different workloads across multiple Sparks simultaneously.

```bash
# Example: 4-Spark setup
# Spark 1 (192.168.1.237): GROOT server (serving models for eval)
# Spark 2: GR00T fine-tuning (new dataset)
# Spark 3: MuJoCo eval (running WBC RoboCasa 38 envs)
# Spark 4: Video2Robot pipeline (batch processing videos)

# From Mac, launch jobs on each Spark:
for SPARK_IP in 192.168.1.237 192.168.1.??? 192.168.1.??? 192.168.1.???; do
    sshpass -p "datamentors" ssh -o StrictHostKeyChecking=no nvidia@$SPARK_IP \
        "docker exec dm-spark-workstation nvidia-smi" &
done
wait
```

---

## Image Comparison

| Feature | `groot-inference:arm64` (current) | `dm-spark-workstation:latest` (new) |
|---------|----------------------------------|-------------------------------------|
| **Purpose** | GROOT inference only | Full workstation |
| **Base** | `nvcr.io/nvidia/pytorch:25.04-py3` | `nvcr.io/nvidia/pytorch:25.04-py3` |
| **GR00T** | Yes | Yes |
| **MuJoCo** | No | Yes (3.2.6 + Menagerie) |
| **WBC** | No | Yes (GR00T-WholeBodyControl-dex1) |
| **Video2Robot** | No | Yes (PromptHMR + GMR) |
| **DeepSpeed** | Yes | Yes |
| **wandb** | Yes | Yes |
| **RSL-RL** | No | Yes |
| **dm-isaac-g1** | Mounted | Mounted |
| **Repos baked in** | Isaac-GR00T only | GR00T + WBC + Video2Robot + Menagerie |
| **Size (est.)** | ~23 GB | ~28 GB |

---

## ARM64 Compatibility Notes

### Packages NOT Available on ARM64

| Package | Why | Workaround |
|---------|-----|-----------|
| `flash-attn` | No ARM64 build | Standard attention (functional, slower, more VRAM) |
| `torchcodec` | No ARM64 wheels | Use `av` (pyav) or `decord` for video decode |
| `isaacsim` | x86_64 only (Omniverse) | Use MuJoCo for sim/eval |
| `IsaacLab` | Depends on Isaac Sim | Use MuJoCo-based RL |

### NVIDIA Container Patches Required

The NVIDIA `nvcr.io/nvidia/pytorch:25.04-py3` container has specific constraints that need patching:

1. **GR00T `pyproject.toml`**: Relax Python requirement from `==3.10.*` to `>=3.10`
2. **torch/torchvision versions**: Relax to `>=2.5.0` (container has pre-built versions)
3. **dm-tree constraint**: Container pins `0.1.9`, GR00T needs `0.1.8` — fix in `/etc/pip/constraint.txt`
4. **Disabled packages**: Comment out `torchcodec` and `flash-attn` in GR00T's pyproject.toml

All patches are handled automatically in the Dockerfile.

---

## Syncing Data Between Machines

### Workstation → Spark (datasets, checkpoints)

```bash
# From workstation host
scp -r /workspace/datasets/gr00t_x_embodiment/unitree_g1.LMPnPAppleToPlateDC \
    nvidia@192.168.1.???:/workspace/datasets/gr00t_x_embodiment/

# Or use rsync for large transfers
rsync -avz --progress \
    /workspace/checkpoints/groot-g1-gripper-hospitality-7ds/ \
    nvidia@192.168.1.???:/workspace/checkpoints/groot-g1-gripper-hospitality-7ds/
```

### Spark → Workstation (trained checkpoints, eval results)

```bash
# From workstation host
scp -r nvidia@192.168.1.???:/workspace/checkpoints/groot-ft-spark/ \
    /workspace/checkpoints/groot-ft-spark/
```

### Spark → Mac (eval videos)

```bash
# From Mac
scp nvidia@192.168.1.???:/workspace/dm-isaac-g1/eval_videos/*.mp4 \
    ~/Documents/DataScienceProjects/Datamentors/dataset_review/
```

---

## Process → Machine Matrix (Updated)

| Process | Workstation | Spark (any) | ECS (cloud) | Notes |
|---------|:-----------:|:-----------:|:-----------:|-------|
| **GR00T Fine-tuning (1-GPU)** | Yes | Yes | Yes | Spark has 128 GB unified mem |
| **GR00T Fine-tuning (8-GPU)** | - | - | Yes (Vast.ai) | Multi-GPU only on cloud |
| **GR00T Inference Server** | Yes | Yes | Yes | Current Spark primary role |
| **MuJoCo Eval** | Yes | Yes | Yes | EGL rendering, no display needed |
| **WBC RoboCasa Eval** | Yes | Yes | Yes | 38 Dex1 envs, MuJoCo-based |
| **Isaac Sim Inference** | Yes | - | Yes | x86_64 only |
| **Isaac Lab RL Training** | Yes | - | Yes | x86_64 only |
| **Mimic Training (Isaac)** | Yes | - | Yes | x86_64 only |
| **Mimic Training (MuJoCo)** | Yes | Yes | Yes | ARM64 compatible |
| **Video2Robot (pose extract)** | Yes | Yes | Yes | No GPU rendering needed |
| **Video2Robot (CSV→NPZ)** | Yes | - | Yes | Needs Isaac Sim |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "CUDA error: no kernel image" | Ensure using NVIDIA's PyTorch container, not pip-installed PyTorch |
| "No module named 'flash_attn'" | Expected on ARM64 — model falls back to standard attention |
| "dm-tree version conflict" | Dockerfile patches `/etc/pip/constraint.txt` automatically |
| First container start slow (5-10 min) | Normal — PyTorch compiles kernels for GB10 on first run |
| "MuJoCo EGL error" | Set `MUJOCO_GL=egl` and ensure NVIDIA runtime is active |
| Out of disk space | Spark has 256 GB NVMe — prune old Docker images: `docker system prune` |
| Container can't reach network | Check `--network host` in compose or Docker bridge settings |
| "Permission denied" on /workspace | Run `sudo chown -R nvidia:nvidia /workspace` on host |

---

## Cost Comparison

| Setup | Hardware | Cost | VRAM | Architecture |
|-------|----------|------|------|-------------|
| DGX Spark (owned) | GB10 | $0/hr (owned) | 128 GB unified | ARM64 |
| Workstation (owned) | RTX PRO 6000 | $0/hr (owned) | 98 GB | x86_64 |
| ECS g5.2xlarge | A10G | $1.45/hr | 24 GB | x86_64 |
| Vast.ai 8xA100 | 8x A100 | ~$15/hr | 8x 80 GB | x86_64 |

**The 4 Sparks give us 4x 128 GB = 512 GB total GPU memory at zero marginal cost.**
