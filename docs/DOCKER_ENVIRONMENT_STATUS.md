# Docker Environment Status

**Last updated**: 2026-03-03

## Infrastructure Overview

| Machine | IP | Role | Container | Image |
|---------|-----|------|-----------|-------|
| **Blackwell Workstation** | 192.168.1.205 | Training + Simulation + Eval | `dm-workstation` | `dm-workstation:latest` |
| **DGX Spark** | 192.168.1.237 | GROOT Inference Server | `groot-server` | `groot-inference:arm64` |
| **Vast.ai** (on-demand) | Cloud | Multi-GPU Fine-tuning | (ephemeral) | `vastai/pytorch` + setup script |
| **Local Mac** | - | Development, editing | (native) | No container |

---

## Process → Image Matrix

| Process | Machine | Image | Container | Python Env | Status | Launch Command |
|---------|---------|-------|-----------|------------|--------|----------------|
| **GR00T Fine-tuning** (1-GPU) | Workstation | `dm-workstation:latest` | `dm-workstation` | `unitree_sim_env` (conda, py3.11) | Ready | `conda run -n unitree_sim_env torchrun launch_finetune.py` |
| **GR00T Fine-tuning** (8-GPU) | Vast.ai | `vastai/pytorch` | (cloud) | setup script | Ready (on-demand) | `cloud/vastai/launch.sh` |
| **Isaac Sim Inference** | Workstation | `dm-workstation:latest` | `dm-workstation` | `unitree_sim_env` | Ready | `scripts/policy_inference_groot_g1.py` (needs VNC `:1`) |
| **Isaac Lab RL Training** | Workstation | `dm-workstation:latest` | `dm-workstation` | `unitree_sim_env` | Ready | `scripts/training/launch_rl_military_march.sh` |
| **Mimic/MimicGen Motion Tracking** | Workstation | `dm-workstation:latest` | `dm-workstation` | `unitree_sim_env` | Ready | `scripts/training/launch_mimic_ronaldo.sh` |
| **MuJoCo Closed-Loop Eval** | Workstation | `dm-workstation:latest` | `dm-workstation` | `unitree_sim_env` | Ready | `scripts/eval/run_mujoco_towel_eval.py` |
| **WBC Eval (RoboCasa, 38 Dex1 envs)** | Workstation | `dm-workstation:latest` | `dm-workstation` | WBC `.venv` (py3.10) | Ready | `scripts/eval/run_official_eval.sh` |
| **GROOT Inference Server** | Spark | `groot-inference:arm64` | `groot-server` | System py3 | Running | `run_gr00t_server.py --use-sim-policy-wrapper` |
| **Video2Robot Pipeline** | Workstation | `dm-workstation:latest` | `dm-workstation` | `phmr` conda env | Ready | `scripts/training/run_video2robot_ronaldo.sh` |

---

## Docker Images Inventory

### Workstation (192.168.1.205)

| Image | Tag | ID | Size | Built | Purpose | ECR Tag |
|-------|-----|-----|------|-------|---------|---------|
| `dm-workstation` | `latest` | `4580a8269c1e` | 42.4 GB | 2026-02-18 | Full env: Isaac Sim + IsaacLab + GR00T + MuJoCo + RL + Mimic | `latest` |
| `dm-workstation-base` | `latest` | `4497315fc101` | 42.1 GB | 2026-02-18 | Stable base: Isaac Sim + IsaacLab + Unitree (no GR00T) | `base-latest` |
| `nvcr.io/nvidia/isaac-sim` | `5.1.0` | `f3563cb2ba0c` | 22.9 GB | - | Upstream reference (unused) | - |

### Spark (192.168.1.237)

| Image | Tag | ID | Size | Purpose |
|-------|-----|-----|------|---------|
| `groot-inference` | `arm64` | `f1a22dc19097` | 23.4 GB | GROOT neural net server (ARM64 Grace Hopper) |
| `dm-spark-inference` | `latest` | - | 23.4 GB | Alternative build (created, never started) |
| `nvcr.io/nvidia/pytorch` | `25.04-py3` | - | 22.4 GB | Base for Spark inference image |

### ECR Registry

**Registry**: `260464233120.dkr.ecr.us-east-1.amazonaws.com/isaac-g1-sim-ft-rl`

| ECR Tag | Maps To | Last Pushed | Notes |
|---------|---------|-------------|-------|
| `base-latest` | `dm-workstation-base:latest` | 2026-02-19 | Stable base, no GR00T |
| `latest` | `dm-workstation:latest` | 2026-02-19 | Full image with GR00T |

**ECR login status**: EXPIRED (tokens expired 2026-02-20). Re-login required before push/pull.

---

## Container State (as of 2026-03-03)

### dm-workstation (Workstation)

- **Status**: Running (up 7 days)
- **Created from image**: 2026-02-19
- **Uncommitted changes**: 440K filesystem diffs (mostly NVIDIA runtime noise)
- **Significant in-container additions since image build**:
  - `mujoco_menagerie/` (Google DeepMind robot models)
  - `launch_finetune_v3.py`, `early_stop_callback.py`
  - `launch_rl_military_march.sh`, `launch_mimic_ronaldo.sh`
  - `run_video2robot_ronaldo.sh`
  - Python packages: `mujoco`, `paramiko`, `hidapi`, `prettytable`, `decord`, and more
- **Active training runs**:
  - RL Military March (4096 envs, ~11h in, ~18h remaining)
  - Mimic Ronaldo Celebration (4096 envs, ~3.5h in, ~23h remaining)

### groot-server (Spark)

- **Status**: Running (healthy, up 11h)
- **Uncommitted changes**: None (clean — only NVIDIA runtime injection)
- **Active model**: `GR00T-N1.6-G1-PnPAppleToPlate-8gpu` (NVIDIA pretrained, 60% success)
- **Stopped**: `groot-server-pnp-ft-v2` (our v2 FT, 0% success, killed 42h ago)

---

## Dockerfile Architecture

```
Dockerfile.unitree (environments/workstation/)
│
├── Stage 1: builder (nvidia/cuda:12.8.0-devel-ubuntu22.04)
│   ├── Miniconda + unitree_sim_env (Python 3.11)
│   ├── PyTorch 2.7.0+cu128
│   ├── Isaac Sim 5.0.0 (pip)
│   ├── IsaacLab v2.2.0
│   ├── CycloneDDS 0.10.x
│   ├── unitree_sdk2_python
│   ├── unitree_sim_isaaclab
│   ├── pink + pinocchio (IK)
│   ├── MuJoCo 3.2.6
│   └── MuJoCo Menagerie (cloned)
│
├── Stage 2: base (nvidia/cuda:12.8.0-runtime-ubuntu22.04)
│   ├── Runtime deps (XFCE4, TurboVNC, Chrome)
│   ├── Copies all builder artifacts
│   └── → dm-workstation-base:latest → ECR base-latest
│
└── Stage 3: groot (extends base)
    ├── requirements-groot.txt (GR00T + eval + SSH + RL deps)
    ├── flash-attn==2.8.3 (compiled with nvcc)
    ├── conda-forge ffmpeg (native .so for torchcodec)
    ├── torchcodec==0.4.0+cu128
    └── → dm-workstation:latest → ECR latest
```

---

## Key Python Packages (dm-workstation conda env)

| Package | Version | Purpose | Pin Reason |
|---------|---------|---------|------------|
| `torch` | 2.7.0+cu128 | Core ML framework | ABI-matched to flash-attn, torchcodec |
| `transformers` | 4.51.3 | GR00T backbone | 4.52+ breaks Eagle3_VL |
| `tokenizers` | 0.21.1 | Required by transformers 4.51.3 | 4.22+ breaks it |
| `torchcodec` | 0.4.0+cu128 | AV1 video decode for LeRobot v3 | ABI-matched to torch |
| `flash-attn` | 2.8.3 | Efficient attention for fine-tuning | Compiled for CUDA 12.8 |
| `mujoco` | 3.2.6 | Physics sim for eval scenes | Flexcomp support |
| `pin` (pinocchio) | 3.9.0 | Inverse kinematics | WBC/IK eval |
| `deepspeed` | 0.17.6 | Distributed training | Multi-GPU support |
| `wandb` | 0.25.0 | Training monitoring | - |
| `paramiko` | 4.0.0 | SSH for remote ops | - |

---

## Volume Mounts (dm-workstation)

| Host Path | Container Path | Contents |
|-----------|---------------|----------|
| `/home/datamentors/dm-isaac-g1` | `/workspace/dm-isaac-g1` | This repo (source code) |
| `/workspace/datasets` | `/workspace/datasets` | Training datasets |
| `/workspace/datasets_inspire` | `/workspace/datasets_inspire` | Legacy Inspire datasets |
| `/workspace/checkpoints` | `/workspace/checkpoints` | Model checkpoints |
| `/workspace/Isaac-GR00T` | `/workspace/Isaac-GR00T` | GR00T framework (patched) |
| `/home/datamentors/unitree_sim_isaaclab` | `/workspace/unitree_sim_isaaclab` | Unitree sim assets + tasks |
| `/dev/dri` | `/dev/dri` | GPU display rendering |

---

## Rebuild & ECR Push Playbook

### Prerequisites

```bash
# 1. Re-login to AWS SSO (requires browser)
aws sso login --profile elianomarques-dm

# 2. Get ECR credentials
source ../../.env
aws ecr get-login-password --region $AWS_REGION --profile $AWS_PROFILE | \
  docker login --username AWS --password-stdin $AWS_ECR_REGISTRY
```

### Build

```bash
cd /home/datamentors/dm-isaac-g1/environments/workstation

# Build base (Stage 2) — stable, pushed to ECR
docker compose -f docker-compose.unitree.yml build base

# Build groot (Stage 3) — full image
docker compose -f docker-compose.unitree.yml build groot
```

### Push to ECR

```bash
source ../../.env

# Tag and push base
docker tag dm-workstation-base:latest $AWS_ECR_REGISTRY/$AWS_ECR_REPO:base-latest
docker push $AWS_ECR_REGISTRY/$AWS_ECR_REPO:base-latest

# Tag and push full
docker tag dm-workstation:latest $AWS_ECR_REGISTRY/$AWS_ECR_REPO:latest
docker push $AWS_ECR_REGISTRY/$AWS_ECR_REPO:latest
```

### Replace Running Container

```bash
# IMPORTANT: Wait for active training runs to finish first!
# Check: docker exec dm-workstation ps aux | grep python

# Stop old container
docker compose -f docker-compose.unitree.yml stop groot

# Remove old container (preserves image)
docker compose -f docker-compose.unitree.yml rm groot

# Start new container from rebuilt image
docker compose -f docker-compose.unitree.yml up -d groot
```

---

## What Changed Since Last Build (Feb 19)

### Added to Dockerfile

1. **MuJoCo 3.2.6** — baked into builder stage (was manually installed Feb 25)
2. **MuJoCo Menagerie** — cloned in builder, copied to base (was manually cloned in container)

### Added to requirements-groot.txt

| Package | Version | Why Added |
|---------|---------|-----------|
| `paramiko` | 4.0.0 | SSH/SCP for remote file transfers |
| `bcrypt` | 5.0.0 | Paramiko dependency |
| `PyNaCl` | 1.6.2 | Paramiko dependency |
| `invoke` | 2.2.1 | Task runner for deployment scripts |
| `argcomplete` | 3.6.3 | CLI autocompletion |
| `prettytable` | 3.17.0 | Table formatting for RL training logs |
| `hidapi` | 0.15.0 | USB/HID device communication |
| `decord` | 0.6.0 | Fast video decoding (alternative to torchcodec) |

### Scripts Extracted from Container → Repo

| Script | Location | Purpose |
|--------|----------|---------|
| `launch_finetune_v3.py` | `scripts/training/` | PnP FT v3 with early stopping + no grad accum |
| `launch_rl_military_march.sh` | `scripts/training/` | RL training for G1 29-DOF military march |
| `launch_mimic_ronaldo.sh` | `scripts/training/` | Mimic training for Ronaldo celebration |
| `run_video2robot_ronaldo.sh` | `scripts/training/` | Video → motion data pipeline |

---

## Environments by Category

### Isaac Sim (GPU rendering, Omniverse)

| Environment | Script | Notes |
|-------------|--------|-------|
| `pickplace_g1_inspire` | `scripts/policy_inference_groot_g1.py` | G1 + Inspire hands, table pick-and-place |
| `locomanipulation_g1` | `scripts/policy_inference_groot_g1.py` | G1 locomotion + manipulation |
| `fixed_base_ik_g1` | Isaac Lab direct | G1 fixed-base IK tasks |

### Isaac Lab RL (GPU-parallel, headless)

| Task | Script | Envs | Notes |
|------|--------|------|-------|
| `DM-G1-29dof-MilitaryMarch` | `src/dm_isaac_g1/rl/scripts/train.py` | 4096 | Locomotion gait training, currently running |

### Mimic/MimicGen (GPU-parallel, headless)

| Task | Script | Envs | Notes |
|------|--------|------|-------|
| `DM-G1-29dof-Mimic-RonaldoCelebration` | `src/dm_isaac_g1/mimic/scripts/train.py` | 4096 | Motion tracking from video, currently running |

### MuJoCo Eval (CPU or EGL, closed-loop)

| Scene | Script | Notes |
|-------|--------|-------|
| `g1_gripper_towel_folding.xml` | `scripts/eval/run_mujoco_towel_eval.py` | Deformable towel, GROOT server required |
| `g1_dex1_towel_folding.xml` | `scripts/eval/run_mujoco_towel_eval_wbc.py` | Dex1 hands + WBC |

### WBC RoboCasa Eval (MuJoCo + GROOT, 38 Dex1 envs)

| Namespace | Example Env | Notes |
|-----------|-------------|-------|
| `gr00tlocomanip_g1_dex1_sim/` | `LMPnPAppleToPlateDC_G1Dex1_gear_wbc` | 38 auto-registered kitchen tasks |
| `gr00tlocomanip_g1_sim/` | `LMPnPAppleToPlateDC_G1_gear_wbc` | Standard G1 (non-Dex1) variants |

### GR00T Inference Server (Spark)

| Checkpoint | Path | Success Rate |
|------------|------|-------------|
| NVIDIA Pretrained PnP | `/workspace/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate-8gpu` | ~60% |
| Hospitality 7-DS | `/workspace/checkpoints/groot-g1-gripper-hospitality-7ds` | Deployed |
| Apple PnP (NVIDIA) | `/workspace/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate` | Baseline |
| PnP FT v2 (ours) | `/workspace/checkpoints/groot-g1-pnp-apple-dex3-v2` | 0% (investigating) |

### GR00T Fine-tuning

| Config | Embodiment | DOF | Status |
|--------|-----------|-----|--------|
| `g1_gripper_unitree.py` | UNITREE_G1 | 31 state / 23 action | Current |
| `g1_inspire_53dof.py` | NEW_EMBODIMENT | 53/53 | Legacy |
| `g1_dex3_28dof.py` | NEW_EMBODIMENT | 28/28 | Legacy |
