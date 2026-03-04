# ECS GPU Cluster — Team Guide

GPU training and development cluster on AWS ECS. Auto-scales from 0 (zero cost when idle) to N GPU instances on demand. Each container runs our full `dm-workstation` Docker image from ECR — identical to the workstation environment.

## What's in the Container

The ECR image (`isaac-g1-sim-ft-rl:latest`) is the same as `dm-workstation:latest`:

| Component | Details |
|-----------|---------|
| **Base** | NVIDIA CUDA 12.8 + Ubuntu 22.04 |
| **Isaac Sim** | 5.0.0 (pip) + IsaacLab v2.2.0 |
| **MuJoCo** | 3.2.6 + Menagerie robot models + unitree_mujoco |
| **PyTorch** | 2.7.0+cu128 with flash-attn, DeepSpeed |
| **GR00T** | Isaac-GR00T + WBC + video2robot |
| **Desktop** | XFCE4 + TurboVNC 3.1.2 + Chrome |
| **RL** | RSL-RL, unitree_rl_lab, dm_isaac_g1, unitree_sim_isaaclab |
| **Conda env** | `unitree_sim_env` (Python 3.11) |
| **VNC** | Port 5901, password: `datament` |

**Source repos baked into the image** (at `/workspace/`):

| Repo | Path | Notes |
|------|------|-------|
| dm-isaac-g1 | `/workspace/dm-isaac-g1` | Installed as editable (`pip install -e .`) |
| unitree_rl_lab | `/workspace/unitree_rl_lab` | Installed as editable |
| Isaac-GR00T | `/workspace/Isaac-GR00T` | GROOT framework |
| unitree_sim_isaaclab | `/home/code/unitree_sim_isaaclab` | USD assets + scene configs |
| unitree_mujoco | `/workspace/unitree_mujoco` | G1 MuJoCo scene XML |

> **Important:** The baked-in repos may be behind `main`. Always run `git pull origin main` after entering the container to get the latest code.

See [DOCKER_ENVIRONMENT_STATUS.md](../../docs/DOCKER_ENVIRONMENT_STATUS.md) for the full Dockerfile architecture and package list.

## Prerequisites

Each team member needs:

1. **AWS CLI v2**: `brew install awscli` (Mac) or `pip install awscli`
2. **Session Manager plugin** (for `exec` command):
   ```bash
   # Mac
   brew install --cask session-manager-plugin
   # Linux
   curl "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/ubuntu_64bit/session-manager-plugin.deb" -o session-manager-plugin.deb
   sudo dpkg -i session-manager-plugin.deb
   ```
3. **AWS credentials**:
   ```bash
   aws configure --profile elianomarques-dm
   # Region: eu-west-1 — ask team lead for access key + secret
   ```
4. **VNC client** (for GUI): RealVNC, TigerVNC Viewer, or TurboVNC Viewer
5. SSH key `dm-isaac-g1-training.pem` (in `cloud/ecs/` after setup)

---

## One-Time Setup (admin only)

```bash
cd dm-isaac-g1/cloud/ecs
./setup.sh --max-instances 10
```

Creates: ECS cluster, ASG (min=0, max=10), IAM roles, S3 bucket, security group, key pair. Takes ~2 minutes.

---

## Quick Reference

```bash
cd dm-isaac-g1/cloud/ecs
./run.sh help              # see all commands

# Training
./run.sh submit ...        # run a training job (auto-terminates when done)
./run.sh status            # cluster + task status
./run.sh logs              # stream training logs
./run.sh download ...      # download checkpoints from S3

# Interactive
./run.sh shell             # launch GPU container (24h, VNC auto-starts)
./run.sh exec              # bash shell into a container
./run.sh ssh               # SSH into the EC2 host
./run.sh vnc               # start/restart VNC in a container
./run.sh stop              # stop a task/container
```

---

## Use Cases

### 1. Team playground — MuJoCo, Isaac Sim, general exploration

Launch interactive containers for each team member. Each gets their own GPU, full desktop, and all tools.

```bash
# Launch 10 containers for the team (one per person)
for i in $(seq 1 10); do
    echo "--- Launching container $i ---"
    ./run.sh shell &
    sleep 5  # stagger API calls
done
wait
# Each prints its Task ARN and instance IP when RUNNING

# Check all containers
./run.sh status
```

**Each team member connects via VNC:**
1. Find your instance IP: `./run.sh ssh` (lists all instances)
2. Open VNC client → `<instance-ip>:5901`, password: `datament`
3. You get a full XFCE4 desktop with Chrome, terminal, GPU access

**Inside the container (via VNC terminal or `./run.sh exec`):**
```bash
# Activate conda
conda activate unitree_sim_env

# Test MuJoCo
python -c "import mujoco; print(f'MuJoCo {mujoco.__version__}')"

# Test Isaac Sim (needs display — works over VNC)
python /workspace/dm-isaac-g1/scripts/test_isaac_sim.py

# Test GPU
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name())"
```

**When done, stop your container:**
```bash
./run.sh stop --task-arn <your-task-arn>
# Instance auto-terminates when no tasks remain
```

---

### 2. RL training — Military March (headless)

Train a locomotion policy with Isaac Lab. Runs headless (no VNC needed).

**Option A: Interactive (inside container)**
```bash
./run.sh shell
# Wait for RUNNING, then:
./run.sh exec

# Inside the container:
conda activate unitree_sim_env
cd /workspace/dm-isaac-g1

# Pull latest code (repos are baked in but may be behind main)
git pull origin main
pip install -e .

# Also update unitree_rl_lab
cd /workspace/unitree_rl_lab && git pull origin main && pip install -e .
cd /workspace/dm-isaac-g1

# Train (headless, 4096 envs)
python -u src/dm_isaac_g1/rl/scripts/train.py \
    --task DM-G1-29dof-MilitaryMarch \
    --num_envs 4096 \
    --max_iterations 30000 \
    --headless
```

**Option B: Batch job (auto-terminates when done)**
```bash
./run.sh submit --task rl --task-id DM-G1-29dof-MilitaryMarch --max-iterations 30000
```

---

### 3. Mimic training — Motion from video (batch job)

Train a mimic policy from retargeted motion data. This is fully automated via `run.sh submit`.

```bash
# Upload data + launch training (auto-scales instance, runs, uploads checkpoints, shuts down)
./run.sh submit --task mimic --motion cr7_06_tiktok_uefa --max-iterations 30000

# Monitor
./run.sh status
./run.sh logs

# When done, download checkpoints
./run.sh download --motion cr7_06_tiktok_uefa
```

**What happens under the hood:**
1. Uploads NPZ + config from `src/dm_isaac_g1/mimic/tasks/cr7_06_tiktok_uefa/` to S3
2. Registers ECS task definition
3. ECS auto-scales a g5.2xlarge GPU instance (~3-5 min)
4. Container pulls latest code, downloads data from S3, runs training
5. Uploads checkpoints to S3 when done
6. Instance auto-terminates

---

### 4. GR00T fine-tuning (1-GPU)

Fine-tune GR00T on a single GPU. Uses the same launch scripts as the workstation.

```bash
./run.sh shell
./run.sh exec

# Inside container:
conda activate unitree_sim_env
cd /workspace/Isaac-GR00T

# Download dataset from S3 (pre-uploaded)
aws s3 sync s3://dm-isaac-g1-training-eu-west-1/datasets/unitree_g1.LMPnPAppleToPlateDC \
    /workspace/datasets/gr00t_x_embodiment/unitree_g1.LMPnPAppleToPlateDC

# Launch fine-tuning
python -u gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path /workspace/datasets/gr00t_x_embodiment/unitree_g1.LMPnPAppleToPlateDC \
    --embodiment_tag UNITREE_G1 \
    --num_gpus 1 \
    --output_dir /workspace/checkpoints/groot-g1-pnp-ecs \
    --max_steps 10000 \
    --use_wandb \
    --global_batch_size 128 \
    --gradient_accumulation_steps 8

# Upload results
aws s3 sync /workspace/checkpoints/groot-g1-pnp-ecs \
    s3://dm-isaac-g1-training-eu-west-1/checkpoints/groot/pnp-ecs/
```

---

### 5. Video2Robot — Retarget video to robot motion

Run the video-to-robot-motion pipeline (PromptHMR + GMR).

```bash
./run.sh shell
./run.sh exec

# Inside container:
# Step 1: Extract poses (PromptHMR)
conda activate phmr
cd /workspace/video2robot

# Start the web UI (optional — for visual track selection)
cd web && uvicorn app:app --host 0.0.0.0 --port 8000 &

# Or run CLI directly:
python scripts/extract_pose.py --video /workspace/data/my_video.mp4 --output /workspace/data/poses/

# Step 2: Retarget to G1 (GMR)
conda activate gmr
python scripts/retarget.py --input /workspace/data/poses/ --robot unitree_g1 --output /workspace/data/retarget/

# Step 3: Convert PKL → CSV → NPZ (for mimic training)
conda activate unitree_sim_env
python /workspace/dm-isaac-g1/src/dm_isaac_g1/mimic/scripts/pkl_to_csv.py \
    --input /workspace/data/retarget/results.pkl \
    --output /workspace/data/my_motion.csv

# NPZ conversion requires Isaac Sim (headless)
python /workspace/dm-isaac-g1/src/dm_isaac_g1/mimic/scripts/csv_to_npz.py \
    --input /workspace/data/my_motion.csv \
    --output /workspace/dm-isaac-g1/src/dm_isaac_g1/mimic/tasks/my_motion/my_motion.npz
```

---

### 6. MuJoCo eval — Closed-loop with GROOT server

Run MuJoCo evaluation scenes. Requires GROOT inference server (Spark or local).

```bash
./run.sh shell
./run.sh exec

# Inside container:
conda activate unitree_sim_env
cd /workspace/dm-isaac-g1

# Run towel folding eval (connect to Spark GROOT server)
MUJOCO_GL=egl python scripts/eval/run_mujoco_towel_eval.py \
    --policy_client_host 192.168.1.237 \
    --policy_client_port 5555 \
    --n_episodes 5

# Or run the WBC eval (RoboCasa Dex1 envs)
cd /workspace/Isaac-GR00T
MUJOCO_GL=egl python gr00t/eval/rollout_policy.py \
    --n_episodes 5 --max_episode_steps 720 \
    --env_name gr00tlocomanip_g1_dex1_sim/LMPnPAppleToPlateDC_G1Dex1_gear_wbc \
    --policy_client_host 192.168.1.237 --policy_client_port 5555 \
    --n_action_steps 20 --n_envs 1
```

**Note:** MuJoCo eval connects to the Spark GROOT server at `192.168.1.237:5555`. Make sure the VPN/network allows the ECS instance to reach it, or run a local GROOT server inside the container.

---

### 7. Isaac Sim with GUI (VNC)

Run Isaac Sim scenes with full 3D visualization over VNC.

```bash
./run.sh shell
# Wait for RUNNING
# Connect via VNC client to <instance-ip>:5901 (password: datament)

# Inside VNC terminal:
conda activate unitree_sim_env

# View G1 robot USD model
python /workspace/dm-isaac-g1/scripts/view_g1_usd.py

# Launch scene UI
python /workspace/dm-isaac-g1/scripts/launch_scene_ui.py

# Run RL play (visualize trained policy)
python /workspace/dm-isaac-g1/src/dm_isaac_g1/rl/scripts/play.py \
    --task DM-G1-29dof-MilitaryMarch \
    --num_envs 16
```

---

### 8. SSH into instance + Docker management

For advanced users who want full host access:

```bash
./run.sh ssh
# Connects to the EC2 host via SSH

# On the host:
docker ps                           # list containers
docker exec -it <id> bash           # shell into container
docker logs <id>                    # container logs
nvidia-smi                          # check GPU on host

# Run additional containers on the same host
docker run --gpus all -it \
    260464233120.dkr.ecr.eu-west-1.amazonaws.com/isaac-g1-sim-ft-rl:latest \
    bash
```

---

## Scaling: 1 VM = 1 Container = 1 GPU

Each g5.2xlarge has **1 GPU** (A10G 24GB). Each container requests **1 GPU**. So ECS places exactly **1 container per instance**.

| Team size | Instances | Cost/hr | 4-hour session |
|-----------|-----------|---------|----------------|
| 1 person | 1 x g5.2xlarge | $1.45 | $5.80 |
| 3 people | 3 x g5.2xlarge | $4.35 | $17.40 |
| 5 people | 5 x g5.2xlarge | $7.25 | $29.00 |
| 10 people | 10 x g5.2xlarge | $14.50 | $58.00 |

ECS manages this automatically. Stop all containers and the ASG scales back to 0 ($0/hr).

**To allow N concurrent containers:** set `--max-instances N` during setup.

---

## Architecture

```
┌───────────────────┐     ┌───────────────────────────────────────────────────┐
│   Team Members     │     │  AWS ECS Cluster (dm-isaac-g1-gpu)                │
│                    │     │                                                   │
│  VNC client ──────────▶  │  ASG: 0 → N x g5.2xlarge (1x A10G 24GB each)    │
│  ./run.sh exec ──────▶  │                                                   │
│  ./run.sh ssh ───────▶  │  ┌──────────────────┐  ┌──────────────────┐      │
│                    │     │  │ EC2 Instance #1   │  │ EC2 Instance #2   │ ... │
│                    │     │  │                    │  │                    │     │
│                    │     │  │ ┌──────────────┐  │  │ ┌──────────────┐  │     │
│                    │     │  │ │ Container    │  │  │ │ Container    │  │     │
│                    │     │  │ │              │  │  │ │              │  │     │
│                    │     │  │ │ Isaac Sim    │  │  │ │ MuJoCo       │  │     │
│                    │     │  │ │ MuJoCo       │  │  │ │ RL Training  │  │     │
│                    │     │  │ │ GR00T        │  │  │ │ Video2Robot  │  │     │
│                    │     │  │ │ TurboVNC     │  │  │ │ TurboVNC     │  │     │
│                    │     │  │ │ XFCE+Chrome  │  │  │ │ XFCE+Chrome  │  │     │
│                    │     │  │ └──────────────┘  │  │ └──────────────┘  │     │
│                    │     │  │  :5901 (VNC)      │  │  :5901 (VNC)      │     │
│                    │     │  │  :22 (SSH)        │  │  :22 (SSH)        │     │
│                    │     │  └──────────────────┘  └──────────────────┘      │
│                    │     │                                                   │
│                    │     │  ECR: isaac-g1-sim-ft-rl:latest (16 GB)          │
│                    │     │  S3:  dm-isaac-g1-training-eu-west-1              │
│                    │     └───────────────────────────────────────────────────┘
└───────────────────┘
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No GPU available" in container | GPU AMI handles NVIDIA drivers. Check `nvidia-smi` on the host via `./run.sh ssh` |
| `execute-command` not supported | Install Session Manager plugin (see Prerequisites) |
| Instance takes 3-5 min to start | Normal for GPU instances. Check `./run.sh status` for ASG state |
| VNC black screen | VNC auto-starts with `./run.sh shell`. Restart: `./run.sh vnc --task-arn <arn>` |
| Task fails immediately | Check `./run.sh logs --task-arn <arn>`. Common: missing env vars, S3 data not uploaded |
| ECR image pull slow (~16 GB) | First pull takes ~5 min. Subsequent pulls on same instance are cached |
| Conda env not found | Run `conda activate unitree_sim_env` — it's the default training env |
| Isaac Sim needs display | Use VNC (`:5901`) or set `--headless` for training. For EGL: `MUJOCO_GL=egl` |
| Can't reach Spark (192.168.1.237) | ECS instances are in AWS VPC, not on the office LAN. Run GROOT server locally in the container, or set up VPN |
