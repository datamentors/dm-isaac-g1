# GROOT Inference Implementation Plan

## Overview

This document outlines the two-track approach for running GROOT N1.6 inference with the Unitree G1 robot, addressing the physics engine compatibility discovered during development.

## Track 1: Locomotion in Isaac Sim

### Objective
Run G1 walking/locomotion policies in Isaac Sim using policies trained in Isaac Gym (same PhysX engine).

### Resources
| Resource | Location |
|----------|----------|
| Unitree RL Gym | https://github.com/unitreerobotics/unitree_rl_gym |
| Unitree RL Lab | https://github.com/unitreerobotics/unitree_rl_lab |
| Pre-trained Policies | HuggingFace: unitreerobotics/G1_locomotion |

### Implementation Steps

#### Step 1.1: Clone Unitree Repositories
```bash
# On workstation (192.168.1.205)
cd ~/
git clone https://github.com/unitreerobotics/unitree_rl_gym.git
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
```

#### Step 1.2: Download Pre-trained Walking Policy
```bash
# Using HuggingFace CLI
pip install huggingface_hub
huggingface-cli download unitreerobotics/G1_locomotion --local-dir ./policies/g1_locomotion
```

#### Step 1.3: Create Locomotion Inference Script
Create `scripts/policy_inference_locomotion.py`:
- Load pre-trained Isaac Gym policy
- Set up G1 robot in Isaac Sim with leg DOFs
- Run inference loop with velocity commands
- Action space: 12 leg joints (hip, knee, ankle × 4 legs)

#### Step 1.4: Test Walking Behaviors
- Forward walking at various speeds
- Turning left/right
- Walking on slopes
- Recovery from pushes

### Expected Observation Space
```python
{
    "base_lin_vel": [3],      # Linear velocity (x, y, z)
    "base_ang_vel": [3],      # Angular velocity
    "projected_gravity": [3], # Gravity projection
    "commands": [3],          # Velocity commands (vx, vy, yaw_rate)
    "dof_pos": [12],          # Joint positions (legs)
    "dof_vel": [12],          # Joint velocities
    "actions": [12],          # Previous actions
}
```

### Success Criteria
- G1 walks forward smoothly
- Can turn and navigate
- Stable on uneven terrain
- No joint explosions or instability

---

## Track 2: Manipulation in MuJoCo

### Objective
Run GROOT PnPAppleToPlate and other manipulation policies in MuJoCo (their native training environment).

### Resources
| Resource | Location |
|----------|----------|
| MuJoCo | https://github.com/google-deepmind/mujoco |
| LeRobot | https://github.com/huggingface/lerobot |
| Unitree IL LeRobot | https://github.com/unitreerobotics/unitree_IL_lerobot |
| GROOT Demo | Isaac-GR00T/getting_started/ |

### Implementation Steps

#### Step 2.1: Install MuJoCo on Workstation
```bash
# On workstation (192.168.1.205)
pip install mujoco mujoco-py gymnasium[mujoco]

# Verify installation
python -c "import mujoco; print(mujoco.__version__)"
```

#### Step 2.2: Clone Required Repositories
```bash
cd ~/
git clone https://github.com/huggingface/lerobot.git
git clone https://github.com/unitreerobotics/unitree_IL_lerobot.git
```

#### Step 2.3: Set Up G1 MuJoCo Model
```bash
# Download G1 MJCF/URDF for MuJoCo
# From: https://github.com/unitreerobotics/unitree_mujoco

git clone https://github.com/unitreerobotics/unitree_mujoco.git
```

#### Step 2.4: Create MuJoCo Inference Script
Create `scripts/policy_inference_mujoco.py`:
- Load MuJoCo environment with G1 model
- Connect to GROOT server (192.168.1.237:5555)
- Use `SimPolicyWrapper` observation format
- Apply actions in MuJoCo physics

#### Step 2.5: Run PnPAppleToPlate Demo
```bash
# Configure GROOT server for manipulation
docker exec groot-server bash -c '
  cd /workspace/gr00t &&
  python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment-tag UNITREE_G1 \
    --port 5555
'

# Run MuJoCo client
python scripts/policy_inference_mujoco.py \
    --server-host 192.168.1.237 \
    --server-port 5555 \
    --task pnp_apple
```

### Expected Observation Space (SimPolicyWrapper)
```python
{
    "video.ego_view": np.array([256, 256, 3]),  # RGB from head camera
    "state.left_arm": np.array([7]),             # Left arm joints
    "state.right_arm": np.array([7]),            # Right arm joints
    "state.left_hand": np.array([6]),            # Left hand joints
    "state.right_hand": np.array([6]),           # Right hand joints
    "state.waist": np.array([3]),                # Torso joints
}
```

### Success Criteria
- G1 reaches for apple
- Successful grasp
- Carries to plate
- Places apple on plate
- Smooth, human-like motion

---

## Track 3: GROOT Fine-Tuning (Overnight Training)

### Objective
Fine-tune GROOT N1.6 on existing datasets for improved G1 manipulation performance.

### Recommended Dataset: GR1 Arms Only ⭐

| Attribute | Value |
|-----------|-------|
| Dataset | `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` |
| Subset | `gr1_arms_only` |
| Size | ~9,000 trajectories |
| Training Time | 8-10 hours |
| Tasks | Threading, LiftTray, Assembly, Transport, Coffee, Pouring |

### Quick Start (Tonight!)

```bash
# SSH to workstation
ssh datamentors@192.168.1.205

# Enter container
docker exec -it isaac-lab bash

# Run overnight fine-tuning
cd /workspace
./dm-isaac-g1/scripts/finetune_groot_overnight.sh gr1_arms_only 0
```

### What Happens
1. Downloads GR1 arms dataset (~50-80GB)
2. Creates UNITREE_G1 embodiment config
3. Fine-tunes `nvidia/GR00T-N1.6-3B` for 5000 steps
4. Saves checkpoints every 1000 steps
5. Completes in ~8-10 hours

### Morning: Deploy Fine-tuned Model

```bash
# Copy checkpoint to Spark
scp -r checkpoints/groot_*/final nvidia@192.168.1.237:/workspace/

# Update GROOT server
docker exec groot-server bash -c '
  pkill -f run_gr00t_server
  cd /workspace/gr00t &&
  python gr00t/eval/run_gr00t_server.py \
    --model-path /workspace/final \
    --embodiment-tag UNITREE_G1 \
    --port 5555
'
```

### Available Datasets

| Dataset | Size | Time | Use Case |
|---------|------|------|----------|
| GR1 Arms Only | 9k | 8-10h | Best for overnight |
| GR1 Arms + Waist | 240k | 24-48h | Weekend training |
| DROID Subset | 16k | 10-12h | Real robot data |

See [FINETUNING_PLAN.md](FINETUNING_PLAN.md) for detailed instructions.

---

## Track 4: Future - RL Training in Isaac Lab

### Objective
Train custom manipulation policies from scratch using reinforcement learning.

### Approaches

#### Option A: PPO/SAC Training
1. Define manipulation tasks in Isaac Lab
2. Train from scratch using PPO/SAC
3. Curriculum: reach → grasp → lift → place

#### Option B: Combined IL + RL
1. Initialize with GROOT weights
2. Fine-tune with RL for specific tasks
3. Use sparse rewards + shaped rewards

#### Option C: Domain Randomization
1. Randomize Isaac Sim physics parameters
2. Train policy robust to physics variations
3. Better sim-to-real transfer

### Implementation (Phase 4)
```bash
# To be developed
./scripts/train_manipulation_rl.sh
./scripts/curriculum_training.sh
```

---

## Infrastructure Setup

### Workstation (192.168.1.205)
```bash
# Current software
- Isaac Sim 5.1.0 (Docker)
- Isaac Lab 2.3.2
- VNC on port 5901

# To install
- MuJoCo (pip install mujoco)
- LeRobot (git clone)
- Unitree MuJoCo models
```

### Spark Server (192.168.1.237)
```bash
# Current setup
- GROOT server on port 5555
- Model: nvidia/GR00T-N1.6-G1-PnPAppleToPlate
- Embodiment: UNITREE_G1

# Can switch models for different tasks
- GR00T-N1.6-3B (general)
- GR00T-N1.6-G1-PnPAppleToPlate (manipulation)
```

---

## File Structure

```
dm-isaac-g1/
├── scripts/
│   ├── policy_inference_groot_g1.py      # Existing (Isaac Sim - needs physics fix)
│   ├── policy_inference_locomotion.py    # NEW: Track 1 - Walking in Isaac
│   ├── policy_inference_mujoco.py        # NEW: Track 2 - Manipulation in MuJoCo
│   └── run_groot_inference.sh            # Updated launcher
├── docs/
│   ├── PHYSICS_ENGINE_LEARNINGS.md       # Why sim-to-sim fails
│   └── IMPLEMENTATION_PLAN.md            # This document
├── configs/
│   ├── locomotion_config.yaml            # Walking policy config
│   └── manipulation_config.yaml          # Manipulation config
└── phases/
    ├── phase1_inference/
    │   ├── locomotion/                   # Track 1 code
    │   └── manipulation/                 # Track 2 code
    └── phase3_finetuning/
        └── isaac_manipulation/           # Future Track 3
```

---

## Timeline & Priority

| Priority | Track | Task | Status |
|----------|-------|------|--------|
| 1 | Track 1 | Set up locomotion inference in Isaac Sim | 🔴 Not started |
| 2 | Track 2 | Install MuJoCo on workstation | 🔴 Not started |
| 3 | Track 2 | Run PnPAppleToPlate in MuJoCo | 🔴 Not started |
| 4 | Track 3 | Plan fine-tuning pipeline | 🔴 Not started |

---

## Quick Reference: Server Configuration

### For Locomotion (Track 1)
No GROOT server needed - uses local Isaac Gym policy files.

### For Manipulation (Track 2)
```bash
# On Spark (192.168.1.237)
docker exec groot-server bash -c '
  pkill -f run_gr00t_server  # Stop existing
  cd /workspace/gr00t &&
  python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment-tag UNITREE_G1 \
    --port 5555
'
```

---

## Validation Checklist

### Track 1: Locomotion
- [ ] Unitree RL repos cloned
- [ ] Pre-trained policy downloaded
- [ ] Isaac Sim scene loads G1
- [ ] Robot walks forward
- [ ] Robot turns left/right
- [ ] Stable on slopes

### Track 2: Manipulation
- [ ] MuJoCo installed
- [ ] G1 MJCF model loaded
- [ ] GROOT server configured
- [ ] Connection to server working
- [ ] Apple pickup working
- [ ] Place on plate working

### Track 3: Fine-tuning (Future)
- [ ] Isaac Lab environment defined
- [ ] Demonstration collection pipeline
- [ ] GROOT fine-tuning script
- [ ] Trained policy works in Isaac
