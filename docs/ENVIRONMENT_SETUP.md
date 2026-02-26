# Environment Setup — MuJoCo Eval with WBC

This documents the complete environment setup for running GROOT G1 MuJoCo evaluation
with Whole-Body-Control (WBC) across the workstation and Spark inference server.

## Architecture

```
Workstation (192.168.1.205)              Spark (192.168.1.237)
┌──────────────────────────┐             ┌──────────────────────────┐
│  dm-workstation container│  ZMQ:5555   │  groot-server container  │
│  - MuJoCo simulation     │ ──────────► │  - GROOT inference       │
│  - WBC ONNX policies     │             │  - GPU (model on CUDA)   │
│  - PolicyClient          │             │  - PolicyServer          │
│  - Video recording       │             │  - --use-sim-policy-wrap │
└──────────────────────────┘             └──────────────────────────┘
```

## 1. Spark Inference Server (192.168.1.237)

**SSH:** `ssh nvidia@192.168.1.237` (password: `datamentors`)

### Container: `groot-server`

- **Base image:** NVIDIA GR00T inference container
- **Python:** 3.10.x
- **GPU:** Required (CUDA)

### Key packages

| Package        | Version                       | Notes                         |
|---------------|-------------------------------|-------------------------------|
| gr00t          | 0.1.0                         | From /workspace/gr00t (editable) |
| torch          | 2.7.0a0+79aa17489c.nv25.4    | NVIDIA custom build           |
| numpy          | 1.26.4                        |                               |
| transformers   | 4.51.3                        |                               |
| pyzmq          | 26.4.0                        | For PolicyServer ZMQ          |

### Starting the server

```bash
# SSH into Spark
ssh nvidia@192.168.1.237

# Start/restart the groot-server container
docker start groot-server
docker exec -it groot-server bash

# Launch inference server (inside container)
cd /workspace/gr00t
python gr00t/eval/run_gr00t_server.py \
    --model-path /workspace/checkpoints/groot-g1-gripper-hospitality-7ds \
    --embodiment-tag UNITREE_G1 \
    --use-sim-policy-wrapper \
    --port 5555
```

### Available checkpoints on Spark (`/home/nvidia/GR00T/checkpoints/`)

| Checkpoint | Description | Hand type |
|-----------|-------------|-----------|
| `groot-g1-gripper-hospitality-7ds` | Our 7-dataset hospitality model | 1-DOF gripper |
| `groot-g1-gripper-fold-towel-full` | Towel folding specific model | 1-DOF gripper |
| `GR00T-N1.6-G1-PnPAppleToPlate` | NVIDIA pre-trained apple PnP | 7-DOF dexterous |
| `GR00T-N1.6-3B` | NVIDIA base model | — |

## 2. Workstation (192.168.1.205)

**SSH:** `ssh datamentors@192.168.1.205` (password: `datamentors`)

### Container: `dm-workstation`

- **Python:** 3.13.11
- **No GPU required** (ONNX runs on CPU, GROOT inference is remote)

### Key packages

| Package               | Version      | Notes                                   |
|----------------------|--------------|------------------------------------------|
| mujoco               | 3.2.6        | MuJoCo physics simulation               |
| onnxruntime          | 1.24.2       | WBC ONNX policy inference (CPU)          |
| torch                | 2.10.0       | Used by gr00t data processing            |
| numpy                | 2.3.5        |                                          |
| opencv-python-headless | 4.13.0.92  | Video recording, image processing        |
| pyzmq                | 27.1.0       | PolicyClient ZMQ communication           |
| PyYAML               | 6.0.3        | WBC config loading                       |
| gr00t                | 0.1.0        | Isaac-GR00T (editable install)           |
| gr00t_wbc            | 0.1.0        | WBC package (editable, from submodule)   |
| robosuite            | 1.5.1        | Robot simulation (for apple PnP WBC env) |
| robocasa             | 0.2.0        | Kitchen environments (for apple PnP)     |

### Install commands (inside dm-workstation container)

```bash
# Core dependencies (most already in the container)
pip install mujoco==3.2.6
pip install onnxruntime  # Added for WBC ONNX inference
pip install pyzmq numpy opencv-python-headless PyYAML

# gr00t (Isaac-GR00T project — editable install)
pip install -e /workspace/Isaac-GR00T --no-deps

# WBC (GR00T-WholeBodyControl submodule)
cd /workspace/Isaac-GR00T
git submodule update --init external_dependencies/GR00T-WholeBodyControl
pip install -e external_dependencies/GR00T-WholeBodyControl --no-deps

# For apple PnP eval only (requires robosuite + robocasa):
pip install -e external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/gr00trobosuite
pip install -e external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/gr00trobocasa
```

### WBC ONNX Resources

Located at: `/workspace/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1/`

| File | Description |
|------|-------------|
| `policy/GR00T-WholeBodyControl-Balance.onnx` (1.9 MB) | Standing/balancing policy |
| `policy/GR00T-WholeBodyControl-Walk.onnx` (1.9 MB) | Walking locomotion policy |
| `g1_gear_wbc.yaml` | WBC config (PD gains, scales, defaults) |
| `g1_gear_wbc.xml` | G1 29-DOF MuJoCo model for WBC |

### MuJoCo Menagerie

```bash
# Clone if not present
git clone https://github.com/google-deepmind/mujoco_menagerie.git /workspace/mujoco_menagerie

# Towel scene symlink
ln -sf /workspace/dm-isaac-g1/scripts/eval/mujoco_towel_scene/g1_towel_folding.xml \
       /workspace/mujoco_menagerie/unitree_g1/g1_towel_folding.xml
```

## 3. Running Evaluations

### Towel Folding with WBC (recommended)

```bash
# Inside dm-workstation container on workstation
# IMPORTANT: Use g1_gripper_towel_folding.xml (has hand joints for gripper state)
#            NOT g1_towel_folding.xml (has no hands — sends 0.0 for gripper state)
python /workspace/dm-isaac-g1/scripts/eval/run_mujoco_towel_eval_wbc.py \
    --scene /workspace/dm-isaac-g1/scripts/eval/mujoco_towel_scene/g1_gripper_towel_folding.xml \
    --wbc-dir /workspace/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1 \
    --host 192.168.1.237 --port 5555 \
    --max-steps 1500 \
    --output-dir /tmp/mujoco_towel_eval_wbc
```

### Towel Folding without WBC (fixed base)

```bash
# Use gripper scene for correct hand state
python /workspace/dm-isaac-g1/scripts/eval/run_mujoco_towel_eval.py \
    --scene /workspace/dm-isaac-g1/scripts/eval/mujoco_towel_scene/g1_gripper_towel_folding.xml \
    --host 192.168.1.237 --port 5555 \
    --max-steps 1500 \
    --output-dir /tmp/mujoco_towel_eval
```

### Apple Pick-and-Place (requires NVIDIA dexterous model)

```bash
# Must use Python 3.10 venv for robocasa compatibility
source /workspace/wbc_uv_env/.venv/bin/activate

# Must load GR00T-N1.6-G1-PnPAppleToPlate on Spark (7-DOF dexterous hands)
python /workspace/Isaac-GR00T/gr00t/eval/rollout_policy.py \
    --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \
    --policy_client_host 192.168.1.237 \
    --policy_client_port 5555 \
    --n_episodes 5 --n_envs 1
```

## 4. Known Issues

- **Gripper vs dexterous hands:** Our models (7ds, fold-towel) use 1-DOF grippers.
  The apple PnP env expects 7-DOF dexterous hands. Only NVIDIA's pre-trained
  `GR00T-N1.6-G1-PnPAppleToPlate` works with apple PnP.
- **Python 3.13 + robocasa:** robocasa has frozen dataclass issues on Python 3.13.
  Use the Python 3.10 UV venv at `/workspace/wbc_uv_env/.venv` for robocasa tasks.
- **YAML config filenames:** The `g1_gear_wbc.yaml` references `ft92.onnx`/`ft109.onnx`
  but the actual files are `GR00T-WholeBodyControl-Balance.onnx` / `Walk.onnx`.
  Our WBC script uses the correct filenames directly.
