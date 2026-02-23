# DEVOPS-002: Migration to Unitree-Based Docker Environment

## Status: COMPLETE ✅ (2026-02-19)

## Background

Previous Docker setup (`environments/workstation/Dockerfile`) used `nvcr.io/nvidia/isaac-sim:5.0.0`
as the base image, which caused an `assimp` symbol conflict between Isaac Sim's bundled assimp
library and `cmeel`-built `hpp-fcl` (from pink/pinocchio IK). See DEVOPS-001 for full details.

Unitree's official approach installs Isaac Sim **via pip** inside a clean conda environment
(`unitree_sim_env`) on top of a vanilla CUDA base image. This avoids the bundled library
conflicts entirely.

## New Approach

**File**: `environments/workstation/Dockerfile.unitree`
**Compose**: `environments/workstation/docker-compose.unitree.yml`
**Container name**: `dm-workstation-unitree`

### Base Image Change

| | Previous | New |
|---|---|---|
| Base | `nvcr.io/nvidia/isaac-sim:5.0.0` | `nvidia/cuda:12.8.0-runtime-ubuntu22.04` |
| Isaac Sim | Pre-baked in NVCR image | pip install inside conda env |
| Python | Isaac Sim's bundled `/isaac-sim/python.sh` | Conda Python 3.11 |
| Driver compat | CUDA 12.x | CUDA 12.8 (Blackwell) |

### CUDA Compatibility

- Workstation NVIDIA driver: **12.8** (RTX Blackwell / GeForce RTX 50xx series)
- Docker base image toolkit: **12.8.0**
- PyTorch wheels: **cu126** (CUDA 12.6 build — driver 12.8 is backward compatible)
- Isaac Sim 5.0.0: ships internal CUDA 12.x runtime; uses pip from `pypi.nvidia.com`

### What's Baked into the Image

- `unitree_sim_env` conda environment (Python 3.11)
- PyTorch 2.7.0 (cu126)
- Isaac Sim 5.0.0 (`isaacsim[all,extscache]`)
- IsaacLab v2.2.0 (+ `./isaaclab.sh --install`)
- CycloneDDS 0.10.x (compiled from source)
- `unitree_sdk2_python`
- `unitree_sim_isaaclab` (USD assets, task definitions, G1 configs)
- Isaac-GR00T (GROOT N1.6 inference + fine-tuning)
- TurboVNC 3.1.2 (remote visualization)
- numpy pinned `>=1.26.0,<2.0.0` (Isaac Sim synthetic data pipeline)

### What's Mounted at Runtime

- `/workspace/dm-isaac-g1` — live repo (all scripts, src code)
- `/workspace/datasets` — training datasets (NEVER delete)
- `/workspace/checkpoints` — model weights (NEVER delete)
- `/workspace/Isaac-GR00T` — live override of baked GR00T version
- `/workspace/unitree_sim_isaaclab` — live override for USD/task configs

## Pre-Build: Workstation Cleanup

### Check Current Disk Usage

```bash
source .env
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 "
  echo '=== Disk Usage ==='
  df -h /
  echo ''
  echo '=== Docker disk usage ==='
  docker system df
  echo ''
  echo '=== /tmp usage ==='
  du -sh /tmp/*  2>/dev/null | sort -rh | head -20
"
```

### Safe Docker Cleanup

```bash
source .env
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 "
  echo '--- Removing stopped containers ---'
  docker container prune -f
  echo '--- Removing dangling images (untagged layers) ---'
  docker image prune -f
  echo '--- Removing unused build cache ---'
  docker builder prune -f
  echo '--- Removing unused networks ---'
  docker network prune -f
  echo '--- After cleanup ---'
  docker system df
  df -h /
"
```

> **DO NOT run `docker image prune -a`** — this would delete the previously built `dm-workstation:latest`.

### /tmp Cleanup

```bash
source .env
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 "
  echo '--- /tmp cleanup ---'
  find /tmp -maxdepth 1 -type f -mtime +1 -delete 2>/dev/null
  find /tmp -maxdepth 1 -type d -name 'tmp.*' -mtime +1 -exec rm -rf {} + 2>/dev/null
  find /tmp -maxdepth 1 -type d -name '*.groot*' -exec rm -rf {} + 2>/dev/null
  echo '/tmp cleaned'
  df -h /
"
```

## Build Commands

### 1. Pull Latest Repo on Workstation

```bash
source .env
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 \
    "cd /home/datamentors/dm-isaac-g1 && git pull origin main"
```

### 2. Build Docker Image (on workstation)

```bash
source .env
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 "
  cd /home/datamentors/dm-isaac-g1/environments/workstation
  docker compose -f docker-compose.unitree.yml build --no-cache 2>&1 | tee /tmp/docker_build.log
"
```

Monitor build progress:
```bash
source .env
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 "tail -f /tmp/docker_build.log"
```

### 3. Start Container

```bash
source .env
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 "
  cd /home/datamentors/dm-isaac-g1/environments/workstation
  docker compose -f docker-compose.unitree.yml up -d
"
```

## Test Plan

### Test 1: Isaac Sim Imports (Smoke Test)

```bash
source .env
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 "
  docker exec dm-workstation-unitree conda run -n unitree_sim_env python -c \"
import isaacsim
print('Isaac Sim imported OK')
from isaacsim import SimulationApp
print('SimulationApp available')
\"
"
```

**Expected**: No import errors, prints confirmation lines.

### Test 2: Isaac Lab Import

```bash
source .env
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 "
  docker exec dm-workstation-unitree conda run -n unitree_sim_env python -c \"
import sys
sys.path.insert(0, '/home/code/IsaacLab/source/isaaclab')
import isaaclab
print('IsaacLab version:', isaaclab.__version__)
\"
"
```

**Expected**: IsaacLab version string printed (e.g. `2.2.0`).

### Test 3: GROOT Server Connection

```bash
source .env
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 "
  docker exec dm-workstation-unitree conda run -n unitree_sim_env python -c \"
import sys
sys.path.insert(0, '/workspace/Isaac-GR00T')
import gr00t
print('GR00T imported OK:', gr00t.__version__ if hasattr(gr00t, '__version__') else 'no version attr')
import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.RCVTIMEO = 3000
try:
    sock.connect('tcp://192.168.1.237:5555')
    print('GROOT server reachable at 192.168.1.237:5555')
except Exception as e:
    print('GROOT server not reachable (may not be running):', e)
finally:
    sock.close()
    ctx.term()
\"
"
```

**Expected**: GR00T imports OK. GROOT server reachability depends on whether the Spark server is running.

### Test 4: Full Inference with VNC (Isaac Sim + GROOT)

```bash
source .env
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 "
  docker exec dm-workstation-unitree conda run -n unitree_sim_env bash -c '
    /opt/TurboVNC/bin/vncserver :1 -geometry 1920x1080 -depth 24 2>/dev/null || true
    export DISPLAY=:1
    export PYTHONPATH=/workspace/dm-isaac-g1/src:/home/code/Isaac-GR00T:/home/code/IsaacLab/source/isaaclab:/home/code/IsaacLab/source/isaaclab_tasks:/home/code/IsaacLab/source/isaaclab_assets:/home/code/unitree_sim_isaaclab:\$PYTHONPATH
    export GR00T_STATS=/workspace/checkpoints/groot_g1_inspire_9datasets/processor/statistics.json
    cd /workspace/dm-isaac-g1
    python scripts/policy_inference_groot_g1.py \
      --server 192.168.1.237:5555 \
      --scene pickplace_g1_inspire \
      --language \"pick up the apple\" \
      --enable_cameras \
      --save_debug_frames
  '
"
# Connect VNC: open vnc://192.168.1.205:5901
```

### Test 5: No assimp / Library Conflicts

After inference starts, confirm no symbol errors:
```bash
source .env
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 "
  docker exec dm-workstation-unitree conda run -n unitree_sim_env python -c \"
import sys
sys.path.insert(0, '/home/code/IsaacLab/source/isaaclab')
# Test pink/pinocchio (IK library) — should load without assimp conflict
try:
    import pink
    print('pink OK:', pink.__version__)
except Exception as e:
    print('pink error:', e)
try:
    import pinocchio
    print('pinocchio OK:', pinocchio.__version__)
except Exception as e:
    print('pinocchio error:', e)
\"
"
```

**Expected**: Both pink and pinocchio import cleanly — no `undefined symbol` errors.

### Test 6: GPU Availability

```bash
source .env
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 "
  docker exec dm-workstation-unitree conda run -n unitree_sim_env python -c \"
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
\"
"
```

**Expected**: CUDA available, GPU name shows RTX Blackwell device.

## Success Criteria

- [x] Docker image builds without errors
- [x] Container starts and GPU is accessible (RTX PRO 6000 Blackwell)
- [x] Isaac Sim 5.0.0 imports cleanly via pip (no nvcr base image needed)
- [x] IsaacLab v2.2.0 (0.44.9) imports cleanly
- [x] Isaac-GR00T imports cleanly
- [x] pink 4.0.0 + pinocchio 3.9.0 import without assimp symbol conflicts
- [x] VNC server starts (TurboVNC 3.1.2, :1)
- [x] GROOT server reachable at tcp://192.168.1.237:5555
- [x] No CUDA symbol errors

## ECR Images (2026-02-19)

| Tag | Digest | Description |
|---|---|---|
| `base-latest` | `sha256:4497315fc101...` | Isaac Sim + IsaacLab + unitree + pink/pinocchio |
| `latest` | `sha256:4580a8269c1e...` | Extends base, adds GR00T inference deps |

Registry: `260464233120.dkr.ecr.us-east-1.amazonaws.com/isaac-g1-sim-ft-rl`

## Rollback

If the new image fails, the old `dm-workstation:latest` image is still available:

```bash
# Revert to old container
cd /home/datamentors/dm-isaac-g1/environments/workstation
docker compose up -d  # uses docker-compose.yml + dm-workstation:latest
```
