# Workstation Environment Map

**Blackwell Workstation: 192.168.1.205**

This document maps every mounted folder, Python environment, and critical environment variable used during inference and training. It explains why they exist, what they contain, and how they interact — including the root cause of past issues where a missing package in one env cascaded into another.

---

## 1. Docker Container Overview

Two containers are active for inference:

| Container | Image | Purpose |
|-----------|-------|---------|
| `dm-workstation` | `dm-workstation:latest` (groot) | Isaac Sim + inference runtime |
| `dm-workstation-base` | `dm-workstation-base:latest` | Base environment for training |
| `isaac-sim` | `nvcr.io/nvidia/isaac-sim:5.1.0` | Legacy NVCR Isaac Sim (team reference) |
| `isaaclab_arena:cuda_gr00t` | Team's custom image | Team changes (separate) |

The active inference container is always `dm-workstation`.

---

## 2. Volume Mounts

Defined in `environments/workstation/docker-compose.unitree.yml`.

### Current Mounts

```yaml
volumes:
  - /home/datamentors/dm-isaac-g1:/workspace/dm-isaac-g1
  - /workspace/datasets:/workspace/datasets
  - /workspace/datasets_inspire:/workspace/datasets_inspire
  - /workspace/checkpoints:/workspace/checkpoints
  - /workspace/Isaac-GR00T:/workspace/Isaac-GR00T
  - /home/datamentors/unitree_sim_isaaclab:/workspace/unitree_sim_isaaclab
  - /dev/dri:/dev/dri
```

### Mount Purpose Table

| Host Path | Container Path | What It Contains | Why Mounted |
|-----------|---------------|-----------------|-------------|
| `/home/datamentors/dm-isaac-g1` | `/workspace/dm-isaac-g1` | This repo (inference scripts, configs) | Live code sync — changes on Mac push via git, container picks up immediately without rebuild |
| `/workspace/datasets` | `/workspace/datasets` | LeRobot/GROOT training datasets | Large files (>100GB) — can't be inside container image |
| `/workspace/datasets_inspire` | `/workspace/datasets_inspire` | Inspire-hand specific datasets | Same reason — too large for image |
| `/workspace/checkpoints` | `/workspace/checkpoints` | Fine-tuned model checkpoints + statistics.json | Shared between training and inference; survives container recreation |
| `/workspace/Isaac-GR00T` | `/workspace/Isaac-GR00T` | NVIDIA Isaac-GR00T framework source | Enables importing `gr00t.*` packages; mounted to allow code edits without image rebuild |
| `/home/datamentors/unitree_sim_isaaclab` | `/workspace/unitree_sim_isaaclab` | Unitree Isaac Sim/Lab scenes, USD assets | Contains robot scene configs and USD asset files (PackingTable, robot models) |
| `/dev/dri` | `/dev/dri` | GPU DRI device nodes | Required for GPU rendering in Isaac Sim |

### What Is NOT Mounted (Lives Inside the Image)

| Internal Path | What It Contains |
|--------------|-----------------|
| `/opt/conda/envs/unitree_sim_env/` | The conda Python environment with Isaac Sim + IsaacLab |
| `/home/code/IsaacLab/` | IsaacLab source code (editable install in conda env) |
| `/home/code/unitree_sim_isaaclab/` | Secondary copy of Unitree scenes (from image build time) |
| `/opt/TurboVNC/` | TurboVNC server for VNC display |

**Critical note**: `/home/code/IsaacLab` is baked into the image (from the base `unitree_sim_env` conda env build). When you import `isaaclab`, the conda env resolves to `/home/code/IsaacLab` — **not** `/workspace/IsaacLab`. See Section 4 below.

---

## 3. Python Environment Architecture

The container has **one conda environment** for inference:

### `unitree_sim_env` (conda)

- **Location**: `/opt/conda/envs/unitree_sim_env/`
- **Python**: 3.11
- **Isaac Sim**: Installed as pip package (`isaacsim`)
- **IsaacLab**: Editable install from `/home/code/IsaacLab/` (baked into image)
- **Key packages**: `isaacsim`, `isaaclab`, `isaaclab_tasks`, `flatdict`, `pink`, `pinocchio`

#### How to run inside it

```bash
# From outside the container:
docker exec dm-workstation conda run -n unitree_sim_env python script.py

# From inside the container:
conda activate unitree_sim_env && python script.py

# Or using the inference command pattern:
conda run --no-capture-output -n unitree_sim_env python scripts/policy_inference_groot_g1.py
```

### There Is No `/isaac-sim/python.sh`

The old NVCR Isaac Sim image (`nvcr.io/nvidia/isaac-sim:5.1.0`) used `/isaac-sim/python.sh` as its Python entry point. **Our `dm-workstation` image does NOT have this.** Isaac Sim is installed as a Python package inside the conda env. Always use `conda run -n unitree_sim_env python` instead.

---

## 4. The IsaacLab Duality Problem

This is the source of many confusing issues.

### Two IsaacLab Locations

| Path | Where It Came From | Who Uses It |
|------|-------------------|-------------|
| `/home/code/IsaacLab/` | Baked into the image during build | `unitree_sim_env` conda env (via `.pth` file) |
| `/workspace/IsaacLab/` | Mounted from host at runtime | NOT used by conda env |

When you `import isaaclab` inside `unitree_sim_env`, Python resolves it to `/home/code/IsaacLab/` — even if `/workspace/IsaacLab` exists and you add it to `PYTHONPATH`.

**Why**: The conda env has `/home/code/IsaacLab/source/isaaclab` registered as an editable install (`.pth` file). These take priority in `sys.path`.

**Impact**: Any code modifications made to `/workspace/IsaacLab` will NOT be picked up by the conda env. To use the workspace copy, you would need to remove the editable install from the conda env (not recommended — this is what the image was built with).

### Verification

```bash
# Check which IsaacLab is actually imported:
docker exec dm-workstation conda run -n unitree_sim_env python -c \
  "import isaaclab; print(isaaclab.__file__)"
# Output: /home/code/IsaacLab/source/isaaclab/isaaclab/__init__.py
```

---

## 5. Critical Environment Variables

### For Inference

| Variable | Value | Why Required |
|----------|-------|-------------|
| `DISPLAY` | `:1` | Points Isaac Sim to TurboVNC display |
| `PROJECT_ROOT` | `/workspace/unitree_sim_isaaclab` | Used by Unitree scene configs to locate USD assets. **Without this, scenes crash with `FileNotFoundError: USD file not found at path 'None/assets/...'`** |
| `GR00T_STATS` | `/workspace/checkpoints/groot_g1_inspire_9datasets/processor/statistics.json` | State normalization statistics for the GROOT model |
| `PYTHONPATH` | `/workspace/dm-isaac-g1/src:/workspace/Isaac-GR00T:$PYTHONPATH` | Adds our scripts and GROOT framework to path |

### Why PROJECT_ROOT Matters

In `/workspace/unitree_sim_isaaclab/tasks/common_scene/base_scene_pickplace_cylindercfg.py`:

```python
import os
project_root = os.environ.get("PROJECT_ROOT")

# Later used in USD paths:
usd_path=f"{project_root}/assets/objects/PackingTable/PackingTable.usd"
```

If `PROJECT_ROOT` is not set, `project_root` is `None`, and the path becomes `None/assets/...`, causing an immediate crash during scene loading.

**Set it before every inference run:**

```bash
export PROJECT_ROOT=/workspace/unitree_sim_isaaclab
```

---

## 6. Mounted vs Baked Assets

The `unitree_sim_isaaclab` scenes reference USD assets in two ways:

### Local Assets (from mount)
```python
# Uses PROJECT_ROOT — found in /workspace/unitree_sim_isaaclab/assets/
usd_path=f"{project_root}/assets/objects/PackingTable/PackingTable.usd"
```
These exist at `/workspace/unitree_sim_isaaclab/assets/` and are accessible because of the mount.

### Nucleus/Remote Assets
```python
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd"
```
These reference NVIDIA's Nucleus server. On first run they are cached locally. If Nucleus is unreachable, the scene may fail or show empty warehouse.

---

## 7. Issues Fixed and Their Root Causes

### Issue 1: `ModuleNotFoundError: No module named 'flatdict'`

**Symptom**: Isaac Sim loads but crashes during extension startup with `flatdict` import error.

**Root cause**: `flatdict` is listed in isaaclab's dependencies but was not installed in the `unitree_sim_env` conda env.

**How the cascade worked**:
1. Isaac Sim starts and loads extensions
2. Extension `isaaclab_tasks` imports `isaaclab`
3. `isaaclab.sim.simulation_context` imports `flatdict`
4. `flatdict` not found → extension fails → Isaac Sim crashes

**Fix**: `conda run -n unitree_sim_env pip install flatdict`

**Permanent fix**: Add `flatdict` to the Dockerfile's pip install step for `unitree_sim_env`.

---

### Issue 2: `FileNotFoundError: USD file not found at path 'None/assets/...'`

**Symptom**: Scene fails to load immediately after env creation.

**Root cause**: `PROJECT_ROOT` environment variable not set. The Unitree scene configs use `os.environ.get("PROJECT_ROOT")` to build USD file paths.

**Fix**: Always set `export PROJECT_ROOT=/workspace/unitree_sim_isaaclab` before running inference.

**Permanent fix**: Add this to the inference launch command or the container's entrypoint.

---

### Issue 3: `ValueError: Invalid action shape, expected: 53, received: 41`

**Symptom**: Simulation starts, camera loads, first action received from GROOT, then crashes on `env.step()`.

**Root cause**: The action vector was initialized from `joint_pos[:, action_joint_ids]` where `action_joint_ids` has 41 entries (waist + arms + hands), but `env.step()` expects a 53-DOF tensor (all robot joints including legs).

**How the cascade worked**:
1. GROOT outputs 53-DOF actions (or they get mapped to 41 via `action_name_to_index`)
2. `current_action_vec = joint_pos[:, action_joint_ids].copy()` creates a `(1, 41)` array
3. `current_action_vec[:, :env_action_dim]` clips to 53 — but array is already smaller, so no effect
4. `env.step(action_tensor)` receives `(1, 41)` but expects `(1, 53)`

**Fix** (committed in `80c9af7`):
```python
# Before (wrong):
current_action_vec = joint_pos[:, action_joint_ids].copy()  # shape (1, 41)

# After (correct):
current_action_vec = joint_pos[:, :env_action_dim].copy()  # shape (1, 53)
```
Also changed `_apply_group` to use `group_joint_ids[key]` (robot joint indices into the 53-DOF array) instead of `action_name_to_index` (which indexed into the old 41-element array).

**Why the 53 vs 41**: The env action space includes all 53 joints (6 left_leg + 6 right_leg + 3 waist + 7 left_arm + 7 right_arm + 12 left_hand + 12 right_hand). GROOT only predicts waist + arms + hands (41 joints) — legs stay at current position. The fix initializes the action vector from current joint positions (all 53) and only overwrites the joints GROOT controls.

---

### Issue 4: Stale X11 Locks Blocking VNC

**Symptom**: After container restart, `vncserver :1` fails with "A VNC server is already running as :1" but no server actually running.

**Root cause**: `/tmp/.X11-unix` was mounted from host. X lock files (`/tmp/.X1-lock`, `/tmp/.X11-unix/X1`) persisted across container restarts in the host filesystem.

**Fix**: Removed `/tmp/.X11-unix:/tmp/.X11-unix` mount from `docker-compose.unitree.yml`.

---

## 8. Complete Inference Launch Command

The canonical command to start inference from the host machine (Mac):

```bash
source /path/to/dm-isaac-g1/.env

sshpass -p "$WORKSTATION_PASSWORD" ssh -o StrictHostKeyChecking=no datamentors@192.168.1.205 '
# Start VNC if needed
docker exec dm-workstation /opt/TurboVNC/bin/vncserver :1 -geometry 1920x1080 -depth 24 2>/dev/null || true

# Launch inference
docker exec -d dm-workstation bash -c "
  export DISPLAY=:1
  export PROJECT_ROOT=/workspace/unitree_sim_isaaclab
  export GR00T_STATS=/workspace/checkpoints/groot_g1_inspire_9datasets/processor/statistics.json
  export PYTHONPATH=/workspace/dm-isaac-g1/src:/workspace/Isaac-GR00T:\$PYTHONPATH
  cd /workspace/dm-isaac-g1
  conda run --no-capture-output -n unitree_sim_env python scripts/policy_inference_groot_g1.py \
    --server 192.168.1.237:5555 \
    --scene pickplace_g1_inspire \
    --language \"pick up the cylinder\" \
    --num_action_steps 30 \
    --action_scale 0.1 \
    --enable_cameras \
    > /tmp/inference.log 2>&1
"
echo "Inference launched. Log: docker exec dm-workstation tail -f /tmp/inference.log"
'

# Check log:
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 \
  "docker exec dm-workstation tail -f /tmp/inference.log"

# Connect VNC to see simulation:
open vnc://192.168.1.205:5901
```

---

## 9. Image Gaps (Manual Installs Not in Dockerfile)

The following packages were installed manually in the running container and need to be added to `Dockerfile.unitree` for a permanent fix:

| Package | Install Command | When Needed |
|---------|----------------|-------------|
| `flatdict==4.1.0` | `pip install flatdict` | Always (isaaclab dependency) |
| `xfce4 xfce4-terminal xfce4-taskmanager` | `apt install` | VNC desktop (already in Dockerfile now) |

### Adding flatdict to Dockerfile

In `environments/workstation/Dockerfile.unitree`, add to the conda env setup:

```dockerfile
# Install missing isaaclab runtime dependencies
RUN conda run -n unitree_sim_env pip install flatdict
```

Or set `PROJECT_ROOT` in the Dockerfile ENV block:

```dockerfile
ENV PROJECT_ROOT=/workspace/unitree_sim_isaaclab
```

---

## 10. Finding Issues Quickly

### Inference crashed — where to look?

```bash
# Check the inference log
docker exec dm-workstation tail -100 /tmp/inference.log

# Check if process is still running
docker exec dm-workstation ps aux | grep python | grep -v grep

# Check Isaac Sim extension errors (look for [Error] lines)
docker exec dm-workstation grep "\[Error\]" /tmp/inference.log | tail -20
```

### VNC not connecting?

```bash
# Check VNC is running
docker exec dm-workstation ps aux | grep vnc

# Check display
docker exec dm-workstation bash -c "echo $DISPLAY"

# Restart VNC
docker exec dm-workstation /opt/TurboVNC/bin/vncserver -kill :1 2>/dev/null || true
docker exec dm-workstation /opt/TurboVNC/bin/vncserver :1 -geometry 1920x1080 -depth 24
```

### Missing module error?

```bash
# Install into unitree_sim_env
docker exec dm-workstation conda run -n unitree_sim_env pip install <package>
```
