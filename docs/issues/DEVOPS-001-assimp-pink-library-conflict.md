# DevOps Issue: Isaac Sim + Pink/Pinocchio Library Conflict

**Priority:** HIGH
**Assigned To:** DevOps Team
**Date Created:** 2026-02-18
**Status:** Open

## Summary

The current Docker environment cannot run GROOT inference with unitree_sim_isaaclab manipulation scenes due to a **library symbol conflict between Isaac Sim's bundled assimp and the cmeel-built hpp-fcl/pinocchio libraries**. This blocks all G1 manipulation scene testing.

## Goal

Get a working Docker environment with:
- Isaac Sim 5.0.0
- IsaacLab v2.2.0
- unitree_sim_isaaclab (G1 manipulation scenes)
- GROOT DDC (inference client)
- pink/pinocchio for IK (required by unitree scenes)

**We had this working before** - the goal is to replicate a working setup.

---

## Error Details

### The Error
```
ImportError: /isaac-sim/kit/python/lib/python3.11/site-packages/cmeel.prefix/lib/libhpp-fcl.so:
undefined symbol: _ZNK6Assimp8IOSystem16CurrentDirectoryB5cxx11Ev
```

### Root Cause
1. **Isaac Sim bundles its own assimp library** at `/isaac-sim/extscache/omni.kit.asset_converter-*/libassimp.so`
2. **cmeel/pinocchio also bundles assimp** at `/isaac-sim/kit/python/lib/python3.11/site-packages/cmeel.prefix/lib/libassimp.so`
3. **hpp-fcl is compiled against cmeel's assimp** but at runtime, Isaac Sim's assimp gets loaded first
4. The two assimp versions have **different ABI symbols**, causing the undefined symbol error

### Why This Happens
- `unitree_sim_isaaclab/tasks/g1_tasks/` imports `from pink.tasks import FrameTask` at module load time
- pink imports pinocchio
- pinocchio loads hpp-fcl
- hpp-fcl needs assimp symbols that don't match what's loaded

---

## What Was Tried (All Failed)

### 1. Installing pin-pink + hpp-fcl via pip
```bash
/isaac-sim/python.sh -m pip install pin-pink hpp-fcl
```
**Result:** Installs but conflicts with Isaac Sim's assimp at runtime.

### 2. LD_PRELOAD to force cmeel assimp first
```bash
export LD_PRELOAD=/isaac-sim/kit/python/lib/python3.11/site-packages/cmeel.prefix/lib/libassimp.so
```
**Result:** Doesn't resolve - Isaac Sim internals still use their own assimp.

### 3. Using IsaacLab's env_isaaclab virtual environment
```bash
# The ./isaaclab.sh --install creates /workspace/IsaacLab/env_isaaclab/
# with its own pink/pinocchio
```
**Result:** Same conflict - the compiled libraries still link against incompatible assimp.

### 4. Patching unitree_sim_isaaclab to lazy-load pink modules
**Result:** Doesn't work because `tasks/__init__.py` auto-imports all submodules.

### 5. Blacklisting pink-dependent modules in unitree_sim_isaaclab
**Result:** Would require blacklisting most manipulation scenes.

---

## Recommended Approach: Follow Unitree's Docker Setup

Unitree has working Docker environments that include all these libraries. Their approach should be investigated:

### Unitree Resources
1. **unitree_sim_isaaclab Dockerfile**: https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/main/Dockerfile
2. **Isaac Sim 5.0.0 install guide**: https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/main/doc/isaacsim5.0_install.md
3. **Their requirements_full.txt**: Contains their tested dependency versions

### Key Differences to Investigate
- How does Unitree install pink/pinocchio without conflicts?
- Do they use a different base image?
- Do they compile libraries from source with matching ABI?
- Do they use conda instead of pip?

---

## Current Environment State

### Docker Image
- **Base:** `nvcr.io/nvidia/isaac-sim:5.0.0`
- **IsaacLab:** v2.2.0 (built into image via `./isaaclab.sh --install`)
- **Container:** `dm-workstation`

### What Works
- Isaac Sim starts and renders
- Cameras work with `--enable_cameras`
- GROOT server connection works (192.168.1.237:5555)
- Scene USD files load correctly
- warp imports correctly

### What Doesn't Work
- Any unitree_sim_isaaclab G1 manipulation scene (requires pink)
- `from pink.tasks import FrameTask` - crashes with assimp symbol error

---

## Files Affected

### Dockerfile Location
```
dm-isaac-g1/environments/workstation/Dockerfile
```

### Key Configuration Files
```
dm-isaac-g1/environments/workstation/docker-compose.yml
dm-isaac-g1/environments/workstation/pyproject.toml
```

### Inference Script
```
dm-isaac-g1/scripts/policy_inference_groot_g1.py
```

### unitree_sim_isaaclab (mounted)
```
/workspace/unitree_sim_isaaclab/tasks/g1_tasks/  # All G1 manipulation scenes
```

---

## Environment Variables Required

```bash
PROJECT_ROOT=/workspace/unitree_sim_isaaclab  # Required for USD asset paths
GROOT_SERVER_HOST=192.168.1.237
GROOT_SERVER_PORT=5555
DISPLAY=:1  # For VNC visualization
```

---

## Testing Commands

### Test Pink Import (Currently Fails)
```bash
docker exec dm-workstation /isaac-sim/python.sh -c "from pink.tasks import FrameTask; print('SUCCESS')"
```

### Test Full Inference (Currently Fails)
```bash
docker exec dm-workstation bash -c '
export DISPLAY=:1
export PROJECT_ROOT=/workspace/unitree_sim_isaaclab
cd /workspace/dm-isaac-g1
/isaac-sim/python.sh scripts/policy_inference_groot_g1.py \
    --server 192.168.1.237:5555 \
    --scene pickplace_redblock_g1_inspire \
    --language "pick up the red block" \
    --enable_cameras
'
```

---

## Success Criteria

1. `from pink.tasks import FrameTask` works without errors
2. All G1 manipulation scenes load without assimp conflicts
3. GROOT inference runs end-to-end with camera input
4. Docker image can be pushed to ECR for deployment

---

## Additional Context

### Library Versions in Container
```
cmeel                    0.59.0
cmeel-assimp             5.4.3.1
cmeel-boost              1.83.0
hpp-fcl                  2.4.4
pin-pink                 4.0.0
pinocchio                (bundled in cmeel.prefix)
```

### Workstation Details
- **IP:** 192.168.1.205
- **GPU:** NVIDIA RTX PRO 6000 Blackwell (98GB)
- **Container:** dm-workstation
- **VNC:** Port 5901

### GROOT Server (DGX Spark)
- **IP:** 192.168.1.237
- **Port:** 5555
- **Status:** Running and reachable

---

## References

- [IsaacLab Docker Deployment Guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html)
- [Unitree Sim IsaacLab](https://github.com/unitreerobotics/unitree_sim_isaaclab)
- [cmeel (CMake Wheels)](https://github.com/cmake-wheel/cmeel) - The build system causing conflicts
- [Isaac Sim 5.0.0 Release Notes](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/)
