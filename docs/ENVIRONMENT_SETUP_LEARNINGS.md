# Environment Setup Learnings & Action Items

## Session: 2026-02-15

### Key Discovery: Environment Consolidation Required

**Problem**: The Blackwell workstation (192.168.1.205) currently has multiple Python environments that create dependency conflicts and reproducibility issues:

1. **Isaac Sim embedded Python** (`/isaac-sim/python.sh`) - Used by Isaac Lab
2. **grootenv conda environment** (`/workspace/grootenv`) - Created for GROOT training
3. **Various pip installs directly in container** - Not reproducible

### Action Items

#### Priority 1: Consolidate to Single Environment

The recommended approach is to use Isaac Sim's embedded Python for everything:

```bash
# All installs should use:
/isaac-sim/python.sh -m pip install <package>

# NOT:
pip install <package>  # System pip
conda install <package>  # Conda
```

**Rationale**: Isaac Lab extensions are compiled against Isaac Sim's Python. Using a different Python can cause ABI mismatches.

#### Priority 2: Create Reproducible Setup Script

A setup script should be created that can be run after container restart to restore all dependencies.

**Location**: `/workspace/unitree_sim_isaaclab/setup_env.sh`

```bash
#!/bin/bash
# Reproducible environment setup for unitree_sim_isaaclab

set -e

echo "=== Setting up unitree_sim_isaaclab environment ==="

# 1. Build CycloneDDS (required for unitree_sdk2py)
export CYCLONEDDS_HOME=/workspace/cyclonedds/install
if [ ! -d "$CYCLONEDDS_HOME" ]; then
    echo "Building CycloneDDS..."
    cd /workspace
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
    cd cyclonedds && mkdir -p build install && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../install
    cmake --build . --target install -j$(nproc)
fi

# 2. Install unitree_sdk2py
if ! /isaac-sim/python.sh -c "import unitree_sdk2py" 2>/dev/null; then
    echo "Installing unitree_sdk2_python..."
    cd /workspace
    git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
    cd unitree_sdk2_python
    /isaac-sim/python.sh -m pip install -e .
fi

# 3. Install requirements
echo "Installing unitree_sim_isaaclab requirements..."
/isaac-sim/python.sh -m pip install -r /workspace/unitree_sim_isaaclab/requirements.txt

# 4. Install additional dependencies
/isaac-sim/python.sh -m pip install \
    gymnasium \
    flatdict \
    h5py \
    aiortc \
    aiohttp \
    pin-pink \
    "qpsolvers[open_source_solvers]"

echo "=== Environment setup complete ==="
```

#### Priority 3: Download Required Assets

The unitree_sim_isaaclab requires USD assets (~1.2GB) from HuggingFace:

```bash
cd /workspace/unitree_sim_isaaclab
bash fetch_assets.sh
```

This downloads the warehouse scenes, robot models, and object assets needed for simulation.

---

## Technical Details

### GROOT Server Response Format

**Discovery**: The GROOT inference server (running on Spark 192.168.1.237:5555) returns responses in a list format:

```python
# Server returns:
[{"action": np.ndarray(shape=(1, 16, 53))}, {}]

# NOT:
{"action": np.ndarray(shape=(1, 16, 53))}
```

**Fix Applied**: Modified `action_provider_groot.py` to handle list responses:

```python
# Handle list response from GROOT server (response is [dict, dict])
if isinstance(response, list) and len(response) > 0:
    response = response[0]  # Extract first dict from list
```

### Required PYTHONPATH

When running unitree_sim_isaaclab, the following paths must be set:

```bash
export PYTHONPATH=/workspace/unitree_sim_isaaclab:\
/workspace/unitree_sim_isaaclab/teleimager/src:\
/workspace/IsaacLab/source/isaaclab:\
/workspace/IsaacLab/source/isaaclab_tasks:\
/workspace/IsaacLab/source/isaaclab_rl:\
/workspace/IsaacLab/source/isaaclab_assets:\
$PYTHONPATH
```

### Git Submodules

The teleimager submodule must be initialized:

```bash
cd /workspace/unitree_sim_isaaclab
git submodule update --init --recursive
```

---

## Dependencies Installed This Session

All installed via `/isaac-sim/python.sh -m pip install`:

| Package | Version | Purpose |
|---------|---------|---------|
| cyclonedds | 0.10.2 | DDS middleware (built from source) |
| unitree_sdk2py | 1.0.1 | Unitree robot SDK |
| gymnasium | latest | RL environment interface |
| flatdict | latest | Flatten nested dicts |
| h5py | latest | HDF5 file handling |
| aiortc | latest | WebRTC for teleimager |
| aiohttp | latest | Async HTTP |
| pin-pink | latest | Inverse kinematics |
| qpsolvers | latest | QP solvers for IK |
| onnxruntime | 1.22.1 | ONNX model inference |
| pyzmq | 27.0.0 | ZeroMQ for GROOT communication |
| logging_mp | 0.1.5 | Multiprocessing logging |
| cmake | latest | Build system (for CycloneDDS) |
| gcc/build-essential | latest | C compiler (for pynput/evdev) |

---

## Known Issues

### 1. Material/Shader Warnings

When loading warehouse scenes, you may see errors like:
```
[Error] [omni.hydra] Unable to find SdrShaderNode for prim...
```

**Impact**: Visual only - materials may appear as default colors
**Solution**: These are NVIDIA Digital Twin assets not included locally. Can be ignored for functionality.

### 2. WebRTC Errors

```
WebRTC Thread Error: [Errno 2] No such file or directory
```

**Impact**: Image streaming via WebRTC won't work
**Solution**: Not critical for local VNC visualization

### 3. Zombie Processes

After simulation crashes, zombie Python processes may remain:
```bash
docker restart isaac-sim  # Clean restart
```

---

## VNC Visualization Checklist

1. Ensure VNC server is running on workstation:
   ```bash
   ps aux | grep Xvnc
   ```

2. Connect via VNC client to `192.168.1.205:5901`

3. In Isaac Sim window:
   - Click on viewport to activate
   - Navigate: PerspectiveCamera -> Cameras -> PerspectiveCamera
   - Use mouse to pan/zoom

---

## Next Steps

1. **Create Dockerfile** with all dependencies pre-installed
2. **Create requirements.txt** capturing all pip packages
3. **Test with different G1+Inspire tasks**:
   - `Isaac-PickPlace-Cylinder-G129-Inspire-Joint`
   - `Isaac-PickPlace-RedBlock-G129-Inspire-Joint`
   - `Isaac-Move-Cylinder-G129-Inspire-Wholebody`
4. **Validate GROOT model actions** - check if robot moves correctly
