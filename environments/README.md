# Environment Management with UV

This folder contains environment configurations for different deployment targets. We use [UV](https://github.com/astral-sh/uv) for fast, reproducible Python dependency management.

## Overview

| Environment | Target | Architecture | Purpose |
|------------|--------|--------------|---------|
| **workstation** | Blackwell (192.168.1.205) | x86_64 + NVIDIA GPU | Training + Simulation |
| **spark** | Spark (192.168.1.237) | ARM64 (Jetson/Orin) | Inference only |

## Design Principles

1. **Single Repository**: Both local development and workstation use the same `dm-isaac-g1` repo
2. **UV for Dependencies**: Fast, reproducible installs with lockfiles
3. **Docker for Isolation**: Container images with mounted volumes for code
4. **Git for Sync**: Workstation clones repo with GitHub token for pull/push

## Environment Structure

```
environments/
├── README.md                    # This file
├── workstation/                 # Blackwell workstation (training + sim)
│   ├── pyproject.toml          # UV project config
│   ├── uv.lock                 # Locked dependencies (generated)
│   ├── Dockerfile              # Container image
│   └── setup.sh                # Environment setup script
└── spark/                       # Spark inference server (ARM64)
    ├── pyproject.toml          # UV project config
    ├── uv.lock                 # Locked dependencies (generated)
    ├── Dockerfile              # Container image (ARM64)
    └── setup.sh                # Environment setup script
```

## Quick Start

### Workstation Setup

```bash
# 1. Clone repo on workstation
git clone https://github.com/datamentors/dm-isaac-g1.git
cd dm-isaac-g1

# 2. Create .env with GitHub token
cp .env.example .env
# Edit .env and add GITHUB_TOKEN, HF_TOKEN

# 3. Build and start container
cd environments/workstation
docker compose build  # Rebuild to get latest fixes
docker compose up -d

# 4. Fetch Unitree robot assets
docker exec dm-workstation bash -c "cd /workspace/unitree_sim_isaaclab && bash fetch_assets.sh"

# 5. Test camera setup (verifies Isaac Sim 5.1.0 workarounds)
docker exec dm-workstation /isaac-sim/python.sh scripts/test_camera_stable.py --headless

# 6. Check output
ls /tmp/groot_debug/  # Should contain camera_*.png
```

### Spark Setup

```bash
# 1. Clone repo on Spark
git clone https://github.com/datamentors/dm-isaac-g1.git
cd dm-isaac-g1

# 2. Build ARM64 container
cd environments/spark
docker compose up -d

# 3. Sync venv inside container
docker exec -it dm-spark-inference bash
cd /workspace/dm-isaac-g1
uv sync
```

## UV Commands Reference

```bash
# Install UV (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
uv sync

# Add a package
uv add package-name

# Add dev dependency
uv add --dev package-name

# Run Python with venv
uv run python script.py

# Lock dependencies (after editing pyproject.toml)
uv lock

# Update all packages
uv lock --upgrade

# Show installed packages
uv pip list
```

## Unified Dependencies

Both workstation and spark environments share core dependencies from the root `pyproject.toml`. Environment-specific dependencies are added via optional groups:

```toml
# Root pyproject.toml
[project.optional-dependencies]
training = [
    # Training-specific deps
]
inference = [
    # Inference-specific deps
]
simulation = [
    # Isaac Sim/Lab deps
]
```

## Git Workflow

### Workstation Git Setup

The workstation should have a GitHub token configured for pulling/pushing:

```bash
# On workstation, create ~/.git-credentials
echo "https://oauth2:${GITHUB_TOKEN}@github.com" > ~/.git-credentials
git config --global credential.helper store

# Or use .env in the repo
echo "GITHUB_TOKEN=ghp_xxx" >> .env
```

### Sync Workflow

```bash
# Local (Mac) → GitHub → Workstation
git push origin main
ssh workstation "cd /workspace/dm-isaac-g1 && git pull"

# Workstation → GitHub → Local (Mac)
ssh workstation "cd /workspace/dm-isaac-g1 && git push origin main"
git pull
```

## Container Volumes

Both containers mount the repo as a volume for live code editing:

```yaml
volumes:
  - /home/datamentors/dm-isaac-g1:/workspace/dm-isaac-g1
  - /workspace/datasets:/workspace/datasets
  - /workspace/checkpoints:/workspace/checkpoints
```

This means:
- Code changes are immediately available in container
- No need to rebuild container for code changes
- UV venv is inside container, not on host

## Key Dependencies

### Robot Communication (DDS + Unitree SDK)

The workstation environment includes CycloneDDS and the Unitree SDK for real robot communication:

- **CycloneDDS**: Built from source (releases/0.10.x) with Python bindings
- **unitree_sdk2py**: Installed from GitHub with manual directory copy (pip bug workaround)

These are installed in the **Dockerfile** (not pyproject.toml) because:
- CycloneDDS requires cmake build from source
- unitree_sdk2py has a known pip bug where subdirectories (b2, g1, h1, comm) aren't copied

### IK Libraries (Pink/Pinocchio)

Isaac Sim 5.1.0 uses NumPy 2.x, which requires updated IK libraries:

- **pin-pink 4.0.0+**: Supports NumPy 2.x
- **pinocchio 3.9.0+**: Compiled for NumPy 2.x

These are installed directly into Isaac Sim's Python environment in the Dockerfile:
```dockerfile
RUN /isaac-sim/python.sh -m pip install --upgrade pin-pink
```

**Note**: Older IsaacLab environments (env_isaaclab) have pin-pink 3.1.0 compiled for NumPy 1.x, which crashes with Isaac Sim's NumPy 2.x. The Dockerfile installs the updated versions directly.

### Unitree Robot Assets

The robot USD files must be fetched from HuggingFace:

```bash
# Inside the container
cd /workspace/unitree_sim_isaaclab
bash fetch_assets.sh
```

This downloads the G1 robot USD files to `/workspace/unitree_sim_isaaclab/assets/`.

**Available G1 USD files** (after fetching):
- `g1-29dof_wholebody_inspire/` - G1 with Inspire hands (for GROOT)
- `g1-29dof_wholebody_dex3/` - G1 with DEX3 hands
- `g1-29dof-inspire-base-fix-usd/` - G1 Inspire with base fixes

**Camera Mount Links** (from USD inspection):
- `d435_link` - Intel RealSense D435 head camera (only in full Unitree USDs)
- `torso_link` - Fallback for IsaacLab tasks (no d435_link)
- `left_hand_camera_base_link` / `right_hand_camera_base_link` - Wrist cameras

**Note**: IsaacLab task environments use simplified robot USDs that don't include the `d435_link`. Always use the fallback camera configuration for IsaacLab tasks.

## Container Persistence

**Important**: Use `docker compose stop/start` instead of `down/up` when possible:

```bash
# PREFERRED - preserves container state (installed packages)
docker compose stop && docker compose start

# ONLY if needed - recreates container (loses pip installs not in Dockerfile)
docker compose down && docker compose up -d
```

Packages installed via pip during development are lost when container is recreated with `down/up`.

## Isaac Sim 5.1.0 Known Issues & Workarounds

The Dockerfile applies several workarounds for known Isaac Sim 5.1.0 issues. See:
- [Official Known Issues](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/known_issues.html)
- [Unitree Installation Guide](https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/main/doc/isaacsim5.0_install.md)

### Camera/Synthetic Data Stability

Isaac Sim 5.1.0 has issues with camera capture and synthetic data generation. The following workarounds are applied:

1. **Disable Async Rendering** - Prevents frame skipping with Replicator
2. **DLSS Quality Mode** - Required for resolutions below 600x600
3. **Increased rt_subframes** - Ensures materials load before capture

To run scripts with all workarounds:
```bash
# Method 1: Use stable wrapper (created in Dockerfile)
docker exec dm-workstation /usr/local/bin/isaac-python-stable scripts/test_camera_stable.py

# Method 2: Use flags directly
docker exec dm-workstation /isaac-sim/python.sh \
  --/exts/isaacsim.core.throttling/enable_async=false \
  --/rtx/post/dlss/mode=2 \
  scripts/your_script.py
```

### Test Camera Setup

A stable camera test script is provided:
```bash
docker exec dm-workstation /isaac-sim/python.sh scripts/test_camera_stable.py

# Headless mode (no VNC needed)
docker exec dm-workstation /isaac-sim/python.sh scripts/test_camera_stable.py --headless

# Custom output directory
docker exec dm-workstation /isaac-sim/python.sh scripts/test_camera_stable.py \
  --output-dir /tmp/my_output
```

The test script outputs images to `/tmp/groot_debug/` by default.

### Camera Segmentation Faults

If you see segfaults in `libomni.syntheticdata.plugin.so`:

1. **Ensure proper initialization order** - SimulationApp → World → Scene → Camera
2. **Render frames before capture** - At least 30-60 frames to let physics settle
3. **Use USD camera API** - More stable than sensor.Camera for some workflows

Example stable initialization:
```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "anti_aliasing": 0})

# Apply workarounds
import carb.settings
settings = carb.settings.get_settings()
settings.set("/exts/isaacsim.core.throttling/enable_async", False)
settings.set("/rtx/post/dlss/mode", 2)

# Create world BEFORE importing other modules
from omni.isaac.core import World
world = World()
world.scene.add_default_ground_plane()

# Load robot...
# Create camera...

# Reset and render many frames
world.reset()
for _ in range(60):
    world.step(render=True)
```

## Troubleshooting

### NumPy Version Mismatch
If you see errors like "A module compiled using NumPy 1.x cannot be run in NumPy 2.x":
```bash
# Fix by reinstalling pin-pink for NumPy 2
docker exec dm-workstation /isaac-sim/python.sh -m pip install --upgrade pin-pink
```

### unitree_sdk2py Import Errors
If importing unitree_sdk2py fails with "cannot import b2" or similar:
```bash
# Manually copy missing directories
docker exec dm-workstation bash -c "
  SITE=$(/isaac-sim/python.sh -c 'import site; print(site.getsitepackages()[0])')
  for dir in b2 g1 h1 comm; do
    cp -r /opt/unitree_sdk2_python/unitree_sdk2py/\$dir \$SITE/unitree_sdk2py/
  done
"
```

### Camera "Unable to write from unknown dtype" Error
This error occurs with TiledCameraCfg in Isaac Sim 5.1.0. Solutions:
```bash
# Option 1: Use stable test script (recommended)
docker exec dm-workstation /isaac-sim/python.sh scripts/test_camera_stable.py

# Option 2: Use Replicator API instead of TiledCameraCfg
# See scripts/test_camera_stable.py for example

# Option 3: Disable async rendering
docker exec dm-workstation /isaac-sim/python.sh \
  --/exts/isaacsim.core.throttling/enable_async=false \
  your_script.py
```

### Git Permission Issues
If git commands fail with permission errors:
```bash
# Fix ownership (run on workstation, not in container)
sudo chown -R datamentors:datamentors /home/datamentors/dm-isaac-g1/.git
```

### Container Keeps Stopping
If the container stops when running scripts:
1. Check Docker logs: `docker logs dm-workstation`
2. Monitor memory: scripts may be OOMing
3. Use headless mode to reduce GPU memory: `--headless`
4. Run scripts in tmux to survive disconnects
