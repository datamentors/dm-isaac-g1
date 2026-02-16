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
# Edit .env and add GITHUB_TOKEN

# 3. Build and start container
cd environments/workstation
docker compose up -d

# 4. Sync venv inside container
docker exec -it dm-workstation bash
cd /workspace/dm-isaac-g1
uv sync
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
