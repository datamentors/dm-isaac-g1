# GROOT N1.6 Inference Setup Guide

## Overview

This document describes how to set up the GROOT N1.6 inference server on the DGX Spark server (192.168.1.237) for the fine-tuned G1+Inspire model (53 DOF).

## Quick Status

| Component | Status | Location |
|-----------|--------|----------|
| Model files | ✅ Deployed | `/home/nvidia/GR00T/checkpoints/groot-g1-inspire-9datasets/` |
| HuggingFace | ✅ Uploaded | `datamentorshf/groot-g1-inspire-9datasets` |
| Docker container | ✅ Running | `groot-server` |
| Dependencies | ⚠️ Needs uv sync | Complex dependency resolution required |
| Inference server | ⏳ Pending | Waiting on dependencies |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Spark Server                              │
│                       192.168.1.237                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    /home/nvidia/GR00T/                      │ │
│  │  ├── checkpoints/groot-g1-inspire-9datasets/  (10GB model)  │ │
│  │  ├── gr00t/                        (GROOT source code)      │ │
│  │  └── inference_venv/               (Python environment)     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Docker: groot-server                           │ │
│  │              Port: 5555                                     │ │
│  │              Mount: /home/nvidia/GR00T → /workspace/gr00t   │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Model Location

The fine-tuned model is stored at:
- **Spark server**: `/home/nvidia/GR00T/checkpoints/groot-g1-inspire-9datasets/`
- **HuggingFace**: `datamentorshf/groot-g1-inspire-9datasets`

## GR00T N1.6 Action Chunking

GR00T 1.6 uses Flow Matching with a 32-layer Diffusion Transformer (DiT) to predict action chunks. Key characteristics:

- **Action horizon**: Configurable at inference time (8-16 steps recommended)
- **Single-step inference**: `action_horizon=1` causes jittering - avoid in production
- **State-relative actions**: Smoother, more accurate motion
- **Flow matching**: Reconstructs continuous actions from noise

### Control Strategies

1. **Receding Horizon (MPC-style, recommended)**:
   - `action_horizon=16, execute_steps=1`
   - Re-plan every step for maximum robustness
   - Best for dynamic environments

2. **Partial Execution**:
   - `action_horizon=16, execute_steps=8`
   - Execute half, then re-plan
   - Balance between efficiency and robustness

3. **Open-Loop**:
   - `action_horizon=8, execute_steps=8`
   - Execute full trajectory before re-planning
   - Use only in controlled environments

## Docker Container Setup

### Starting the Container

```bash
docker run -d \
  --name groot-server \
  --runtime=nvidia \
  --gpus all \
  -p 5555:5555 \
  -v /home/nvidia/GR00T:/workspace/gr00t \
  -v /home/nvidia/.cache/huggingface:/root/.cache/huggingface \
  --shm-size=32g \
  nvcr.io/nvidia/pytorch:25.04-py3 \
  sleep infinity
```

### Required Dependencies

The GROOT inference requires these Python packages:
- transformers
- safetensors
- torch (with CUDA)
- tyro
- av (pyav)
- fastapi (for HTTP server)
- uvicorn
- pydantic

Install them with:
```bash
docker exec groot-server pip install transformers safetensors tyro av fastapi uvicorn pydantic
```

**Note**: The gr00t package has complex dependencies. Best practice is to use the gr00t virtualenv or install gr00t as a package:
```bash
docker exec groot-server bash -c 'cd /workspace/gr00t && pip install -e .'
```

## Inference Server

### Server Script

The inference server (`groot_inference_server.py`) provides a FastAPI endpoint:

```python
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

policy = Gr00tPolicy(
    embodiment_tag=EmbodimentTag.G1,
    model_path="/workspace/gr00t/checkpoints/groot-g1-inspire-9datasets",
    device="cuda:0",
    strict=False,
)
```

### API Endpoints

- `GET /health` - Health check
- `GET /policy/info` - Model information
- `POST /inference` - Get action from observation

### Inference Request Format

```json
{
    "observation": [53 float values],
    "image": "base64_encoded_image (optional)",
    "image_shape": [H, W, 3],
    "task": "Pick up the red block",
    "action_horizon": 16,
    "execute_steps": 1
}
```

### Response Format

```json
{
    "action": [53 float values],
    "action_horizon": 16,
    "execute_steps": 1
}
```

## Client Usage

From dm-isaac-g1:

```python
from dm_isaac_g1.inference.client import GrootClient

client = GrootClient(host="192.168.1.237", port=5555)

# Receding horizon control (recommended)
action = client.get_action(
    observation=robot_state,  # 53 DOF
    image=camera_image,       # Optional
    task="Pick up the block",
    action_horizon=16,        # Predict 16 steps
    execute_steps=1,          # Execute 1 step
)

# Execute action and get new observation
# Repeat until task complete
```

## Network Configuration

- **Spark Server IP**: 192.168.1.237
- **Inference Port**: 5555
- **Container Name**: groot-server

## Troubleshooting

### Common Issues

1. **Model not loading**: Check if model files exist at checkpoint path
2. **CUDA errors**: Ensure GPU is available and drivers are correct
3. **Import errors**: Install missing dependencies
4. **Connection refused**: Verify container is running and port is exposed

### Checking Server Status

```bash
# Check container
docker ps | grep groot-server

# Check logs
docker exec groot-server cat /tmp/server.log

# Test health endpoint
curl http://192.168.1.237:5555/health
```

## Using uv for Dependency Management

The recommended approach is to use `uv` for managing Python dependencies to ensure consistency between environments.

### Setting Up with uv

```bash
# On Spark server
cd /home/nvidia/GR00T

# Sync dependencies (may need to resolve conflicts)
uv sync

# If there are dependency conflicts, modify pyproject.toml:
# - Remove tensorrt optional dependency if not needed
# - Pin numpy version compatible with other packages
```

### Known Dependency Conflicts

gr00t has complex dependencies that may conflict:
- `numpy>=2.1.0` required by `onnx>=1.20.0` (for tensorrt)
- `numpy==1.26.4` required by other packages

**Workaround**: Use inference without tensorrt optimization:
```bash
uv sync --no-optional-groups
```

## Comparison with Workstation Setup

The workstation (192.168.1.205) uses a working grootenv with these key packages:

| Package | Workstation Version | Notes |
|---------|-------------------|-------|
| torch | 2.7.0a0+nv25.4 | NVIDIA optimized |
| transformers | 5.1.0 | Latest |
| gr00t | 0.1.0 | From /workspace/Isaac-GR00T |
| accelerate | 1.12.0 | For multi-GPU |
| flash_attn | 2.8.3 | Attention optimization |

## Servers Reference

| Server | IP | User | Purpose |
|--------|-----|------|---------|
| Blackwell Workstation | 192.168.1.205 | datamentors | Training, Isaac Sim |
| DGX Spark | 192.168.1.237 | nvidia | Inference |

## Repository Structure

- **dm-isaac-g1** (local): Main project with CLI, inference client
- **dm-groot-inference** (Spark): Server-side inference (consider renaming to dm-spark-groot-inf)
- **GR00T** (Spark): NVIDIA GROOT source code + checkpoints
