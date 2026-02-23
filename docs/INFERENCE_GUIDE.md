# GROOT N1.6 Inference Guide

Deploying and running GROOT N1.6 models fine-tuned with the UNITREE_G1 gripper embodiment.

## Architecture Overview

```
┌─────────────────┐     ZMQ      ┌─────────────────┐
│   Robot/Client   │◄────────────►│  GROOT Server   │
│   (Blackwell)   │   Port 5555  │    (Spark)      │
│ 192.168.1.205   │              │ 192.168.1.237   │
└─────────────────┘              └─────────────────┘
        │                                │
        ▼                                ▼
   Observations                    Fine-tuned
   (ego_view + state)              GROOT Model
        │                                │
        └───────► Actions ◄──────────────┘
              (30-step trajectory)
```

## Prerequisites

1. **GROOT Server** running on Spark (192.168.1.237) in `groot-server` container
2. **Fine-tuned model** — currently `groot-g1-gripper-hospitality-7ds`
3. Network access between client and Spark on port 5555

## Current Deployment

| | |
|---|---|
| **Server** | Spark (192.168.1.237), `groot-server` container |
| **Model** | [datamentorshf/groot-g1-gripper-hospitality-7ds](https://huggingface.co/datamentorshf/groot-g1-gripper-hospitality-7ds) |
| **Embodiment** | `UNITREE_G1` (pre-registered in Isaac-GR00T) |
| **Port** | 5555 (ZMQ) |
| **Action horizon** | 30 steps |
| **State** | 31 DOF flat vector |
| **Action** | 23 DOF flat vector (arms RELATIVE, grippers/waist/nav ABSOLUTE) |
| **Camera** | 1 ego-view (`observation.images.ego_view`) |

## Quick Start

### 1. Start/Restart GROOT Server (on Spark)

```bash
# SSH to Spark
ssh nvidia@192.168.1.237

# Check server health
docker exec groot-server bash -c 'curl -s http://localhost:5555/health || echo "not running"'

# Restart if needed
docker restart groot-server

# Or manually start with a different model
docker exec -it groot-server bash
cd /workspace/Isaac-GR00T
python gr00t/eval/run_gr00t_server.py \
    --model-path /workspace/checkpoints/groot-g1-gripper-hospitality-7ds \
    --embodiment-tag UNITREE_G1 \
    --port 5555
```

### 2. Switch Models on Spark

```bash
# Download a different model from HuggingFace
docker exec groot-server bash -c "
    huggingface-cli download datamentorshf/groot-g1-gripper-fold-towel-full \
        --local-dir /workspace/checkpoints/groot-g1-gripper-fold-towel-full
"

# Update the model path in .env or docker-compose.yml and restart
docker restart groot-server
```

### 3. Test Connection

```bash
# From workstation
docker exec dm-workstation bash -c "
    python -c \"
from gr00t.policy.server_client import PolicyClient
client = PolicyClient(host='192.168.1.237', port=5555, strict=False)
config = client.get_modality_config()
print('Connected! Modality config:', list(config.keys()))
\"
"
```

## Observation Format (UNITREE_G1)

The UNITREE_G1 embodiment expects observations as a flat dictionary:

```python
observation = {
    # Camera image: (B, T, H, W, C) uint8 — 1 ego-view
    "video.ego_view": np.ndarray,  # shape (1, 1, 480, 640, 3)

    # State: (B, T, 31) float32 — flat vector
    "state.observation.state": np.ndarray,

    # Language: list of strings
    "annotation.human.task_description": ["fold the towel"],
}
```

State vector layout (31 DOF):
| Index | Component | DOF |
|-------|-----------|-----|
| 0-5 | Left Leg | 6 |
| 6-11 | Right Leg | 6 |
| 12-14 | Waist | 3 |
| 15-21 | Left Arm | 7 |
| 22-28 | Right Arm | 7 |
| 29 | Left Gripper | 1 |
| 30 | Right Gripper | 1 |

## Action Format (UNITREE_G1)

GROOT returns a 30-step action trajectory:

```python
action = {
    # Action: (B, 30, 23) float32 — flat vector
    "action": np.ndarray,
}
```

Action vector layout (23 DOF):
| Index | Component | DOF | Representation |
|-------|-----------|-----|----------------|
| 0-2 | Waist | 3 | ABSOLUTE |
| 3-9 | Left Arm | 7 | RELATIVE |
| 10-16 | Right Arm | 7 | RELATIVE |
| 17 | Left Gripper | 1 | ABSOLUTE |
| 18 | Right Gripper | 1 | ABSOLUTE |
| 19 | Base Height | 1 | ABSOLUTE |
| 20-22 | Navigate (VX, VY, AngZ) | 3 | ABSOLUTE |

**RELATIVE arms**: actions are deltas from the trajectory start position.
**ABSOLUTE grippers/waist/nav**: actions are direct position/velocity targets.

## GROOT Server API

```python
from gr00t.policy.server_client import PolicyClient

# Connect
client = PolicyClient(host="192.168.1.237", port=5555, strict=False)

# Get modality config
config = client.get_modality_config()

# Get action
action_dict, info = client.get_action(observation)

# Reset (call at episode boundaries)
client.reset()
```

## Deploying a New Model

After fine-tuning (see [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md)):

```bash
# 1. Upload model to HuggingFace (from workstation)
mkdir -p /workspace/checkpoints/model-upload
cp /workspace/checkpoints/<model>/checkpoint-10000/model-*.safetensors \
   /workspace/checkpoints/<model>/checkpoint-10000/config.json \
   /workspace/checkpoints/<model>/checkpoint-10000/generation_config.json \
   /workspace/checkpoints/model-upload/

huggingface-cli upload datamentorshf/<model-name> \
    /workspace/checkpoints/model-upload . \
    --repo-type model --private

# 2. Download on Spark
docker exec groot-server bash -c "
    huggingface-cli download datamentorshf/<model-name> \
        --local-dir /workspace/checkpoints/<model-name>
"

# 3. Update config and restart
docker restart groot-server
```

## Available Models

| Model | HuggingFace | Use Case |
|-------|-------------|----------|
| **groot-g1-gripper-hospitality-7ds** | [datamentorshf/groot-g1-gripper-hospitality-7ds](https://huggingface.co/datamentorshf/groot-g1-gripper-hospitality-7ds) | Multi-task (fold, clean, wipe, fruit, medicine, tools, pingpong) |
| groot-g1-gripper-fold-towel-full | [datamentorshf/groot-g1-gripper-fold-towel-full](https://huggingface.co/datamentorshf/groot-g1-gripper-fold-towel-full) | Towel folding specialist |
| groot-g1-gripper-fold-towel | [datamentorshf/groot-g1-gripper-fold-towel](https://huggingface.co/datamentorshf/groot-g1-gripper-fold-towel) | Towel folding (partial, 6000 steps) |

## Troubleshooting

### Connection Refused
```bash
# Check server is listening
ssh nvidia@192.168.1.237 "docker exec groot-server netstat -tlpn | grep 5555"

# Check container is running
ssh nvidia@192.168.1.237 "docker ps | grep groot-server"

# Restart
ssh nvidia@192.168.1.237 "docker restart groot-server"
```

### Robot Not Moving / Wrong Actions
1. Verify embodiment tag is `UNITREE_G1` (not `NEW_EMBODIMENT`)
2. Check state vector is 31 DOF and action is 23 DOF
3. Verify camera key is `ego_view` (not `cam_left_high`)
4. Ensure language prompt matches training data tasks

### Model Loading Error
```bash
# Check model files exist
docker exec groot-server ls /workspace/checkpoints/groot-g1-gripper-hospitality-7ds/

# Re-download if needed
docker exec groot-server bash -c "
    huggingface-cli download datamentorshf/groot-g1-gripper-hospitality-7ds \
        --local-dir /workspace/checkpoints/groot-g1-gripper-hospitality-7ds
"
```

## References

- [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) — Train your own model
- [Isaac-GR00T Repository](https://github.com/NVIDIA/Isaac-GR00T)
- [GROOT N1.6 Model Card](https://huggingface.co/nvidia/GR00T-N1.6-3B)
