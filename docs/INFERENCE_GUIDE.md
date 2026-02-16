# GROOT N1.6 Inference Guide

This guide covers running GROOT inference for G1+Inspire robot manipulation in Isaac Sim.

## Architecture Overview

```
┌─────────────────┐     ZMQ      ┌─────────────────┐
│   Isaac Sim     │◄────────────►│  GROOT Server   │
│   (Blackwell)   │   Port 5555  │    (Spark)      │
│ 192.168.1.205   │              │ 192.168.1.237   │
└─────────────────┘              └─────────────────┘
        │                                │
        ▼                                ▼
   Observations                    Fine-tuned
   (camera, state)                 GROOT Model
        │                                │
        └───────► Actions ◄──────────────┘
                (16-step trajectory)
```

## Prerequisites

1. **GROOT Server** running on Spark (192.168.1.237)
2. **Isaac Sim** running on Blackwell (192.168.1.205)
3. **Fine-tuned model** checkpoint with statistics.json
4. **VNC access** to visualize the simulation

## Quick Start

### 1. Start GROOT Server (on Spark)

```bash
# SSH to Spark
ssh nvidia@192.168.1.237

# Start inference container
cd /home/nvidia/dm-groot-inference
docker compose up -d

# Or start manually
docker exec -it groot-inference bash
cd /workspace/gr00t
python gr00t/eval/run_gr00t_server.py \
    --model-path /workspace/checkpoints/groot-g1-inspire-9datasets \
    --embodiment-tag NEW_EMBODIMENT \
    --port 5555
```

### 2. Start Isaac Sim (on Blackwell)

```bash
# SSH to workstation
ssh datamentors@192.168.1.205

# Enter Isaac Sim container
docker exec -it isaac-sim bash

# Set environment
export PYTHONPATH=/workspace/Isaac-GR00T:/workspace/IsaacLab/source/isaaclab:$PYTHONPATH
export GR00T_STATS=/workspace/checkpoints/groot-g1-inspire-9datasets/statistics.json

# Run inference
/isaac-sim/python.sh /workspace/IsaacLab/scripts/policy_inference_groot_g1.py \
    --server 192.168.1.237:5555 \
    --language "pick up the red block" \
    --enable_cameras \
    --num_action_steps 16 \
    --action_scale 0.1
```

### 3. View in VNC

Connect to `192.168.1.205:5901` with a VNC client to see the simulation.

## Inference Script Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--server` | required | GROOT server endpoint (host:port) |
| `--language` | "pick up the object" | Task instruction |
| `--num_action_steps` | 30 | Steps to execute per inference |
| `--action_scale` | 0.1 | Scale factor for actions |
| `--video_h` | 64 | Camera image height |
| `--video_w` | 64 | Camera image width |
| `--max_episode_steps` | 1000 | Max steps before reset |

## Observation Format

The inference script sends observations to GROOT in flat dictionary format:

```python
observation = {
    # Camera image: (B, T, H, W, C) uint8
    "video.ego_view": np.ndarray,

    # State vectors: (B, T, D) float32
    "state.left_leg": np.ndarray,    # (B, T, 6)
    "state.right_leg": np.ndarray,   # (B, T, 6)
    "state.waist": np.ndarray,       # (B, T, 3)
    "state.left_arm": np.ndarray,    # (B, T, 7)
    "state.right_arm": np.ndarray,   # (B, T, 7)
    "state.left_hand": np.ndarray,   # (B, T, 12)
    "state.right_hand": np.ndarray,  # (B, T, 12)

    # Language: list of strings
    "annotation.human.task_description": ["task"] * B,
}
```

## Action Format

GROOT returns a 16-step action trajectory:

```python
action = {
    # Each action: (B, 16, D) float32
    "action.waist": np.ndarray,      # (B, 16, 3)
    "action.left_arm": np.ndarray,   # (B, 16, 7)
    "action.right_arm": np.ndarray,  # (B, 16, 7)
    "action.left_hand": np.ndarray,  # (B, 16, 12)
    "action.right_hand": np.ndarray, # (B, 16, 12)
}
```

### Action Application

Actions are applied as **relative deltas** from the trajectory start position:

```python
# For each timestep t in [0, 15]:
target_position = trajectory_start_position + action_delta[t] * action_scale
```

This means:
- `action[0]` = delta to reach first target
- `action[15]` = delta to reach final target
- All deltas are relative to the position when inference was called

## Inference Loop

```
┌──────────────────────────────────────────────────────────────┐
│                     Inference Loop                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Get current joint positions                              │
│  2. Capture camera image                                     │
│  3. Build observation dict                                   │
│  4. Send to GROOT server                     ◄─── Every 16   │
│  5. Receive 16-step trajectory                    steps      │
│  6. Execute steps 0-15:                                      │
│     - target = start_pos + delta[t] * scale                 │
│     - Send to robot                                          │
│  7. Repeat from step 1                                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## GROOT Server API

### Connect to Server

```python
from gr00t.policy.server_client import PolicyClient

client = PolicyClient(
    host="192.168.1.237",
    port=5555,
    api_token=None,  # Optional
    strict=False
)
```

### Get Modality Config

```python
modality_cfg = client.get_modality_config()
# Returns dict with video, state, action modality configs
```

### Get Action

```python
action_dict, info = client.get_action(observation)
# observation: flat dict with video, state, language
# action_dict: flat dict with action trajectories
```

### Reset

```python
client.reset()
# Call at episode start/end
```

## Troubleshooting

### Robot Not Moving

1. Check GROOT server is running: `docker logs groot-inference`
2. Verify network connectivity: `ping 192.168.1.237`
3. Check action scale - try `--action_scale 1.0`

### Robot Moving But Not Completing Task

1. Verify camera is capturing: Check `--enable_cameras`
2. Check language prompt matches training data
3. Verify statistics.json matches model checkpoint

### Connection Refused

```bash
# Check server is listening
ssh nvidia@192.168.1.237 "netstat -tlpn | grep 5555"

# Restart server if needed
docker restart groot-inference
```

### Actions Look Wrong

1. Check action format (absolute vs relative)
2. Verify joint ordering matches training
3. Enable debug logging in inference script

## Example: Pick and Place

```bash
# Start server with pick-place model
docker exec -it groot-inference bash
python gr00t/eval/run_gr00t_server.py \
    --model-path /workspace/checkpoints/groot-g1-inspire-9datasets \
    --embodiment-tag NEW_EMBODIMENT \
    --port 5555

# Run simulation with pick-place task
/isaac-sim/python.sh /workspace/IsaacLab/scripts/policy_inference_groot_g1.py \
    --server 192.168.1.237:5555 \
    --language "pick up the red block and place it on the table" \
    --enable_cameras \
    --num_action_steps 16 \
    --action_scale 0.1 \
    --max_episode_steps 2000
```

## Server Response Format

**Important**: The GROOT server returns responses in a list format:

```python
# Server returns:
[{"action": {...}}, {}]

# Extract first element:
if isinstance(response, list):
    response = response[0]
```

## Environment Variables

```bash
# Required
export GR00T_STATS=/path/to/statistics.json
export PYTHONPATH=/workspace/Isaac-GR00T:$PYTHONPATH

# Optional
export GROOT_SERVER_HOST=192.168.1.237
export GROOT_SERVER_PORT=5555
```

## Files Reference

| File | Location | Purpose |
|------|----------|---------|
| Inference script | `/workspace/IsaacLab/scripts/policy_inference_groot_g1.py` | Main inference loop |
| GROOT server | `/workspace/gr00t/gr00t/eval/run_gr00t_server.py` | Model server |
| Statistics | `/workspace/checkpoints/*/statistics.json` | State normalization |
| Model checkpoint | `/workspace/checkpoints/*/checkpoint-*` | Fine-tuned weights |

## References

- [INFERENCE_DEBUGGING.md](INFERENCE_DEBUGGING.md) - Debug action application issues
- [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) - Train your own model
- [Isaac-GR00T Repository](https://github.com/NVIDIA/Isaac-GR00T)
