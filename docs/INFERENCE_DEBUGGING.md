# GROOT Inference Debugging Guide

## Current Issue: Action Application to Robot

**Status**: WIP - Robot moves but does not complete manipulation tasks

### Problem Description

When running GROOT inference in Isaac Sim with the fine-tuned model:
1. GROOT server receives observations correctly
2. Actions are returned (shape: 1x16x53 for 16-step horizon, 53 DOF)
3. Robot joints move
4. **BUT** the robot does not successfully complete pick-and-place tasks

### Key Question: Absolute vs Delta Actions

The critical question is whether GROOT outputs:
- **Absolute joint positions**: `target = action` (use action directly as target)
- **Delta/relative actions**: `target = current_pos + action` (add action to current position)

#### Evidence for Absolute Actions
- Model config shows `action_configs: rep=ABSOLUTE` for new_embodiment
- Action values range [-0.4, 1.3] which are reasonable joint positions in radians

#### Evidence for Delta Actions
- Model config has `use_relative_action: true`
- Training typically uses relative actions for imitation learning

### Current Action Provider Implementation

Located at: `/workspace/unitree_sim_isaaclab/action_provider/action_provider_groot.py`

```python
# Current implementation sends action directly as target
action = full_action[0]  # First timestep from trajectory
return torch.tensor(action, dtype=torch.float32, device=env.device).unsqueeze(0)
```

### Testing Procedure

1. **Test absolute actions**: Send GROOT output directly as joint targets
   - If robot moves meaningfully toward objects: absolute is correct

2. **Test delta actions**: Add GROOT output to current joint positions
   ```python
   target = current_joint_pos + action_delta
   ```
   - If robot moves more smoothly: delta is correct

3. **Compare trajectory consistency**:
   - Send same observation twice, check if actions are similar
   - Absolute: actions should be similar
   - Delta: deltas should be similar when added to same state

### Debug Logging

Add to action provider to diagnose:
```python
if self._request_count <= 5:
    print(f"Current pos: {current_pos[:10]}")
    print(f"GROOT action: {action[:10]}")
    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
```

### Environment Variables

```bash
GROOT_SERVER_HOST=192.168.1.237
GROOT_SERVER_PORT=5555
```

### Observation Format

GROOT expects:
```python
obs_dict = {
    "state": {"observation.state": np.array([1, 1, 53], dtype=np.float32)},
    "video": {"cam_left_high": np.array([1, 1, 256, 256, 3], dtype=np.uint8)},
    "language": {"task": [["Pick up the red block"]]}
}
```

### Action Output Format

GROOT returns:
```python
response = {
    "action": np.array([1, 16, 53])  # (batch, horizon, dof)
}
# Note: Server returns as list: [{"action": ...}, {}]
```

### Potential Issues to Investigate

1. **Training data format**: What was the action representation during fine-tuning?
2. **Normalization**: Are actions normalized? Do they need denormalization?
3. **Joint ordering**: Does GROOT output match Isaac Sim joint order?
4. **Camera view**: Does inference camera match training camera?
5. **Task prompt**: Does language prompt match training data?

### Files to Check

- Action provider: `/workspace/unitree_sim_isaaclab/action_provider/action_provider_groot.py`
- Model config: `/workspace/checkpoints/groot-g1-gripper-hospitality-7ds/processor_config.json`
- Statistics: `/workspace/checkpoints/groot-g1-gripper-hospitality-7ds/statistics.json`
- Training config: `/workspace/Isaac-GR00T/g1_inspire_unified_config.py`

### Next Steps

1. Verify action format from training data
2. Test with both absolute and delta application
3. Add action scaling if needed
4. Verify camera/observation matching
5. Check task language prompts
