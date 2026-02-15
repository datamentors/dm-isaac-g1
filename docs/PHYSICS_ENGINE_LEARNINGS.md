# Physics Engine Compatibility: Key Learnings

## Executive Summary

**Critical Discovery**: GROOT N1.6 policies trained on one physics engine (MuJoCo) do not transfer directly to another physics engine (Isaac Sim/PhysX). This is **sim-to-sim** transfer failure, which can be harder than sim-to-real in some cases.

## The Problem We Encountered

### Symptoms
When running `GROOT-N1.6-G1-PnPAppleToPlate` (pick-and-place policy) in Isaac Sim:
- Joint values "exploded" to extreme positions (±3.14 radians)
- Action deltas grew progressively larger over time
- Step 0: small deltas (~0.05-0.12)
- Step 50+: huge deltas (~3.0+)
- Robot arms drifted uncontrollably despite correct observation/action formats

### Root Cause
The `PnPAppleToPlate` checkpoint was trained in **MuJoCo**, not Isaac Sim. MuJoCo and PhysX (Isaac's physics engine) have fundamentally different:
- Contact dynamics
- Joint friction models
- Gravity compensation
- Numerical integration methods
- Constraint solving approaches

The policy learned to compensate for MuJoCo's specific physics quirks. When deployed in Isaac/PhysX, those compensations become destabilizing.

## Why NVIDIA Uses MuJoCo for Manipulation

1. **Industry Standard**: MuJoCo is the de-facto standard for manipulation research
2. **Contact-Rich Tasks**: MuJoCo excels at contact-rich manipulation (grasping, placing)
3. **Real Robot Validation**: Easier to validate against real hardware
4. **Faster Iteration**: Lighter weight than full Isaac Sim
5. **Existing Datasets**: Most imitation learning datasets (ALOHA, DROID, etc.) use MuJoCo

## What Works Where

| Task Type | Physics Engine | Why |
|-----------|---------------|-----|
| **Locomotion** | Isaac Sim | Trained in Isaac Gym (same PhysX engine) |
| **Walking/Running** | Isaac Sim | Unitree policies use Isaac Gym |
| **Pick-and-Place** | MuJoCo | GROOT manipulation trained on MuJoCo |
| **Dexterous Manipulation** | MuJoCo | Contact dynamics matter |

## The Sim-to-Sim vs Sim-to-Real Paradox

**Key Insight**: Sim-to-sim can actually be **harder** than sim-to-real because:

1. **Real World Has Noise**: Real sensors/actuators have natural variation that helps generalization
2. **Simulators Are Precisely Wrong**: Each simulator is deterministically wrong in its own unique way
3. **Domain Randomization Works for Real**: You can randomize physics params to cover real-world variation
4. **No Randomization Covers Another Sim**: You can't randomize to match another simulator's specific errors

### How Sim-to-Real Works
```
Training Sim → Domain Randomization → Real World
     ↓              ↓                    ↓
  MuJoCo     Mass, friction,      Natural variation
             delay, noise          falls within
                                   randomized range
```

### Why Sim-to-Sim Fails
```
Training Sim → Target Sim
     ↓             ↓
  MuJoCo       Isaac/PhysX
  Specific     Specific but
  dynamics     DIFFERENT dynamics
              (not in training distribution)
```

## Recommended Architecture

### Two-Track Approach

```
┌─────────────────────────────────────────────────────────────┐
│                    GROOT N1.6 Inference                      │
├─────────────────────────────┬───────────────────────────────┤
│      LOCOMOTION TRACK       │     MANIPULATION TRACK        │
│                             │                               │
│  ┌─────────────────────┐    │   ┌─────────────────────┐     │
│  │    Isaac Sim        │    │   │      MuJoCo         │     │
│  │    (PhysX)          │    │   │                     │     │
│  └─────────────────────┘    │   └─────────────────────┘     │
│           ↓                 │            ↓                  │
│  ┌─────────────────────┐    │   ┌─────────────────────┐     │
│  │ Unitree G1 Walking  │    │   │ GROOT PnPApple     │     │
│  │ Isaac Gym Policies  │    │   │ GROOT Manipulation │     │
│  └─────────────────────┘    │   └─────────────────────┘     │
│           ↓                 │            ↓                  │
│  Works: Same engine        │   Works: Same engine          │
│  (Isaac Gym → Isaac Sim)   │   (MuJoCo → MuJoCo)           │
└─────────────────────────────┴───────────────────────────────┘
```

### Future: Unified Isaac Lab Training

To run manipulation in Isaac Sim, you would need to:
1. Fine-tune GROOT on Isaac Sim physics
2. Or train from scratch in Isaac Lab with RL
3. Or use imitation learning with Isaac Sim demonstrations

## Technical Details

### GROOT Observation Format (SimPolicyWrapper)
```python
{
    "video.ego_view": np.array([H, W, 3]),  # RGB image
    "state.left_arm": np.array([7]),         # 7 joint positions
    "state.right_arm": np.array([7]),        # 7 joint positions
    "state.left_hand": np.array([6]),        # 6 hand joints
    "state.right_hand": np.array([6]),       # 6 hand joints
    "state.waist": np.array([3]),            # 3 torso joints
}
```

### Action Output
- 30 timesteps predicted per inference
- 8-30 timesteps typically executed before next inference
- Actions are **state-relative deltas** (not absolute positions)
- Apply as: `new_position = current_position + action_delta`

### Isaac Sim PD Control Settings
```python
# Arm joints (very stiff)
stiffness = 3000.0
damping = 10.0

# These settings don't match MuJoCo's more compliant control
# Contributing to the transfer failure
```

## Files Modified During Investigation

### dm-isaac-g1/scripts/policy_inference_groot_g1.py
- Added `--action_scale` parameter (0.1 default) to dampen actions
- Added `--num_action_steps` parameter for trajectory control
- Implemented trajectory tracking with `current_timestep`
- Built `build_flat_observation()` for SimPolicyWrapper format
- Tried both RELATIVE and ABSOLUTE action application

### Key Learnings for Code
1. **Observation format matters**: GROOT expects flat dictionary, not nested
2. **Action scaling doesn't fix physics mismatch**: Slows drift but doesn't solve root cause
3. **Check training environment**: Always verify what physics engine was used for training

## References

- [Isaac-GR00T Repository](https://github.com/NVIDIA/Isaac-GR00T)
- [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym) - Isaac Gym based
- [Unitree IL LeRobot](https://github.com/unitreerobotics/unitree_IL_lerobot) - MuJoCo based
- [LeRobot](https://github.com/huggingface/lerobot) - MuJoCo manipulation
- [IsaacLabEvalTasks](https://github.com/NVlabs/IsaacLabEvalTasks) - Isaac Lab benchmarks

## Next Steps

1. **Locomotion in Isaac Sim**: Use Unitree's Isaac Gym policies
2. **Manipulation in MuJoCo**: Install MuJoCo, run GROOT PnP there
3. **Future Fine-tuning**: Train GROOT on Isaac Sim physics if needed
4. **RL Training**: Use Isaac Lab for custom manipulation policies
