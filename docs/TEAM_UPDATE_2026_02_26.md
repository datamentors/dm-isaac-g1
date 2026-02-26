# Team Update — MuJoCo Simulation Eval Pipeline
**Date:** 2026-02-26

## Summary

We built a full MuJoCo closed-loop evaluation pipeline for our fine-tuned GROOT G1 models, connecting the workstation (192.168.1.205) to the Spark inference server (192.168.1.237). We also integrated NVIDIA's Whole-Body-Control (WBC) ONNX balance policies. The pipeline works end-to-end, but the robot does not interact with the towel in the scene — the arms make small movements but don't reach down to the table surface.

---

## What Was Done

### 1. MuJoCo Eval Pipeline (Towel Folding)

- Built `run_mujoco_towel_eval.py` — a custom MuJoCo eval script that:
  - Loads a G1 robot + table + deformable towel scene
  - Captures ego-view camera images (224x224) from the robot's head
  - Sends observations to the GROOT inference server via ZMQ (PolicyClient)
  - Applies returned actions to the robot joints
  - Records video for review
- Tested with both `groot-g1-gripper-fold-towel-full` and `groot-g1-gripper-hospitality-7ds` models
- Ran evaluations at 500 steps and 1500 steps

### 2. Action Processing — ABSOLUTE vs RELATIVE (Confirmed)

- **Confirmed:** The GROOT server returns ALL actions as ABSOLUTE joint targets
- The model internally predicts RELATIVE deltas for arms, but `StateActionProcessor.unapply_action()` converts them to ABSOLUTE positions server-side before returning
- Our eval script applies these as direct position targets — no double-delta issue
- This was verified by tracing the full code path: `Gr00tPolicy._get_action()` → `processor.decode_action()` → `_convert_to_absolute_action()`

### 3. Whole-Body-Control (WBC) Integration

- Built `run_mujoco_towel_eval_wbc.py` — extends the eval with NVIDIA's WBC ONNX balance policies
- **WBC architecture:** Two ONNX models (Balance + Walk) control the lower body (12 leg DOFs + 3 waist DOFs = 15 total) at 50 Hz, while GROOT controls upper body (14 arm DOFs + grippers) at 12.5 Hz
- **Key fix:** Discovered the Menagerie G1 uses position-servo actuators (PD controller built into the actuator with gain=500), but WBC outputs raw torques (gain=1). Converted all actuators to direct-torque mode to match WBC expectations. Without this fix, the robot falls over immediately.
- With WBC enabled, the robot stands and balances throughout the episode

### 4. Apple Pick-and-Place (NVIDIA Standard Eval)

- Successfully ran NVIDIA's standard `rollout_policy.py` eval with the `gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc` environment
- Used NVIDIA's pre-trained `GR00T-N1.6-G1-PnPAppleToPlate` checkpoint — the robot actively reaches for and interacts with the apple
- **Our models cannot run this eval** — the apple PnP environment expects 7-DOF dexterous hands, but our models use 1-DOF grippers (shape mismatch error)

### 5. Infrastructure & Documentation

- Created `docs/ENVIRONMENT_SETUP.md` — complete dependency list, SSH credentials, container setup, and eval commands
- Installed `onnxruntime` in the workstation container (was missing)
- Set up GR00T-WholeBodyControl submodule with all dependencies (robosuite, robocasa, pinocchio) in a separate Python 3.10 UV venv on the workstation
- All code committed and pushed to `datamentors/dm-isaac-g1`

---

## What's Not Working

### The robot does not interact with the towel

Both with and without WBC, the robot's arms make small movements but do not reach down to the table to touch or fold the towel. Observations:

- **Arm action ranges are very small:** GROOT returns arm targets in the range of approximately ±0.1 to ±0.4 radians — these are subtle joint adjustments, not the large sweeping motions needed to reach a table surface
- **The ego-view looks correct:** The head-mounted camera sees the table and towel from the correct first-person perspective
- **Joint state is being sent correctly:** All 31 DOF state values are extracted and sent in the expected UNITREE_G1 format
- **Both models show the same issue:** `fold-towel-full` and `hospitality-7ds` both produce insufficient arm range

### Possible root causes

1. **Sim-to-real gap in scene setup:** Our MuJoCo towel scene may not match the training environment geometry (robot-to-table distance, table height, robot posture). The model may have been trained with a different spatial relationship.

2. **Fine-tuning data vs eval mismatch:** Need to verify how the official Isaac-GR00T fine-tuning pipeline sets up evaluation — there may be specific scene configurations, observation preprocessing, or action post-processing steps we're missing.

3. **Missing observation context:** The model may expect additional context frames or a specific observation history that we're not providing (we send single-frame observations).

4. **Wrist joint remapping:** We remap wrist joints between Menagerie order (roll, pitch, yaw) and GROOT training order (yaw, roll, pitch). If this mapping is wrong, the model receives incorrect proprioception and produces incorrect actions.

---

## Eval Results Summary

| Eval | Model | WBC | Balance | Towel Interaction | Video |
|------|-------|-----|---------|-------------------|-------|
| Fixed-base, 1500 steps | fold-towel-full | No | Locked base | None — small arm movements | `dataset_review/mujoco/v6/` |
| Fixed-base, 1500 steps | hospitality-7ds | No | Locked base | None — small arm movements | `dataset_review/mujoco/7ds_towel/` |
| WBC, 1500 steps | hospitality-7ds | Yes (Balance ONNX) | Standing, balanced | None — small arm movements | `dataset_review/mujoco/wbc_towel_v2/` |
| Apple PnP (NVIDIA eval) | NVIDIA PnPAppleToPlate | Yes (WBC wrapper) | Full WBC | **Yes — reaches and grasps apple** | `dataset_review/mujoco/apple_pnp_eval/` |

---

## Next Steps

1. **Review official Isaac-GR00T fine-tuning and evaluation pipeline** — compare their training data format, observation construction, action decoding, and scene setup against what we're doing to identify discrepancies
2. **Validate our fine-tuning config** — check if `g1_fold_towel_config.py` and the training scripts match what the official repo expects
3. **Compare observation format with training data** — verify that the ego-view images and state vectors we construct match what the model was trained on (resolution, crop, normalization, joint ordering)
4. **Test with NVIDIA's base model** — run `GR00T-N1.6-3B` to see if the base model also produces small actions, which would indicate a scene/observation issue rather than a fine-tuning issue

---

## Commits

| Hash | Description |
|------|-------------|
| `a859dc4` | fix: treat ALL server actions as ABSOLUTE, set default host to Spark |
| `cb0ad76` | fix: update server port to 5555 and host to Spark across docs/scripts |
| `dfab107` | feat: add WBC-enabled MuJoCo towel eval script |
| `59c248c` | fix: convert Menagerie actuators to direct-torque mode for WBC |
| `2404f50` | docs: add environment setup guide for MuJoCo eval with WBC |

---

## Key Files

- `scripts/eval/run_mujoco_towel_eval.py` — Fixed-base towel eval (no WBC)
- `scripts/eval/run_mujoco_towel_eval_wbc.py` — WBC-enabled towel eval
- `scripts/eval/mujoco_towel_scene/g1_towel_folding.xml` — MuJoCo towel scene
- `docs/ENVIRONMENT_SETUP.md` — Full environment/dependency documentation
- `docs/MUJOCO_EVAL_GUIDE.md` — Eval pipeline architecture guide
- `docs/SIMULATION_INFERENCE_GUIDE.md` — Inference setup guide
