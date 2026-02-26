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

---

## Audit: Root Causes Identified

A detailed audit of the Isaac-GR00T fine-tuning pipeline vs our eval setup uncovered **three critical discrepancies** that likely explain why the model produces only small arm movements:

### 1. Wrong Robot Model — No Gripper Hands (CRITICAL)

Our MuJoCo scene (`g1_towel_folding.xml`) used Menagerie's `g1.xml` which has **no hands at all** (29 DOF). The GROOT model was trained with the UNITREE_G1 embodiment that includes **1-DOF Dex1 grippers** per hand (31 DOF total). This means:

- **State indices 29-30 (left_hand, right_hand) were sent as 0.0** instead of the actual gripper state
- Training data hand values: left_hand mean ≈ 2.9, right_hand mean ≈ 2.25 (range [0, 4.74])
- Sending 0.0 is far outside the training distribution, causing the model to receive normalized state values clipped at -1.0

**Fix:** Switched to `g1_gripper_towel_folding.xml` which uses `g1_with_hands.xml` (Inspire dexterous hands). The eval script maps between Inspire's 7 finger DOFs and GROOT's 1-DOF gripper value.

### 2. Initial Joint Pose Mismatch (CRITICAL)

MuJoCo initializes all joints to 0.0, but training data shows the robot in a specific bent posture:

| Joint Group | Training Mean | MuJoCo Default | Issue |
|-------------|--------------|----------------|-------|
| left_leg hip_pitch | -0.43 rad | 0.0 | Knees straight instead of bent |
| left_leg knee | 0.64 rad | 0.0 | Missing knee bend |
| waist pitch | 0.24 rad | 0.0 | Not leaning forward |
| left_arm shoulder_pitch | -0.75 rad | 0.0 | Arms hanging instead of raised |
| left_arm elbow | 0.42 rad | 0.0 | Elbows straight |
| left_hand | 2.9 | 0.0 | Grippers reported as fully open |

The StateActionProcessor normalizes state with min/max from training data. With `clip_outliers: true`, joint values at 0.0 (when training range is e.g. [-0.9, -0.1]) get normalized to a value far outside [-1, 1] and clipped. The model effectively receives garbage proprioception.

**Fix:** Initialize all joints to training data mean state from `dataset_statistics.json` in the checkpoint.

### 3. Server-Side Processing Confirmed Correct

The following were verified as correctly handled:
- **Sin/cos state encoding** — applied server-side by `StateActionProcessor` (not client)
- **Language instruction** — format `tuple[str]` → `list[list[str]]` handled by `Gr00tSimPolicyWrapper`
- **Relative→Absolute action conversion** — done server-side by `_convert_to_absolute_action()`
- **navigate_command** — 3 DOF (confirmed, not 2)
- **Wrist joint remapping** — GROOT order (yaw, roll, pitch) ↔ Menagerie order (roll, pitch, yaw) is correct

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

1. **Run eval with fixes applied** — use `g1_gripper_towel_folding.xml` scene with training-mean initial pose and gripper state. This addresses both root causes identified above.
2. **Test with both models** — run with `fold-towel-full` and `hospitality-7ds` to verify improvement
3. **Test with NVIDIA's base model** — run `GR00T-N1.6-3B` to see if the base model produces meaningful actions with the corrected setup
4. **Fine-tune scene geometry if needed** — if actions improve but don't reach the table, adjust robot-to-table distance/height to better match training environment

---

## Commits

| Hash | Description |
|------|-------------|
| `a859dc4` | fix: treat ALL server actions as ABSOLUTE, set default host to Spark |
| `cb0ad76` | fix: update server port to 5555 and host to Spark across docs/scripts |
| `dfab107` | feat: add WBC-enabled MuJoCo towel eval script |
| `59c248c` | fix: convert Menagerie actuators to direct-torque mode for WBC |
| `2404f50` | docs: add environment setup guide for MuJoCo eval with WBC |
| `7dbf309` | docs: add team update for MuJoCo eval pipeline progress |
| `9bd8490` | docs: add gripper inference changes doc and loco client example script |

---

## Key Files

- `scripts/eval/run_mujoco_towel_eval.py` — Fixed-base towel eval (no WBC)
- `scripts/eval/run_mujoco_towel_eval_wbc.py` — WBC-enabled towel eval (with gripper + training pose)
- `scripts/eval/mujoco_towel_scene/g1_towel_folding.xml` — MuJoCo towel scene (base g1, no hands)
- `scripts/eval/mujoco_towel_scene/g1_gripper_towel_folding.xml` — MuJoCo towel scene (with hands for gripper)
- `docs/ENVIRONMENT_SETUP.md` — Full environment/dependency documentation
- `docs/MUJOCO_EVAL_GUIDE.md` — Eval pipeline architecture guide
- `docs/SIMULATION_INFERENCE_GUIDE.md` — Inference setup guide
