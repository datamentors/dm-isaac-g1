# G1 EDU 2 + Inspire Hands Training Plan

> **DEPRECATED**: This plan was for the Inspire Robotics Dexterous Hands (53 DOF) with `NEW_EMBODIMENT` tag.
> We have since switched to the **UNITREE_G1 gripper embodiment** (31 DOF state / 23 DOF action) which uses
> the pre-registered `UNITREE_G1` tag in Isaac-GR00T. See [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) for
> the current workflow. This document is kept for historical reference.

## Overview

This document outlines the comprehensive strategy for training a GROOT model for the **Unitree G1 EDU 2** robot with **Inspire Robotics Dexterous Hands**. It consolidates all available datasets, joint configurations, and training approaches.

---

## Target Robot Configuration

| Component | Details |
|-----------|---------|
| Robot Body | Unitree G1 EDU 2 |
| Hands | Inspire Robotics Dexterous Hands |
| Body DOF | 29 (legs + waist + arms) |
| Hand DOF | 24 (12 per hand) |
| **Total DOF** | **53** |

---

## Complete Joint Specification (from unitree_sim_isaaclab)

### Body Joints (29 DOF)

```
Legs (12 DOF):
├── Left Leg (6):
│   ├── left_hip_yaw_joint
│   ├── left_hip_roll_joint
│   ├── left_hip_pitch_joint
│   ├── left_knee_joint
│   ├── left_ankle_pitch_joint
│   └── left_ankle_roll_joint
└── Right Leg (6):
    ├── right_hip_yaw_joint
    ├── right_hip_roll_joint
    ├── right_hip_pitch_joint
    ├── right_knee_joint
    ├── right_ankle_pitch_joint
    └── right_ankle_roll_joint

Waist (3 DOF):
├── waist_yaw_joint
├── waist_roll_joint
└── waist_pitch_joint

Arms (14 DOF):
├── Left Arm (7):
│   ├── left_shoulder_pitch_joint
│   ├── left_shoulder_roll_joint
│   ├── left_shoulder_yaw_joint
│   ├── left_elbow_joint
│   ├── left_wrist_roll_joint
│   ├── left_wrist_pitch_joint
│   └── left_wrist_yaw_joint
└── Right Arm (7):
    ├── right_shoulder_pitch_joint
    ├── right_shoulder_roll_joint
    ├── right_shoulder_yaw_joint
    ├── right_elbow_joint
    ├── right_wrist_roll_joint
    ├── right_wrist_pitch_joint
    └── right_wrist_yaw_joint
```

### Inspire Hand Joints (24 DOF)

```
Left Hand (12 DOF):
├── L_index_proximal_joint
├── L_index_intermediate_joint
├── L_middle_proximal_joint
├── L_middle_intermediate_joint
├── L_pinky_proximal_joint
├── L_pinky_intermediate_joint
├── L_ring_proximal_joint
├── L_ring_intermediate_joint
├── L_thumb_proximal_yaw_joint
├── L_thumb_proximal_pitch_joint
├── L_thumb_intermediate_joint
└── L_thumb_distal_joint

Right Hand (12 DOF):
├── R_index_proximal_joint
├── R_index_intermediate_joint
├── R_middle_proximal_joint
├── R_middle_intermediate_joint
├── R_pinky_proximal_joint
├── R_pinky_intermediate_joint
├── R_ring_proximal_joint
├── R_ring_intermediate_joint
├── R_thumb_proximal_yaw_joint
├── R_thumb_proximal_pitch_joint
├── R_thumb_intermediate_joint
└── R_thumb_distal_joint
```

### Inspire Hand Control Parameters

| Parameter | Value |
|-----------|-------|
| Effort Limit | 100.0 N·m |
| Velocity Limit | 50 rad/s |
| Stiffness | 1000.0 |
| Damping | 15.0 |
| Default Position | 0.0 rad (all joints) |

---

## Isaac Sim Tasks with G1 + Inspire Hands

These tasks from `unitree_sim_isaaclab` are specifically designed for G1 with Inspire hands:

| Task ID | Description | Control Type |
|---------|-------------|--------------|
| `Isaac-Move-Cylinder-G129-Inspire-Wholebody` | Move cylinder with whole-body control | Wholebody |
| `Isaac-PickPlace-Cylinder-G129-Inspire-Joint` | Pick and place cylinder | Joint control |
| `Isaac-PickPlace-RedBlock-G129-Inspire-Joint` | Pick and place red block | Joint control |
| `Isaac-Stack-RgyBlock-G129-Inspire-Joint` | Stack RGB blocks | Joint control |

### Task Configuration Details

**Simulation Parameters:**
- Timestep: 0.005 seconds
- Decimation: 2-4 (varies by task)
- Episode Length: 20.0 seconds
- Control Scale: 1.0

**Observation Groups:**
1. Robot body joint states
2. Robot Inspire hand joint states
3. Camera image data

---

## ALL Available Datasets Summary

### Category 1: Native G1 Simulated Data (Best for Initial Training)

| Dataset | Robot | Hand Type | DOF | Episodes | Status |
|---------|-------|-----------|-----|----------|--------|
| `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` (unitree_g1.LMPnPAppleToPlateDC) | G1 (sim) | Default | Variable | 103 | ✅ Trained |

### Category 2: Real G1 Teleoperation Data

| Dataset | Robot | Hand Type | Hand DOF | Episodes | Status |
|---------|-------|-----------|----------|----------|--------|
| `nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1` (g1-pick-apple) | G1 (real) | Tri-finger | 7/hand | 311 | ✅ Trained |
| `nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1` (g1-pick-pear) | G1 (real) | Tri-finger | 7/hand | ~311 | ⏳ Available |
| `nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1` (g1-pick-grapes) | G1 (real) | Tri-finger | 7/hand | ~311 | ⏳ Available |
| `nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1` (g1-pick-starfruit) | G1 (real) | Tri-finger | 7/hand | ~311 | ⏳ Available |

### Category 3: Dex3 Dexterous Hand Tasks (Closest to Inspire - 3 Fingers)

| Dataset | Episodes | Frames | Task | Hand DOF |
|---------|----------|--------|------|----------|
| [G1_Dex3_ToastedBread](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset) | 418 | 352k | Making toast | 14 (7/hand) |
| [G1_Dex3_BlockStacking](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_BlockStacking_Dataset) | 301 | 281k | Block stacking | 14 (7/hand) |
| [G1_Dex3_Pouring](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_Pouring_Dataset) | ~300 | ~250k | Pouring liquid | 14 (7/hand) |
| [G1_Dex3_PickApple](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_PickApple_Dataset) | 201 | 152k | Apple picking | 14 (7/hand) |
| [G1_Dex3_ObjectPlacement](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ObjectPlacement_Dataset) | 210 | 98k | Object placement | 14 (7/hand) |
| [G1_Dex3_GraspSquare](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_GraspSquare_Dataset) | ~200 | ~150k | Square grasping | 14 (7/hand) |
| G1_Dex3_PickBottle | Various | Various | Bottle picking | 14 (7/hand) |
| G1_Dex3_PickCharger | Various | Various | Charger picking | 14 (7/hand) |
| G1_Dex3_PickDoll | Various | Various | Doll picking | 14 (7/hand) |
| G1_Dex3_PickGum | Various | Various | Gum picking | 14 (7/hand) |
| G1_Dex3_PickSnack | Various | Various | Snack picking | 14 (7/hand) |
| G1_Dex3_PickTissue | Various | Various | Tissue picking | 14 (7/hand) |

### Category 4: Simple Gripper Tasks (1 DOF per hand) - HOSPITALITY FOCUS

**Priority Hospitality Tasks (Recommended for Training):**

| Dataset | Episodes | Frames | Task | Hospitality Use |
|---------|----------|--------|------|-----------------|
| [G1_Fold_Towel](https://huggingface.co/datasets/unitreerobotics/G1_Fold_Towel) | ~714 | 311k | Towel folding | ⭐ Hotel housekeeping |
| [G1_Clean_Table](https://huggingface.co/datasets/unitreerobotics/G1_Clean_Table) | ~775 | 266k | Table cleaning | ⭐ Restaurant/Hotel |
| [G1_Wipe_Table](https://huggingface.co/datasets/unitreerobotics/G1_Wipe_Table) | ~526 | 76k | Table wiping | ⭐ Restaurant/Hotel |
| [G1_DualRobot_Clean_Table](https://huggingface.co/datasets/unitreerobotics/G1_DualRobot_Clean_Table) | ~274 | 171k | Dual-robot cleaning | ⭐ Large venue cleaning |
| [G1_Prepare_Fruit](https://huggingface.co/datasets/unitreerobotics/G1_Prepare_Fruit) | ~427 | 124k | Fruit preparation | ⭐ Kitchen/Buffet prep |
| [G1_Pour_Medicine](https://huggingface.co/datasets/unitreerobotics/G1_Pour_Medicine) | ~596 | - | Pouring liquid | ⭐ Beverage service |
| [G1_Organize_Tools](https://huggingface.co/datasets/unitreerobotics/G1_Organize_Tools) | ~407 | 183k | Organization | ⭐ Room tidying |

**Additional Useful Tasks:**

| Dataset | Episodes | Frames | Task |
|---------|----------|--------|------|
| [G1_Pack_PingPong](https://huggingface.co/datasets/unitreerobotics/G1_Pack_PingPong) | ~506 | 161k | Object packing |
| [G1_Pack_PencilBox](https://huggingface.co/datasets/unitreerobotics/G1_Pack_PencilBox) | ~358 | 163k | Box packing |
| [G1_Stack_Block](https://huggingface.co/datasets/unitreerobotics/G1_Stack_Block) | ~500 | - | Stacking objects |
| [G1_Bag_Insert](https://huggingface.co/datasets/unitreerobotics/G1_Bag_Insert) | ~683 | - | Bag insertion |
| [G1_Erase_Board](https://huggingface.co/datasets/unitreerobotics/G1_Erase_Board) | ~331 | 128k | Board erasing |

### Category 5: Dex1 & Brainco Hand Tasks

| Dataset | Hand Type | Task |
|---------|-----------|------|
| [G1_Dex1_MountCamera](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_MountCamera_Dataset) | Dex1 | Camera mounting |
| G1_Dex1_PickPlaceCylinder_Sim | Dex1 | Pick-place (sim) |
| G1_Dex1_PickPlaceRedBlock_Sim | Dex1 | Pick-place (sim) |
| G1_Dex1_StackRygBlock_Sim | Dex1 | Block stacking (sim) |
| G1_Brainco_PickToothpaste | Brainco | Toothpaste picking |
| G1_Brainco_PickTissues | Brainco | Tissue picking |
| G1_Brainco_GraspRubiksCube | Brainco | Cube grasping |
| G1_Brainco_GraspOreo | Brainco | Food grasping |
| G1_Brainco_PickDrink | Brainco | Beverage handling |

### Category 6: Z1 Dual-Arm Tasks (Hospitality-Relevant)

These Z1 arm datasets demonstrate dual-arm coordination useful for hospitality:

| Dataset | Task | Hospitality Application |
|---------|------|------------------------|
| [Z1_Dual_Dex1_FoldClothes](https://huggingface.co/datasets/unitreerobotics/Z1_Dual_Dex1_FoldClothes_Dataset) | Clothes folding | ⭐ Laundry/housekeeping |
| [Z1_Dual_Dex1_PourCoffee](https://huggingface.co/datasets/unitreerobotics/Z1_Dual_Dex1_PourCoffee_Dataset) | Coffee pouring | ⭐ Beverage service |
| [Z1_Dual_Dex1_CleanupPencils](https://huggingface.co/datasets/unitreerobotics/Z1_Dual_Dex1_CleanupPencils_Dataset) | Cleanup/organizing | ⭐ Room tidying |
| [Z1_Dual_Dex1_StackBox](https://huggingface.co/datasets/unitreerobotics/Z1_Dual_Dex1_StackBox_Dataset) | Box stacking | Storage/organization |

---

## Hand Type Comparison

| Hand Type | DOF per Hand | Fingers | Best Dataset Match for Inspire |
|-----------|--------------|---------|-------------------------------|
| **Inspire** (Target) | 12 | 5 (full dexterous) | - |
| Dex3 | 7 | 3 (index, middle, thumb) | ⭐⭐⭐ Best (dexterous manipulation) |
| Tri-finger | 7 | 3 | ⭐⭐ Good (real robot data) |
| Simple Gripper | 1 | 2 (parallel) | ⭐ Limited (grasp only) |
| Dex1 | 2 | 2 | ⭐ Limited |

---

## Joint Mapping Strategy

### Dex3 (7 DOF) → Inspire (12 DOF)

The Dex3 hand has 3 fingers (index, middle, thumb) with 7 DOF total. Map to Inspire:

| Dex3 Joint | DOF | Inspire Joint | Notes |
|------------|-----|---------------|-------|
| index_proximal | 1 | L/R_index_proximal_joint | Direct map |
| index_intermediate | 1 | L/R_index_intermediate_joint | Direct map |
| middle_proximal | 1 | L/R_middle_proximal_joint | Direct map |
| middle_intermediate | 1 | L/R_middle_intermediate_joint | Direct map |
| thumb_proximal | 1 | L/R_thumb_proximal_pitch_joint | Direct map |
| thumb_intermediate | 1 | L/R_thumb_intermediate_joint | Direct map |
| thumb_distal | 1 | L/R_thumb_distal_joint | Direct map |
| - | - | L/R_thumb_proximal_yaw_joint | Zero-pad |
| - | - | L/R_ring_proximal_joint | Zero-pad |
| - | - | L/R_ring_intermediate_joint | Zero-pad |
| - | - | L/R_pinky_proximal_joint | Zero-pad |
| - | - | L/R_pinky_intermediate_joint | Zero-pad |

### Tri-finger (7 DOF) → Inspire (12 DOF)

| Tri-finger Joint | Inspire Joint |
|------------------|---------------|
| finger_0 | thumb_proximal_pitch |
| finger_1 | thumb_intermediate |
| finger_2 | thumb_distal |
| finger_3 | index_proximal |
| finger_4 | index_intermediate |
| finger_5 | middle_proximal |
| finger_6 | middle_intermediate |
| (zero) | remaining 5 joints |

### Simple Gripper (1 DOF) → Inspire (12 DOF)

| Gripper | Inspire Mapping |
|---------|-----------------|
| gripper_open/close | All finger proximal joints move together |
| (zero) | All other joints |

---

## Detailed Hand Remapping Examples

### Example 1: Simple Gripper → Inspire (Hospitality Datasets)

**Source Data (G1_Fold_Towel):**
```python
# Original format - 1 DOF per hand
action.left_gripper = [0.8]   # gripper close amount (0-1)
action.right_gripper = [0.3]  # gripper open
```

**Target Format (Inspire 12 DOF per hand):**
```python
# Inspire hand joint order
inspire_joints = [
    "index_proximal", "index_intermediate",
    "middle_proximal", "middle_intermediate",
    "pinky_proximal", "pinky_intermediate",
    "ring_proximal", "ring_intermediate",
    "thumb_proximal_yaw", "thumb_proximal_pitch",
    "thumb_intermediate", "thumb_distal"
]

# Mapping: gripper value controls all proximal joints for grasping
def gripper_to_inspire(gripper_value):
    inspire_hand = [0.0] * 12
    # Proximal joints (indices 0, 2, 4, 6, 9) control finger closure
    inspire_hand[0] = gripper_value  # index_proximal
    inspire_hand[2] = gripper_value  # middle_proximal
    inspire_hand[4] = gripper_value  # pinky_proximal
    inspire_hand[6] = gripper_value  # ring_proximal
    inspire_hand[9] = gripper_value  # thumb_proximal_pitch
    return inspire_hand

# Result:
left_inspire = gripper_to_inspire(0.8)
# [0.8, 0.0, 0.8, 0.0, 0.8, 0.0, 0.8, 0.0, 0.0, 0.8, 0.0, 0.0]
```

### Example 2: Dex3 (7 DOF) → Inspire (12 DOF)

**Source Data (G1_Dex3_ToastedBread):**
```python
# Dex3 format - 7 DOF per hand (index, middle, thumb only)
action.left_hand = [
    0.5,  # index_proximal
    0.3,  # index_intermediate
    0.6,  # middle_proximal
    0.4,  # middle_intermediate
    0.7,  # thumb_proximal
    0.2,  # thumb_intermediate
    0.1   # thumb_distal
]
```

**Mapping to Inspire:**
```python
def dex3_to_inspire(dex3_hand):
    inspire_hand = [0.0] * 12
    # Direct mapping for index, middle, thumb
    inspire_hand[0] = dex3_hand[0]   # index_proximal
    inspire_hand[1] = dex3_hand[1]   # index_intermediate
    inspire_hand[2] = dex3_hand[2]   # middle_proximal
    inspire_hand[3] = dex3_hand[3]   # middle_intermediate
    # Pinky and ring are zero-padded (indices 4-7)
    inspire_hand[9] = dex3_hand[4]   # thumb_proximal_pitch
    inspire_hand[10] = dex3_hand[5]  # thumb_intermediate
    inspire_hand[11] = dex3_hand[6]  # thumb_distal
    # thumb_proximal_yaw (index 8) stays zero
    return inspire_hand

# Result:
left_inspire = dex3_to_inspire([0.5, 0.3, 0.6, 0.4, 0.7, 0.2, 0.1])
# [0.5, 0.3, 0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1]
```

### Example 3: Full State Vector Assembly (53 DOF)

**Final concatenated observation.state for GROOT:**
```python
# Body joints (29 DOF)
body = [
    # Legs (12): hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll × 2
    *left_leg,   # 6 DOF
    *right_leg,  # 6 DOF
    # Waist (3)
    *waist,      # 3 DOF
    # Arms (14): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
    #            wrist_roll, wrist_pitch, wrist_yaw × 2
    *left_arm,   # 7 DOF
    *right_arm,  # 7 DOF
]  # Total: 29 DOF

# Inspire hands (24 DOF) - after remapping
left_inspire = gripper_to_inspire(left_gripper)   # 12 DOF
right_inspire = gripper_to_inspire(right_gripper) # 12 DOF

# Final state vector
observation_state = body + left_inspire + right_inspire  # 53 DOF
action = body_action + left_inspire_action + right_inspire_action  # 53 DOF
```

### Modality Config for 53 DOF Inspire

```python
# /workspace/Isaac-GR00T/g1_inspire_unified_config.py

g1_inspire_unified_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_left_high"],  # or appropriate camera key
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_leg",         # 6 DOF (indices 0-5)
            "right_leg",        # 6 DOF (indices 6-11)
            "waist",            # 3 DOF (indices 12-14)
            "left_arm",         # 7 DOF (indices 15-21)
            "right_arm",        # 7 DOF (indices 22-28)
            "left_inspire",     # 12 DOF (indices 29-40)
            "right_inspire",    # 12 DOF (indices 41-52)
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "left_leg", "right_leg", "waist",
            "left_arm", "right_arm",
            "left_inspire", "right_inspire",
        ],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.ABSOLUTE,
                        type=ActionType.NON_EEF,
                        format=ActionFormat.DEFAULT)
        ] * 7,
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}
```

---

## Recommended Training Approach

### Multi-Dataset Strategy (Recommended)

GROOT N1.6 supports **cross-embodiment training** - we can combine datasets with different body/hand configurations by using joint mapping. This maximizes available data while targeting the Inspire hand configuration.

### Data Sources (Priority Order)

| Priority | Source | Native DOF | Mapping Required | Data Type | Status |
|----------|--------|------------|------------------|-----------|--------|
| 1 | **Generate Inspire data** (unitree_sim_isaaclab) | 53 (exact) | None | Sim | ⏳ To generate |
| 2 | **G1 Loco-Manipulation** (already trained) | Variable | Body only, zero-pad hands | Sim | ✅ Downloaded |
| 3 | **G1 Teleop Tri-finger** (already trained) | 43 | 7→12 per hand | Real | ✅ Downloaded |
| 4 | **Dex3 datasets** (HuggingFace) | 28 | 7→12 per hand | Real | ⏳ To download |
| 5 | **Simple gripper tasks** | 31 | 1→12 per hand | Real | ⏳ Optional |

### Already Available Data (From Previous Training)

These datasets are already downloaded and trained on - include them in the combined dataset:

| Dataset | Location | Episodes | DOF | Hand Mapping |
|---------|----------|----------|-----|--------------|
| `unitree_g1.LMPnPAppleToPlateDC` | `/workspace/datasets/gr00t_x_embodiment/` | 103 | Variable (loco+manip) | Zero-pad to 24 hand DOF |
| `g1-pick-apple` (teleop) | `/workspace/datasets/g1_teleop/` | 311 | 43 (upper body + tri-finger) | Map 7→12 per hand |

---

### Phase 1: Generate Native Inspire Data (Best Quality)

Use `unitree_sim_isaaclab` to generate demonstration data with **exact G1 + Inspire joint configuration**:

```bash
# Clone the repo
git clone https://github.com/unitreerobotics/unitree_sim_isaaclab
cd unitree_sim_isaaclab

# Run Inspire tasks with data generation
python sim_main.py --device cuda --enable_cameras \
  --task Isaac-PickPlace-RedBlock-G129-Inspire-Joint \
  --robot_type g129 \
  --generate_data --generate_data_dir ./inspire_data/pick_place_redblock

python sim_main.py --device cuda --enable_cameras \
  --task Isaac-Stack-RgyBlock-G129-Inspire-Joint \
  --robot_type g129 \
  --generate_data --generate_data_dir ./inspire_data/stack_blocks

python sim_main.py --device cuda --enable_cameras \
  --task Isaac-Move-Cylinder-G129-Inspire-Wholebody \
  --robot_type g129 \
  --generate_data --generate_data_dir ./inspire_data/move_cylinder
```

**Available Inspire Tasks:**
- `Isaac-PickPlace-Cylinder-G129-Inspire-Joint` - Cylinder manipulation
- `Isaac-PickPlace-RedBlock-G129-Inspire-Joint` - Block manipulation
- `Isaac-Stack-RgyBlock-G129-Inspire-Joint` - Block stacking (RGB)
- `Isaac-Move-Cylinder-G129-Inspire-Wholebody` - Whole-body locomotion + manipulation

### Phase 2: Download Dex3 Data (Dexterous Manipulation)

Dex3 has 3-finger dexterous hands closest to Inspire. Download and apply joint mapping:

```bash
huggingface-cli download unitreerobotics/G1_Dex3_ToastedBread_Dataset \
  --repo-type dataset \
  --local-dir ~/datasets/g1_dex3_toasted_bread

huggingface-cli download unitreerobotics/G1_Dex3_BlockStacking_Dataset \
  --repo-type dataset \
  --local-dir ~/datasets/g1_dex3_block_stacking

huggingface-cli download unitreerobotics/G1_Dex3_Pouring_Dataset \
  --repo-type dataset \
  --local-dir ~/datasets/g1_dex3_pouring
```

### Phase 3: Add Teleop Data (Real-World Sensor Data)

Real robot data provides better sensor characteristics:

```bash
# Use git LFS to avoid rate limits
git lfs install
git clone https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1 ~/datasets/g1_teleop
```

### Phase 4: Combine ALL Datasets (Including Previous)

Merge ALL available datasets with joint mapping into unified training set:

```bash
# Combine all data sources:
python scripts/combine_datasets.py \
  --inspire-data ./inspire_data \
  --loco-manip-data /workspace/datasets/gr00t_x_embodiment/unitree_g1.LMPnPAppleToPlateDC \
  --teleop-data /workspace/datasets/g1_teleop \
  --dex3-data ~/datasets/g1_dex3_* \
  --output ~/datasets/g1_inspire_combined \
  --target-embodiment G1_INSPIRE_53DOF
```

### Combined Dataset Summary

| Source | Episodes | Task Types | Hand Mapping |
|--------|----------|------------|--------------|
| Native Inspire (Isaac Sim) | ~500+ | Pick/place, stack, move | None (exact) |
| Loco-Manipulation (sim) | 103 | Pick apple + locomotion | Zero-pad hands |
| G1 Teleop (real) | 311+ | Fruit picking | Tri-finger → Inspire |
| Dex3 (real) | 1000+ | Toast, stack, pour, grasp | Dex3 → Inspire |
| **Hospitality (simple gripper)** | **3000+** | **Clean, fold, wipe, organize, serve** | **Gripper → Inspire** |
| **Total** | **~5000+** | **Diverse manipulation + hospitality** | **Unified 53 DOF** |

### Hospitality-Focused Dataset Download

```bash
# Priority hospitality datasets (simple gripper - map 1→12 DOF per hand)
huggingface-cli download unitreerobotics/G1_Fold_Towel --repo-type dataset --local-dir ~/datasets/g1_fold_towel
huggingface-cli download unitreerobotics/G1_Clean_Table --repo-type dataset --local-dir ~/datasets/g1_clean_table
huggingface-cli download unitreerobotics/G1_Wipe_Table --repo-type dataset --local-dir ~/datasets/g1_wipe_table
huggingface-cli download unitreerobotics/G1_Prepare_Fruit --repo-type dataset --local-dir ~/datasets/g1_prepare_fruit
huggingface-cli download unitreerobotics/G1_Pour_Medicine --repo-type dataset --local-dir ~/datasets/g1_pour_medicine
huggingface-cli download unitreerobotics/G1_Organize_Tools --repo-type dataset --local-dir ~/datasets/g1_organize_tools
huggingface-cli download unitreerobotics/G1_DualRobot_Clean_Table --repo-type dataset --local-dir ~/datasets/g1_dual_clean
```

---

## G1 Inspire Embodiment Configuration

### Full 53-DOF Config for GROOT Training

```python
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat
from gr00t.data.embodiment_tags import EmbodimentTag

# Complete joint list (53 DOF)
G1_INSPIRE_JOINTS = {
    # Legs (12)
    "left_leg": [
        "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"
    ],
    "right_leg": [
        "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
    ],
    # Waist (3)
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    # Arms (14)
    "left_arm": [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"
    ],
    "right_arm": [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
    ],
    # Inspire Hands (24)
    "left_inspire_hand": [
        "L_index_proximal_joint", "L_index_intermediate_joint",
        "L_middle_proximal_joint", "L_middle_intermediate_joint",
        "L_pinky_proximal_joint", "L_pinky_intermediate_joint",
        "L_ring_proximal_joint", "L_ring_intermediate_joint",
        "L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint",
        "L_thumb_intermediate_joint", "L_thumb_distal_joint"
    ],
    "right_inspire_hand": [
        "R_index_proximal_joint", "R_index_intermediate_joint",
        "R_middle_proximal_joint", "R_middle_intermediate_joint",
        "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
        "R_ring_proximal_joint", "R_ring_intermediate_joint",
        "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
        "R_thumb_intermediate_joint", "R_thumb_distal_joint"
    ],
}

g1_inspire_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],  # or "rs_view" for RealSense
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_leg", "right_leg",      # 12 DOF
            "waist",                       # 3 DOF
            "left_arm", "right_arm",       # 14 DOF
            "left_inspire_hand", "right_inspire_hand",  # 24 DOF
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "left_leg", "right_leg",
            "waist",
            "left_arm", "right_arm",
            "left_inspire_hand", "right_inspire_hand",
        ],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ] * 7,
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1_inspire_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

---

## Training Commands

### Training with Dex3 Data + Joint Mapping

```bash
# Activate environment
source /opt/conda/etc/profile.d/conda.sh
conda activate grootenv
cd /workspace/Isaac-GR00T

# Train with custom embodiment config
python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path ./datasets/g1_dex3_toasted_bread \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path ./g1_inspire_config.py \
    --output-dir /workspace/checkpoints/groot_g1_inspire \
    --max-steps 10000 \
    --save-steps 2000 \
    --global-batch-size 8 \
    --learning-rate 1e-4 \
    --dataloader-num-workers 4
```

### Multi-Dataset Training

```bash
# Combine multiple datasets
python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path ./datasets/g1_dex3_combined \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path ./g1_inspire_config.py \
    --output-dir /workspace/checkpoints/groot_g1_inspire_multi \
    --max-steps 20000 \
    --save-steps 5000 \
    --global-batch-size 8 \
    --learning-rate 5e-5
```

---

## Completed Training Sessions

| Session | Dataset | Steps | Duration | Checkpoint |
|---------|---------|-------|----------|------------|
| 1 | unitree_g1.LMPnPAppleToPlateDC (sim) | 5000 | 22 min | `/workspace/checkpoints/groot_g1_full/checkpoint-5000/` |
| 2 | g1-pick-apple (teleop, real) | 5000 | ~17 min | `/workspace/checkpoints/groot_g1_teleop/checkpoint-5000/` |

---

## Dataset Download Status (Updated 2026-02-15)

### Downloaded & Ready for Training

| Dataset | Location | Size | Episodes | Category | Status |
|---------|----------|------|----------|----------|--------|
| gr00t_x_embodiment (LMPnPAppleToPlateDC) | `/workspace/Isaac-GR00T/datasets/` | 360MB | 103 | Simulated | ✅ Ready |
| PhysicalAI-Robotics-GR00T-Teleop-G1 | `/workspace/datasets/` | 1.1GB | ~1000 | Real Teleop | ✅ Ready |
| G1_Dex3_ToastedBread_Dataset | `/workspace/datasets/` | 28GB | 418 | Dex3 | ✅ Ready |
| G1_Dex3_BlockStacking_Dataset | `/workspace/datasets/` | 18GB | 301 | Dex3 | ✅ Ready |
| G1_Fold_Towel | `/workspace/datasets/` | 21GB | ~714 | Hospitality | ✅ Ready |
| G1_Clean_Table | `/workspace/datasets/` | 19GB | ~775 | Hospitality | ✅ Ready |
| G1_Wipe_Table | `/workspace/datasets/` | 4.9GB | ~526 | Hospitality | ✅ Ready |
| G1_Prepare_Fruit | `/workspace/datasets/` | 8.1GB | ~427 | Hospitality | ✅ Ready |
| G1_Pour_Medicine | `/workspace/datasets/` | 12GB | ~596 | Hospitality | ✅ Ready |
| G1_Organize_Tools | `/workspace/datasets/` | 21GB | ~407 | Hospitality | ✅ Ready |
| **Total** | | **~130GB** | **~5000+** | | |

### Not Downloaded (Lower Priority)

| Dataset | Reason | Priority |
|---------|--------|----------|
| G1_DualRobot_Clean_Table | Dual-robot coordination (different control scheme) | Low |
| G1_Dex3_Pouring | Similar to Pour_Medicine | Low |
| G1_Dex3_PickApple | Already have teleop pick-apple | Low |
| Other Dex3 variants | Limited added value | Low |

---

## Understanding the Simulated Data (gr00t_x_embodiment)

The `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` dataset is **NOT** using generative AI (like Cosmos) for scene generation.

### How Isaac Sim Generates Data:

1. **Deterministic Physics Simulation**: NVIDIA Isaac Sim creates a physics-accurate environment with:
   - G1 robot model with exact joint kinematics
   - Objects (apple, plate) with collision and physics properties
   - Table and environment geometry

2. **Domain Randomization (DR)**: Each episode varies:
   - Object initial positions (random placement on table)
   - Lighting conditions (intensity, direction, color)
   - Camera noise and slight angle variations
   - Physics parameters (friction coefficients)
   - Material textures (subtle variations)

3. **Expert Policy Generation**: Actions come from:
   - Pre-trained RL policies that solve the task
   - OR scripted motion planners with IK solvers
   - NOT human teleoperation

4. **103 Episodes = 103 Rollouts**: Each is a unique execution with:
   - Different initial object positions
   - Different random seeds for DR
   - Same fundamental task structure
   - Clean, noise-free joint labels

### Key Insight:
This data provides **consistent joint/action space learning** but lacks the visual complexity and sensor noise of real-world data. That's why combining with real teleop and hospitality datasets is crucial.

---

## Previous Training Sessions (Uploaded to HuggingFace)

These models were trained on **single datasets** and uploaded to HuggingFace. Local checkpoints were deleted to save space.

| Model | HuggingFace Repo | Dataset | Steps | Status |
|-------|-----------------|---------|-------|--------|
| G1 Loco-Manipulation | [datamentorshf/groot-g1-loco-manip](https://huggingface.co/datamentorshf/groot-g1-loco-manip) | unitree_g1.LMPnPAppleToPlateDC (sim) | 5000 | ✅ Uploaded |
| G1 Teleop | [datamentorshf/groot-g1-teleop](https://huggingface.co/datamentorshf/groot-g1-teleop) | g1-pick-apple (real) | 4000 | ✅ Uploaded |

**Note**: These were single-task models. The goal now is multi-dataset training for generalization.

---

## Dataset Format Requirements

### GROOT N1.6 Expected Format (LeRobot v2)

GROOT expects datasets with these specific field names:
- `observation.state` - Concatenated state vector (all joints)
- `action` - Concatenated action vector (NOT `action.action`)
- `observation.images.*` - Video frames (MP4 format)
- `task` - Language task description

### Unitree Hospitality Format (Needs Conversion)

Hospitality datasets use individual fields:
- `observation.left_arm`, `observation.right_arm`, `observation.body`, etc.
- `action.left_arm`, `action.right_arm`, `action.body`, etc.

**Conversion Required**: Use `/workspace/Isaac-GR00T/scripts/convert_g1_format.py`

### Hand Joint Remapping for Inspire (Critical!)

All datasets must be remapped to Inspire hand configuration (12 DOF per hand):

| Source Hand | DOF | Mapping to Inspire (12 DOF) |
|-------------|-----|----------------------------|
| Simple Gripper | 1 | `gripper_value` → all proximal joints (index, middle, ring, pinky, thumb_pitch) |
| Tri-finger | 7 | Map to index/middle/thumb joints, zero-pad ring/pinky |
| Dex3 | 7 | Map to index/middle/thumb joints, zero-pad ring/pinky |
| Inspire (native) | 12 | No mapping needed |

**Inspire Hand Joint Order (per hand):**
```
[0] index_proximal
[1] index_intermediate
[2] middle_proximal
[3] middle_intermediate
[4] pinky_proximal      ← Zero-padded for most datasets
[5] pinky_intermediate  ← Zero-padded for most datasets
[6] ring_proximal       ← Zero-padded for most datasets
[7] ring_intermediate   ← Zero-padded for most datasets
[8] thumb_proximal_yaw  ← Zero-padded for Dex3/Tri-finger
[9] thumb_proximal_pitch
[10] thumb_intermediate
[11] thumb_distal
```

**Total Target DOF: 53** (29 body + 12 left hand + 12 right hand)

### Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `KeyError: 'observation.state'` | Dataset uses individual fields | Convert dataset to concatenated format |
| `KeyError: 'action'` | Dataset uses `action.action` | Rename field to `action` |
| `Language modality must have exactly one key` | Missing task field | Add `modality_keys=["task"]` to config |
| `FileNotFoundError: modality.json` | Missing meta file | Create `meta/modality.json` |

---

## Next Steps

### Immediate Actions
1. [x] **Download all priority datasets** ✅
2. [ ] **Convert hospitality datasets to GROOT format**
3. [ ] **Create unified multi-dataset config**
4. [ ] **Run test training with combined datasets (100 steps)**

### Training Pipeline
5. [ ] **Launch full multi-dataset training** (~10,000 steps)
6. [ ] **Evaluate on multiple tasks**
7. [ ] **Upload best checkpoint to HuggingFace**

### Future Work
8. [ ] **Generate native Inspire hand data** using `unitree_sim_isaaclab`
9. [ ] **Deploy model** to Spark servers (192.168.1.237)
10. [ ] **Collect real teleoperation data** with Inspire hands

---

## References

- [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) - G1 Inspire task definitions
- [NVIDIA GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) - Base model
- [Unitree G1 Datasets](https://huggingface.co/unitreerobotics) - HuggingFace datasets
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) - Fine-tuning framework
