"""
Inference Setup Configurations for GR00T G1 Inference

This module defines named experimental setups (camera + scene layout) that can be
selected when running policy_inference_groot_g1.py via --setup <name>.

Each setup is an InferenceSetup dataclass with:
  - camera_parent: robot link the camera is attached to (None = use scene default d435_link)
  - camera_pos: (x, y, z) offset from camera_parent link origin
  - camera_rot: (w, x, y, z) ROS-convention quaternion
  - object_pos: (x, y, z) initial position of the apple/object in world frame
  - plate_pos:  (x, y, z) initial position of the plate in world frame
  - description: human-readable explanation of the setup

Adding a new setup:
  1. Add an InferenceSetup instance to SETUPS dict below.
  2. Run inference with --setup <your_setup_name>.
  3. No other changes needed.

=============================================================================
SCENE COORDINATE SYSTEM — CRITICAL REFERENCE
=============================================================================

Robot spawn:   (-0.15, 0.0, 0.76) in world frame
Robot rotation: (0.7071, 0, 0, 0.7071) = 90° around X → robot faces +Y direction

Table (packing_table): world pos = (0.0, 0.55, -0.2)
  Table model height ≈ 1.07 m → table surface at world Z ≈ -0.2 + 1.07 = 0.87 m
  Table center is at Y=0.55 — directly in FRONT of robot (robot faces +Y)

Object reach zone (on table surface, reachable by arm):
  X: -0.35 to +0.05  (left-right relative to robot facing direction)
  Y:  0.35 to 0.55   (in front of robot — toward table center)
  Z:  0.84 to 0.87   (table surface + object height)

Default scene cylinder position (from base_scene_pickplace_cylindercfg.py):
  pos = [-0.35, 0.40, 0.84]  ← this is the canonical "on table" position

Camera (d435_link/front_cam):
  Parent: d435_link on robot head
  ROS quaternion: (0.5, -0.5, 0.5, -0.5) = forward-facing in robot frame
  Because robot is rotated 90° around X, "forward" in robot frame = +Y in world frame
  The camera is already forward-facing relative to the robot — it looks toward the table.

=============================================================================
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceSetup:
    """Complete camera + scene layout for one inference experiment."""

    description: str

    # Camera configuration
    # camera_parent=None → use d435_link (the scene's built-in head camera)
    camera_parent: Optional[str]    # Robot link name (relative to Robot prim), or None
    camera_pos: tuple               # (x, y, z) offset from camera_parent link origin
    camera_rot: tuple               # (w, x, y, z) quaternion, ROS convention

    # Camera optics (defaults match Unitree D435 specs)
    focal_length: float = 7.6
    horizontal_aperture: float = 20.0  # ~75° horizontal FOV

    # Scene layout — initial world positions (x, y, z)
    # Default: on the table surface, reachable by the left arm
    object_pos: tuple = (-0.35, 0.40, 0.87)   # Apple: on table, left of center
    plate_pos: tuple = (-0.05, 0.45, 0.865)   # Plate: on table, closer to center-right


# =============================================================================
# Named setups
# =============================================================================

SETUPS: dict[str, InferenceSetup] = {

    # -------------------------------------------------------------------------
    "default": InferenceSetup(
        description=(
            "Original d435_link camera (Unitree default). "
            "Robot faces +Y, table is at Y=0.55. "
            "Objects placed on table surface matching training data layout: "
            "apple at (-0.35, 0.40, 0.87), plate at (-0.05, 0.45, 0.865)."
        ),
        camera_parent=None,           # Use d435_link as configured in the scene
        camera_pos=(0.0, 0.0, 0.0),
        camera_rot=(0.5, -0.5, 0.5, -0.5),
        object_pos=(-0.35, 0.40, 0.87),
        plate_pos=(-0.05, 0.45, 0.865),
    ),

    # -------------------------------------------------------------------------
    "option_a": InferenceSetup(
        description=(
            "Option A: d435_link camera with corrected forward-facing orientation. "
            "Objects placed ON the table in correct world positions. "
            "Robot faces +Y; table at Y=0.55, surface Z=0.87. "
            "Apple at (-0.35, 0.40, 0.87) — matches training data cylinder position. "
            "Plate at (-0.05, 0.45, 0.865) — slightly right of apple on table."
        ),
        camera_parent=None,           # Use d435_link — it IS the correct forward camera
        camera_pos=(0.0, 0.0, 0.0),
        camera_rot=(0.5, -0.5, 0.5, -0.5),
        # Correct positions — ON the table, in front of robot (+Y direction)
        object_pos=(-0.35, 0.40, 0.87),
        plate_pos=(-0.05, 0.45, 0.865),
    ),

    # -------------------------------------------------------------------------
    "option_a_apple_center": InferenceSetup(
        description=(
            "Apple centered on table (Y=0.45), plate further right (Y=0.55). "
            "Tests pick from the center of the robot's reach. "
            "Same d435_link camera as default."
        ),
        camera_parent=None,
        camera_pos=(0.0, 0.0, 0.0),
        camera_rot=(0.5, -0.5, 0.5, -0.5),
        object_pos=(-0.15, 0.45, 0.87),
        plate_pos=(0.10, 0.52, 0.865),
    ),

    # -------------------------------------------------------------------------
    "option_a_apple_left": InferenceSetup(
        description=(
            "Apple far left on table (X=-0.45, Y=0.35), plate center (X=-0.10, Y=0.50). "
            "Tests reaching to the far-left side of the table — "
            "matches training episodes where apple is on robot's left."
        ),
        camera_parent=None,
        camera_pos=(0.0, 0.0, 0.0),
        camera_rot=(0.5, -0.5, 0.5, -0.5),
        object_pos=(-0.45, 0.35, 0.87),
        plate_pos=(-0.10, 0.50, 0.865),
    ),

    # -------------------------------------------------------------------------
    "option_a_closer": InferenceSetup(
        description=(
            "Apple closer to robot (Y=0.30), plate at table center (Y=0.45). "
            "Tests reaching to the near edge of the table — "
            "useful if model trained with close-range pickup."
        ),
        camera_parent=None,
        camera_pos=(0.0, 0.0, 0.0),
        camera_rot=(0.5, -0.5, 0.5, -0.5),
        object_pos=(-0.30, 0.30, 0.87),
        plate_pos=(-0.05, 0.45, 0.865),
    ),
}


def get_setup(name: str) -> InferenceSetup:
    """Return an InferenceSetup by name, raising ValueError if not found."""
    if name not in SETUPS:
        available = ", ".join(sorted(SETUPS.keys()))
        raise ValueError(f"Unknown setup '{name}'. Available: {available}")
    return SETUPS[name]


def list_setups() -> str:
    """Return a formatted string listing all available setups."""
    lines = ["Available inference setups:", ""]
    for name, setup in SETUPS.items():
        lines.append(f"  {name}")
        lines.append(f"    {setup.description}")
        lines.append(f"    camera_parent={setup.camera_parent}, pos={setup.camera_pos}, rot={setup.camera_rot}")
        lines.append(f"    object_pos={setup.object_pos}, plate_pos={setup.plate_pos}")
        lines.append("")
    return "\n".join(lines)
