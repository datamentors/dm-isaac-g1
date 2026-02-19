"""
Inference Setup Configurations for GR00T G1 Inference

This module defines named experimental setups (camera + scene layout) that can be
selected when running policy_inference_groot_g1.py via --setup <name>.

Each setup is an InferenceSetup dataclass with:
  - camera_parent: robot link the camera is attached to (None = use scene default)
  - camera_pos: (x, y, z) offset from camera_parent
  - camera_rot: (w, x, y, z) ROS-convention quaternion
  - object_pos: (x, y, z) initial position of the apple/object in world frame
  - plate_pos:  (x, y, z) initial position of the plate in world frame
  - description: human-readable explanation of the setup

Available setups:
  default       Original setup — uses the scene's built-in d435_link camera (top-down view)
  option_a      Forward-facing chest-height camera matching training data cam_left_high
  option_a_v2   Same as option_a but with steeper downward pitch (30°) for wider table view
  option_a_v3   Torso camera, 20° pitch, objects further forward for longer reach test

Adding a new setup:
  1. Add an InferenceSetup instance to SETUPS dict below.
  2. Run inference with --setup <your_setup_name>.
  3. No other changes needed.

Camera quaternion reference (ROS convention, w x y z):
  - Pure forward (no tilt):     (0.5, -0.5, 0.5, -0.5)
  - ~15° downward pitch:        (0.56, -0.50, 0.43, -0.50)
  - ~30° downward pitch:        (0.61, -0.50, 0.35, -0.50)
  - ~45° downward pitch:        (0.65, -0.50, 0.27, -0.50)

Object/plate position reference (Isaac Sim: +X=forward, +Y=left, +Z=up):
  - Table surface height: ~0.87 m
  - Safe arm reach from robot: ~0.3-0.5 m forward (+X)
  - Apple left of center:  Y ≈ +0.10 to +0.20
  - Plate right of apple:  Y ≈ -0.05 to +0.05
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceSetup:
    """Complete camera + scene layout for one inference experiment."""

    description: str

    # Camera configuration
    # camera_parent=None means use the scene's built-in camera (d435_link)
    camera_parent: Optional[str]    # Robot link name (relative to Robot prim), or None
    camera_pos: tuple               # (x, y, z) offset from camera_parent link origin
    camera_rot: tuple               # (w, x, y, z) quaternion, ROS convention

    # Camera optics (shared defaults match Unitree D435 specs)
    focal_length: float = 7.6
    horizontal_aperture: float = 20.0  # ~75° horizontal FOV

    # Scene layout — initial world positions (x, y, z)
    object_pos: tuple = (0.35, 0.15, 0.87)   # Apple: front-left, on table
    plate_pos: tuple = (0.40, -0.05, 0.865)  # Plate: front-center, on table


# =============================================================================
# Named setups
# =============================================================================

SETUPS: dict[str, InferenceSetup] = {

    # -------------------------------------------------------------------------
    "default": InferenceSetup(
        description=(
            "Original setup: uses the scene's built-in d435_link camera. "
            "In the unitree_sim_isaaclab USD this produces a nearly top-down "
            "birds-eye view (not matching training data)."
        ),
        camera_parent=None,         # Use d435_link as configured in the scene
        camera_pos=(0.0, 0.0, 0.0),
        camera_rot=(0.5, -0.5, 0.5, -0.5),
        # Objects behind robot (original placement from previous experiments)
        object_pos=(-0.30, 0.45, 0.87),
        plate_pos=(-0.10, 0.45, 0.865),
    ),

    # -------------------------------------------------------------------------
    "option_a": InferenceSetup(
        description=(
            "Option A: torso_link chest camera matching training cam_left_high. "
            "Camera attached 15 cm forward, 42 cm above torso_link → ~1.14 m world height. "
            "Forward-facing with ~15° downward pitch to see table + hands. "
            "Apple front-left (0.35, 0.15, 0.87), plate front-center (0.40, -0.05, 0.865)."
        ),
        camera_parent="torso_link",
        camera_pos=(0.15, 0.0, 0.42),
        # Base forward (0.5,-0.5,0.5,-0.5) + ~15° pitch-down adjustment
        camera_rot=(0.56, -0.50, 0.43, -0.50),
        object_pos=(0.35, 0.15, 0.87),
        plate_pos=(0.40, -0.05, 0.865),
    ),

    # -------------------------------------------------------------------------
    "option_a_v2": InferenceSetup(
        description=(
            "Option A v2: same torso_link mount, steeper 30° downward pitch. "
            "Shows more of the table surface and less sky/background. "
            "Useful if option_a shows too much background above the table."
        ),
        camera_parent="torso_link",
        camera_pos=(0.15, 0.0, 0.42),
        # ~30° downward pitch: more table visible in frame
        camera_rot=(0.61, -0.50, 0.35, -0.50),
        object_pos=(0.35, 0.15, 0.87),
        plate_pos=(0.40, -0.05, 0.865),
    ),

    # -------------------------------------------------------------------------
    "option_a_v3": InferenceSetup(
        description=(
            "Option A v3: slightly higher mount (0.45 m above torso_link), "
            "20° pitch, objects placed further forward (0.45 m) to test longer reach. "
            "Tests whether model can generalize to slightly different object positions."
        ),
        camera_parent="torso_link",
        camera_pos=(0.15, 0.0, 0.45),
        # ~20° downward pitch
        camera_rot=(0.59, -0.50, 0.39, -0.50),
        object_pos=(0.40, 0.12, 0.87),
        plate_pos=(0.45, -0.05, 0.865),
    ),

    # -------------------------------------------------------------------------
    "option_a_left": InferenceSetup(
        description=(
            "Option A left: same as option_a but apple placed further left (0.20 m). "
            "Tests model's ability to reach for apple on the left side of the workspace, "
            "matching the left-biased placement in some training episodes."
        ),
        camera_parent="torso_link",
        camera_pos=(0.15, 0.0, 0.42),
        camera_rot=(0.56, -0.50, 0.43, -0.50),
        object_pos=(0.32, 0.20, 0.87),
        plate_pos=(0.38, 0.02, 0.865),
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
