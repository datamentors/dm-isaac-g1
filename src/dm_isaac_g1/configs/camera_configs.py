"""
Camera Configuration Module for Unitree G1 Robot in Isaac Sim/Lab.

This module provides standardized camera configurations for the Unitree G1 robot
with various hand types (DEX3, Inspire, Gripper). Camera configurations are based
on the official Unitree Isaac Sim/Lab integration:
https://github.com/unitreerobotics/unitree_sim_isaaclab

Key Concepts:
- HEAD CAMERA: Mounted on the robot's head (d435_link). This camera is AGNOSTIC
  to hand type - the same configuration works regardless of which hands are used.

- WRIST CAMERAS: Mounted on each hand. These configurations ARE DEPENDENT on
  hand type, as different hands have different link structures and camera mounts.

- WORLD CAMERA: External camera at a fixed position in the world frame, useful
  for third-person observation views.

Usage:
    from dm_isaac_g1.configs.camera_configs import (
        get_head_camera_config,
        get_wrist_camera_configs,
        get_all_camera_configs,
        HandType,
    )

    # Get head camera (hand-agnostic)
    head_cam = get_head_camera_config()

    # Get wrist cameras for DEX3 hands
    wrist_cams = get_wrist_camera_configs(HandType.DEX3)

    # Get all cameras for a robot with Inspire hands
    all_cams = get_all_camera_configs(HandType.INSPIRE, include_world=True)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


class RobotType(Enum):
    """Supported robot base types."""
    G1 = auto()  # Unitree G1 Humanoid


class HandType(Enum):
    """Supported hand/gripper types for G1 robot."""
    NONE = auto()        # No hands (arms only)
    DEX3 = auto()        # Unitree DEX3-1 Dexterous Hand
    INSPIRE = auto()     # Inspire Robotics Dexterous Hand
    GRIPPER = auto()     # Simple parallel gripper


@dataclass
class CameraConfig:
    """Configuration for a single camera in Isaac Sim/Lab.

    Attributes:
        name: Unique identifier for the camera (e.g., "front_cam", "left_wrist_cam")
        prim_path: USD prim path for the camera. Use {ENV_REGEX_NS} for env regex.
        parent_link: Parent link name the camera is attached to.
        position: (x, y, z) position offset from parent link in meters.
        rotation: Quaternion (w, x, y, z) rotation from parent link.
        width: Image width in pixels.
        height: Image height in pixels.
        focal_length: Camera focal length in mm.
        horizontal_aperture: Horizontal aperture in mm.
        clipping_range: (near, far) clipping planes in meters.
        is_hand_dependent: Whether this camera config depends on hand type.
        description: Human-readable description of the camera placement.
    """
    name: str
    prim_path: str
    parent_link: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]  # wxyz quaternion
    width: int = 640
    height: int = 480
    focal_length: float = 24.0
    horizontal_aperture: float = 20.955
    clipping_range: Tuple[float, float] = (0.01, 100.0)
    is_hand_dependent: bool = False
    description: str = ""

    def to_isaac_lab_dict(self) -> Dict:
        """Convert to Isaac Lab TiledCameraCfg-compatible dictionary."""
        return {
            "prim_path": self.prim_path,
            "offset": {
                "pos": self.position,
                "rot": self.rotation,
            },
            "spawn": {
                "clipping_range": self.clipping_range,
                "focal_length": self.focal_length,
                "horizontal_aperture": self.horizontal_aperture,
            },
            "width": self.width,
            "height": self.height,
        }


# =============================================================================
# G1 Head Camera Configuration (Hand-Agnostic)
# =============================================================================

# Intel RealSense D435 mounted on G1 head
# This is the primary observation camera for manipulation tasks
G1_HEAD_CAMERA = CameraConfig(
    name="front_cam",
    prim_path="{ENV_REGEX_NS}/Robot/d435_link/front_cam",
    parent_link="d435_link",
    position=(0.0, 0.0, 0.0),
    # Rotation from Unitree config: (0.5, -0.5, 0.5, -0.5) in xyzw
    # Converting to wxyz: (-0.5, 0.5, -0.5, 0.5)
    rotation=(0.5, -0.5, 0.5, -0.5),  # wxyz - camera looking forward
    width=640,
    height=480,
    is_hand_dependent=False,
    description="Intel RealSense D435 on G1 head, looking forward with slight downward inclination",
)

# Alternative head camera config with explicit offsets for scenes without d435_link
# For G1 robot in Isaac Sim: robot faces +X direction, Y is left, Z is up
# Camera should be mounted at head height looking forward-down at the workspace
G1_HEAD_CAMERA_FALLBACK = CameraConfig(
    name="front_cam_fallback",
    prim_path="{ENV_REGEX_NS}/Robot/torso_link/front_cam",
    parent_link="torso_link",
    # Position: forward and up from torso to approximate head position
    # In robot frame: X=forward, Y=left, Z=up
    position=(0.25, 0.0, 0.45),  # 25cm forward, 45cm up (head height)
    # Rotation: Using Unitree d435 camera rotation (0.5, -0.5, 0.5, -0.5)
    # This is xyzw format from Unitree, same as Isaac Lab convention
    rotation=(0.5, -0.5, 0.5, -0.5),  # wxyz - same as Unitree d435
    width=640,
    height=480,
    is_hand_dependent=False,
    description="Fallback head camera attached to torso_link for scenes without d435_link",
)


# =============================================================================
# G1 Wrist Camera Configurations (Hand-Dependent)
# =============================================================================

# DEX3 Hand Wrist Cameras
# Based on unitree_sim_isaaclab camera_configs.py
DEX3_LEFT_WRIST_CAMERA = CameraConfig(
    name="left_wrist_cam",
    prim_path="{ENV_REGEX_NS}/Robot/left_hand_camera_base_link/left_wrist_cam",
    parent_link="left_hand_camera_base_link",
    position=(-0.04012, 0.07441, 0.15711),
    rotation=(0.50809, 0.00539, 0.86024, 0.0424),  # wxyz from Unitree config
    width=640,
    height=480,
    is_hand_dependent=True,
    description="Left wrist camera on DEX3-1 hand",
)

DEX3_RIGHT_WRIST_CAMERA = CameraConfig(
    name="right_wrist_cam",
    prim_path="{ENV_REGEX_NS}/Robot/right_hand_camera_base_link/right_wrist_cam",
    parent_link="right_hand_camera_base_link",
    position=(-0.04012, -0.07441, 0.15711),
    rotation=(0.50809, -0.00539, 0.86024, -0.0424),  # Mirrored for right hand
    width=640,
    height=480,
    is_hand_dependent=True,
    description="Right wrist camera on DEX3-1 hand",
)

# Inspire Hand Wrist Cameras
# Inspire hands use same camera mount links as DEX3 (verified from USD inspection)
# Links: left_hand_camera_base_link, right_hand_camera_base_link
INSPIRE_LEFT_WRIST_CAMERA = CameraConfig(
    name="left_wrist_cam",
    prim_path="{ENV_REGEX_NS}/Robot/left_hand_camera_base_link/left_wrist_cam",
    parent_link="left_hand_camera_base_link",
    position=(-0.04012, 0.07441, 0.15711),  # Same as DEX3
    rotation=(0.50809, 0.00539, 0.86024, 0.0424),  # wxyz - Same as DEX3
    width=640,
    height=480,
    is_hand_dependent=True,
    description="Left wrist camera on Inspire hand (same mount as DEX3)",
)

INSPIRE_RIGHT_WRIST_CAMERA = CameraConfig(
    name="right_wrist_cam",
    prim_path="{ENV_REGEX_NS}/Robot/right_hand_camera_base_link/right_wrist_cam",
    parent_link="right_hand_camera_base_link",
    position=(-0.04012, -0.07441, 0.15711),  # Same as DEX3, mirrored
    rotation=(0.50809, -0.00539, 0.86024, -0.0424),  # Mirrored for right hand
    width=640,
    height=480,
    is_hand_dependent=True,
    description="Right wrist camera on Inspire hand (same mount as DEX3)",
)

# Gripper Wrist Cameras (simple parallel gripper)
GRIPPER_LEFT_WRIST_CAMERA = CameraConfig(
    name="left_wrist_cam",
    prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link/left_wrist_cam",
    parent_link="left_wrist_yaw_link",
    position=(0.0, 0.03, 0.08),
    rotation=(0.5, -0.5, 0.5, -0.5),
    width=640,
    height=480,
    is_hand_dependent=True,
    description="Left wrist camera on parallel gripper",
)

GRIPPER_RIGHT_WRIST_CAMERA = CameraConfig(
    name="right_wrist_cam",
    prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/right_wrist_cam",
    parent_link="right_wrist_yaw_link",
    position=(0.0, -0.03, 0.08),
    rotation=(0.5, 0.5, 0.5, 0.5),
    width=640,
    height=480,
    is_hand_dependent=True,
    description="Right wrist camera on parallel gripper",
)


# =============================================================================
# World Camera Configuration
# =============================================================================

WORLD_CAMERA = CameraConfig(
    name="world_cam",
    prim_path="/World/PerspectiveCamera",
    parent_link="",  # World frame, no parent link
    position=(2.0, 0.0, 1.5),  # 2m in front, 1.5m high
    rotation=(0.9239, 0.0, 0.3827, 0.0),  # wxyz - 45 deg down
    width=1280,
    height=720,
    is_hand_dependent=False,
    description="External world camera for third-person observation",
)


# =============================================================================
# Camera Registry
# =============================================================================

@dataclass
class RobotCameraRegistry:
    """Registry of camera configurations for different robot and hand combinations."""

    # Head cameras (keyed by RobotType, hand-agnostic)
    head_cameras: Dict[RobotType, CameraConfig] = field(default_factory=dict)
    head_camera_fallbacks: Dict[RobotType, CameraConfig] = field(default_factory=dict)

    # Wrist cameras (keyed by (RobotType, HandType))
    wrist_cameras: Dict[Tuple[RobotType, HandType], Tuple[CameraConfig, CameraConfig]] = field(default_factory=dict)

    # World camera
    world_camera: CameraConfig = field(default_factory=lambda: WORLD_CAMERA)

    def register_head_camera(
        self,
        robot_type: RobotType,
        camera: CameraConfig,
        fallback: Optional[CameraConfig] = None,
    ):
        """Register a head camera for a robot type."""
        self.head_cameras[robot_type] = camera
        if fallback:
            self.head_camera_fallbacks[robot_type] = fallback

    def register_wrist_cameras(
        self,
        robot_type: RobotType,
        hand_type: HandType,
        left_camera: CameraConfig,
        right_camera: CameraConfig,
    ):
        """Register wrist cameras for a robot/hand combination."""
        self.wrist_cameras[(robot_type, hand_type)] = (left_camera, right_camera)

    def get_head_camera(
        self,
        robot_type: RobotType = RobotType.G1,
        use_fallback: bool = False,
    ) -> Optional[CameraConfig]:
        """Get head camera config for a robot type."""
        if use_fallback:
            return self.head_camera_fallbacks.get(robot_type)
        return self.head_cameras.get(robot_type)

    def get_wrist_cameras(
        self,
        robot_type: RobotType = RobotType.G1,
        hand_type: HandType = HandType.DEX3,
    ) -> Optional[Tuple[CameraConfig, CameraConfig]]:
        """Get wrist camera configs for a robot/hand combination."""
        return self.wrist_cameras.get((robot_type, hand_type))


# Global registry instance
CAMERA_REGISTRY = RobotCameraRegistry()

# Register G1 head camera
CAMERA_REGISTRY.register_head_camera(
    RobotType.G1,
    G1_HEAD_CAMERA,
    fallback=G1_HEAD_CAMERA_FALLBACK,
)

# Register G1 wrist cameras for each hand type
CAMERA_REGISTRY.register_wrist_cameras(
    RobotType.G1,
    HandType.DEX3,
    DEX3_LEFT_WRIST_CAMERA,
    DEX3_RIGHT_WRIST_CAMERA,
)

CAMERA_REGISTRY.register_wrist_cameras(
    RobotType.G1,
    HandType.INSPIRE,
    INSPIRE_LEFT_WRIST_CAMERA,
    INSPIRE_RIGHT_WRIST_CAMERA,
)

CAMERA_REGISTRY.register_wrist_cameras(
    RobotType.G1,
    HandType.GRIPPER,
    GRIPPER_LEFT_WRIST_CAMERA,
    GRIPPER_RIGHT_WRIST_CAMERA,
)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_head_camera_config(
    robot_type: RobotType = RobotType.G1,
    use_fallback: bool = False,
) -> CameraConfig:
    """Get the head camera configuration for a robot.

    The head camera is AGNOSTIC to hand type - the same configuration works
    regardless of which hands (DEX3, Inspire, Gripper) are used.

    Args:
        robot_type: The robot type (default: G1).
        use_fallback: If True, return fallback config for scenes without d435_link.

    Returns:
        CameraConfig for the head camera.
    """
    camera = CAMERA_REGISTRY.get_head_camera(robot_type, use_fallback)
    if camera is None:
        raise ValueError(f"No head camera registered for robot type: {robot_type}")
    return camera


def get_wrist_camera_configs(
    hand_type: HandType,
    robot_type: RobotType = RobotType.G1,
) -> Tuple[CameraConfig, CameraConfig]:
    """Get wrist camera configurations for a hand type.

    Wrist cameras ARE DEPENDENT on hand type, as different hands have
    different link structures and camera mount positions.

    Args:
        hand_type: The hand type (DEX3, INSPIRE, GRIPPER).
        robot_type: The robot type (default: G1).

    Returns:
        Tuple of (left_wrist_camera, right_wrist_camera) CameraConfigs.

    Raises:
        ValueError: If hand_type is NONE or no config is registered.
    """
    if hand_type == HandType.NONE:
        raise ValueError("Cannot get wrist cameras for HandType.NONE")

    cameras = CAMERA_REGISTRY.get_wrist_cameras(robot_type, hand_type)
    if cameras is None:
        raise ValueError(
            f"No wrist cameras registered for robot={robot_type}, hand={hand_type}"
        )
    return cameras


def get_world_camera_config() -> CameraConfig:
    """Get the world (external) camera configuration."""
    return CAMERA_REGISTRY.world_camera


def get_all_camera_configs(
    hand_type: HandType = HandType.NONE,
    robot_type: RobotType = RobotType.G1,
    include_head: bool = True,
    include_wrist: bool = False,
    include_world: bool = False,
    use_head_fallback: bool = False,
) -> List[CameraConfig]:
    """Get all camera configurations for a robot setup.

    Args:
        hand_type: The hand type for wrist cameras (required if include_wrist=True).
        robot_type: The robot type (default: G1).
        include_head: Whether to include head camera (default: True).
        include_wrist: Whether to include wrist cameras (default: False).
        include_world: Whether to include world camera (default: False).
        use_head_fallback: Use fallback head camera for scenes without d435_link.

    Returns:
        List of CameraConfig objects.
    """
    cameras = []

    if include_head:
        cameras.append(get_head_camera_config(robot_type, use_head_fallback))

    if include_wrist and hand_type != HandType.NONE:
        left, right = get_wrist_camera_configs(hand_type, robot_type)
        cameras.extend([left, right])

    if include_world:
        cameras.append(get_world_camera_config())

    return cameras


# =============================================================================
# Isaac Lab Integration Helpers
# =============================================================================

def create_tiled_camera_cfg_dict(camera: CameraConfig, env_regex: str = "env_.*") -> Dict:
    """Create a dictionary compatible with Isaac Lab's TiledCameraCfg.

    Args:
        camera: CameraConfig to convert.
        env_regex: Environment regex pattern (default: "env_.*").

    Returns:
        Dictionary that can be used to instantiate TiledCameraCfg.
    """
    prim_path = camera.prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/{env_regex}")

    return {
        "prim_path": prim_path,
        "offset": {
            "pos": camera.position,
            "rot": camera.rotation,
            "convention": "world",
        },
        "data_types": ["rgb"],
        "spawn": {
            "focal_length": camera.focal_length,
            "horizontal_aperture": camera.horizontal_aperture,
            "clipping_range": camera.clipping_range,
        },
        "width": camera.width,
        "height": camera.height,
    }


def check_link_exists_in_scene(stage, link_name: str, robot_prim_path: str) -> bool:
    """Check if a link exists in the USD stage.

    Args:
        stage: USD stage object.
        link_name: Name of the link to check.
        robot_prim_path: Base path to the robot prim.

    Returns:
        True if link exists, False otherwise.
    """
    full_path = f"{robot_prim_path}/{link_name}"
    prim = stage.GetPrimAtPath(full_path)
    return prim.IsValid() if prim else False


def get_available_camera_links(stage, robot_prim_path: str) -> Dict[str, bool]:
    """Check which camera-related links are available in a scene.

    Args:
        stage: USD stage object.
        robot_prim_path: Base path to the robot prim.

    Returns:
        Dictionary mapping link names to availability (True/False).
    """
    camera_links = [
        "d435_link",                    # Head camera
        "head_link",                     # Alternative head mount
        "torso_link",                    # Fallback
        "left_hand_camera_base_link",   # DEX3 left wrist
        "right_hand_camera_base_link",  # DEX3 right wrist
        "left_wrist_yaw_link",          # Inspire/Gripper left wrist
        "right_wrist_yaw_link",         # Inspire/Gripper right wrist
    ]

    return {
        link: check_link_exists_in_scene(stage, link, robot_prim_path)
        for link in camera_links
    }
