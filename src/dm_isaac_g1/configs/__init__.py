"""Configuration modules for dm-isaac-g1."""

from dm_isaac_g1.configs.camera_configs import (
    # Camera configuration classes
    CameraConfig,
    RobotCameraRegistry,
    # Registry instance
    CAMERA_REGISTRY,
    # Helper functions
    get_head_camera_config,
    get_wrist_camera_configs,
    get_world_camera_config,
    get_all_camera_configs,
    # Robot types
    RobotType,
    HandType,
)

__all__ = [
    "CameraConfig",
    "RobotCameraRegistry",
    "CAMERA_REGISTRY",
    "get_head_camera_config",
    "get_wrist_camera_configs",
    "get_world_camera_config",
    "get_all_camera_configs",
    "RobotType",
    "HandType",
]
