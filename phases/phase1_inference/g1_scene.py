"""
G1 Isaac Sim Scene Setup
Creates and manages the G1 robot scene in Isaac Sim
"""

import os
from pathlib import Path
from typing import Optional

import yaml

# Isaac Sim imports will be available when running on the workstation
# These are stubbed for local development
try:
    import omni
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.objects import DynamicCuboid, VisualSphere
    from omni.isaac.core.prims import XFormPrim
    from pxr import Gf, UsdPhysics
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    print("Isaac Sim not available. Running in stub mode.")


class G1Scene:
    """Manages the G1 robot scene in Isaac Sim."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        physics_dt: float = 0.005,
        rendering_dt: float = 0.0167,
    ):
        """Initialize the G1 scene.

        Args:
            config_path: Path to scene configuration YAML
            physics_dt: Physics simulation timestep
            rendering_dt: Rendering timestep
        """
        self.physics_dt = physics_dt
        self.rendering_dt = rendering_dt

        # Load configuration
        if config_path:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        self.world = None
        self.robot = None
        self.objects = {}

    def _default_config(self) -> dict:
        """Return default scene configuration."""
        return {
            "scene": {
                "robot": {
                    "spawn_position": [0.0, 0.0, 0.95],
                    "spawn_orientation": [0.0, 0.0, 0.0, 1.0],
                },
                "objects": {
                    "table": {"enabled": True},
                    "target_sphere": {"enabled": True, "radius": 0.05},
                },
            }
        }

    def setup(self) -> None:
        """Set up the Isaac Sim scene."""
        if not ISAAC_AVAILABLE:
            print("Isaac Sim not available. Cannot set up scene.")
            return

        # Create world
        self.world = World(
            physics_dt=self.physics_dt,
            rendering_dt=self.rendering_dt,
            stage_units_in_meters=1.0,
        )

        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Load G1 robot
        self._add_robot()

        # Add scene objects
        self._add_objects()

        # Reset world
        self.world.reset()

    def _add_robot(self) -> None:
        """Add the G1 robot to the scene."""
        robot_config = self.config["scene"]["robot"]
        position = robot_config.get("spawn_position", [0.0, 0.0, 0.95])
        orientation = robot_config.get("spawn_orientation", [0.0, 0.0, 0.0, 1.0])

        # Try to find G1 USD asset
        asset_paths = [
            "assets/g1_29dof_rev_1_0.usd",
            os.path.expanduser("~/dm-isaac-g1/assets/g1_29dof_rev_1_0.usd"),
            "/isaac-sim/standalone_examples/api/omni.isaac.lab/unitree_g1/g1.usd",
        ]

        usd_path = None
        for path in asset_paths:
            if os.path.exists(path):
                usd_path = path
                break

        if usd_path is None:
            print("Warning: G1 USD asset not found. Using default humanoid.")
            # Fall back to built-in humanoid or placeholder
            return

        # Add robot to stage
        add_reference_to_stage(
            usd_path=usd_path,
            prim_path="/World/G1",
        )

        self.robot = self.world.scene.add(
            Robot(
                prim_path="/World/G1",
                name="g1_robot",
                position=position,
                orientation=orientation,
            )
        )

    def _add_objects(self) -> None:
        """Add objects to the scene."""
        objects_config = self.config["scene"].get("objects", {})

        # Add table
        if objects_config.get("table", {}).get("enabled", False):
            table_pos = objects_config["table"].get("position", [0.6, 0.0, 0.4])
            table_size = objects_config["table"].get("size", [0.8, 0.6, 0.02])

            self.objects["table"] = self.world.scene.add(
                DynamicCuboid(
                    prim_path="/World/Table",
                    name="table",
                    position=table_pos,
                    size=table_size,
                    color=[0.5, 0.35, 0.2],  # wood color
                )
            )

        # Add target sphere
        if objects_config.get("target_sphere", {}).get("enabled", False):
            sphere_config = objects_config["target_sphere"]
            radius = sphere_config.get("radius", 0.05)
            color = sphere_config.get("color", [1.0, 0.0, 0.0])

            self.objects["target"] = self.world.scene.add(
                VisualSphere(
                    prim_path="/World/Target",
                    name="target_sphere",
                    position=[0.5, 0.0, 1.0],
                    radius=radius,
                    color=color,
                )
            )

    def get_robot_state(self) -> dict:
        """Get the current robot state."""
        if self.robot is None:
            return {}

        return {
            "joint_positions": self.robot.get_joint_positions(),
            "joint_velocities": self.robot.get_joint_velocities(),
            "world_position": self.robot.get_world_pose()[0],
            "world_orientation": self.robot.get_world_pose()[1],
        }

    def set_robot_action(self, joint_positions: list) -> None:
        """Set robot joint position targets."""
        if self.robot is not None:
            self.robot.set_joint_position_targets(joint_positions)

    def set_target_position(self, position: list) -> None:
        """Set the target sphere position."""
        if "target" in self.objects:
            self.objects["target"].set_world_pose(position=position)

    def step(self) -> None:
        """Step the simulation."""
        if self.world is not None:
            self.world.step(render=True)

    def reset(self) -> None:
        """Reset the scene."""
        if self.world is not None:
            self.world.reset()

    def close(self) -> None:
        """Close and clean up the scene."""
        if self.world is not None:
            self.world.stop()


def create_standalone_scene():
    """Create a standalone Isaac Sim scene for testing."""
    if not ISAAC_AVAILABLE:
        print("Isaac Sim not available. Cannot create standalone scene.")
        return None

    # Initialize Omniverse
    from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"headless": False})

    # Create scene
    scene = G1Scene()
    scene.setup()

    return scene, simulation_app
