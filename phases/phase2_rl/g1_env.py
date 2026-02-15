"""
G1 Isaac Lab RL Environment
Implements the G1 robot environment for reinforcement learning using Isaac Lab.
Based on isaac-g1-ulc-vlm architecture.
"""

import os
from typing import Optional

import numpy as np
import torch

# Isaac Lab imports (available on workstation)
try:
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.assets import Articulation, ArticulationCfg
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
    from omni.isaac.lab.managers import EventTermCfg as EventTerm
    from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
    from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
    from omni.isaac.lab.managers import RewardTermCfg as RewTerm
    from omni.isaac.lab.managers import SceneEntityCfg
    from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
    from omni.isaac.lab.scene import InteractiveSceneCfg
    from omni.isaac.lab.utils import configclass
    ISAAC_LAB_AVAILABLE = True
except ImportError:
    ISAAC_LAB_AVAILABLE = False
    print("Isaac Lab not available. Running in stub mode.")


# Environment configuration
if ISAAC_LAB_AVAILABLE:
    @configclass
    class G1SceneCfg(InteractiveSceneCfg):
        """Configuration for the G1 scene."""

        # Ground plane
        ground = sim_utils.GroundPlaneCfg()

        # Dome light
        dome_light = sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(1.0, 1.0, 1.0),
        )

        # G1 Robot
        robot: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path="assets/g1_29dof_rev_1_0.usd",
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.95),
                joint_pos={
                    ".*": 0.0,
                },
            ),
            actuators={
                "legs": sim_utils.ImplicitActuatorCfg(
                    joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"],
                    stiffness=100.0,
                    damping=10.0,
                ),
                "arms": sim_utils.ImplicitActuatorCfg(
                    joint_names_expr=[".*shoulder.*", ".*elbow.*", ".*wrist.*"],
                    stiffness=50.0,
                    damping=5.0,
                ),
            },
        )

    @configclass
    class G1EnvCfg(ManagerBasedEnvCfg):
        """Configuration for the G1 RL environment."""

        # Scene
        scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)

        # Simulation settings
        sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
            dt=0.005,
            render_interval=4,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )

        # Episode settings
        episode_length_s: float = 20.0


class G1RLEnvironment:
    """G1 RL Environment wrapper for training."""

    def __init__(
        self,
        num_envs: int = 4096,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.device = device
        self.seed = seed

        # Observation and action dimensions
        # Locomotion: 57 obs -> 12 actions (legs)
        # Arms: 55 obs -> 5 actions (per arm)
        self.obs_dim_locomotion = 57
        self.obs_dim_arm = 55
        self.action_dim_legs = 12
        self.action_dim_arm = 5

        self.env = None

    def create(self) -> None:
        """Create the Isaac Lab environment."""
        if not ISAAC_LAB_AVAILABLE:
            print("Isaac Lab not available. Cannot create environment.")
            return

        # Create environment
        env_cfg = G1EnvCfg()
        env_cfg.scene.num_envs = self.num_envs

        self.env = ManagerBasedEnv(cfg=env_cfg)

    def reset(self) -> dict:
        """Reset all environments."""
        if self.env is None:
            return self._stub_observation()

        obs_dict, _ = self.env.reset()
        return self._process_observation(obs_dict)

    def step(self, actions: torch.Tensor) -> tuple:
        """Step the environment.

        Args:
            actions: Joint position targets (num_envs, action_dim)

        Returns:
            observations, rewards, dones, infos
        """
        if self.env is None:
            return self._stub_step()

        obs_dict, rewards, dones, _, infos = self.env.step(actions)
        observations = self._process_observation(obs_dict)

        return observations, rewards, dones, infos

    def _process_observation(self, obs_dict: dict) -> dict:
        """Process raw observations into locomotion and arm observations."""
        # Extract relevant observations
        joint_pos = obs_dict.get("joint_pos", torch.zeros(self.num_envs, 29))
        joint_vel = obs_dict.get("joint_vel", torch.zeros(self.num_envs, 29))
        base_lin_vel = obs_dict.get("base_lin_vel", torch.zeros(self.num_envs, 3))
        base_ang_vel = obs_dict.get("base_ang_vel", torch.zeros(self.num_envs, 3))
        projected_gravity = obs_dict.get("projected_gravity", torch.zeros(self.num_envs, 3))

        # Build locomotion observation (57 dim)
        # [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
        #  leg_joint_pos(12), leg_joint_vel(12), commands(3), ...]
        locomotion_obs = torch.cat([
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            joint_pos[:, :12],  # leg joints
            joint_vel[:, :12],
        ], dim=-1)

        # Build arm observation (55 dim)
        # [arm_joint_pos(5), arm_joint_vel(5), target_pos(3), ...]
        arm_obs = torch.cat([
            joint_pos[:, 12:17],  # right arm joints
            joint_vel[:, 12:17],
        ], dim=-1)

        return {
            "locomotion": locomotion_obs,
            "arm": arm_obs,
            "full_state": torch.cat([joint_pos, joint_vel], dim=-1),
        }

    def _stub_observation(self) -> dict:
        """Return stub observation when Isaac Lab is not available."""
        return {
            "locomotion": torch.zeros(self.num_envs, self.obs_dim_locomotion),
            "arm": torch.zeros(self.num_envs, self.obs_dim_arm),
            "full_state": torch.zeros(self.num_envs, 58),  # 29 pos + 29 vel
        }

    def _stub_step(self) -> tuple:
        """Return stub step result when Isaac Lab is not available."""
        obs = self._stub_observation()
        rewards = torch.zeros(self.num_envs)
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        infos = {}
        return obs, rewards, dones, infos

    def close(self) -> None:
        """Close the environment."""
        if self.env is not None:
            self.env.close()
