"""
Reward Functions for G1 RL Training
Based on isaac-g1-ulc-vlm curriculum approach.
"""

import torch
from typing import Dict, Optional


class RewardManager:
    """Manages reward computation for G1 training."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.reward_scales = {}

    def set_scales(self, scales: Dict[str, float]) -> None:
        """Set reward scales."""
        self.reward_scales = scales

    def compute_rewards(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute total reward from all reward terms."""
        total_reward = torch.zeros(obs["full_state"].shape[0], device=self.device)

        for name, scale in self.reward_scales.items():
            if scale == 0.0:
                continue

            reward_fn = getattr(self, f"_reward_{name}", None)
            if reward_fn is not None:
                reward = reward_fn(obs, actions, next_obs, targets)
                total_reward += scale * reward

        return total_reward

    # === Standing/Balance Rewards ===

    def _reward_base_height(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Reward for maintaining target base height."""
        target_height = 0.95  # G1 standing height
        # Extract base height from observation
        base_pos = obs.get("base_pos", torch.zeros(obs["full_state"].shape[0], 3))
        current_height = base_pos[:, 2]
        height_error = torch.abs(current_height - target_height)
        return torch.exp(-height_error * 10.0)

    def _reward_orientation(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Reward for maintaining upright orientation."""
        # Projected gravity should be [0, 0, -1] when upright
        projected_gravity = obs.get(
            "projected_gravity",
            torch.zeros(obs["full_state"].shape[0], 3)
        )
        projected_gravity[:, 2] = -1.0  # default to upright if not available
        upright_error = torch.sum(
            torch.square(projected_gravity[:, :2]), dim=-1
        )
        return torch.exp(-upright_error * 5.0)

    def _reward_joint_regularization(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Penalize large joint deviations from default pose."""
        joint_pos = obs["full_state"][:, :29]
        default_pos = torch.zeros_like(joint_pos)  # default pose
        deviation = torch.sum(torch.square(joint_pos - default_pos), dim=-1)
        return -deviation

    # === Locomotion Rewards ===

    def _reward_velocity_tracking(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Reward for tracking commanded velocity."""
        if targets is None or "velocity_cmd" not in targets:
            return torch.zeros(obs["full_state"].shape[0], device=self.device)

        velocity_cmd = targets["velocity_cmd"]  # (N, 3)
        base_lin_vel = obs.get(
            "base_lin_vel",
            torch.zeros(obs["full_state"].shape[0], 3)
        )

        # Linear velocity tracking
        lin_vel_error = torch.sum(
            torch.square(base_lin_vel[:, :2] - velocity_cmd[:, :2]),
            dim=-1
        )
        return torch.exp(-lin_vel_error * 2.0)

    def _reward_energy(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Penalize energy consumption (action magnitude)."""
        return -torch.sum(torch.square(actions), dim=-1)

    # === Reaching Rewards ===

    def _reward_reaching_distance(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Reward for reaching target position."""
        if targets is None or "reach_target" not in targets:
            return torch.zeros(obs["full_state"].shape[0], device=self.device)

        target_pos = targets["reach_target"]  # (N, 3)
        ee_pos = obs.get(
            "end_effector_pos",
            torch.zeros(obs["full_state"].shape[0], 3)
        )

        distance = torch.norm(ee_pos - target_pos, dim=-1)
        return torch.exp(-distance * 10.0)

    def _reward_end_effector_orientation(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Reward for correct end-effector orientation."""
        if targets is None or "reach_orientation" not in targets:
            return torch.zeros(obs["full_state"].shape[0], device=self.device)

        target_quat = targets["reach_orientation"]
        ee_quat = obs.get(
            "end_effector_quat",
            torch.zeros(obs["full_state"].shape[0], 4)
        )
        ee_quat[:, -1] = 1.0  # default to identity quaternion

        # Quaternion distance
        dot_product = torch.sum(ee_quat * target_quat, dim=-1)
        orientation_error = 1.0 - torch.square(dot_product)
        return torch.exp(-orientation_error * 5.0)

    # === Anti-Gaming Rewards (Stage 7) ===

    def _reward_movement_incentive(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Incentivize movement to prevent standing still."""
        joint_vel = obs["full_state"][:, 29:]  # velocities
        movement = torch.sum(torch.abs(joint_vel), dim=-1)
        return torch.clamp(movement, 0.0, 1.0)

    def _reward_end_effector_displacement(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Reward actual end-effector displacement."""
        ee_pos = obs.get(
            "end_effector_pos",
            torch.zeros(obs["full_state"].shape[0], 3)
        )
        next_ee_pos = next_obs.get(
            "end_effector_pos",
            torch.zeros(obs["full_state"].shape[0], 3)
        )

        displacement = torch.norm(next_ee_pos - ee_pos, dim=-1)
        return displacement


# Stage-specific reward configurations
STAGE_REWARDS = {
    1: {  # Standing
        "base_height": 1.0,
        "orientation": 0.5,
        "joint_regularization": 0.1,
    },
    2: {  # Locomotion
        "velocity_tracking": 1.0,
        "orientation": 0.5,
        "energy": -0.01,
    },
    3: {  # Torso control
        "base_height": 1.0,
        "orientation": 0.5,
    },
    4: {  # Fixed-base reaching
        "reaching_distance": 1.0,
        "end_effector_orientation": 0.3,
    },
    5: {  # Dynamic arm reaching
        "reaching_distance": 1.0,
        "orientation": 0.5,
    },
    6: {  # Loco-manipulation
        "reaching_distance": 1.0,
        "velocity_tracking": 0.5,
        "orientation": 0.5,
    },
    7: {  # Anti-gaming reaching
        "reaching_distance": 1.0,
        "movement_incentive": 0.3,
        "end_effector_displacement": 0.2,
    },
}
