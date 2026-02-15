#!/usr/bin/env python3
"""
Phase 2: Train G1 with Reinforcement Learning
Curriculum-based training following isaac-g1-ulc-vlm approach.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phases.phase2_rl.g1_env import G1RLEnvironment
from phases.phase2_rl.rewards import RewardManager, STAGE_REWARDS


def parse_args():
    parser = argparse.ArgumentParser(description="Train G1 with RL")
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Curriculum stage to train",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4096,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10000,
        help="Maximum training iterations",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML",
    )
    return parser.parse_args()


class DualActorCriticPPO:
    """Dual Actor-Critic PPO for G1 training.

    Uses separate networks for locomotion and arm control.
    """

    def __init__(
        self,
        obs_dim_locomotion: int = 57,
        obs_dim_arm: int = 55,
        action_dim_legs: int = 12,
        action_dim_arm: int = 5,
        hidden_dims: list = [256, 256, 128],
        learning_rate: float = 3e-4,
        device: str = "cuda",
    ):
        self.device = device
        self.obs_dim_locomotion = obs_dim_locomotion
        self.obs_dim_arm = obs_dim_arm
        self.action_dim_legs = action_dim_legs
        self.action_dim_arm = action_dim_arm

        # Build locomotion network
        self.locomotion_actor = self._build_mlp(
            obs_dim_locomotion, action_dim_legs, hidden_dims
        ).to(device)
        self.locomotion_critic = self._build_mlp(
            obs_dim_locomotion, 1, hidden_dims
        ).to(device)

        # Build arm network
        self.arm_actor = self._build_mlp(
            obs_dim_arm, action_dim_arm, hidden_dims
        ).to(device)
        self.arm_critic = self._build_mlp(
            obs_dim_arm, 1, hidden_dims
        ).to(device)

        # Optimizers
        self.locomotion_optimizer = torch.optim.Adam(
            list(self.locomotion_actor.parameters()) +
            list(self.locomotion_critic.parameters()),
            lr=learning_rate,
        )
        self.arm_optimizer = torch.optim.Adam(
            list(self.arm_actor.parameters()) +
            list(self.arm_critic.parameters()),
            lr=learning_rate,
        )

        # PPO parameters
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 1.0
        self.gamma = 0.99
        self.lam = 0.95

    def _build_mlp(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list,
    ) -> torch.nn.Module:
        """Build MLP network."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.ELU(),
            ])
            prev_dim = hidden_dim

        layers.append(torch.nn.Linear(prev_dim, output_dim))

        return torch.nn.Sequential(*layers)

    def get_action(
        self,
        obs: dict,
        deterministic: bool = False,
    ) -> tuple:
        """Get action from both actors.

        Returns:
            actions: Combined leg and arm actions
            log_probs: Log probabilities
            values: Value estimates
        """
        locomotion_obs = obs["locomotion"]
        arm_obs = obs["arm"]

        # Locomotion actions
        leg_mean = self.locomotion_actor(locomotion_obs)
        leg_std = torch.ones_like(leg_mean) * 0.3
        leg_dist = torch.distributions.Normal(leg_mean, leg_std)

        if deterministic:
            leg_actions = leg_mean
        else:
            leg_actions = leg_dist.sample()

        leg_log_prob = leg_dist.log_prob(leg_actions).sum(dim=-1)
        leg_value = self.locomotion_critic(locomotion_obs).squeeze(-1)

        # Arm actions
        arm_mean = self.arm_actor(arm_obs)
        arm_std = torch.ones_like(arm_mean) * 0.2
        arm_dist = torch.distributions.Normal(arm_mean, arm_std)

        if deterministic:
            arm_actions = arm_mean
        else:
            arm_actions = arm_dist.sample()

        arm_log_prob = arm_dist.log_prob(arm_actions).sum(dim=-1)
        arm_value = self.arm_critic(arm_obs).squeeze(-1)

        # Combine actions
        # Note: Full action includes all 29 joints, we only control 17 (12 legs + 5 arm)
        actions = torch.cat([leg_actions, arm_actions], dim=-1)
        log_probs = leg_log_prob + arm_log_prob
        values = (leg_value + arm_value) / 2.0  # Average values

        return actions, log_probs, values

    def update(
        self,
        obs_batch: dict,
        actions_batch: torch.Tensor,
        log_probs_old: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        num_epochs: int = 5,
        num_mini_batches: int = 4,
    ) -> dict:
        """Update policy using PPO."""
        batch_size = obs_batch["locomotion"].shape[0]
        mini_batch_size = batch_size // num_mini_batches

        losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(num_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]

                # Get mini-batch
                mb_obs = {
                    "locomotion": obs_batch["locomotion"][mb_indices],
                    "arm": obs_batch["arm"][mb_indices],
                }
                mb_actions = actions_batch[mb_indices]
                mb_log_probs_old = log_probs_old[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]

                # Get current policy outputs
                _, log_probs_new, values = self.get_action(mb_obs, deterministic=True)

                # PPO loss
                ratio = torch.exp(log_probs_new - mb_log_probs_old)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = torch.nn.functional.mse_loss(values, mb_returns)

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss

                # Update
                self.locomotion_optimizer.zero_grad()
                self.arm_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.locomotion_actor.parameters()) +
                    list(self.locomotion_critic.parameters()) +
                    list(self.arm_actor.parameters()) +
                    list(self.arm_critic.parameters()),
                    self.max_grad_norm,
                )
                self.locomotion_optimizer.step()
                self.arm_optimizer.step()

                losses["policy_loss"] += policy_loss.item()
                losses["value_loss"] += value_loss.item()

        # Average losses
        num_updates = num_epochs * num_mini_batches
        losses = {k: v / num_updates for k, v in losses.items()}

        return losses

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "locomotion_actor": self.locomotion_actor.state_dict(),
            "locomotion_critic": self.locomotion_critic.state_dict(),
            "arm_actor": self.arm_actor.state_dict(),
            "arm_critic": self.arm_critic.state_dict(),
            "locomotion_optimizer": self.locomotion_optimizer.state_dict(),
            "arm_optimizer": self.arm_optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.locomotion_actor.load_state_dict(checkpoint["locomotion_actor"])
        self.locomotion_critic.load_state_dict(checkpoint["locomotion_critic"])
        self.arm_actor.load_state_dict(checkpoint["arm_actor"])
        self.arm_critic.load_state_dict(checkpoint["arm_critic"])
        self.locomotion_optimizer.load_state_dict(checkpoint["locomotion_optimizer"])
        self.arm_optimizer.load_state_dict(checkpoint["arm_optimizer"])


def train(args):
    """Main training loop."""
    # Load environment
    load_dotenv(PROJECT_ROOT / ".env")

    print("=" * 60)
    print(f"Phase 2: Training G1 - Stage {args.stage}")
    print("=" * 60)

    # Set random seed
    torch.manual_seed(args.seed)

    # Create environment
    print(f"\nCreating environment with {args.num_envs} parallel instances...")
    env = G1RLEnvironment(
        num_envs=args.num_envs,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=args.seed,
    )
    env.create()

    # Create policy
    print("Creating Dual Actor-Critic PPO policy...")
    policy = DualActorCriticPPO(
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        policy.load(args.checkpoint)

    # Setup reward manager
    reward_manager = RewardManager(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    reward_manager.set_scales(STAGE_REWARDS[args.stage])
    print(f"Stage {args.stage} rewards: {STAGE_REWARDS[args.stage]}")

    # Setup logging
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "dm-isaac-g1"),
                config={
                    "stage": args.stage,
                    "num_envs": args.num_envs,
                    "max_iterations": args.max_iterations,
                    "seed": args.seed,
                },
            )
        except ImportError:
            print("wandb not available, skipping...")

    # Training loop
    print(f"\nStarting training for {args.max_iterations} iterations...")
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / f"stage_{args.stage}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    obs = env.reset()
    best_reward = float("-inf")

    for iteration in range(args.max_iterations):
        # Collect rollout
        rollout_obs = {"locomotion": [], "arm": []}
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []

        for step in range(128):  # Rollout length
            # Get action
            actions, log_probs, values = policy.get_action(obs)

            # Step environment
            next_obs, env_rewards, dones, infos = env.step(actions)

            # Compute custom rewards
            custom_rewards = reward_manager.compute_rewards(
                obs, actions, next_obs, targets=infos.get("targets")
            )
            rewards = custom_rewards  # Use custom rewards

            # Store rollout
            rollout_obs["locomotion"].append(obs["locomotion"])
            rollout_obs["arm"].append(obs["arm"])
            rollout_actions.append(actions)
            rollout_log_probs.append(log_probs)
            rollout_rewards.append(rewards)
            rollout_dones.append(dones)
            rollout_values.append(values)

            obs = next_obs

        # Stack rollout
        rollout_obs = {
            "locomotion": torch.stack(rollout_obs["locomotion"], dim=1),
            "arm": torch.stack(rollout_obs["arm"], dim=1),
        }
        rollout_actions = torch.stack(rollout_actions, dim=1)
        rollout_log_probs = torch.stack(rollout_log_probs, dim=1)
        rollout_rewards = torch.stack(rollout_rewards, dim=1)
        rollout_dones = torch.stack(rollout_dones, dim=1)
        rollout_values = torch.stack(rollout_values, dim=1)

        # Compute returns and advantages (GAE)
        returns = torch.zeros_like(rollout_rewards)
        advantages = torch.zeros_like(rollout_rewards)
        last_gae = 0

        for t in reversed(range(128)):
            if t == 127:
                next_value = rollout_values[:, t]
            else:
                next_value = rollout_values[:, t + 1]

            delta = rollout_rewards[:, t] + policy.gamma * next_value * (~rollout_dones[:, t]) - rollout_values[:, t]
            advantages[:, t] = last_gae = delta + policy.gamma * policy.lam * (~rollout_dones[:, t]) * last_gae
            returns[:, t] = advantages[:, t] + rollout_values[:, t]

        # Flatten rollout
        flat_obs = {
            "locomotion": rollout_obs["locomotion"].view(-1, rollout_obs["locomotion"].shape[-1]),
            "arm": rollout_obs["arm"].view(-1, rollout_obs["arm"].shape[-1]),
        }
        flat_actions = rollout_actions.view(-1, rollout_actions.shape[-1])
        flat_log_probs = rollout_log_probs.view(-1)
        flat_returns = returns.view(-1)
        flat_advantages = advantages.view(-1)

        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # Update policy
        losses = policy.update(
            flat_obs, flat_actions, flat_log_probs, flat_returns, flat_advantages
        )

        # Logging
        mean_reward = rollout_rewards.mean().item()
        if iteration % 10 == 0:
            print(
                f"Iter {iteration}/{args.max_iterations} | "
                f"Reward: {mean_reward:.4f} | "
                f"Policy Loss: {losses['policy_loss']:.4f} | "
                f"Value Loss: {losses['value_loss']:.4f}"
            )

        # Save checkpoint
        if iteration % 500 == 0 or mean_reward > best_reward:
            if mean_reward > best_reward:
                best_reward = mean_reward
                checkpoint_path = checkpoint_dir / "best.pt"
            else:
                checkpoint_path = checkpoint_dir / f"iter_{iteration}.pt"

            policy.save(str(checkpoint_path))
            print(f"Saved checkpoint: {checkpoint_path}")

    # Final save
    policy.save(str(checkpoint_dir / "final.pt"))
    print(f"\nTraining complete! Final checkpoint saved.")

    env.close()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
