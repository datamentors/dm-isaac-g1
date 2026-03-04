#!/usr/bin/env python3
"""Standalone policy export script — no Isaac Sim required.

Loads an RSL-RL checkpoint and exports ONNX + JIT policy files.
Works on any machine with PyTorch, no GPU rendering needed.

Usage:
    python src/dm_isaac_g1/mimic/scripts/export_policy.py \
        --task DM-G1-29dof-Mimic-CR7TiktokUEFA \
        --checkpoint logs/rsl_rl/.../model_14500.pt
"""

import argparse
import importlib
import os
import sys

import torch

# Minimal Isaac Lab / RSL-RL imports (no Isaac Sim dependency)
# We need the task registration to get the agent config (network arch, etc.)
# This import does NOT require Isaac Sim to be running.

parser = argparse.ArgumentParser(description="Export trained policy to ONNX + JIT")
parser.add_argument("--task", type=str, required=True, help="Gymnasium task ID")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
args = parser.parse_args()


def main():
    # Register our tasks with gymnasium (this imports task configs but NOT Isaac Sim)
    import dm_isaac_g1.mimic.tasks  # noqa: F401
    import gymnasium as gym

    # Get agent config from task registration
    spec = gym.spec(args.task)
    rsl_rl_cfg_entry = spec.kwargs["rsl_rl_cfg_entry_point"]
    module_path, class_name = rsl_rl_cfg_entry.rsplit(":", 1)
    agent_cfg = getattr(importlib.import_module(module_path), class_name)()

    # Get environment config to determine observation/action dims
    play_cfg_entry = spec.kwargs["play_env_cfg_entry_point"]
    module_path, class_name = play_cfg_entry.rsplit(":", 1)
    env_cfg = getattr(importlib.import_module(module_path), class_name)()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Checkpoint keys: {list(ckpt.keys())}")

    # The checkpoint contains 'model_state_dict' and optionally 'normalizer_state_dict'
    model_state_dict = ckpt.get("model_state_dict", ckpt)

    # Determine network architecture from agent config
    policy_cfg = agent_cfg.to_dict().get("policy", {})
    print(f"Policy config: {policy_cfg}")

    # Infer observation and action dimensions from model weights
    # The actor network's first layer tells us obs_dim, last layer tells us action_dim
    actor_keys = [k for k in model_state_dict.keys() if "actor" in k]
    if actor_keys:
        first_weight = [k for k in actor_keys if "weight" in k][0]
        last_weight = [k for k in actor_keys if "weight" in k][-1]
        obs_dim = model_state_dict[first_weight].shape[1]
        action_dim = model_state_dict[last_weight].shape[0]
        print(f"Inferred dims: obs={obs_dim}, action={action_dim}")
    else:
        print("WARNING: Could not find actor weights in checkpoint")
        obs_dim = None
        action_dim = None

    # Build the policy network
    from rsl_rl.modules import ActorCritic, ActorCriticRecurrent

    # Try to reconstruct the policy from the config
    init_noise_std = policy_cfg.get("init_noise_std", 1.0)
    actor_hidden_dims = policy_cfg.get("actor_hidden_dims", [256, 256, 256])
    critic_hidden_dims = policy_cfg.get("critic_hidden_dims", [256, 256, 256])
    activation = policy_cfg.get("activation", "elu")

    if obs_dim is None:
        print("ERROR: Cannot determine observation dimension from checkpoint")
        sys.exit(1)

    # Check if recurrent
    is_recurrent = any("rnn" in k or "memory" in k for k in model_state_dict.keys())

    if is_recurrent:
        rnn_type = policy_cfg.get("rnn_type", "lstm")
        rnn_hidden_size = policy_cfg.get("rnn_hidden_size", 256)
        rnn_num_layers = policy_cfg.get("rnn_num_layers", 1)
        policy_nn = ActorCriticRecurrent(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            rnn_type=rnn_type,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
        )
    else:
        policy_nn = ActorCritic(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

    # Load weights
    policy_nn.load_state_dict(model_state_dict)
    policy_nn.eval()
    print(f"Policy network loaded ({type(policy_nn).__name__})")

    # Handle normalizer
    normalizer = None
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer

    # Export
    log_dir = os.path.dirname(args.checkpoint)
    export_dir = os.path.join(log_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)

    # JIT export
    jit_path = os.path.join(export_dir, "policy.pt")
    try:
        # Create dummy input for tracing
        dummy_obs = torch.zeros(1, obs_dim)

        # Extract just the actor for export
        if hasattr(policy_nn, "actor"):
            actor = policy_nn.actor
        elif hasattr(policy_nn, "act"):
            actor = policy_nn
        else:
            actor = policy_nn

        # Trace the actor
        traced = torch.jit.trace(actor, dummy_obs)
        torch.jit.save(traced, jit_path)
        print(f"Exported JIT policy: {jit_path}")
    except Exception as e:
        print(f"JIT export failed: {e}")
        # Fallback: save state dict
        jit_path = os.path.join(export_dir, "policy.pt")
        torch.save({"model_state_dict": model_state_dict}, jit_path)
        print(f"Saved state dict instead: {jit_path}")

    # ONNX export
    onnx_path = os.path.join(export_dir, "policy.onnx")
    try:
        dummy_obs = torch.zeros(1, obs_dim)
        torch.onnx.export(
            actor,
            dummy_obs,
            onnx_path,
            input_names=["obs"],
            output_names=["actions"],
            opset_version=11,
        )
        print(f"Exported ONNX policy: {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")

    print(f"\nExport complete. Files in: {export_dir}")
    for f in sorted(os.listdir(export_dir)):
        size = os.path.getsize(os.path.join(export_dir, f))
        print(f"  {f} ({size:,} bytes)")


if __name__ == "__main__":
    main()
