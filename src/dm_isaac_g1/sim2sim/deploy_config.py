"""Deploy config loading and defaults for sim2sim."""

import os
from pathlib import Path

import yaml


def load_deploy_config(yaml_path):
    """Load deploy.yaml and return config dict."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def auto_detect_deploy_yaml(policy_path):
    """Find deploy.yaml near the policy file.

    Searches:
      1. {policy_dir}/../params/deploy.yaml  (standard export layout)
      2. {policy_dir}/deploy.yaml
      3. {policy_dir}/../deploy.yaml
    """
    policy_dir = Path(policy_path).parent
    candidates = [
        policy_dir.parent / "params" / "deploy.yaml",
        policy_dir / "deploy.yaml",
        policy_dir.parent / "deploy.yaml",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def get_default_g1_29dof_deploy_cfg():
    """Return a default deploy config for G1 29-DOF velocity policy.

    Used when no deploy.yaml is available, allowing quick testing.
    Matches the standard velocity/v0 deploy.yaml structure.
    """
    n = 29
    return {
        "joint_ids_map": list(range(n)),
        "step_dt": 0.02,
        "stiffness": [100.0] * 6 + [100.0] * 6 + [200.0, 200.0, 200.0] + [40.0] * 14,
        "damping": [2.0] * 6 + [2.0] * 6 + [5.0, 5.0, 5.0] + [10.0] * 14,
        "default_joint_pos": [
            -0.1, -0.1, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.3, 0.3, 0.3,
            0.3, -0.2, -0.2, 0.25, -0.25,
            0.0, 0.0, 0.0, 0.0,
            0.97, 0.97, 0.15, -0.15,
            0.0, 0.0, 0.0, 0.0,
        ],
        "commands": {
            "base_velocity": {
                "ranges": {
                    "lin_vel_x": [-0.5, 1.0],
                    "lin_vel_y": [-0.3, 0.3],
                    "ang_vel_z": [-0.2, 0.2],
                    "heading": None,
                },
            },
        },
        "actions": {
            "JointPositionAction": {
                "scale": [0.25] * n,
                "offset": [
                    -0.1, -0.1, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.3, 0.3, 0.3,
                    0.3, -0.2, -0.2, 0.25, -0.25,
                    0.0, 0.0, 0.0, 0.0,
                    0.97, 0.97, 0.15, -0.15,
                    0.0, 0.0, 0.0, 0.0,
                ],
                "clip": None,
                "joint_names": [".*"],
                "joint_ids": None,
            },
        },
        "observations": {
            "base_ang_vel": {
                "params": {},
                "clip": None,
                "scale": [0.2, 0.2, 0.2],
                "history_length": 1,
            },
            "projected_gravity": {
                "params": {},
                "clip": None,
                "scale": [1.0, 1.0, 1.0],
                "history_length": 1,
            },
            "velocity_commands": {
                "params": {"command_name": "base_velocity"},
                "clip": None,
                "scale": [1.0, 1.0, 1.0],
                "history_length": 1,
            },
            "joint_pos_rel": {
                "params": {},
                "clip": None,
                "scale": [1.0] * n,
                "history_length": 1,
            },
            "joint_vel_rel": {
                "params": {},
                "clip": None,
                "scale": [0.05] * n,
                "history_length": 1,
            },
            "last_action": {
                "params": {},
                "clip": None,
                "scale": [1.0] * n,
                "history_length": 1,
            },
        },
    }


def find_mujoco_scene():
    """Auto-detect G1 MuJoCo scene XML."""
    candidates = [
        # Workstation / ECS
        "/workspace/unitree_mujoco/unitree_robots/g1/scene.xml",
        "/workspace/unitree_model/G1/g1.xml",
        # Home directory clones
        os.path.expanduser("~/unitree_mujoco/unitree_robots/g1/scene.xml"),
    ]
    # Search common repo locations
    for base in ["/workspace", os.path.expanduser("~"), "."]:
        candidates.append(os.path.join(
            base, "unitree_rl_lab", "deploy", "robots", "g1_29dof", "resources", "scene.xml"
        ))

    # Search relative to this package (repo root)
    repo_root = Path(__file__).parents[4]  # sim2sim -> dm_isaac_g1 -> src -> dm-isaac-g1
    candidates.append(str(
        repo_root / "unitree_rl_lab" / "deploy" / "robots" / "g1_29dof" / "resources" / "scene.xml"
    ))

    # macOS Homebrew / pip-installed mujoco menagerie
    candidates.append(os.path.expanduser(
        "~/mujoco_menagerie/unitree_g1/scene.xml"
    ))

    for p in candidates:
        if os.path.exists(p):
            return p
    return None
