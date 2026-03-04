# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""FALCON-style force-adaptive locomotion task for G1 29-DOF.

Ports the reward structure from LeCAR-Lab/FALCON (Isaac Gym) to Isaac Lab.
Force-adaptive locomotion with payload handling and dual-agent decomposition.

Reference: https://github.com/LeCAR-Lab/FALCON
Paper: "FALCON: Learning Force-Adaptive Humanoid Loco-Manipulation" (L4DC 2026)
"""

import gymnasium as gym

gym.register(
    id="DM-G1-29dof-FALCON",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg:FALCONPPORunnerCfg",
    },
)
