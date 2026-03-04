# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""SoFTA-style smooth locomotion task for G1 29-DOF.

Ports the reward structure from LeCAR-Lab/SoFTA (Isaac Gym) to Isaac Lab.
Smooth force-torque-aware locomotion with end-effector stabilization.

Reference: https://github.com/LeCAR-Lab/SoFTA
Paper: "Hold My Beer: Learning Gentle Humanoid Locomotion" (2025)
"""

import gymnasium as gym

gym.register(
    id="DM-G1-29dof-SoFTA",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg:SoFTAPPORunnerCfg",
    },
)
