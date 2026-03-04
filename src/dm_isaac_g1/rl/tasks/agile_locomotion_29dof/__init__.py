# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""AGILE-style locomotion task for G1 29-DOF.

Based on NVIDIA WBC-AGILE reward structure with 24 reward terms for robust,
natural humanoid locomotion with sim-to-real transfer capabilities.

Reference: https://github.com/nvidia-isaac/WBC-AGILE
"""

import gymnasium as gym

gym.register(
    id="DM-G1-29dof-AgileLocomotion",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg:AgilePPORunnerCfg",
    },
)
