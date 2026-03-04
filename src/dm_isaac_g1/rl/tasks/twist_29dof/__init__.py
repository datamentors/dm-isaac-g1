# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""TWIST-style motion tracking locomotion task for G1 29-DOF.

Ports the reward structure from YanjieZe/TWIST (Isaac Gym) to Isaac Lab.
Full-body motion tracking with exponential tracking kernels for dancing/boxing.

Reference: https://github.com/YanjieZe/TWIST
Paper: "TWIST: Teleoperated Whole-Body Imitation System" (CoRL 2025)
"""

import gymnasium as gym

gym.register(
    id="DM-G1-29dof-TWIST",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg:TWISTPPORunnerCfg",
    },
)
