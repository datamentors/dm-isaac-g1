# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""29-DOF military march velocity-tracking environment config.

Re-exports the full config from unitree_rl_lab's 29dof_military_march task.
The directory name starts with a digit so we use importlib to load it.

All reward weights, observations, events, and curriculum are inherited as-is:
  - Gait period: 0.8s (faster cadence)
  - Waist/leg deviation: -1.0 (strong control)
  - Flat orientation: -5.0
  - Forward velocity: 0.3-0.5 m/s
  - Flat terrain (parade ground)
"""

import importlib

# The unitree_rl_lab directory 29dof_military_march starts with a digit,
# so we need importlib to load it.
_module = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof_military_march.velocity_env_cfg"
)

RobotEnvCfg = _module.RobotEnvCfg
RobotPlayEnvCfg = _module.RobotPlayEnvCfg
