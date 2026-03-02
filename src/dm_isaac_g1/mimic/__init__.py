# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""Mimic (imitation learning) training pipeline for G1 humanoid.

Leverages unitree_rl_lab's mimic MDP (commands, rewards, observations, terminations,
events) and adds dm-isaac-g1-specific tasks and training scripts.

Requires unitree_rl_lab to be installed:
    pip install -e /path/to/unitree_rl_lab/source/unitree_rl_lab
"""
