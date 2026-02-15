"""
DM-ISAAC-G1: G1 Robot Training Suite

Fine-tuning, Inference, Imitation Learning, and Reinforcement Learning
for the Unitree G1 EDU 2 robot with Inspire Robotics Dexterous Hands.
"""

__version__ = "0.2.0"

from dm_isaac_g1.core.config import Config, load_config
from dm_isaac_g1.core.robot import G1InspireRobot

__all__ = [
    "Config",
    "load_config",
    "G1InspireRobot",
    "__version__",
]
