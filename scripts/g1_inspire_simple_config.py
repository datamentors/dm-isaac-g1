"""
Simplified G1 + Inspire Hand Config for GROOT N1.6 Training

Uses a single state/action key matching the dataset structure.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import (
    ModalityConfig,
    ActionConfig,
    ActionRepresentation,
    ActionType,
    ActionFormat,
)
from gr00t.data.embodiment_tags import EmbodimentTag

# Simplified config using single observation.state key
g1_inspire_simple_config = {
    # Video modality - just the camera name without observation.images prefix
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_left_high"],
    ),

    # State modality - single 53 DOF vector
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["observation.state"],  # Single key for full state
    ),

    # Action modality - single 53 DOF vector
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["action"],  # Single key for full action
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),

    # Language modality
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}

register_modality_config(
    g1_inspire_simple_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)

if __name__ == "__main__":
    print("G1 Inspire Simple Config registered")
    print("State: 53 DOF (observation.state)")
    print("Action: 53 DOF")
