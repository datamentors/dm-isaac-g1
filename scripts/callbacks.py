# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""
Custom training callbacks for GROOT fine-tuning.

This file is deployed to /workspace/scripts/ on the workstation container
and imported by launch_finetune_v3.py during training.
"""

import logging

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class EarlyStopOnLossCallback(TrainerCallback):
    """Stop training when the smoothed loss drops below a threshold.

    This prevents overfitting on small datasets where the model can memorize
    trajectories. A very low training loss (e.g., < 0.01) on a 103-episode
    dataset typically indicates the model is memorizing rather than
    generalizing.

    Args:
        loss_threshold: Stop when smoothed loss drops below this value.
        window_size: Number of log entries to average over.
    """

    def __init__(self, loss_threshold: float = 0.01, window_size: int = 50):
        self.loss_threshold = loss_threshold
        self.window_size = window_size
        self.recent_losses: list[float] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if logs is None or "loss" not in logs:
            return
        loss = logs["loss"]
        self.recent_losses.append(loss)
        if len(self.recent_losses) > self.window_size:
            self.recent_losses.pop(0)

        avg_loss = sum(self.recent_losses) / len(self.recent_losses)
        if len(self.recent_losses) >= self.window_size and avg_loss < self.loss_threshold:
            logger.warning(
                f"Early stopping: smoothed loss {avg_loss:.6f} < threshold "
                f"{self.loss_threshold}. Step {state.global_step}. "
                f"Stopping to prevent overfitting."
            )
            control.should_training_stop = True
