"""Keyboard velocity controller for sim2sim GUI mode.

Key bindings (MuJoCo viewer key codes):
    W / Up      Forward velocity (+vx)
    S / Down    Backward velocity (-vx)
    A / Left    Strafe left (+vy)
    D / Right   Strafe right (-vy)
    Q           Turn left (+wz)
    E           Turn right (-wz)
    Space       Stop all movement (zero velocity)
    R           Reset robot to initial pose
    1           FSM: Passive (damping)
    2           FSM: Stand
    3           FSM: Walk (enable policy)
"""

import numpy as np


# MuJoCo viewer GLFW key codes (subset)
_KEY_W = 87
_KEY_S = 83
_KEY_A = 65
_KEY_D = 68
_KEY_Q = 81
_KEY_E = 69
_KEY_R = 82
_KEY_SPACE = 32
_KEY_1 = 49
_KEY_2 = 50
_KEY_3 = 51
_KEY_UP = 265
_KEY_DOWN = 264
_KEY_LEFT = 263
_KEY_RIGHT = 262


class KeyboardController:
    """Incremental keyboard velocity controller.

    Each key press adjusts velocity by `vel_step`. Velocity is clamped
    to the configured limits. Space resets to zero.
    """

    def __init__(self, vel_step=0.1, max_vx=1.0, max_vy=0.5, max_wz=0.5):
        self.cmd_vel = np.zeros(3, dtype=np.float32)  # [vx, vy, wz]
        self.vel_step = vel_step
        self.limits = np.array([max_vx, max_vy, max_wz])
        self.reset_requested = False
        self.fsm_transition = None  # None, "passive", "stand", "walk"

    def process_key(self, key_code):
        """Process a single key press event."""
        if key_code in (_KEY_W, _KEY_UP):
            self.cmd_vel[0] = min(self.cmd_vel[0] + self.vel_step, self.limits[0])
        elif key_code in (_KEY_S, _KEY_DOWN):
            self.cmd_vel[0] = max(self.cmd_vel[0] - self.vel_step, -self.limits[0])
        elif key_code in (_KEY_A, _KEY_LEFT):
            self.cmd_vel[1] = min(self.cmd_vel[1] + self.vel_step, self.limits[1])
        elif key_code in (_KEY_D, _KEY_RIGHT):
            self.cmd_vel[1] = max(self.cmd_vel[1] - self.vel_step, -self.limits[1])
        elif key_code == _KEY_Q:
            self.cmd_vel[2] = min(self.cmd_vel[2] + self.vel_step, self.limits[2])
        elif key_code == _KEY_E:
            self.cmd_vel[2] = max(self.cmd_vel[2] - self.vel_step, -self.limits[2])
        elif key_code == _KEY_SPACE:
            self.cmd_vel[:] = 0.0
        elif key_code == _KEY_R:
            self.reset_requested = True
        elif key_code == _KEY_1:
            self.fsm_transition = "passive"
        elif key_code == _KEY_2:
            self.fsm_transition = "stand"
        elif key_code == _KEY_3:
            self.fsm_transition = "walk"

    def get_cmd_vel(self):
        return self.cmd_vel.copy()

    def consume_reset(self):
        """Check and clear reset flag."""
        if self.reset_requested:
            self.reset_requested = False
            return True
        return False

    def consume_fsm_transition(self):
        """Check and clear FSM transition request."""
        t = self.fsm_transition
        self.fsm_transition = None
        return t

    def status_line(self, fsm_state="walk", sim_time=0.0, step_count=0):
        """Return a one-line status string for terminal HUD."""
        return (f"[{fsm_state.upper():7s}] "
                f"vx={self.cmd_vel[0]:+.2f} vy={self.cmd_vel[1]:+.2f} "
                f"wz={self.cmd_vel[2]:+.2f}  "
                f"t={sim_time:.1f}s  step={step_count}")
