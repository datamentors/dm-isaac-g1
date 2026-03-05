"""Finite State Machine for sim2sim robot control.

Replicates the C++ CtrlFSM behavior (State_Passive, State_FixStand,
State_RLBase) in Python for sim2sim MuJoCo validation.

States:
    PASSIVE  - Damping mode (kp=0, kd=damping). Robot goes limp.
    STAND    - Linear interpolation to standing pose over stand_duration.
    WALK     - Policy inference active, velocity commands accepted.

Transitions:
    PASSIVE -> STAND  : Key '2'
    STAND   -> WALK   : Key '3'
    WALK    -> STAND  : Key '2'
    ANY     -> PASSIVE: Key '1'
"""

from enum import Enum

import numpy as np


class FSMState(Enum):
    PASSIVE = "passive"
    STAND = "stand"
    WALK = "walk"


class RobotFSM:
    """Finite state machine for robot control in sim2sim."""

    def __init__(self, deploy_cfg, n_joints):
        self.state = FSMState.STAND
        self.n_joints = n_joints
        self.deploy_cfg = deploy_cfg

        self.transition_time = 0.0
        self.stand_duration = 2.0  # seconds to interpolate to stand
        self.start_pos = None  # captured at transition start

        # PD gains from config
        self.kp = np.ones(n_joints) * 100.0
        self.kd = np.ones(n_joints) * 2.0
        if "stiffness" in deploy_cfg:
            kp_cfg = deploy_cfg["stiffness"]
            self.kp[:min(len(kp_cfg), n_joints)] = kp_cfg[:n_joints]
        if "damping" in deploy_cfg:
            kd_cfg = deploy_cfg["damping"]
            self.kd[:min(len(kd_cfg), n_joints)] = kd_cfg[:n_joints]

        # Default standing pose
        self.default_pos = np.zeros(n_joints)
        if "default_joint_pos" in deploy_cfg:
            dp = deploy_cfg["default_joint_pos"]
            self.default_pos[:min(len(dp), n_joints)] = dp[:n_joints]

    def transition_to(self, new_state_name, data, sim_time):
        """Transition to a new state.

        Args:
            new_state_name: "passive", "stand", or "walk"
            data: MuJoCo MjData (to capture current joint positions)
            sim_time: current simulation time
        """
        new_state = FSMState(new_state_name)
        if new_state == self.state:
            return

        old = self.state
        self.state = new_state
        self.transition_time = sim_time

        n_qpos = min(self.n_joints, data.qpos.shape[0] - 7)
        self.start_pos = np.zeros(self.n_joints)
        self.start_pos[:n_qpos] = data.qpos[7:7 + n_qpos].copy()

        print(f"[FSM] {old.value} -> {new_state.value}")

    @property
    def policy_active(self):
        return self.state == FSMState.WALK

    def compute_torque(self, data, policy_target_pos, sim_time):
        """Compute actuator torques based on current FSM state.

        Args:
            data: MuJoCo MjData
            policy_target_pos: target positions from policy (only used in WALK)
            sim_time: current simulation time

        Returns:
            np.ndarray of torques (n_joints,)
        """
        n = self.n_joints
        n_qpos = min(n, data.qpos.shape[0] - 7)
        n_qvel = min(n, data.qvel.shape[0] - 6)
        current_pos = data.qpos[7:7 + n_qpos]
        current_vel = data.qvel[6:6 + n_qvel]

        if self.state == FSMState.PASSIVE:
            # Damping only, no position tracking
            torque = np.zeros(n)
            torque[:n_qvel] = -self.kd[:n_qvel] * current_vel
            return torque

        elif self.state == FSMState.STAND:
            # Linear interpolation to default standing pose
            t = sim_time - self.transition_time
            alpha = min(t / self.stand_duration, 1.0)
            target = self.start_pos.copy()
            target[:n_qpos] = (
                self.start_pos[:n_qpos]
                + alpha * (self.default_pos[:n_qpos] - self.start_pos[:n_qpos])
            )
            torque = np.zeros(n)
            torque[:n_qpos] = (
                self.kp[:n_qpos] * (target[:n_qpos] - current_pos)
                - self.kd[:n_qvel] * current_vel[:min(n_qpos, n_qvel)]
            )
            return torque

        elif self.state == FSMState.WALK:
            # Policy-driven: PD control to policy targets
            if policy_target_pos is None:
                return np.zeros(n)
            n_act = min(len(policy_target_pos), n_qpos)
            torque = np.zeros(n)
            torque[:n_act] = (
                self.kp[:n_act] * (policy_target_pos[:n_act] - current_pos[:n_act])
                - self.kd[:n_act] * current_vel[:min(n_act, n_qvel)]
            )
            return torque

        return np.zeros(n)
