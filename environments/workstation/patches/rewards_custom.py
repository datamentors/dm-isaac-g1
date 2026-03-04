# ============================================================================
# Datamentors Custom Reward Functions
# Appended to unitree_rl_lab/tasks/locomotion/mdp/rewards.py at build time.
# Source: dm-isaac-g1/environments/workstation/patches/rewards_custom.py
# ============================================================================


def arm_leg_coordination(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_arm_joints: list[str],
    right_arm_joints: list[str],
    left_leg_joints: list[str],
    right_leg_joints: list[str],
) -> torch.Tensor:
    """Penalize when arms don't swing opposite to legs (military march style).

    Left arm should swing forward when right leg steps forward, and vice versa.
    Measured via shoulder_pitch and hip_pitch joints - they should have
    OPPOSITE signs for contralateral (cross-body) limbs.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "_arm_leg_coord_cache") or env._arm_leg_coord_cache is None:
        env._arm_leg_coord_cache = {
            "left_arm": [asset.find_joints(j) for j in left_arm_joints],
            "right_arm": [asset.find_joints(j) for j in right_arm_joints],
            "left_leg": [asset.find_joints(j) for j in left_leg_joints],
            "right_leg": [asset.find_joints(j) for j in right_leg_joints],
        }

    cache = env._arm_leg_coord_cache
    joint_pos = asset.data.joint_pos
    default_pos = asset.data.default_joint_pos

    reward = torch.zeros(env.num_envs, device=env.device)
    for la, rl in zip(cache["left_arm"], cache["right_leg"]):
        left_arm_delta = joint_pos[:, la[0]] - default_pos[:, la[0]]
        right_leg_delta = joint_pos[:, rl[0]] - default_pos[:, rl[0]]
        reward += torch.sum(torch.clamp(left_arm_delta * right_leg_delta, min=0), dim=-1)

    for ra, ll in zip(cache["right_arm"], cache["left_leg"]):
        right_arm_delta = joint_pos[:, ra[0]] - default_pos[:, ra[0]]
        left_leg_delta = joint_pos[:, ll[0]] - default_pos[:, ll[0]]
        reward += torch.sum(torch.clamp(right_arm_delta * left_leg_delta, min=0), dim=-1)

    return reward


def lateral_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral (side-to-side) body velocity.
    Military march should be straight with minimal sway."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_b = quat_apply_inverse(asset.data.root_quat_w, asset.data.root_lin_vel_w)
    lateral_vel = lin_vel_b[:, 1]
    return torch.square(lateral_vel)
