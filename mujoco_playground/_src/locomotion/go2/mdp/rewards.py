import jax
import jax.numpy as jp
from mujoco_playground._src.locomotion.go2.Util.TrotUtil import quaternion_to_rotation_6d, rotate_inv

# =========================================================================
# Trot Rewards
# =========================================================================

def reference_tracking(data, info, cfg, **kwargs):
    """Reward for tracking the reference trajectory (position, rotation, velocity)."""
    ref_data = kwargs['ref_data']
    f = lambda a, b: ((a - b) ** 2).sum(-1).mean()

    mse_pos = f(data.xpos[1:], ref_data.xpos[1:])
    mse_rot = f(quaternion_to_rotation_6d(data.xquat[1:]), quaternion_to_rotation_6d(ref_data.xquat[1:]))
    mse_vel = f(data.cvel[1:, 3:], ref_data.cvel[1:, 3:])
    mse_ang = f(data.cvel[1:, :3], ref_data.cvel[1:, :3])

    return mse_pos + 0.1 * mse_rot + 0.01 * mse_vel + 0.001 * mse_ang


def min_reference_tracking(data, info, cfg, **kwargs):
    """Simplified reference tracking focusing on joint positions and velocities."""
    ref_qpos = kwargs['ref_qpos']
    ref_qvel = kwargs['ref_qvel']

    pos = jp.concatenate([data.qpos[:3], data.qpos[7:]])
    pos_targ = jp.concatenate([ref_qpos[:3], ref_qpos[7:]])
    pos_err = jp.linalg.norm(pos_targ - pos)
    vel_err = jp.linalg.norm(data.qvel - ref_qvel)

    return pos_err + vel_err


def feet_height(data, info, cfg, **kwargs):
    """Penalty for feet height deviation from reference."""
    ref_data = kwargs['ref_data']
    feet_inds = kwargs['feet_inds']

    feet_z = data.geom_xpos[feet_inds][:, 2]
    feet_z_ref = ref_data.geom_xpos[feet_inds][:, 2]

    return jp.sum(jp.abs(feet_z - feet_z_ref))


def base_tracking(data, info, cfg, **kwargs):
    """Reward for base position, rotation, and velocity tracking."""
    ref_data = kwargs['ref_data']

    pos_err = jp.linalg.norm(data.xpos[1] - ref_data.xpos[1])
    q = data.xquat[1]
    q_ref = ref_data.xquat[1]

    dot = jp.abs(jp.dot(q, q_ref))
    dot = jp.clip(dot, -1.0, 1.0)
    rot_err = jp.arccos(2 * dot**2 - 1)
    vel_err = jp.linalg.norm(data.cvel[1] - ref_data.cvel[1])

    return pos_err + 0.5 * rot_err + 0.1 * vel_err


# =========================================================================
# Joystick Rewards
# =========================================================================

def tracking_lin_vel(data, info, cfg, **kwargs):
    """Reward for tracking linear velocity commands."""
    q = data.xquat[1]
    v_local = rotate_inv(data.cvel[1, 3:], q)
    cmd = info['command'][:2]
    err = jp.sum(jp.square(cmd - v_local[:2]))
    return jp.exp(-err / cfg.rewards.tracking_sigma)


def tracking_ang_vel(data, info, cfg, **kwargs):
    """Reward for tracking angular velocity commands."""
    q = data.xquat[1]
    w_local = rotate_inv(data.cvel[1, :3], q)
    cmd = info['command'][2]
    err = jp.square(cmd - w_local[2])
    return jp.exp(-err / cfg.rewards.tracking_sigma)


def base_height_tracking(data, info, cfg, **kwargs):
    """Reward for maintaining nominal base height."""
    del info
    nominal_base_height = kwargs['nominal_base_height']
    err = jp.square(data.qpos[2] - nominal_base_height)
    return jp.exp(-err / cfg.rewards.base_height_sigma)


def joint_pose_tracking(data, info, cfg, **kwargs):
    """Reward for tracking kinematic reference joint positions.

    Uses gait_step (resets to 0 when stationary) for phase indexing.
    move_mask ensures this reward is only active during movement.
    """
    move_mask = kwargs.get('move_mask', 1.0)
    kinematic_ref_qpos = kwargs['kinematic_ref_qpos']
    l_cycle = kwargs['l_cycle']

    step_idx = jp.array(info['gait_step'] % l_cycle, int)
    ref_qpos = kinematic_ref_qpos[step_idx][7:]
    qpos = data.qpos[7:19]
    weight = jp.array([1.0, 1.0, 0.1] * 4)
    err = jp.sum(jp.square(qpos - ref_qpos) * weight)
    return jp.exp(-err / cfg.rewards.joint_pose_tracking_sigma) * move_mask


def joint_vel_tracking(data, info, cfg, **kwargs):
    """Reward for tracking kinematic reference joint velocities.

    Uses gait_step (resets to 0 when stationary) for phase indexing.
    move_mask ensures this reward is only active during movement.
    """
    move_mask = kwargs.get('move_mask', 1.0)
    kinematic_ref_qvel = kwargs['kinematic_ref_qvel']
    l_cycle = kwargs['l_cycle']

    step_idx = jp.array(info['gait_step'] % l_cycle, int)
    ref_qvel = kinematic_ref_qvel[step_idx][6:]
    qvel = data.qvel[6:]
    err = jp.sum(jp.square(qvel - ref_qvel))
    return jp.exp(-err / cfg.rewards.joint_vel_tracking_sigma) * move_mask


def gait_phase_tracking(data, info, cfg, **kwargs):
    """Reward for matching contact pattern to expected gait phase."""
    del data
    contact = kwargs['contact']
    expected_stance = 1.0 - info['foot_swing']
    err = jp.sum(jp.square(contact - expected_stance))
    return jp.exp(-err / cfg.rewards.gait_phase_tracking_sigma)


def feet_traj(data, info, cfg, **kwargs):
    """Cost for foot trajectory tracking based on Raibert heuristic.

    - Swing legs: track cycloid reference trajectory
    - Stance legs: hold position at touchdown anchor (xy0, z0)

    Self-consistent: when stationary, all feet are stance legs holding position.
    """
    feet_inds = kwargs['feet_inds']
    foot_linvel_sensor_adr = kwargs.get('foot_linvel_sensor_adr', None)

    curr_feet = data.geom_xpos[feet_inds]
    ref_pos = info['foot_ref_pos']
    swing_mask = info['foot_swing'][:, None]
    stance_mask = 1.0 - swing_mask  # Stance legs = 1, Swing legs = 0

    # Swing leg error: track cycloid trajectory
    swing_err = jp.sum(jp.square((curr_feet - ref_pos) * swing_mask))

    # Stance leg error: hold position at touchdown anchor
    stance_ref = jp.concatenate([info['xy0'], info['z0'][:, None]], axis=1)
    stance_err = jp.sum(jp.square((curr_feet - stance_ref) * stance_mask))

    # Total position error
    pos_err = swing_err + stance_err

    # Velocity error (swing legs only)
    vel_err = 0.0
    if foot_linvel_sensor_adr is not None:
        feet_vel = data.sensordata[foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        ref_v_xy = info['foot_ref_v_xy']
        vel_err = jp.sum(jp.square((vel_xy - ref_v_xy) * swing_mask[:, :2]))

    return pos_err + cfg.env.foot_traj_vel_weight * vel_err


def lin_vel_z(data, info, cfg, **kwargs):
    """Penalty for base vertical velocity."""
    del info, cfg, kwargs
    return jp.square(data.cvel[1, 5])


def ang_vel_xy(data, info, cfg, **kwargs):
    """Penalty for base roll/pitch angular velocity."""
    del info, cfg, kwargs
    return jp.sum(jp.square(data.cvel[1, :2]))


def orientation(data, info, cfg, **kwargs):
    """Penalty for base tilt away from upright."""
    del info, cfg
    get_upvector = kwargs['get_upvector']
    up_vec = get_upvector(data)
    return jp.sum(jp.square(up_vec[:2]))


def torques(data, info, cfg, **kwargs):
    """Penalty for large actuator torques."""
    del info, cfg, kwargs
    tau = data.actuator_force
    return jp.sqrt(jp.sum(jp.square(tau))) + jp.sum(jp.abs(tau))


def energy(data, info, cfg, **kwargs):
    """Penalty for joint power usage."""
    del info, cfg, kwargs
    return jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force))


def action_rate(data, info, cfg, **kwargs):
    """Penalty for rapid action changes."""
    del data, cfg
    current_action = kwargs['current_action']
    return jp.sum(jp.square(current_action - info['last_action']))


def feet_slip(data, info, cfg, **kwargs):
    """Penalty for foot slipping while in contact.

    Self-consistent: penalizes sliding when in contact, regardless of command.
    Stationary feet should not slip.
    """
    del info, cfg
    foot_linvel_sensor_adr = kwargs.get('foot_linvel_sensor_adr', None)

    if foot_linvel_sensor_adr is None:
        return 0.0

    feet_vel = data.sensordata[foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    effective_contact = jax.nn.relu(kwargs['contact'] - 0.5) * 2.0
    return jp.sum(vel_xy_norm_sq * effective_contact)


def feet_air_time(data, info, cfg, **kwargs):
    """Reward for increasing feet air time during movement."""
    move_mask = kwargs.get('move_mask', 1.0)
    first_contact = kwargs['first_contact']
    air_time = info['feet_air_time']

    rew = jp.sum((air_time - 0.1) * first_contact)
    return rew * move_mask


def dof_pos_limits(data, info, cfg, **kwargs):
    """Penalty for violating soft joint limits."""
    del info, cfg
    soft_lowers = kwargs['soft_lowers']
    soft_uppers = kwargs['soft_uppers']
    qpos = data.qpos[7:]
    out_of_limits = -jp.clip(qpos - soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)


def stand_still(data, info, cfg, **kwargs):
    """Penalty for moving joints when the robot should be standing still."""
    still_mask = kwargs.get('still_mask', 1.0)
    default_ap_pose = kwargs['default_ap_pose']
    qpos = data.qpos[7:]

    return jp.sum(jp.abs(qpos - default_ap_pose)) * still_mask


def pose(data, info, cfg, **kwargs):
    """Reward for staying close to the default pose."""
    del info, cfg
    default_ap_pose = kwargs['default_ap_pose']
    qpos = data.qpos[7:]
    weight = jp.array([1.0, 1.0, 0.1] * 4)
    err = jp.sum(jp.square(qpos - default_ap_pose) * weight)
    return jp.exp(-err)


def feet_clearance(data, info, cfg, **kwargs):
    """Penalty for swing feet deviating from the configured target height."""
    del info
    feet_inds = kwargs['feet_inds']
    foot_linvel_sensor_adr = kwargs.get('foot_linvel_sensor_adr', None)

    if foot_linvel_sensor_adr is None:
        return 0.0

    feet_vel = data.sensordata[foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_z = data.geom_xpos[feet_inds][:, 2]
    target_height = cfg.rewards.max_foot_height
    delta = jp.abs(foot_z - target_height)
    return jp.sum(delta * vel_norm)


def feet_height(data, info, cfg, **kwargs):
    """Penalty for swing peak height differing from the configured target."""
    del data
    move_mask = kwargs.get('move_mask', 1.0)
    first_contact = kwargs['first_contact']
    swing_peak = info['swing_peak']
    target_height = cfg.rewards.max_foot_height

    error = swing_peak / target_height - 1.0
    return jp.sum(jp.square(error) * first_contact) * move_mask


def termination(data, info, cfg, **kwargs):
    """Soft termination penalty."""
    del data, info, cfg
    return kwargs['soft_done']


# =========================================================================
# Trot Gait Rewards
# =========================================================================

def contact_count_penalty(data, info, cfg, **kwargs):
    """Penalize deviation from 2 feet in contact (ideal trot pattern).

    In a proper trot gait, diagonal leg pairs move together, resulting in
    exactly 2 feet in contact at any time (FL+RR or FR+RL).

    Only applied when velocity command is non-zero (during movement).
    """
    contact = kwargs['contact']
    n_contact = jp.sum(contact)
    move_mask = kwargs.get('move_mask', 1.0)
    # 完美逻辑，限制接触脚数为 2，仅在速度命令非 0 时应用
    return jp.square(n_contact - 2.0) * move_mask


def diagonal_sync_penalty(data, info, cfg, **kwargs):
    """Penalize asymmetry between diagonal leg pairs on Hip and Calf joints.

    In trot gait, diagonal legs should move in phase:
    - FL (front-left) should mirror RR (rear-right)
    - FR (front-right) should mirror RL (rear-left)

    This reward encourages this diagonal symmetry pattern by penalizing
    differences in Hip and Calf joint positions.

    Only applied when velocity command is non-zero (during movement).
    """
    # action 按 [FL, FR, RL, RR] 排序，每条腿 3 个关节 (abduction, hip, calf)
    action = kwargs['current_action']
    move_mask = kwargs.get('move_mask', 1.0)

    # 只取 Hip 和 Calf 关节 (每条腿的索引 1 和 2)
    fl_hip_calf = action[1:3]    # FL 腿的 hip, calf
    fr_hip_calf = action[4:6]    # FR 腿的 hip, calf
    rl_hip_calf = action[7:9]    # RL 腿的 hip, calf
    rr_hip_calf = action[10:12]  # RR 腿的 hip, calf

    # 计算对角线的平方差
    diag_fl_rr = jp.sum(jp.square(fl_hip_calf - rr_hip_calf))
    diag_fr_rl = jp.sum(jp.square(fr_hip_calf - rl_hip_calf))

    return (diag_fl_rr + diag_fr_rl) * move_mask
