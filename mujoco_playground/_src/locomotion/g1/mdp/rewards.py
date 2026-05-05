# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
# ==============================================================================
"""Reward functions for G1 joystick environments."""

import jax
import jax.numpy as jp

from mujoco_playground._src import gait as gait_lib
from mujoco_playground._src import mjx_env


# =========================================================================
# Tracking rewards
# =========================================================================

def tracking_lin_vel(data, info, cfg, **kwargs):
    local_vel = kwargs["local_linvel"]
    cmd = info["command"][:2]
    err = jp.sum(jp.square(cmd - local_vel[:2]))
    return jp.exp(-err / cfg.rewards.tracking_sigma)


def tracking_ang_vel(data, info, cfg, **kwargs):
    gyro = kwargs["gyro"]
    cmd = info["command"][2]
    err = jp.square(cmd - gyro[2])
    return jp.exp(-err / cfg.rewards.tracking_sigma)


# =========================================================================
# Base-related rewards
# =========================================================================

def lin_vel_z(data, info, cfg, **kwargs):
    global_linvel_torso = kwargs["global_linvel_torso"]
    global_linvel_pelvis = kwargs["global_linvel_pelvis"]
    return jp.square(global_linvel_torso[2]) + jp.square(global_linvel_pelvis[2])


def ang_vel_xy(data, info, cfg, **kwargs):
    global_angvel_torso = kwargs["global_angvel_torso"]
    return jp.sum(jp.square(global_angvel_torso[:2]))


def orientation(data, info, cfg, **kwargs):
    torso_zaxis = kwargs["gravity_torso"]
    return jp.sum(jp.square(torso_zaxis - jp.array([0.073, 0.0, 1.0])))


def base_height(data, info, cfg, **kwargs):
    return jp.square(data.qpos[2] - cfg.rewards.base_height_target)


# =========================================================================
# Energy related rewards
# =========================================================================

def torques(data, info, cfg, **kwargs):
    return jp.sum(jp.abs(data.actuator_force))


def energy(data, info, cfg, **kwargs):
    return jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force))


def action_rate(data, info, cfg, **kwargs):
    current_action = kwargs["current_action"]
    last_act = info["last_action"]
    return jp.sum(jp.square(current_action - last_act))


def dof_acc(data, info, cfg, **kwargs):
    return jp.sum(jp.square(data.qacc[6:]))


# =========================================================================
# Feet related rewards
# =========================================================================

def feet_slip(data, info, cfg, **kwargs):
    body_vel = kwargs["global_linvel_pelvis"][:2]
    contact = kwargs["contact"]
    reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)
    return reward


def feet_clearance(data, info, cfg, **kwargs):
    foot_linvel_sensor_adr = kwargs.get("foot_linvel_sensor_adr")
    feet_site_id = kwargs.get("feet_site_id")
    feet_vel = data.sensordata[foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - cfg.rewards.max_foot_height)
    return jp.sum(delta * vel_norm)


def feet_height(data, info, cfg, **kwargs):
    swing_peak = info["swing_peak"]
    first_contact = kwargs["first_contact"]
    error = swing_peak / cfg.rewards.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact)


def feet_air_time(data, info, cfg, **kwargs):
    air_time = info["feet_air_time"]
    first_contact = kwargs["first_contact"]
    threshold_min = 0.2
    threshold_max = 0.5
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    return jp.sum(air_time)


def feet_phase(data, info, cfg, **kwargs):
    feet_site_id = kwargs["feet_site_id"]
    foot_pos = data.site_xpos[feet_site_id]
    foot_z = foot_pos[..., -1]
    phase = info["phase"]
    rz = gait_lib.get_rz(phase, swing_height=cfg.rewards.max_foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    reward = jp.exp(-error / 0.01)
    # Mask: only reward when moving.
    body_linvel = kwargs["global_linvel_pelvis"][:2]
    body_angvel = kwargs["global_angvel_pelvis"][2]
    linvel_mask = jp.logical_or(
        jp.linalg.norm(body_linvel) > 0.1,
        jp.abs(body_angvel) > 0.1,
    )
    mask = jp.logical_or(linvel_mask, jp.linalg.norm(info["command"]) > 0.01)
    return reward * mask


def feet_traj(data, info, cfg, **kwargs):
    feet_inds = kwargs["feet_inds"]
    foot_linvel_sensor_adr = kwargs.get("foot_linvel_sensor_adr")

    curr_feet = data.geom_xpos[feet_inds]
    ref_pos = info["foot_ref_pos"]
    swing_mask = info["foot_swing"][:, None]
    stance_mask = 1.0 - swing_mask

    # Swing leg error: track cycloid trajectory.
    swing_err = jp.sum(jp.square((curr_feet - ref_pos) * swing_mask))

    # Stance leg error: hold position at touchdown anchor.
    stance_ref = jp.concatenate([info["xy0"], info["z0"][:, None]], axis=1)
    stance_err = jp.sum(jp.square((curr_feet - stance_ref) * stance_mask))

    pos_err = swing_err + stance_err

    vel_err = 0.0
    if foot_linvel_sensor_adr is not None:
        feet_vel = data.sensordata[foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        ref_v_xy = info["foot_ref_v_xy"]
        vel_err = jp.sum(jp.square((vel_xy - ref_v_xy) * swing_mask[:, :2]))

    return pos_err + cfg.env.foot_traj_vel_weight * vel_err


def gait_phase_tracking(data, info, cfg, **kwargs):
    contact = kwargs["contact"]
    expected_stance = 1.0 - info["foot_swing"]
    err = jp.sum(jp.square(contact - expected_stance))
    return jp.exp(-err / cfg.rewards.gait_phase_tracking_sigma)


# =========================================================================
# Other rewards
# =========================================================================

def alive(data, info, cfg, **kwargs):
    return jp.array(1.0)


def termination(data, info, cfg, **kwargs):
    del data, info, cfg
    if "soft_done" in kwargs:
        return kwargs["soft_done"]
    return kwargs["done"]


def stand_still(data, info, cfg, **kwargs):
    cmd_norm = jp.linalg.norm(info["command"])
    cost = jp.sum(jp.abs(data.qpos[7:] - kwargs["default_pose"]))
    return cost * (cmd_norm < 0.01)


def _segment_distance(p0, p1, q0, q1):
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0

    a = jp.sum(u * u, axis=-1)
    b = jp.sum(u * v, axis=-1)
    c = jp.sum(v * v, axis=-1)
    d = jp.sum(u * w, axis=-1)
    e = jp.sum(v * w, axis=-1)
    eps = 1e-8

    denom = a * c - b * b
    s = jp.clip((b * e - c * d) / (denom + eps), 0.0, 1.0)
    t = jp.clip((b * s + e) / (c + eps), 0.0, 1.0)
    s = jp.clip((b * t - d) / (a + eps), 0.0, 1.0)

    closest_p = p0 + s[:, None] * u
    closest_q = q0 + t[:, None] * v
    return jp.sqrt(jp.sum(jp.square(closest_p - closest_q), axis=-1) + eps)


def collision(data, info, cfg, **kwargs):
    del info
    geom_pairs = kwargs["hand_thigh_geom_pairs"]
    radius = kwargs["hand_thigh_capsule_radius"]
    half_length = kwargs["hand_thigh_capsule_half_length"]

    centers = data.geom_xpos[geom_pairs]
    axes = data.geom_xmat[geom_pairs, :, 2]
    p0 = centers[:, 0] - axes[:, 0] * half_length[:, 0:1]
    p1 = centers[:, 0] + axes[:, 0] * half_length[:, 0:1]
    q0 = centers[:, 1] - axes[:, 1] * half_length[:, 1:2]
    q1 = centers[:, 1] + axes[:, 1] * half_length[:, 1:2]

    signed_dist = _segment_distance(p0, p1, q0, q1) - jp.sum(radius, axis=-1)
    x = (cfg.rewards.collision_margin - signed_dist) / cfg.rewards.collision_temp
    penalty = jax.nn.softplus(x) * cfg.rewards.collision_temp
    return jp.sum(penalty)


def contact_force(data, info, cfg, **kwargs):
    left_foot_force = mjx_env.get_sensor_data(
        kwargs["mj_model"], data, "left_foot_force"
    )
    right_foot_force = mjx_env.get_sensor_data(
        kwargs["mj_model"], data, "right_foot_force"
    )
    cost = jp.clip(jp.abs(left_foot_force[2]) - cfg.rewards.max_contact_force, min=0.0)
    cost += jp.clip(jp.abs(right_foot_force[2]) - cfg.rewards.max_contact_force, min=0.0)
    return cost


# =========================================================================
# Pose related rewards
# =========================================================================

def joint_deviation_hip(data, info, cfg, **kwargs):
    hip_indices = kwargs["hip_indices"]
    default_pose = kwargs["default_pose"]
    qpos = data.qpos[7:]
    error = qpos[hip_indices] - default_pose[hip_indices]
    cmd = info["command"]
    weight = jp.where(
        jp.abs(cmd[1]) > 0.1,
        jp.array([0.0, 1.0, 0.0, 1.0]),
        jp.array([1.0, 1.0, 1.0, 1.0]),
    )
    return jp.sum(jp.abs(error) * weight)


def joint_deviation_knee(data, info, cfg, **kwargs):
    knee_indices = kwargs["knee_indices"]
    default_pose = kwargs["default_pose"]
    qpos = data.qpos[7:]
    error = qpos[knee_indices] - default_pose[knee_indices]
    return jp.sum(jp.abs(error))


def dof_pos_limits(data, info, cfg, **kwargs):
    qpos = data.qpos[7:]
    soft_lowers = kwargs["soft_lowers"]
    soft_uppers = kwargs["soft_uppers"]
    out_of_limits = -jp.clip(qpos - soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)


def pose(data, info, cfg, **kwargs):
    qpos = data.qpos[7:]
    default_pose = kwargs["default_pose"]
    weights = kwargs["weights"]
    return jp.sum(jp.square(qpos - default_pose) * weights)
