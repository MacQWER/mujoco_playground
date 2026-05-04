# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
# ==============================================================================
"""Raibert heuristic, cycloid foot trajectory, and visualization for G1 bipedal."""

import jax
import jax.numpy as jp
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2.Util.TrotUtil import (
    cos_wave,
    dcos_wave,
    rotate,
    rotate_inv,
)
from mujoco_playground._src.locomotion.g1 import g1_constants as consts


# ---------------------------------------------------------------------------
# 1. Kinematic reference generator (bipedal walking)
# ---------------------------------------------------------------------------

def make_kinematic_ref(sinusoid, step_k, scale=0.3, dt=1/50):
    """Build a bipedal walking kinematic reference for G1 (29-DOF).

    Half-cycle 1: left leg swings, right leg stances.
    Half-cycle 2: right leg swings, left leg stances.
    Waist and arms stay at default pose (0 reference).

    Returns (l_cycle, 29) array of joint offsets from default pose.
    """
    _steps = jp.arange(step_k)
    step_period = step_k * dt
    t = _steps * dt

    wave = sinusoid(t, step_period, scale)

    # Leg block for one leg (6 joints): [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
    # Only modulate hip_pitch, knee, ankle_pitch; others stay at 0.
    swing_leg_block = jp.concatenate([
        wave.reshape(step_k, 1),           # hip_pitch
        jp.zeros((step_k, 1)),            # hip_roll
        jp.zeros((step_k, 1)),            # hip_yaw
        wave.reshape(step_k, 1),           # knee
        jp.zeros((step_k, 1)),            # ankle_pitch
        jp.zeros((step_k, 1)),            # ankle_roll
    ], axis=1)  # (step_k, 6)

    stance_leg_block = jp.zeros((step_k, 6))  # (step_k, 6)
    zero_rest = jp.zeros((step_k, 17))  # waist(3) + arms(14)

    # Block 1: left swings, right stances
    block1 = jp.concatenate([swing_leg_block, stance_leg_block, zero_rest], axis=1)

    # Block 2: left stances, right swings
    block2 = jp.concatenate([stance_leg_block, swing_leg_block, zero_rest], axis=1)

    step_cycle = jp.concatenate([block1, block2], axis=0)  # (2*step_k, 29)
    return step_cycle


# ---------------------------------------------------------------------------
# 2. Swing mask (2 feet: left, right)
# ---------------------------------------------------------------------------

def get_swing_mask(env, info):
    gait_step = info["gait_step"]
    is_stationary = info["is_stationary"]

    stationary_swing = jp.array([0.0, 0.0])  # both feet planted

    gait_step = jp.asarray(gait_step, dtype=jp.int32)
    chunk_idx = gait_step // env.step_k
    even_chunk = chunk_idx % 2 == 0
    # even_chunk: left swings [1, 0], odd_chunk: right swings [0, 1]
    moving_swing = jp.where(even_chunk, jp.array([1.0, 0.0]), jp.array([0.0, 1.0]))

    return jp.where(is_stationary, stationary_swing, moving_swing)


# ---------------------------------------------------------------------------
# 3. Raibert heuristic target updater (bipedal)
# ---------------------------------------------------------------------------

def update_raibert_target(env, data, info):
    s = info["gait_step"]
    step_k = env.step_k
    new_step = s % step_k == 0
    even_step = (s // step_k) % 2 == 0

    v_cmd_local = info["command"][:2]
    w_cmd_local = info["command"][2]

    quat = data.xquat[1]

    # Rotational velocity contribution: v_rot = w × r
    v_ff_rot_x = -w_cmd_local * env.leg_offsets_y
    v_ff_rot_y = w_cmd_local * env.leg_offsets_x

    v_leg_local = jp.stack([
        v_cmd_local[0] + v_ff_rot_x,
        v_cmd_local[1] + v_ff_rot_y,
    ], axis=1)  # (2, 2)

    v_leg_local_3d = jp.concatenate([v_leg_local, jp.zeros((2, 1))], axis=1)
    v_leg_global = jax.vmap(rotate, in_axes=(0, None))(v_leg_local_3d, quat)[:, :2]

    hip_pos = data.xpos[env.hip_inds][:, :2]
    t_stance = env.gait_period / 2.0
    t_swing = env.gait_period / 2.0

    raibert_offset = (t_swing + 0.5 * t_stance) * v_leg_global

    foot_offset_local_3d = jp.concatenate(
        [env.foot_offsets_xy, jp.zeros((2, 1))], axis=1
    )
    foot_offset_global = jax.vmap(rotate, in_axes=(0, None))(
        foot_offset_local_3d, quat
    )[:, :2]

    raibert_xy = hip_pos + foot_offset_global + raibert_offset

    cur_tars = info["xy*"]
    # new_step & even_step: left foot update (index 0)
    # new_step & ~even_step: right foot update (index 1)
    xy_tars = jp.where(new_step & even_step, cur_tars.at[0].set(raibert_xy[0]), cur_tars)
    xy_tars = jp.where(new_step & (~even_step), xy_tars.at[1].set(raibert_xy[1]), xy_tars)
    info["xy*"] = xy_tars

    feet_pos = data.geom_xpos[env.feet_inds][:, :2]
    info["xy0"] = jp.where(new_step, feet_pos, info["xy0"])
    info["k0"] = jp.where(new_step, s, info["k0"])

    feet_z = data.site_xpos[env._feet_site_id][:, 2]
    info["z0"] = jp.where(new_step, feet_z, info["z0"])


# ---------------------------------------------------------------------------
# 4. Cycloid foot trajectory reference
# ---------------------------------------------------------------------------

def update_foot_cycloid_ref(env, info):
    step = info["gait_step"]
    swing_period = env.gait_period / 2.0
    dt_step = (step - info["k0"]) * env.dt
    phi = jp.clip(dt_step / swing_period, 0.0, 1.0)

    s = phi - jp.sin(2.0 * jp.pi * phi) / (2.0 * jp.pi)
    ds_dt = (1.0 - jp.cos(2.0 * jp.pi * phi)) / swing_period

    xy0 = info["xy0"]
    xys = info["xy*"]
    delta_xy = xys - xy0

    xy_ref = xy0 + delta_xy * s
    v_xy_ref = delta_xy * ds_dt

    swing_mask = get_swing_mask(env, info)
    swing_mask_col = swing_mask[:, None]

    xy_ref = xy0 * (1.0 - swing_mask_col) + xy_ref * swing_mask_col
    v_xy_ref = v_xy_ref * swing_mask_col

    step_len = jp.linalg.norm(delta_xy, axis=1)
    r = step_len / (2.0 * jp.pi)
    r_min = 0.5 * env._step_height_min
    r_max = 0.5 * env._step_height_max
    r = jp.clip(r, r_min, r_max)

    z0 = info["z0"]
    z_ref = z0 + r * (1.0 - jp.cos(2.0 * jp.pi * phi))
    z_ref = z0 * (1.0 - swing_mask) + z_ref * swing_mask

    ref_pos = jp.concatenate([xy_ref, z_ref[:, None]], axis=1)

    info["foot_phase"] = jax.lax.stop_gradient(phi)
    info["foot_swing"] = jax.lax.stop_gradient(swing_mask)
    info["foot_ref_xy"] = jax.lax.stop_gradient(xy_ref)
    info["foot_ref_z"] = jax.lax.stop_gradient(z_ref)
    info["foot_ref_pos"] = jax.lax.stop_gradient(ref_pos)
    info["foot_ref_v_xy"] = jax.lax.stop_gradient(v_xy_ref)


# =============================================================================
# Native MuJoCo helpers (numpy) – for fast diagnostics without JAX/MJX
# =============================================================================

def _q_to_mat_np(q):
    """Quaternion to rotation matrix (numpy)."""
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two_s = 2.0 / (q * q).sum(-1)
    o = np.stack([
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j),
    ], axis=-1)
    return o.reshape(q.shape[:-1] + (3, 3))


def _rotate_np(v, q):
    """Rotate vector v by quaternion q."""
    return _q_to_mat_np(q) @ v


def _get_swing_mask_np(gait_step, is_stationary, step_k):
    """Swing mask for G1 bipedal (numpy)."""
    if is_stationary:
        return np.array([0.0, 0.0])
    chunk_idx = gait_step // step_k
    if chunk_idx % 2 == 0:
        return np.array([1.0, 0.0])  # left swings
    else:
        return np.array([0.0, 1.0])  # right swings


def _update_raibert_target_np(env, data, info):
    """Numpy version of update_raibert_target for G1 bipedal diagnostics."""
    s = int(info["gait_step"])
    new_step = s % env.step_k == 0
    even_step = (s // env.step_k) % 2 == 0

    v_cmd_local = np.asarray(info["command"][:2])
    w_cmd_local = float(info["command"][2])

    v_ff_rot_x = -w_cmd_local * np.asarray(env.leg_offsets_y)
    v_ff_rot_y = w_cmd_local * np.asarray(env.leg_offsets_x)

    v_leg_local = np.stack([
        v_cmd_local[0] + v_ff_rot_x,
        v_cmd_local[1] + v_ff_rot_y,
    ], axis=1)

    v_leg_local_3d = np.concatenate([v_leg_local, np.zeros((2, 1))], axis=1)
    quat = data.xquat[1]
    v_leg_global = np.stack([
        _rotate_np(v_leg_local_3d[0], quat)[:2],
        _rotate_np(v_leg_local_3d[1], quat)[:2],
    ])

    t_stance = env.gait_period / 2.0
    t_swing = env.gait_period / 2.0
    raibert_offset = (t_swing + 0.5 * t_stance) * v_leg_global

    foot_offset_local_3d = np.concatenate(
        [np.asarray(env.foot_offsets_xy), np.zeros((2, 1))], axis=1)
    foot_offset_global = np.stack([
        _rotate_np(foot_offset_local_3d[0], quat)[:2],
        _rotate_np(foot_offset_local_3d[1], quat)[:2],
    ])

    hip_pos = data.xpos[env.hip_inds][:, :2]
    raibert_xy = hip_pos + foot_offset_global + raibert_offset

    if new_step and even_step:
        info["xy*"][0] = raibert_xy[0]
    if new_step and not even_step:
        info["xy*"][1] = raibert_xy[1]

    if new_step:
        info["xy0"] = data.geom_xpos[env.feet_inds][:, :2].copy()
        info["k0"] = s
        info["z0"] = data.site_xpos[env._feet_site_id][:, 2].copy()


def _update_foot_cycloid_ref_np(env, info):
    """Numpy version of update_foot_cycloid_ref for G1 bipedal diagnostics."""
    step = int(info["gait_step"])
    swing_period = env.gait_period / 2.0
    dt_step = (step - info["k0"]) * env.dt
    phi = np.clip(dt_step / swing_period, 0.0, 1.0)

    cyc_s = phi - np.sin(2.0 * np.pi * phi) / (2.0 * np.pi)
    ds_dt = (1.0 - np.cos(2.0 * np.pi * phi)) / swing_period

    delta_xy = info["xy*"] - info["xy0"]
    xy_ref = info["xy0"] + delta_xy * cyc_s
    v_xy_ref = delta_xy * ds_dt

    swing_mask = _get_swing_mask_np(step, info["is_stationary"], env.step_k)
    swing_mask_col = swing_mask[:, None]

    xy_ref = info["xy0"] * (1.0 - swing_mask_col) + xy_ref * swing_mask_col
    v_xy_ref = v_xy_ref * swing_mask_col

    step_len = np.linalg.norm(delta_xy, axis=1)
    r = step_len / (2.0 * np.pi)
    r = np.clip(r, 0.5 * env._step_height_min, 0.5 * env._step_height_max)

    z0 = info["z0"]
    z_ref = z0 + r * (1.0 - np.cos(2.0 * np.pi * phi))
    z_ref = z0 * (1.0 - swing_mask) + z_ref * swing_mask

    ref_pos = np.concatenate([xy_ref, z_ref[:, None]], axis=1)

    info["foot_phase"] = phi
    info["foot_swing"] = swing_mask
    info["foot_ref_xy"] = xy_ref
    info["foot_ref_z"] = z_ref
    info["foot_ref_pos"] = ref_pos
    info["foot_ref_v_xy"] = v_xy_ref


# ---------------------------------------------------------------------------
# 5. Phase alignment check (bipedal adapted from Go2)
# ---------------------------------------------------------------------------

def check_phase_alignment(env):
    print("Running bipedal phase alignment check (G1) [native MuJoCo]...")

    steps = np.arange(env.step_k * 4)
    foot_idx = 0
    half_cycle = env.step_k
    l_cycle = int(env.l_cycle)

    height_target_log = []
    height_actual_log = []
    raibert_change_log = []
    kino_thigh_log = []
    swing_mask_l_log = []
    swing_mask_r_log = []

    gait_rew_log = []
    gait_half_rew_log = []
    gait_err_log = []
    gait_half_err_log = []

    feet_traj_err_log = []
    feet_traj_half_err_log = []
    feet_traj_pos_err_log = []
    feet_traj_pos_half_err_log = []
    feet_traj_vel_err_log = []
    feet_traj_vel_half_err_log = []

    gait_step_log = []
    is_stationary_log = []

    cmd = np.array([1.0, 0.0, 0.0])

    def make_info(gait_step):
        return {
            "step": int(gait_step),
            "gait_step": int(gait_step),
            "is_stationary": False,
            "k0": 0,
            "command": cmd,
            "xy0": np.zeros((2, 2)),
            "xy*": np.zeros((2, 2)),
            "foot_phase": 0.0,
            "foot_swing": np.zeros(2),
            "foot_ref_xy": np.zeros((2, 2)),
            "foot_ref_z": np.zeros(2),
            "foot_ref_pos": np.zeros((2, 3)),
            "foot_ref_v_xy": np.zeros((2, 2)),
            "z0": np.zeros(2),
        }

    # Pre-build all unique kinematic reference frames (native MuJoCo)
    print("  Building kinematic reference frames (%d unique)..." % l_cycle)
    ref_data_cache = []
    for i in range(l_cycle):
        d = mujoco.MjData(env.mj_model)
        d.qpos[:] = env.kinematic_ref_qpos[i]
        d.qvel[:] = 0
        d.ctrl[:] = 0
        mujoco.mj_forward(env.mj_model, d)
        ref_data_cache.append(d)

    def prepare_info(gait_step, data):
        info = make_info(gait_step)
        _update_raibert_target_np(env, data, info)
        _update_foot_cycloid_ref_np(env, info)
        return info

    def foot_traj_terms(data, info):
        curr_feet = data.geom_xpos[env.feet_inds]
        ref_pos = info["foot_ref_pos"]
        swing_mask = info["foot_swing"][:, None]
        pos_err = np.sum(((curr_feet - ref_pos) * swing_mask) ** 2)
        vel_err = 0.0
        if env._foot_linvel_sensor_adr is not None:
            feet_vel = data.sensordata[env._foot_linvel_sensor_adr]
            vel_xy = feet_vel[..., :2]
            ref_v_xy = info["foot_ref_v_xy"]
            vel_err = np.sum(((vel_xy - ref_v_xy) * swing_mask[:, :2]) ** 2)
        total_err = pos_err + env._foot_traj_vel_weight * vel_err
        return float(total_err), float(pos_err), float(vel_err)

    def gait_terms(data, info):
        foot_z = data.site_xpos[env._feet_site_id][:, 2]
        contact = 1.0 / (1.0 + np.exp(-(0.025 - foot_z) * 100.0))
        expected_stance = 1.0 - info["foot_swing"]
        err = np.sum((contact - expected_stance) ** 2)
        rew = np.exp(-err / 0.25)
        return float(rew), float(err)

    last_left_target_x = 0.0

    print("  Running phase alignment check (%d steps)..." % len(steps))
    for s in steps:
        step_idx = int(s % l_cycle)

        ref_data = ref_data_cache[step_idx]
        info = prepare_info(s, ref_data)
        half_info = prepare_info(s + half_cycle, ref_data)

        gait_step_log.append(info["gait_step"])
        is_stationary_log.append(0.0)

        height_target_log.append(float(info["foot_ref_z"][foot_idx]))
        height_actual_log.append(float(ref_data.geom_xpos[env.feet_inds[foot_idx], 2]))

        current_left_target_x = float(info["xy*"][0, 0])
        if s > 0 and abs(current_left_target_x - last_left_target_x) > 1e-4:
            raibert_change_log.append(0.12)
        else:
            raibert_change_log.append(0.0)
        last_left_target_x = current_left_target_x

        kin_ref = env.kinematic_ref_qpos[step_idx]
        kino_thigh_log.append(float(kin_ref[7]))
        swing_mask_l_log.append(float(info["foot_swing"][0]))
        swing_mask_r_log.append(float(info["foot_swing"][1]))

        gait_rew, gait_err = gait_terms(ref_data, info)
        gait_half_rew, gait_half_err = gait_terms(ref_data, half_info)
        gait_rew_log.append(gait_rew)
        gait_half_rew_log.append(gait_half_rew)
        gait_err_log.append(gait_err)
        gait_half_err_log.append(gait_half_err)

        ft_err, ft_pos_err, ft_vel_err = foot_traj_terms(ref_data, info)
        ft_half_err, ft_pos_half_err, ft_vel_half_err = foot_traj_terms(ref_data, half_info)
        feet_traj_err_log.append(ft_err)
        feet_traj_half_err_log.append(ft_half_err)
        feet_traj_pos_err_log.append(ft_pos_err)
        feet_traj_pos_half_err_log.append(ft_pos_half_err)
        feet_traj_vel_err_log.append(ft_vel_err)
        feet_traj_vel_half_err_log.append(ft_vel_half_err)

    # Plotting (5 subplots).
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

    axes[0].step(steps, is_stationary_log, "r-", where="mid", linewidth=2, label="is_stationary")
    axes[0].plot(steps, gait_step_log, "g-", linewidth=1, label="gait_step")
    axes[0].set_ylabel("State")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, height_target_log, "g-", label="Foot Ref Z (L)", linewidth=2)
    axes[1].plot(steps, height_actual_log, color="orange", linestyle="-",
                 label="Ref-Kino Foot Z (L)", linewidth=1.5, alpha=0.85)
    axes[1].bar(steps, raibert_change_log, width=1.0, color="red", alpha=0.25,
                label="Raibert target update")
    axes[1].plot(steps, np.array(swing_mask_l_log) * 0.015, color="black",
                 linestyle="--", linewidth=1.5, label="Swing mask L (scaled)")
    axes[1].plot(steps, np.array(swing_mask_r_log) * 0.015, color="gray",
                 linestyle="--", linewidth=1.5, label="Swing mask R (scaled)")
    ax1_twin = axes[1].twinx()
    ax1_twin.plot(steps, kino_thigh_log, "b--", label="Ref-kino L hip_pitch",
                  linewidth=2, alpha=0.6)
    axes[1].set_ylabel("Foot Height (m)")
    ax1_twin.set_ylabel("Joint Angle (rad)")
    axes[1].set_title(f"G1 Bipedal Phase Alignment Dashboard (step_k={env.step_k})")

    axes[2].plot(steps, gait_rew_log, color="tab:brown", linewidth=2,
                 label="gait_phase reward (current)")
    axes[2].plot(steps, gait_half_rew_log, color="tab:brown", linestyle="--",
                 linewidth=2, label="gait_phase reward (half-cycle)")
    axes[2].set_ylabel("Reward")
    axes[2].set_ylim(-0.05, 1.05)

    axes[3].plot(steps, gait_err_log, color="tab:brown", linewidth=2,
                 label="gait_phase err (current)")
    axes[3].plot(steps, gait_half_err_log, color="tab:brown", linestyle="--",
                 linewidth=2, label="gait_phase err (half-cycle)")
    axes[3].set_ylabel("Error")

    axes[4].plot(steps, gait_rew_log, color="tab:brown", linewidth=2,
                 label="gait_phase reward (current)")
    axes[4].plot(steps, gait_half_rew_log, color="tab:brown", linestyle="--",
                 linewidth=2, label="gait_phase reward (half-cycle)")
    axes[4].plot(steps, feet_traj_err_log, color="tab:green", linewidth=2,
                 label="feet_traj total err (current)")
    axes[4].plot(steps, feet_traj_half_err_log, color="tab:green", linestyle="--",
                 linewidth=2, label="feet_traj total err (half-cycle)")
    axes[4].plot(steps, feet_traj_pos_err_log, color="tab:olive", linewidth=1.5,
                 alpha=0.85, label="feet_traj pos err (current)")
    axes[4].plot(steps, feet_traj_pos_half_err_log, color="tab:olive", linestyle="--",
                 linewidth=1.5, alpha=0.85, label="feet_traj pos err (half-cycle)")
    axes[4].plot(steps, feet_traj_vel_err_log, color="tab:cyan", linewidth=1.5,
                 alpha=0.85, label="feet_traj vel err (current)")
    axes[4].plot(steps, feet_traj_vel_half_err_log, color="tab:cyan", linestyle="--",
                 linewidth=1.5, alpha=0.85, label="feet_traj vel err (half-cycle)")
    axes[4].set_ylabel("Reward / Error")
    axes[4].set_xlabel("Step")

    for ax in axes:
        ax.axvspan(0, env.step_k, color="gray", alpha=0.08)
        ax.axvspan(env.step_k, env.step_k * 2, color="green", alpha=0.06)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", ncol=2)

    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines1b, labels1b = ax1_twin.get_legend_handles_labels()
    axes[1].legend(lines1 + lines1b, labels1 + labels1b, loc="upper left", ncol=2)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6. Gait step transition test (bipedal)
# ---------------------------------------------------------------------------

def check_gait_step_transition(env):
    print("Running G1 gait step transition test (moving -> stationary -> moving) [native MuJoCo]...")

    total_steps = 200
    stationary_start = 75
    stationary_end = 125

    # ── Pre-build kinematic reference frames (50 unique) ──
    l_cycle = int(env.l_cycle)
    print("  Building kinematic reference frames (%d unique)..." % l_cycle)
    ref_data_cache = []
    for i in range(l_cycle):
        d = mujoco.MjData(env.mj_model)
        d.qpos[:] = env.kinematic_ref_qpos[i]
        d.qvel[:] = 0
        d.ctrl[:] = 0
        mujoco.mj_forward(env.mj_model, d)
        ref_data_cache.append(d)

    # ── Run the 200-step transition test ──
    print("  Running 200-step transition test...")
    info = {
        "step": 0,
        "gait_step": 0,
        "is_stationary": False,
        "k0": 0,
        "command": np.array([1.0, 0.0, 0.0]),
        "xy0": np.zeros((2, 2)),
        "xy*": np.zeros((2, 2)),
        "foot_phase": 0.0,
        "foot_swing": np.zeros(2),
        "foot_ref_xy": np.zeros((2, 2)),
        "foot_ref_z": np.zeros(2),
        "foot_ref_pos": np.zeros((2, 3)),
        "foot_ref_v_xy": np.zeros((2, 2)),
        "z0": np.zeros(2),
    }

    step_log = []
    gait_step_log = []
    is_stationary_log = []
    swing_mask_l_log = []
    swing_mask_r_log = []
    raibert_update_log = []
    cmd_norm_log = []

    for s in range(total_steps):
        if stationary_start <= s < stationary_end:
            cmd = np.array([0.0, 0.0, 0.0])
        else:
            cmd = np.array([1.0, 0.0, 0.0])
        info["command"] = cmd

        cmd_norm = float(np.linalg.norm(cmd[:2]))
        w_cmd = float(abs(cmd[2]))
        is_stationary = (cmd_norm < 0.01) and (w_cmd < 0.01)
        info["is_stationary"] = is_stationary

        if is_stationary:
            info["gait_step"] = 0
        else:
            info["gait_step"] = info["gait_step"] + 1

        info["step"] = s

        step_idx = int(s % l_cycle)
        ref_data = ref_data_cache[step_idx]

        xy_star_before = info["xy*"].copy()
        _update_raibert_target_np(env, ref_data, info)
        _update_foot_cycloid_ref_np(env, info)

        xy_star_after = info["xy*"]
        raibert_updated = np.any(np.abs(xy_star_after - xy_star_before) > 1e-6)

        step_log.append(s)
        gait_step_log.append(info["gait_step"])
        is_stationary_log.append(float(is_stationary))
        swing_mask_l_log.append(float(info["foot_swing"][0]))
        swing_mask_r_log.append(float(info["foot_swing"][1]))
        raibert_update_log.append(float(raibert_updated))
        cmd_norm_log.append(cmd_norm)

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(step_log, cmd_norm_log, "b-", linewidth=2, label="cmd_norm")
    axes[0].axvline(x=stationary_start, color="r", linestyle="--", label="stationary phase")
    axes[0].axvline(x=stationary_end, color="r", linestyle="--")
    axes[0].set_ylabel("Command Norm")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].step(step_log, is_stationary_log, "r-", where="mid", linewidth=2, label="is_stationary")
    axes[1].set_ylabel("Stationary")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(step_log, gait_step_log, "g-", linewidth=2, label="gait_step")
    axes[2].set_ylabel("Gait Step")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(step_log, swing_mask_l_log, "b-", linewidth=2, label="L swing_mask")
    axes[3].plot(step_log, swing_mask_r_log, "r-", linewidth=2, label="R swing_mask")
    axes[3].set_ylabel("Swing Mask")
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    axes[4].scatter(step_log, raibert_update_log, c="purple", s=10, label="Raibert update")
    axes[4].set_ylabel("Target Updated")
    axes[4].set_ylim(-0.1, 1.1)
    axes[4].set_xlabel("Step")
    axes[4].legend(loc="upper right")
    axes[4].grid(True, alpha=0.3)

    for ax in axes:
        ax.axvspan(stationary_start, stationary_end, color="gray", alpha=0.1)

    plt.suptitle(f"G1 Gait Step Transition Test (step_k={env.step_k})")
    plt.tight_layout()
    plt.show()

    print("\n=== Assertion Checks ===")
    stationary_gait_steps = gait_step_log[stationary_start:stationary_end]
    assert all(gs == 0 for gs in stationary_gait_steps), (
        f"FAIL: gait_step should be 0 during stationary, got {stationary_gait_steps[:10]}..."
    )
    print("✓ Check 1: gait_step = 0 during stationary phase")

    stationary_swing_l = swing_mask_l_log[stationary_start:stationary_end]
    stationary_swing_r = swing_mask_r_log[stationary_start:stationary_end]
    assert all(s == 0 for s in stationary_swing_l), "FAIL: L swing_mask should be 0 during stationary"
    assert all(s == 0 for s in stationary_swing_r), "FAIL: R swing_mask should be 0 during stationary"
    print("✓ Check 2: swing_mask = [0, 0] during stationary phase")

    post_stationary_gait_steps = gait_step_log[stationary_end:stationary_end + 10]
    expected_steps = list(range(1, 11))
    assert post_stationary_gait_steps == expected_steps, (
        f"FAIL: gait_step should resume from 0, got {post_stationary_gait_steps}"
    )
    print("✓ Check 3: gait_step resumes from 0 after stationary phase")

    moving_updates_before = raibert_update_log[:stationary_start]
    moving_updates_after = raibert_update_log[stationary_end:]
    assert sum(moving_updates_before) > 0, "FAIL: Raibert target should update during moving phase (before)"
    assert sum(moving_updates_after) > 0, "FAIL: Raibert target should update during moving phase (after)"
    print("✓ Check 4: Raibert target updates normally during moving phase")

    print("\n=== All Checks Passed! ===")
    return fig


# ---------------------------------------------------------------------------
# 7. Cycloid foot trajectory playback
# ---------------------------------------------------------------------------

def play_cycloid_foot_trajectory(
    env,
    command=None,
    foot="left_foot",
    num_steps=None,
    render_every=2,
    save_path=None,
    camera=None,
    trail_stride=2,
    trail_max=80,
):
    print("Playing G1 bipedal foot trajectory with ref kino [native MuJoCo]...")

    if command is None:
        command = np.array([1.0, 0.0, 0.0])
    command = np.asarray(command, dtype=np.float64)

    if foot in consts.FEET_SITES:
        foot_idx = consts.FEET_SITES.index(foot)
    else:
        raise ValueError(f"Unknown foot site: {foot}. Expected one of {consts.FEET_SITES}.")

    l_cycle = int(env.l_cycle)
    if num_steps is None:
        num_steps = l_cycle
    ref_qpos = np.array(env.kinematic_ref_qpos[:l_cycle])

    # ── Pre-build kinematic reference frames (native MuJoCo) ──
    print("  Building kinematic reference frames (%d unique)..." % l_cycle)
    ref_data_cache = []
    for i in range(l_cycle):
        d = mujoco.MjData(env.mj_model)
        d.qpos[:] = env.kinematic_ref_qpos[i]
        d.qvel[:] = 0
        d.ctrl[:] = 0
        mujoco.mj_forward(env.mj_model, d)
        ref_data_cache.append(d)

    # ── Compute foot trajectory ──
    print("  Computing foot trajectory (%d steps)..." % num_steps)
    info = {
        "step": 0,
        "gait_step": 0,
        "is_stationary": False,
        "k0": 0,
        "command": command,
        "xy0": np.zeros((2, 2)),
        "xy*": np.zeros((2, 2)),
        "foot_phase": 0.0,
        "foot_swing": np.zeros(2),
        "foot_ref_xy": np.zeros((2, 2)),
        "foot_ref_z": np.zeros(2),
        "foot_ref_pos": np.zeros((2, 3)),
        "foot_ref_v_xy": np.zeros((2, 2)),
        "z0": np.zeros(2),
    }

    path_xyz = np.zeros((num_steps, 3), dtype=np.float32)

    for i in range(num_steps):
        info["step"] = i
        info["gait_step"] = i
        info["is_stationary"] = False

        _update_raibert_target_np(env, ref_data_cache[i % l_cycle], info)
        _update_foot_cycloid_ref_np(env, info)

        path_xyz[i] = np.array(info["foot_ref_pos"][foot_idx], dtype=np.float32)

    # ── Build scene modification callbacks ──
    def _add_sphere(scn, pos, radius, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        scn.ngeom += 1
        scn.geoms[scn.ngeom - 1].category = mujoco.mjtCatBit.mjCAT_DECOR
        mujoco.mjv_initGeom(
            geom=scn.geoms[scn.ngeom - 1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0.0, 0.0]),
            pos=np.array(pos),
            mat=np.eye(3).flatten().astype(np.float32),
            rgba=np.asarray(rgba).astype(np.float32),
        )

    modify_scene_fns = []
    for i in range(num_steps):
        def make_fn(idx=i):
            def _fn(scn):
                trail = path_xyz[: idx + 1 : max(1, trail_stride)]
                if trail_max is not None and len(trail) > trail_max:
                    trail = trail[-trail_max:]
                for p in trail:
                    _add_sphere(scn, p, radius=0.007, rgba=[0.2, 0.8, 0.2, 0.6])
                _add_sphere(scn, path_xyz[idx], radius=0.012, rgba=[0.9, 0.3, 0.2, 0.9])
                _add_sphere(scn, path_xyz[0], radius=0.009, rgba=[0.2, 0.4, 0.9, 0.9])
                _add_sphere(scn, path_xyz[-1], radius=0.009, rgba=[0.2, 0.4, 0.9, 0.9])
            return _fn
        modify_scene_fns.append(make_fn())

    # ── Render trajectory with native MuJoCo ──
    print("  Rendering (%d frames)..." % (num_steps // render_every))
    renderer = mujoco.Renderer(env.mj_model, height=480, width=640)

    if camera is not None:
        camera_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    else:
        camera_id = -1

    render_data = mujoco.MjData(env.mj_model)
    frames_out = []

    try:
        for i, idx in enumerate(range(0, num_steps, render_every)):
            render_data.qpos[:] = ref_qpos[idx % l_cycle]
            render_data.qvel[:] = 0
            mujoco.mj_forward(env.mj_model, render_data)
            renderer.update_scene(render_data, camera=camera_id)

            if idx < len(modify_scene_fns):
                modify_scene_fns[idx](renderer.scene)

            frames_out.append(renderer.render())
    finally:
        renderer.close()

    if save_path is not None:
        media.write_video(save_path, frames_out, fps=int(1.0 / (env.dt * render_every)))

    fps = 1.0 / (env.dt * render_every)
    media.show_video(frames_out, fps=fps, loop=True)
    return frames_out, path_xyz
