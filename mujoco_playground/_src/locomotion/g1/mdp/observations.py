# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
# ==============================================================================
"""Observation functions for G1 joystick environments."""

import jax
import jax.numpy as jp


def apply_uniform_noise(rng: jax.Array, x: jax.Array, noise_level: float, scale: float):
    rng, noise_rng = jax.random.split(rng)
    noise = (2 * jax.random.uniform(noise_rng, shape=x.shape) - 1) * noise_level * scale
    return x + noise, rng


def base_angular_velocity(data, info, **kwargs) -> jax.Array:
    del info
    get_gyro = kwargs.get("get_gyro")
    if get_gyro is not None:
        return get_gyro(data, "pelvis")
    return data.cvel[1, :3]


def base_linear_velocity(data, info, **kwargs) -> jax.Array:
    del info
    return kwargs["get_local_linvel"](data, "pelvis")


def projected_gravity(data, info, **kwargs) -> jax.Array:
    del info
    pelvis_imu_site_id = kwargs.get("pelvis_imu_site_id", 1)
    return data.site_xmat[pelvis_imu_site_id].T @ jp.array([0, 0, -1])


def joint_positions(data, info, **kwargs) -> jax.Array:
    angles = data.qpos[7:]
    return angles - kwargs["default_ap_pose"]


def joint_velocities(data, info, **kwargs) -> jax.Array:
    del info, kwargs
    return data.qvel[6:]


def command(data, info, **kwargs) -> jax.Array:
    del data, kwargs
    return info["command"]


def last_action(data, info, **kwargs) -> jax.Array:
    del data, kwargs
    return info["last_action"]


def kinematic_reference(data, info, **kwargs) -> jax.Array:
    l_cycle = kwargs["l_cycle"]
    step_idx = jp.array(info["gait_step"] % l_cycle, int)
    return kwargs["kin_ref_qpos"][step_idx][7:]


def gait_phase(data, info, **kwargs) -> jax.Array:
    del data, kwargs
    phase = info["phase"]
    return jp.concatenate([jp.cos(phase), jp.sin(phase)])
