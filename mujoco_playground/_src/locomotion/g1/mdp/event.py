# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
# ==============================================================================
"""Disturbance (kick) pipeline for G1 environments."""

import jax
import jax.numpy as jp


def init_disturbance(info, *, disturbance_cfg, dt, prefix, rng_key="rng"):
    info[f"{prefix}_dir"] = jp.zeros(2)
    info[f"{prefix}_mag"] = 0.0
    info[f"steps_until_next_{prefix}"] = jp.array(
        int(jp.mean(jp.array(disturbance_cfg.kick_wait_times)) / dt), dtype=jp.int32
    )
    info[f"{prefix}_duration"] = jp.array(0, dtype=jp.int32)
    info[f"steps_since_last_{prefix}"] = jp.array(0, dtype=jp.int32)
    info[f"{prefix}_steps"] = jp.array(0, dtype=jp.int32)
    return info


def maybe_apply_disturbance(
    state, *, disturbance_cfg, dt, base_mass, nbody, base_id, prefix, rng_key="rng"
):
    info = state.info
    info[f"steps_until_next_{prefix}"] = info[f"steps_until_next_{prefix}"] - 1
    info[f"steps_since_last_{prefix}"] = info[f"steps_since_last_{prefix}"] + 1

    # Sample new kick when timer expires.
    info[rng_key], dir_rng, dur_rng, mag_rng = jax.random.split(info[rng_key], 4)
    theta = jax.random.uniform(dir_rng, maxval=2 * jp.pi)
    info[f"{prefix}_dir"] = jp.array([jp.cos(theta), jp.sin(theta)])

    kick_dur = jax.random.uniform(
        dur_rng,
        minval=disturbance_cfg.kick_durations[0],
        maxval=disturbance_cfg.kick_durations[1],
    )
    info[f"{prefix}_duration_seconds"] = kick_dur
    info[f"{prefix}_duration"] = jp.array(int(kick_dur / dt), dtype=jp.int32)

    mag = jax.random.uniform(
        mag_rng,
        minval=disturbance_cfg.velocity_kick[0],
        maxval=disturbance_cfg.velocity_kick[1],
    )
    info[f"{prefix}_mag"] = mag

    # Apply kick.
    info[f"{prefix}_steps"] = info[f"{prefix}_steps"] + 1
    active = info[f"{prefix}_steps"] <= info[f"{prefix}_duration"]
    active &= info[f"steps_until_next_{prefix}"] <= 0

    force = info[f"{prefix}_dir"] * info[f"{prefix}_mag"] * base_mass / dt
    qvel = state.data.qvel.at[:2].add(force * active.astype(jp.float32))
    data = state.data.replace(qvel=qvel)

    # Reset timer after kick.
    wait_time = jax.random.uniform(
        info[rng_key],
        minval=disturbance_cfg.kick_wait_times[0],
        maxval=disturbance_cfg.kick_wait_times[1],
    )
    new_wait = jp.array(int(wait_time / dt), dtype=jp.int32)

    info[f"steps_until_next_{prefix}"] = jp.where(
        active, info[f"steps_until_next_{prefix}"], new_wait
    )
    info[f"steps_since_last_{prefix}"] = jp.where(
        active, jp.array(0, dtype=jp.int32), info[f"steps_since_last_{prefix}"]
    )
    info[f"{prefix}_steps"] = jp.where(
        active, info[f"{prefix}_steps"], jp.array(0, dtype=jp.int32)
    )

    return state.replace(data=data, info=info)
