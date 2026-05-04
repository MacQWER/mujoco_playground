# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
# ==============================================================================
"""Command sampling and update logic for G1 environments."""

import jax
import jax.numpy as jp


def sample_command(rng: jax.Array, *, cmd_a: jax.Array, cmd_b: jax.Array) -> jax.Array:
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    lin_vel_x = jax.random.uniform(rng1, minval=-cmd_a[0], maxval=cmd_a[0])
    lin_vel_y = jax.random.uniform(rng2, minval=-cmd_a[1], maxval=cmd_a[1])
    ang_vel_yaw = jax.random.uniform(rng3, minval=-cmd_a[2], maxval=cmd_a[2])

    return jp.where(
        jax.random.bernoulli(rng4, p=jp.sum(1.0 - cmd_b) / 3.0),
        jp.zeros(3),
        jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
    )


def init_command_state(info, *, command=None, steps_until_next_cmd=None):
    if command is None:
        command = jp.zeros(3)
    if steps_until_next_cmd is None:
        steps_until_next_cmd = jp.array(500, dtype=jp.int32)
    info["command"] = command
    info["steps_until_next_cmd"] = steps_until_next_cmd
    return info


def update_command(info, *, dt, cmd_a, cmd_b, mean_replan_time=2.5):
    info["steps_until_next_cmd"] = info["steps_until_next_cmd"] - 1

    info["rng"], cmd_rng = jax.random.split(info["rng"])
    new_cmd = sample_command(cmd_rng, cmd_a=cmd_a, cmd_b=cmd_b)

    info["command"] = jp.where(
        info["steps_until_next_cmd"] <= 0,
        new_cmd,
        info["command"],
    )
    info["steps_until_next_cmd"] = jp.where(
        info["steps_until_next_cmd"] <= 0,
        jp.array(int(mean_replan_time / dt)),
        info["steps_until_next_cmd"],
    )
    return info
