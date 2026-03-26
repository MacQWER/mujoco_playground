from typing import Any, Dict

import jax
import jax.numpy as jp


def sample_command(rng: jax.Array, cmd_a: jax.Array, cmd_b: jax.Array) -> jax.Array:
    """Samples a masked velocity command."""
    _, key_cmd, key_mask = jax.random.split(rng, 3)
    cmd = jax.random.uniform(key_cmd, (3,), minval=-cmd_a, maxval=cmd_a)
    mask = jax.random.bernoulli(key_mask, cmd_b, (3,))
    return cmd * mask


def init_command_state(
    state_info: Dict[str, Any],
    *,
    command: jax.Array | None = None,
    steps_until_next_cmd: int | jax.Array | None = None,
) -> Dict[str, Any]:
    if command is None:
        command = jp.zeros(3)
    state_info["command"] = command
    if steps_until_next_cmd is not None:
        state_info["steps_until_next_cmd"] = steps_until_next_cmd
    return state_info


def update_command(
    info: Dict[str, Any],
    *,
    dt: float,
    cmd_a: jax.Array,
    cmd_b: jax.Array,
    mean_replan_time: float = 2.5,
) -> Dict[str, Any]:
    info["rng"], key_cmd, key_time = jax.random.split(info["rng"], 3)
    steps_until_next_cmd = info["steps_until_next_cmd"] - 1
    should_update = steps_until_next_cmd <= 0
    new_cmd = sample_command(key_cmd, cmd_a=cmd_a, cmd_b=cmd_b)
    new_timer = jp.round(
        jax.random.exponential(key_time) * mean_replan_time / dt
    ).astype(jp.int32)
    info["command"] = jp.where(should_update, new_cmd, info["command"])
    info["steps_until_next_cmd"] = jp.where(
        should_update, new_timer, steps_until_next_cmd
    )
    return info
