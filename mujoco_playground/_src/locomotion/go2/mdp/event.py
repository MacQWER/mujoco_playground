from typing import Any, Dict

import jax
import jax.numpy as jp


def _sample_disturbance(rng: jax.Array, disturbance_cfg: Any, dt: float, prefix: str) -> tuple[jax.Array, Dict[str, jax.Array]]:
    rng, key_wait, key_dur, key_mag = jax.random.split(rng, 4)
    time_until_next = jax.random.uniform(
        key_wait,
        minval=disturbance_cfg.kick_wait_times[0],
        maxval=disturbance_cfg.kick_wait_times[1],
    )
    duration_seconds = jax.random.uniform(
        key_dur,
        minval=disturbance_cfg.kick_durations[0],
        maxval=disturbance_cfg.kick_durations[1],
    )
    magnitude = jax.random.uniform(
        key_mag,
        minval=disturbance_cfg.velocity_kick[0],
        maxval=disturbance_cfg.velocity_kick[1],
    )
    return rng, {
        f"steps_until_next_{prefix}": jp.round(time_until_next / dt).astype(jp.int32),
        f"{prefix}_duration_seconds": duration_seconds,
        f"{prefix}_duration": jp.round(duration_seconds / dt).astype(jp.int32),
        f"steps_since_last_{prefix}": jp.array(0, dtype=jp.int32),
        f"{prefix}_steps": jp.array(0, dtype=jp.int32),
        f"{prefix}_dir": jp.zeros(3),
        f"{prefix}_mag": magnitude,
    }


def init_disturbance(
    state_info: Dict[str, Any],
    disturbance_cfg: Any,
    dt: float,
    prefix: str,
    rng_key: str = "rng",
) -> Dict[str, Any]:
    rng, fields = _sample_disturbance(state_info[rng_key], disturbance_cfg, dt, prefix)
    state_info[rng_key] = rng
    state_info.update(fields)
    return state_info


def maybe_apply_disturbance(
    state,
    disturbance_cfg: Any,
    dt: float,
    base_mass: float,
    nbody: int,
    base_id: int,
    prefix: str,
    rng_key: str = "rng",
):
    steps_until_next_key = f"steps_until_next_{prefix}"
    duration_seconds_key = f"{prefix}_duration_seconds"
    duration_key = f"{prefix}_duration"
    steps_since_last_key = f"steps_since_last_{prefix}"
    steps_key = f"{prefix}_steps"
    dir_key = f"{prefix}_dir"
    mag_key = f"{prefix}_mag"

    def gen_dir(rng: jax.Array) -> jax.Array:
        angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
        return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

    def apply_disturbance(state):
        t = state.info[steps_key] * dt
        u_t = 0.5 * jp.sin(jp.pi * t / state.info[duration_seconds_key])
        force = u_t * base_mass * state.info[mag_key] / state.info[duration_seconds_key]
        xfrc_applied = jp.zeros((nbody, 6))
        xfrc_applied = xfrc_applied.at[base_id, :3].set(force * state.info[dir_key])

        state.info[rng_key], fields = _sample_disturbance(
            state.info[rng_key], disturbance_cfg, dt, prefix
        )
        done_kick = state.info[steps_key] >= state.info[duration_key]

        data = state.data.replace(xfrc_applied=xfrc_applied)
        state = state.replace(data=data)
        state.info[steps_since_last_key] = jp.where(
            done_kick,
            0,
            state.info[steps_since_last_key],
        )
        state.info[steps_until_next_key] = jp.where(
            done_kick,
            fields[steps_until_next_key],
            state.info[steps_until_next_key],
        )
        state.info[duration_seconds_key] = jp.where(
            done_kick,
            fields[duration_seconds_key],
            state.info[duration_seconds_key],
        )
        state.info[duration_key] = jp.where(
            done_kick,
            fields[duration_key],
            state.info[duration_key],
        )
        state.info[mag_key] = jp.where(
            done_kick,
            fields[mag_key],
            state.info[mag_key],
        )
        state.info[steps_key] += 1
        return state

    def wait(state):
        state.info[rng_key], rng = jax.random.split(state.info[rng_key])
        state.info[steps_since_last_key] += 1
        xfrc_applied = jp.zeros((nbody, 6))
        data = state.data.replace(xfrc_applied=xfrc_applied)
        should_start = state.info[steps_since_last_key] >= state.info[steps_until_next_key]
        state.info[steps_key] = jp.where(should_start, 0, state.info[steps_key])
        state.info[dir_key] = jp.where(should_start, gen_dir(rng), state.info[dir_key])
        return state.replace(data=data)

    return jax.lax.cond(
        state.info[steps_since_last_key] >= state.info[steps_until_next_key],
        apply_disturbance,
        wait,
        state,
    )
