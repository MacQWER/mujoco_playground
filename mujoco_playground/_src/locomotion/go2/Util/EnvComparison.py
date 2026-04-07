"""Utilities for comparing MJX and native MuJoCo environments."""

from typing import List

import jax
import jax.numpy as jp

from mujoco_playground import locomotion


def compare_mjx_and_mujoco(
    env_name_mjx: str = "Go2Joystick2",
    env_name_mj: str = "Go2JoystickMujoco",
    seed: int = 0,
    horizon: int = 10,
    action_scale: float = 0.0,
    use_mjx_xml_for_mujoco: bool = True,
) -> List[dict]:
    """Run open-loop rollouts on both backends and return per-step mismatch metrics.

    Args:
        env_name_mjx: MJX environment name.
        env_name_mj: Native MuJoCo environment name.
        seed: RNG seed for reset and action generation.
        horizon: Number of steps to roll out.
        action_scale: If 0, uses zero actions. Otherwise samples uniformly in
            ``[-action_scale, action_scale]``.
        use_mjx_xml_for_mujoco: Whether to force the MuJoCo env to use the MJX
            XML path so the models start as close as possible.

    Returns:
        A list of dicts with keys ``tag``, ``obs_linf``, ``qpos_linf``,
        ``qvel_linf``, ``reward_abs_diff``, ``done_abs_diff``.
    """
    from mujoco_playground._src.locomotion.go2 import go2_constants as consts

    mjx_env_cfg = locomotion.get_default_config(env_name_mjx)
    mj_env_cfg = locomotion.get_default_config(env_name_mj)
    mjx_env_cfg.disturbance.enable = False
    mj_env_cfg.disturbance.enable = False
    if use_mjx_xml_for_mujoco:
        mj_env_cfg.mujoco_model.xml_path = consts.MJX_XML_SENSOR_PATH.as_posix()

    mjx_env = locomotion.load(env_name_mjx, config=mjx_env_cfg)
    mj_env = locomotion.load(env_name_mj, config=mj_env_cfg)

    rng = jax.random.PRNGKey(seed)
    state_mjx = mjx_env.reset(rng)
    state_mj = mj_env.reset(rng)

    rows: List[dict] = []

    def collect(tag: str, s_mjx, s_mj) -> None:
        rows.append(
            {
                "tag": tag,
                "obs_linf": float(jp.max(jp.abs(s_mjx.obs["state"] - s_mj.obs["state"]))),
                "qpos_linf": float(jp.max(jp.abs(s_mjx.data.qpos - s_mj.data.qpos))),
                "qvel_linf": float(jp.max(jp.abs(s_mjx.data.qvel - s_mj.data.qvel))),
                "reward_abs_diff": float(jp.abs(s_mjx.reward - s_mj.reward)),
                "done_abs_diff": float(jp.abs(s_mjx.done - s_mj.done)),
            }
        )

    collect("reset", state_mjx, state_mj)

    action_rng = jax.random.PRNGKey(seed + 1)
    for step in range(horizon):
        if action_scale == 0.0:
            action = jp.zeros(mjx_env.action_size)
        else:
            action_rng, subkey = jax.random.split(action_rng)
            action = jax.random.uniform(
                subkey,
                (mjx_env.action_size,),
                minval=-action_scale,
                maxval=action_scale,
            )
        state_mjx = mjx_env.step(state_mjx, action)
        state_mj = mj_env.step(state_mj, action)
        collect(f"step_{step + 1}", state_mjx, state_mj)

    return rows
