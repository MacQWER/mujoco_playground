"""Validation tools for state alignment and gradient analysis."""

from typing import Any, Dict, Sequence

import jax
import jax.numpy as jp
import numpy as np
import pandas as pd


def make_alignment_configs(
    use_mjx_xml_for_mujoco: bool = True,
) -> tuple:
    """Create a pair of matched MJX / MuJoCo configs for alignment verification.

    Disables disturbance and assistive wrench; clears anchor path.
    """
    from mujoco_playground._src.locomotion.go2 import go2_constants as consts
    from mujoco_playground._src.locomotion.go2.JoystickGo2 import (
        JoystickGo2,
        default_config as mjx_default_config,
    )
    from mujoco_playground._src.locomotion.go2.JoystickGo2Mujoco import (
        JoystickGo2Mujoco,
        default_config as mujoco_default_config,
    )

    mjx_cfg = mjx_default_config()
    mjx_cfg.anchor.path = None
    mjx_cfg.disturbance.enable = False
    mjx_cfg.assistive_wrench.enable = False

    mj_cfg = mujoco_default_config()
    mj_cfg.anchor.path = None
    mj_cfg.disturbance.enable = False
    mj_cfg.assistive_wrench.enable = False
    if use_mjx_xml_for_mujoco:
        mj_cfg.mujoco_model.xml_path = consts.MJX_XML_SENSOR_PATH.as_posix()

    return JoystickGo2, JoystickGo2Mujoco, mjx_cfg, mj_cfg


def verification_loss_fn(env, data) -> jax.Array:
    """Simple verification loss: base height + foot position penalty."""
    base_height = data.xpos[env.base_id, 2]
    foot_term = 0.01 * jp.sum(jp.square(data.site_xpos[env._feet_site_id]))
    return base_height + foot_term


# ---------------------------------------------------------------------------
# Data-frame builders
# ---------------------------------------------------------------------------


def build_trajectory_dataframe(
    rollout: Sequence[Any], env
) -> pd.DataFrame:
    """Build a trajectory DataFrame from a list of state objects."""
    rows = []
    for step, state in enumerate(rollout):
        local_linvel = np.asarray(jax.device_get(env.get_local_linvel(state.data)))
        command = np.asarray(jax.device_get(state.info["command"]))
        rows.append(
            {
                "step": step,
                "base_height": float(jax.device_get(state.data.xpos[env.base_id, 2])),
                "linvel_x": float(local_linvel[0]),
                "linvel_y": float(local_linvel[1]),
                "yaw_rate": float(jax.device_get(env.get_gyro(state.data)[2])),
                "action_norm": float(
                    np.linalg.norm(np.asarray(jax.device_get(state.info["last_residual"])))
                ),
                "command_x": float(command[0]),
                "command_y": float(command[1]),
                "command_yaw": float(command[2]),
                "reward": float(jax.device_get(state.reward)),
                "done": float(jax.device_get(state.done)),
            }
        )
    return pd.DataFrame(rows)


def build_state_gap_dataframe(
    mjx_rollout: Sequence[Any],
    align_rollout: Sequence[Any],
    native_rollout: Sequence[Any],
) -> pd.DataFrame:
    """Compute state-gap metrics between three rollout trajectories."""
    return pd.DataFrame(
        {
            "step": np.arange(len(mjx_rollout)),
            "qpos_gap_align_vs_mjx": [
                float(
                    np.linalg.norm(
                        np.asarray(jax.device_get(align_state.data.qpos))
                        - np.asarray(jax.device_get(mjx_state.data.qpos))
                    )
                )
                for mjx_state, align_state in zip(mjx_rollout, align_rollout)
            ],
            "qvel_gap_align_vs_mjx": [
                float(
                    np.linalg.norm(
                        np.asarray(jax.device_get(align_state.data.qvel))
                        - np.asarray(jax.device_get(mjx_state.data.qvel))
                    )
                )
                for mjx_state, align_state in zip(mjx_rollout, align_rollout)
            ],
            "qpos_gap_native_vs_align": [
                float(
                    np.linalg.norm(
                        np.asarray(jax.device_get(native_state.data.qpos))
                        - np.asarray(jax.device_get(align_state.data.qpos))
                    )
                )
                for native_state, align_state in zip(native_rollout, align_rollout)
            ],
        }
    )


# ---------------------------------------------------------------------------
# Gradient comparison
# ---------------------------------------------------------------------------


def gradient_comparison_metrics(
    grad_mjx: jax.Array,
    grad_alpha1: jax.Array,
    grad_alpha0: jax.Array,
    loss_mjx: float,
    loss_alpha1: float,
    loss_alpha0: float,
) -> pd.DataFrame:
    """Compute cosine similarity and norm metrics between gradients.

    Args:
        grad_mjx: Gradient from pure-MJX rollout (reference).
        grad_alpha1: Gradient from aligned rollout with alpha=1.
        grad_alpha0: Gradient from aligned rollout with alpha=0.
        loss_mjx: Loss value from MJX-only rollout.
        loss_alpha1: Loss value from aligned rollout with alpha=1.
        loss_alpha0: Loss value from aligned rollout with alpha=0.

    Returns:
        DataFrame with cosine similarity and norm comparisons.
    """
    g_mjx = jp.asarray(grad_mjx)
    g_a1 = jp.asarray(grad_alpha1)
    g_a0 = jp.asarray(grad_alpha0)

    def cosine_sim(a, b):
        return float(
            jp.dot(a.flatten(), b.flatten())
            / (jp.linalg.norm(a.flatten()) * jp.linalg.norm(b.flatten()) + 1e-12)
        )

    def norm_val(a):
        return float(jp.linalg.norm(a.flatten()))

    return pd.DataFrame(
        [
            {
                "case": "alpha_1",
                "cosine_vs_mjx": cosine_sim(g_mjx, g_a1),
                "norm": norm_val(g_a1),
                "norm_mjx": norm_val(g_mjx),
                "norm_diff": norm_val(g_a1) - norm_val(g_mjx),
                "loss": loss_alpha1,
                "loss_mjx": loss_mjx,
            },
            {
                "case": "alpha_0",
                "cosine_vs_mjx": cosine_sim(g_mjx, g_a0),
                "norm": norm_val(g_a0),
                "norm_mjx": norm_val(g_mjx),
                "norm_diff": norm_val(g_a0) - norm_val(g_mjx),
                "loss": loss_alpha0,
                "loss_mjx": loss_mjx,
            },
        ]
    )
