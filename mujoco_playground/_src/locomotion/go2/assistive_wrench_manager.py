from typing import Any, Dict, Sequence

import jax
import jax.numpy as jp
from ml_collections import config_dict


def _zeros3() -> jax.Array:
    return jp.zeros(3)


def _clip_vector_norm(vec: jax.Array, max_norm: float) -> jax.Array:
    norm = jp.linalg.norm(vec)
    scale = jp.minimum(1.0, max_norm / (norm + 1e-6))
    return vec * scale


class AssistiveWrenchManager:
    """Computes and applies an assistive base wrench before physics."""

    def __init__(
        self,
        cfg: config_dict.ConfigDict,
        *,
        base_id: int,
        base_mass: float,
        subtree_mass: float,
        base_inertia: jax.Array,
    ) -> None:
        self._cfg = cfg
        self._base_id = base_id
        self._base_mass = float(jp.asarray(base_mass).reshape(-1)[0])
        self._subtree_mass = float(jp.asarray(subtree_mass).reshape(-1)[0])
        self._base_inertia = jp.asarray(base_inertia).reshape(-1)[:3]

    def init_state(self, info: Dict[str, Any]) -> Dict[str, Any]:
        info["assist_beta"] = jp.array(self._cfg.beta.initial)
        info["assist_tracking_ema"] = jp.array(0.0)
        info["assist_force_world"] = _zeros3()
        info["assist_torque_world"] = _zeros3()
        info["assist_force_norm"] = jp.array(0.0)
        info["assist_torque_norm"] = jp.array(0.0)
        return info

    def compute_and_apply_wrench(
        self,
        data: Any,
        info: Dict[str, Any],
        command: jax.Array,
        *,
        get_local_linvel: Any,
        get_global_linvel: Any,
        get_gyro: Any,
        get_gravity: Any,
    ) -> tuple[Any, Dict[str, Any]]:
        if not self._cfg.enable:
            info["assist_force_world"] = _zeros3()
            info["assist_torque_world"] = _zeros3()
            info["assist_force_norm"] = jp.array(0.0)
            info["assist_torque_norm"] = jp.array(0.0)
            return data, info

        local_linvel = get_local_linvel(data)
        global_linvel = get_global_linvel(data)
        angvel_body = get_gyro(data)
        gravity_body = get_gravity(data)
        base_xmat = data.xmat[self._base_id]
        base_height = data.qpos[2]

        roll = jp.arctan2(gravity_body[1], -gravity_body[2])
        pitch = jp.arctan2(
            -gravity_body[0],
            jp.sqrt(gravity_body[1] ** 2 + gravity_body[2] ** 2) + 1e-6,
        )

        cmd_xy = command[:2]
        vel_err_xy = cmd_xy - local_linvel[:2]
        force_xy_body = self._cfg.gains.xy_d * vel_err_xy
        force_xy_world = base_xmat @ jp.array([force_xy_body[0], force_xy_body[1], 0.0])

        height_err = self._cfg.z_ref - base_height
        force_z = (
            self._cfg.gains.z_p * height_err
            + self._cfg.gains.z_d * (-global_linvel[2])
        )
        ff_mass = self._base_mass if self._cfg.ff_mass_mode == "base" else self._subtree_mass
        gravity_comp = jp.where(
            self._cfg.enable_feedforward,
            jp.array([0.0, 0.0, ff_mass * 9.81]),
            _zeros3(),
        )
        force_world_raw = gravity_comp + force_xy_world + jp.array([0.0, 0.0, force_z])

        inertia_omega = self._base_inertia * angvel_body
        torque_ff_body = jp.where(
            self._cfg.enable_feedforward,
            jp.cross(angvel_body, inertia_omega),
            _zeros3(),
        )
        torque_fb_body = jp.array(
            [
                self._cfg.gains.roll_p * roll - self._cfg.gains.roll_d * angvel_body[0],
                self._cfg.gains.pitch_p * pitch - self._cfg.gains.pitch_d * angvel_body[1],
                self._cfg.gains.yaw_d * (command[2] - angvel_body[2]),
            ]
        )
        torque_world_raw = base_xmat @ (torque_ff_body + torque_fb_body)

        beta = jp.clip(info["assist_beta"], 0.0, self._cfg.beta.max)
        force_world = _clip_vector_norm(beta * force_world_raw, self._cfg.force_limit)
        torque_world = _clip_vector_norm(beta * torque_world_raw, self._cfg.torque_limit)

        # Cut off the gradient flow for the assistive wrench to prevent APG explosion.
        # The simulator will still feel the force, but JAX won't backpropagate through its calculation.
        force_world = jax.lax.stop_gradient(force_world)
        torque_world = jax.lax.stop_gradient(torque_world)

        xfrc_applied = data.xfrc_applied.at[self._base_id, :3].add(force_world)
        xfrc_applied = xfrc_applied.at[self._base_id, 3:].add(torque_world)
        data = data.replace(xfrc_applied=xfrc_applied)

        info["assist_force_world"] = force_world
        info["assist_torque_world"] = torque_world
        info["assist_force_norm"] = jp.linalg.norm(force_world)
        info["assist_torque_norm"] = jp.linalg.norm(torque_world)
        return data, info

    def update_curriculum(
        self,
        info: Dict[str, Any],
        command: jax.Array,
        *,
        local_linvel: jax.Array,
        yaw_rate: jax.Array,
    ) -> Dict[str, Any]:
        if not self._cfg.enable:
            return info

        mode = self._cfg.curriculum.mode
        step = info["step"]

        if mode in ("linear", "staircase"):
            start = self._cfg.curriculum.start_step
            end = max(self._cfg.curriculum.end_step, start + 1)
            progress = jp.clip((step - start) / (end - start), 0.0, 1.0)

            if mode == "linear":
                beta = (
                    self._cfg.beta.initial
                    + progress * (self._cfg.beta.final - self._cfg.beta.initial)
                )
            else:
                levels = max(int(self._cfg.curriculum.staircase_levels), 1)
                step_index = jp.floor(progress * levels)
                step_index = jp.minimum(step_index, levels - 1)
                level_progress = step_index / jp.maximum(levels - 1, 1)
                beta = (
                    self._cfg.beta.initial
                    + level_progress * (self._cfg.beta.final - self._cfg.beta.initial)
                )

            beta = jp.clip(beta, 0.0, self._cfg.beta.max)
            if self._cfg.curriculum.hard_disable_after_end:
                beta = jp.where(step >= end, 0.0, beta)
            info["assist_beta"] = beta
            return info

        lin_err = jp.linalg.norm(command[:2] - local_linvel[:2])
        yaw_err = jp.abs(command[2] - yaw_rate)
        tracking_err = lin_err + self._cfg.curriculum.yaw_error_weight * yaw_err
        ema = (
            self._cfg.curriculum.ema_alpha * info["assist_tracking_ema"]
            + (1.0 - self._cfg.curriculum.ema_alpha) * tracking_err
        )
        should_decay = ema < self._cfg.curriculum.tracking_error_threshold
        beta = jp.where(
            should_decay,
            info["assist_beta"] - self._cfg.curriculum.decay_per_step,
            info["assist_beta"],
        )
        info["assist_tracking_ema"] = ema
        info["assist_beta"] = jp.clip(beta, self._cfg.beta.final, self._cfg.beta.max)
        return info

    def build_modify_scene_fns(
        self,
        trajectory: Sequence[Any],
        *,
        render_every: int,
        force_arrow_scale: float,
        torque_arrow_scale: float,
        arrow_radius: float,
    ) -> list[Any]:
        import mujoco
        import numpy as np

        states = trajectory[::render_every]
        base_pos = np.asarray(
            jax.device_get(jp.stack([s.data.xpos[self._base_id] for s in states]))
        )
        force_world = np.asarray(
            jax.device_get(jp.stack([s.info["assist_force_world"] for s in states]))
        )
        torque_world = np.asarray(
            jax.device_get(jp.stack([s.info["assist_torque_world"] for s in states]))
        )
        force_norm = np.asarray(
            jax.device_get(jp.stack([s.info["assist_force_norm"] for s in states]))
        )
        torque_norm = np.asarray(
            jax.device_get(jp.stack([s.info["assist_torque_norm"] for s in states]))
        )

        modify_scene_fns = []
        for idx in range(len(states)):
            origin = base_pos[idx]
            force = force_world[idx]
            torque = torque_world[idx]
            norm_force = float(force_norm[idx])
            norm_torque = float(torque_norm[idx])

            def make_fn(p=origin, f=force, tau=torque, f_norm=norm_force, tau_norm=norm_torque):
                def _add_arrow(scn, start, vec, max_norm, arrow_scale, rgba):
                    if scn.ngeom >= scn.maxgeom:
                        return
                    vec_norm = np.linalg.norm(vec)
                    if vec_norm < 1e-6:
                        return
                    direction = vec / (vec_norm + 1e-6)
                    end = start + direction * arrow_scale * vec_norm / max(max_norm, 1e-6)
                    scn.ngeom += 1
                    scn.geoms[scn.ngeom - 1].category = mujoco.mjtCatBit.mjCAT_DECOR
                    mujoco.mjv_initGeom(
                        geom=scn.geoms[scn.ngeom - 1],
                        type=mujoco.mjtGeom.mjGEOM_ARROW.value,
                        size=np.zeros(3),
                        pos=np.zeros(3),
                        mat=np.zeros(9),
                        rgba=np.asarray(rgba, dtype=np.float32),
                    )
                    mujoco.mjv_connector(
                        geom=scn.geoms[scn.ngeom - 1],
                        type=mujoco.mjtGeom.mjGEOM_ARROW.value,
                        width=arrow_radius,
                        from_=start,
                        to=end,
                    )

                def _fn(scn):
                    if f_norm >= 1e-6:
                        _add_arrow(
                            scn,
                            p,
                            f,
                            self._cfg.force_limit,
                            force_arrow_scale,
                            [0.95, 0.25, 0.2, 0.9],
                        )
                    if tau_norm >= 1e-6:
                        _add_arrow(
                            scn,
                            p + np.array([0.0, 0.0, 0.03]),
                            tau,
                            self._cfg.torque_limit,
                            torque_arrow_scale,
                            [0.2, 0.55, 0.95, 0.9],
                        )
                return _fn

            modify_scene_fns.append(make_fn())
        return modify_scene_fns
