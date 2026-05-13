# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
# ==============================================================================
"""G1 Joystick v2 environment with Raibert heuristic + cycloid foot trajectories."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import g1_constants as consts
from mujoco_playground._src.locomotion.g1.mdp import rewards as reward_lib
from mujoco_playground._src.locomotion.g1.Util import G1Util as g1_utils
from mujoco_playground._src.locomotion.go2.Util.TrotUtil import (
    cos_wave,
    dcos_wave,
    rotate,
    rotate_inv,
)


def default_config() -> config_dict.ConfigDict:
    from mujoco_playground._src.locomotion.g1.configs.joystick_config import default_config as dc
    return dc()


class G1Joystick2(g1_base.G1Env):
    """G1 bipedal joystick tracking with Raibert heuristic and cycloid trajectories."""

    def __init__(
        self,
        task: Optional[str] = None,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        del task  # Unused; kept for API compatibility.
        super().__init__(
            xml_path=consts.MJX_XML_SENSOR_PATH.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        # Keyframe poses (use knees_bent like original G1).
        self._init_q = jp.array(self._mj_model.keyframe("knees_bent").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("knees_bent").qpos[7:])

        # Joint limits.
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        # Joint group indices.
        waist_indices = []
        for joint_name in ["waist_yaw", "waist_roll", "waist_pitch"]:
            waist_indices.append(self._mj_model.joint(f"{joint_name}_joint").qposadr - 7)
        self._waist_indices = jp.array(waist_indices)

        arm_indices = []
        for side in ["left", "right"]:
            for joint_name in ["shoulder_roll", "shoulder_yaw", "wrist_roll", "wrist_pitch", "wrist_yaw"]:
                arm_indices.append(self._mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7)
        self._arm_indices = jp.array(arm_indices)

        hip_indices = []
        for side in ["left", "right"]:
            for joint_name in ["hip_roll", "hip_yaw"]:
                hip_indices.append(self._mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7)
        self._hip_indices = jp.array(hip_indices)

        knee_indices = []
        for side in ["left", "right"]:
            knee_indices.append(self._mj_model.joint(f"{side}_knee_joint").qposadr - 7)
        self._knee_indices = jp.array(knee_indices)

        # Joint weights for pose reward (same as original G1 joystick).
        self._weights = jp.array([
            0.01, 1.0, 1.0, 0.01, 1.0, 1.0,   # left leg
            0.01, 1.0, 1.0, 0.01, 1.0, 1.0,   # right leg
            1.0, 1.0, 1.0,                      # waist
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left arm
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right arm
        ])

        # Body / geom / site IDs.
        self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._pelvis_imu_site_id = self._mj_model.site("imu_in_pelvis").id

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in consts.FEET_SITES]
        )
        self._hands_site_id = np.array(
            [self._mj_model.site(name).id for name in consts.HAND_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
        )
        self._hand_thigh_geom_pairs = jp.array([
            [
                self._mj_model.geom("left_hand_collision").id,
                self._mj_model.geom("left_thigh").id,
            ],
            [
                self._mj_model.geom("right_hand_collision").id,
                self._mj_model.geom("right_thigh").id,
            ],
        ], dtype=jp.int32)
        hand_thigh_geom_pairs_np = np.array(self._hand_thigh_geom_pairs)
        self._hand_thigh_capsule_radius = jp.array(
            self._mj_model.geom_size[hand_thigh_geom_pairs_np, 0]
        )
        self._hand_thigh_capsule_half_length = jp.array(
            self._mj_model.geom_size[hand_thigh_geom_pairs_np, 1]
        )

        # Hip body indices for Raibert.
        self.hip_inds = jp.array(
            [self._mj_model.body(name).id for name in consts.HIP_NAMES]
        )

        # Foot geometry indices for feet_traj reward.
        self.feet_inds = jp.array(self._feet_geom_id, dtype=jp.int32)
        self._feet_geom_size = jp.array(self._mj_model.geom_size[self._feet_geom_id])
        self._foot_capsule_half_length = self._feet_geom_size[:, 0]
        self._foot_capsule_radius = jp.sqrt(
            self._feet_geom_size[:, 1] ** 2 + self._feet_geom_size[:, 2] ** 2
        )

        shin_geom_id = np.array(
            [
                self._mj_model.geom("left_shin").id,
                self._mj_model.geom("right_shin").id,
            ]
        )
        self._shin_geom_id = jp.array(shin_geom_id, dtype=jp.int32)
        self._shin_capsule_radius = jp.array(
            self._mj_model.geom_size[shin_geom_id, 0]
        )
        self._shin_capsule_half_length = jp.array(
            self._mj_model.geom_size[shin_geom_id, 1]
        )
        self._soft_collision_margin = float(
            getattr(self._config.rewards, "soft_collision_margin", 0.02)
        )
        self._soft_collision_temp = float(
            getattr(self._config.rewards, "soft_collision_temp", 0.01)
        )

        # Foot linvel sensor addresses.
        foot_linvel_sensor_adr = []
        for site in consts.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr, dtype=jp.int32)

        # Contact sensors.
        self._feet_floor_found_sensor = [
            self._mj_model.sensor(f"{foot_geom}_floor_found").id
            for foot_geom in ["left_foot", "right_foot"]
        ]
        self._right_foot_left_foot_found_sensor = self._mj_model.sensor(
            "right_foot_left_foot_found"
        ).id
        self._left_foot_right_shin_found_sensor = self._mj_model.sensor(
            "left_foot_right_shin_found"
        ).id
        self._right_foot_left_shin_found_sensor = self._mj_model.sensor(
            "right_foot_left_shin_found"
        ).id
        self._left_hand_left_thigh_found_sensor = self._mj_model.sensor(
            "left_hand_left_thigh_found"
        ).id
        self._right_hand_right_thigh_found_sensor = self._mj_model.sensor(
            "right_hand_right_thigh_found"
        ).id

        # Action scaling.
        self.action_scale = self._config.env.action_scale

        # Command config.
        self._cmd_a = jp.array(self._config.command_config.a)
        self._cmd_b = jp.array(self._config.command_config.b)

        # Gait parameters.
        self.step_k = self._config.env.step_k
        self.gait_scale = self._config.env.gait_scale
        self.gait_period = self.step_k * self.dt * 2.0  # full cycle time

        # Kinematic reference.
        kinematic_ref_qpos = g1_utils.make_kinematic_ref(
            cos_wave, self.step_k, scale=self.gait_scale, dt=self.dt
        )
        kinematic_ref_qvel = g1_utils.make_kinematic_ref(
            dcos_wave, self.step_k, scale=self.gait_scale, dt=self.dt
        )
        self.l_cycle = int(kinematic_ref_qpos.shape[0])

        kinematic_ref_qpos = np.array(kinematic_ref_qpos) + np.array(self._default_pose)
        ref_qs = np.tile(self._init_q.reshape(1, 36), (self.l_cycle, 1))
        ref_qs[:, 7:] = kinematic_ref_qpos
        self.kinematic_ref_qpos = jp.array(ref_qs)

        ref_qvels = np.zeros((self.l_cycle, 29))
        ref_qvels[:, :] = np.array(kinematic_ref_qvel)
        self.kinematic_ref_qvel = jp.array(ref_qvels)

        # Hip positions and foot offsets for Raibert heuristic.
        d = mjx_env.make_data(self.mj_model, qpos=self._init_q)
        d = mjx.forward(self.mjx_model, d)

        base_pos = d.xpos[self._torso_body_id]
        hip_pos = d.xpos[self.hip_inds]
        rel_pos = hip_pos - base_pos
        self.leg_offsets_x = rel_pos[:, 0]
        self.leg_offsets_y = rel_pos[:, 1]

        foot_pos = d.geom_xpos[self.feet_inds]
        base_quat = d.xquat[self._torso_body_id]
        hip_local = jax.vmap(rotate_inv, in_axes=(0, None))(hip_pos - base_pos, base_quat)
        foot_local = jax.vmap(rotate_inv, in_axes=(0, None))(foot_pos - base_pos, base_quat)
        foot_offset_local = foot_local - hip_local
        self.foot_offsets_xy = foot_offset_local[:, :2]

        self.base_mass = self._mj_model.body(consts.ROOT_BODY).mass
        self._nominal_base_height = self._init_q[2]

        # Step height parameters.
        self._step_height_max = float(self._config.env.step_height)
        self._step_height_min = float(self._config.env.step_height_min)
        self._foot_traj_vel_weight = float(self._config.env.foot_traj_vel_weight)

        # Initialize rewards.
        self._init_active_rewards(reward_lib)

    # ------------------------------------------------------------------
    # Swing mask and trajectory helpers (thin wrappers).
    # ------------------------------------------------------------------

    def _get_swing_mask(self, info):
        return g1_utils.get_swing_mask(self, info)

    def _update_foot_cycloid_ref(self, info):
        g1_utils.update_foot_cycloid_ref(self, info)

    def _update_raibert_target(self, data, info):
        g1_utils.update_raibert_target(self, data, info)

    # ------------------------------------------------------------------
    # Observastion context.
    # ------------------------------------------------------------------

    def _get_obs_context(self) -> Dict[str, Any]:
        return {
            "default_ap_pose": self._default_pose,
            "get_gyro": self.get_gyro,
            "get_local_linvel": self.get_local_linvel,
            "kin_ref_qpos": self.kinematic_ref_qpos,
            "l_cycle": self.l_cycle,
            "pelvis_imu_site_id": self._pelvis_imu_site_id,
        }

    # ------------------------------------------------------------------
    # Reset.
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # # Randomize x, y position.
        # rng, key = jax.random.split(rng)
        # dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        # qpos = qpos.at[0:2].set(qpos[0:2] + dxy)

        # # Randomize yaw.
        # rng, key = jax.random.split(rng)
        # yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        # quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        # new_quat = math.quat_mul(qpos[3:7], quat)
        # qpos = qpos.at[3:7].set(new_quat)

        # # Randomize joint positions: *U(0.5, 1.5).
        # rng, key = jax.random.split(rng)
        # qpos = qpos.at[7:].set(
        #     qpos[7:] * jax.random.uniform(key, (29,), minval=0.5, maxval=1.5)
        # )

        # # Randomize base velocity.
        # rng, key = jax.random.split(rng)
        # qvel = qvel.at[0:6].set(
        #     jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        # )

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=qpos[7:],
            impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        # Lift the root only for true penetration. Positive contact distances
        # can appear inside the contact margin and should not push the body down.
        min_dist = jp.where(
            data._impl.ncon > 0,
            jp.min(data._impl.contact.dist),
            0.0,
        )
        penetration = jp.minimum(min_dist, 0.0)
        qpos = qpos.at[2].set(qpos[2] - penetration)
        data = data.replace(qpos=qpos)
        data = mjx.forward(self.mjx_model, data)

        # Gait phase: freq ~ U(1.25, 1.5) Hz.
        rng, key = jax.random.split(rng)
        gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
        phase_dt = 2 * jp.pi * self.dt * gait_freq
        phase = jp.array([0, jp.pi])

        # Sample command.
        rng, cmd_rng = jax.random.split(rng)
        cmd = self.command_manager.sample(cmd_rng, cmd_a=self._cmd_a, cmd_b=self._cmd_b)

        # Determine initial stationary state.
        cmd_norm = jp.linalg.norm(cmd[:2])
        w_cmd = jp.abs(cmd[2])
        is_stationary = (cmd_norm < self._config.env.stationary_cmd_threshold) & (
            w_cmd < self._config.env.stationary_w_cmd_threshold
        )

        # Initialize Raibert / cycloid references in the same coordinate used by
        # the feet_traj reward: the foot geom center.
        feet_pos = data.geom_xpos[self.feet_inds]
        feet_xy = feet_pos[:, :2]
        feet_z = feet_pos[:, 2]

        state_info = {
            "rng": rng,
            "step": jp.array(0, dtype=jp.int32),
            "gait_step": jp.array(0, dtype=jp.int32),
            "is_stationary": is_stationary,
            "command": cmd,
            "last_action": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2),
            "swing_peak": jp.zeros(2),
            # Raibert / cycloid state.
            "xy0": feet_xy,
            "xy*": feet_xy,
            "k0": jp.array(0, dtype=jp.int32),
            "z0": feet_z,
            "foot_phase": 0.0,
            "foot_swing": jp.zeros(2),
            "foot_ref_xy": jp.zeros((2, 2)),
            "foot_ref_z": jp.zeros(2),
            "foot_ref_pos": feet_pos,
            "foot_ref_v_xy": jp.zeros((2, 2)),
            # Phase.
            "phase_dt": phase_dt,
            "phase": phase,
            "reward_tuple": {k: 0.0 for k in self.reward_manager.all_term_names},
        }

        # Initialize manager states.
        state_info = self.command_manager.init_state(
            state_info,
            command=cmd,
            steps_until_next_cmd=jp.array(500, dtype=jp.int32),
        )
        state_info = self.event_manager.init_state(
            state_info,
            disturbance_cfg=self._config.disturbance,
            dt=self.dt,
            prefix="disturbance",
        )
        state_info = self._sync_info(state_info)
        self._update_raibert_target(data, state_info)
        self._update_foot_cycloid_ref(state_info)

        metrics = {}
        for k in self._config.rewards.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        obs = self._build_obs(data, state_info, self._config.obs.policy_terms)
        return self._init_state(data, obs, state_info, metrics=metrics)

    # ------------------------------------------------------------------
    # Step.
    # ------------------------------------------------------------------

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # 1. Disturbance pipeline.
        state = self._run_event_pipeline(
            state,
            enabled=self._config.disturbance.enable,
            disturbance_cfg=self._config.disturbance,
            base_mass=self.base_mass,
            base_id=self._torso_body_id,
            prefix="disturbance",
        )
        info = state.info

        # 2. Action: ctrl = default_pose + action * action_scale.
        action = self._clip_action(action)
        ctrl = self._default_pose + action * self.action_scale
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        # 3. Physics step.
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

        # 4. Differentiable contact proxy from foot-bottom height.
        # G1 foot sites are at the ankle/foot frame, unlike Go2 where the foot
        # site is colocated with the contact geom.  Use the box geom bottom so
        # the sigmoid tracks floor contact while remaining differentiable.
        foot_bottom_z = self._get_foot_bottom_z(data)
        contact = jax.nn.sigmoid((0.005 - foot_bottom_z) * 100.0)
        delta_contact = jax.nn.relu(contact - info["last_contact"])
        first_contact = (info["feet_air_time"] > 0.0) * delta_contact
        info["feet_air_time"] += self.dt
        info["swing_peak"] = jp.maximum(info["swing_peak"], foot_bottom_z)

        # 6. Stationary detection and gait_step management.
        cmd_norm = jp.linalg.norm(info["command"][:2])
        w_cmd = jp.abs(info["command"][2])
        is_stationary = (
            cmd_norm < self._config.env.stationary_cmd_threshold
        ) & (w_cmd < self._config.env.stationary_w_cmd_threshold)
        info["is_stationary"] = is_stationary

        # When stationary: reset gait_step to 0.
        # When moving: increment gait_step.
        info["gait_step"] = jp.where(
            is_stationary,
            jp.array(0, dtype=jp.int32),
            info["gait_step"] + 1,
        )

        # 7. Update kinematic reference, Raibert targets, and cycloid ref.
        self._update_raibert_target(data, info)
        self._update_foot_cycloid_ref(info)

        # 8. Termination check.
        done = self._get_termination(data, info)

        # 9. Reward computation.
        reward_kwargs = self._get_reward_kwargs(data, action, info, contact, first_contact, done)
        reward_dict = self._get_reward(data, action, info, reward_kwargs, done)
        reward = self._sum_reward_dict(reward_dict, self.dt)

        # 10. Update info for next step.
        info["last_action"] = action
        info["feet_air_time"] *= (1.0 - contact)
        info["last_contact"] = contact
        info["swing_peak"] *= (1.0 - contact)
        info["step"] += 1

        # 11. Manager updates.
        info = self.command_manager.update(
            info, dt=self.dt, cmd_a=self._cmd_a, cmd_b=self._cmd_b
        )

        # 12. Metrics.
        info["reward_tuple"] = reward_dict
        for k, v in reward_dict.items():
            state.metrics[f"reward/{k}"] = v

        # 13. Build next observation.
        obs = self._build_obs(data, info, self._config.obs.policy_terms)

        done = done.astype(reward.dtype)
        return self._finalize_step(state, data=data, obs=obs, reward=reward, done=done, info=info)

    def _get_foot_bottom_z(self, data: mjx.Data) -> jax.Array:
        foot_xmat = data.geom_xmat[self.feet_inds]
        vertical_radius = jp.sum(
            jp.abs(foot_xmat[:, 2, :]) * self._feet_geom_size,
            axis=-1,
        )
        return data.geom_xpos[self.feet_inds, 2] - vertical_radius

    def _capsule_segments(
        self,
        data: mjx.Data,
        geom_ids: jax.Array,
        axis: int,
        half_lengths: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        centers = data.geom_xpos[geom_ids]
        axes = data.geom_xmat[geom_ids, :, axis]
        return (
            centers - axes * half_lengths[:, None],
            centers + axes * half_lengths[:, None],
        )

    def _segment_distance(
        self,
        p0: jax.Array,
        p1: jax.Array,
        q0: jax.Array,
        q1: jax.Array,
    ) -> jax.Array:
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

        closest_p = p0 + s[..., None] * u
        closest_q = q0 + t[..., None] * v
        return jp.sqrt(jp.sum(jp.square(closest_p - closest_q), axis=-1) + eps)

    def _get_soft_contact_done(self, data: mjx.Data) -> jax.Array:
        foot_p0, foot_p1 = self._capsule_segments(
            data, self.feet_inds, 0, self._foot_capsule_half_length
        )
        shin_p0, shin_p1 = self._capsule_segments(
            data, self._shin_geom_id, 2, self._shin_capsule_half_length
        )

        foot_foot_clearance = self._segment_distance(
            foot_p0[0], foot_p1[0], foot_p0[1], foot_p1[1]
        ) - (self._foot_capsule_radius[0] + self._foot_capsule_radius[1])
        left_foot_right_shin_clearance = self._segment_distance(
            foot_p0[0], foot_p1[0], shin_p0[1], shin_p1[1]
        ) - (self._foot_capsule_radius[0] + self._shin_capsule_radius[1])
        right_foot_left_shin_clearance = self._segment_distance(
            foot_p0[1], foot_p1[1], shin_p0[0], shin_p1[0]
        ) - (self._foot_capsule_radius[1] + self._shin_capsule_radius[0])

        clearances = jp.stack([
            foot_foot_clearance,
            left_foot_right_shin_clearance,
            right_foot_left_shin_clearance,
        ])
        return jp.max(
            jax.nn.sigmoid(
                (self._soft_collision_margin - clearances)
                / self._soft_collision_temp
            )
        )

    # ------------------------------------------------------------------
    # Termination.
    # ------------------------------------------------------------------

    def _get_termination(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        del info
        fall_termination = self.get_gravity(data, "torso")[-1] < 0.0

        contact_termination = data.sensordata[
            self._mj_model.sensor_adr[self._right_foot_left_foot_found_sensor]
        ] > 0
        contact_termination |= data.sensordata[
            self._mj_model.sensor_adr[self._left_foot_right_shin_found_sensor]
        ] > 0
        contact_termination |= data.sensordata[
            self._mj_model.sensor_adr[self._right_foot_left_shin_found_sensor]
        ] > 0

        return (
            fall_termination
            | contact_termination
            | jp.isnan(data.qpos).any()
            | jp.isnan(data.qvel).any()
        )

    # ------------------------------------------------------------------
    # Reward context (built in step, passed through to reward functions).
    # ------------------------------------------------------------------

    def _get_reward_kwargs(self, data, action, info, contact, first_contact, done):
        gravity_torso = self.get_gravity(data, "torso")
        soft_fall_done = jax.nn.sigmoid((0.0 - gravity_torso[-1]) * 100.0)
        soft_contact_done = self._get_soft_contact_done(data)
        move_mask = 1.0 - info["is_stationary"].astype(jp.float32)
        return {
            # Sensor accessors.
            "local_linvel": self.get_local_linvel(data, "pelvis"),
            "gyro": self.get_gyro(data, "pelvis"),
            "global_linvel_torso": self.get_global_linvel(data, "torso"),
            "global_linvel_pelvis": self.get_global_linvel(data, "pelvis"),
            "global_angvel_torso": self.get_global_angvel(data, "torso"),
            "global_angvel_pelvis": self.get_global_angvel(data, "pelvis"),
            "gravity_torso": gravity_torso,
            # Action.
            "current_action": action,
            # Joint groups.
            "default_pose": self._default_pose,
            "hip_indices": self._hip_indices,
            "knee_indices": self._knee_indices,
            "weights": self._weights,
            "soft_lowers": self._soft_lowers,
            "soft_uppers": self._soft_uppers,
            "nominal_base_height": self._nominal_base_height,
            "kinematic_ref_qpos": self.kinematic_ref_qpos,
            "l_cycle": self.l_cycle,
            "move_mask": move_mask,
            # Contact.
            "contact": contact,
            "first_contact": first_contact,
            "done": done,
            "soft_done": jp.maximum(soft_fall_done, soft_contact_done),
            # Feet.
            "feet_inds": self.feet_inds,
            "feet_site_id": self._feet_site_id,
            "foot_linvel_sensor_adr": self._foot_linvel_sensor_adr,
            # Smooth hand-thigh collision penalty geometry.
            "hand_thigh_geom_pairs": self._hand_thigh_geom_pairs,
            "hand_thigh_capsule_radius": self._hand_thigh_capsule_radius,
            "hand_thigh_capsule_half_length": self._hand_thigh_capsule_half_length,
            # Sensors for force.
            "mj_model": self.mj_model,
        }

    # ------------------------------------------------------------------
    # Visualization entry points.
    # ------------------------------------------------------------------

    def check_phase_alignment(self):
        g1_utils.check_phase_alignment(self)

    def check_gait_step_transition(self):
        g1_utils.check_gait_step_transition(self)

    def visualize_keyframes(self, keyframes="both", **kwargs):
        return g1_utils.visualize_keyframes(
            self, keyframes=keyframes, **kwargs
        )

    def visualize_keyframe(self, keyframe="knees_bent", **kwargs):
        return self.visualize_keyframes(keyframe, **kwargs)

    def play_cycloid_foot_trajectory(self, **kwargs):
        return g1_utils.play_cycloid_foot_trajectory(self, **kwargs)
