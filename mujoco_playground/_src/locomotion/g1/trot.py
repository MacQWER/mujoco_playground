# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple in-place trot task for G1."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import joystick2
from mujoco_playground._src.locomotion.g1.configs import trot_config


def default_config() -> config_dict.ConfigDict:
    return trot_config.default_config()


class G1Trot(joystick2.G1Joystick2):
    """G1 fixed-command in-place stepping task.

    Observation and action layouts intentionally match G1Joystick2.
    """

    def __init__(
        self,
        task: Optional[str] = None,
        config: config_dict.ConfigDict = trot_config.default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            task=task,
            config=config,
            config_overrides=config_overrides,
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        if self._config.env.reference_state_init:
            rng, step_rng = jax.random.split(rng)
            init_step = jax.random.randint(step_rng, (), 0, self.l_cycle)
            qpos = self.kinematic_ref_qpos[init_step]
            qvel = jp.zeros(self.mjx_model.nv).at[6:].set(
                self.kinematic_ref_qvel[init_step]
            )
        else:
            init_step = jp.array(0, dtype=jp.int32)
            qpos = self._init_q
            qvel = jp.zeros(self.mjx_model.nv)

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

        phase_dt = jp.array(2.0 * jp.pi * self.dt / self.gait_period)
        phase = jp.array([0.0, jp.pi]) + phase_dt * init_step
        phase = jp.fmod(phase + jp.pi, 2.0 * jp.pi) - jp.pi

        feet_pos = data.geom_xpos[self.feet_inds]
        feet_xy = feet_pos[:, :2]
        feet_z = feet_pos[:, 2]
        k0 = (init_step // self.step_k) * self.step_k

        state_info = {
            "rng": rng,
            "step": jp.array(0, dtype=jp.int32),
            "gait_step": init_step,
            "is_stationary": jp.array(False),
            "command": jp.zeros(3),
            "last_action": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2),
            "swing_peak": jp.zeros(2),
            "xy0": feet_xy,
            "xy*": feet_xy,
            "k0": k0,
            "z0": feet_z,
            "foot_phase": 0.0,
            "foot_swing": jp.zeros(2),
            "foot_ref_xy": feet_xy,
            "foot_ref_z": feet_z,
            "foot_ref_pos": feet_pos,
            "foot_ref_v_xy": jp.zeros((2, 2)),
            "phase_dt": phase_dt,
            "phase": phase,
            "reward_tuple": {k: 0.0 for k in self.reward_manager.all_term_names},
        }

        self._update_raibert_target(data, state_info)
        self._update_foot_cycloid_ref(state_info)

        state_info = self.command_manager.init_state(
            state_info,
            command=jp.zeros(3),
            steps_until_next_cmd=jp.array(1_000_000, dtype=jp.int32),
        )
        state_info = self.event_manager.init_state(
            state_info,
            disturbance_cfg=self._config.disturbance,
            dt=self.dt,
            prefix="disturbance",
        )
        state_info = self._sync_info(state_info)

        metrics = {}
        for k in self._config.rewards.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        obs = self._build_obs(data, state_info, self._config.obs.policy_terms)
        state = self._init_state(data, obs, state_info, metrics=metrics)
        return jax.lax.stop_gradient(state)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = state.info

        state = self._run_event_pipeline(
            state,
            enabled=self._config.disturbance.enable,
            disturbance_cfg=self._config.disturbance,
            base_mass=self.base_mass,
            base_id=self._torso_body_id,
            prefix="disturbance",
        )
        info = state.info
        info["command"] = jp.zeros(3)
        info["is_stationary"] = jp.array(False)

        action = self._clip_action(action)
        ctrl = self._default_pose + action * self.action_scale
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

        foot_bottom_z = self._get_foot_bottom_z(data)
        contact = jax.nn.sigmoid((0.005 - foot_bottom_z) * 100.0)
        delta_contact = jax.nn.relu(contact - info["last_contact"])
        first_contact = (info["feet_air_time"] > 0.0) * delta_contact
        info["feet_air_time"] += self.dt
        info["swing_peak"] = jp.maximum(info["swing_peak"], foot_bottom_z)

        phase_tp1 = info["phase"] + info["phase_dt"]
        info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2.0 * jp.pi) - jp.pi
        info["gait_step"] += 1

        self._update_raibert_target(data, info)
        self._update_foot_cycloid_ref(info)

        done = self._get_termination(data, info)

        reward_kwargs = self._get_reward_kwargs(
            data, action, info, contact, first_contact, done
        )
        reward_dict = self._get_reward(data, action, info, reward_kwargs, done)
        reward = self._sum_reward_dict(reward_dict, self.dt)

        info["last_action"] = action
        info["feet_air_time"] *= 1.0 - contact
        info["last_contact"] = contact
        info["swing_peak"] *= 1.0 - contact
        info["step"] += 1
        info["reward_tuple"] = reward_dict

        for k, v in reward_dict.items():
            state.metrics[f"reward/{k}"] = v

        obs = self._build_obs(data, info, self._config.obs.policy_terms)
        done = done.astype(reward.dtype)
        return self._finalize_step(
            state, data=data, obs=obs, reward=reward, done=done, info=info
        )
