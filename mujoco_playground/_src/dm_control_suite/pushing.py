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
"""1D box pushing environment."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "pushing.xml"


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=256,
      action_repeat=1,
      impl="jax",
      nconmax=4,
      njmax=20,
      target_x=0.2,
      solimp=[0.015, 1.0, 0.031],
      solref=[0.02, 1.0],
  )


class PushBox(mjx_env.MjxEnv):
  """1D box pushing environment.

  The agent applies force to the ball to push the box toward a target position.
  Reward is based on the distance between the box and the target.
  """

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)

    self._xml_path = _XML_PATH.as_posix()
    self._model_assets = common.get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        _XML_PATH.read_text(), self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt

    # Apply solimp/solref from config to ball and box geoms before put_model.
    solimp = self._config.solimp
    solref = self._config.solref
    full_solimp = np.array([solimp[0], solimp[1], solimp[2], 0.5, 2.0])
    full_solref = np.array([solref[0], solref[1]])
    for geom_name in ("ball_geom", "box_geom"):
      geom_id = self._mj_model.geom(geom_name).id
      self._mj_model.geom_solimp[geom_id] = full_solimp
      self._mj_model.geom_solref[geom_id] = full_solref

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._target_x = self._config.target_x
    self._box_qpos_addr = self.mj_model.joint("box_x").qposadr[0]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, _ = jax.random.split(rng, 2)

    data = mjx_env.make_data(
        self.mj_model,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    metrics = {
        "reward/box_to_target": jp.zeros(()),
        "reward/ball_to_box": jp.zeros(()),
        "reward/distance_reward": jp.zeros(()),
        "reward/contact_reward": jp.zeros(()),
        "reward/action_penalty": jp.zeros(()),
        "reward/action_rate_penalty": jp.zeros(()),
        "reward/box_vel_penalty": jp.zeros(()),
    }
    info = {"rng": rng, "prev_action": jp.zeros(self.mjx_model.nu)}

    reward = jp.zeros(())
    done = jp.zeros(())
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    action = jp.clip(action, -1.0, 1.0)
    action = action * 0.4 + 0.4  # Rescale from [-1, 1] to [0, 0.8]
    data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
    reward = self._get_reward(data, action, state.info, state.metrics)
    obs = self._get_obs(data, state.info)
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    info = {**state.info, "prev_action": action}
    return mjx_env.State(data, obs, reward, done, state.metrics, info)

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info
    return jp.concatenate([
        data.qpos,
        data.qvel,
    ])

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    box_x = data.qpos[self._box_qpos_addr]
    ball_x = data.qpos[0]
    box_vel = data.qvel[self._box_qpos_addr]
    box_to_target = jp.abs(box_x - self._target_x)
    ball_to_box = jp.abs(box_x - ball_x - 0.2)
    distance_reward = -box_to_target
    contact_reward = -0.5 * ball_to_box
    action_penalty = -0.0001 * jp.sum(action**2)
    action_rate_penalty = -0.01 * jp.sum((action - info["prev_action"]) ** 2)
    box_vel_penalty = -0.1 * box_vel**2

    metrics["reward/box_to_target"] = box_to_target
    metrics["reward/ball_to_box"] = ball_to_box
    metrics["reward/distance_reward"] = distance_reward
    metrics["reward/contact_reward"] = contact_reward
    metrics["reward/action_penalty"] = action_penalty
    metrics["reward/action_rate_penalty"] = action_rate_penalty
    metrics["reward/box_vel_penalty"] = box_vel_penalty

    return distance_reward + contact_reward + action_penalty + action_rate_penalty + box_vel_penalty

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
