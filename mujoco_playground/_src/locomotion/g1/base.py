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
"""Base classes for G1."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import g1_constants as consts
from mujoco_playground._src.locomotion.go2 import managers as manager_lib
from mujoco_playground._src.locomotion.g1.mdp import observations as observation_lib
from mujoco_playground._src.locomotion.go2.Util.render_utils import render_trajectory


def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "assets")
    path = mjx_env.MENAGERIE_PATH / "unitree_g1"
    mjx_env.update_assets(assets, path, "*.xml")
    mjx_env.update_assets(assets, path / "assets")
    return assets


class G1Env(mjx_env.MjxEnv):
    """Base class for G1 environments."""

    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        self._model_assets = get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(
            epath.Path(xml_path).read_text(), assets=self._model_assets
        )
        self._mj_model.opt.timestep = self.sim_dt
        if hasattr(config, "env"):
            self._mj_model.opt.impratio = config.env.impratio
            self._mj_model.opt.iterations = config.env.iterations

        if self._config.restricted_joint_range:
            self._mj_model.jnt_range[1:] = consts.RESTRICTED_JOINT_RANGE
            self._mj_model.actuator_ctrlrange[:] = consts.RESTRICTED_JOINT_RANGE

        # Modify PD gains (only if config provides them).
        if hasattr(config, "Kd") and hasattr(config, "Kp"):
            self._mj_model.dof_damping[6:] = config.Kd
            self._mj_model.actuator_gainprm[:, 0] = config.Kp
            self._mj_model.actuator_biasprm[:, 1] = -config.Kp

        # Apply foot solimp/solref from config (only if config.env exists).
        if hasattr(config, "env") and hasattr(config.env, "solimp"):
            solimp = config.env.solimp
            solref = config.env.solref
            full_solimp = np.array([solimp[0], solimp[1], solimp[2], 0.5, 2.0])
            full_solref = np.array([solref[0], solref[1]])
            for foot_name in consts.FEET_GEOMS:
                geom_id = self._mj_model.geom(foot_name).id
                self._mj_model.geom_solimp[geom_id] = full_solimp
                self._mj_model.geom_solref[geom_id] = full_solref

        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = xml_path

        # Manager instantiation — only for the new config style (joystick2).
        if hasattr(config, "obs") and hasattr(config, "rewards"):
            self.observation_manager = manager_lib.ObservationManager(
                obs_module=observation_lib,
                context_fn=self._get_obs_context,
                noise_fn=self._apply_obs_noise,
            )
            self.action_manager = manager_lib.ActionManager()
            self.command_manager = manager_lib.CommandManager()
            self.event_manager = manager_lib.EventManager()
            self.termination_manager = manager_lib.TerminationManager()
            self.reward_manager = None
        else:
            self.observation_manager = None
            self.action_manager = None
            self.command_manager = None
            self.event_manager = None
            self.termination_manager = None
            self.reward_manager = None

    # Sensor readings.

    def get_gravity(self, data: mjx.Data, frame: str) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, f"{consts.GRAVITY_SENSOR}_{frame}"
        )

    def get_global_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, f"{consts.GLOBAL_LINVEL_SENSOR}_{frame}"
        )

    def get_global_angvel(self, data: mjx.Data, frame: str) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, f"{consts.GLOBAL_ANGVEL_SENSOR}_{frame}"
        )

    def get_local_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, f"{consts.LOCAL_LINVEL_SENSOR}_{frame}"
        )

    def get_accelerometer(self, data: mjx.Data, frame: str) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, f"{consts.ACCELEROMETER_SENSOR}_{frame}"
        )

    def get_gyro(self, data: mjx.Data, frame: str) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, f"{consts.GYRO_SENSOR}_{frame}"
        )

    def get_feet_pos(self, data: mjx.Data) -> jax.Array:
        return jp.vstack([
            mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
            for sensor_name in consts.FEET_POS_SENSOR
        ])

    # Accessors.

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # Manager cooperative methods.

    def _get_obs_context(self) -> Dict[str, Any]:
        return {}

    def _apply_obs_noise(
        self,
        info: Dict[str, Any],
        x: jax.Array,
        noise_name: Optional[str],
    ) -> jax.Array:
        if noise_name is None:
            return x
        x, info["rng"] = observation_lib.apply_uniform_noise(
            info["rng"],
            x,
            self._config.noise_config.level,
            self._config.noise_config.scales[noise_name],
        )
        return x

    def _build_obs(
        self,
        data: mjx.Data,
        info: Dict[str, Any],
        terms: Sequence[tuple[str, Optional[str], float]],
    ) -> Dict[str, jax.Array]:
        return self.observation_manager.build(data, info, terms)

    def _get_reward_context(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: Dict[str, Any],
        extra_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        del data, action, info
        return dict(extra_args)

    def _init_active_rewards(self, reward_lib: Any) -> None:
        self.reward_manager = manager_lib.RewardManager(
            reward_module=reward_lib,
            cfg=self._config,
            context_fn=self._get_reward_context,
        )
        self.active_rewards = {
            name: (scale, func)
            for name, scale, func in self.reward_manager.active_terms
        }

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: Dict[str, Any],
        extra_args: Dict[str, Any],
        done: jax.Array,
    ) -> Dict[str, jax.Array]:
        return self.reward_manager.compute(data, action, info, extra_args, done)

    def _sync_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        return manager_lib.sync_manager_state(info)

    def _init_state(
        self,
        data: mjx.Data,
        obs: Dict[str, jax.Array],
        info: Dict[str, Any],
        reward: Optional[jax.Array] = None,
        done: Optional[jax.Array] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> mjx_env.State:
        info = self._sync_info(info)
        if reward is None or done is None:
            reward, done = jp.zeros(2)
        if metrics is None:
            reward_tuple = info.get("reward_tuple", {})
            metrics = dict(reward_tuple)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _run_event_pipeline(
        self,
        state: mjx_env.State,
        *,
        enabled: bool,
        disturbance_cfg: Any,
        base_mass: float,
        base_id: int,
        prefix: str,
    ) -> mjx_env.State:
        if not enabled:
            return state
        return self.event_manager.maybe_apply(
            state,
            disturbance_cfg=disturbance_cfg,
            dt=self.dt,
            base_mass=base_mass,
            nbody=self.mjx_model.nbody,
            base_id=base_id,
            prefix=prefix,
        )

    def _clip_action(self, action: jax.Array, low: float = -1.0, high: float = 1.0) -> jax.Array:
        return self.action_manager.clip(action, low=low, high=high)

    def _sum_reward_dict(
        self, reward_dict: Dict[str, jax.Array], scale: float = 1.0
    ) -> jax.Array:
        return sum(reward_dict.values()) * scale

    def _update_metrics(
        self, metrics: Dict[str, Any], reward_dict: Dict[str, jax.Array]
    ) -> Dict[str, Any]:
        for k, v in reward_dict.items():
            metrics[k] = v
        return metrics

    def _finalize_step(
        self,
        state: mjx_env.State,
        *,
        data: mjx.Data,
        obs: Dict[str, jax.Array],
        reward: jax.Array,
        done: jax.Array,
        info: Dict[str, Any],
    ) -> mjx_env.State:
        info = self._sync_info(info)
        return state.replace(data=data, obs=obs, reward=reward, done=done, info=info)

    def _render_trajectory(
        self,
        trajectory: Union[List[Any], jax.Array, np.ndarray],
        render_every: int = 1,
        height: int = 480,
        width: int = 640,
        camera: Optional[str] = None,
        save_path: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[Sequence[Callable[[mujoco.MjvScene], None]]] = None,
    ) -> List[np.ndarray]:
        return render_trajectory(
            mj_model=self._mj_model,
            dt=self.dt,
            trajectory=trajectory,
            render_every=render_every,
            height=height,
            width=width,
            camera=camera,
            save_path=save_path,
            scene_option=scene_option,
            modify_scene_fns=modify_scene_fns,
        )
