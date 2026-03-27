from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

from mujoco_playground._src.locomotion.go2 import go2_constants as consts
from mujoco_playground._src.locomotion.go2 import managers as manager_lib
from mujoco_playground._src.locomotion.go2.mdp import observations as observation_lib
from mujoco_playground._src.locomotion.go2.Util.render_utils import render_trajectory

class Go2Env(mjx_env.MjxEnv):
    """Base class for Go2 environments."""

    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        self._xml_path = xml_path
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = config.sim_dt
        self._mj_model.opt.impratio = config.env.impratio
        self._mj_model.opt.iterations = config.env.iterations
        # Modify PD gains.
        self._mj_model.dof_damping[6:] = config.Kd
        self._mj_model.actuator_gainprm[:, 0] = config.Kp
        self._mj_model.actuator_biasprm[:, 1] = -config.Kp

        # Increase offscreen framebuffer size to render at higher resolutions.
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        self._imu_site_id = self._mj_model.site("imu").id
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
    
    # Sensor readings.

    def get_upvector(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.UPVECTOR_SENSOR)

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
        self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR
    )

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR
        )

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.LOCAL_LINVEL_SENSOR
        )

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.ACCELEROMETER_SENSOR
    )

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)
    
    def get_feet_pos(self, data: mjx.Data) -> jax.Array:
        return jp.vstack([
            mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
            for sensor_name in consts.FEET_POS_SENSOR
        ])
    
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
