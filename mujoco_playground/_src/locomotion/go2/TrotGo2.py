from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import os
import mediapy as media
import tqdm
# Math
import jax.numpy as jp
import numpy as np
import jax

from ml_collections import config_dict

from mujoco_playground._src.locomotion.go2.Util.TrotUtil import (
    cos_wave, dcos_wave, make_kinematic_ref,
    quaternion_to_matrix, matrix_to_rotation_6d,
    quaternion_to_rotation_6d,
    rotate, rotate_inv
)

from mujoco_playground._src.locomotion.go2 import go2_constants as consts

# Sim
import mujoco
import mujoco.mjx as mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import make_data
from mujoco_playground._src.locomotion.go2.base import Go2Env
from mujoco_playground._src.locomotion.go2.configs import trot_config
from mujoco_playground._src.locomotion.go2.mdp import commands as command_lib
from mujoco_playground._src.locomotion.go2.mdp import event as event_lib
from mujoco_playground._src.locomotion.go2.mdp import rewards as reward_lib

def default_config() -> config_dict.ConfigDict:
    return trot_config.default_config()

# ----------------- Env -----------------
class TrotGo2(Go2Env):
    """
    MJX-based TrotGo2 environment.
    Signature required by locomotion: __init__(self, config, config_overrides=None)
    """

    def __init__(self,
                 task: str = None, 
                 config: config_dict.ConfigDict = trot_config.default_config(), 
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        default_xml = consts.MJX_XML_SENSOR_PATH.as_posix()
        super().__init__(
            xml_path=default_xml,
            config=config,
            config_overrides=config_overrides,
        )

        self._post_init()

    def _post_init(self):    
        # 基本姿态 / 初始 qpos
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos.copy())
        self._default_ap_pose = jp.array(self._mj_model.keyframe("home").qpos[7:].copy())

        # actions limits
        self.lowers, self.uppers = self.mj_model.jnt_range[1:].T

        # 动作中心与缩放（3 joints per leg）
        self.action_loc = jp.array(self._default_ap_pose)
        self.action_scale = jp.array(self._config.env.action_scale)

        # 其他参数
        self.termination_height = float(getattr(self._config.env, "termination_height", 0.1))
        self.err_threshold = self._config.env.err_threshold
        self.reward_config = self._config.rewards
        self.feet_inds = jp.array( [self._mj_model.geom(name).id for name in consts.FEET_GEOMS] )
        # print("Feet geom ids:", self.feet_inds)
        self.base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.base_mass = self.mj_model.body("base").mass

        # imitation reference
        step_k = int(getattr(self._config.env, "step_k", 25))
        gait_scale = float(getattr(self._config.env, "gait_scale", 0.3))
        kinematic_ref_qpos = make_kinematic_ref(cos_wave, step_k, scale=gait_scale, dt=self.dt)
        kinematic_ref_qvel = make_kinematic_ref(dcos_wave, step_k, scale=gait_scale, dt=self.dt)
        self.l_cycle = int(kinematic_ref_qpos.shape[0])

        kinematic_ref_qpos = np.array(kinematic_ref_qpos) + np.array(self._default_ap_pose)
        ref_qs = np.tile(self._init_q.reshape(1, 19), (self.l_cycle, 1))
        ref_qs[:, 7:] = kinematic_ref_qpos
        self.kinematic_ref_qpos = jp.array(ref_qs)

        ref_qvels = np.zeros((self.l_cycle, 18))
        ref_qvels[:, 6:] = np.array(kinematic_ref_qvel)
        self.kinematic_ref_qvel = jp.array(ref_qvels)

        self.reset2ref = self._config.env.reset2ref
        self.reference_state_init = self._config.env.reference_state_init

        self._init_active_rewards(reward_lib)

    # -------- Envs API: reset/step ----------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        # RSI
        if self.reference_state_init:
            rng, step_rng = jax.random.split(rng)
            init_step = jax.random.randint(step_rng, (), 0, self.l_cycle)
            qpos = self.kinematic_ref_qpos[init_step]
            qvel = self.kinematic_ref_qvel[init_step]
        else:
            # Deterministic init
            init_step = 0
            qpos = self._init_q
            qvel = jp.zeros(self.mjx_model.nv)

        # 创建 mjx data
        data = make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros(self.mjx_model.nu),
            impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        # 将机器人放到地面上（和原始 reset 一样）
        pen = jp.where(data._impl.ncon > 0, jp.min(data._impl.contact.dist), 0.0)
        qpos = qpos.at[2].set(qpos[2] - pen)

        data = make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros(self.mjx_model.nu),
            impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        # state_info 保持和原版一致
        state_info = {
            'rng': rng,
            'step': jp.array(init_step, dtype=jp.float32),
            'reward_tuple': {
                'reference_tracking': 0.0,
                'min_reference_tracking': 0.0,
                'feet_height': 0.0,
                'base_tracking': 0.0
            },
            'last_action': jp.zeros(self.mjx_model.nu),  # 12 通道动作
            'kinematic_ref': qpos,
            # Unified obs fields (match JoystickGo2 full obs layout)
            'anchor_action': jp.zeros(self.mjx_model.nu),
        }
        state_info = command_lib.init_command_state(state_info)
        state_info = event_lib.init_disturbance(
            state_info,
            disturbance_cfg=self._config.disturbance,
            dt=self.dt,
            prefix="disturbance",
        )

        # 生成 obs
        obs = self._get_obs(data, state_info)

        # 初始化 reward 和 metrics
        reward, done = jp.zeros(2)
        metrics = {}
        for k in state_info['reward_tuple']:
            metrics[k] = state_info['reward_tuple'][k]

        # 返回 mjx_env.State
        state = mjx_env.State(data, obs, reward, done, metrics, state_info)
        return jax.lax.stop_gradient(state)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # add disturbance
        if self._config.disturbance.enable:
            state = event_lib.maybe_apply_disturbance(
                state,
                disturbance_cfg=self._config.disturbance,
                dt=self.dt,
                base_mass=self.base_mass,
                nbody=self.mjx_model.nbody,
                base_id=self.base_id,
                prefix="disturbance",
            )
    
        action = jp.clip(action, -1, 1)
        ctrl = self.action_loc + (action * self.action_scale)

        data = mjx_env.step(
            self.mjx_model, state.data, ctrl, self.n_substeps
        )

        step_idx = jp.array(state.info["step"] % self.l_cycle, int)
        ref_qpos = self.kinematic_ref_qpos[step_idx]
        ref_qvel = self.kinematic_ref_qvel[step_idx]

        ref_data = data.replace(qpos=ref_qpos, qvel=ref_qvel)
        ref_data = mjx.forward(self.mjx_model, ref_data)

        state.info["kinematic_ref"] = ref_qpos

        obs = self._get_obs(data, state.info)

        # 结束条件
        base_z = data.xpos[self.base_id, 2]
        done = jp.where(base_z < self.termination_height, 1.0, 0.0)
        R_base = quaternion_to_matrix(data.xquat[1])
        up = jp.array([0.0, 0.0, 1.0])
        base_z_axis_world = R_base @ up
        done = jp.where(jp.dot(base_z_axis_world, up) < 0.0, 1.0, done)

        # 奖励
        reward_kwargs = {
            'ref_data': ref_data,
            'ref_qpos': ref_qpos,
            'ref_qvel': ref_qvel,
            'feet_inds': self.feet_inds,
            }
        reward_tuple = self._get_reward(data, ctrl, state.info, reward_kwargs, done)

        state.info["last_action"] = ctrl

        if self.reset2ref:
            err = (((data.xpos[1:] - ref_data.xpos[1:]) ** 2).sum(-1) ** 0.5).mean()
            to_ref = err > self.err_threshold

            reward_tuple['reference_tracking'] *= jax.numpy.where(to_ref, 10.0, 1.0)
            reward_tuple['base_tracking'] *= jax.numpy.where(to_ref, 10.0, 1.0)
            reward = sum(reward_tuple.values())
            for k in reward_tuple.keys():
                state.metrics[k] = reward_tuple[k]
            state.info["reward_tuple"] = reward_tuple

            def safe_select(a, b):
                return jp.where(to_ref, b, a)
            data_blend = jax.tree_util.tree_map(safe_select, data, ref_data)

            obs = self._get_obs(data_blend, state.info)
            state.info["step"] = state.info["step"] + 1.0

            return state.replace(data=data_blend, obs=obs, reward=reward, done=done)
        
        else:
            reward = sum(reward_tuple.values())
            state.info["reward_tuple"] = reward_tuple
            for k in reward_tuple.keys():
                state.metrics[k] = reward_tuple[k]

            state.info["step"] = state.info["step"] + 1.0

            return state.replace(data=data, obs=obs, reward=reward, done=done)
    
    # -------- Render reference motion (optimized) ----------
    def play_ref_motion(self, render_every: int = 2, seed: int = 0, save_path: str = None):
        """
        Play the built-in kinematic reference trajectory.
        """
        print("Playing reference motion (Optimized)...")

        # 1. Prepare data (Slicing array)
        ref_qpos = self.kinematic_ref_qpos[:self.l_cycle]

        # 2. Render
        # CHANGE: Call internal method, remove 'mj_model' and 'dt' (access via self)
        frames = self._render_trajectory(
            trajectory=ref_qpos,
            render_every=render_every,
            height=480,
            width=640,
            save_path=save_path
        )

        # 3. Display
        fps = 1.0 / (self.dt * render_every)
        media.show_video(frames, fps=fps, loop=True)

        return frames

    # -------- obs & reward helpers ----------
    def _get_obs_context(self) -> Dict[str, Any]:
        return {
            "default_ap_pose": self._default_ap_pose,
            "l_cycle": self.l_cycle,
            "kin_ref_qpos": self.kinematic_ref_qpos,
        }

    def _get_obs(self, data, state_info: Dict[str, Any]):
        return self._build_obs(data, state_info, self._config.obs.policy_terms)

    def _get_reward_context(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        extra_args: dict[str, Any],
    ) -> dict[str, Any]:
        del data, action, info
        return dict(extra_args)
