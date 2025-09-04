import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8" # 0.9 causes too much lag. 
os.environ['MUJOCO_GL'] = 'egl'

from typing import Any, Dict, Optional, Union

# Math
import jax.numpy as jp
import numpy as np
import jax
from jax import config # Analytical gradients work much better with double precision.
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
config.update('jax_default_matmul_precision', 'high')

from mujoco_playground._src.locomotion.anymal.TrotUtil import (
    cos_wave, dcos_wave, make_kinematic_ref,
    quaternion_to_matrix, matrix_to_rotation_6d,
    quaternion_to_rotation_6d,
    rotate, rotate_inv
)

# Sim
import mujoco
import mujoco.mjx as mjx
from mujoco_playground._src import mjx_env

try:
    from mujoco_playground._src.mjx_env import make_data
except ImportError:
    from typing import Optional
    import jax
    import mujoco
    from mujoco import mjx
    def make_data(
        model: mujoco.MjModel,
        qpos: Optional[jax.Array] = None,
        qvel: Optional[jax.Array] = None,
        ctrl: Optional[jax.Array] = None,
        act: Optional[jax.Array] = None,
        mocap_pos: Optional[jax.Array] = None,
        mocap_quat: Optional[jax.Array] = None,
        impl: Optional[str] = None,
        nconmax: Optional[int] = None,
        njmax: Optional[int] = None,
        device: Optional[jax.Device] = None, # type: ignore
    ) -> mjx.Data:
        """Initialize MJX Data."""
        data = mjx.make_data(
            model, impl=impl, nconmax=nconmax, njmax=njmax, device=device
        )
        if qpos is not None:
            data = data.replace(qpos=qpos)
        if qvel is not None:
            data = data.replace(qvel=qvel)
        if ctrl is not None:
            data = data.replace(ctrl=ctrl)
        if act is not None:
            data = data.replace(act=act)
        if mocap_pos is not None:
            data = data.replace(mocap_pos=mocap_pos.reshape(model.nmocap, -1))
        if mocap_quat is not None:
            data = data.replace(mocap_quat=mocap_quat.reshape(model.nmocap, -1))
        return data

from mujoco_playground._src.locomotion.anymal.base import AnymalEnv

# Supporting
from ml_collections import config_dict
from typing import Any, Dict


# ----------------- default config -----------------
def default_config() -> config_dict.ConfigDict:
    # 注意：MjxEnv 要求 config 里必须包含 sim_dt 和 ctrl_dt
    cfg = config_dict.ConfigDict()
    cfg.Kp = 230.0          # PD 控制器的比例增益
    cfg.sim_dt = 0.002          # 物理仿真步长（s）
    cfg.ctrl_dt = 0.02          # 控制步长（s） => n_frames = ctrl_dt / sim_dt = 10
    # 环境超参
    cfg.env = config_dict.ConfigDict()
    cfg.env.termination_height = 0.25
    cfg.env.step_k = 13         # 每条腿抬起/落下的子步数量
    cfg.env.err_threshold = 0.4
    cfg.env.action_scale = [0.2, 0.8, 0.8] * 4  # 每条腿3个关节，共4条腿
    # 奖励权重
    cfg.rewards = config_dict.ConfigDict()
    cfg.rewards.scales = config_dict.ConfigDict()
    cfg.rewards.scales.min_reference_tracking = -2.5 * 3e-3
    cfg.rewards.scales.reference_tracking = -1.0
    cfg.rewards.scales.feet_height = -1.0
    # 其他
    cfg.impl = "jax"
    cfg.nconmax = 4 * 8192
    cfg.njmax = 40
    return cfg


# ----------------- Env -----------------
class TrotAnymal(AnymalEnv):
    """
    MJX-based TrotAnymal environment.
    Signature required by locomotion: __init__(self, config, config_overrides=None)
    """

    def __init__(self,
                 task: str = None, 
                 config: config_dict.ConfigDict = default_config(), 
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        default_xml = os.path.normpath(
            os.path.join(CURRENT_DIR, "xmls", "scene_mjx.xml")
        )

        super().__init__(
            xml_path=default_xml,
            config=config,
            config_overrides=config_overrides,
        )

        self._post_init()

    def _post_init(self):    
        # 基本姿态 / 初始 qpos
        self._init_q = jp.array(self._mj_model.keyframe("standing").qpos.copy())
        self._default_ap_pose = jp.array(self._mj_model.keyframe("standing").qpos[7:].copy())

        # actions limits
        self.lowers, self.uppers = self.mj_model.jnt_range[1:].T

        # 动作中心与缩放（3 joints per leg）
        self.action_loc = jp.array(self._default_ap_pose)
        self.action_scale = jp.array(self._config.env.action_scale)

        # 其他参数
        self.termination_height = float(getattr(self._config.env, "termination_height", 0.25))
        self.err_threshold = 0.4
        self.reward_config = self._config.rewards
        self.feet_inds = jp.array([21, 28, 35, 42])  # LF, RF, LH, RH
        self.base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")

        # imitation reference
        step_k = int(getattr(self._config.env, "step_k", 25))
        kinematic_ref_qpos = make_kinematic_ref(cos_wave, step_k, scale=0.3, dt=self.dt)
        kinematic_ref_qvel = make_kinematic_ref(dcos_wave, step_k, scale=0.3, dt=self.dt)
        self.l_cycle = int(kinematic_ref_qpos.shape[0])

        kinematic_ref_qpos = np.array(kinematic_ref_qpos) + np.array(self._default_ap_pose)
        ref_qs = np.tile(self._init_q.reshape(1, 19), (self.l_cycle, 1))
        ref_qs[:, 7:] = kinematic_ref_qpos
        self.kinematic_ref_qpos = jp.array(ref_qs)

        ref_qvels = np.zeros((self.l_cycle, 18))
        ref_qvels[:, 6:] = np.array(kinematic_ref_qvel)
        self.kinematic_ref_qvel = jp.array(ref_qvels)

    # -------- Envs API: reset/step ----------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Deterministic init
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
        pen = jp.min(data._impl.contact.dist)
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
            'steps': 0.0,
            'reward_tuple': {
                'reference_tracking': 0.0,
                'min_reference_tracking': 0.0,
                'feet_height': 0.0
            },
            'last_action': jp.zeros(self.mjx_model.nu),  # 12 通道动作
            'kinematic_ref': jp.zeros(19),
        }

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
        # TODO 扰动
        # if self._config.pert_config.enable:
        #     state = self._maybe_apply_perturbation(state)
    
        action = jp.clip(action, -1, 1)
        ctrl = self.action_loc + (action * self.action_scale)

        data = mjx_env.step(
            self.mjx_model, state.data, ctrl, self.n_substeps
        )

        step_idx = jp.array(state.info["steps"] % self.l_cycle, int)
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
        reward_tuple = dict(
            reference_tracking=self._reward_reference_tracking(data, ref_data) * self.reward_config.scales.reference_tracking,
            min_reference_tracking=self._reward_min_reference_tracking(ref_qpos, ref_qvel, data)
            * self.reward_config.scales.min_reference_tracking,
            feet_height=self._reward_feet_height(
                data.geom_xpos[self.feet_inds][:, 2], ref_data.geom_xpos[self.feet_inds][:, 2]
            )
            * self.reward_config.scales.feet_height,
        )
        reward = sum(reward_tuple.values())

        state.info["reward_tuple"] = reward_tuple
        state.info["last_action"] = ctrl
        for k in reward_tuple.keys():
            state.metrics[k] = reward_tuple[k]

        err = (((data.xpos[1:] - ref_data.xpos[1:]) ** 2).sum(-1) ** 0.5).mean()
        # to_ref = jp.where(err > self.err_threshold, 1.0, 0.0)
        # data_blend = jax.tree_util.tree_map(lambda a, b: (1 - to_ref) * a + to_ref * b, data, ref_data)
        to_ref = err > self.err_threshold
        def safe_select(a, b):
            return jp.where(to_ref, b, a)
        data_blend = jax.tree_util.tree_map(safe_select, data, ref_data)

        obs = self._get_obs(data_blend, state.info)
        state.info["steps"] = state.info["steps"] + 1.0

        return state.replace(data=data_blend, obs=obs, reward=reward, done=done)

    # -------- obs & reward helpers ----------
    def _get_obs(self, data, state_info: Dict[str, Any]):
        local_omega = data.cvel[1, :3]
        yaw_rate = local_omega[2]
        g_world = jp.array([0.0, 0.0, -1.0])
        g_local = rotate_inv(g_world, data.xquat[1])
        angles = data.qpos[7:19]
        last_action = state_info["last_action"]
        step_idx = jp.array(state_info["steps"] % self.l_cycle, int)
        kin_ref = self.kinematic_ref_qpos[step_idx][7:]
        obs_list = [jp.array([yaw_rate]) * 0.25, g_local, angles - jp.array(self._default_ap_pose), last_action, kin_ref]
        obs = jp.clip(jp.concatenate(obs_list), -100.0, 100.0)
        return obs

    def _reward_reference_tracking(self, data, ref_data):
        f = lambda a, b: ((a - b) ** 2).sum(-1).mean()
        mse_pos = f(data.xpos[1:], ref_data.xpos[1:])
        mse_rot = f(quaternion_to_rotation_6d(data.xquat[1:]), quaternion_to_rotation_6d(ref_data.xquat[1:]))
        vel = data.cvel[1:, 3:]
        ang = data.cvel[1:, :3]
        ref_vel = ref_data.cvel[1:, 3:]
        ref_ang = ref_data.cvel[1:, :3]
        mse_vel = f(vel, ref_vel)
        mse_ang = f(ang, ref_ang)
        return mse_pos + 0.1 * mse_rot + 0.01 * mse_vel + 0.001 * mse_ang

    def _reward_min_reference_tracking(self, ref_qpos, ref_qvel, data):
        pos = jp.concatenate([data.qpos[:3], data.qpos[7:]])
        pos_targ = jp.concatenate([ref_qpos[:3], ref_qpos[7:]])
        pos_err = jp.linalg.norm(pos_targ - pos)
        vel_err = jp.linalg.norm(data.qvel - ref_qvel)
        return pos_err + vel_err

    def _reward_feet_height(self, feet_z, feet_z_ref):
        return jp.sum(jp.abs(feet_z - feet_z_ref))

# # ----------------- 注册到 playground -----------------
# locomotion.register_environment(
#     'TrotAnymal',   # 环境名字
#     TrotAnymal,     # 环境类
#     default_config      # 默认配置函数
# )

