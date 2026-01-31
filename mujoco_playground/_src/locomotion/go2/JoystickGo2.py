from typing import Any, Dict, Optional, Union, Callable
import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from ml_collections import config_dict

from brax.io import model

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2.base import Go2Env
from mujoco_playground._src.locomotion.go2 import go2_constants as consts
from mujoco_playground._src.locomotion.go2.TrotUtil import (
    make_kinematic_ref, cos_wave, rotate_inv, rotate, get_anchor_inference_fn
)

# ----------------- Default Config -----------------
def default_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    
    # 1. 物理参数 (CRITICAL: 必须严格对齐 Anchor Policy 的训练设置)
    # 你提到 Baseline 是用 Kp=35, Kd=0.5 训练的
    cfg.Kp = 35.0          
    cfg.Kd = 0.5           
    cfg.sim_dt = 0.002
    cfg.ctrl_dt = 0.02
    cfg.episode_length = 240  # 5s
    
    # 2. 环境参数
    cfg.env = config_dict.ConfigDict()
    # [CRITICAL] 必须和 TrotGo2 一致，否则 Anchor 输出的动作幅度不对
    cfg.env.action_scale = [0.2, 0.8, 0.8] * 4 
    cfg.env.termination_height = 0.1
    cfg.env.step_k = 13    # 保持和 TrotGo2 一致
    cfg.env.impratio = 100
    cfg.env.iterations = 1
    
    # 3. 指令配置 (Joystick)
    cfg.command_config = config_dict.ConfigDict()
    cfg.command_config.a = [1.5, 0.80, 1.2]  # [Vx_max, Vy_max, Wz_max]
    cfg.command_config.b = [0.9, 0.25, 0.5]  # Probability of keeping non-zero
    
    # 4. 奖励权重
    cfg.rewards = config_dict.ConfigDict()
    cfg.rewards.scales = config_dict.ConfigDict()
    
    # Task: Velocity Tracking (Joystick)
    cfg.rewards.scales.tracking_lin_vel = 1.5
    cfg.rewards.scales.tracking_ang_vel = 0.8
    
    # Constraints: APG Guidance (from FwdTrot)
    cfg.rewards.scales.feet_pos = -1.0     # Raibert Heuristic (Cost)
    cfg.rewards.scales.feet_height = -1.0  # Sine Wave Height (Cost)
    
    # Regularization / Survival
    cfg.rewards.scales.lin_vel_z = -1.0
    cfg.rewards.scales.orientation = -1.0
    cfg.rewards.scales.torques = -0.001
    
    cfg.rewards.tracking_sigma = 0.25

    # 5. Anchor (原 Baseline) 配置
    cfg.anchor = config_dict.ConfigDict()
    cfg.anchor.path = consts.ANCHOR_PATH  # Anchor Policy 路径
    
    cfg.impl = "jax"
    cfg.nconmax = 4 * 8192
    cfg.njmax = 40
    return cfg


class JoystickGo2(Go2Env):
    """
    JoystickGo2 with Residual Learning support.
    Uses an 'Anchor Policy' (trained on TrotGo2) to provide a base gait,
    and learns a residual policy to achieve omnidirectional velocity tracking.
    """
    
    def __init__(self,
                 task: str = None, 
                 config: config_dict.ConfigDict = default_config(), 
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,):
        
        super().__init__(
            xml_path=consts.MJX_XML_PATH.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )

        self._post_init()

    def _post_init(self):
        # 加载 Anchor Policy
        if not self._config.anchor.path:
            self._anchor_inference_fn = None
        else:
            self._anchor_inference_fn = get_anchor_inference_fn(
                self._config.anchor.path
            )
            
        # 基础姿态
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos.copy())
        self._default_ap_pose = jp.array(self._mj_model.keyframe("home").qpos[7:].copy())
        
        # 动作空间 (和 TrotGo2 对齐)
        self.action_loc = jp.array(self._default_ap_pose)
        self.action_scale = jp.array(self._config.env.action_scale) 
        
        # 身体 ID
        self.base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.feet_inds = jp.array([self._mj_model.geom(name).id for name in consts.FEET_GEOMS])
        
        # 步态参数
        step_k = int(getattr(self._config.env, "step_k", 25))
        self.step_k = step_k
        self.gait_period = step_k * 2 * self.dt
        
        # 预计算参考轨迹 (用于欺骗 Anchor Policy，让它以为在做 Trot 任务)
        kinematic_ref_qpos = make_kinematic_ref(cos_wave, step_k, scale=0.3, dt=self.dt)
        self.l_cycle = int(kinematic_ref_qpos.shape[0])

        kinematic_ref_qpos = np.array(kinematic_ref_qpos) + np.array(self._default_ap_pose)
        ref_qs = np.tile(self._init_q.reshape(1, 19), (self.l_cycle, 1))
        ref_qs[:, 7:] = kinematic_ref_qpos
        self.kinematic_ref_qpos = jp.array(ref_qs)
        
        # Command 参数
        self._cmd_a = jp.array(self._config.command_config.a)
        self._cmd_b = jp.array(self._config.command_config.b)

    # -------- Reset --------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, key_cmd = jax.random.split(rng)
        
        # 1. 物理 Reset
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)
        
        # (可选) 稍微随机化初始 yaw 或 位置
        # rng, key_yaw = jax.random.split(rng)
        # qpos = qpos.at[3:7].set(...) 

        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=jp.zeros(12),
                                 impl=self.mjx_model.impl.value, 
                                 nconmax=self._config.nconmax, njmax=self._config.njmax)
        data = mjx.forward(self.mjx_model, data)
        
        # 落地处理 (避免一开始就穿模摔倒)
        pen = jp.where(data._impl.ncon > 0, jp.min(data._impl.contact.dist), 0.0)
        data = data.replace(qpos=qpos.at[2].set(qpos[2] - pen))
        data = mjx.forward(self.mjx_model, data)

        # 2. 初始化 Command
        cmd = self.sample_command(key_cmd, jp.zeros(3))
        
        # 3. 初始化 State Info
        state_info = {
            'rng': rng,
            'step': 0.0,
            'command': cmd,
            'steps_until_next_cmd': 100,
            'last_action': self.action_loc, # 这是物理层面的 Normalized Action
            
            # Raibert & Gait 变量 (用于 Reward)
            'xy0': data.geom_xpos[self.feet_inds][:, :2],
            'xy*': data.geom_xpos[self.feet_inds][:, :2],
            'k0': 0.0,
            
            # Anchor 变量
            'anchor_action': jp.zeros(12),
            
            # Metrics
            'reward_tuple': {k: 0.0 for k in self._config.rewards.scales.keys()}
        }

        # 4. Anchor Policy Inference (初始帧)
        # 获取 Anchor 专用的 Obs (模仿 TrotGo2)
        anchor_obs = self._get_anchor_obs(data, state_info)
        rng, key_anchor = jax.random.split(rng)
        
        # 执行策略
        anchor_act, _ = self._anchor_inference_fn(anchor_obs, key_anchor)
        
        # [CRITICAL] 冻结 Anchor 参数，阻断梯度
        anchor_act = jax.lax.stop_gradient(anchor_act)
        
        state_info['anchor_action'] = anchor_act
        state_info['rng'] = rng

        # 5. 获取 Residual Policy 的 Obs
        residual_obs = self._get_residual_obs(data, state_info)
        
        reward, done = jp.zeros(2)
        metrics = state_info['reward_tuple'].copy()
        
        return mjx_env.State(data, {'state': residual_obs}, reward, done, metrics, state_info)

    # -------- Step --------
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # action: 这里是 Residual Policy 输出的 "残差"
        
        # 1. 动作混合
        anchor_act = state.info['anchor_action'] # 已经在上一步 stop_gradient
        mixed_action = anchor_act + action
        
        # Clip 并应用 Scale (Scale 必须是 TrotGo2 的 Scale)
        mixed_action = jp.clip(mixed_action, -1, 1)
        ctrl = self.action_loc + (mixed_action * self.action_scale)
        
        # 2. 物理步进
        data_next = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)
        data = data_next.replace(qacc=jax.lax.stop_gradient(data_next.qacc), 
                                 qfrc_constraint=jax.lax.stop_gradient(data_next.qfrc_constraint))

        # 3. Command 更新逻辑 (Timer based)
        state.info['rng'], key_cmd, key_time = jax.random.split(state.info['rng'], 3)
        state.info['steps_until_next_cmd'] -= 1
        should_update = state.info['steps_until_next_cmd'] <= 0
        
        new_cmd = self.sample_command(key_cmd, state.info['command'])
        # 指令持续时间随机化 (0.5s ~ 2.5s 左右)
        new_timer = jp.round(jax.random.exponential(key_time) * 5.0 / self.dt).astype(jp.int32)
        
        state.info['command'] = jp.where(should_update, new_cmd, state.info['command'])
        state.info['steps_until_next_cmd'] = jp.where(should_update, new_timer, state.info['steps_until_next_cmd'])

        # 4. 更新 Raibert Target (用于 Reward 计算)
        self._update_raibert_target(data, state.info)
        
        state.info['step'] += 1.0
        state.info['last_action'] = ctrl # 记录物理 ctrl 用于下一帧 obs

        # 5. 计算下一帧的 Anchor Action
        anchor_obs = self._get_anchor_obs(data, state.info)
        
        # Inference
        next_anchor_act, _ = self._anchor_inference_fn(anchor_obs, key_cmd) # key reused
        
        # [CRITICAL] 再次阻断梯度
        next_anchor_act = jax.lax.stop_gradient(next_anchor_act)
        
        state.info['anchor_action'] = next_anchor_act

        # 6. 计算奖励
        reward_tuple = self._compute_rewards(data, state.info)
        reward = sum(reward_tuple.values())
        
        # 更新 Metrics
        for k, v in reward_tuple.items(): 
            state.metrics[k] = v
        state.info['reward_tuple'] = reward_tuple
        
        # 7. 终止条件 check
        base_z = data.xpos[self.base_id, 2]
        done = jp.where(base_z < self._config.env.termination_height, 1.0, 0.0)

        # 8. 获取下一帧 Residual Obs
        residual_obs = self._get_residual_obs(data, state.info)
        
        return state.replace(data=data, obs={'state': residual_obs}, reward=reward, done=done)

    # -------- Observation Generators --------
    
    def _get_anchor_obs(self, data, info):
        """
        生成 Anchor Policy 需要的 Obs。
        必须严格模仿 TrotGo2 的 obs 结构：
        [yaw_rate(1), g_local(3), joints(12), last_action(12), kin_ref(12)] = 40 dims
        """
        q = data.xquat[1]
        local_omega = data.cvel[1, :3]
        yaw_rate = rotate_inv(local_omega, q)[2]
        
        g_local = rotate_inv(jp.array([0., 0., -1.]), q)
        angles = data.qpos[7:19]
        
        # 注意: TrotGo2 使用的是上一帧输出的 raw action (scaled 之前还是之后? 通常是 normalized)
        # 但在 TrotGo2 代码中: obs_list.append(last_action) 且 state.info["last_action"] = ctrl
        # 如果 TrotGo2 的 last_action 是 ctrl (物理值)，这里也要用 ctrl。
        # 根据提供的 TrotGo2.py: state.info["last_action"] = ctrl (Line 183)
        last_action = info['last_action'] 
        
        # 生成 Reference (相位信息)
        step_idx = jp.array(info['step'] % self.l_cycle, int)
        kin_ref = self.kinematic_ref_qpos[step_idx][7:]
        
        obs_list = [
            jp.array([yaw_rate]) * 0.25, # TrotGo2 的 scaling
            g_local,
            angles - self._default_ap_pose,
            last_action, 
            kin_ref
        ]
        obs_vec = jp.clip(jp.concatenate(obs_list), -100.0, 100.0)

        return {"state": obs_vec}

    def _get_residual_obs(self, data, info):
        """
        生成 Residual Policy (Agent) 的 Obs。
        包含了 Command 和 Anchor Action。
        """
        q = data.xquat[1]
        v_local = rotate_inv(data.cvel[1, 3:], q)
        w_local = rotate_inv(data.cvel[1, :3], q)
        g_local = rotate_inv(jp.array([0., 0., -1.]), q)
        
        obs_list = [
            v_local,          # 3
            w_local,          # 3
            g_local,          # 3
            info['command'],  # 3 (Vx, Vy, Wz)
            data.qpos[7:19] - self._default_ap_pose, # 12
            data.qvel[6:],    # 12
            info['anchor_action'], # 12 (让 Agent 知道 Anchor 想做什么)
        ]
        return jp.clip(jp.concatenate(obs_list), -100.0, 100.0)

    # -------- Helpers --------

    def _update_raibert_target(self, data, info):
        """
        计算 Raibert Heuristic 落点，用于构造 APG 的 Dense Reward。
        """
        s = info['step']
        step_k = self.step_k
        new_step = (s % step_k == 0)
        even_step = ((s // step_k) % 2 == 0)
        
        # 1. 计算当前 Command 下的理论落点
        v_cmd_local = info['command'][:2]
        quat = data.xquat[1]
        # 将局部指令速度转到世界系，因为 feet_pos 是世界系
        v_cmd_global = rotate(jp.concatenate([v_cmd_local, jp.array([0.])]), quat)[:2]
        
        step_period = self.gait_period / 2
        feet_pos = data.geom_xpos[self.feet_inds][:, :2]
        raibert_xy = feet_pos + (step_period / 2) * v_cmd_global
        
        # 2. 根据步态相位更新目标
        # Go2 Indices: 0:FL, 1:FR, 2:RL, 3:RR
        # Trot Pairs: (0, 3) 和 (1, 2)
        pair1 = jp.array([0, 3])
        pair2 = jp.array([1, 2])
        
        cur_tars = info['xy*']
        tars_p1 = cur_tars.at[pair1].set(raibert_xy[pair1])
        tars_p2 = cur_tars.at[pair2].set(raibert_xy[pair2])
        
        # 如果是新的一步，且是偶数步 -> 更新 Pair1
        xy_tars = jp.where(new_step & even_step, tars_p1, cur_tars)
        # 如果是新的一步，且是奇数步 -> 更新 Pair2
        xy_tars = jp.where(new_step & (~even_step), tars_p2, xy_tars)
        
        info['xy*'] = xy_tars
        info['xy0'] = jp.where(new_step, feet_pos, info['xy0'])
        info['k0'] = jp.where(new_step, s, info['k0'])

    def sample_command(self, rng, old_cmd):
        """
        采样新的速度指令，包含 Masking 逻辑。
        """
        rng, key1, key2 = jax.random.split(rng, 3)
        new_cmd = jax.random.uniform(key1, (3,), minval=-self._cmd_a, maxval=self._cmd_a)
        mask = jax.random.bernoulli(key2, self._cmd_b, (3,))
        return new_cmd * mask

    def _compute_rewards(self, data, info):
        """
        计算奖励。结合了 Tracking (Joystick) 和 Geometry (APG)。
        """
        # 1. Raibert Position Cost (Dense Gradient for APG)
        dt_step = (info['step'] - info['k0']) * self.dt
        step_period = self.gait_period / 2
        # 插值比例 0 -> 1
        ratio = jp.clip(dt_step / step_period, 0.0, 1.0)
        # 当前时刻的期望足端位置
        xyt = info['xy0'] + (info['xy*'] - info['xy0']) * ratio
        
        curr_feet = data.geom_xpos[self.feet_inds][:, :2]
        r_feet_pos = jp.sum(jp.square(curr_feet - xyt)) # MSE Cost
        
        # 2. Feet Height Cost (Sine Wave)
        t = info['step'] * self.dt
        phase = (2 * jp.pi / self.gait_period) * t
        # Trot Phase Logic:
        sin1 = jp.sin(phase)
        sin2 = jp.sin(phase - jp.pi)
        h_tar = 0.08 # Max swing height
        h_ref1 = jp.clip(sin1, 0, 1) * h_tar + 0.02
        h_ref2 = jp.clip(sin2, 0, 1) * h_tar + 0.02
        # Pair1 (0,3) match h_ref2? 
        # 注意: TrotGo2 的 ref 逻辑比较复杂，这里简化为两组对角线
        # 0:FL, 1:FR, 2:RL, 3:RR
        h_tars = jp.array([h_ref1, h_ref2, h_ref2, h_ref1]) 
        curr_feet_z = data.geom_xpos[self.feet_inds][:, 2]
        r_feet_height = jp.sum(jp.square(curr_feet_z - h_tars))

        # 3. Velocity Tracking
        q = data.xquat[1]
        v_local = rotate_inv(data.cvel[1, 3:], q)
        w_local = rotate_inv(data.cvel[1, :3], q)
        
        r_lin_vel = jp.exp(-jp.sum(jp.square(info['command'][:2] - v_local[:2])) / 0.25)
        r_ang_vel = jp.exp(-jp.square(info['command'][2] - w_local[2]) / 0.25)
        
        # 4. Regularization
        r_lin_vel_z = jp.square(data.cvel[1, 5]) # global z vel
        g_local = rotate_inv(jp.array([0., 0., -1.]), q)
        r_ori = jp.sum(jp.square(g_local[:2])) # projected gravity xy
        r_torques = jp.sum(jp.square(data.qfrc_actuator[6:]))
        
        scales = self._config.rewards.scales
        return {
            'tracking_lin_vel': r_lin_vel * scales.tracking_lin_vel,
            'tracking_ang_vel': r_ang_vel * scales.tracking_ang_vel,
            'feet_pos': r_feet_pos * scales.feet_pos, 
            'feet_height': r_feet_height * scales.feet_height,
            'lin_vel_z': r_lin_vel_z * scales.lin_vel_z,
            'orientation': r_ori * scales.orientation,
            'torques': r_torques * scales.torques,
        }