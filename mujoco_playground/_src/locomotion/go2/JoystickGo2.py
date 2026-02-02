from typing import Any, Dict, Optional, Union, Callable
import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from ml_collections import config_dict

from brax.io import model

import matplotlib.pyplot as plt

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
    cfg.env.action_scale = [0.5, 0.5, 0.5] * 4 
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
        self.hip_inds = jp.array([self._mj_model.body(name).id for name in consts.HIP_NAMES])
        
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

        # Raibert Preparation
        d = mjx_env.make_data(self.mj_model, qpos=self._init_q)
        d = mjx.forward(self.mjx_model, d)
        
        base_pos = d.xpos[self.base_id]
        hip_pos = d.xpos[self.hip_inds]
        rel_pos = hip_pos - base_pos
        self.leg_offsets_x = rel_pos[:, 0] # [FL, FR, RL, RR]
        self.leg_offsets_y = rel_pos[:, 1]

    # -------- Reset --------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, key_cmd = jax.random.split(rng)
        
        # 1. 物理 Reset
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)
        
        # TODO 稍微随机化初始 yaw 或 位置
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

        # Init Raibert
        hip_pos = data.xpos[self.hip_inds][:, :2]
        feet_pos = data.geom_xpos[self.feet_inds][:, :2]
        
        # 3. 初始化 State Info
        state_info = {
            'rng': rng,
            'step': 0.0,
            'command': cmd,
            'steps_until_next_cmd': 100,
            'last_action': jp.zeros(self.mjx_model.nu), # ctrl
            
            # Raibert & Gait 变量 (用于 Reward)
            'xy0': feet_pos,
            'xy*': feet_pos,
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
        anchor_act = jp.clip(anchor_act, -1.0, 1.0)
        
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
        # 1. Action Mixing
        action = jp.clip(action, -1.0, 1.0)
        anchor_act = state.info['anchor_action']
        
        # [Trick] Residual Boost if needed (Optional, based on previous discussion)
        # residual_boost = jp.array([2.5, 1.0, 1.0] * 4) 
        # mixed_action = anchor_act + action * residual_boost
        mixed_action = anchor_act + action # Standard mixing
        
        ctrl = self.action_loc + (mixed_action * self.action_scale)
        
        # 2. Physics Step
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

        # 3. Command Update
        state.info['rng'], key_cmd, key_time = jax.random.split(state.info['rng'], 3)
        state.info['steps_until_next_cmd'] -= 1
        should_update = state.info['steps_until_next_cmd'] <= 0
        
        new_cmd = self.sample_command(key_cmd, state.info['command'])
        new_timer = jp.round(jax.random.exponential(key_time) * 2.5 / self.dt).astype(jp.int32)
        
        state.info['command'] = jp.where(should_update, new_cmd, state.info['command'])
        state.info['steps_until_next_cmd'] = jp.where(should_update, new_timer, state.info['steps_until_next_cmd'])

        # 4. Update Raibert Target (Standard Hip Logic)
        self._update_raibert_target(data, state.info)
        
        state.info['step'] += 1.0
        state.info['last_action'] = ctrl

        # 5. Anchor Inference for Next Step
        anchor_obs = self._get_anchor_obs(data, state.info)
        next_anchor_act, _ = self._anchor_inference_fn(anchor_obs, key_cmd) 
        next_anchor_act = jp.clip(next_anchor_act, -1.0, 1.0)
        next_anchor_act = jax.lax.stop_gradient(next_anchor_act)
        state.info['anchor_action'] = next_anchor_act

        # 6. Compute Rewards (Modular)
        # 终止条件
        base_z = data.xpos[self.base_id, 2]
        done = jp.where(base_z < self._config.env.termination_height, 1.0, 0.0)
        
        # 计算具体 Reward
        reward_dict = self._get_reward(data, action, state.info, {}, done)
        reward = sum(reward_dict.values())
        
        # Metrics update
        for k, v in reward_dict.items():
            state.metrics[k] = v
        state.info['reward_tuple'] = reward_dict

        # 7. Obs
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

    # -------- Raibert Heuristic (Standard Hip Logic) --------
    def _update_raibert_target(self, data, info):
        """
        Raibert Heuristic Target Updater.
        Phase Logic: Even Step -> FR/RL (Pair 2) Swing.
        Reference Frame: Hip-Centric.
        """
        s = info['step']
        step_k = self.step_k
        new_step = (s % step_k == 0)
        even_step = ((s // step_k) % 2 == 0)
        
        # 1. 解析指令
        v_cmd_local = info['command'][:2]
        w_cmd_local = info['command'][2]
        
        # 2. 计算旋转引起的线速度分量 (v = w x r)
        # 使用 _post_init 中自动计算的偏移量，不再硬编码
        v_rot_x = -w_cmd_local * self.leg_offsets_y
        v_rot_y =  w_cmd_local * self.leg_offsets_x

        # 3. 合成局部线速度
        v_leg_local = jp.stack([
            v_cmd_local[0] + v_rot_x,
            v_cmd_local[1] + v_rot_y
        ], axis=1)

        # 4. 旋转到世界坐标系
        # 使用 vmap 批量旋转 4 条腿的速度向量
        quat = data.xquat[1]
        v_leg_local_3d = jp.concatenate([v_leg_local, jp.zeros((4, 1))], axis=1)
        v_leg_global = jax.vmap(rotate, in_axes=(0, None))(v_leg_local_3d, quat)[:, :2]

        # 5. 计算 Raibert 落点 (Hip-Centric)
        # 物理公式: Target = Hip + (T_stance / 2) * v
        # 在 Trot 中 T_stance = T_cycle / 2, 所以系数是 T_cycle / 4
        hip_pos = data.xpos[self.hip_inds][:, :2] 
        raibert_offset = (self.gait_period / 4.0) * v_leg_global
        raibert_xy = hip_pos + raibert_offset
        
        # 6. 更新目标 (Phase Alignment)
        # Pair 1: FL(0), RR(3) -> 对应 Odd Step (Block 2) 摆动
        # Pair 2: FR(1), RL(2) -> 对应 Even Step (Block 1) 摆动
        pair1 = jp.array([0, 3]) 
        pair2 = jp.array([1, 2]) 
        
        cur_tars = info['xy*']
        
        # 逻辑: 
        # Even Step 开始瞬间 -> FR/RL (Pair 2) 变成摆动腿 -> 更新它们的目标
        tars_p2 = cur_tars.at[pair2].set(raibert_xy[pair2])
        
        # Odd Step 开始瞬间 -> FL/RR (Pair 1) 变成摆动腿 -> 更新它们的目标
        tars_p1 = cur_tars.at[pair1].set(raibert_xy[pair1])
        
        # 选择
        xy_tars = jp.where(new_step & even_step, tars_p2, cur_tars)
        xy_tars = jp.where(new_step & (~even_step), tars_p1, xy_tars)
        
        info['xy*'] = xy_tars
        
        # 记录起跳点 (用于插值 Reward)
        feet_pos = data.geom_xpos[self.feet_inds][:, :2]
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

    # -------- Modular Rewards --------
    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        """
        聚合所有 Reward 组件。
        """
        scales = self._config.rewards.scales
        
        rewards = {
            "tracking_lin_vel": self._reward_tracking_lin_vel(data, info) * scales.tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel(data, info) * scales.tracking_ang_vel,
            "feet_pos": self._cost_feet_pos(data, info) * scales.feet_pos,
            "feet_height": self._cost_feet_height(data, info) * scales.feet_height,
            "lin_vel_z": self._cost_lin_vel_z(data) * scales.lin_vel_z,
            "orientation": self._cost_orientation(data) * scales.orientation,
            "torques": self._cost_torques(data) * scales.torques,
        }
        
        return rewards

    # --- Individual Reward Functions ---
    
    def _reward_tracking_lin_vel(self, data, info):
        q = data.xquat[1]
        v_local = rotate_inv(data.cvel[1, 3:], q)
        cmd = info['command'][:2]
        err = jp.sum(jp.square(cmd - v_local[:2]))
        return jp.exp(-err / self._config.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self, data, info):
        q = data.xquat[1]
        w_local = rotate_inv(data.cvel[1, :3], q)
        cmd = info['command'][2]
        err = jp.square(cmd - w_local[2])
        return jp.exp(-err / self._config.rewards.tracking_sigma)

    def _cost_feet_pos(self, data, info):
        # Raibert Cost
        # 计算当前时刻应该在的位置 (插值)
        dt_step = (info['step'] - info['k0']) * self.dt
        step_period = self.gait_period / 2
        ratio = jp.clip(dt_step / step_period, 0.0, 1.0)
        
        # 目标轨迹：从起跳点(xy0) 移动到 Raibert目标点(xy*)
        xyt = info['xy0'] + (info['xy*'] - info['xy0']) * ratio
        
        curr_feet = data.geom_xpos[self.feet_inds][:, :2]
        dist_sq = jp.sum(jp.square(curr_feet - xyt))
        return dist_sq

    def _cost_feet_height(self, data, info):
        """
        Feet Height Tracking Cost.
        Must strictly align with Kinematic Reference phase.
        """
        # 1. 计算当前在 step_k 周期内的时间
        t = (info['step'] % self.step_k) * self.dt
        step_period = self.step_k * self.dt
        
        # 2. 复刻 Reference 的 cos 波形 (0 -> 1 -> 0)
        # 原始公式: -cos(2*pi*t/T) * 0.5 + 0.5
        wave_phase = ((2 * jp.pi) / step_period) * t
        ref_wave = -jp.cos(wave_phase) * 0.5 + 0.5
        
        # 3. 设定高度
        h_tar_swing = 0.08  # 摆动高度
        h_tar_stance = 0.02 # 支撑高度 (稍微离地一点点避免穿模噪音)
        
        h_swing = ref_wave * h_tar_swing + h_tar_stance
        h_stance = h_tar_stance
        
        # 4. 相位分配
        # Even Step: FR(1)/RL(2) Swing
        # Odd Step:  FL(0)/RR(3) Swing
        chunk_idx = (info['step'] // self.step_k).astype(int)
        even_chunk = (chunk_idx % 2 == 0)
        
        # 目标向量构建 [FL, FR, RL, RR]
        # Even: [Stance, Swing, Swing, Stance]
        targets_even = jp.array([h_stance, h_swing, h_swing, h_stance])
        # Odd:  [Swing, Stance, Stance, Swing]
        targets_odd  = jp.array([h_swing, h_stance, h_stance, h_swing])
        
        h_tars = jp.where(even_chunk, targets_even, targets_odd)
        
        # 5. 计算 Cost
        curr_feet_z = data.geom_xpos[self.feet_inds][:, 2]
        errs = jp.sum(jp.square(curr_feet_z - h_tars))
        return errs

    def _cost_lin_vel_z(self, data):
        return jp.square(data.cvel[1, 5])

    def _cost_orientation(self, data):
        q = data.xquat[1]
        g_local = rotate_inv(jp.array([0., 0., -1.]), q)
        return jp.sum(jp.square(g_local[:2])) # xy component should be 0

    def _cost_torques(self, data):
        return jp.sum(jp.square(data.qfrc_actuator[6:]))
    
    # --- Debug Util ---

    def check_phase_alignment(self):
        """
        Debug工具：绘制 Reference Motion Z轴高度 vs Heuristic Target Z轴高度。
        用于验证相位是否对齐。
        """
        print("Checking Phase Alignment...")
        
        # 模拟一个完整的步态周期 (2 * step_k)
        cycle_len = self.l_cycle
        steps = np.arange(cycle_len)
        
        ref_z_log = []
        heuristic_z_log = []
        
        # 创建一个 dummy data 用于调用 _cost_feet_height
        # 我们不需要真实的物理 step，只需要 info['step']
        dummy_data = mjx.make_data(self.mj_model) # 空数据
        
        for s in steps:
            # 1. 获取 Reference Motion 的 Z 高度 (Ground Truth)
            # Reference Qpos 结构: [Base(7), Joints(12)]
            # 我们需要通过正向运动学算出 Ref 对应的脚高度
            # 这里简单起见，我们直接复用 make_kinematic_ref 里的波形逻辑来生成 Truth
            # 或者更严谨地，直接读 kinematic_ref_qpos
            
            # 这里我们用你提供的 cos_wave 逻辑重算一遍 "理论上的 Reference Z"
            t = (s % self.step_k) * self.dt
            step_period = self.step_k * self.dt
            wave_phase = ((2 * np.pi) / step_period) * t
            ref_val = -np.cos(wave_phase) * 0.5 + 0.5
            
            # 判断当前脚的状态 (Block 1 vs Block 2)
            is_even = (s // self.step_k) % 2 == 0
            
            # 记录 FL (左前, index 0) 的高度
            # Even Step (Block 1): FL 是支撑 (Stance) -> 高度应该是 0
            # Odd Step (Block 2): FL 是摆动 (Swing) -> 高度应该是 ref_val * scale
            if is_even:
                ref_z_fl = 0.02 # stance
            else:
                ref_z_fl = ref_val * 0.08 + 0.02 # swing
            
            ref_z_log.append(ref_z_fl)
            
            # 2. 获取 Heuristic Target 的 Z 高度
            # 调用你的函数
            info = {'step': jp.array(s)}
            
            # 我们 hack 一下 _cost_feet_height 里的逻辑提取出 h_tars
            # 直接复制 _cost_feet_height 的逻辑片段:
            t_jax = (info['step'] % self.step_k) * self.dt
            wave_phase_jax = ((2 * jp.pi) / step_period) * t_jax
            ref_wave_jax = -jp.cos(wave_phase_jax) * 0.5 + 0.5
            h_swing = ref_wave_jax * 0.08 + 0.02
            h_stance = 0.02
            
            chunk_idx = (info['step'] // self.step_k).astype(int)
            even_chunk = (chunk_idx % 2 == 0)
            
            # 提取 FL (index 0) 的目标
            # Even: targets_even[0] = h_stance
            # Odd:  targets_odd[0]  = h_swing
            fl_target = jp.where(even_chunk, h_stance, h_swing)
            
            heuristic_z_log.append(float(fl_target))

        # 3. 绘图
        plt.figure(figsize=(10, 4))
        plt.plot(steps, ref_z_log, label='Reference Motion (FL Leg)', linewidth=3, alpha=0.5)
        plt.plot(steps, heuristic_z_log, 'r--', label='Heuristic Target (FL Leg)')
        
        # 标注相位区域
        plt.axvspan(0, self.step_k, color='gray', alpha=0.1, label='Even Step (Phase 1)')
        plt.axvspan(self.step_k, self.step_k*2, color='green', alpha=0.1, label='Odd Step (Phase 2)')
        
        plt.title(f"Phase Alignment Check (Step K = {self.step_k})")
        plt.xlabel("Step")
        plt.ylabel("Z Height (m)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print("Analysis:")
        print("Even Step (0~13): FL should be Stance (Low).")
        print("Odd Step (13~26): FL should be Swing (High).")

    