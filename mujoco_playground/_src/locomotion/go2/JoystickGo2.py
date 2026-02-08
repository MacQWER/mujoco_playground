from typing import Any, Dict, Optional, Union, Callable
import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
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
    
    # 1. 物理与环境参数
    cfg.Kp = 35.0          
    cfg.Kd = 0.5           
    cfg.sim_dt = 0.002
    cfg.ctrl_dt = 0.02
    cfg.episode_length = 240
    
    cfg.env = config_dict.ConfigDict()
    cfg.env.action_scale = [0.5, 0.5, 0.5] * 4 
    cfg.env.step_k = 13
    cfg.env.impratio = 100
    cfg.env.iterations = 1
    
    # 2. 指令配置
    cfg.command_config = config_dict.ConfigDict()
    cfg.command_config.a = [1.5, 0.80, 1.2]
    cfg.command_config.b = [0.9, 0.25, 0.5]

    # 3. Disturbance 配置 (新增)
    cfg.disturbance = config_dict.ConfigDict()
    cfg.disturbance.enable = False
    cfg.disturbance.velocity_kick = [0.0, 3.0]
    cfg.disturbance.kick_durations = [0.05, 0.2]
    cfg.disturbance.kick_wait_times = [1.0, 3.0]
    
    # 4. 奖励配置 (核心修改)
    cfg.rewards = config_dict.ConfigDict()
    cfg.rewards.scales = config_dict.ConfigDict()
    
    # Tracking
    cfg.rewards.scales.tracking_lin_vel = 1.5
    cfg.rewards.scales.tracking_ang_vel = 0.8
    
    # Anchor Heuristics
    cfg.rewards.scales.feet_pos = -1.0
    cfg.rewards.scales.feet_height = -2.0
    
    # Smoothness & Physics (新增)
    cfg.rewards.scales.lin_vel_z = -2.0
    cfg.rewards.scales.ang_vel_xy = -0.05
    cfg.rewards.scales.orientation = -1.0
    cfg.rewards.scales.torques = -0.0002
    cfg.rewards.scales.action_rate = -0.01
    cfg.rewards.scales.energy = -0.001
    
    # Feet Interaction (新增)
    cfg.rewards.scales.feet_slip = -0.1
    cfg.rewards.scales.feet_clearance = -1.0
    cfg.rewards.scales.feet_air_time = 1.0
    cfg.rewards.scales.dof_pos_limits = -1.0
    cfg.rewards.scales.stand_still = -0.5
    cfg.rewards.scales.termination = -1.0  # Soft termination
    
    cfg.rewards.tracking_sigma = 0.25
    cfg.rewards.max_foot_height = 0.08
    cfg.soft_joint_pos_limit_factor = 0.95

    # 5. Anchor 配置
    cfg.anchor = config_dict.ConfigDict()
    cfg.anchor.path = consts.ANCHOR_PATH
    
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
            xml_path=consts.MJX_XML_SENSOR_PATH.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )

        self._post_init()

    def _post_init(self):
        # 1. Anchor Init
        if not self._config.anchor.path:
            self._anchor_inference_fn = None
        else:
            self._anchor_inference_fn = get_anchor_inference_fn(
                self._config.anchor.path
            )
            
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos.copy())
        self._default_ap_pose = jp.array(self._mj_model.keyframe("home").qpos[7:].copy())
        
        self.action_loc = jp.array(self._default_ap_pose)
        self.action_scale = jp.array(self._config.env.action_scale) 
        
        self.base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.feet_inds = jp.array([self._mj_model.geom(name).id for name in consts.FEET_GEOMS])
        self.hip_inds = jp.array([self._mj_model.body(name).id for name in consts.HIP_NAMES])
        
        # 2. 获取关节限位 (新增)
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

        # 3. 获取足端传感器 (新增，适配 *_sensor.xml)
        self._feet_site_id = np.array([self._mj_model.site(name).id for name in consts.FEET_SITES])
        
        foot_linvel_sensor_adr = []
        try:
            for site in consts.FEET_SITES:
                sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
                sensor_adr = self._mj_model.sensor_adr[sensor_id]
                sensor_dim = self._mj_model.sensor_dim[sensor_id]
                foot_linvel_sensor_adr.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
            self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)
        except Exception:
            self._foot_linvel_sensor_adr = None

        # 4. 步态参数
        step_k = int(getattr(self._config.env, "step_k", 25))
        self.step_k = step_k
        self.gait_period = step_k * 2 * self.dt
        
        # 5. Kinematic Reference
        kinematic_ref_qpos = make_kinematic_ref(cos_wave, step_k, scale=0.3, dt=self.dt)
        self.l_cycle = int(kinematic_ref_qpos.shape[0])

        kinematic_ref_qpos = np.array(kinematic_ref_qpos) + np.array(self._default_ap_pose)
        ref_qs = np.tile(self._init_q.reshape(1, 19), (self.l_cycle, 1))
        ref_qs[:, 7:] = kinematic_ref_qpos
        self.kinematic_ref_qpos = jp.array(ref_qs)
        
        # 6. Command & Raibert Offset
        self._cmd_a = jp.array(self._config.command_config.a)
        self._cmd_b = jp.array(self._config.command_config.b)

        d = mjx_env.make_data(self.mj_model, qpos=self._init_q)
        d = mjx.forward(self.mjx_model, d)
        
        base_pos = d.xpos[self.base_id]
        hip_pos = d.xpos[self.hip_inds]
        rel_pos = hip_pos - base_pos
        self.leg_offsets_x = rel_pos[:, 0]
        self.leg_offsets_y = rel_pos[:, 1]
        self.base_mass = self.mj_model.body("base").mass

    # -------- Reset --------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, key_cmd = jax.random.split(rng)
        
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)
        
        # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        # d(xyzrpy)=U(-0.5, 0.5)
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        )

        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=jp.zeros(12),
                                 impl=self.mjx_model.impl.value, 
                                 nconmax=self._config.nconmax, njmax=self._config.njmax)
        data = mjx.forward(self.mjx_model, data)
        
        # 落地防穿模
        pen = jp.where(data._impl.ncon > 0, jp.min(data._impl.contact.dist), 0.0)
        data = data.replace(qpos=qpos.at[2].set(qpos[2] - pen))
        data = mjx.forward(self.mjx_model, data)

        cmd = self.sample_command(key_cmd, jp.zeros(3))

        hip_pos = data.xpos[self.hip_inds][:, :2]
        feet_pos = data.geom_xpos[self.feet_inds][:, :2]
        
        rng, key_wait, key_dur, key_mag = jax.random.split(rng, 4)
        time_until_next_pert = jax.random.uniform(
            key_wait,
            minval=self._config.disturbance.kick_wait_times[0],
            maxval=self._config.disturbance.kick_wait_times[1],
        )
        steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
            jp.int32
        )
        pert_duration_seconds = jax.random.uniform(
            key_dur,
            minval=self._config.disturbance.kick_durations[0],
            maxval=self._config.disturbance.kick_durations[1],
        )
        pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
            jp.int32
        )
        pert_mag = jax.random.uniform(
            key_mag,
            minval=self._config.disturbance.velocity_kick[0],
            maxval=self._config.disturbance.velocity_kick[1],
        )
        
        # 初始化 State Info (包含新增变量)
        state_info = {
            'rng': rng,
            'step': 0.0,
            'command': cmd,
            'steps_until_next_cmd': 100,
            'last_action': jp.zeros(self.mjx_model.nu), 
            
            # --- 新增变量开始 ---
            'last_residual': jp.zeros(self.mjx_model.nu),      
            'feet_air_time': jp.zeros(4),                      
            'last_contact': jp.zeros(4),                       
            'swing_peak': jp.zeros(4),                         
            # --- 新增变量结束 ---
            
            'xy0': feet_pos,
            'xy*': feet_pos,
            'k0': 0.0,
            'anchor_action': jp.zeros(12),
            'steps_until_next_pert': steps_until_next_pert,
            'pert_duration_seconds': pert_duration_seconds,
            'pert_duration': pert_duration_steps,
            'steps_since_last_pert': 0,
            'pert_steps': 0,
            'pert_dir': jp.zeros(3),
            'pert_mag': pert_mag,
            'reward_tuple': {k: 0.0 for k in self._config.rewards.scales.keys()}
        }

        # Anchor Inference
        anchor_obs = self._get_anchor_obs(data, state_info)
        rng, key_anchor = jax.random.split(rng)
        anchor_act, _ = self._anchor_inference_fn(anchor_obs, key_anchor)
        anchor_act = jp.clip(anchor_act, -1.0, 1.0)
        anchor_act = jax.lax.stop_gradient(anchor_act)
        
        state_info['anchor_action'] = anchor_act
        state_info['rng'] = rng

        residual_obs = self._get_residual_obs(data, state_info)
        
        reward, done = jp.zeros(2)
        metrics = state_info['reward_tuple'].copy()
        
        return mjx_env.State(data, {'state': residual_obs}, reward, done, metrics, state_info)

    # -------- Step --------
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # 0. add disturbance (新增)
        if self._config.disturbance.enable:
            state = self._maybe_apply_perturbation(state)
        # 1. Action Mixing
        action = jp.clip(action, -1.0, 1.0)
        anchor_act = state.info['anchor_action']
        mixed_action = anchor_act + action 
        ctrl = self.action_loc + (mixed_action * self.action_scale)
        
        # 2. Physics Step
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

        # 3. 状态更新 (新增逻辑)
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        contact = jax.nn.sigmoid((0.02 - foot_z) * 50.0) 
        
        delta_contact = jax.nn.relu(contact - state.info["last_contact"])
        first_contact = (state.info["feet_air_time"] > 0.0) * delta_contact
        state.info["feet_air_time"] += self.dt
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], foot_z)

        # 4. Command Update
        state.info['rng'], key_cmd, key_time = jax.random.split(state.info['rng'], 3)
        state.info['steps_until_next_cmd'] -= 1
        should_update = state.info['steps_until_next_cmd'] <= 0
        new_cmd = self.sample_command(key_cmd, state.info['command'])
        new_timer = jp.round(jax.random.exponential(key_time) * 2.5 / self.dt).astype(jp.int32)
        state.info['command'] = jp.where(should_update, new_cmd, state.info['command'])
        state.info['steps_until_next_cmd'] = jp.where(should_update, new_timer, state.info['steps_until_next_cmd'])

        # 5. Raibert & Anchor
        self._update_raibert_target(data, state.info)
        state.info['step'] += 1.0
        state.info['last_action'] = ctrl

        anchor_obs = self._get_anchor_obs(data, state.info)
        next_anchor_act, _ = self._anchor_inference_fn(anchor_obs, key_cmd) 
        next_anchor_act = jp.clip(next_anchor_act, -1.0, 1.0)
        next_anchor_act = jax.lax.stop_gradient(next_anchor_act)
        state.info['anchor_action'] = next_anchor_act

        # 6. Rewards & Termination (修改逻辑)
        up_z = self.get_upvector(data)[-1]
        
        # Hard Termination (翻车保护 + 飞天保护)
        fall_termination = up_z < 0.0
        base_z = data.xpos[self.base_id, 2]
        height_termination = (base_z < 0.05) | (base_z > 0.8)
        done = jp.where(fall_termination | height_termination, 1.0, 0.0)
        
        # Soft Done
        soft_done = jax.nn.sigmoid((0.25 - up_z) * 20.0) 

        reward_kwargs = {
            'first_contact': first_contact,
            'contact': contact,
            'soft_done': soft_done
        }
        
        reward_dict = self._get_reward(data, action, state.info, reward_kwargs, done)
        reward = sum(reward_dict.values()) * self.dt
        
        # Reset counters
        state.info["feet_air_time"] *= (1.0 - contact)
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= (1.0 - contact)

        state.info["last_residual"] = action 
        
        for k, v in reward_dict.items():
            state.metrics[k] = v
        state.info['reward_tuple'] = reward_dict

        # 7. Obs
        residual_obs = self._get_residual_obs(data, state.info)
        
        return state.replace(data=data, obs={'state': residual_obs}, reward=reward, done=done)

    # ----------------- Disturbance -----------------
    def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
        def gen_dir(rng: jax.Array) -> jax.Array:
            angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
            return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

        def apply_pert(state: mjx_env.State) -> mjx_env.State:
            t = state.info["pert_steps"] * self.dt
            u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
            force = (
                u_t
                * self.base_mass
                * state.info["pert_mag"]
                / state.info["pert_duration_seconds"]
            )
            xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
            xfrc_applied = xfrc_applied.at[self.base_id, :3].set(
                force * state.info["pert_dir"]
            )
            state.info["rng"], key_wait, key_dur, key_mag = jax.random.split(
                state.info["rng"], 4
            )
            done_kick = state.info["pert_steps"] >= state.info["pert_duration"]
            time_until_next_pert = jax.random.uniform(
                key_wait,
                minval=self._config.disturbance.kick_wait_times[0],
                maxval=self._config.disturbance.kick_wait_times[1],
            )
            steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
                jp.int32
            )
            pert_duration_seconds = jax.random.uniform(
                key_dur,
                minval=self._config.disturbance.kick_durations[0],
                maxval=self._config.disturbance.kick_durations[1],
            )
            pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
                jp.int32
            )
            pert_mag = jax.random.uniform(
                key_mag,
                minval=self._config.disturbance.velocity_kick[0],
                maxval=self._config.disturbance.velocity_kick[1],
            )
            data = state.data.replace(xfrc_applied=xfrc_applied)
            state = state.replace(data=data)
            state.info["steps_since_last_pert"] = jp.where(
                done_kick,
                0,
                state.info["steps_since_last_pert"],
            )
            state.info["steps_until_next_pert"] = jp.where(
                done_kick,
                steps_until_next_pert,
                state.info["steps_until_next_pert"],
            )
            state.info["pert_duration_seconds"] = jp.where(
                done_kick,
                pert_duration_seconds,
                state.info["pert_duration_seconds"],
            )
            state.info["pert_duration"] = jp.where(
                done_kick,
                pert_duration_steps,
                state.info["pert_duration"],
            )
            state.info["pert_mag"] = jp.where(
                done_kick,
                pert_mag,
                state.info["pert_mag"],
            )
            state.info["pert_steps"] += 1
            return state

        def wait(state: mjx_env.State) -> mjx_env.State:
            state.info["rng"], rng = jax.random.split(state.info["rng"])
            state.info["steps_since_last_pert"] += 1
            xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
            data = state.data.replace(xfrc_applied=xfrc_applied)
            state.info["pert_steps"] = jp.where(
                state.info["steps_since_last_pert"]
                >= state.info["steps_until_next_pert"],
                0,
                state.info["pert_steps"],
            )
            state.info["pert_dir"] = jp.where(
                state.info["steps_since_last_pert"]
                >= state.info["steps_until_next_pert"],
                gen_dir(rng),
                state.info["pert_dir"],
            )
            return state.replace(data=data)
        
        return jax.lax.cond(
            state.info["steps_since_last_pert"]
            >= state.info["steps_until_next_pert"],
            apply_pert,
            wait,
            state,
        )

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

    # =========================================================================
    #                    Reward Logic (Updated with Masks)
    # =========================================================================

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        extra_args: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        """
        聚合所有 Reward，并处理静止/运动状态的冲突。
        """
        scales = self._config.rewards.scales
        
        # === [核心修改] 统一计算可微 Mask ===
        cmd_norm = jp.linalg.norm(info['command'])
        
        # 1. 运动掩码: 当 cmd > 0.01 时接近 1，否则接近 0
        # turned off masking for now
        # 系数 200.0 决定了 Sigmoid 的陡峭程度
        # move_mask = jax.nn.sigmoid((cmd_norm - 0.01) * 200.0)
        move_mask = 1.0
        
        # 2. 静止掩码: 当 cmd < 0.01 时接近 1，否则接近 0
        # still_mask = jax.nn.sigmoid((0.01 - cmd_norm) * 200.0)
        still_mask = 0.0
        
        # 将 Mask 存入 extra_args 供子函数调用
        extra_args['move_mask'] = move_mask
        extra_args['still_mask'] = still_mask
        
        rewards = {
            # --- 1. Tracking Task ---
            "tracking_lin_vel": self._reward_tracking_lin_vel(data, info, extra_args) * scales.tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel(data, info, extra_args) * scales.tracking_ang_vel,
            
            # --- 2. Anchor Heuristics (Masked) ---
            # 静止时关掉 Anchor 引导，避免原地踏步
            "feet_pos":        self._cost_feet_pos(data, info, extra_args) * scales.feet_pos,
            "feet_height":     self._cost_feet_height_wave(data, info, extra_args) * scales.feet_height,
            
            # --- 3. Base Stability & Physics ---
            "lin_vel_z":       self._cost_lin_vel_z(data, info, extra_args) * scales.lin_vel_z,
            "ang_vel_xy":      self._cost_ang_vel_xy(data, info, extra_args) * scales.ang_vel_xy,
            "orientation":     self._cost_orientation(data, info, extra_args) * scales.orientation,
            "torques":         self._cost_torques(data, info, extra_args) * scales.torques,
            "energy":          self._cost_energy(data, info, extra_args) * scales.energy,
            
            # --- 4. Action Smoothness ---
            "action_rate":     self._cost_action_rate(data, info, extra_args, action) * scales.action_rate,
            
            # --- 5. Feet Interaction (Masked) ---
            "feet_slip":       self._cost_feet_slip(data, info, extra_args) * scales.feet_slip,
            "feet_clearance":  self._cost_feet_clearance(data, info, extra_args) * scales.feet_clearance,
            "feet_air_time":   self._reward_feet_air_time(data, info, extra_args) * scales.feet_air_time,
            
            # --- 6. Safety & Limits ---
            "dof_pos_limits":  self._cost_joint_pos_limits(data, info, extra_args) * scales.dof_pos_limits,
            # 静止时专用惩罚
            "stand_still":     self._cost_stand_still(data, info, extra_args) * scales.stand_still,
            "termination":     self._cost_termination(data, info, extra_args) * scales.termination,
        }
        
        return rewards

    # =========================================================================
    #                  Individual Reward Functions (Optimized)
    # =========================================================================

    # --- 1. Tracking Task ---

    def _reward_tracking_lin_vel(self, data, info, extra_args):
        q = data.xquat[1]
        v_local = rotate_inv(data.cvel[1, 3:], q)
        cmd = info['command'][:2]
        err = jp.sum(jp.square(cmd - v_local[:2]))
        return jp.exp(-err / self._config.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self, data, info, extra_args):
        q = data.xquat[1]
        w_local = rotate_inv(data.cvel[1, :3], q)
        cmd = info['command'][2]
        err = jp.square(cmd - w_local[2])
        return jp.exp(-err / self._config.rewards.tracking_sigma)

    # --- 2. Anchor Heuristics (With Masks) ---

    def _cost_feet_pos(self, data, info, extra_args):
        # Anchor 引导的落点位置
        dt_step = (info['step'] - info['k0']) * self.dt
        step_period = self.gait_period / 2
        ratio = jp.clip(dt_step / step_period, 0.0, 1.0)
        
        xyt = info['xy0'] + (info['xy*'] - info['xy0']) * ratio
        curr_feet = data.geom_xpos[self.feet_inds][:, :2]
        dist_sq = jp.sum(jp.square(curr_feet - xyt))
        
        # [Fix] 只有在移动时 (move_mask=1) 才要求追踪落点
        return dist_sq * extra_args['move_mask']

    def _cost_feet_height_wave(self, data, info, extra_args):
        # Anchor 引导的抬腿波形
        t = (info['step'] % self.step_k) * self.dt
        step_period = self.step_k * self.dt
        wave_phase = ((2 * jp.pi) / step_period) * t
        
        ref_wave = -jp.cos(wave_phase) * 0.5 + 0.5
        h_swing = ref_wave * 0.08 + 0.02
        h_stance = 0.02
        
        chunk_idx = (info['step'] // self.step_k).astype(int)
        even_chunk = (chunk_idx % 2 == 0)
        
        targets_even = jp.array([h_stance, h_swing, h_swing, h_stance])
        targets_odd  = jp.array([h_swing, h_stance, h_stance, h_swing])
        h_tars = jp.where(even_chunk, targets_even, targets_odd)
        
        curr_feet_z = data.geom_xpos[self.feet_inds][:, 2]
        
        # [Fix] 只有在移动时 (move_mask=1) 才要求腿做正弦运动，防止静止时原地踏步
        return jp.sum(jp.square(curr_feet_z - h_tars)) * extra_args['move_mask']

    # --- 3. Base Stability & Physics ---

    def _cost_lin_vel_z(self, data, info, extra_args):
        return jp.square(data.cvel[1, 5])

    def _cost_ang_vel_xy(self, data, info, extra_args):
        return jp.sum(jp.square(data.cvel[1, :2]))

    def _cost_orientation(self, data, info, extra_args):
        up_vec = self.get_upvector(data)
        return jp.sum(jp.square(up_vec[:2]))

    def _cost_torques(self, data, info, extra_args):
        torques = data.actuator_force
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _cost_energy(self, data, info, extra_args):
        return jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force))

    # --- 4. Action Smoothness ---

    def _cost_action_rate(self, data, info, extra_args, current_action):
        return jp.sum(jp.square(current_action - info['last_residual']))

    # --- 5. Feet Interaction (With Masks) ---

    def _cost_feet_slip(self, data, info, extra_args):
        contact = extra_args['contact']
        
        if self._foot_linvel_sensor_adr is not None:
            feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
            vel_xy = feet_vel[..., :2]
            vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
            
            effective_contact = jax.nn.relu(contact - 0.5) * 2.0
            
            # [Fix] 只有在移动时才惩罚滑动 (静止时由 stand_still 负责锁死)
            return jp.sum(vel_xy_norm_sq * effective_contact) * extra_args['move_mask']
        
        return 0.0

    def _cost_feet_clearance(self, data, info, extra_args):
        # 无需 Mask，因为静止时 vel_norm 为 0，Cost 自动为 0
        if self._foot_linvel_sensor_adr is not None:
            feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
            vel_xy = feet_vel[..., :2]
            vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
            
            foot_pos = data.site_xpos[self._feet_site_id]
            foot_z = foot_pos[..., -1]
            
            delta = jp.abs(foot_z - self._config.rewards.max_foot_height)
            return jp.sum(delta * vel_norm)
            
        return 0.0

    def _reward_feet_air_time(self, data, info, extra_args):
        air_time = info['feet_air_time']
        first_contact = extra_args['first_contact']
        
        rew = jp.sum((air_time - 0.1) * first_contact)
        
        # [Fix] 只有在移动时才奖励腾空
        return rew * extra_args['move_mask']

    # --- 6. Safety & Limits ---

    def _cost_joint_pos_limits(self, data, info, extra_args):
        qpos = data.qpos[7:]
        out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
        return jp.sum(out_of_limits)

    def _cost_stand_still(self, data, info, extra_args):
        qpos = data.qpos[7:]
        # [Fix] 使用 still_mask，只在静止时惩罚关节偏离默认姿态
        return jp.sum(jp.abs(qpos - self._default_ap_pose)) * extra_args['still_mask']

    def _cost_termination(self, data, info, extra_args):
        return extra_args['soft_done']
    
    # --- Debug Util ---

    def check_phase_alignment(self):
        """
        [终极验证版 - 已修复 MockData Bug]
        同时检查以下三者的相位是否严格对齐：
        1. Reward: 抬腿高度波形
        2. Raibert: 落点目标更新
        3. Ref Kino: 参考动作的关节角度
        """
        print("Running Ultimate Phase Alignment Check (Height + Raibert + Kino)...")
        import matplotlib.pyplot as plt
        from collections import namedtuple
        
        # 模拟 4 个周期
        steps = np.arange(self.step_k * 4)
        
        # 日志
        height_target_log = []   # Reward 要求的高度
        raibert_change_log = []  # Raibert 落点是否发生更新 (Event)
        kino_thigh_log = []      # Ref Kino 中的髋关节角度
        
        # Mock Data 定义
        # [Fix] 确保 geom_xpos 不是 None
        MockData = namedtuple('MockData', ['geom_xpos', 'xpos', 'xquat'])
        
        # 计算最大的 geom index 以创建正确大小的数组
        max_geom_id = int(jp.max(self.feet_inds)) + 1
        fake_geom_xpos = jp.zeros((max_geom_id, 3)) # <--- 修复点：全零数组
        
        # 伪造一个恒定的指令：向前走 1.0 m/s
        cmd = jp.array([1.0, 0.0, 0.0])
        
        # 初始化状态用于 Raibert 迭代
        info = {
            'step': 0.0,
            'k0': 0.0,
            'command': cmd,
            'xy0': jp.zeros((4, 2)), 
            'xy*': jp.zeros((4, 2)), 
        }
        
        last_fl_target_x = 0.0
        
        # 伪造物理数据
        fake_xpos = jp.zeros((self.mjx_model.nbody, 3))
        fake_xquat = jp.array([[1., 0., 0., 0.]] * self.mjx_model.nbody)
        
        # [Fix] 传入 fake_geom_xpos 而不是 None
        mock_data_raibert = MockData(geom_xpos=fake_geom_xpos, xpos=fake_xpos, xquat=fake_xquat)

        for s in steps:
            # 更新 step
            info['step'] = jp.array(float(s))
            
            # --- 1. Check Reward (Height Wave) ---
            t = (s % self.step_k) * self.dt
            step_period = self.step_k * self.dt
            wave_phase = ((2 * np.pi) / step_period) * t
            ref_wave = -np.cos(wave_phase) * 0.5 + 0.5
            
            chunk_idx = int(s // self.step_k)
            is_even = (chunk_idx % 2 == 0)
            
            if is_even:
                h_target = 0.02 # Stance
            else:
                h_target = ref_wave * 0.08 + 0.02 # Swing
            
            height_target_log.append(h_target)
            
            # --- 2. Check Raibert (Target Update) ---
            # 现在调用不会报错了
            self._update_raibert_target(mock_data_raibert, info)
            
            current_fl_target_x = float(info['xy*'][0, 0]) 
            
            if s > 0 and abs(current_fl_target_x - last_fl_target_x) > 1e-4:
                raibert_change_log.append(0.12) 
            else:
                raibert_change_log.append(0.0)
            
            last_fl_target_x = current_fl_target_x
            
            # --- 3. Check Ref Kino (Joint Angle) ---
            step_idx = int(s % self.l_cycle)
            kin_ref = self.kinematic_ref_qpos[step_idx] 
            fl_thigh_angle = float(kin_ref[8])
            kino_thigh_log.append(fl_thigh_angle)

        # --- 绘图验证 ---
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 1. Height Wave (Green)
        ax1.plot(steps, height_target_log, 'g-', label='Reward: Height Target (FL)', linewidth=2)
        ax1.set_ylabel('Height (m)')
        ax1.set_ylim(0, 0.15)
        
        # 2. Ref Kino (Blue)
        ax2 = ax1.twinx()
        ax2.plot(steps, kino_thigh_log, 'b--', label='Ref Kino: FL Thigh Angle', linewidth=2, alpha=0.6)
        ax2.set_ylabel('Joint Angle (rad)')
        
        # 3. Raibert Update (Red Bars)
        ax1.bar(steps, raibert_change_log, width=1.0, color='red', alpha=0.3, label='Raibert: Target Update Event')
        
        plt.title(f"Multi-Phase Alignment Check (Step K={self.step_k})")
        plt.axvspan(0, self.step_k, color='gray', alpha=0.1, label='Even Step (Stance)')
        plt.axvspan(self.step_k, self.step_k*2, color='green', alpha=0.1, label='Odd Step (Swing)')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.grid(True)
        plt.show()
    
