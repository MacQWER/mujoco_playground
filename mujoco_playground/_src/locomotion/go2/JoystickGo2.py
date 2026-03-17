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
import mediapy as media

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
    cfg.env.anchor_action_scale = [0.5, 0.5, 0.5] * 4
    cfg.env.residual_action_scale = [1.0, 0.5, 0.5] * 4
    cfg.env.step_k = consts.STEP_K
    cfg.env.gait_scale = consts.GAIT_SCALE
    cfg.env.raibert_k = 0.5
    cfg.env.step_height = 0.1128  # max peak cycloid height (cap)
    cfg.env.step_height_min = 0.0
    cfg.env.foot_traj_vel_weight = 0.2
    cfg.env.impratio = 100
    cfg.env.iterations = 1

    # 1.5. Observation Noise (参考 joystick.py)
    cfg.noise_config = config_dict.ConfigDict()
    cfg.noise_config.level = 1.0  # Set to 0.0 to disable noise.
    cfg.noise_config.scales = config_dict.ConfigDict()
    cfg.noise_config.scales.joint_pos = 0.03
    cfg.noise_config.scales.joint_vel = 1.5
    cfg.noise_config.scales.gyro = 0.2
    cfg.noise_config.scales.gravity = 0.05
    cfg.noise_config.scales.linvel = 0.1
    
    # 2. 指令配置
    cfg.command_config = config_dict.ConfigDict()
    cfg.command_config.a = [0.5, 0.2, 0.5]
    cfg.command_config.b = [1.0, 1.0, 1.0]

    # 3. Disturbance 配置 (新增)
    cfg.disturbance = config_dict.ConfigDict()
    cfg.disturbance.enable = True
    cfg.disturbance.velocity_kick = [0.0, 1.0]
    cfg.disturbance.kick_durations = [0.05, 0.2]
    cfg.disturbance.kick_wait_times = [1.0, 3.0]
    
    # 4. 奖励配置 (核心修改)
    cfg.rewards = config_dict.ConfigDict()
    cfg.rewards.scales = config_dict.ConfigDict()
    
    # Tracking
    cfg.rewards.scales.tracking_lin_vel = 1.5
    cfg.rewards.scales.tracking_ang_vel = 1.0
    
    # Anchor Heuristics
    cfg.rewards.scales.feet_traj = -5.0
    
    # Smoothness & Physics (新增)
    cfg.rewards.scales.lin_vel_z = -0.5
    cfg.rewards.scales.ang_vel_xy = -0.05
    cfg.rewards.scales.orientation = -10.0
    cfg.rewards.scales.torques = -0.0002
    cfg.rewards.scales.action_rate = -0.05
    cfg.rewards.scales.energy = -0.001
    
    # Feet Interaction (新增)
    cfg.rewards.scales.feet_slip = -0.1
    cfg.rewards.scales.feet_clearance = -1.0
    cfg.rewards.scales.feet_air_time = 5.0
    cfg.rewards.scales.dof_pos_limits = -1.0
    cfg.rewards.scales.stand_still = -0.5
    cfg.rewards.scales.termination = -10.0  # Soft termination
    
    cfg.rewards.tracking_sigma = 0.25
    cfg.rewards.max_foot_height = 0.075
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
        self.anchor_action_scale = jp.array(self._config.env.anchor_action_scale)
        self.residual_action_scale = jp.array(self._config.env.residual_action_scale)
        
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
        gait_scale = float(getattr(self._config.env, "gait_scale", 0.3))
        self.step_k = step_k
        self.gait_scale = gait_scale
        self.gait_period = step_k * 2 * self.dt
        self.raibert_k = float(getattr(self._config.env, "raibert_k", 0.2))
        
        # 5. Kinematic Reference
        kinematic_ref_qpos = make_kinematic_ref(cos_wave, step_k, scale=gait_scale, dt=self.dt)
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

        # Hip -> foot lateral offset in base local frame
        foot_pos = d.site_xpos[self._feet_site_id]
        base_quat = d.xquat[self.base_id]
        hip_local = jax.vmap(rotate_inv, in_axes=(0, None))(hip_pos - base_pos, base_quat)
        foot_local = jax.vmap(rotate_inv, in_axes=(0, None))(foot_pos - base_pos, base_quat)
        foot_offset_local = foot_local - hip_local
        self.foot_offsets_xy = foot_offset_local[:, :2]

        self.base_mass = self.mj_model.body("base").mass

        # Step height parameters (coupled to step length; capped)
        self._step_height_max = float(getattr(self._config.env, "step_height", 0.1128))
        self._step_height_min = float(getattr(self._config.env, "step_height_min", 0.0))
        self._foot_traj_vel_weight = float(getattr(self._config.env, "foot_traj_vel_weight", 0.2))


    def _get_swing_mask(self, step):
        """Return swing mask (4,) for FL, FR, RL, RR at current phase block."""
        step = jp.asarray(step, dtype=jp.int32)
        chunk_idx = step // self.step_k
        even_chunk = (chunk_idx % 2 == 0)
        swing_even = jp.array([0.0, 1.0, 1.0, 0.0])  # FR, RL
        swing_odd = jp.array([1.0, 0.0, 0.0, 1.0])   # FL, RR
        return jp.where(even_chunk, swing_even, swing_odd)

    def _update_foot_cycloid_ref(self, info):
        """
        Build foot reference trajectory from Raibert endpoints + cycloid phase.
        ref_pos/ref_v_xy are stored in info for reward tracking.
        """
        step = info['step']
        swing_period = self.gait_period / 2.0
        dt_step = (step - info['k0']) * self.dt
        phi = jp.clip(dt_step / swing_period, 0.0, 1.0)

        s = phi - jp.sin(2.0 * jp.pi * phi) / (2.0 * jp.pi)
        ds_dt = (1.0 - jp.cos(2.0 * jp.pi * phi)) / swing_period

        xy0 = info['xy0']
        xys = info['xy*']
        delta_xy = xys - xy0

        xy_ref = xy0 + delta_xy * s
        v_xy_ref = delta_xy * ds_dt

        swing_mask = self._get_swing_mask(step)
        swing_mask_col = swing_mask[:, None]

        # Stance legs hold touchdown anchor; swing legs follow cycloid.
        xy_ref = xy0 * (1.0 - swing_mask_col) + xy_ref * swing_mask_col
        v_xy_ref = v_xy_ref * swing_mask_col

        # Cycloid height coupled to step length (2*pi*r = step length)
        step_len = jp.linalg.norm(delta_xy, axis=1)
        r = step_len / (2.0 * jp.pi)
        r_min = 0.5 * self._step_height_min
        r_max = 0.5 * self._step_height_max
        r = jp.clip(r, r_min, r_max)

        z0 = info['z0']
        z_ref = z0 + r * (1.0 - jp.cos(2.0 * jp.pi * phi))
        z_ref = z0 * (1.0 - swing_mask) + z_ref * swing_mask

        ref_pos = jp.concatenate([xy_ref, z_ref[:, None]], axis=1)

        info['foot_phase'] = jax.lax.stop_gradient(phi)
        info['foot_swing'] = jax.lax.stop_gradient(swing_mask)
        info['foot_ref_xy'] = jax.lax.stop_gradient(xy_ref)
        info['foot_ref_z'] = jax.lax.stop_gradient(z_ref)
        info['foot_ref_pos'] = jax.lax.stop_gradient(ref_pos)
        info['foot_ref_v_xy'] = jax.lax.stop_gradient(v_xy_ref)

    # -------- Reset --------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, key_cmd = jax.random.split(rng)
        
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)
        
        # # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
        # rng, key = jax.random.split(rng)
        # dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        # qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        # rng, key = jax.random.split(rng)
        # yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        # quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        # new_quat = math.quat_mul(qpos[3:7], quat)
        # qpos = qpos.at[3:7].set(new_quat)

        # # d(xyzrpy)=U(-0.5, 0.5)
        # rng, key = jax.random.split(rng)
        # qvel = qvel.at[0:6].set(
        #     jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        # )

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
            'step': jp.array(0, dtype=jp.int32),
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
            'k0': jp.array(0, dtype=jp.int32),
            'foot_phase': 0.0,
            'foot_swing': jp.zeros(4),
            'foot_ref_xy': feet_pos,
            'foot_ref_z': data.geom_xpos[self.feet_inds][:, 2],
            'foot_ref_pos': jp.concatenate([feet_pos, data.geom_xpos[self.feet_inds][:, 2:3]], axis=1),
            'z0': data.site_xpos[self._feet_site_id][:, 2],
            'foot_ref_v_xy': jp.zeros((4, 2)),
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
        self._update_foot_cycloid_ref(state_info)

        # Anchor Inference
        anchor_obs = self._get_anchor_obs(data, state_info)
        state_info["rng"], key_anchor = jax.random.split(state_info["rng"])
        anchor_act, _ = self._anchor_inference_fn(anchor_obs, key_anchor)
        anchor_act = jp.clip(anchor_act, -1.0, 1.0)
        anchor_act = jax.lax.stop_gradient(anchor_act)
        
        state_info['anchor_action'] = anchor_act

        residual_obs = self._get_residual_obs(data, state_info)
        
        reward, done = jp.zeros(2)
        metrics = state_info['reward_tuple'].copy()
        
        return mjx_env.State(data, residual_obs, reward, done, metrics, state_info)

    # -------- Step --------
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # 0. add disturbance (新增)
        if self._config.disturbance.enable:
            state = self._maybe_apply_perturbation(state)
        # 1. Action Mixing
        action = jp.clip(action, -1.0, 1.0)
        anchor_act = state.info['anchor_action']
        mixed_action = anchor_act * self.anchor_action_scale + action * self.residual_action_scale
        ctrl = self.action_loc + mixed_action
        
        # 2. Physics Step
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

        # 3. 状态更新 (新增逻辑)
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        contact = jax.nn.sigmoid((0.025 - foot_z) * 100.0) 
        
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
        state.info['step'] += 1
        self._update_foot_cycloid_ref(state.info)
        state.info['last_action'] = ctrl

        anchor_obs = self._get_anchor_obs(data, state.info)
        next_anchor_act, _ = self._anchor_inference_fn(anchor_obs, key_cmd) 
        next_anchor_act = jp.clip(next_anchor_act, -1.0, 1.0)
        next_anchor_act = jax.lax.stop_gradient(next_anchor_act)
        state.info['anchor_action'] = next_anchor_act

        # 6. Rewards & Termination (修改逻辑)
        up_z = self.get_upvector(data)[-1]
        
        # Hard Termination (翻车保护)
        tilt_threshold = jp.cos(jp.deg2rad(45.0))
        fall_termination = up_z < tilt_threshold
        done = jp.where(fall_termination, 1.0, 0.0)
        
        # Soft Done
        soft_done = jax.nn.sigmoid((tilt_threshold - up_z) * 100.0)

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

        # 7. Obs (dict)
        residual_obs = self._get_residual_obs(data, state.info)
        
        return state.replace(data=data, obs=residual_obs, reward=reward, done=done)

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

    def _apply_obs_noise(self, info: dict[str, Any], x: jax.Array, scale: float):
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noise = (
            (2 * jax.random.uniform(noise_rng, shape=x.shape) - 1)
            * self._config.noise_config.level
            * scale
        )
        return x + noise
    
    def _get_obs(self, data, info):
        """
        Unified observation (aligned with TrotGo2):
        Order (72 dims): v_local(3), w_local(3), g_local(3), command(3),
        angles(12), joint_vels(12), last_action(12), kin_ref(12), anchor_action(12).
        """
        q = data.xquat[1]
        v_local = rotate_inv(data.cvel[1, 3:], q)
        v_local = self._apply_obs_noise(
            info, v_local, self._config.noise_config.scales.linvel
        )
        w_local = rotate_inv(data.cvel[1, :3], q)
        w_local = self._apply_obs_noise(
            info, w_local, self._config.noise_config.scales.gyro
        )
        g_local = rotate_inv(jp.array([0., 0., -1.]), q)
        g_local = self._apply_obs_noise(
            info, g_local, self._config.noise_config.scales.gravity
        )
        angles = data.qpos[7:19]
        angles = self._apply_obs_noise(
            info, angles, self._config.noise_config.scales.joint_pos,
        )
        joint_vels = data.qvel[6:]
        joint_vels = self._apply_obs_noise(
            info, joint_vels, self._config.noise_config.scales.joint_vel,
        )
        last_action = info['last_action']
        step_idx = jp.array(info['step'] % self.l_cycle, int)
        kin_ref = self.kinematic_ref_qpos[step_idx][7:]
        command = info['command']
        anchor_action = info['anchor_action']

        obs_list = [
            v_local,                        # 3
            w_local,                        # 3
            g_local,                        # 3
            command,                        # 3 (Vx, Vy, Wz)
            angles - self._default_ap_pose, # 12
            joint_vels,                     # 12
            last_action,                    # 12
            kin_ref,                        # 12
            anchor_action,                  # 12
        ]
        obs = jp.clip(jp.concatenate(obs_list), -100.0, 100.0)
        return {"state": obs}

    def _get_anchor_obs(self, data, info):
        obs_dict = self._get_obs(data, info)
        obs = obs_dict["state"]
        # Mask command (indices 9 to 11)
        obs = obs.at[9:12].set(0.0)
        # Mask anchor_action (indices 60 to 71)
        obs = obs.at[60:72].set(0.0)
        
        return {"state": obs}

    def _get_residual_obs(self, data, info):
        return self._get_obs(data, info)

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

        # 1. Parse command velocities (use command for feedforward)
        v_cmd_local = info['command'][:2]
        w_cmd_local = info['command'][2]

        quat = data.xquat[1]

        # 2. Feedforward: Kinematic drift prediction using COMMAND velocities
        v_ff_rot_x = -w_cmd_local * self.leg_offsets_y
        v_ff_rot_y =  w_cmd_local * self.leg_offsets_x

        v_leg_local = jp.stack([
            v_cmd_local[0] + v_ff_rot_x,
            v_cmd_local[1] + v_ff_rot_y
        ], axis=1)

        # 3. Global rotation
        v_leg_local_3d = jp.concatenate([v_leg_local, jp.zeros((4, 1))], axis=1)
        v_leg_global = jax.vmap(rotate, in_axes=(0, None))(v_leg_local_3d, quat)[:, :2]

        # 4. Calculate Raibert target (Mid-stance projection)
        hip_pos = data.xpos[self.hip_inds][:, :2]
        t_stance = self.gait_period / 2.0
        t_swing = self.gait_period / 2.0

        raibert_offset = (t_swing + 0.5 * t_stance) * v_leg_global

        # Hip -> foot offsets (base-local) rotated to global
        foot_offset_local_3d = jp.concatenate([self.foot_offsets_xy, jp.zeros((4, 1))], axis=1)
        foot_offset_global = jax.vmap(rotate, in_axes=(0, None))(foot_offset_local_3d, quat)[:, :2]

        raibert_xy = hip_pos + foot_offset_global + raibert_offset

        # 6. 更新目标 (Phase Alignment)
        # Pair 1: FL(0), RR(3) -> 对应 Odd Step (Block 2) 摆动
        # Pair 2: FR(1), RL(2) -> 对应 Even Step (Block 1) 摆动
        pair1 = jp.array([0, 3])
        pair2 = jp.array([1, 2])

        cur_tars = info['xy*']

        # Even Step 开始瞬间 -> FR/RL (Pair 2) 更新目标
        tars_p2 = cur_tars.at[pair2].set(raibert_xy[pair2])

        # Odd Step 开始瞬间 -> FL/RR (Pair 1) 更新目标
        tars_p1 = cur_tars.at[pair1].set(raibert_xy[pair1])

        xy_tars = jp.where(new_step & even_step, tars_p2, cur_tars)
        xy_tars = jp.where(new_step & (~even_step), tars_p1, xy_tars)
        info['xy*'] = xy_tars

        # 记录起跳点 (用于插值 Reward)
        feet_pos = data.geom_xpos[self.feet_inds][:, :2]
        info['xy0'] = jp.where(new_step, feet_pos, info['xy0'])
        info['k0'] = jp.where(new_step, s, info['k0'])

        # 记录当前触地点高度 (用于摆线高度)
        feet_z = data.site_xpos[self._feet_site_id][:, 2]
        info['z0'] = jp.where(new_step, feet_z, info['z0'])

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
            "feet_traj":      self._cost_foot_traj_tracking(data, info, extra_args) * scales.feet_traj,
            
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

    def _cost_foot_traj_tracking(self, data, info, extra_args):
        curr_feet = data.geom_xpos[self.feet_inds]
        ref_pos = info['foot_ref_pos']
        swing_mask = info['foot_swing'][:, None]
        pos_err = jp.sum(jp.square((curr_feet - ref_pos) * swing_mask))

        vel_err = 0.0
        if self._foot_linvel_sensor_adr is not None:
            feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
            vel_xy = feet_vel[..., :2]
            ref_v_xy = info['foot_ref_v_xy']
            vel_err = jp.sum(jp.square((vel_xy - ref_v_xy) * swing_mask[:, :2]))

        return (pos_err + self._foot_traj_vel_weight * vel_err) * extra_args['move_mask']

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
        同时检查以下内容是否严格对齐：
        1. Reward: 抬腿高度波形
        2. Raibert: 落点目标更新
        3. Ref Kino: 参考动作的关节角度
        4. Swing mask 与相位切换
        并额外记录一个脚的实际高度（来自当前模型前向运动学）
        """
        print("Running Ultimate Phase Alignment Check (Height + Actual + Raibert + Kino)...")
        import matplotlib.pyplot as plt

        # 模拟 4 个周期
        steps = np.arange(self.step_k * 4)

        # 日志
        height_target_log = []   # Reward 要求的高度（摆线参考）
        height_actual_log = []   # 实际高度（当前模型前向运动学）
        raibert_change_log = []  # Raibert 落点是否发生更新 (Event)
        kino_thigh_log = []      # Ref Kino 中的髋关节角度
        swing_mask_fl_log = []   # FL 摆动 mask
        swing_mask_fr_log = []   # FR 摆动 mask
        foot_idx = 0             # 随便选一个脚，这里选 FL

        # 伪造一个恒定的指令：向前走 1.0 m/s
        cmd = jp.array([1.0, 0.0, 0.0])

        # 初始化状态用于 Raibert 迭代
        info = {
            'step': jp.array(0, dtype=jp.int32),
            'k0': jp.array(0, dtype=jp.int32),
            'command': cmd,
            'xy0': jp.zeros((4, 2)),
            'xy*': jp.zeros((4, 2)),
            'foot_phase': 0.0,
            'foot_swing': jp.zeros(4),
            'foot_ref_xy': jp.zeros((4, 2)),
            'foot_ref_z': jp.zeros(4),
            'foot_ref_pos': jp.zeros((4, 3)),
            'foot_ref_v_xy': jp.zeros((4, 2)),
            'z0': jp.zeros(4),
        }

        last_fl_target_x = 0.0

        for s in steps:
            # 更新 step
            info['step'] = jp.array(s, dtype=jp.int32)
            step_idx = int(s % self.l_cycle)

            # Build ref data
            qpos_ref = self.kinematic_ref_qpos[step_idx]
            ref_data = mjx_env.make_data(
                self.mj_model,
                qpos=qpos_ref,
                qvel=jp.zeros(self.mjx_model.nv),
                ctrl=jp.zeros(self.mjx_model.nu),
                impl=self.mjx_model.impl.value,
                nconmax=self._config.nconmax,
                njmax=self._config.njmax,
            )
            ref_data = mjx.forward(self.mjx_model, ref_data)

            # Update Raibert + cycloid ref
            self._update_raibert_target(ref_data, info)
            self._update_foot_cycloid_ref(info)

            # --- 1. Check Reward (Height Wave) ---
            h_target = float(info['foot_ref_z'][foot_idx])
            height_target_log.append(h_target)

            h_actual = float(ref_data.geom_xpos[self.feet_inds[foot_idx], 2])
            height_actual_log.append(h_actual)

            # --- 2. Check Raibert (Target Update) ---
            current_fl_target_x = float(info['xy*'][0, 0])

            if s > 0 and abs(current_fl_target_x - last_fl_target_x) > 1e-4:
                raibert_change_log.append(0.12)
            else:
                raibert_change_log.append(0.0)

            last_fl_target_x = current_fl_target_x

            # --- 3. Check Ref Kino (Joint Angle) ---
            kin_ref = self.kinematic_ref_qpos[step_idx]
            fl_thigh_angle = float(kin_ref[8])
            kino_thigh_log.append(fl_thigh_angle)
            swing_mask_fl_log.append(float(info['foot_swing'][0]))
            swing_mask_fr_log.append(float(info['foot_swing'][1]))

        # --- 绘图验证 ---
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # 1. Height target + actual (Green + Orange)
        ax1.plot(steps, height_target_log, 'g-', label='Reward Target Height (FL)', linewidth=2)
        ax1.plot(steps, height_actual_log, color='orange', linestyle='-', label='Actual Height (FL)', linewidth=1.5, alpha=0.8)
        ax1.set_ylabel('Height (m)')
        ax1.set_ylim(0, 0.15)

        # 2. Ref Kino (Blue)
        ax2 = ax1.twinx()
        ax2.plot(steps, kino_thigh_log, 'b--', label='Ref Kino: FL Thigh Angle', linewidth=2, alpha=0.6)
        ax2.set_ylabel('Joint Angle (rad)')

        # 3. Raibert Update (Red Bars)
        ax1.bar(steps, raibert_change_log, width=1.0, color='red', alpha=0.3, label='Raibert: Target Update Event')
        ax1.plot(steps, np.array(swing_mask_fl_log) * 0.015, color='black', linestyle='--', linewidth=1.5, label='Swing Mask FL (scaled)')
        ax1.plot(steps, np.array(swing_mask_fr_log) * 0.015, color='gray', linestyle='--', linewidth=1.5, label='Swing Mask FR (scaled)')

        # 4. 标注足端最高点（实际高度）
        height_actual_arr = np.array(height_actual_log)
        peak_idx = int(np.argmax(height_actual_arr))
        peak_step = steps[peak_idx]
        peak_height = float(height_actual_arr[peak_idx])
        ax1.scatter([peak_step], [peak_height], color='orange', s=50, zorder=5)
        ax1.annotate(
            f"peak={peak_height:.4f} m",
            xy=(peak_step, peak_height),
            xytext=(peak_step + self.step_k * 0.1, peak_height + 0.01),
            arrowprops=dict(arrowstyle="->", color="orange", lw=1.2),
            color="orange",
            fontsize=10,
        )

        plt.title(f"Multi-Phase Alignment Check (Step K={self.step_k})")
        plt.axvspan(0, self.step_k, color='gray', alpha=0.1, label='Even Step (Stance)')
        plt.axvspan(self.step_k, self.step_k*2, color='green', alpha=0.1, label='Odd Step (Swing)')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.grid(True)
        plt.show()


    def play_cycloid_foot_trajectory(
        self,
        command: Optional[np.ndarray] = None,
        foot: str = "FL_foot",
        num_steps: Optional[int] = None,
        render_every: int = 2,
        save_path: Optional[str] = None,
        camera: Optional[str] = None,
        trail_stride: int = 2,
        trail_max: int = 80,
    ):
        """
        Visualize the reward foot trajectory while playing ref kino.

        Args:
            command: Fixed command [vx, vy, wz] used by Raibert target update.
            foot: Foot site name, e.g. "FL_foot", "FR_foot", "RL_foot", "RR_foot".
            num_steps: Number of frames to visualize. Defaults to one ref cycle.
            render_every: Render every Nth frame.
            save_path: Optional output video path.
            camera: Optional camera name.
            trail_stride: Downsampled trail stride for path visualization.
            trail_max: Max number of trail points to draw per frame.
        """
        print("Playing reward foot trajectory with ref kino...")

        if command is None:
            command = np.array([1.0, 0.0, 0.0])
        command = np.asarray(command, dtype=np.float32)

        # Resolve foot index
        if foot in consts.FEET_SITES:
            foot_idx = consts.FEET_SITES.index(foot)
        else:
            raise ValueError(f"Unknown foot site: {foot}. Expected one of {consts.FEET_SITES}.")

        # Use ref kino trajectory for rendering
        if num_steps is None:
            num_steps = int(self.l_cycle)
        ref_qpos = np.array(self.kinematic_ref_qpos[:num_steps])

        # Prepare Raibert info
        info = {
            'step': jp.array(0, dtype=jp.int32),
            'k0': jp.array(0, dtype=jp.int32),
            'command': jp.array(command),
            'xy0': jp.zeros((4, 2)),
            'xy*': jp.zeros((4, 2)),
            'foot_phase': 0.0,
            'foot_swing': jp.zeros(4),
            'foot_ref_xy': jp.zeros((4, 2)),
            'foot_ref_z': jp.zeros(4),
            'foot_ref_pos': jp.zeros((4, 3)),
            'foot_ref_v_xy': jp.zeros((4, 2)),
            'z0': jp.zeros(4),
        }

        # Trajectory container
        path_xyz = np.zeros((num_steps, 3), dtype=np.float32)

        for i in range(num_steps):
            info['step'] = jp.array(i, dtype=jp.int32)

            # Build mjx data from ref kino at this step (for Raibert hip positions + quat)
            ref_data = mjx_env.make_data(
                self.mj_model,
                qpos=self.kinematic_ref_qpos[i],
                qvel=jp.zeros(self.mjx_model.nv),
                ctrl=jp.zeros(self.mjx_model.nu),
                impl=self.mjx_model.impl.value,
                nconmax=self._config.nconmax,
                njmax=self._config.njmax,
            )
            ref_data = mjx.forward(self.mjx_model, ref_data)

            # Update Raibert targets from fixed command
            self._update_raibert_target(ref_data, info)
            self._update_foot_cycloid_ref(info)

            path_xyz[i] = np.array(info['foot_ref_pos'][foot_idx])

        # Static rendering uses ref kino qpos (playing ref motion)
        qpos_traj = ref_qpos

        def _add_sphere(scn, pos, radius, rgba):
            if scn.ngeom >= scn.maxgeom:
                return
            scn.ngeom += 1
            scn.geoms[scn.ngeom - 1].category = mujoco.mjtCatBit.mjCAT_DECOR
            mujoco.mjv_initGeom(
                geom=scn.geoms[scn.ngeom - 1],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([radius, 0.0, 0.0]),
                pos=np.array(pos),
                mat=np.eye(3).flatten().astype(np.float32),
                rgba=np.asarray(rgba).astype(np.float32),
            )

        modify_scene_fns = []
        for i in range(num_steps):
            def make_fn(idx=i):
                def _fn(scn):
                    trail = path_xyz[: idx + 1 : max(1, trail_stride)]
                    if trail_max is not None and len(trail) > trail_max:
                        trail = trail[-trail_max:]
                    for p in trail:
                        _add_sphere(scn, p, radius=0.007, rgba=[0.2, 0.8, 0.2, 0.6])
                    _add_sphere(scn, path_xyz[idx], radius=0.012, rgba=[0.9, 0.3, 0.2, 0.9])
                    _add_sphere(scn, path_xyz[0], radius=0.009, rgba=[0.2, 0.4, 0.9, 0.9])
                    _add_sphere(scn, path_xyz[-1], radius=0.009, rgba=[0.2, 0.4, 0.9, 0.9])
                return _fn
            modify_scene_fns.append(make_fn())

        frames = self._render_trajectory(
            trajectory=qpos_traj,
            render_every=render_every,
            height=480,
            width=640,
            camera=camera,
            save_path=save_path,
            modify_scene_fns=modify_scene_fns[::render_every],
        )

        fps = 1.0 / (self.dt * render_every)
        media.show_video(frames, fps=fps, loop=True)
        return frames, path_xyz
