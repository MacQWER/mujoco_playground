from typing import Any, Dict, Optional, Union, Callable
import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
from ml_collections import config_dict

from brax.io import model

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2.base import Go2Env
from mujoco_playground._src.locomotion.go2 import go2_constants as consts
from mujoco_playground._src.locomotion.go2.Util.TrotUtil import (
    make_kinematic_ref, cos_wave, dcos_wave, rotate_inv
)

from mujoco_playground._src.locomotion.go2.configs import joystick_config
from mujoco_playground._src.locomotion.go2.mdp import commands as command_lib
from mujoco_playground._src.locomotion.go2.mdp import event as event_lib
from mujoco_playground._src.locomotion.go2.mdp import rewards as reward_lib
from mujoco_playground._src.locomotion.go2.Util import JoystickUtil as joystick_utils


def default_config() -> config_dict.ConfigDict:
    return joystick_config.default_config()


class JoystickGo2(Go2Env):
    """
    JoystickGo2 with Residual Learning support.
    Uses an 'Anchor Policy' (trained on TrotGo2) to provide a base gait,
    and learns a residual policy to achieve omnidirectional velocity tracking.
    """
    
    def __init__(self,
                 task: str = None, 
                 config: config_dict.ConfigDict = joystick_config.default_config(), 
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
            self._anchor_inference_fn = joystick_utils.get_anchor_inference_fn(
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
        
        # 5. Kinematic Reference
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
        self._nominal_base_height = self._init_q[2]

        # Step height parameters (coupled to step length; capped)
        self._step_height_max = float(getattr(self._config.env, "step_height", 0.1128))
        self._step_height_min = float(getattr(self._config.env, "step_height_min", 0.0))
        self._foot_traj_vel_weight = float(getattr(self._config.env, "foot_traj_vel_weight", 0.2))

        self._init_active_rewards(reward_lib)


    def _get_swing_mask(self, step):
        return joystick_utils.get_swing_mask(self, step)

    def _update_foot_cycloid_ref(self, info):
        joystick_utils.update_foot_cycloid_ref(self, info)

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

        cmd = command_lib.sample_command(key_cmd, cmd_a=self._cmd_a, cmd_b=self._cmd_b)

        hip_pos = data.xpos[self.hip_inds][:, :2]
        feet_pos = data.geom_xpos[self.feet_inds][:, :2]
        

        # 初始化 State Info (包含新增变量)
        state_info = {
            'rng': rng,
            'step': jp.array(0, dtype=jp.int32),
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
            'reward_tuple': {k: 0.0 for k in self._config.rewards.scales.keys()}
        }
        state_info = command_lib.init_command_state(
            state_info,
            command=cmd,
            steps_until_next_cmd=jp.array(100, dtype=jp.int32),
        )
        state_info = event_lib.init_disturbance(
            state_info,
            disturbance_cfg=self._config.disturbance,
            dt=self.dt,
            prefix="pert",
        )
        self._update_foot_cycloid_ref(state_info)

        # Anchor Inference
        anchor_obs = self._get_anchor_obs(data, state_info)
        state_info["rng"], key_anchor = jax.random.split(state_info["rng"])
        if self._anchor_inference_fn is None:
            anchor_act = jp.zeros(12)
        else:
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
        # 0. Add disturbance.
        if self._config.disturbance.enable:
            state = event_lib.maybe_apply_disturbance(
                state,
                disturbance_cfg=self._config.disturbance,
                dt=self.dt,
                base_mass=self.base_mass,
                nbody=self.mjx_model.nbody,
                base_id=self.base_id,
                prefix="pert",
            )

        info = state.info

        # 1. Mix anchor and residual actions into the applied control.
        action = jp.clip(action, -1.0, 1.0)
        anchor_act = info['anchor_action']
        mixed_action = (
            anchor_act * self.anchor_action_scale
            + action * self.residual_action_scale
        )
        ctrl = self.action_loc + mixed_action

        # 2. Physics step.
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

        # 3. Contact-derived transition quantities use pre-step history.
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        contact = jax.nn.sigmoid((0.025 - foot_z) * 100.0)
        delta_contact = jax.nn.relu(contact - info["last_contact"])
        first_contact = (info["feet_air_time"] > 0.0) * delta_contact
        feet_air_time = info["feet_air_time"] + self.dt
        swing_peak = jp.maximum(info["swing_peak"], foot_z)

        # 4. Advance gait-phase references for this transition reward.
        self._update_raibert_target(data, info)
        info['step'] += 1
        self._update_foot_cycloid_ref(info)

        # 5. Reward and termination use the transition state plus pre-step action history.
        up_z = self.get_upvector(data)[-1]
        tilt_threshold = jp.cos(jp.deg2rad(45.0))
        fall_termination = up_z < tilt_threshold
        done = jp.where(fall_termination, 1.0, 0.0)
        soft_done = jax.nn.sigmoid((tilt_threshold - up_z) * 100.0)

        reward_kwargs = {
            'first_contact': first_contact,
            'contact': contact,
            'soft_done': soft_done,
        }
        reward_dict = self._get_reward(data, ctrl, info, reward_kwargs, done)
        reward = sum(reward_dict.values()) * self.dt

        # 6. Commit post-step history for the next observation / next action.
        info["feet_air_time"] = feet_air_time * (1.0 - contact)
        info["last_contact"] = contact
        info["swing_peak"] = swing_peak * (1.0 - contact)
        info["last_residual"] = action
        info["last_action"] = ctrl

        info = command_lib.update_command(
            info,
            dt=self.dt,
            cmd_a=self._cmd_a,
            cmd_b=self._cmd_b,
        )

        anchor_obs = self._get_anchor_obs(data, info)
        info["rng"], key_anchor = jax.random.split(info["rng"])
        if self._anchor_inference_fn is None:
            next_anchor_act = jp.zeros(12)
        else:
            next_anchor_act, _ = self._anchor_inference_fn(anchor_obs, key_anchor)
            next_anchor_act = jp.clip(next_anchor_act, -1.0, 1.0)
            next_anchor_act = jax.lax.stop_gradient(next_anchor_act)
        info['anchor_action'] = next_anchor_act

        for k, v in reward_dict.items():
            state.metrics[k] = v
        info['reward_tuple'] = reward_dict

        # 7. Next observation.
        residual_obs = self._get_residual_obs(data, info)

        return state.replace(data=data, obs=residual_obs, reward=reward, done=done)

    # -------- Observation Generators --------

    def _get_obs_context(self) -> Dict[str, Any]:
        return {
            "default_ap_pose": self._default_ap_pose,
            "l_cycle": self.l_cycle,
            "kin_ref_qpos": self.kinematic_ref_qpos,
        }

    def _get_obs(self, data, info):
        return self._build_obs(data, info, self._config.obs.policy_terms)

    def _get_anchor_obs(self, data, info):
        return self._build_obs(data, info, self._config.obs.anchor_terms)

    def _get_residual_obs(self, data, info):
        return self._get_obs(data, info)

    # -------- Helpers --------

    # -------- Raibert Heuristic (Standard Hip Logic) --------
    def _update_raibert_target(self, data, info):
        joystick_utils.update_raibert_target(self, data, info)

    # =========================================================================
    #                    Reward Logic (Updated with Masks)
    # =========================================================================

    def _get_reward_context(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        extra_args: dict[str, Any],
    ) -> dict[str, Any]:
        del data

        # Keep masks behavior unchanged for now.
        # move_mask = jax.nn.sigmoid((jp.linalg.norm(info['command']) - 0.01) * 200.0)
        # still_mask = jax.nn.sigmoid((0.01 - jp.linalg.norm(info['command'])) * 200.0)
        move_mask = 1.0
        still_mask = 0.0

        reward_kwargs = dict(extra_args)
        reward_kwargs.update(
            move_mask=move_mask,
            still_mask=still_mask,
            feet_inds=self.feet_inds,
            foot_linvel_sensor_adr=self._foot_linvel_sensor_adr,
            nominal_base_height=self._nominal_base_height,
            kinematic_ref_qpos=self.kinematic_ref_qpos,
            kinematic_ref_qvel=self.kinematic_ref_qvel,
            l_cycle=self.l_cycle,
            get_upvector=self.get_upvector,
            soft_lowers=self._soft_lowers,
            soft_uppers=self._soft_uppers,
            default_ap_pose=self._default_ap_pose,
            current_action=action,
        )
        return reward_kwargs

    # --- Debug Util ---

    def check_phase_alignment(self):
        return joystick_utils.check_phase_alignment(self)

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
        return joystick_utils.play_cycloid_foot_trajectory(
            self,
            command=command,
            foot=foot,
            num_steps=num_steps,
            render_every=render_every,
            save_path=save_path,
            camera=camera,
            trail_stride=trail_stride,
            trail_max=trail_max,
        )
