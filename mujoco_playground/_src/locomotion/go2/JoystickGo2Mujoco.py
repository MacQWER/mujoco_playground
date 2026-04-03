from typing import Any, Dict, Mapping, Optional, Sequence, Union

from flax import struct
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import go2_constants as consts
from mujoco_playground._src.locomotion.go2 import managers as manager_lib
from mujoco_playground._src.locomotion.go2.Util.JoystickUtil import (
    get_swing_mask,
    update_foot_cycloid_ref,
    update_raibert_target,
)
from mujoco_playground._src.locomotion.go2.Util.TrotUtil import (
    cos_wave,
    dcos_wave,
    make_kinematic_ref,
    rotate_inv,
)
from mujoco_playground._src.locomotion.go2.configs import joystick_config
from mujoco_playground._src.locomotion.go2.mdp import observations as observation_lib
from mujoco_playground._src.locomotion.go2.mdp import rewards as reward_lib


@struct.dataclass
class MujocoDataSnapshot:
    """Array snapshot of MuJoCo state used by shared reward/obs code."""

    qpos: jax.Array
    qvel: jax.Array
    ctrl: jax.Array
    act: jax.Array
    mocap_pos: jax.Array
    mocap_quat: jax.Array
    xfrc_applied: jax.Array
    sensordata: jax.Array
    xpos: jax.Array
    xquat: jax.Array
    site_xpos: jax.Array
    site_xmat: jax.Array
    geom_xpos: jax.Array
    cvel: jax.Array
    actuator_force: jax.Array


def default_config() -> config_dict.ConfigDict:
    cfg = joystick_config.default_config()
    cfg.backend = "mujoco"
    cfg.assistive_wrench.enable = False
    cfg.disturbance.enable = False

    cfg.mujoco_motor_controller = config_dict.ConfigDict()
    cfg.mujoco_motor_controller.kp = 20.0
    cfg.mujoco_motor_controller.kd = 1.0

    cfg.mujoco_model = config_dict.ConfigDict()
    cfg.mujoco_model.xml_path = consts.MUJOCO_XML_PATH.as_posix()
    cfg.mujoco_model.body_mass = config_dict.ConfigDict()
    cfg.mujoco_model.body_inertia = config_dict.ConfigDict()
    cfg.mujoco_model.geom_friction = config_dict.ConfigDict()
    cfg.mujoco_model.geom_solref = config_dict.ConfigDict()
    cfg.mujoco_model.geom_solimp = config_dict.ConfigDict()
    cfg.mujoco_model.joint_damping = config_dict.ConfigDict()
    cfg.mujoco_model.joint_armature = config_dict.ConfigDict()
    cfg.mujoco_model.joint_frictionloss = config_dict.ConfigDict()
    return cfg




def make_mujoco_model_template() -> config_dict.ConfigDict:
    """Example twin-model overrides for JoystickGo2Mujoco.

    Fill in only the entries you want to override. Keys must match MuJoCo
    body / geom / joint names in the loaded XML. Scalar joint overrides apply
    to 1-DoF joints; multi-DoF joints should use a full vector.
    """
    cfg = config_dict.ConfigDict()
    cfg.xml_path = consts.MUJOCO_XML_PATH.as_posix()

    cfg.body_mass = config_dict.ConfigDict({
        # "base": 7.8,
    })
    cfg.body_inertia = config_dict.ConfigDict({
        # "base": [0.12, 0.25, 0.29],
    })
    cfg.geom_friction = config_dict.ConfigDict({
        # "FL": [1.2, 0.02, 0.01],
        # "FR": [1.2, 0.02, 0.01],
        # "RL": [1.2, 0.02, 0.01],
        # "RR": [1.2, 0.02, 0.01],
    })
    cfg.geom_solref = config_dict.ConfigDict({
        # "FL": [0.015, 1.0],
    })
    cfg.geom_solimp = config_dict.ConfigDict({
        # "FL": [0.9, 0.95, 0.001, 0.5, 2.0],
    })
    cfg.joint_damping = config_dict.ConfigDict({
        # "FL_thigh_joint": 0.8,
    })
    cfg.joint_armature = config_dict.ConfigDict({
        # "FL_thigh_joint": 0.01,
    })
    cfg.joint_frictionloss = config_dict.ConfigDict({
        # "FL_thigh_joint": 0.02,
    })
    return cfg

def _cfg_mapping(cfg: Optional[config_dict.ConfigDict]) -> Mapping[str, Any]:
    if cfg is None:
        return {}
    return cfg.to_dict()


def _snapshot_from_data(data: mujoco.MjData) -> MujocoDataSnapshot:
    return MujocoDataSnapshot(
        qpos=jp.asarray(np.array(data.qpos, copy=True)),
        qvel=jp.asarray(np.array(data.qvel, copy=True)),
        ctrl=jp.asarray(np.array(data.ctrl, copy=True)),
        act=jp.asarray(np.array(data.act, copy=True)),
        mocap_pos=jp.asarray(np.array(data.mocap_pos, copy=True)),
        mocap_quat=jp.asarray(np.array(data.mocap_quat, copy=True)),
        xfrc_applied=jp.asarray(np.array(data.xfrc_applied, copy=True)),
        sensordata=jp.asarray(np.array(data.sensordata, copy=True)),
        xpos=jp.asarray(np.array(data.xpos, copy=True)),
        xquat=jp.asarray(np.array(data.xquat, copy=True)),
        site_xpos=jp.asarray(np.array(data.site_xpos, copy=True)),
        site_xmat=jp.asarray(np.array(data.site_xmat, copy=True).reshape(-1, 3, 3)),
        geom_xpos=jp.asarray(np.array(data.geom_xpos, copy=True)),
        cvel=jp.asarray(np.array(data.cvel, copy=True)),
        actuator_force=jp.asarray(np.array(data.actuator_force, copy=True)),
    )


def _build_data_from_snapshot(
    model: mujoco.MjModel, snapshot: Optional[MujocoDataSnapshot]
) -> mujoco.MjData:
    data = mujoco.MjData(model)
    if snapshot is None:
        return data

    data.qpos[:] = np.asarray(snapshot.qpos)
    data.qvel[:] = np.asarray(snapshot.qvel)
    if model.nu:
        data.ctrl[:] = np.asarray(snapshot.ctrl)
    if model.na:
        data.act[:] = np.asarray(snapshot.act)
    if model.nmocap:
        data.mocap_pos[:] = np.asarray(snapshot.mocap_pos)
        data.mocap_quat[:] = np.asarray(snapshot.mocap_quat)
    data.xfrc_applied[:] = np.asarray(snapshot.xfrc_applied)
    mujoco.mj_forward(model, data)
    return data


def _min_penetration(data: mujoco.MjData) -> float:
    if data.ncon == 0:
        return 0.0
    return min(float(data.contact[i].dist) for i in range(data.ncon))


def _set_named_vector(
    array: np.ndarray,
    idx: Union[int, slice],
    value: Union[float, Sequence[float]],
    expected_dim: int,
) -> None:
    if np.isscalar(value):
        if expected_dim != 1:
            raise ValueError(f"Expected sequence of length {expected_dim}, got scalar.")
        array[idx] = float(value)
        return
    value_arr = np.asarray(value, dtype=array.dtype)
    if value_arr.shape != (expected_dim,):
        raise ValueError(f"Expected shape {(expected_dim,)}, got {value_arr.shape}.")
    array[idx] = value_arr


def _joint_dof_slice(model: mujoco.MjModel, joint_id: int) -> slice:
    start = model.jnt_dofadr[joint_id]
    width = mjx_env.dof_width(model.jnt_type[joint_id])
    return slice(start, start + width)


def _apply_model_overrides(model: mujoco.MjModel, cfg: Optional[config_dict.ConfigDict]) -> None:
    model_cfg = _cfg_mapping(cfg)
    for body_name, mass in model_cfg.get("body_mass", {}).items():
        model.body_mass[model.body(body_name).id] = float(mass)
    for body_name, inertia in model_cfg.get("body_inertia", {}).items():
        _set_named_vector(model.body_inertia, model.body(body_name).id, inertia, 3)
    for geom_name, friction in model_cfg.get("geom_friction", {}).items():
        _set_named_vector(model.geom_friction, model.geom(geom_name).id, friction, 3)
    for geom_name, solref in model_cfg.get("geom_solref", {}).items():
        _set_named_vector(model.geom_solref, model.geom(geom_name).id, solref, 2)
    for geom_name, solimp in model_cfg.get("geom_solimp", {}).items():
        _set_named_vector(model.geom_solimp, model.geom(geom_name).id, solimp, 5)
    for joint_name, damping in model_cfg.get("joint_damping", {}).items():
        joint_id = model.joint(joint_name).id
        dof_slice = _joint_dof_slice(model, joint_id)
        width = dof_slice.stop - dof_slice.start
        _set_named_vector(model.dof_damping, dof_slice, damping, width)
    for joint_name, armature in model_cfg.get("joint_armature", {}).items():
        joint_id = model.joint(joint_name).id
        dof_slice = _joint_dof_slice(model, joint_id)
        width = dof_slice.stop - dof_slice.start
        _set_named_vector(model.dof_armature, dof_slice, armature, width)
    for joint_name, frictionloss in model_cfg.get("joint_frictionloss", {}).items():
        joint_id = model.joint(joint_name).id
        dof_slice = _joint_dof_slice(model, joint_id)
        width = dof_slice.stop - dof_slice.start
        _set_named_vector(model.dof_frictionloss, dof_slice, frictionloss, width)


class JoystickGo2Mujoco(mjx_env.MjxEnv):
    """Native MuJoCo twin environment for JoystickGo2."""

    def __init__(
        self,
        task: str = None,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        del task
        super().__init__(config, config_overrides)

        self._xml_path = str(
            getattr(self._config.mujoco_model, "xml_path", consts.MUJOCO_XML_PATH.as_posix())
        )
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.actuator_gainprm[:, 0] = self._config.Kp
        self._mj_model.actuator_biasprm[:, 1] = -self._config.Kp
        _apply_model_overrides(self._mj_model, self._config.mujoco_model)

        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160
        self._mjx_model = None

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

        self._post_init()

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mj_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> None:
        return self._mjx_model

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        if not hasattr(self, "_observation_size"):
            obs = self.reset(jax.random.PRNGKey(0)).obs
            self._observation_size = jax.tree_util.tree_map(lambda x: x.shape, obs)
        return self._observation_size

    def get_upvector(self, data: MujocoDataSnapshot) -> jax.Array:
        return data.site_xmat[self._imu_site_id, :, 2]

    def get_gravity(self, data: MujocoDataSnapshot) -> jax.Array:
        return data.site_xmat[self._imu_site_id].T @ jp.array([0.0, 0.0, -1.0])

    def get_global_linvel(self, data: MujocoDataSnapshot) -> jax.Array:
        sensor_name = consts.GLOBAL_LINVEL_SENSOR
        if self._has_sensor(sensor_name):
            return mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
        return data.cvel[self.base_id, 3:]

    def get_global_angvel(self, data: MujocoDataSnapshot) -> jax.Array:
        sensor_name = consts.GLOBAL_ANGVEL_SENSOR
        if self._has_sensor(sensor_name):
            return mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
        return data.cvel[self.base_id, :3]

    def get_local_linvel(self, data: MujocoDataSnapshot) -> jax.Array:
        sensor_name = consts.LOCAL_LINVEL_SENSOR
        if self._has_sensor(sensor_name):
            return mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
        return rotate_inv(data.cvel[self.base_id, 3:], data.xquat[self.base_id])

    def get_accelerometer(self, data: MujocoDataSnapshot) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.ACCELEROMETER_SENSOR)

    def get_gyro(self, data: MujocoDataSnapshot) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

    def get_feet_pos(self, data: MujocoDataSnapshot) -> jax.Array:
        if self._feet_site_id is None:
            return data.geom_xpos[self.feet_inds]
        return data.site_xpos[self._feet_site_id]

    def _has_sensor(self, sensor_name: str) -> bool:
        try:
            self._mj_model.sensor(sensor_name)
        except KeyError:
            return False
        return True

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
        data: MujocoDataSnapshot,
        info: Dict[str, Any],
        terms: Sequence[tuple[str, Optional[str], float]],
    ) -> Dict[str, jax.Array]:
        return self.observation_manager.build(data, info, terms)

    def _get_reward_context(
        self,
        data: MujocoDataSnapshot,
        action: jax.Array,
        info: Dict[str, Any],
        extra_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        del data, info
        reward_kwargs = dict(extra_args)
        reward_kwargs.update(
            move_mask=1.0,
            still_mask=0.0,
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

    def _init_active_rewards(self, reward_lib_module: Any) -> None:
        self.reward_manager = manager_lib.RewardManager(
            reward_module=reward_lib_module,
            cfg=self._config,
            context_fn=self._get_reward_context,
        )
        self.active_rewards = {
            name: (scale, func) for name, scale, func in self.reward_manager.active_terms
        }

    def _get_reward(
        self,
        data: MujocoDataSnapshot,
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
        data: MujocoDataSnapshot,
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
            metrics = dict(info.get("reward_tuple", {}))
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
            nbody=self.mj_model.nbody,
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
        data: MujocoDataSnapshot,
        obs: Dict[str, jax.Array],
        reward: jax.Array,
        done: jax.Array,
        info: Dict[str, Any],
    ) -> mjx_env.State:
        info = self._sync_info(info)
        return state.replace(data=data, obs=obs, reward=reward, done=done, info=info)

    def _post_init(self) -> None:
        self._anchor_inference_fn = None
        if self._config.anchor.path:
            from mujoco_playground._src.locomotion.go2.Util import JoystickUtil as joystick_utils

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
        try:
            self._feet_site_id = np.array([self._mj_model.site(name).id for name in consts.FEET_SITES])
        except KeyError:
            self._feet_site_id = None

        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

        foot_linvel_sensor_adr = []
        try:
            for site in consts.FEET_SITES:
                sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
                sensor_adr = self._mj_model.sensor_adr[sensor_id]
                sensor_dim = self._mj_model.sensor_dim[sensor_id]
                foot_linvel_sensor_adr.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
            self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)
        except KeyError:
            self._foot_linvel_sensor_adr = None

        self.step_k = int(getattr(self._config.env, "step_k", 25))
        self.gait_scale = float(getattr(self._config.env, "gait_scale", 0.3))
        self.gait_period = self.step_k * 2 * self.dt

        kinematic_ref_qpos = make_kinematic_ref(cos_wave, self.step_k, scale=self.gait_scale, dt=self.dt)
        kinematic_ref_qvel = make_kinematic_ref(dcos_wave, self.step_k, scale=self.gait_scale, dt=self.dt)
        self.l_cycle = int(kinematic_ref_qpos.shape[0])

        kinematic_ref_qpos = np.array(kinematic_ref_qpos) + np.array(self._default_ap_pose)
        ref_qs = np.tile(self._init_q.reshape(1, 19), (self.l_cycle, 1))
        ref_qs[:, 7:] = kinematic_ref_qpos
        self.kinematic_ref_qpos = jp.array(ref_qs)

        ref_qvels = np.zeros((self.l_cycle, 18))
        ref_qvels[:, 6:] = np.array(kinematic_ref_qvel)
        self.kinematic_ref_qvel = jp.array(ref_qvels)

        self._cmd_a = jp.array(self._config.command_config.a)
        self._cmd_b = jp.array(self._config.command_config.b)

        init_data = mujoco.MjData(self.mj_model)
        init_data.qpos[:] = np.asarray(self._init_q)
        mujoco.mj_forward(self.mj_model, init_data)
        init_snapshot = _snapshot_from_data(init_data)

        base_pos = init_snapshot.xpos[self.base_id]
        hip_pos = init_snapshot.xpos[self.hip_inds]
        rel_pos = hip_pos - base_pos
        self.leg_offsets_x = rel_pos[:, 0]
        self.leg_offsets_y = rel_pos[:, 1]

        foot_pos = self.get_feet_pos(init_snapshot)
        base_quat = init_snapshot.xquat[self.base_id]
        hip_local = jax.vmap(rotate_inv, in_axes=(0, None))(hip_pos - base_pos, base_quat)
        foot_local = jax.vmap(rotate_inv, in_axes=(0, None))(foot_pos - base_pos, base_quat)
        self.foot_offsets_xy = (foot_local - hip_local)[:, :2]

        self.base_mass = self.mj_model.body("base").mass
        self._nominal_base_height = self._init_q[2]
        self._step_height_max = float(getattr(self._config.env, "step_height", 0.1128))
        self._step_height_min = float(getattr(self._config.env, "step_height_min", 0.0))
        self._foot_traj_vel_weight = float(getattr(self._config.env, "foot_traj_vel_weight", 0.2))
        self._actuator_ctrl_low = jp.array(self.mj_model.actuator_ctrlrange[:, 0])
        self._actuator_ctrl_high = jp.array(self.mj_model.actuator_ctrlrange[:, 1])
        self._ctrl_is_position_target = bool(np.any(self.mj_model.actuator_biastype != 0))

        self._init_active_rewards(reward_lib)

    def _get_swing_mask(self, step: jax.Array) -> jax.Array:
        return get_swing_mask(self, step)

    def _update_foot_cycloid_ref(self, info: Dict[str, Any]) -> None:
        update_foot_cycloid_ref(self, info)

    def _update_raibert_target(self, data: MujocoDataSnapshot, info: Dict[str, Any]) -> None:
        update_raibert_target(self, data, info)

    def _get_obs_context(self) -> Dict[str, Any]:
        return {
            "default_ap_pose": self._default_ap_pose,
            "l_cycle": self.l_cycle,
            "kin_ref_qpos": self.kinematic_ref_qpos,
        }

    def _get_anchor_obs(self, data: MujocoDataSnapshot, info: Dict[str, Any]) -> Dict[str, jax.Array]:
        return self._build_obs(data, info, self._config.obs.anchor_terms)

    def _get_residual_obs(self, data: MujocoDataSnapshot, info: Dict[str, Any]) -> Dict[str, jax.Array]:
        return self._build_obs(data, info, self._config.obs.policy_terms)

    def _compute_motor_ctrl(
        self, snapshot: MujocoDataSnapshot, target_qpos: jax.Array
    ) -> jax.Array:
        q = snapshot.qpos[7:19]
        qd = snapshot.qvel[6:18]
        kp = self._config.mujoco_motor_controller.kp
        kd = self._config.mujoco_motor_controller.kd
        tau = kp * (target_qpos - q) - kd * qd
        return jp.clip(tau, self._actuator_ctrl_low, self._actuator_ctrl_high)

    def _physics_step(
        self, snapshot: MujocoDataSnapshot, ctrl: jax.Array, n_substeps: int
    ) -> MujocoDataSnapshot:
        data = _build_data_from_snapshot(self.mj_model, snapshot)
        ctrl_np = np.asarray(ctrl)
        for _ in range(n_substeps):
            if self.mj_model.nu:
                data.ctrl[:] = ctrl_np
            mujoco.mj_step(self.mj_model, data)
        return _snapshot_from_data(data)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, key_cmd = jax.random.split(rng)
        qpos = self._init_q
        qvel = jp.zeros(self.mj_model.nv)

        data = mujoco.MjData(self.mj_model)
        data.qpos[:] = np.asarray(qpos)
        data.qvel[:] = np.asarray(qvel)
        if self.mj_model.nu:
            data.ctrl[:] = 0.0
        mujoco.mj_forward(self.mj_model, data)

        # The native MuJoCo model's home keyframe is already a valid standing pose.
        # Reusing the MJX penetration correction here can over-raise the base because
        # self-collisions are included in the raw contact set.
        snapshot = _snapshot_from_data(data)
        cmd = self.command_manager.sample(key_cmd, cmd_a=self._cmd_a, cmd_b=self._cmd_b)

        feet_pos = snapshot.geom_xpos[self.feet_inds][:, :2]
        state_info = {
            "rng": rng,
            "step": jp.array(0, dtype=jp.int32),
            "last_action": jp.zeros(self.mj_model.nu),
            "last_residual": jp.zeros(self.mj_model.nu),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4),
            "swing_peak": jp.zeros(4),
            "xy0": feet_pos,
            "xy*": feet_pos,
            "k0": jp.array(0, dtype=jp.int32),
            "foot_phase": jp.array(0.0),
            "foot_swing": jp.zeros(4),
            "foot_ref_xy": feet_pos,
            "foot_ref_z": snapshot.geom_xpos[self.feet_inds][:, 2],
            "foot_ref_pos": jp.concatenate([feet_pos, snapshot.geom_xpos[self.feet_inds][:, 2:3]], axis=1),
            "z0": self.get_feet_pos(snapshot)[:, 2],
            "foot_ref_v_xy": jp.zeros((4, 2)),
            "anchor_action": jp.zeros(12),
            "reward_tuple": {k: 0.0 for k in self._config.rewards.scales.keys()},
        }
        state_info = self.command_manager.init_state(
            state_info,
            command=cmd,
            steps_until_next_cmd=jp.array(100, dtype=jp.int32),
        )
        state_info = self.event_manager.init_state(
            state_info,
            disturbance_cfg=self._config.disturbance,
            dt=self.dt,
            prefix="pert",
        )
        state_info = self._sync_info(state_info)
        self._update_foot_cycloid_ref(state_info)

        anchor_obs = self._get_anchor_obs(snapshot, state_info)
        state_info["rng"], key_anchor = jax.random.split(state_info["rng"])
        if self._anchor_inference_fn is None:
            anchor_act = jp.zeros(12)
        else:
            anchor_act, _ = self._anchor_inference_fn(anchor_obs, key_anchor)
            anchor_act = jp.clip(anchor_act, -1.0, 1.0)
            anchor_act = jax.lax.stop_gradient(anchor_act)
        state_info["anchor_action"] = anchor_act

        residual_obs = self._get_residual_obs(snapshot, state_info)
        return self._init_state(snapshot, residual_obs, state_info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = self._run_event_pipeline(
            state,
            enabled=self._config.disturbance.enable,
            disturbance_cfg=self._config.disturbance,
            base_mass=self.base_mass,
            base_id=self.base_id,
            prefix="pert",
        )

        info = state.info
        action = self._clip_action(action, -1.0, 1.0)
        anchor_act = info["anchor_action"]
        mixed_action = anchor_act * self.anchor_action_scale + action * self.residual_action_scale
        target_qpos = self.action_loc + mixed_action
        if self._ctrl_is_position_target:
            applied_ctrl = target_qpos
        else:
            applied_ctrl = self._compute_motor_ctrl(state.data, target_qpos)

        data = self._physics_step(state.data, applied_ctrl, self.n_substeps)

        foot_pos = self.get_feet_pos(data)
        foot_z = foot_pos[..., -1]
        contact = jax.nn.sigmoid((0.025 - foot_z) * 100.0)
        delta_contact = jax.nn.relu(contact - info["last_contact"])
        first_contact = (info["feet_air_time"] > 0.0) * delta_contact
        feet_air_time = info["feet_air_time"] + self.dt
        swing_peak = jp.maximum(info["swing_peak"], foot_z)

        self._update_raibert_target(data, info)
        info["step"] += 1
        self._update_foot_cycloid_ref(info)

        up_z = self.get_upvector(data)[-1]
        tilt_threshold = jp.cos(jp.deg2rad(45.0))
        fall_termination = up_z < tilt_threshold
        done = self.termination_manager.build_done(jp.where(fall_termination, 1.0, 0.0))
        soft_done = jax.nn.sigmoid((tilt_threshold - up_z) * 100.0)

        reward_kwargs = {
            "first_contact": first_contact,
            "contact": contact,
            "soft_done": soft_done,
        }
        reward_dict = self._get_reward(data, target_qpos, info, reward_kwargs, done)
        reward = self._sum_reward_dict(reward_dict, self.dt)

        info["feet_air_time"] = feet_air_time * (1.0 - contact)
        info["last_contact"] = contact
        info["swing_peak"] = swing_peak * (1.0 - contact)
        info["last_residual"] = action
        info["last_action"] = target_qpos
        info = self.command_manager.update(
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
        info["anchor_action"] = next_anchor_act

        self._update_metrics(state.metrics, reward_dict)
        info["reward_tuple"] = reward_dict
        residual_obs = self._get_residual_obs(data, info)
        return self._finalize_step(
            state,
            data=data,
            obs=residual_obs,
            reward=reward,
            done=done,
            info=info,
        )
