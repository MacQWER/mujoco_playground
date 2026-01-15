from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import mediapy as media
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from mujoco_playground._src import mjx_env

from mujoco_playground._src.locomotion.go2 import go2_constants as consts

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
        # Modify PD gains.
        self._mj_model.dof_damping[6:] = config.Kd
        self._mj_model.actuator_gainprm[:, 0] = config.Kp
        self._mj_model.actuator_biasprm[:, 1] = -config.Kp

        # Increase offscreen framebuffer size to render at higher resolutions.
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        self._imu_site_id = self._mj_model.site("imu").id

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
        """
        Unified optimized renderer supporting both qpos arrays and Brax States,
        with optional scene modifications.
        """
        # 1. 初始化渲染器
        renderer = mujoco.Renderer(self._mj_model, height=height, width=width)
        camera_id = -1
        if camera is not None:
            camera_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            
        d = mujoco.MjData(self._mj_model)
        out = []

        # 2. 数据准备与搬运 (GPU -> CPU)
        # ---------------------------------------------------------
        # 情况 A: 输入是 Numpy 或 JAX 数组 (通常是 qpos)
        # 来自: play_ref_motion, render_rollout
        if isinstance(trajectory, (jp.ndarray, np.ndarray)):
            traj_qpos = jax.device_get(trajectory)
            traj_qvel, traj_mocap_pos, traj_mocap_quat = None, None, None
            
        # 情况 B: 输入是 Brax State 列表
        # 来自: 你的复杂场景渲染 (traj = rollout[::render_every])
        elif isinstance(trajectory, list):
            # 辅助函数：如果属性存在，则堆叠并传回 CPU；否则返回 None
            def get_attr(states, attr):
                if not hasattr(states[0].data, attr): return None
                return jax.device_get(jp.stack([getattr(s.data, attr) for s in states]))

            traj_qpos = get_attr(trajectory, 'qpos')
            traj_qvel = get_attr(trajectory, 'qvel')
            traj_mocap_pos = get_attr(trajectory, 'mocap_pos')
            traj_mocap_quat = get_attr(trajectory, 'mocap_quat')
        else:
            raise ValueError(f"Unsupported trajectory type: {type(trajectory)}")

        # 3. 渲染循环 (CPU)
        # ---------------------------------------------------------
        # 处理下采样：
        # 如果外部已经切片了(render_every=1)，这里 indices 就是 [0, 1, 2...]
        # 如果外部传了完整序列，这里负责切片
        n_frames = len(traj_qpos)
        indices = np.arange(0, n_frames, render_every)

        try:
            for i, idx in enumerate(indices):
                # 更新 mj_data
                d.qpos[:] = traj_qpos[idx]
                if traj_qvel is not None: d.qvel[:] = traj_qvel[idx]
                if traj_mocap_pos is not None: d.mocap_pos[:] = traj_mocap_pos[idx]
                if traj_mocap_quat is not None: d.mocap_quat[:] = traj_mocap_quat[idx]
                
                # 正向运动学
                mujoco.mj_forward(self._mj_model, d)
                
                # 更新场景 (支持 scene_option)
                renderer.update_scene(d, camera=camera_id, scene_option=scene_option)
                
                # 应用自定义场景修改 (支持 modify_scene_fns)
                if modify_scene_fns is not None:
                    # 这里的逻辑假设 modify_scene_fns 的长度与"最终渲染帧数"一致
                    # 如果外部切片了 traj，也必须切片 modify_scene_fns
                    if i < len(modify_scene_fns):
                        modify_scene_fns[i](renderer.scene)
                
                out.append(renderer.render())
                
        except Exception as e:
            print(f"Rendering failed at frame {i}: {e}")
        finally:
            renderer.close()

        # 4. 保存视频
        if save_path:
            fps = 1.0 / (self.dt * render_every)
            print(f"Saving video to {save_path} (FPS={fps:.1f})...")
            media.write_video(save_path, out, fps=fps)

        return out
