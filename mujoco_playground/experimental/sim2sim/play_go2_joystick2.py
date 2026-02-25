"""Deploy JoystickGo2 (anchor + residual ONNX) to MuJoCo C API."""

import mujoco
import numpy as np
import onnxruntime as rt
from etils import epath
import mediapy as media

from mujoco_playground._src.locomotion.go2 import go2_constants as consts
from mujoco_playground._src.locomotion.go2.TrotUtil import (
    cos_wave,
    make_kinematic_ref,
    rotate_inv,
)

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"   


class Go2Joystick2OnnxController:
    """Sim2sim controller aligned with JoystickGo2.py logic."""

    JOINT_POS_SENSOR_NAMES = [
        "abduction_front_left_pos", "hip_front_left_pos", "knee_front_left_pos",
        "abduction_front_right_pos", "hip_front_right_pos", "knee_front_right_pos",
        "abduction_hind_left_pos", "hip_hind_left_pos", "knee_hind_left_pos",
        "abduction_hind_right_pos", "hip_hind_right_pos", "knee_hind_right_pos",
    ]

    JOINT_VEL_SENSOR_NAMES = [
        "abduction_front_left_vel", "hip_front_left_vel", "knee_front_left_vel",
        "abduction_front_right_vel", "hip_front_right_vel", "knee_front_right_vel",
        "abduction_hind_left_vel", "hip_hind_left_vel", "knee_hind_left_vel",
        "abduction_hind_right_vel", "hip_hind_right_vel", "knee_hind_right_vel",
    ]

    def __init__(
        self,
        anchor_policy_path: str,
        residual_policy_path: str,
        default_qpos: np.ndarray,
        ctrl_dt: float,
        n_substeps: int,
        action_scale: np.ndarray,
        command: np.ndarray,
    ):
        self._anchor_policy = rt.InferenceSession(
            anchor_policy_path, providers=["CPUExecutionProvider"]
        )
        self._residual_policy = rt.InferenceSession(
            residual_policy_path, providers=["CPUExecutionProvider"]
        )

        self._default_angles = default_qpos[7:].astype(np.float32)
        self._action_scale = action_scale.astype(np.float32)

        # Match JoystickGo2 state info semantics.
        self._last_action = np.zeros(12, dtype=np.float32)  # ctrl (not normalized action)
        self._anchor_action = np.zeros(12, dtype=np.float32)
        self._command = command.astype(np.float32)

        self._counter = 0
        self._n_substeps = n_substeps

        # Same gait reference setup as JoystickGo2._post_init.
        step_k = 13
        kin_q = make_kinematic_ref(cos_wave, step_k, scale=0.3, dt=ctrl_dt)
        self._kinematic_ref_angles = (np.array(kin_q) + self._default_angles).astype(np.float32)
        self._l_cycle = self._kinematic_ref_angles.shape[0]
        self._step = 0  # increments once per control update

    def _read_joint_pos(self, data: mujoco.MjData) -> np.ndarray:
        return np.array([data.sensor(n).data[0] for n in self.JOINT_POS_SENSOR_NAMES], dtype=np.float32)

    def _read_joint_vel(self, data: mujoco.MjData) -> np.ndarray:
        return np.array([data.sensor(n).data[0] for n in self.JOINT_VEL_SENSOR_NAMES], dtype=np.float32)

    def _get_g_local(self, data: mujoco.MjData) -> np.ndarray:
        quat = np.array(data.sensor("orientation").data, dtype=np.float32)
        return rotate_inv(np.array([0.0, 0.0, -1.0], dtype=np.float32), quat).astype(np.float32)

    def _get_anchor_obs(self, data: mujoco.MjData) -> np.ndarray:
        # Equivalent to JoystickGo2._get_anchor_obs without observation noise.
        yaw_rate = np.float32(data.sensor("gyro").data[2] * 0.25)
        g_local = self._get_g_local(data)
        angles = self._read_joint_pos(data)
        step_idx = self._step % self._l_cycle
        kin_ref = self._kinematic_ref_angles[step_idx]

        obs = np.concatenate([
            np.array([yaw_rate], dtype=np.float32),
            g_local,
            angles - self._default_angles,
            self._last_action,
            kin_ref,
        ])
        return np.clip(obs, -100.0, 100.0).astype(np.float32)

    def _get_residual_obs(self, data: mujoco.MjData) -> np.ndarray:
        # Equivalent to JoystickGo2._get_residual_obs without observation noise.
        v_local = np.array(data.sensor("local_linvel").data, dtype=np.float32)
        w_local = np.array(data.sensor("gyro").data, dtype=np.float32)
        g_local = self._get_g_local(data)
        angles = self._read_joint_pos(data)
        joint_vel = self._read_joint_vel(data)

        obs = np.concatenate([
            v_local,
            w_local,
            g_local,
            self._command,
            angles - self._default_angles,
            joint_vel,
            self._anchor_action,
        ])
        return np.clip(obs, -100.0, 100.0).astype(np.float32)

    def _infer_anchor(self, anchor_obs: np.ndarray) -> np.ndarray:
        actions, _ = self._anchor_policy.run(["actions", "std"], {"obs": anchor_obs.reshape(1, -1)})
        return np.clip(actions[0], -1.0, 1.0).astype(np.float32)

    def _infer_residual(self, residual_obs: np.ndarray) -> np.ndarray:
        actions, _ = self._residual_policy.run(["actions", "std"], {"obs": residual_obs.reshape(1, -1)})
        return np.clip(actions[0], -1.0, 1.0).astype(np.float32)

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData):
        if self._counter % self._n_substeps == 0:
            # Compute anchor action on current state.
            anchor_obs = self._get_anchor_obs(data)
            self._anchor_action = self._infer_anchor(anchor_obs)

            # Residual policy sees anchor action in observation.
            residual_obs = self._get_residual_obs(data)
            residual_action = self._infer_residual(residual_obs)

            # Same mixing as JoystickGo2.step:
            # mixed_action = anchor_action + residual_action
            mixed_action = self._anchor_action + residual_action
            ctrl = self._default_angles + mixed_action * self._action_scale
            data.ctrl[:] = ctrl

            # Keep state vars aligned with JoystickGo2 info updates.
            self._last_action = ctrl.astype(np.float32)
            self._step += 1

        self._counter += 1


def load_callback(model=None, data=None):
    del model, data
    mujoco.set_mjcb_control(None)

    # Use sensor-enabled XML required by JoystickGo2 observations.
    xml_path = consts.MJX_XML_SENSOR_PATH.as_posix()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Match JoystickGo2 default config.
    ctrl_dt = 0.02
    sim_dt = 0.002
    model.opt.timestep = sim_dt
    n_substeps = int(ctrl_dt / sim_dt)

    # Match JoystickGo2 default gains.
    kp = 35.0
    kd = 0.5
    model.dof_damping[6:] = kd
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 1] = -kp

    controller = Go2Joystick2OnnxController(
        anchor_policy_path=(_ONNX_DIR / "go2_apg2_anchor_policy.onnx").as_posix(),
        residual_policy_path=(_ONNX_DIR / "go2_apg2_residual_policy.onnx").as_posix(),
        default_qpos=np.array(model.keyframe("home").qpos, dtype=np.float32),
        ctrl_dt=ctrl_dt,
        n_substeps=n_substeps,
        action_scale=np.array([0.5, 0.5, 0.5] * 4, dtype=np.float32),
        command=np.array([1.0, -0.5, -0.5], dtype=np.float32),
    )

    mujoco.set_mjcb_control(controller.get_control)
    renderer = mujoco.Renderer(model, height=480, width=640)
    return model, data, renderer


if __name__ == "__main__":
    model, data, renderer = load_callback()

    frames = []
    run_time = 10.0
    print("Running simulation...")

    while data.time < run_time:
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera="track")
        frames.append(renderer.render())

        if len(frames) % 100 == 0:
            print(f"time={data.time:.2f}s, frames={len(frames)}")

    print("Saving video...")
    media.write_video("go2_joystick2_anchor_residual_onnx.mp4", frames, fps=int(1 / model.opt.timestep))
    print("Saved to go2_joystick2_anchor_residual_onnx.mp4")

    renderer.close()
