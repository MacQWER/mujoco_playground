from typing import Any, Callable, List, Optional, Sequence, Union

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np


def render_trajectory(
    mj_model: mujoco.MjModel,
    dt: float,
    trajectory: Union[List[Any], jax.Array, np.ndarray],
    render_every: int = 1,
    height: int = 480,
    width: int = 640,
    camera: Optional[str] = None,
    save_path: Optional[str] = None,
    scene_option: Optional[mujoco.MjvOption] = None,
    modify_scene_fns: Optional[Sequence[Callable[[mujoco.MjvScene], None]]] = None,
) -> List[np.ndarray]:
    """Renders a trajectory from qpos arrays or state objects."""
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    camera_id = -1
    if camera is not None:
        camera_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)

    data = mujoco.MjData(mj_model)
    out = []

    if isinstance(trajectory, (jp.ndarray, np.ndarray)):
        traj_qpos = jax.device_get(trajectory)
        traj_qvel, traj_mocap_pos, traj_mocap_quat = None, None, None
    elif isinstance(trajectory, list):
        def get_attr(states, attr):
            if not hasattr(states[0].data, attr):
                return None
            return jax.device_get(jp.stack([getattr(s.data, attr) for s in states]))

        traj_qpos = get_attr(trajectory, "qpos")
        traj_qvel = get_attr(trajectory, "qvel")
        traj_mocap_pos = get_attr(trajectory, "mocap_pos")
        traj_mocap_quat = get_attr(trajectory, "mocap_quat")
    else:
        raise ValueError(f"Unsupported trajectory type: {type(trajectory)}")

    n_frames = len(traj_qpos)
    indices = np.arange(0, n_frames, render_every)

    try:
        for i, idx in enumerate(indices):
            data.qpos[:] = traj_qpos[idx]
            if traj_qvel is not None:
                data.qvel[:] = traj_qvel[idx]
            if traj_mocap_pos is not None:
                data.mocap_pos[:] = traj_mocap_pos[idx]
            if traj_mocap_quat is not None:
                data.mocap_quat[:] = traj_mocap_quat[idx]

            mujoco.mj_forward(mj_model, data)
            renderer.update_scene(data, camera=camera_id, scene_option=scene_option)

            if modify_scene_fns is not None and i < len(modify_scene_fns):
                modify_scene_fns[i](renderer.scene)

            out.append(renderer.render())
    except Exception as e:
        print(f"Rendering failed at frame {i}: {e}")
    finally:
        renderer.close()

    if save_path:
        fps = 1.0 / (dt * render_every)
        print(f"Saving video to {save_path} (FPS={fps:.1f})...")
        media.write_video(save_path, out, fps=fps)

    return out
