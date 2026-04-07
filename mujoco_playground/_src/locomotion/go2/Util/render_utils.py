import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np


def render_trajectory(
    mj_model: mujoco.MjModel,
    dt: float,
    trajectory,
    render_every: int = 1,
    height: int = 480,
    width: int = 640,
    camera: str = "track",
    save_path: str = None,
    scene_option = None,
    modify_scene_fns = None,
) -> list:
    """Renders a trajectory from state objects.

    Args:
        mj_model: MuJoCo model.
        dt: Environment timestep.
        trajectory: List of state objects.
        render_every: Render every N steps.
        height: Render height.
        width: Render width.
        camera: Camera name. Default ``"track"`` for the built-in tracking cam.
        save_path: Optional path to save the video.
        scene_option: Optional scene options.
        modify_scene_fns: Per-frame scene modification callbacks.

    Returns:
        List of rendered frames as numpy arrays.
    """
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    camera_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)

    data = mujoco.MjData(mj_model)
    out = []

    def get_attr(states, attr):
        if not hasattr(states[0].data, attr):
            return None
        return jax.device_get(jp.stack([getattr(s.data, attr) for s in states]))

    traj_qpos = get_attr(trajectory, "qpos")
    traj_qvel = get_attr(trajectory, "qvel")
    traj_mocap_pos = get_attr(trajectory, "mocap_pos")
    traj_mocap_quat = get_attr(trajectory, "mocap_quat")

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
