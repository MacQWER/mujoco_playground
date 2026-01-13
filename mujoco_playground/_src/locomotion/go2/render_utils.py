import os
import mujoco
import jax
import jax.numpy as jnp
import numpy as np
import mediapy as media
from typing import Union, List, Optional, Any

def render_trajectory(
    mj_model: mujoco.MjModel,
    trajectory: Union[Any, List[Any], np.ndarray],
    dt: float = 0.02,
    render_every: int = 1,
    height: int = 480,
    width: int = 640,
    camera: Optional[str] = None,
    save_path: Optional[str] = None
) -> List[np.ndarray]:
    """
    A robust and fast renderer for Brax/JAX trajectories.
    Safe for EGL (Server) environments.

    Args:
        mj_model: The CPU mujoco.MjModel (NOT mjx.Model).
        trajectory: Input data. Can be:
                    1. Brax State (batched/vmap output).
                    2. List of Brax States.
                    3. Numpy/JAX array (qpos).
        dt: Simulation timestep (for FPS calculation).
        render_every: Render every Nth frame (downsampling).
        save_path: If provided (e.g., "video.mp4"), saves the video file.
    """
    print("Preparing data for rendering...")

    # --- 1. Data Extraction & Transfer (GPU -> CPU) ---
    # This step is critical for performance. We move all data to CPU at once
    # to avoid slow communication during the render loop.

    # Case A: Input is a Brax State (likely from vmap)
    if hasattr(trajectory, 'data') and hasattr(trajectory.data, 'qpos'):
        if trajectory.data.qpos.ndim > 1: 
            raw_qpos = trajectory.data.qpos
        else:
            # Handle single frame case
            raw_qpos = trajectory.data.qpos[None, :]
            
    # Case B: Input is a List of States
    elif isinstance(trajectory, list) and hasattr(trajectory[0], 'data'):
        # Stack list into an array
        raw_qpos = jnp.stack([s.data.qpos for s in trajectory])
        
    # Case C: Input is already a qpos array
    else:
        raw_qpos = trajectory

    # Apply downsampling (skip frames)
    n_steps = len(raw_qpos)
    indices = np.arange(0, n_steps, render_every)
    
    # Move data to CPU (JAX -> Numpy)
    traj_qpos = jax.device_get(raw_qpos[indices])

    # --- 2. Safety Checks ---
    # Check for NaNs to prevent rendering glitches (static noise).
    if np.isnan(traj_qpos).any():
        print("⚠️ WARNING: Trajectory contains NaNs! Replacing with zeros.")
        traj_qpos = np.nan_to_num(traj_qpos)

    # --- 3. Rendering Loop (CPU-side) ---
    print(f"Rendering {len(traj_qpos)} frames using MuJoCo...")
    
    # Create a fresh renderer to prevent EGL context conflicts
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    mj_data = mujoco.MjData(mj_model)
    
    # Get camera ID if specified
    camera_id = -1
    if camera is not None:
        camera_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)

    frames = []
    
    try:
        for i, qpos in enumerate(traj_qpos):
            # Update simulation state
            mj_data.qpos = qpos
            
            # CRITICAL: Run forward kinematics on CPU to update geometry positions
            mujoco.mj_forward(mj_model, mj_data)
            
            # Render frame
            renderer.update_scene(mj_data, camera=camera_id)
            pixels = renderer.render()
            frames.append(pixels)
            
    except Exception as e:
        print(f"Rendering failed at frame {i}: {e}")
    finally:
        # Clean up the renderer to release resources
        renderer.close()

    # --- 4. Output ---
    fps = 1.0 / (dt * render_every)
    print(f"Rendered {len(frames)} frames. FPS: {fps:.1f}")

    if save_path:
        print(f"Saving video to {save_path}...")
        media.write_video(save_path, frames, fps=fps)
    
    return frames