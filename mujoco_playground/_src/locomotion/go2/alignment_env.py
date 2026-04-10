"""AlignmentEnv for real-world fine-tuning with APG.

This module provides an environment wrapper that combines:
- MJX (differentiable) for backward pass
- Native MuJoCo or Real Robot (non-differentiable) for forward pass

The gradient flows through MJX while the value comes from the native/real transition.
"""

from typing import Any, Callable, Dict, Optional, Sequence
import functools

import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from ml_collections import config_dict

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import managers as manager_lib
from mujoco_playground._src.locomotion.go2.managers import state_alignment_manager as state_align_lib


def _to_numpy_tree(pytree: Any) -> Any:
    """Convert JAX arrays in pytree to numpy arrays."""
    return jax.tree_util.tree_map(lambda x: np.asarray(x), pytree)


def _native_reset_numpy(native_env: Any) -> Any:
    """Pure numpy reset for native environment.

    This function calls native_env.reset() and converts the result to numpy.
    It must be called inside pure_callback to avoid tracer issues.

    Args:
        native_env: Native environment instance (Go2JoystickMujoco)

    Returns:
        State with numpy arrays matching Go2JoystickMujoco state structure
    """
    # Call the native reset with a fixed key
    native_key = np.array([0, 0], dtype=np.uint32)
    state = native_env.reset(native_key)
    return _to_numpy_tree(state)


class NativeEnvWrapper(mjx_env.MjxEnv):
    """Wrapper for native MuJoCo environments to make them JIT-compatible.

    This wrapper uses pure_callback to call the native environment's reset
    and step methods, avoiding tracer issues when used with wrap_for_training.

    Note: This wrapper has overhead from pure_callback and numpy<->JAX conversions.
    For faster training without physics mismatch, use Go2Joystick2 directly.

    Usage:
        native_env = locomotion.load("Go2JoystickMujoco", config=cfg)
        wrapped_env = NativeEnvWrapper(native_env)
        # Now wrapped_env can be used with wrap_for_training
    """

    def __init__(self, native_env: Any):
        """Initialize the wrapper.

        Args:
            native_env: Native environment instance (e.g., Go2JoystickMujoco)
        """
        self._native_env = native_env
        self._observation_size = native_env.observation_size
        self._action_size = native_env.action_size

        # Create result spec for pure_callback (computed once at init)
        # Use a fixed seed for consistency
        dummy_key = np.array([0, 0], dtype=np.uint32)
        dummy_state = native_env.reset(dummy_key)
        self._result_spec = _to_numpy_tree(dummy_state)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment via pure_callback.

        Args:
            rng: Random key (not used, fixed seed for determinism)

        Returns:
            State with numpy arrays converted back to JAX arrays
        """
        # Use fixed seed for native reset (deterministic, no RNG overhead)
        # This is fine for evaluation where we want consistent behavior
        def _do_reset():
            native_key = np.array([0, 0], dtype=np.uint32)
            state = self._native_env.reset(native_key)
            return _to_numpy_tree(state)

        native_state = jax.pure_callback(
            _do_reset,
            self._result_spec,
            vmap_method='sequential',
        )
        # Convert numpy arrays back to JAX arrays
        return jax.tree_util.tree_map(jp.asarray, native_state)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step the environment via pure_callback.

        Args:
            state: Current state
            action: Action array

        Returns:
            Next state
        """
        # Note: We pass state and action directly to pure_callback.
        # Inside the callback, they become numpy arrays, which is what
        # the native environment expects.
        def _do_step(np_state, np_action):
            # The inputs are already numpy arrays from pure_callback
            next_state = self._native_env.step(np_state, np_action)
            return _to_numpy_tree(next_state)

        native_next = jax.pure_callback(
            _do_step,
            self._result_spec,
            _to_numpy_tree(state),  # Convert state to numpy before callback
            np.asarray(action),     # Convert action to numpy before callback
            vmap_method='sequential',
        )
        # Convert numpy arrays back to JAX arrays
        return jax.tree_util.tree_map(jp.asarray, native_next)

    @property
    def observation_size(self) -> Any:
        """Return observation size."""
        return self._observation_size

    @property
    def action_size(self) -> int:
        """Return action size."""
        return self._action_size

    @property
    def xml_path(self) -> str:
        """Return XML path."""
        return self._native_env.xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        """Return MJ model."""
        return self._native_env.mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        """Return MJX model (same as mj_model for native env)."""
        return self._native_env.mjx_model if hasattr(self._native_env, 'mjx_model') else self._native_env.mj_model

    def render(self, trajectory, height=240, width=320, camera=None, scene_option=None, modify_scene_fns=None):
        """Render using native env's renderer."""
        return self._native_env.render(trajectory, height, width, camera, scene_option, modify_scene_fns)


def default_config() -> config_dict.ConfigDict:
    """Default configuration for AlignmentEnv.

    This merges the MJX env config with alignment-specific parameters.
    """
    from mujoco_playground._src.locomotion.go2.configs import joystick_config
    cfg = joystick_config.default_config()

    # Add alignment-specific parameters
    cfg.align = config_dict.ConfigDict(
        dict(
            alpha=1.0,  # Alignment strength (1.0 = full alignment)
            eta=1.0,    # Action scaling factor
            horizon=1,  # Rollout horizon per step (1 for single-step fine-tuning)
        )
    )
    return cfg


def _default_env_factory(
    config: config_dict.ConfigDict = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> "AlignmentEnv":
    """Default factory for creating AlignmentEnv with standard MJX + MuJoCo pairing.

    Args:
        config: Configuration dictionary for AlignmentEnv
        config_overrides: Optional config overrides. Can contain:
            - mjx_cfg: Config for MJX environment (Go2Joystick2)
            - native_cfg: Config for native environment (Go2JoystickMujoco)

    Returns:
        AlignmentEnv instance
    """
    from mujoco_playground import locomotion

    # Use default configs if not provided via config_overrides
    if config_overrides is None:
        config_overrides = {}

    mjx_cfg = config_overrides.get("mjx_cfg", None)
    native_cfg = config_overrides.get("native_cfg", None)

    # Load MJX env (Go2Joystick2 for training)
    mjx_env_inst = locomotion.load("Go2Joystick2", config=mjx_cfg)

    # Load native env (Go2JoystickMujoco for forward pass)
    native_env_inst = locomotion.load("Go2JoystickMujoco", config=native_cfg)

    # Get alignment parameters from config
    if config is None:
        config = default_config()

    return AlignmentEnv(
        mjx_env=mjx_env_inst,
        native_env=native_env_inst,
        alpha=config.get("align", {}).get("alpha", 1.0),
        eta=config.get("align", {}).get("eta", 1.0),
        horizon=config.get("align", {}).get("horizon", 1),
    )


class AlignmentEnv(mjx_env.MjxEnv):
    """Environment wrapper for sim-to-real fine-tuning with state alignment.

    This environment:
    1. Uses MJX for differentiable simulation (gradient backprop)
    2. Uses a native oracle (MuJoCo or real robot) for forward transition
    3. Aligns the states using StateAlignmentManager

    The key insight is that:
    - Forward pass: Uses PURE native MuJoCo state (with mj_twin_template physics mismatch)
    - Backward pass: Gradients flow through MJX, scaled by alpha parameter

    This allows learning corrections that work with real-world physics while
    benefiting from differentiable simulation for gradient computation.

    State format:
    - State.data: MJX data aligned with native values (qpos, qvel, etc.)
    - State.info['native_state']: Native MuJoCo state (stored for step)
    - State.info['rng']: RNG key for native reset

    Example usage:
        # Create MJX and native environments
        mjx_env_inst = locomotion.load("Go2Joystick2", config=mjx_cfg)
        native_env_inst = locomotion.load("Go2JoystickMujoco", config=native_cfg)

        # Create alignment environment
        align_env = AlignmentEnv(
            mjx_env=mjx_env_inst,
            native_env=native_env_inst,
            alpha=1.0,
            eta=1.0,
            horizon=1,
        )

        # Use with APG
        from apg_alg.algorithm import apg
        make_inference_fn, params, _ = apg.train(
            environment=align_env,
            eval_env=align_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            randomization_fn=None,
            num_envs=1,
            **apg_params
        )
    """

    def __init__(
        self,
        mjx_env: Any,
        native_env: Any,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        alpha: float = 1.0,
        eta: float = 1.0,
        horizon: int = 1,
    ):
        """Initialize AlignmentEnv.

        Args:
            mjx_env: MJX environment instance (differentiable)
            native_env: Native environment instance (non-differentiable, e.g., MuJoCo or real robot)
            config: Configuration dictionary
            config_overrides: Optional config overrides
            alpha: Alignment strength. 1.0 = full alignment, 0.0 = MJX only
            eta: Action scaling factor
            horizon: Number of steps per rollout (1 for single-step fine-tuning)
        """
        # Use mjx_env's config as base
        if config is None:
            config = default_config()

        super().__init__(
            config=config,
            config_overrides=config_overrides,
        )

        self._mjx_env = mjx_env
        self._native_env = native_env
        self._alpha = jp.asarray(alpha, dtype=jp.float64)
        self._eta = jp.asarray(eta, dtype=jp.float64)
        self._horizon = horizon

        # Create state alignment manager
        self._alignment_mgr = manager_lib.StateAlignmentManager(
            mjx_env_inst=mjx_env,
            native_env_inst=native_env,
        )

        # Get observation and action sizes from MJX env
        self._observation_size = mjx_env.observation_size
        self._action_size = mjx_env.action_size

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset both environments and initialize the state.

        Args:
            rng: Random key

        Returns:
            Initial MJX state with native_state stored in info
        """
        # Split RNG for MJX environment only
        rng, mjx_key = jax.random.split(rng)

        # Reset MJX environment (JIT-compatible)
        mjx_state = self._mjx_env.reset(mjx_key)

        # Reset native environment via pure_callback (non-JIT, pure numpy)
        # Note: native reset doesn't affect gradients, so fixed seed is fine
        # Use a closure to capture native_env
        native_env = self._native_env
        def _do_reset():
            return _native_reset_numpy(native_env)

        native_state = jax.pure_callback(
            _do_reset,
            self._alignment_mgr.oracle_result_spec,
            vmap_method='sequential',
        )

        # Store native_state and rng in info for step()
        info = dict(mjx_state.info)
        info['native_state'] = native_state
        info['rng'] = rng

        return mjx_state.replace(info=info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step the alignment environment.

        This executes one step of aligned simulation:
        1. Clip and scale the action
        2. Step MJX environment (differentiable)
        3. Step native environment via pure_callback (non-differentiable)
        4. Align the states using StateAlignmentManager
        5. Return the aligned MJX state

        Args:
            state: Current MJX state (with native_state stored in info)
            action: Action from policy (unnormalized, in [-1, 1])

        Returns:
            Aligned MJX state with:
            - data from aligned state (native forward + MJX gradient)
            - obs from aligned state
            - reward from MJX env
            - done from MJX env
        """
        # Extract native_state and rng from info
        info = dict(state.info)
        native_state = info['native_state']
        rng = info['rng']

        # Check if episode is still active
        is_active = 1.0 - jp.asarray(state.done, dtype=jp.float64)

        # Scale and clip action
        action = jp.clip(self._eta * jp.asarray(action, dtype=jp.float64), -1.0, 1.0)
        action = action * is_active

        # Step MJX environment (differentiable)
        diff_next = self._mjx_env.step(state, action)

        # Step native environment via pure_callback (non-differentiable)
        native_next = state_align_lib.oracle_step_callback(
            self._alignment_mgr.oracle.step,
            self._alignment_mgr.oracle_result_spec,
            native_state,
            action,
        )

        # Align states: native forward + MJX gradient
        aligned_next = self._alignment_mgr.build_aligned_state(
            diff_next, native_next, self._alpha
        )

        # Freeze state after termination (no implicit reset)
        mjx_next = jax.tree_util.tree_map(
            lambda new, old: jp.where(is_active, new, old),
            aligned_next,
            state,
        )
        native_next = jax.tree_util.tree_map(
            lambda new, old: jp.where(is_active, new, old),
            native_next,
            native_state,
        )

        # Update info with new native_state and rng
        info_next = dict(mjx_next.info)
        info_next['native_state'] = native_next
        info_next['rng'] = rng

        return mjx_next.replace(info=info_next)

    @property
    def observation_size(self) -> Any:
        """Return observation size from MJX env."""
        return self._observation_size

    @property
    def action_size(self) -> int:
        """Return action size from MJX env."""
        return self._action_size

    @property
    def xml_path(self) -> str:
        """Return XML path from MJX env."""
        return self._mjx_env.xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        """Return MJX model (for rendering)."""
        return self._mjx_env.mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        """Return MJX model (for alignment)."""
        return self._mjx_env.mjx_model

    def render(
        self,
        trajectory: Sequence[mjx_env.State],
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[Sequence[Callable[[mujoco.MjvScene], None]]] = None,
    ) -> Sequence[np.ndarray]:
        """Render trajectory using MJX env's renderer."""
        return self._mjx_env.render(
            trajectory, height, width, camera, scene_option, modify_scene_fns
        )

    def get_native_state(self) -> Any:
        """Get current native environment state (for debugging/logging).

        Note: This method is for non-JIT debugging only. For use inside JIT,
        extract native_state from state.info['native_state'] directly.
        """
        # This method is only useful for non-JIT debugging.
        # In JIT context, native_state should be extracted from state.info
        return None

    def set_alpha(self, alpha: float) -> None:
        """Update alignment strength dynamically.

        Args:
            alpha: New alignment strength (0.0 = MJX only, 1.0 = full alignment)
        """
        self._alpha = jp.asarray(alpha, dtype=jp.float64)

    def set_eta(self, eta: float) -> None:
        """Update action scaling factor dynamically.

        Args:
            eta: New action scaling factor
        """
        self._eta = jp.asarray(eta, dtype=jp.float64)


def create_alignment_env(
    mjx_env_name: str = "Go2Joystick2",
    native_env_name: str = "Go2JoystickMujoco",
    mjx_cfg: Optional[config_dict.ConfigDict] = None,
    native_cfg: Optional[config_dict.ConfigDict] = None,
    alpha: float = 1.0,
    eta: float = 1.0,
    horizon: int = 1,
) -> AlignmentEnv:
    """Factory function to create AlignmentEnv with default settings.

    Args:
        mjx_env_name: Name of the MJX environment
        native_env_name: Name of the native environment
        mjx_cfg: Optional config for MJX env (Go2Joystick2)
        native_cfg: Optional config for native env (Go2JoystickMujoco).
                   Pass mj_twin_template here to override physics parameters.
        alpha: Alignment strength
        eta: Action scaling factor
        horizon: Rollout horizon

    Returns:
        AlignmentEnv instance
    """
    from mujoco_playground import locomotion

    # Load environments
    mjx_env_inst = locomotion.load(mjx_env_name, config=mjx_cfg)
    native_env_inst = locomotion.load(native_env_name, config=native_cfg)

    # Create alignment environment
    return AlignmentEnv(
        mjx_env=mjx_env_inst,
        native_env=native_env_inst,
        alpha=alpha,
        eta=eta,
        horizon=horizon,
    )
