from typing import Any, Callable

from flax import struct
import functools
import jax
import jax.numpy as jp
import jax.numpy as jnp
from jax import dtypes
from mujoco import mjx
import mujoco
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import go2_constants as consts


@struct.dataclass
class CanonicalState:
    qpos: jax.Array
    qvel: jax.Array
    ctrl: jax.Array


def tree_shape_dtype(pytree: Any) -> Any:
    def _shape_dtype(x: Any) -> jax.ShapeDtypeStruct:
        arr = np.asarray(x)
        return jax.ShapeDtypeStruct(arr.shape, arr.dtype)

    return jax.tree_util.tree_map(_shape_dtype, pytree)


def align_tree(non_diff: Any, diff: Any, alpha: jax.Array) -> Any:
    alpha = jp.asarray(alpha)
    return jax.tree_util.tree_map(
        lambda x_nd, x_df: x_nd + alpha * (x_df - jax.lax.stop_gradient(x_df)),
        non_diff,
        diff,
    )


def extract_canonical_state(data: Any) -> CanonicalState:
    return CanonicalState(qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl)


def load_residual_inference_fn(path: str):
    from brax.io import model
    from brax.training.acme import running_statistics
    from apg_alg.networks import apg_networks

    full_params = model.load_params(path)
    if isinstance(full_params, dict):
        if "normalizer_params" in full_params and "policy_params" in full_params:
            full_params = (
                full_params["normalizer_params"],
                full_params["policy_params"],
            )
        else:
            raise ValueError(
                f"Unsupported checkpoint format keys: {sorted(full_params.keys())}"
            )
    network_factory = functools.partial(
        apg_networks.make_apg_networks,
        hidden_layer_sizes=(256, 128),
        policy_obs_key="state",
    )
    network = network_factory(
        observation_size=consts.RESIDUAL_OBS_DIM,
        action_size=consts.RESIDUAL_ACT_DIM,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_inference_fn = apg_networks.make_inference_fn(network)
    return jax.jit(make_inference_fn(full_params, deterministic=True))


def _to_numpy_tree(pytree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: np.asarray(x), pytree)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def oracle_step_callback(
    callback: Callable[[Any, Any], Any],
    result_spec: Any,
    native_state: Any,
    action: Any,
) -> Any:
    return jax.pure_callback(callback, result_spec, native_state, action)


@oracle_step_callback.defjvp
def _oracle_step_callback_jvp(
    callback: Callable[[Any, Any], Any],
    result_spec: Any,
    primals: tuple[Any, Any],
    tangents: tuple[Any, Any],
) -> tuple[Any, Any]:
    native_state, action = primals
    del tangents
    primal_out = jax.pure_callback(callback, result_spec, native_state, action)

    def _zero_tangent(x: Any) -> Any:
        arr = jnp.asarray(x)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            return jnp.zeros_like(arr)
        return jnp.zeros(arr.shape, dtype=dtypes.float0)

    tangent_out = jax.tree_util.tree_map(_zero_tangent, primal_out)
    return primal_out, tangent_out


class LocalNativeOracle:
    """Synchronous native MuJoCo oracle used by pure_callback."""

    def __init__(self, native_env_inst: Any) -> None:
        self._native_env = native_env_inst

    def step(self, native_state: Any, action: Any) -> Any:
        native_state = jax.tree_util.tree_map(jp.asarray, native_state)
        action = jp.asarray(action)
        next_state = self._native_env.step(native_state, action)
        return _to_numpy_tree(next_state)


class StateAlignmentManager:
    """Builds aligned rollouts between MJX and a non-differentiable oracle."""

    def __init__(self, mjx_env_inst: Any, native_env_inst: Any) -> None:
        self._mjx_env = mjx_env_inst
        self._oracle = LocalNativeOracle(native_env_inst)

    def build_aligned_data(self, state: CanonicalState) -> Any:
        data = mjx_env.make_data(
            self._mjx_env.mj_model,
            qpos=state.qpos,
            qvel=state.qvel,
            ctrl=state.ctrl,
            impl=self._mjx_env.mjx_model.impl.value,
            nconmax=self._mjx_env._config.nconmax,
            njmax=self._mjx_env._config.njmax,
        )
        return mjx.forward(self._mjx_env.mjx_model, data)

    def _build_aligned_state(self, diff_state: Any, non_diff_state: Any, alpha: jax.Array) -> Any:
        aligned_core = align_tree(
            extract_canonical_state(non_diff_state.data),
            extract_canonical_state(diff_state.data),
            alpha,
        )
        aligned_data = self.build_aligned_data(aligned_core)
        aligned_obs = self._mjx_env._get_residual_obs(aligned_data, diff_state.info)
        return diff_state.replace(data=aligned_data, obs=aligned_obs)

    def make_step_fns(
        self,
        *,
        initial_mjx_state: Any,
        initial_native_state: Any,
        policy_fn: Callable[[Any, Any], tuple[Any, Any]],
        horizon: int,
        loss_fn: Callable[[Any, Any], jax.Array],
    ) -> tuple[Callable[[jax.Array], dict[str, jax.Array]], Callable[[jax.Array, jax.Array], dict[str, jax.Array]]]:
        native_state_spec = tree_shape_dtype(initial_native_state)

        def scan_policy_action(obs: Any, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
            action, _ = policy_fn(obs, rng)
            action = jp.asarray(action)
            next_rng = jax.random.split(rng, 2)[0]
            return action, next_rng

        def rollout_once(alpha: jax.Array | None, eta: jax.Array) -> dict[str, jax.Array]:
            def body_fn(carry: tuple[Any, Any, jax.Array], _unused: Any) -> tuple[tuple[Any, Any, jax.Array], dict[str, jax.Array]]:
                mjx_state, native_state, rng = carry
                is_active = 1.0 - jp.asarray(mjx_state.done)
                action_rng, next_rng = jax.random.split(rng)
                base_action, _ = policy_fn(mjx_state.obs, action_rng)
                action = jp.clip(eta * jp.asarray(base_action), -1.0, 1.0)
                action = action * is_active

                diff_next = self._mjx_env.step(mjx_state, action)
                native_next = oracle_step_callback(
                    self._oracle.step,
                    native_state_spec,
                    native_state,
                    action,
                )

                if alpha is None:
                    aligned_next = diff_next
                else:
                    aligned_next = self._build_aligned_state(diff_next, native_next, alpha)

                # Freeze the rollout after termination instead of implicitly
                # resetting, which better matches real-world fine-tuning.
                mjx_next = jax.tree_util.tree_map(
                    lambda new, old: jp.where(is_active, new, old),
                    aligned_next,
                    mjx_state,
                )
                raw_native_next = native_next
                native_next = jax.tree_util.tree_map(
                    lambda new, old: jp.where(is_active, new, old),
                    native_next,
                    native_state,
                )

                metrics = {
                    "loss": loss_fn(self._mjx_env, aligned_next.data),
                    "reward": aligned_next.reward,
                    "done": aligned_next.done,
                    "active": is_active,
                    "ctrl_norm": jp.linalg.norm(aligned_next.data.ctrl),
                    "qpos_gap": jp.linalg.norm(diff_next.data.qpos - raw_native_next.data.qpos),
                    "qvel_gap": jp.linalg.norm(diff_next.data.qvel - raw_native_next.data.qvel),
                }
                return (mjx_next, native_next, next_rng), metrics

            init_rng = initial_mjx_state.info["rng"]
            (_, _, _), traj = jax.lax.scan(
                body_fn,
                (initial_mjx_state, initial_native_state, init_rng),
                xs=None,
                length=horizon,
            )
            valid_mask = traj["active"]
            valid_steps = jp.maximum(jp.sum(valid_mask), 1.0)
            total_loss = jp.sum(traj["loss"] * valid_mask)
            return {
                "loss": total_loss,
                "reward_sum": jp.sum(traj["reward"] * valid_mask),
                "mean_ctrl_norm": jp.sum(traj["ctrl_norm"] * valid_mask) / valid_steps,
                "mean_qpos_gap": jp.sum(traj["qpos_gap"] * valid_mask) / valid_steps,
                "mean_qvel_gap": jp.sum(traj["qvel_gap"] * valid_mask) / valid_steps,
                "final_done": jp.max(traj["done"]),
                "valid_steps": valid_steps,
            }

        @jax.jit
        def mjx_step(eta: jax.Array) -> dict[str, jax.Array]:
            return rollout_once(None, eta)

        @jax.jit
        def align_step(eta: jax.Array, alpha: jax.Array) -> dict[str, jax.Array]:
            return rollout_once(alpha, eta)

        return mjx_step, align_step
