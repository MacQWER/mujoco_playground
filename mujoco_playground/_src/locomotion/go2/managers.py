from typing import Any, Callable, Dict, Optional, Sequence

import jax

from ml_collections import config_dict

from mujoco_playground._src.locomotion.go2.mdp import commands as command_lib
from mujoco_playground._src.locomotion.go2.mdp import event as event_lib
from mujoco_playground._src.locomotion.go2 import assistive_wrench_manager as assistive_wrench_lib


def _to_config_dict(value: Any) -> config_dict.ConfigDict:
    if isinstance(value, config_dict.ConfigDict):
        return value
    return config_dict.ConfigDict(value)


def normalize_obs_term(term: Any) -> config_dict.ConfigDict:
    if isinstance(term, tuple):
        func, noise, scale = term
        return config_dict.ConfigDict(
            dict(
                func=func,
                noise=noise,
                scale=scale,
                enabled=True,
            )
        )

    cfg = _to_config_dict(term)
    if "func" not in cfg:
        raise ValueError(f"Observation term is missing `func`: {term}")
    if "enabled" not in cfg:
        cfg.enabled = True
    if "noise" not in cfg:
        cfg.noise = None
    if "scale" not in cfg:
        cfg.scale = 1.0
    return cfg


def normalize_reward_term(name: str, term: Any) -> config_dict.ConfigDict:
    if isinstance(term, (int, float)):
        return config_dict.ConfigDict(
            dict(
                func=name,
                scale=float(term),
                enabled=True,
            )
        )

    cfg = _to_config_dict(term)
    if "func" not in cfg:
        cfg.func = name
    if "enabled" not in cfg:
        cfg.enabled = True
    if "scale" not in cfg:
        raise ValueError(f"Reward term `{name}` is missing `scale`.")
    return cfg


def sync_manager_state(info: Dict[str, Any]) -> Dict[str, Any]:
    manager_state = info.setdefault("manager_state", {})

    manager_state["command"] = {
        "value": info.get("command"),
        "steps_until_next": info.get("steps_until_next_cmd"),
    }
    manager_state["actions"] = {
        "last_action": info.get("last_action"),
        "last_residual": info.get("last_residual"),
        "anchor_action": info.get("anchor_action"),
    }
    manager_state["rewards"] = {
        "current": info.get("reward_tuple"),
    }

    if "assist_beta" in info:
        manager_state["assistive_wrench"] = {
            "beta": info.get("assist_beta"),
            "force_world": info.get("assist_force_world"),
            "torque_world": info.get("assist_torque_world"),
            "force_norm": info.get("assist_force_norm"),
            "torque_norm": info.get("assist_torque_norm"),
        }

    events = manager_state.setdefault("events", {})
    for prefix in ("pert", "disturbance"):
        if (
            f"steps_until_next_{prefix}" in info
            or f"{prefix}_duration" in info
            or f"{prefix}_mag" in info
        ):
            events[prefix] = {
                "steps_until_next": info.get(f"steps_until_next_{prefix}"),
                "duration_seconds": info.get(f"{prefix}_duration_seconds"),
                "duration_steps": info.get(f"{prefix}_duration"),
                "steps_since_last": info.get(f"steps_since_last_{prefix}"),
                "steps": info.get(f"{prefix}_steps"),
                "direction": info.get(f"{prefix}_dir"),
                "magnitude": info.get(f"{prefix}_mag"),
            }

    return info


class ObservationManager:
    def __init__(
        self,
        obs_module: Any,
        context_fn: Callable[[], Dict[str, Any]],
        noise_fn: Callable[[Dict[str, Any], Any, Optional[str]], Any],
    ) -> None:
        self._obs_module = obs_module
        self._context_fn = context_fn
        self._noise_fn = noise_fn

    def build(
        self,
        data: Any,
        info: Dict[str, Any],
        terms: Sequence[Any],
    ) -> Dict[str, Any]:
        obs_kwargs = self._context_fn()
        obs_terms = []
        for raw_term in terms:
            term = normalize_obs_term(raw_term)
            if not term.enabled:
                continue
            term_fn = getattr(self._obs_module, term.func)
            value = term_fn(data, info, **obs_kwargs)
            value = self._noise_fn(info, value, term.noise)
            obs_terms.append(value * term.scale)
        obs = jax.numpy.clip(jax.numpy.concatenate(obs_terms), -100.0, 100.0)
        return {"state": obs}


class RewardManager:
    def __init__(
        self,
        reward_module: Any,
        cfg: config_dict.ConfigDict,
        context_fn: Callable[[Any, Any, Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        self._cfg = cfg
        self._context_fn = context_fn
        self._active_terms = []

        if "terms" in cfg.rewards:
            terms_cfg = cfg.rewards.terms
        else:
            terms_cfg = {
                name: normalize_reward_term(name, scale)
                for name, scale in cfg.rewards.scales.items()
            }

        for name, raw_term in terms_cfg.items():
            term = normalize_reward_term(name, raw_term)
            if not term.enabled or term.scale == 0.0:
                continue
            self._active_terms.append((name, term.scale, getattr(reward_module, term.func)))

    @property
    def active_terms(self) -> Sequence[tuple[str, float, Callable[..., Any]]]:
        return self._active_terms

    def compute(
        self,
        data: Any,
        action: Any,
        info: Dict[str, Any],
        extra_args: Dict[str, Any],
        done: Any,
    ) -> Dict[str, Any]:
        del done
        reward_kwargs = self._context_fn(data, action, info, extra_args)
        reward_dict = {}
        for name, scale, func in self._active_terms:
            reward_value = func(data, info, cfg=self._cfg, **reward_kwargs)
            reward_dict[name] = reward_value * scale
        return reward_dict


class ActionManager:
    def clip(self, action: jax.Array, low: float = -1.0, high: float = 1.0) -> jax.Array:
        return jax.numpy.clip(action, low, high)


class TerminationManager:
    def build_done(self, terminated: Any) -> Any:
        return terminated


class CommandManager:
    def sample(self, rng: jax.Array, *, cmd_a: jax.Array, cmd_b: jax.Array) -> jax.Array:
        return command_lib.sample_command(rng, cmd_a=cmd_a, cmd_b=cmd_b)

    def init_state(
        self,
        info: Dict[str, Any],
        *,
        command: Optional[jax.Array] = None,
        steps_until_next_cmd: Optional[Any] = None,
    ) -> Dict[str, Any]:
        info = command_lib.init_command_state(
            info,
            command=command,
            steps_until_next_cmd=steps_until_next_cmd,
        )
        return sync_manager_state(info)

    def update(
        self,
        info: Dict[str, Any],
        *,
        dt: float,
        cmd_a: jax.Array,
        cmd_b: jax.Array,
        mean_replan_time: float = 2.5,
    ) -> Dict[str, Any]:
        info = command_lib.update_command(
            info,
            dt=dt,
            cmd_a=cmd_a,
            cmd_b=cmd_b,
            mean_replan_time=mean_replan_time,
        )
        return sync_manager_state(info)


class EventManager:
    def init_state(
        self,
        info: Dict[str, Any],
        *,
        disturbance_cfg: Any,
        dt: float,
        prefix: str,
        rng_key: str = "rng",
    ) -> Dict[str, Any]:
        info = event_lib.init_disturbance(
            info,
            disturbance_cfg=disturbance_cfg,
            dt=dt,
            prefix=prefix,
            rng_key=rng_key,
        )
        return sync_manager_state(info)

    def maybe_apply(
        self,
        state: Any,
        *,
        disturbance_cfg: Any,
        dt: float,
        base_mass: float,
        nbody: int,
        base_id: int,
        prefix: str,
        rng_key: str = "rng",
    ) -> Any:
        state = event_lib.maybe_apply_disturbance(
            state,
            disturbance_cfg=disturbance_cfg,
            dt=dt,
            base_mass=base_mass,
            nbody=nbody,
            base_id=base_id,
            prefix=prefix,
            rng_key=rng_key,
        )
        info = sync_manager_state(state.info)
        return state.replace(info=info)


class AssistiveWrenchManager:
    def __init__(
        self,
        cfg: config_dict.ConfigDict,
        *,
        base_id: int,
        base_mass: float,
        subtree_mass: float,
        base_inertia: Any,
    ) -> None:
        self._manager = assistive_wrench_lib.AssistiveWrenchManager(
            cfg,
            base_id=base_id,
            base_mass=base_mass,
            subtree_mass=subtree_mass,
            base_inertia=base_inertia,
        )

    def init_state(self, info: Dict[str, Any]) -> Dict[str, Any]:
        info = self._manager.init_state(info)
        return sync_manager_state(info)

    def maybe_apply(
        self,
        data: Any,
        info: Dict[str, Any],
        command: Any,
        *,
        get_local_linvel: Any,
        get_global_linvel: Any,
        get_gyro: Any,
        get_gravity: Any,
    ) -> tuple[Any, Dict[str, Any]]:
        data, info = self._manager.compute_and_apply_wrench(
            data,
            info,
            command,
            get_local_linvel=get_local_linvel,
            get_global_linvel=get_global_linvel,
            get_gyro=get_gyro,
            get_gravity=get_gravity,
        )
        return data, sync_manager_state(info)

    def update_curriculum(
        self,
        info: Dict[str, Any],
        command: Any,
        *,
        local_linvel: Any,
        yaw_rate: Any,
    ) -> Dict[str, Any]:
        info = self._manager.update_curriculum(
            info,
            command,
            local_linvel=local_linvel,
            yaw_rate=yaw_rate,
        )
        return sync_manager_state(info)

    def build_modify_scene_fns(self, trajectory: Sequence[Any], **kwargs: Any) -> list[Any]:
        return self._manager.build_modify_scene_fns(trajectory, **kwargs)
