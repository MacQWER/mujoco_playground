# Extending Go2 Environments

This document explains how to add a new Go2 environment after the manager-style refactor.

## Current Architecture

The Go2 stack is split into a few layers:

- `base.py`: shared robot base class and shared reset/step pipeline helpers.
- `managers.py`: explicit managers for observation, reward, action, command, event, and termination.
- `configs/`: environment assembly through structured config terms.
- `mdp/`: reusable observation terms, reward terms, command logic, and event logic.
- task files such as `JoystickGo2.py` and `TrotGo2.py`: task-specific logic only.

In practice, a new environment should reuse as much as possible from these layers.

## Fast Decision Guide

Use this checklist first:

1. If you only want to change observation composition, reward composition, command ranges, disturbance settings, or a few task constants, add a new config.
2. If you need a different reset flow, a different action interpretation, or a different transition update logic, add a new task class.
3. If multiple tasks should share the same new logic, add it to `mdp/`, `managers.py`, or `base.py` instead of duplicating it.

## Option 1: Add a Config-Only Environment

This is the preferred path when the environment behavior is the same but the composition is different.

Typical examples:

- same task flow, different reward weights
- same task flow, different observation terms
- same task flow, different command ranges
- same task flow, disturbance on/off or different disturbance magnitudes

### Steps

1. Create a new config file in `configs/`.
2. Start from an existing config like `joystick_config.py` or `trot_config.py`.
3. Define `cfg.obs.policy_terms` using `config_blocks.make_obs_term(...)`.
4. Define `cfg.rewards.terms` using `config_blocks.make_reward_term(...)`.
5. Keep `cfg.rewards.scales` synchronized from `cfg.rewards.terms`.
6. Reuse the existing task class if the reset/step logic is unchanged.
7. Register the new environment name in `mujoco_playground/_src/locomotion/__init__.py`.

### Example

```python
from ml_collections import config_dict
from mujoco_playground._src.locomotion.go2.configs import config_blocks
from mujoco_playground._src.locomotion.go2 import go2_constants as consts


def default_config() -> config_dict.ConfigDict:
    cfg = config_blocks.get_sim_config()

    cfg.env = config_blocks.get_env_config()
    cfg.env.anchor_action_scale = [0.3, 0.5, 0.5] * 4
    cfg.env.residual_action_scale = [0.4, 0.6, 0.6] * 4

    cfg.obs = config_dict.ConfigDict()
    cfg.obs.policy_terms = [
        config_blocks.make_obs_term("base_angular_velocity", "gyro", consts.OBS_W_LOCAL_SCALE),
        config_blocks.make_obs_term("projected_gravity", "gravity", 1.0),
        config_blocks.make_obs_term("command", None, 1.0),
        config_blocks.make_obs_term("joint_positions", "joint_pos", 1.0),
        config_blocks.make_obs_term("joint_velocities", "joint_vel", consts.OBS_JOINT_VELS_SCALE),
        config_blocks.make_obs_term("last_action", None, 1.0),
    ]

    cfg.rewards = config_blocks.get_base_rewards_config()
    cfg.rewards.terms.tracking_lin_vel = config_blocks.make_reward_term("tracking_lin_vel", 3.0)
    cfg.rewards.terms.orientation = config_blocks.make_reward_term("orientation", -8.0)
    cfg.rewards.terms.torques = config_blocks.make_reward_term("torques", -0.0001)
    for name, term in cfg.rewards.terms.items():
        cfg.rewards.scales[name] = term.scale

    cfg.disturbance = config_blocks.get_disturbance_config()
    cfg.noise_config = config_blocks.get_noise_config()
    cfg.command_config = config_blocks.get_base_command_config()
    cfg.command_config.a = [0.6, 0.2, 0.5]
    cfg.command_config.b = [1.0, 1.0, 1.0]
    return cfg
```

Then register it to an existing task class.

## Option 2: Add a New Task Class

Create a new task class when config changes are not enough.

Typical reasons:

- custom reset logic
- custom action mixing or action post-processing
- custom per-step state updates
- custom contact/phase bookkeeping
- custom termination logic

### Steps

1. Create a new task file, for example `MyNewGo2Task.py`.
2. Inherit from `Go2Env`.
3. Build robot-specific cached values in `_post_init()`.
4. Use `_init_state(...)` in `reset()` instead of directly constructing `mjx_env.State`.
5. Use `_run_event_pipeline(...)`, `_clip_action(...)`, `_sum_reward_dict(...)`, `_update_metrics(...)`, and `_finalize_step(...)` in `step()`.
6. Put reusable observation/reward/event functions into `mdp/` if other tasks might reuse them.
7. Create a config for the task and register the environment name.

### Minimal Skeleton

```python
from typing import Dict, Optional, Union, Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2.base import Go2Env
from mujoco_playground._src.locomotion.go2 import go2_constants as consts
from mujoco_playground._src.locomotion.go2.configs import my_task_config
from mujoco_playground._src.locomotion.go2.mdp import rewards as reward_lib


def default_config() -> config_dict.ConfigDict:
    return my_task_config.default_config()


class MyNewGo2Task(Go2Env):
    def __init__(
        self,
        task: str = None,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.MJX_XML_SENSOR_PATH.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self):
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos.copy())
        self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:].copy())
        self._init_active_rewards(reward_lib)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        data = mjx_env.make_data(
            self.mj_model,
            qpos=self._init_q,
            qvel=jp.zeros(self.mjx_model.nv),
            ctrl=jp.zeros(self.mjx_model.nu),
            impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        info = {
            "rng": rng,
            "last_action": jp.zeros(self.mjx_model.nu),
            "reward_tuple": {k: 0.0 for k in self._config.rewards.scales.keys()},
        }
        info = self.command_manager.init_state(info)
        info = self.event_manager.init_state(
            info,
            disturbance_cfg=self._config.disturbance,
            dt=self.dt,
            prefix="disturbance",
        )

        obs = self._get_obs(data, info)
        metrics = info["reward_tuple"].copy()
        return self._init_state(data, obs, info, metrics=metrics)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = self._run_event_pipeline(
            state,
            enabled=self._config.disturbance.enable,
            disturbance_cfg=self._config.disturbance,
            base_mass=self.mj_model.body("base").mass,
            base_id=self.mj_model.body("base").id,
            prefix="disturbance",
        )

        action = self._clip_action(action)
        ctrl = self._default_pose + action
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

        done = self.termination_manager.build_done(jp.array(0.0))
        reward_dict = self._get_reward(data, ctrl, state.info, {}, done)
        reward = self._sum_reward_dict(reward_dict)
        self._update_metrics(state.metrics, reward_dict)
        state.info["reward_tuple"] = reward_dict
        state.info["last_action"] = ctrl

        obs = self._get_obs(data, state.info)
        return self._finalize_step(
            state,
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            info=state.info,
        )

    def _get_obs(self, data, info):
        return self._build_obs(data, info, self._config.obs.policy_terms)
```

## When To Add Code To `mdp/`

Add code to `mdp/` when the logic is a reusable term, not a one-off task flow.

Good candidates:

- a new observation term in `mdp/observations.py`
- a new reward term in `mdp/rewards.py`
- a new command update mode in `mdp/commands.py`
- a new disturbance/event utility in `mdp/event.py`

Do not put task-specific step sequencing into `mdp/`.
That sequencing should stay in the task class, using the shared base pipeline helpers.

## When To Add Code To `base.py`

Add code to `base.py` only if the behavior should be shared across multiple Go2 tasks.

Good candidates:

- shared state initialization helpers
- shared step finalization helpers
- shared action clipping / reward accumulation / metric updates
- shared pipeline scheduling helpers

Avoid putting task-specific policy logic into `base.py`.

## When To Add Code To `managers.py`

Add logic to `managers.py` when it represents a manager responsibility, not just a helper function.

Good candidates:

- observation term normalization
- reward term normalization and activation
- action transforms shared across tasks
- termination post-processing shared across tasks
- future curriculum manager or reset manager

If the logic is only used by one task and is not really a manager concern, keep it in the task file instead.

## Registration

After writing the new task/config, add it to `mujoco_playground/_src/locomotion/__init__.py`.

Typical changes:

1. import the new task module
2. add the environment class to `_envs`
3. add its config builder to `_cfgs`
4. add a randomizer to `_randomizer` if needed

## Recommended Workflow

For most new environments, follow this order:

1. Copy the closest existing config.
2. Try to solve the problem with config-only composition first.
3. Only create a new task class if the reset/step flow truly differs.
4. If you create reusable math or term logic, move it into `mdp/`.
5. If you notice repeated scheduling code across tasks, promote it into `base.py` helpers.

## Common Mistakes

- Duplicating a full `step()` method when only reward composition changed.
- Adding one-off task bookkeeping to `base.py`.
- Putting reusable reward or observation logic directly in a task file instead of `mdp/`.
- Forgetting to keep `cfg.rewards.scales` in sync with `cfg.rewards.terms`.
- Registering the config but forgetting to register the environment class.

## Rule of Thumb

Use this layering rule:

- `configs/`: what is enabled
- `mdp/`: how a reusable term is computed
- `managers.py`: how groups of terms are normalized and orchestrated
- task file: what is unique about this task’s reset/step behavior
- `base.py`: what all Go2 tasks should share
