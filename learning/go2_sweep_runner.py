# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sweep runner for Go2Joystick2 solimp+solref parameter sweep.

Each invocation handles a BATCH of configs on a SINGLE GPU,
running them sequentially. JAX persistent cache ensures compilation
happens once per architecture.

Usage:
    CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=go2-joystick2-sweep-ppo \
        python learning/go2_sweep_runner.py --configs '[[0.015,0.95,0.001,0.02],...]'
"""

import copy
import functools
import json
import os
import random
import time
import warnings

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import jax
from jax import config as jax_config
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")

_cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "jit_cache")
os.makedirs(_cache_path, exist_ok=True)
jax_config.update("jax_compilation_cache_dir", _cache_path)
jax_config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax_config.update("jax_persistent_cache_min_compile_time_secs", 1)

import jax.numpy as jp
import mediapy as media
from absl import app
from absl import flags
from absl import logging

# Will be set based on algorithm
_ppo_train = None
_ppo_networks = None
_apg_train = None
_apg_networks = None

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
import wandb

warnings.filterwarnings("ignore")
logging.set_verbosity(logging.WARNING)

ENV_NAME = "Go2Joystick2"
EVAL_SOLIMP = [0.9, 0.95, 0.001]
EVAL_SOLREF = [0.004, 1.0]
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "logs",
    "go2_sweep",
    "go2joystick2_ppo_x64_65eval",
)


def build_params(s0, s1, s2, sr0):
  solimp = [float(s0), float(s1), float(s2)]
  if solimp[0] > solimp[1]:
    raise ValueError(f"Invalid solimp, expected solimp[0] <= solimp[1]: {solimp}")
  solref = [float(sr0), 1.0]
  return solimp, solref



def _import_ppo():
  global _ppo_train, _ppo_networks
  if _ppo_train is None:
    from brax.training.agents.ppo import networks as _nn
    from brax.training.agents.ppo import train as _tr
    _ppo_train = _tr
    _ppo_networks = _nn
  return _ppo_train, _ppo_networks


def _import_apg():
  global _apg_train, _apg_networks
  if _apg_train is None:
    from apg_alg.algorithm import apg as _apg
    from apg_alg.networks import apg_networks as _apn
    _apg_train = _apg
    _apg_networks = _apn
  return _apg_train, _apg_networks


def _render_video(eval_env, make_inference_fn, params, suffix, algorithm):
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)
  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)

  rng = jax.random.PRNGKey(0)
  state = jit_reset(rng)
  rollout = []
  for _ in range(eval_env._config.episode_length):  # pylint: disable=protected-access
    act_rng, rng = jax.random.split(rng)
    act = jit_inference_fn(state.obs, act_rng)[0]
    state = jit_step(state, act)
    rollout.append(state)

  render_every = 2
  fps = 1.0 / eval_env.dt / render_every
  traj = rollout[::render_every]
  frames = eval_env.render(traj, height=480, width=640, camera="track")

  video_dir = os.path.join(_OUTPUT_DIR.value, "videos", algorithm)
  os.makedirs(video_dir, exist_ok=True)
  video_path = os.path.join(video_dir, f"rollout-{algorithm}-{suffix}.mp4")
  media.write_video(video_path, frames, fps=fps)

  total_reward = float(jp.sum(jp.array([float(s.reward) for s in rollout])))
  # Track max penetration depth: contact.dist < 0 means penetration.
  contact_dists = []
  for state in rollout:
    dist = state.data.contact.dist
    valid = jp.arange(dist.shape[0]) < state.data.ncon
    contact_dists.append(float(jp.where(valid, dist, 1e6).min()))
  contact_dists = jp.array(contact_dists)
  has_contact = contact_dists < 1e6
  if bool(has_contact.any()):
    max_penetration = max(0.0, float(-contact_dists[has_contact].min()))
  else:
    max_penetration = 0.0
  wandb.log({
      "video": wandb.Video(video_path, fps=fps, format="mp4"),
      "total_reward": total_reward,
      "max_penetration_depth": max_penetration,
  })
  wandb.finish()
  return max_penetration


def run_ppo(s0, s1, s2, sr0, suffix):
  print(f"  > import ppo ...")
  ppo_train, ppo_networks = _import_ppo()
  solimp, solref = build_params(s0, s1, s2, sr0)

  env_cfg = registry.get_default_config(ENV_NAME)
  env_cfg["impl"] = "jax"
  train_env_cfg = copy.deepcopy(env_cfg)
  train_env_cfg.env.solimp = solimp
  train_env_cfg.env.solref = solref
  eval_env_cfg = copy.deepcopy(env_cfg)
  eval_env_cfg.env.solimp = EVAL_SOLIMP
  eval_env_cfg.env.solref = EVAL_SOLREF

  ppo_params = locomotion_params.brax_ppo_config(ENV_NAME, "jax")
  wandb.config.update({
      "ppo_params": ppo_params.to_dict(),
      "train_env_cfg": train_env_cfg.to_dict(),
      "eval_env_cfg": eval_env_cfg.to_dict(),
      "jax_enable_x64": True,
      "jax_default_matmul_precision": "high",
  })

  print(f"  > load env ...")
  env = registry.load(ENV_NAME, config=train_env_cfg)
  eval_env = registry.load(ENV_NAME, config=eval_env_cfg)

  nf_kwargs = dict(ppo_params.network_factory)
  network_factory = functools.partial(ppo_networks.make_ppo_networks, **nf_kwargs)

  training_params = dict(ppo_params)
  training_params.pop("network_factory", None)
  training_params.pop("num_eval_envs", None)

  def progress(num_steps, metrics):
    if num_steps == 0:
      print("JIT_READY", flush=True)
    wandb.log(metrics, step=num_steps)

  def noop_policy_params_fn(step, make_policy, params):
    pass

  num_eval_envs = ppo_params.get("num_eval_envs", 128)

  train_fn = functools.partial(
      ppo_train.train,
      **training_params,
      network_factory=network_factory,
      seed=1,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
  )

  print(f"  > compiling + training (may take several mins on first run) ...")
  make_inference_fn, params, _ = train_fn(
      environment=env, progress_fn=progress,
      policy_params_fn=noop_policy_params_fn, eval_env=eval_env,
  )
  print(f"  > training done, rendering video ...")
  max_penetration = _render_video(
      eval_env, make_inference_fn, params, suffix, "ppo"
  )
  print(f"  > video done (max penetration: {max_penetration:.6f})")


def run_apg(s0, s1, s2, sr0, suffix):
  print(f"  > import apg ...")
  apg_train, apg_networks = _import_apg()
  solimp, solref = build_params(s0, s1, s2, sr0)

  env_cfg = registry.get_default_config(ENV_NAME)
  env_cfg["impl"] = "jax"
  train_env_cfg = copy.deepcopy(env_cfg)
  train_env_cfg.env.solimp = solimp
  train_env_cfg.env.solref = solref
  train_env_cfg.env.iterations = TRAIN_ITERATIONS
  eval_env_cfg = copy.deepcopy(env_cfg)
  eval_env_cfg.env.solimp = EVAL_SOLIMP
  eval_env_cfg.env.solref = EVAL_SOLREF
  eval_env_cfg.env.iterations = EVAL_ITERATIONS

  apg_params = locomotion_params.brax_apg_config(ENV_NAME)

  print(f"  > load env ...")
  env = registry.load(ENV_NAME, config=train_env_cfg)
  eval_env = registry.load(ENV_NAME, config=eval_env_cfg)

  training_params = dict(apg_params)
  training_params.pop("network_factory", None)
  training_params["randomization_fn"] = None

  def progress(num_steps, metrics):
    if num_steps == 0:
      print("JIT_READY", flush=True)
    wandb.log(metrics, step=num_steps)

  def noop_policy_params_fn(step, make_policy, params):
    pass

  train_fn = functools.partial(
      apg_train.train,
      **training_params,
      network_factory=functools.partial(
          apg_networks.make_apg_networks,
          **apg_params.network_factory,
      ),
      seed=1,
      progress_fn=progress,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      policy_params_fn=noop_policy_params_fn,
      eval_env=eval_env,
  )

  print(f"  > compiling + training (may take several mins on first run) ...")
  make_inference_fn, params, _ = train_fn(environment=env)
  print(f"  > training done, rendering video ...")
  max_penetration = _render_video(
      eval_env, make_inference_fn, params, suffix, "apg"
  )
  print(f"  > video done (max penetration: {max_penetration:.6f})")


def main(argv):
  del argv
  configs = json.loads(_CONFIGS.value)
  project = os.environ.get("WANDB_PROJECT", "go2-joystick2-sweep-ppo")
  algorithm = _ALGORITHM.value
  if algorithm != "ppo":
    raise ValueError(f"{ENV_NAME} sweep runner is PPO-only; use --algorithm=ppo.")

  # Random delay (0-5s) to stagger JIT compilation across processes.
  # Earlier processes compile and write to jit_cache; later ones read it.
  delay = random.uniform(0, 5)
  time.sleep(delay)

  print(f"[{algorithm}] {len(configs)} configs (staggered by {delay:.1f}s)")

  for i, (s0, s1, s2, sr0) in enumerate(configs):
    solimp, solref = build_params(s0, s1, s2, sr0)
    suffix = f"s0{s0:.3f}_s1{s1:.3f}_s2{s2:.3f}_sr{sr0:.3f}"
    t0 = time.time()
    print(f"[{i+1}/{len(configs)}] {suffix} (started at {time.strftime('%H:%M:%S')})")

    wandb.init(project=project, name=f"{ENV_NAME}-{suffix}", config={
        "env_name": ENV_NAME,
        "solimp0": float(s0), "solimp1": float(s1), "solimp2": float(s2),
        "solref0": float(sr0),
        "train_solimp": solimp, "train_solref": solref,
        "eval_solimp": EVAL_SOLIMP, "eval_solref": EVAL_SOLREF,
        "train_iterations": int(
            registry.get_default_config(ENV_NAME).env.iterations
        ),
        "eval_iterations": int(
            registry.get_default_config(ENV_NAME).env.iterations
        ),
    })

    if algorithm == "ppo":
      run_ppo(s0, s1, s2, sr0, suffix)
    else:
      run_apg(s0, s1, s2, sr0, suffix)

    print(f"  done ({time.time() - t0:.1f}s)")


_ALGORITHM = flags.DEFINE_string("algorithm", "ppo", "PPO only")
_CONFIGS = flags.DEFINE_string("configs", "[]", "JSON list of [solimp0,solimp1,solimp2,solref0]")
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    DEFAULT_OUTPUT_DIR,
    "Experiment-specific directory for rollout videos.",
)

if __name__ == "__main__":
  app.run(main)
