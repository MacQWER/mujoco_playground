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
"""Sweep runner for solimp/solref parameter sweep on PushBox.

Each invocation handles a BATCH of configs on a SINGLE GPU,
running them sequentially. JAX persistent cache ensures compilation
happens once per architecture.

Usage:
    CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=pushbox-sweep-ppo \
        python learning/sweep_runner.py --configs '[[0.0,0.0,0.0,0.0],...]'
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
from mujoco_playground.config import dm_control_suite_params
import wandb

warnings.filterwarnings("ignore")
logging.set_verbosity(logging.WARNING)

BASE_SOLIMP = [0.01, 0.5, 0.03]
HIGH_SOLIMP = [0.95, 0.99, 0.001]
BASE_SOLREF0 = 0.004
HIGH_SOLREF0 = 0.02

EVAL_SOLIMP = [0.95, 0.99, 0.001]
EVAL_SOLREF = [0.004, 1.0]


def build_params(a0, a1, a2, ar0):
  solimp = [
      (1 - a0) * BASE_SOLIMP[0] + a0 * HIGH_SOLIMP[0],
      (1 - a1) * BASE_SOLIMP[1] + a1 * HIGH_SOLIMP[1],
      (1 - a2) * BASE_SOLIMP[2] + a2 * HIGH_SOLIMP[2],
  ]
  solref = [(1 - ar0) * BASE_SOLREF0 + ar0 * HIGH_SOLREF0, 1.0]
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


def _render_video(eval_env, make_inference_fn, params, suffix):
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)
  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)

  rng = jax.random.PRNGKey(0)
  state = jit_reset(rng)
  rollout = []
  for _ in range(256):  # PushBox episode_length
    act_rng, rng = jax.random.split(rng)
    act = jit_inference_fn(state.obs, act_rng)[0]
    state = jit_step(state, act)
    rollout.append(state)

  render_every = 2
  fps = 1.0 / eval_env.dt / render_every
  traj = rollout[::render_every]
  frames = eval_env.render(traj, height=480, width=640, camera="track")

  video_dir = "logs/sweep/videos"
  os.makedirs(video_dir, exist_ok=True)
  video_path = os.path.join(video_dir, f"rollout-{suffix}.mp4")
  media.write_video(video_path, frames, fps=fps)

  total_reward = float(jp.sum(jp.array([float(s.reward) for s in rollout])))
  # Track max penetration depth: contact.dist < 0 means penetration
  contact_dists = jp.array([
      float(state.data.contact.dist[:state.data.ncon].min())
      for state in rollout
  ])
  has_contact = contact_dists > -1e10  # MuJoCo uses large negative for no contact
  max_penetration = float(-contact_dists[has_contact].min()) if has_contact.any() else 0.0
  wandb.log({
      "video": wandb.Video(video_path, fps=fps, format="mp4"),
      "total_reward": total_reward,
      "max_penetration_depth": max_penetration,
  })
  wandb.finish()
  return max_penetration


def run_ppo(a0, a1, a2, ar0, suffix):
  print(f"  > import ppo ...")
  ppo_train, ppo_networks = _import_ppo()
  solimp, solref = build_params(a0, a1, a2, ar0)

  env_cfg = registry.get_default_config("PushBox")
  env_cfg["impl"] = "jax"
  train_env_cfg = copy.deepcopy(env_cfg)
  train_env_cfg.solimp = solimp
  train_env_cfg.solref = solref
  eval_env_cfg = copy.deepcopy(env_cfg)
  eval_env_cfg.solimp = EVAL_SOLIMP
  eval_env_cfg.solref = EVAL_SOLREF

  ppo_params = dm_control_suite_params.brax_ppo_config("PushBox", "jax")

  print(f"  > load env ...")
  env = registry.load("PushBox", config=train_env_cfg)
  eval_env = registry.load("PushBox", config=eval_env_cfg)

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
  max_penetration = _render_video(eval_env, make_inference_fn, params, suffix)
  print(f"  > video done (max penetration: {max_penetration:.6f})")


def run_apg(a0, a1, a2, ar0, suffix):
  print(f"  > import apg ...")
  apg_train, apg_networks = _import_apg()
  solimp, solref = build_params(a0, a1, a2, ar0)

  env_cfg = registry.get_default_config("PushBox")
  env_cfg["impl"] = "jax"
  train_env_cfg = copy.deepcopy(env_cfg)
  train_env_cfg.solimp = solimp
  train_env_cfg.solref = solref
  eval_env_cfg = copy.deepcopy(env_cfg)
  eval_env_cfg.solimp = EVAL_SOLIMP
  eval_env_cfg.solref = EVAL_SOLREF

  apg_params = dm_control_suite_params.brax_apg_config("PushBox")

  print(f"  > load env ...")
  env = registry.load("PushBox", config=train_env_cfg)
  eval_env = registry.load("PushBox", config=eval_env_cfg)

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
  max_penetration = _render_video(eval_env, make_inference_fn, params, suffix)
  print(f"  > video done (max penetration: {max_penetration:.6f})")


def main(argv):
  del argv
  configs = json.loads(_CONFIGS.value)
  project = os.environ.get("WANDB_PROJECT", "pushbox-sweep")
  algorithm = _ALGORITHM.value

  # Random delay (0-5s) to stagger JIT compilation across processes.
  # Earlier processes compile and write to jit_cache; later ones read it.
  delay = random.uniform(0, 5)
  time.sleep(delay)

  print(f"[{algorithm}] {len(configs)} configs (staggered by {delay:.1f}s)")

  for i, (a0, a1, a2, ar0) in enumerate(configs):
    solimp, solref = build_params(a0, a1, a2, ar0)
    suffix = f"a0{a0:.1f}_a1{a1:.1f}_a2{a2:.1f}_ar0{ar0:.1f}"
    t0 = time.time()
    print(f"[{i+1}/{len(configs)}] {suffix} (started at {time.strftime('%H:%M:%S')})")

    wandb.init(project=project, name=f"PushBox-{suffix}", config={
        "a0": float(a0), "a1": float(a1), "a2": float(a2), "ar0": float(ar0),
        "train_solimp": solimp, "train_solref": solref,
        "eval_solimp": EVAL_SOLIMP, "eval_solref": EVAL_SOLREF,
    })

    if algorithm == "ppo":
      run_ppo(a0, a1, a2, ar0, suffix)
    else:
      run_apg(a0, a1, a2, ar0, suffix)

    print(f"  done ({time.time() - t0:.1f}s)")


_ALGORITHM = flags.DEFINE_string("algorithm", "ppo", "ppo or apg")
_CONFIGS = flags.DEFINE_string("configs", "[]", "JSON list of [a0,a1,a2,ar0]")

if __name__ == "__main__":
  app.run(main)
