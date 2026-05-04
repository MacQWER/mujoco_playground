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
"""Launch solimp/solref sweep across 4 GPUs.

Grid: 5×5×5×4 = 500 combos per algorithm, 1000 total.
PPO project: pushbox-sweep-ppo
APG project: pushbox-sweep-apg

Each GPU runs 72 concurrent subprocesses, each processing 4-5 configs.
Total: 288 concurrent processes for PPO, 60 for APG.

PPO: ~1GB per subprocess (num_envs=256), 80GB/1GB ≈ 72.
APG: ~5GB per subprocess (num_envs=32), 80GB/5GB ≈ 15.

Usage:
    cd /data/mujoco_playground
    python learning/launch_sweep.py [--dry_run]
"""

import datetime
import fcntl
import itertools
import json
import math
import os
import select
import subprocess
import sys
import time

from absl import app
from absl import flags

PPO_PROJECT = "pushbox-sweep-ppo"
APG_PROJECT = "pushbox-sweep-apg"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(SCRIPT_DIR, "sweep_runner.py")

# H20 80GB: PPO ~1GB/run, APG ~5GB/run.
PPO_MEM_LIMIT = 72
APG_MEM_LIMIT = 15

_parent_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _parent_cuda:
  GPUS = [int(x.strip()) for x in _parent_cuda.split(",") if x.strip()]
else:
  GPUS = [0, 1, 2, 3]


def launch(dry_run=False):
  print("=" * 60)
  print("PushBox Solimp/Solref Sweep Launcher")
  print("=" * 60)
  print(f"PPO Project: {PPO_PROJECT}")
  print(f"APG Project: {APG_PROJECT}")
  print(f"Grid: 4×4×4×2 = {4*4*4*2} combos per algorithm")
  print(f"Total runs per algo: {4*4*4*2}")
  algo_name = _ALGORITHM.value or "both"
  print(f"Algorithm: {algo_name.upper()}")
  print(f"GPUs: {GPUS}")
  print()

  print("W&B projects (runs auto-created on first wandb.init):")
  if _ALGORITHM.value is None or _ALGORITHM.value == "ppo":
    print(f"  PPO: {PPO_PROJECT}")
  if _ALGORITHM.value is None or _ALGORITHM.value == "apg":
    print(f"  APG: {APG_PROJECT}")

  grid = list(itertools.product(
      [i / 3 for i in range(4)],  # a0
      [i / 3 for i in range(4)],  # a1
      [i / 3 for i in range(4)],  # a2
      [i / 1 for i in range(2)],  # ar0
  ))
  grid = grid[:_MAX_CONFIGS.value]
  print(f"Grid: {len(grid)} combos per algorithm")
  print()

  if dry_run:
    print("DRY RUN: first 10 configs, not launching.")
    for i, (a0, a1, a2, ar0) in enumerate(grid[:10]):
      suffix = f"a0{a0:.1f}_a1{a1:.1f}_a2{a2:.1f}_ar0{ar0:.1f}"
      print(f"  {suffix}")
    return

  log_dir = os.path.join(SCRIPT_DIR, "..", "logs", "sweep")
  os.makedirs(log_dir, exist_ok=True)
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

  algos_to_run = [
      (algo, project, mem_limit)
      for algo, project, mem_limit in [
          ("ppo", PPO_PROJECT, PPO_MEM_LIMIT),
          ("apg", APG_PROJECT, APG_MEM_LIMIT),
      ]
      if _ALGORITHM.value is None or algo == _ALGORITHM.value
  ]

  for algo, project, mem_limit in algos_to_run:
    per_gpu = min(mem_limit, math.ceil(len(grid) / len(GPUS)))
    print(f"\n{'='*60}")
    print(f"Running {algo.upper()} ({len(grid)} configs)")
    print(f"  Max concurrent per GPU: {per_gpu}")
    print(f"{'='*60}")

    total = len(grid)
    slots = [[combo] for combo in grid]
    gpu_slots = {gpu: [] for gpu in GPUS}
    for slot_id in range(total):
      gpu_slots[GPUS[slot_id % len(GPUS)]].append(slot_id)

    active = {}  # proc -> {slot_id, gpu, log_fh, phase}
    gpu_count = {gpu: 0 for gpu in GPUS}
    ok = fail = 0

    def spawn(gpu, slot_id):
      log_file = os.path.join(log_dir, f"{algo}_{timestamp}_gpu{gpu}_slot{slot_id}.log")
      cmd = [
          sys.executable, RUNNER,
          f"--algorithm={algo}",
          f"--configs={json.dumps(slots[slot_id])}",
      ]
      env = os.environ.copy()
      env["CUDA_VISIBLE_DEVICES"] = str(gpu)
      env["WANDB_MODE"] = "online"
      env["WANDB_PROJECT"] = project
      env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
      env["OMP_NUM_THREADS"] = "1"
      env["MKL_NUM_THREADS"] = "1"
      env["OPENBLAS_NUM_THREADS"] = "1"
      env["PYTHONUNBUFFERED"] = "1"
      proc = subprocess.Popen(
          cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
      )
      fd = proc.stdout.fileno()
      fl = fcntl.fcntl(fd, fcntl.F_GETFL)
      fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
      log_fh = open(log_file, "w")
      active[proc] = {"slot": slot_id, "gpu": gpu, "log": log_fh, "phase": "compiling"}
      gpu_count[gpu] += 1
      prefix = f"[gpu{gpu}/slot{slot_id}]"
      print(f"  {prefix} launched (compiling)")
      return proc

    # Seed: launch one process per GPU
    for gpu in GPUS:
      if gpu_slots[gpu]:
        spawn(gpu, gpu_slots[gpu].pop(0))

    # Dynamic event loop
    try:
      while active:
        fds = [p.stdout.fileno() for p in active]
        readable, _, _ = select.select(fds, [], [], 1.0)
        for fd in readable:
          proc = next(p for p in active if p.stdout.fileno() == fd)
          info = active[proc]
          try:
            data = os.read(fd, 65536)
          except BlockingIOError:
            continue
          if not data:
            # Process exited
            info["log"].close()
            proc.wait()
            gpu = info["gpu"]
            gpu_count[gpu] -= 1
            if proc.returncode == 0:
              ok += 1
            else:
              fail += 1
            del active[proc]
            # Launch next pending on this GPU if any
            if gpu_slots[gpu]:
              spawn(gpu, gpu_slots[gpu].pop(0))
            continue
          # Write output
          text = data.decode("utf-8", errors="replace")
          info["log"].write(text)
          info["log"].flush()
          for line in text.splitlines():
            line = line.rstrip()
            if line:
              print(f"  [gpu{info['gpu']}/slot{info['slot']}] {line}")
          # Detect JIT completion → launch next on same GPU
          if b"JIT_READY" in data:
            info["phase"] = "training"
            gpu = info["gpu"]
            if gpu_slots[gpu] and gpu_count[gpu] < per_gpu:
              time.sleep(0.5)  # small stagger
              spawn(gpu, gpu_slots[gpu].pop(0))
    except KeyboardInterrupt:
      print("\n  Interrupted. Killing remaining processes...")
      for proc in list(active.keys()):
        proc.kill()
      print("  Killed.")
      return

    print(f"  {algo.upper()}: OK={ok}, FAIL={fail}")

  print(f"\nAll complete!")


_ALGORITHM = flags.DEFINE_string("algo", None, "Run only this algorithm (ppo or apg). If None, run both.")
_DRY_RUN = flags.DEFINE_boolean("dry_run", False, "Show schedule without launching")
_MAX_CONFIGS = flags.DEFINE_integer(
    "max_configs", 128,
    "Limit total configs per algorithm (for testing)",
)
if __name__ == "__main__":
  app.run(lambda argv: launch(dry_run=_DRY_RUN.value))
