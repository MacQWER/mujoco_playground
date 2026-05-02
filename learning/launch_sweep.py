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

import fcntl
import itertools
import json
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

GPUS = [0, 1, 2, 3]
# H20 80GB: PPO ~1GB/run, APG ~5GB/run.
# Leave ~10% headroom.
PPO_PER_GPU = 72
APG_PER_GPU = 15


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
  print(f"Wave size: {_WAVE_SIZE.value}")
  print()

  if dry_run:
    print("DRY RUN: first 10 configs, not launching.")
    for i, (a0, a1, a2, ar0) in enumerate(grid[:10]):
      suffix = f"a0{a0:.1f}_a1{a1:.1f}_a2{a2:.1f}_ar0{ar0:.1f}"
      print(f"  {suffix}")
    return

  log_dir = os.path.join(SCRIPT_DIR, "..", "logs", "sweep")
  os.makedirs(log_dir, exist_ok=True)

  algos_to_run = [
      (algo, project, per_gpu)
      for algo, project, per_gpu in [
          ("ppo", PPO_PROJECT, PPO_PER_GPU),
          ("apg", APG_PROJECT, APG_PER_GPU),
      ]
      if _ALGORITHM.value is None or algo == _ALGORITHM.value
  ]

  for algo, project, per_gpu in algos_to_run:
    print(f"\n{'='*60}")
    print(f"Running {algo.upper()} ({len(grid)} configs)")
    print(f"{'='*60}")

    slots = [[combo] for combo in grid]
    total_slots = len(grid)
    wave_size = _WAVE_SIZE.value
    ok = fail = 0
    total = total_slots

    for wave_start in range(0, total, wave_size):
      wave = range(wave_start, min(wave_start + wave_size, total))
      procs = []

      for slot_id in wave:
        gpu = slot_id % 4
        log_file = os.path.join(log_dir, f"{algo}_gpu{gpu}_slot{slot_id}.log")

        cmd = [
            sys.executable, RUNNER,
            f"--algorithm={algo}",
            f"--configs={json.dumps(slots[slot_id])}",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["WANDB_PROJECT"] = project
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append((proc, slot_id, gpu, log_file))
        time.sleep(0.05)

      # Stream output from all subprocesses concurrently using select
      wave_ok = wave_fail = 0
      # Make stdout non-blocking
      for proc, slot_id, gpu, log_file in procs:
        fd = proc.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

      # Track open file handles for log writing
      log_files = {}
      for proc, slot_id, gpu, log_file in procs:
        log_files[proc] = open(log_file, "w")

      # Track which processes have finished reading
      remaining = {proc for proc, _, _, _ in procs}

      while remaining:
        readable, _, _ = select.select(
            [proc.stdout for proc in remaining], [], [], 1.0
        )
        for fd in readable:
          proc = next(p for p in remaining if p.stdout.fileno() == fd.fileno())
          try:
            data = os.read(fd.fileno(), 65536)
          except BlockingIOError:
            continue
          if not data:
            remaining.discard(proc)
            continue
          text = data.decode("utf-8", errors="replace")
          for line in text.splitlines(True):
            line = line.rstrip()
            if not line:
              continue
            proc_info = next(
                (p for p in procs if p[0] is proc), None
            )
            if proc_info:
              _, slot_id, gpu, _ = proc_info
              prefix = f"[gpu{gpu}/slot{slot_id}]"
              print(f"  {prefix} {line}")
            log_files[proc].write(line + "\n")
            log_files[proc].flush()

      # Close log files and wait for processes
      for proc, slot_id, gpu, log_file in procs:
        log_files[proc].close()
        proc.wait()
        if proc.returncode == 0:
          wave_ok += 1
        else:
          wave_fail += 1
        ok += 1 if proc.returncode == 0 else 0
        fail += 1 if proc.returncode != 0 else 0

      print(f"\n  Wave {wave_start//wave_size + 1}: {wave_ok} OK, {wave_fail} FAIL [{ok}/{total}]\n")

    print(f"  {algo.upper()}: OK={ok}, FAIL={fail}")

  print(f"\nAll complete!")


_ALGORITHM = flags.DEFINE_string("algo", None, "Run only this algorithm (ppo or apg). If None, run both.")
_DRY_RUN = flags.DEFINE_boolean("dry_run", False, "Show schedule without launching")
_MAX_CONFIGS = flags.DEFINE_integer(
    "max_configs", 128,
    "Limit total configs per algorithm (for testing)",
)
_WAVE_SIZE = flags.DEFINE_integer(
    "wave_size", 8,
    "Concurrent processes per wave",
)

if __name__ == "__main__":
  app.run(lambda argv: launch(dry_run=_DRY_RUN.value))
