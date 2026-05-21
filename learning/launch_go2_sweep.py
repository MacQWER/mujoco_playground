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
"""Launch Go2Joystick PPO solimp+solref sweep across GPUs.

Grid: 57 train configs with solimp[1] fixed at 0.95, preserving the
old base/light/mid slot families needed for the 3D solimp0-solimp2 surfaces.
Eval solimp/solref is fixed in go2_sweep_runner.py.

Usage:
    cd /data/mujoco_playground
    python learning/launch_go2_sweep.py [--dry_run]
"""

import datetime
import fcntl
import itertools
import json
import math
import os
import re
import select
import subprocess
import sys
import time

from absl import app
from absl import flags

PPO_PROJECT = "go2-joystick-sweep-ppo"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(SCRIPT_DIR, "go2_sweep_runner.py")
LOG_DIR = os.path.join(SCRIPT_DIR, "..", "logs", "go2_sweep")

# Go2 PPO is lightweight in memory on A100s (~1 GiB/process after JIT), and the
# launcher only starts the next process on a GPU after the previous one reports
# JIT_READY.  This keeps JIT compilation serialized while allowing multiple
# training processes to share the same GPU.
PPO_MEM_LIMIT = 1
SOLIMP0_VALUES = [0.015, 0.9]
SOLIMP1_VALUES = [0.95]
SOLIMP2_VALUES = [0.03, 0.001, 0.5]
SOLREF0_VALUES = [0.1, 0.02, 0.004]

LIGHT27_SOLIMP2_VALUES = [0.006, 0.01, 0.05, 0.1]
LIGHT27_FULL_PAIRS = [(0.015, 0.95)]
LIGHT27_DIAGNOSTIC_PAIR = (0.9, 0.95)
LIGHT27_DIAGNOSTIC_SOLIMP2_VALUES = [0.1]

MID42_SOLIMP0_VALUES = [0.03, 0.1, 0.35, 0.7]
MID42_SOLIMP2_VALUES = [0.03, 0.1]

_parent_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _parent_cuda:
  GPUS = [int(x.strip()) for x in _parent_cuda.split(",") if x.strip()]
else:
  GPUS = [0, 1, 2, 3]


def build_base_grid():
  """Returns the solimp[1]=0.95 base grid."""
  return [
      combo
      for combo in itertools.product(
          SOLIMP0_VALUES,
          SOLIMP1_VALUES,
          SOLIMP2_VALUES,
          SOLREF0_VALUES,
      )
      if combo[0] <= combo[1]
  ]


def build_light27_grid():
  """Returns the solimp[1]=0.95 light15 solimp[2] supplement grid."""
  light27_grid = []
  for sr0 in SOLREF0_VALUES:
    for s0, s1 in LIGHT27_FULL_PAIRS:
      for s2 in LIGHT27_SOLIMP2_VALUES:
        light27_grid.append((s0, s1, s2, sr0))
    s0, s1 = LIGHT27_DIAGNOSTIC_PAIR
    for s2 in LIGHT27_DIAGNOSTIC_SOLIMP2_VALUES:
      light27_grid.append((s0, s1, s2, sr0))

  return light27_grid


def build_mid42_grid():
  """Returns intermediate solimp[0] slots appended after light27."""
  return [
      combo
      for combo in itertools.product(
          MID42_SOLIMP0_VALUES,
          SOLIMP1_VALUES,
          MID42_SOLIMP2_VALUES,
          SOLREF0_VALUES,
      )
      if combo[0] <= combo[1]
  ]


def build_grid():
  """Returns the 57-slot solimp[1]=0.95 PPO sweep grid."""
  return build_base_grid() + build_light27_grid() + build_mid42_grid()


def _find_completed_slots(algo, log_dir):
  """Returns slot ids with completed training/video logs."""
  if not os.path.isdir(log_dir):
    return set()

  completed = set()
  pattern = re.compile(rf"^{re.escape(algo)}_.*_slot(\d+)\.log$")
  for filename in os.listdir(log_dir):
    match = pattern.match(filename)
    if match is None:
      continue

    path = os.path.join(log_dir, filename)
    try:
      with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    except OSError:
      continue

    if "video done" in text:
      completed.add(int(match.group(1)))

  return completed


def _format_slots(slot_ids):
  slot_ids = sorted(slot_ids)
  if len(slot_ids) <= 24:
    return ", ".join(str(x) for x in slot_ids)
  head = ", ".join(str(x) for x in slot_ids[:16])
  tail = ", ".join(str(x) for x in slot_ids[-6:])
  return f"{head}, ..., {tail}"


def launch(dry_run=False):
  base_count = len(build_base_grid())
  light27_count = len(build_light27_grid())
  mid42_count = len(build_mid42_grid())
  light27_start = base_count
  light27_end = light27_start + light27_count - 1
  mid42_start = light27_end + 1
  mid42_end = mid42_start + mid42_count - 1

  print("=" * 60)
  print("Go2 Solimp Sweep Launcher")
  print("=" * 60)
  if _ALGORITHM.value not in (None, "ppo"):
    raise ValueError("Go2Joystick sweep is PPO-only; use --algo=ppo.")

  print(f"PPO Project: {PPO_PROJECT}")
  print(f"Base grid: {base_count} configs in slots 0-{base_count - 1}.")
  print(
      f"Light15 supplement: slots {light27_start}-{light27_end}; "
      "pair (0.015,0.95) uses solimp2=[0.006,0.01,0.05,0.1], "
      "pair (0.9,0.95) keeps diagnostic solimp2=[0.1]."
  )
  print(
      f"Mid42 supplement: slots {mid42_start}-{mid42_end}; "
      "solimp0=[0.03,0.1,0.35,0.7], "
      "solimp1=0.95, solimp2=[0.03,0.1]."
  )
  print("solref0=[0.1, 0.02, 0.004]")
  print("Filter: solimp[0] <= solimp[1]")
  algo_name = _ALGORITHM.value or "ppo"
  print(f"Algorithm: {algo_name.upper()}")
  print(f"GPUs: {GPUS}")
  print()

  print("W&B project (runs auto-created on first wandb.init):")
  print(f"  PPO: {PPO_PROJECT}")

  grid = build_grid()
  grid = grid[:_MAX_CONFIGS.value]
  print(f"Grid: {len(grid)} PPO configs")
  print()

  log_dir = LOG_DIR
  all_slots = list(enumerate(grid))

  algos_to_run = [("ppo", PPO_PROJECT, PPO_MEM_LIMIT)]

  if dry_run:
    print("DRY RUN: not launching.")
    for algo, _, mem_limit in algos_to_run:
      completed_slots = _find_completed_slots(algo, log_dir)
      if _SKIP_COMPLETED.value:
        pending_slots = [
            (slot_id, combo)
            for slot_id, combo in all_slots
            if slot_id not in completed_slots
        ]
      else:
        pending_slots = all_slots

      print(f"\n{algo.upper()}: {len(pending_slots)}/{len(grid)} pending")
      per_gpu = min(mem_limit, math.ceil(len(pending_slots) / len(GPUS)))
      print(f"  Max concurrent per GPU: {per_gpu}")
      if _SKIP_COMPLETED.value:
        skipped = [slot_id for slot_id, _ in all_slots if slot_id in completed_slots]
        print(f"  Skipping completed: {len(skipped)}")
        if skipped:
          print(f"  Completed slots: {_format_slots(skipped)}")
      print("  First pending configs:")
      for slot_id, (s0, s1, s2, sr0) in pending_slots[:10]:
        print(f"    slot {slot_id}: solimp=[{s0:.3f}, {s1:.3f}, {s2:.3f}] solref=[{sr0:.3f}, 1.0]")
    return

  os.makedirs(log_dir, exist_ok=True)
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

  for algo, project, mem_limit in algos_to_run:
    completed_slots = _find_completed_slots(algo, log_dir)
    if _SKIP_COMPLETED.value:
      pending_slots = [
          (slot_id, combo)
          for slot_id, combo in all_slots
          if slot_id not in completed_slots
      ]
    else:
      pending_slots = all_slots

    per_gpu = min(mem_limit, math.ceil(len(pending_slots) / len(GPUS)))
    print(f"\n{'='*60}")
    print(f"Running {algo.upper()} ({len(pending_slots)}/{len(grid)} configs)")
    if _SKIP_COMPLETED.value:
      skipped = [slot_id for slot_id, _ in all_slots if slot_id in completed_slots]
      print(f"  Skipping completed: {len(skipped)}")
      if skipped:
        print(f"  Completed slots: {_format_slots(skipped)}")
    print(f"  Max concurrent per GPU: {per_gpu}")
    print(f"{'='*60}")

    if not pending_slots:
      print(f"  {algo.upper()}: nothing to run")
      continue

    total = len(pending_slots)
    slots = {slot_id: [combo] for slot_id, combo in pending_slots}
    gpu_slots = {gpu: [] for gpu in GPUS}
    for i, (slot_id, _) in enumerate(pending_slots):
      gpu_slots[GPUS[i % len(GPUS)]].append(slot_id)

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
      env["WANDB_MODE"] = "offline"
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


_ALGORITHM = flags.DEFINE_string("algo", "ppo", "Run PPO sweep. APG is not supported for Go2Joystick sweep.")
_DRY_RUN = flags.DEFINE_boolean("dry_run", False, "Show schedule without launching")
_MAX_CONFIGS = flags.DEFINE_integer(
    "max_configs", 57,
    "Limit total PPO configs (for testing)",
)
_SKIP_COMPLETED = flags.DEFINE_boolean(
    "skip_completed", False,
    "Skip slots whose previous log contains the 'video done' completion marker.",
)
if __name__ == "__main__":
  app.run(lambda argv: launch(dry_run=_DRY_RUN.value))
