# Go2 Sweep Handoff

Recorded: 2026-05-06
Updated: 2026-05-14

## Goal

Run a small Go2 solimp sweep for `Go2Joystick2`.

## Files

Go2-specific sweep files:

- `learning/launch_go2_sweep.py`
- `learning/go2_sweep_runner.py`

Original PushBox sweep files remain separate:

- `learning/launch_sweep.py`
- `learning/sweep_runner.py`

Do not use `learning/launch_sweep.py` for the Go2 sweep.

## Go2 Sweep Setup

Environment:

- `Go2Joystick2`

Current launcher grid keeps old slot IDs stable and only appends new work.

Base grid, slots `0-26`:

```python
solimp0 = [0.015, 0.9]
solimp1 = [0.5, 0.95]
solimp2 = [0.03, 0.001, 0.5]
solref0 = [0.1, 0.02, 0.004]
```

The launcher filters out invalid triplets where `solimp0 > solimp1`, leaving
9 valid train solimp triplets and 27 total base slots:

```python
[0.015, 0.500, 0.001]
[0.015, 0.500, 0.030]
[0.015, 0.500, 0.500]
[0.015, 0.950, 0.001]
[0.015, 0.950, 0.030]
[0.015, 0.950, 0.500]
[0.900, 0.950, 0.001]
[0.900, 0.950, 0.030]
[0.900, 0.950, 0.500]
```

Completed light27 supplement, slots `27-53`:

```python
solref0 = [0.1, 0.02, 0.004]
pairs = [(0.015, 0.5), (0.015, 0.95)]
solimp2 = [0.006, 0.01, 0.05, 0.1]
diagnostic_pair = (0.9, 0.95)
diagnostic_solimp2 = [0.1]
```

New mid42 solimp0 supplement, slots `54-95`:

```python
solimp0 = [0.03, 0.1, 0.35, 0.7]
solimp1 = [0.5, 0.95]
solimp2 = [0.03, 0.1]
solref0 = [0.1, 0.02, 0.004]
```

This gives 42 new configs because `(0.7, 0.5)` is filtered out by
`solimp0 <= solimp1`.

The exact mid42 train pairs are:

```python
(0.03, 0.5)
(0.03, 0.95)
(0.1, 0.5)
(0.1, 0.95)
(0.35, 0.5)
(0.35, 0.95)
(0.7, 0.95)
```

Eval config is fixed:

```python
eval_solimp = [0.9, 0.95, 0.001]
eval_solref = [0.004, 1.0]
```

Train solref values:

```python
train_solref = [[0.1, 1.0], [0.02, 1.0], [0.004, 1.0]]
```

## Outputs

Go2 logs:

```text
logs/go2_sweep/
```

Go2 rollout videos:

```text
logs/go2_sweep/videos/apg/
logs/go2_sweep/videos/ppo/
```

Go2 softness calibration:

```text
logs/go2_sweep/softness_ranking_augmented_mid42.csv
```

W&B projects:

```text
go2-sweep-apg
go2-sweep-ppo
```

Concurrency:

- Go2 is large; the launcher is configured to run at most 1 training process
  per GPU for both APG and PPO.

## Commands

Dry-run current APG grid with completed slots skipped:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python learning/launch_go2_sweep.py --dry_run --algo=apg --skip_completed
```

Current launcher grid keeps the original 27 configs in slots `0-26`, the
completed light27 supplement in slots `27-53`, and appends the new mid42
supplement in slots `54-95`. With old APG logs present, `--skip_completed`
should skip slots `0-53` and leave 42 pending APG configs.

Start the APG mid42 supplement in the background on four A100s:

```bash
ts=$(date +%Y%m%d_%H%M%S) && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u learning/launch_go2_sweep.py --algo=apg --skip_completed > logs/go2_sweep/launch_mid42_${ts}.out 2>&1 &
```

The older `launch_light27_${ts}.out` filename still works because the launcher
schedule is determined by the Python grid and `--skip_completed`, not the log
filename.

If the shell prints something like `[1] 12478`, `[1]` is the shell job id and
`12478` is the launcher process id.

Follow the launcher log:

```bash
tail -f logs/go2_sweep/launch_mid42_*.out
```

`tail -f` prints new log lines as they are written. Press `Ctrl-C` to stop
watching the log; this does not stop the background sweep.

Check which mid42 slots have completed:

```bash
for i in $(seq 54 95); do grep -q "video done" logs/go2_sweep/apg_*_slot${i}.log 2>/dev/null && echo "slot $i done" || echo "slot $i pending/running"; done
```

Regenerate the full base + light27 + mid42 ball-drop softness ranking:

```bash
python learning/ball_drop_softness.py
```

Check whether the launcher and runner processes are still alive:

```bash
ps -f -u $(whoami) | grep -E "launch_go2_sweep|go2_sweep_runner" | grep -v grep
```

Dry-run APG:

```bash
python learning/launch_go2_sweep.py --dry_run --algo=apg --skip_completed
```

Dry-run PPO:

```bash
python learning/launch_go2_sweep.py --dry_run --algo=ppo --skip_completed
```

Start APG sweep:

```bash
python learning/launch_go2_sweep.py --algo=apg
```

Start PPO sweep:

```bash
python learning/launch_go2_sweep.py --algo=ppo
```

Resume and skip completed slots:

```bash
python learning/launch_go2_sweep.py --algo=apg --skip_completed
python learning/launch_go2_sweep.py --algo=ppo --skip_completed
```

## Verified

The following checks passed before handoff:

```bash
python -m py_compile learning/launch_sweep.py learning/sweep_runner.py learning/launch_go2_sweep.py learning/go2_sweep_runner.py
python learning/launch_sweep.py --dry_run --algo=apg --max_configs=3
python learning/launch_go2_sweep.py --dry_run --algo=apg --skip_completed
python learning/launch_go2_sweep.py --dry_run --algo=ppo --skip_completed
python learning/sweep_runner.py --configs='[]' --algorithm=apg
python learning/go2_sweep_runner.py --configs='[]' --algorithm=apg
```

Also verified by loading `Go2Joystick2` once that:

- train foot geom solimp was set to `[0.015, 0.5, 0.001]`
- eval foot geom solimp was set to `[0.9, 0.95, 0.001]`

2026-05-14 mid42 update checks:

```bash
python -m py_compile learning/launch_go2_sweep.py learning/plot_go2_sweep.py learning/ball_drop_softness.py
python learning/launch_go2_sweep.py --dry_run --algo=apg --skip_completed
```

The APG dry-run reported:

```text
Grid: 96 combos per algorithm
APG: 42/96 pending
Skipping completed: 54
First pending slot: 54 -> solimp=[0.030, 0.500, 0.030] solref=[0.100, 1.0]
Last mid42 slot: 95 -> solimp=[0.700, 0.950, 0.100] solref=[0.004, 1.0]
```

## 2026-05-15 APG Kill Checkpoint

Search keywords: `2026-05-15 kill checkpoint`, `APG slot 74`,
`mid42 resume`.

The APG mid42 sweep launched from:

```bash
ts=$(date +%Y%m%d_%H%M%S) && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u learning/launch_go2_sweep.py --algo=apg --skip_completed > logs/go2_sweep/launch_mid42_${ts}.out 2>&1 &
```

was stopped on 2026-05-15 at about 01:23 CST. The launcher process group
`9864` was killed with `SIGKILL` after it had previously been stopped with
`SIGSTOP`. A follow-up `ps` showed no remaining `launch_go2_sweep.py`,
`go2_sweep_runner.py`, or matching wandb worker processes. `nvidia-smi` showed
GPUs 0-3 at `0 / 81920 MiB`.

Current APG progress from:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python learning/launch_go2_sweep.py --dry_run --algo=apg --skip_completed
```

was:

```text
Grid: 96 combos per algorithm
APG: 22/96 pending
Skipping completed: 74
Completed slots: 0-73
First pending slot: 74 -> solimp=[0.100, 0.950, 0.030] solref=[0.004, 1.0]
```

The killed launcher log was:

```text
logs/go2_sweep/launch_mid42_20260514_211334.out
```

Slots `74`, `75`, `76`, and `77` had started and printed `JIT_READY`, but they
did not reach `video done` before the kill:

```text
slot 74: solimp=[0.100, 0.950, 0.030] solref=[0.004, 1.0]
slot 75: solimp=[0.100, 0.950, 0.100] solref=[0.100, 1.0]
slot 76: solimp=[0.100, 0.950, 0.100] solref=[0.020, 1.0]
slot 77: solimp=[0.100, 0.950, 0.100] solref=[0.004, 1.0]
```

They should be treated as incomplete and will rerun from scratch. The launcher
does not resume inside a single slot; `--skip_completed` only skips slot logs
that contain `video done`.

To resume APG later:

```bash
ts=$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u learning/launch_go2_sweep.py --algo=apg --skip_completed > logs/go2_sweep/launch_mid42_${ts}.out 2>&1 &
```

To verify before resuming:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python learning/launch_go2_sweep.py --dry_run --algo=apg --skip_completed
```

## 2026-05-17 PPO Missing Slots

PPO currently only has completed logs for slots `0-26`, which are the original
base grid. The light27 supplement (`27-53`) and mid42 supplement (`54-95`) are
still missing for PPO. Use `--skip_completed` so the launcher keeps slots `0-26`
and runs only the missing slots.

Verify the pending PPO schedule first:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python learning/launch_go2_sweep.py --dry_run --algo=ppo --skip_completed
```

Expected status before rerunning:

```text
Grid: 96 combos per algorithm
PPO: 69/96 pending
Completed slots: 0-26
First pending slot: 27
Last pending slot: 95
```

Start the missing PPO sweep in the background:

```bash
mkdir -p logs/go2_sweep_apg && ts=$(date +%Y%m%d_%H%M%S) && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u learning/launch_go2_sweep.py --algo=ppo --skip_completed > logs/go2_sweep_apg/launch_ppo_missing_${ts}.out 2>&1 &
```

Follow the launcher log:

```bash
tail -f logs/go2_sweep_apg/launch_ppo_missing_*.out
```

```bash
tail -f logs/go2_sweep_apg/launch_ppo_missing_20260518_005552.out
```

Check launcher/runner processes:

```bash
ps -f -u $(whoami) | grep -E "launch_go2_sweep|go2_sweep_runner" | grep -v grep
```

Check completion for the missing PPO slots:

```bash
for i in $(seq 27 95); do grep -q "video done" logs/go2_sweep/ppo_*_slot${i}.log 2>/dev/null && echo "slot $i done" || echo "slot $i pending/running"; done
```

After PPO finishes, regenerate the PPO plots and CSV:

```bash
python -u learning/plot_go2_sweep.py --algo=ppo
```

Regenerate both APG and PPO plots if needed:

```bash
python -u learning/plot_go2_sweep.py --algo=both
```

## Historical Worktree Notes

At the 2026-05-06 handoff, relevant sweep files were modified/untracked:

```text
 M learning/launch_sweep.py
 M learning/sweep_runner.py
?? learning/go2_sweep_runner.py
?? learning/launch_go2_sweep.py
?? learning/go2_sweep_handoff.md
```

There were also pre-existing untracked plotting files:

```text
?? learning/plot_apg_sweep.py
?? learning/plot_sweep_shared_colors.py
```


## 2026-05-21 Go2Joystick PPO Rerun Notes

The old PPO sweep data was discarded because it used the buggy
`Go2Joystick2` setup. The rerun uses `Go2Joystick`, PPO only, with
`solimp1=0.95` fixed and the normal environment random command sampling.
The sweep has 57 slots:

- base grid: 18 slots
- light supplement: 15 slots
- mid supplement: 24 slots

Runner settings for this rerun:

- `TRAIN_ITERATIONS = 10`
- `EVAL_ITERATIONS = 10`
- `jax_enable_x64 = False`
- no fixed command wrapper
- no `jax_default_matmul_precision` override

Latest completed run:

```text
launch log: logs/go2_sweep/launch_ppo_20260521_112618.out
result csv: logs/go2_sweep/figures/ppo_go2_sweep_results.csv
status: PPO OK=57, FAIL=0
rows: 57
solimp1 values: [0.95]
solref0 groups: 19 points each for 0.1, 0.02, and 0.004
reward range: 13.55 to 27.71, mean 24.54
```

Generated surface figures:

```text
logs/go2_sweep/figures/ppo_go2_solimp0_solimp2_reward_surface_solref_0p1.png
logs/go2_sweep/figures/ppo_go2_solimp0_solimp2_reward_surface_solref_0p02.png
logs/go2_sweep/figures/ppo_go2_solimp0_solimp2_reward_surface_solref_0p004.png
```

Surface interpretation:

- `solref0=0.1` is the softest train-contact group and is visibly unstable:
  reward ranges from 13.55 to 26.24 across its 19 points.
- `solref0=0.02` and `solref0=0.004` both form high-reward plateau regions:
  rewards stay in the narrower ranges 26.26 to 27.45 and 26.31 to 27.71.
- In the two harder groups, darker/harder regions generally trend toward
  higher reward, but the relation is not strictly monotonic point-by-point.
- Practical takeaway: avoid the very soft `solref0=0.1` regime; harder contact
  settings are more stable and usually better, with diminishing differences
  once the setting is already in the `0.02` or `0.004` range.
