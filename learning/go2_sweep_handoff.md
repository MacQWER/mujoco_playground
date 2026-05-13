# Go2 Sweep Handoff

Recorded: 2026-05-06

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

Train grid:

```python
solimp0 = [0.015, 0.95]
solimp1 = [0.5, 0.99]
solimp2 = [0.001, 0.031, 0.5]
```

The launcher filters out invalid triplets where `solimp0 > solimp1`, leaving
9 valid train configurations:

```python
[0.015, 0.500, 0.001]
[0.015, 0.500, 0.031]
[0.015, 0.500, 0.500]
[0.015, 0.990, 0.001]
[0.015, 0.990, 0.031]
[0.015, 0.990, 0.500]
[0.950, 0.990, 0.001]
[0.950, 0.990, 0.031]
[0.950, 0.990, 0.500]
```

Eval config is fixed:

```python
eval_solimp = [0.95, 0.99, 0.001]
eval_solref = [0.02, 1.0]
```

Train solref is fixed:

```python
train_solref = [0.02, 1.0]
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

Current launcher grid keeps the original 27 configs in slots `0-26` and appends
the targeted light27 supplement in slots `27-53`. With old APG logs present,
`--skip_completed` should skip slots `0-26` and leave 27 pending APG configs.

Start the APG light27 supplement in the background on four A100s:

```bash
ts=$(date +%Y%m%d_%H%M%S) && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u learning/launch_go2_sweep.py --algo=apg --skip_completed > logs/go2_sweep/launch_light27_${ts}.out 2>&1 &
```

If the shell prints something like `[1] 12478`, `[1]` is the shell job id and
`12478` is the launcher process id.

Follow the launcher log:

```bash
tail -f logs/go2_sweep/launch_light27_*.out
```

`tail -f` prints new log lines as they are written. Press `Ctrl-C` to stop
watching the log; this does not stop the background sweep.

Check which light27 slots have completed:

```bash
for i in $(seq 27 53); do grep -q "video done" logs/go2_sweep/apg_*_slot${i}.log 2>/dev/null && echo "slot $i done" || echo "slot $i pending/running"; done
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
- eval foot geom solimp was set to `[0.95, 0.99, 0.001]`

## Current Worktree Notes

At handoff, relevant sweep files are modified/untracked:

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
