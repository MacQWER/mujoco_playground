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
