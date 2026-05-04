# Sweep Notes

PushBox solimp/solref parameter sweep with PPO and APG.

## Grid

```
a0, a1, a2 ∈ {0, 0.33, 0.67, 1.0}   (solimp interpolation)
ar0 ∈ {0, 1.0}                         (solref interpolation)
Total: 4×4×4×2 = 128 combos per algorithm

solimp[i] = (1-a)×BASE[i] + a×HIGH[i]
  BASE = [0.01, 0.5, 0.03]     (soft)
  HIGH = [0.95, 0.99, 0.001]   (hard)

solref[0] = (1-ar0)×0.004 + ar0×0.02
solref[1] = 1.0 (fixed)
```

Eval always uses hard params: `solimp=[0.95, 0.99, 0.001], solref=[0.004, 1.0]`.

## PPO Results (2026-05-04, 128/128 completed)

Reward range: **-44.67 to -20.86** (higher is better). All penetration depths < 0.00002.

**Best**: `solimp=[0.292, 0.5, 0.001], solref=[0.02, 1.0]` (`a=[0.30,0.00,1.00,1.00]`), reward = **-20.86**

Patterns:
- a2=1.0 (low impedance on error term) consistently strong
- ar0=1.0 (hard solref) + high a2 is the winning combo
- a0=0 (very soft first dimension) is universally bad, especially with ar0=1.0
- ar0=0 (soft solref) gives mid-range but never top-tier results
- a1 (impedance interpolation) has the weakest individual effect

See `logs/sweep/figures/` for visualizations.

## APG Sweep

Not yet run. Command:
```bash
CUDA_VISIBLE_DEVICES=0,1,3 python learning/launch_sweep.py --algo=apg --max_configs=128
```

## Sweep Infrastructure

- `launch_sweep.py`: Orchestrator with per-GPU dynamic scheduling
  - Reads `CUDA_VISIBLE_DEVICES` env var
  - Adaptive concurrency: `per_gpu = min(mem_limit, ceil(total / len(GPUS)))`
  - PPO_MEM_LIMIT=72, APG_MEM_LIMIT=15 (H20 80GB)
  - JIT_READY gate serializes XLA compilation per GPU
- `sweep_runner.py`: Single-process worker, runs configs sequentially
  - Random 0-5s stagger on startup

## Performance Findings

1. **XLA compilation is the dominant bottleneck.** Each config has unique solimp/solref → unique XLA graph → no reuse. 128 compilations × ~2-5 min each dominates wall-clock.

2. **Per-GPU JIT serialization works.** XLA limits one compilation per GPU at a time. The JIT_READY gate prevents multiple processes from compiling on the same GPU simultaneously.

3. **GPU oversubscription after JIT phase.** Once compilations finish, too many concurrent training processes on one GPU causes memory bandwidth contention and per-process slowdown. The adaptive cap (max 10 PPO, 6 APG) mitigates this.

4. **Adding more CPUs helps marginally.** XLA has internal serial passes that don't scale linearly beyond ~32-48 cores. More GPUs would help more (parallel compilations).

5. **JAX persistent cache may hurt more than help.** With all-unique XLA keys (zero hits), the shared SQLite cache only adds lock contention. Hypothesis: disabling it (`jax_compilation_cache_dir` unset) would let each process compile independently without SQLITE_BUSY stalls. **Not yet tested.**
