# Repository Guidelines

## Project Structure & Module Organization

Core package code lives in `mujoco_playground/`. The main implementation is under `mujoco_playground/_src/`, with shared infrastructure in `_src/mjx_env.py`, `_src/registry.py`, and `_src/wrapper*.py`. Environment suites are grouped by domain: `_src/locomotion/`, `_src/manipulation/`, and `_src/dm_control_suite/`. XMLs, meshes, textures, and robot assets are colocated under `xmls/` or `assets/`. RL hyperparameter configs live in `mujoco_playground/config/`. Training and sweep entry points are in `learning/`; notebooks and prototypes are in `learning/notebooks/` and `mujoco_playground/experimental/`. Tests are colocated with source files and named `*_test.py`.

## Build, Test, and Development Commands

- `uv venv --python 3.11 && source .venv/bin/activate`: create and activate the recommended local environment.
- `uv pip install -e ".[cuda,all]"`: install CUDA JAX support, dev tools, notebooks, and learning extras from `uv.lock`.
- `pip install -e ".[dev]"`: install the lighter development dependency set when CUDA/training extras are not needed.
- `pytest`: run tests configured for `mujoco_playground/_src/`.
- `pre-commit run --all-files`: run formatting and import-sorting hooks.
- `python learning/train_jax_ppo.py --env_name CartpoleBalance`: smoke-test a training entry point.

## Coding Style & Naming Conventions

Use Python 3.11 or 3.12. Formatting is controlled by Pyink (`line-length = 80`, 2-space indentation) and imports by isort with one import per line. Keep environment classes and robot constants close to their suite, for example `mujoco_playground/_src/locomotion/go2/`. Prefer `snake_case` for functions, variables, and modules; use `CamelCase` for classes and config objects where existing code does. Run `pyink .`, `isort .`, `pylint . --rcfile=pylintrc`, and `pytype .` before larger changes.

## Testing Guidelines

Tests use `pytest` plus `absl-py` conventions in some files. Add tests next to covered code with names like `registry_test.py` or `locomotion_test.py`. For new tasks, verify registration, reset/step behavior, and config paths. Contribution docs expect new tasks to work across at least three seeds and include reproducible RL hyperparameters.

## Commit & Pull Request Guidelines

Recent history uses short, imperative commit summaries, often lowercase, with occasional Conventional Commit prefixes such as `fix:`. Keep subjects specific, for example `fix Go2 reward scale overwrite`. PRs should describe the behavioral change, list test commands run, link relevant issues, and include videos or plots for new robot tasks or policy behavior. Do not commit generated experiment outputs such as `logs/`, `wandb/`, `jit_cache/`, rollout videos, or local checkpoints unless explicitly requested.

## Security & Configuration Tips

Do not hard-code credentials, W&B keys, or machine-specific CUDA paths. Prefer CLI flags and environment variables such as `CUDA_VISIBLE_DEVICES` and `JAX_DEFAULT_MATMUL_PRECISION=highest` for local runtime differences.
