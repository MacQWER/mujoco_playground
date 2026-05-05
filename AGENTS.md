# Repository Guidelines

## Project Structure & Module Organization

`mujoco_playground/` is the importable package. Core environments live in
`mujoco_playground/_src/`, grouped into `dm_control_suite/`, `locomotion/`, and
`manipulation/`; tests are colocated as `*_test.py`. Robot XMLs, meshes, and
textures live under each environment's `xmls/` and `assets/` directories.
`mujoco_playground/config/` contains RL hyperparameter configs. `learning/`
contains training CLIs, sweep helpers, and notebooks. `mujoco_playground/experimental/`
holds sim2sim tools, learning notebooks, and benchmarking. Treat root-level
logs, `wandb/`, `jit_cache/`, and rollout videos as generated artifacts unless
you are intentionally updating experiment outputs.

## Build, Test, and Development Commands

- `uv venv --python 3.11` then `source .venv/bin/activate`: create the supported
  local environment.
- `uv pip install -e ".[cuda,all]"`: install the package editably with CUDA, dev,
  notebook, and learning extras using locked versions.
- `python -c "import mujoco_playground"`: verify imports and external assets.
- `python learning/train_jax_ppo.py --env_name CartpoleBalance`: smoke-test the
  training CLI.
- `pytest` or `pytest mujoco_playground/_src/locomotion/locomotion_test.py`: run
  all tests or a focused suite.
- `pre-commit run --all-files`: run formatting and import hooks before a PR.

## Coding Style & Naming Conventions

Python is formatted with Pyink (`line-length = 80`, two-space indentation) and
imports are sorted with isort's single-line style. Ruff, pylint, mypy, and
pytype settings live in `pyproject.toml` and `pylintrc`. Use snake_case for
Python files, functions, and config helpers. Public environment IDs in registry
maps use clear CamelCase names, for example `Go2Joystick2`. Avoid casual
dependency upgrades; core JAX, MuJoCo, MJX, and Warp versions are pinned for
stability.

## Testing Guidelines

Pytest is configured to discover tests under `mujoco_playground/_src/`. Name new
tests `*_test.py` and colocate them with the module or suite they cover. For new
tasks, document objectives and rewards, add reproducible config entries in
`mujoco_playground/config/`, pass the test suite, and verify behavior across at
least three seeds as described in `CONTRIBUTING.md`.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, often lowercase, such as `pass a
simple test` or `update readme for command no-symloss and no-heuristic`. Keep
subjects specific and under about 72 characters; put experiment context in the
body when needed. PRs should include a clear description, linked issues when
applicable, passing `pytest` and `pre-commit` results, and videos or screenshots
for new robotics behaviors or visual changes. Contributors must have a Google
CLA on file.
