# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MuJoCo Playground** is a Google DeepMind open-source library providing GPU-accelerated robotics environments for robot learning research and sim-to-real transfer. Built with MuJoCo MJX (JAX-based MuJoCo) and MuJoCo Warp. It includes locomotion (quadrupeds, humanoids), manipulation (dexterous hands, pick-and-place), and classic control environments.

This fork focuses on Unitree Go2 quadruped locomotion with residual learning (APG/PPO) and state alignment for sim-to-real transfer. The `Go2Joystick2` and `Go2JoystickMujoco` envs are custom additions supporting anchor+residual policy architecture.

## Code Architecture

```
mujoco_playground/
├── _src/
│   ├── mjx_env.py              # Base MjxEnv class; State dataclass; step() fn
│   ├── registry.py             # Unified registry across all env suites
│   ├── locomotion/
│   │   ├── __init__.py         # Registers all locomotion envs (_envs, _cfgs, _randomizer maps)
│   │   ├── go2/                # Unitree Go2 environments (main focus)
│   │   │   ├── base.py         # Go2Env base class (extends MjxEnv), sensor methods
│   │   │   ├── JoystickGo2.py  # Residual-learning joystick env (MJX backend)
│   │   │   ├── JoystickGo2Mujoco.py  # MuJoCo native backend version
│   │   │   ├── go2_constants.py  # Paths (XML, checkpoints), FEET_SITES, HIP_NAMES, etc.
│   │   │   ├── configs/          # Environment config definitions (joystick_config.py)
│   │   │   ├── managers/         # State alignment, assistive wrench managers
│   │   │   ├── mdp/              # MDP components: observations, rewards, randomize
│   │   │   └── xmls/             # MuJoCo XML model files
│   │   └── {other robots}/     # anymal, g1, h1, spot, t1, etc.
│   ├── dm_control_suite/       # Classic control tasks
│   └── manipulation/           # Manipulation tasks (Panda, Allegro, etc.)
├── config/
│   ├── locomotion_params.py    # RL hyperparameter configs (brax_apg_config, etc.)
│   ├── dm_control_suite_params.py
│   └── manipulation_params.py
├── learning/
│   └── train_jax_ppo.py        # Main training CLI entry point
└── experimental/
    └── learning/               # Notebooks: fine-tuning, sim2sim, ablation studies
```

### Key Design Patterns

1. **Environment Registration**: Environments are registered in `_src/locomotion/__init__.py` via `_envs`, `_cfgs`, and `_randomizer` dicts. Use `locomotion.load(env_name)` and `locomotion.get_default_config(env_name)`.

2. **Manager Pattern**: Go2 envs use composable managers (`ObservationManager`, `ActionManager`, `CommandManager`, `EventManager`, `TerminationManager`, `RewardManager`) defined in `mujoco_playground/_src/locomotion/go2/managers/`.

3. **Config System**: Uses `ml_collections.ConfigDict`. Each env has a `default_config()` function. Configs are layered: base config → env-specific overrides → user overrides.

4. **Dual-Backend (MJX + MuJoCo)**: `Go2Joystick2` uses MJX (JAX) backend for training. `Go2JoystickMujoco` uses native MuJoCo for sim-to-real validation. State Alignment Manager bridges the two by computing gradients through MJX while forward-passing through MuJoCo.

5. **Residual Policy Architecture**: An anchor policy (pre-trained TrotGo2) provides a base gait, and a residual policy learns corrections. This is the core of the `Go2Joystick2` design.

## Development Commands

### Setup
```bash
# Using uv (recommended)
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -U "jax[cuda12]"
uv pip install -e ".[all]"
```

### Training
```bash
python learning/train_jax_ppo.py --env_name Go2Joystick2
```

### Testing
```bash
pytest                            # Run all tests
pytest mujoco_playground/_src/    # Run specific test dirs
```

### Code Quality
```bash
pre-commit run --all-files        # Run all linters (pyink, isort, pylint, pytype)
pyink .                           # Format code
isort .                           # Sort imports
pylint . --rcfile=pylintrc        # Lint
pytype .                          # Type checking
```

### Key Files for Go2 Work
- `mujoco_playground/_src/locomotion/go2/JoystickGo2.py` — Main MJX env
- `mujoco_playground/_src/locomotion/go2/JoystickGo2Mujoco.py` — MuJoCo native env
- `mujoco_playground/_src/locomotion/go2/managers/state_alignment_manager.py` — Sim-to-real alignment
- `mujoco_playground/_src/locomotion/go2/go2_constants.py` — Paths and constants
- `mujoco_playground/_src/locomotion/go2/configs/joystick_config.py` — Default env config
- `experimental/learning/real_world_fine_tuning.ipynb` — Sim-to-real fine-tuning notebook
