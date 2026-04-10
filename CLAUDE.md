# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MuJoCo Playground** is a Google DeepMind open-source library providing GPU-accelerated robotics environments for robot learning research and sim-to-real transfer. Built with MuJoCo MJX (JAX-based MuJoCo) and MuJoCo Warp. It includes locomotion (quadrupeds, humanoids), manipulation (dexterous hands, pick-and-place), and classic control environments.

This fork focuses on Unitree Go2 quadruped locomotion with residual learning (APG/PPO) and state alignment for sim-to-real transfer. The `Go2Joystick2` and `Go2JoystickMujoco` envs are custom additions supporting anchor+residual policy architecture.

## Code Architecture

```bash
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

## Real World Fine-tuning Project (Current Focus)

### Goal

Use APG (Advantage Policy Gradient) to fine-tune a residual policy for sim-to-real transfer on Unitree Go2.

### Approach

1. **Sim-to-Real Strategy**: Use `Go2AlignmentEnv` which combines MJX (backward pass) + Native MuJoCo (forward pass)
2. **Physics Mismatch**: Modify parameters via `mj_twin_template` to simulate real-world differences
3. **Single-Environment Training**: `num_envs=1` to simulate single real robot constraint
4. **Residual Architecture**: Anchor policy (pre-trained TrotGo2) provides base gait; residual policy learns corrections
5. **No Domain Randomization**: DR is disabled for real-world fine-tuning (not available on real robot)

### Key Files

- `mujoco_playground/experimental/learning/real_world_fine_tuning.ipynb` — Main notebook for APG fine-tuning
- `mujoco_playground/config/locomotion_params.py` — `brax_apg_config()` supports `Go2AlignmentEnv`
- `mujoco_playground/_src/locomotion/go2/alignment_env.py` — `AlignmentEnv` and `Go2AlignmentEnv` registry

### Go2AlignmentEnv Design

`Go2AlignmentEnv` is a wrapper environment that:

- Uses `Go2Joystick2` (MJX) for differentiable backward pass (gradient computation)
- Uses `Go2JoystickMujoco` (native MuJoCo) for forward transition (value computation)
- Aligns states using `StateAlignmentManager` with configurable `alpha` parameter
- Supports physics overrides via `mj_twin_template` injected into native env config

**State Alignment Mechanism:**

The `align_tree` function implements the key insight:

```python
def align_tree(non_diff: Any, diff: Any, alpha: jax.Array) -> Any:
    return jax.tree_util.tree_map(
        lambda x_nd, x_df: x_nd + alpha * (x_df - jax.lax.stop_gradient(x_df)),
        non_diff,
        diff,
    )
```

- **Forward pass**: Output is `x_nd` (pure native MuJoCo state with mj_twin_template physics)
- **Backward pass**: Gradients flow as `alpha * gradient(x_df)` through the MJX path

This allows learning corrections that work with real-world physics while benefiting from differentiable simulation for gradient computation.

**Train vs Eval:**

- **Training**: Uses `Go2AlignmentEnv` (MJX + Native) for gradients through MJX
- **Evaluation**: Uses `Go2JoystickMujoco` (Native only) — no backprop needed, faster

```python
# Creating Go2AlignmentEnv with physics overrides
from mujoco_playground import locomotion
from mujoco_playground._src.locomotion.go2.JoystickGo2Mujoco import make_mujoco_model_template

mj_twin_template = make_mujoco_model_template()

# To keep environment unchanged: leave mj_twin_template empty (uses XML defaults)
# Default values from go2.xml:
#   body_mass["base"]: 6.921 kg
#   body_inertia["base"]: [0.107, 0.098, 0.024]
#   geom_friction["FL/FR/RL/RR"]: [0.8, 0.02, 0.01]
#   joint_damping["*_joint"]: 2.0
#   joint_armature["*_joint"]: 0.01
#   joint_frictionloss["*_joint"]: 0.2

# For sim2real mismatch, perturb these values:
# mj_twin_template["body_mass"]["base"] = 7.5  # +8% mass
# mj_twin_template["geom_friction"]["FL"] = [0.7, 0.02, 0.01]  # -12% friction

native_env_cfg = locomotion.get_default_config("Go2JoystickMujoco")
native_env_cfg.mujoco_model = mj_twin_template

# Training uses Go2AlignmentEnv (MJX + Native for gradients)
train_env = locomotion.load("Go2AlignmentEnv", config=None, config_overrides={
    "mjx_cfg": None,  # Use default MJX config
    "native_cfg": native_env_cfg,
})

# Evaluation uses pure Go2JoystickMujoco (Native only, no MJX overhead)
eval_env = locomotion.load("Go2JoystickMujoco", config=native_env_cfg)
```

### Current Status

- ✅ `AlignmentEnv` class implemented and tested (reset/step work correctly)
- ✅ `Go2AlignmentEnv` registered in `locomotion` registry
- ✅ `brax_apg_config("Go2AlignmentEnv")` supported (shares config with `Go2Joystick2`)
- ✅ `_default_env_factory` supports `config_overrides` for `mjx_cfg` and `native_cfg` injection
- ✅ `real_world_fine_tuning.ipynb` updated to use `Go2AlignmentEnv`
- ✅ Domain randomization removed (set to `None`) for real-world fine-tuning
- ✅ Eval environment uses `Go2JoystickMujoco` (native only, no MJX overhead)
- ⏳ Ready to run APG fine-tuning experiments

### How to Run

```python
# In real_world_fine_tuning.ipynb:
USE_WANDB = False  # Set True to enable W&B logging
apg_params = locomotion_params.brax_apg_config("Go2AlignmentEnv")
apg_params["num_envs"] = 1  # Single env for real-robot simulation
apg_params["num_eval_envs"] = 1  # Single eval env (native MuJoCo is CPU-only)

# Create mj_twin_template for physics overrides
mj_twin_template = make_mujoco_model_template()
# Optionally modify for sim2real mismatch (see above)

# Load environments
native_env_cfg = locomotion.get_default_config("Go2JoystickMujoco")
native_env_cfg.mujoco_model = mj_twin_template

train_env = locomotion.load("Go2AlignmentEnv", config=None, config_overrides={
    "mjx_cfg": None,
    "native_cfg": native_env_cfg,
})
eval_env = locomotion.load("Go2JoystickMujoco", config=native_env_cfg)  # Native only for eval

# Run training cells...
```

### Next Steps

1. Run full APG fine-tuning with `Go2AlignmentEnv` and validate training converges
2. Modify physics parameters in `mj_twin_template` to introduce sim2real mismatch:
   - Body mass perturbations (±10%)
   - Joint damping variations
   - Foot friction changes
3. Test robustness of fine-tuned policy under different physics perturbations
4. (Optional) Extend `AlignmentEnv` to support real robot oracle (DDS interface)
