# Learning RL Agents

In this directory, we demonstrate learning RL agents from MuJoCo Playground environments using [Brax](https://github.com/google/brax) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl). We provide two entrypoints from the command line: `python train_jax_ppo.py` and `python train_rsl_rl.py`.

For more detailed tutorials on using MuJoCo Playground for RL, see:

1. Intro. to the Playground with DM Control Suite [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb)
2. Locomotion Environments [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb)
3. Manipulation Environments [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/manipulation.ipynb)
4. Training CartPole from Vision [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_1.ipynb)
5. Robotic Manipulation from Vision [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_2.ipynb)

## Training with brax PPO

To train with brax PPO, you can use the `train_jax_ppo.py` script. This script uses the brax PPO algorithm to train an agent on a given environment.

```bash
python train_jax_ppo.py --env_name=CartpoleBalance
```

To train a vision-based policy using pixel observations:

```bash
python train_jax_ppo.py --env_name=CartpoleBalance --vision
```

Use `python train_jax_ppo.py --help` to see possible options and usage. Logs and checkpoints are saved in `logs` directory.

## Training with RSL-RL

To train with RSL-RL, you can use the `train_rsl_rl.py` script. This script uses the RSL-RL algorithm to train an agent on a given environment.

```bash
python train_rsl_rl.py --env_name=LeapCubeReorient
```

To render the behaviour from the resulting policy:

```bash
python learning/train_rsl_rl.py --env_name LeapCubeReorient --play_only --load_run_name <run_name>
```

where `run_name` is the name of the run you want to load (will be printed in the console when the training run is started).

Logs and checkpoints are saved in `logs` directory.

## Training with APG

To train with Advantage Policy Gradient (APG):

```bash
python train_jax_apg.py --env_name Go2Joystick2
```

Train and eval environments can be configured separately via JSON-style overrides:

default parameters:

```bash
CUDA_VISIBLE_DEVICES=3 python train_jax_apg.py --env_name Go2Joystick2 --train_env_cfg_overrides '{"env.solimp": [0.015, 1.0, 0.031], "env.solref": [0.02, 1.0]}' --eval_env_cfg_overrides '{"env.solimp": [0.9, 0.95, 0.001], "env.solref": [0.004, 1.0], "env.iterations": 100}' --use_wandb --suffix "apg"
```

hard parameters:

```bash
CUDA_VISIBLE_DEVICES=3 python learning/train_jax_apg.py --env_name Go2Joystick2 --train_env_cfg_overrides '{"env.solimp": [0.95, 0.99, 0.001], "env.solref": [0.004, 1.0], "env.iterations": 1}' --eval_env_cfg_overrides '{"env.solimp": [0.95, 0.99, 0.001], "env.solref": [0.004, 1.0], "env.iterations": 100}' --use_wandb --suffix "apg-train-hard-eval-hard"
```

soft parameters:

```bash
CUDA_VISIBLE_DEVICES=3 python learning/train_jax_apg.py --env_name Go2Joystick2 --train_env_cfg_overrides '{"env.solimp": [0.015, 0.5, 0.31], "env.solref": [0.02, 1.0], "env.iterations": 1}' --eval_env_cfg_overrides '{"env.solimp": [0.95, 0.99, 0.001], "env.solref": [0.004, 1.0], "env.iterations": 100}' --use_wandb --suffix "apg-train-soft-eval-hard"
```

no symloss:

```bash
CUDA_VISIBLE_DEVICES=0 python learning/train_jax_apg.py --env_name Go2Joystick2 --nosym_loss --use_wandb --suffix 'no-symloss'
```

no heuristic

```bash
 CUDA_VISIBLE_DEVICES=1 python learning/train_jax_apg.py --env_name Go2Joystick2 --train_env_cfg_overrides '{"rewards.terms.feet_traj.scale": 0.0}' --eval_env_cfg_overrides '{"rewards.terms.feet_traj.scale": 0.0}' --use_wandb --suffix 'no-heuristic'
```

play and plot Gait Contact Diagram

```bash
CUDA_VISIBLE_DEVICES=0 python learning/train_jax_apg.py --env_name Go2Joystick2 --play_only --load_checkpoint_path logs/Go2Joystick2-20260429-162255-no-symloss/checkpoints/params.pkl --gait_diagram --suffix 'no-symloss-play'
```

APG for Pushbox

```bash
CUDA_VISIBLE_DEVICES=4 WANDB_PROJECT=test python learning/train_jax_apg.py --env_name=PushBox --use_wandb --train_env_cfg_overrides='{"solimp": [0.01
5, 0.99, 0.031], "solref": [0.02, 1.0]}' --eval_env_cfg_overrides='{"solimp": [0.95, 0.99, 0.001], "solref": [0.004, 1.0]}' --suffix 'apg-test7' --seed 0
```
