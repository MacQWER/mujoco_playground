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
# ==============================================================================
"""Train an APG agent using JAX on the specified environment."""

import datetime
import functools
import json
import os
import pickle
import time
import warnings

from absl import app
from absl import flags
from absl import logging
from etils import epath
from flax.training import orbax_utils
from orbax import checkpoint as ocp
import jax
from jax import config
import jax.numpy as jp

# JAX configuration
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")

# Compilation cache configuration
cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "jit_cache")
os.makedirs(cache_path, exist_ok=True)

config.update("jax_compilation_cache_dir", cache_path)
config.update("jax_persistent_cache_min_entry_size_bytes", -1)
config.update("jax_persistent_cache_min_compile_time_secs", 1)

# Set Mujoco rendering backend (must be set before importing mujoco)
# Use environment variable if already set, otherwise default to egl
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
import tensorboardX
import wandb

# Import APG from apg_alg
from apg_alg.algorithm import apg
from apg_alg.networks import apg_networks


xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "Go2Joystick2",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 240, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_HORIZON_LENGTH = flags.DEFINE_integer("horizon_length", 64, "Horizon length")
_POLICY_UPDATES = flags.DEFINE_integer("policy_updates", 256, "Number of policy updates")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-4, "Learning rate")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 256, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 64, "Number of evaluation environments"
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 65, "Number of evaluations")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 1e-4, "Entropy cost")
_ENTROPY_COST_DECAY = flags.DEFINE_float("entropy_cost_decay", 0.999, "Entropy cost decay")
_MIN_ENTROPY_COST = flags.DEFINE_float("min_entropy_cost", 1e-8, "Minimum entropy cost")
_REWARD_SCALE = flags.DEFINE_float("reward_scale", 10.0, "Reward scale")
_USE_FLOAT64 = flags.DEFINE_boolean("use_float64", True, "Use float64 precision")
_USE_MIXED_PRECISION = flags.DEFINE_boolean("use_mixed_precision", True, "Use mixed precision")
_UNROLLOUT_LENGTH = flags.DEFINE_integer("unrollout_length", 4, "Unrollout length")
_DETERMINISTIC_EVAL = flags.DEFINE_boolean("deterministic_eval", True, "Deterministic eval")
_SYM_LOSS = flags.DEFINE_boolean("sym_loss", False, "Use symmetry loss")
_SYM_COEF = flags.DEFINE_float("sym_coef", 2.0, "Symmetry loss coefficient")
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [256, 128],
    "Policy hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_OBS_PERMUTATION = flags.DEFINE_list(
    "obs_permutation", None, "Observation permutation for symmetry loss"
)
_ACT_PERMUTATION = flags.DEFINE_list(
    "act_permutation", None, "Action permutation for symmetry loss"
)
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "Number of parallel environment rollouts to save for the rscope viewer",
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "Run deterministic rollouts for the rscope viewer",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    False,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
)
_SAVE_CHECKPOINTS = flags.DEFINE_boolean(
    "save_checkpoints",
    True,
    "Save checkpoints during training",
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of videos to record after training."
)
_TRAIN_ENV_CFG_OVERRIDES = flags.DEFINE_string(
    "train_env_cfg_overrides", None,
    "JSON string to override train env config (e.g., '{\"noise_config.level\": 0.0}')"
)
_EVAL_ENV_CFG_OVERRIDES = flags.DEFINE_string(
    "eval_env_cfg_overrides", None,
    "JSON string to override eval env config (e.g., '{\"assistive_wrench.enable\": false}')"
)


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
    if env_name in mujoco_playground.manipulation._envs:
        raise ValueError("APG is not supported for manipulation environments")
    elif env_name in mujoco_playground.locomotion._envs:
        return locomotion_params.brax_apg_config(env_name)
    elif env_name in mujoco_playground.dm_control_suite._envs:
        return dm_control_suite_params.brax_apg_config(env_name)

    raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def rscope_fn(full_states, obs, rew, done):
    """
    All arrays are of shape (unroll_length, rscope_envs, ...)
    full_states: dict with keys 'qpos', 'qvel', 'time', 'metrics'
    obs: nd.array or dict obs based on env configuration
    rew: nd.array rewards
    done: nd.array done flags
    """
    del full_states, obs  # Unused.
    # Calculate cumulative rewards per episode, stopping at first done flag
    done_mask = jp.cumsum(done, axis=0)
    valid_rewards = rew * (done_mask == 0)
    episode_rewards = jp.sum(valid_rewards, axis=0)
    print(
        "Collected rscope rollouts with reward"
        f" {episode_rewards.mean():.3f} +- {episode_rewards.std():.3f}"
    )


def main(argv):
    """Run training and evaluation for the specified environment."""

    del argv

    # Load environment configuration
    env_cfg = registry.get_default_config(_ENV_NAME.value)
    env_cfg["impl"] = _IMPL.value

    apg_params = get_rl_config(_ENV_NAME.value)

    # Override with command line flags
    if _PLAY_ONLY.present:
        apg_params.policy_updates = 0
        apg_params.num_evals = 1  # Skip training loop, only run initial eval
    if _EPISODE_LENGTH.present:
        apg_params.episode_length = _EPISODE_LENGTH.value
    if _NORMALIZE_OBSERVATIONS.present:
        apg_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
    if _ACTION_REPEAT.present:
        apg_params.action_repeat = _ACTION_REPEAT.value
    if _HORIZON_LENGTH.present:
        apg_params.horizon_length = _HORIZON_LENGTH.value
    if _POLICY_UPDATES.present:
        apg_params.policy_updates = _POLICY_UPDATES.value
    if _LEARNING_RATE.present:
        apg_params.learning_rate = _LEARNING_RATE.value
    if _NUM_ENVS.present:
        apg_params.num_envs = _NUM_ENVS.value
    if _NUM_EVAL_ENVS.present:
        apg_params.num_eval_envs = _NUM_EVAL_ENVS.value
    if _NUM_EVALS.present:
        apg_params.num_evals = _NUM_EVALS.value
    if _MAX_GRAD_NORM.present:
        apg_params.max_gradient_norm = _MAX_GRAD_NORM.value
    if _ENTROPY_COST.present:
        apg_params.entropy_cost = _ENTROPY_COST.value
    if _ENTROPY_COST_DECAY.present:
        apg_params.entropy_cost_decay = _ENTROPY_COST_DECAY.value
    if _MIN_ENTROPY_COST.present:
        apg_params.min_entropy_cost = _MIN_ENTROPY_COST.value
    if _REWARD_SCALE.present:
        apg_params.reward_scale = _REWARD_SCALE.value
    if _USE_FLOAT64.present:
        apg_params.use_float64 = _USE_FLOAT64.value
    if _USE_MIXED_PRECISION.present:
        apg_params.use_mixed_precision = _USE_MIXED_PRECISION.value
    if _UNROLLOUT_LENGTH.present:
        apg_params.unrollout_length = _UNROLLOUT_LENGTH.value
    if _DETERMINISTIC_EVAL.present:
        apg_params.deterministic_eval = _DETERMINISTIC_EVAL.value
    if _SYM_LOSS.present:
        apg_params.sym_loss = _SYM_LOSS.value
    if _SYM_COEF.present:
        apg_params.sym_coef = _SYM_COEF.value
    if _POLICY_HIDDEN_LAYER_SIZES.present:
        apg_params.network_factory.hidden_layer_sizes = tuple(
            map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
        )
    if _POLICY_OBS_KEY.present:
        apg_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
    if _OBS_PERMUTATION.present:
        apg_params.obs_permutation = tuple(map(float, _OBS_PERMUTATION.value))
    if _ACT_PERMUTATION.present:
        apg_params.act_permutation = tuple(map(float, _ACT_PERMUTATION.value))
    if _LOG_TRAINING_METRICS.present:
        apg_params.log_training_metrics = _LOG_TRAINING_METRICS.value

    # Handle domain randomization
    training_params = dict(apg_params)
    if "network_factory" in training_params:
        del training_params["network_factory"]

    if _DOMAIN_RANDOMIZATION.value:
        training_params["randomization_fn"] = registry.get_domain_randomizer(
            _ENV_NAME.value
        )
    else:
        training_params["randomization_fn"] = None

    # Generate unique experiment name
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{_ENV_NAME.value}-{timestamp}"
    if _SUFFIX.value is not None:
        exp_name += f"-{_SUFFIX.value}"
    print(f"Experiment name: {exp_name}")

    # Set up logging directory
    logdir = epath.Path("logs").resolve() / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logs are being stored in: {logdir}")

    # Initialize Weights & Biases if required
    if _USE_WANDB.value and not _PLAY_ONLY.value:
        wandb.init(project="mjplayground-apg", name=exp_name)
        wandb.config.update(env_cfg.to_dict())
        wandb.config.update({"env_name": _ENV_NAME.value})

    # Initialize TensorBoard if required
    if _USE_TB.value and not _PLAY_ONLY.value:
        writer = tensorboardX.SummaryWriter(logdir)

    # Handle checkpoint loading
    if _LOAD_CHECKPOINT_PATH.value is not None:
        # Convert to absolute path
        ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
        if ckpt_path.is_dir():
            latest_ckpts = list(ckpt_path.glob("*"))
            latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
            latest_ckpts.sort(key=lambda x: int(x.name))
            latest_ckpt = latest_ckpts[-1]
            restore_checkpoint_path = latest_ckpt
            print(f"Restoring from: {restore_checkpoint_path}")
        else:
            restore_checkpoint_path = ckpt_path
            print(f"Restoring from checkpoint: {restore_checkpoint_path}")
    else:
        print("No checkpoint path provided, not restoring from checkpoint")
        restore_checkpoint_path = None

    # Set up checkpoint directory
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    # Save environment configuration
    with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
        json.dump(env_cfg.to_dict(), fp, indent=4)

    print(f"Environment Config:\n{env_cfg}")
    print(f"APG Training Parameters:\n{apg_params}")

    times = [time.monotonic()]

    # # Calculate scale factor to convert policy update count to environment steps
    # updates_per_epoch = round(apg_params.policy_updates / max(apg_params.num_evals - 1, 1))
    # scale_it = updates_per_epoch * apg_params.horizon_length * apg_params.num_envs

    # Progress function for logging
    def progress(num_steps, metrics):
        times.append(time.monotonic())

        # Convert to actual environment steps
        # env_steps = num_steps * scale_it
        env_steps = num_steps

        # Log to Weights & Biases
        if _USE_WANDB.value and not _PLAY_ONLY.value:
            wandb.log(metrics, step=env_steps)

        # Log to TensorBoard
        if _USE_TB.value and not _PLAY_ONLY.value:
            for key, value in metrics.items():
                writer.add_scalar(key, value, env_steps)
            writer.flush()

        if _NUM_EVALS.value > 1:
            print(f"{env_steps}: reward={metrics.get('eval/episode_reward', 'N/A'):.3f}")

    # Load environment with optional config overrides
    def apply_cfg_overrides(base_cfg, overrides_json):
        if overrides_json is None:
            return base_cfg
        import json as json_module
        overrides = json_module.loads(overrides_json)
        for key, value in overrides.items():
            keys = key.split(".")
            cfg = base_cfg
            for k in keys[:-1]:
                cfg = cfg[k]
            cfg[keys[-1]] = value
        return base_cfg

    # Load train environment
    train_env_cfg = apply_cfg_overrides(env_cfg, _TRAIN_ENV_CFG_OVERRIDES.value)
    env = registry.load(_ENV_NAME.value, config=train_env_cfg)

    # Load evaluation environment with optional overrides
    if _EVAL_ENV_CFG_OVERRIDES.value is not None:
        eval_env_cfg = apply_cfg_overrides(env_cfg, _EVAL_ENV_CFG_OVERRIDES.value)
    else:
        eval_env_cfg = env_cfg
    eval_env = registry.load(_ENV_NAME.value, config=eval_env_cfg)

    # Set up rscope if requested
    if _RSCOPE_ENVS.value:
        from rscope import brax as rscope_utils

        rscope_env = registry.load(_ENV_NAME.value, config=env_cfg)
        rscope_env = wrapper.wrap_for_brax_training(
            rscope_env,
            episode_length=apg_params.episode_length,
            action_repeat=apg_params.action_repeat,
            randomization_fn=training_params.get("randomization_fn"),
        )

        rscope_handle = rscope_utils.BraxRolloutSaver(
            rscope_env,
            apg_params,
            False,  # vision=False
            _RSCOPE_ENVS.value,
            _DETERMINISTIC_RSCOPE.value,
            jax.random.PRNGKey(_SEED.value),
            rscope_fn,
        )

        def policy_params_fn_rscope(current_step, make_policy, params):
            del current_step  # Unused.
            rscope_handle.set_make_policy(make_policy)
            rscope_handle.dump_rollout(params)

        policy_params_fn = policy_params_fn_rscope
    elif _SAVE_CHECKPOINTS.value:
        # Save checkpoints during training
        def policy_params_fn_checkpoint(current_step, make_policy, params):
            del make_policy  # Unused.
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(params)
            path = ckpt_path / f"{current_step}"
            orbax_checkpointer.save(path, params, force=True, save_args=save_args)
            # print(f"Saved checkpoint at step {current_step}")

        policy_params_fn = policy_params_fn_checkpoint
    else:
        policy_params_fn = None

    # Train the model or load from checkpoint
    if _PLAY_ONLY.value and restore_checkpoint_path is not None:
        # Skip training, load params directly from pickle file
        with open(restore_checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        normalizer_params = data['normalizer_params']
        policy_params = data['policy_params']
        params = (normalizer_params, policy_params)

        # Create inference function directly
        apg_network = apg_networks.make_apg_networks(
            env.observation_size,
            env.action_size,
            **apg_params.network_factory
        )
        make_inference_fn = apg_networks.make_inference_fn(apg_network)

        print(f"Loaded params from {restore_checkpoint_path}")
    else:
        # Train the model
        train_fn = functools.partial(
            apg.train,
            **training_params,
            network_factory=functools.partial(
                apg_networks.make_apg_networks,
                **apg_params.network_factory
            ),
            seed=_SEED.value,
            restore_checkpoint_path=restore_checkpoint_path,
            progress_fn=progress,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            policy_params_fn=policy_params_fn,
            eval_env=eval_env,
        )

        make_inference_fn, params, _ = train_fn(environment=env)

        print("Done training.")
        if len(times) > 1:
            print(f"Time to JIT compile: {times[1] - times[0]}")
            print(f"Time to train: {times[-1] - times[1]}")

        # Save final checkpoint
        if _SAVE_CHECKPOINTS.value and not _PLAY_ONLY.value:
            normalizer_params, policy_params = params
            with open(ckpt_path / "params.pkl", "wb") as f:
                data = {
                    "normalizer_params": normalizer_params,
                    "policy_params": policy_params,
                }
                pickle.dump(data, f)
            print(f"Saved final policy and normalizer params to {ckpt_path / 'params.pkl'}")

    print("Starting inference...")

    # Create inference function.
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    # Run evaluation rollouts.
    rng = jax.random.PRNGKey(_SEED.value)
    target_command = jp.array([0.5, 0.0, 0.0])

    rollout = []
    state = jit_reset(rng)
    for _ in range(apg_params.episode_length):
        # Set command
        state.info["command"] = target_command

        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)

    # Render and save the rollout.
    render_every = 2
    fps = 1.0 / eval_env.dt / render_every
    print(f"FPS for rendering: {fps}")
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    traj = rollout[::render_every]
    frames = eval_env.render(
        traj, height=480, width=640, scene_option=scene_option, camera="track"
    )
    media.write_video(f"rollout0.mp4", frames, fps=fps)
    print(f"Rollout video saved as 'rollout0.mp4'.")


if __name__ == "__main__":
    app.run(main)
