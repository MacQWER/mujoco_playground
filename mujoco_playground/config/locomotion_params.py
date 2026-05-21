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
"""RL config for Locomotion envs."""

from typing import Optional
from ml_collections import config_dict
from mujoco_playground._src import locomotion

def brax_apg_config(
    env_name: str, unused_impl: Optional[str] = None
) -> config_dict.ConfigDict:
  """Returns tuned Brax APG config for the given environment."""
  rl_config = config_dict.create(
      num_evals=10,
      episode_length=240,
      policy_updates=499,
      horizon_length=32,
      num_eval_envs=64,
      use_float64=True,
      normalize_observations=True,
      action_repeat=1,
      learning_rate=3e-4,
      # entropy_cost=1e-2, # TODO add it later
      num_envs=8192,
      max_gradient_norm=1e9,
      network_factory=config_dict.create(
          policy_hidden_layer_sizes=(128, 128, 128, 128),
          value_hidden_layer_sizes=(256, 256, 256, 256, 256),
          policy_obs_key="state",
          value_obs_key="state",
      ),
  )

  if env_name in ("AnymalTrot",):
    rl_config.episode_length=240
    rl_config.policy_updates=499
    rl_config.horizon_length=32
    rl_config.num_envs=64
    rl_config.learning_rate=1e-4
    rl_config.num_eval_envs=64
    rl_config.num_evals=10 + 1
    rl_config.use_float64=True
    rl_config.normalize_observations=True
    rl_config.network_factory = config_dict.create(
        hidden_layer_sizes=(256, 128),
    )

  elif env_name in ("Go2Trot",):
    rl_config.episode_length=240
    rl_config.policy_updates=499
    rl_config.horizon_length=32
    rl_config.num_envs=64
    rl_config.learning_rate=1e-4
    rl_config.num_eval_envs=64
    rl_config.num_evals=10 + 1
    rl_config.use_float64=True
    rl_config.normalize_observations=True
    rl_config.network_factory = config_dict.create(
        hidden_layer_sizes=(256, 128),
        policy_obs_key="state",
    )
  
  elif env_name in ("Go2SmoothJoystickAPG",):
    rl_config.episode_length=240
    rl_config.policy_updates=499
    rl_config.horizon_length=32
    rl_config.num_envs=1024
    rl_config.learning_rate=1e-4
    rl_config.num_eval_envs=64
    rl_config.num_evals=10 + 1
    rl_config.use_float64=True
    rl_config.normalize_observations=True
    rl_config.network_factory = config_dict.create(
        hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
    )
  elif env_name in ("Go2Joystick2",):
    rl_config.episode_length=240
    rl_config.policy_updates=256
    rl_config.horizon_length=64
    rl_config.num_envs=256
    rl_config.deterministic_eval=True
    rl_config.learning_rate=1e-4
    rl_config.num_eval_envs=64
    rl_config.num_evals=64 + 1
    rl_config.max_gradient_norm=1.0
    rl_config.reward_scale = 10.0
    rl_config.use_float64=True
    rl_config.use_mixed_precision=True
    rl_config.unrollout_length=4
    rl_config.normalize_observations=True
    rl_config.network_factory = config_dict.create(
        hidden_layer_sizes=(256, 128),
        policy_obs_key="state",
    )
    # Symmetry loss for JoystickGo2 residual policy.
    # Obs layout (71): w(3), g(3), cmd(3), qpos(12), qvel(12),
    # last_action(12), kin_ref(12), anchor_action(12), gait_phase(2)
    # Action layout (12): [FL, FR, RL, RR] x [hip, thigh, calf]
    # Signed permutation encoding:
    #   new[i] = sign(perm[i]) * old[floor(abs(perm[i]) + 1e-3)]
    # Use -0.0001 to represent "- index 0".
    rl_config.sym_loss = True
    rl_config.sym_coef = 2.0
    rl_config.sym_obs_key = "state"
    rl_config.obs_permutation = (
        # 0-8: base states and command
      -0.0001,    1.0,  -2.0,     # [0-2]   w_local (-wx, wy, -wz)
        3.0,     -4.0,   5.0,     # [3-5]   g_local (gx, -gy, gz)
        6.0,     -7.0,  -8.0,     # [6-8]   command (vx, -vy, -wz)

        # 9-20: angles (qpos - default_ap_pose)
      -12.0,     13.0,  14.0,     # [9-11]  angles FL <- FR
       -9.0,     10.0,  11.0,     # [12-14] angles FR <- FL
      -18.0,     19.0,  20.0,     # [15-17] angles RL <- RR
      -15.0,     16.0,  17.0,     # [18-20] angles RR <- RL

        # 21-32: joint_vels (qvel)
      -24.0,     25.0,  26.0,     # [21-23] joint_vels FL <- FR
      -21.0,     22.0,  23.0,     # [24-26] joint_vels FR <- FL
      -30.0,     31.0,  32.0,     # [27-29] joint_vels RL <- RR
      -27.0,     28.0,  29.0,     # [30-32] joint_vels RR <- RL

        # 33-44: last_action
      -36.0,     37.0,  38.0,     # [33-35] last_action FL <- FR
      -33.0,     34.0,  35.0,     # [36-38] last_action FR <- FL
      -42.0,     43.0,  44.0,     # [39-41] last_action RL <- RR
      -39.0,     40.0,  41.0,     # [42-44] last_action RR <- RL

        # 45-56: kin_ref
      -48.0,     49.0,  50.0,     # [45-47] kin_ref FL <- FR
      -45.0,     46.0,  47.0,     # [48-50] kin_ref FR <- FL
      -54.0,     55.0,  56.0,     # [51-53] kin_ref RL <- RR
      -51.0,     52.0,  53.0,     # [54-56] kin_ref RR <- RL

        # 57-68: anchor_action
      -60.0,     61.0,  62.0,     # [57-59] anchor_action FL <- FR
      -57.0,     58.0,  59.0,     # [60-62] anchor_action FR <- FL
      -66.0,     67.0,  68.0,     # [63-65] anchor_action RL <- RR
      -63.0,     64.0,  65.0,     # [66-68] anchor_action RR <- RL

        # 69-70: gait_phase [sin(θ), cos(θ)]
        # Mirror symmetry: θ -> -θ, so sin(-θ)=-sin(θ), cos(-θ)=cos(θ)
      -69.0,     70.0,            # [69-70] gait_phase (-sin, cos)
    )
    rl_config.act_permutation = (
      -3.0, 4.0, 5.0,         # FL <- FR
      -0.0001, 1.0, 2.0,      # FR <- FL  (negative sign on source index 0)
      -9.0, 10.0, 11.0,       # RL <- RR
      -6.0, 7.0, 8.0,         # RR <- RL
    )
  else:
    raise ValueError(f"Unsupported env: {env_name}")

  return rl_config


def brax_ppo_config(
    env_name: str, impl: Optional[str] = None
) -> config_dict.ConfigDict:
  """Returns tuned Brax PPO config for the given environment."""
  env_config = locomotion.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=100_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=20,
      num_minibatches=32,
      num_updates_per_batch=4,
      discounting=0.97,
      learning_rate=3e-4,
      entropy_cost=1e-2,
      num_envs=8192,
      batch_size=256,
      max_grad_norm=1.0,
      network_factory=config_dict.create(
          policy_hidden_layer_sizes=(128, 128, 128, 128),
          value_hidden_layer_sizes=(256, 256, 256, 256, 256),
          policy_obs_key="state",
          value_obs_key="state",
      ),
      num_resets_per_eval=10,
  )

  if env_name in ("Go1JoystickFlatTerrain", "Go1JoystickRoughTerrain"):
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 10
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name in ("Go2Joystick", "Go2SmoothJoystickPPO"):
    rl_config.num_timesteps = 100_000_000
    rl_config.num_evals = 10
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name in ("Go1Handstand", "Go1Footstand"):
    rl_config.num_timesteps = 100_000_000
    rl_config.num_evals = 5
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name == "Go1Backflip":
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 10
    rl_config.discounting = 0.95
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name == "Go1Getup":
    rl_config.num_timesteps = 50_000_000
    rl_config.num_evals = 5
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name in ("G1JoystickFlatTerrain", "G1JoystickRoughTerrain"):
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 20
    rl_config.clipping_epsilon = 0.2
    rl_config.num_resets_per_eval = 1
    rl_config.entropy_cost = 0.005
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name in (
      "BerkeleyHumanoidJoystickFlatTerrain",
      "BerkeleyHumanoidJoystickRoughTerrain",
  ):
    rl_config.num_timesteps = 150_000_000
    rl_config.num_evals = 15
    rl_config.clipping_epsilon = 0.2
    rl_config.entropy_cost = 0.005
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name in (
      "T1JoystickFlatTerrain",
      "T1JoystickRoughTerrain",
  ):
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 20
    rl_config.clipping_epsilon = 0.2
    rl_config.num_resets_per_eval = 1
    rl_config.entropy_cost = 0.005
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name in ("ApolloJoystickFlatTerrain",):
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 20
    rl_config.clipping_epsilon = 0.2
    rl_config.num_resets_per_eval = 1
    rl_config.entropy_cost = 0.005
    rl_config.network_factory = config_dict.create(
      policy_hidden_layer_sizes=(512, 256, 128),
      value_hidden_layer_sizes=(512, 256, 128),
      policy_obs_key="state",
      value_obs_key="privileged_state",
    )
  
  elif env_name in ("AnymalTrot",):
    rl_config.num_timesteps = 10_000_000
    rl_config.num_evals = 10
    rl_config.reward_scaling=0.1
    rl_config.episode_length=240
    rl_config.normalize_observations=True
    rl_config.action_repeat=1
    rl_config.unroll_length=32
    rl_config.num_minibatches=32
    rl_config.num_updates_per_batch=8
    rl_config.discounting=0.97
    rl_config.learning_rate=3e-4
    rl_config.entropy_cost = 1e-3
    rl_config.num_envs=1024
    rl_config.batch_size=1024
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="state",
    )

  elif env_name in ("Go2Trot",):
    rl_config.num_timesteps = 10_000_000
    rl_config.num_evals = 10
    rl_config.reward_scaling=0.1
    rl_config.episode_length=240
    rl_config.normalize_observations=True
    rl_config.action_repeat=1
    rl_config.unroll_length=32
    rl_config.num_minibatches=32
    rl_config.num_updates_per_batch=8
    rl_config.discounting=0.97
    rl_config.learning_rate=3e-4
    rl_config.entropy_cost = 1e-3
    rl_config.num_envs=1024
    rl_config.batch_size=1024
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="state",
    )

  elif env_name in ("Go2Joystick2",):
    # Match APG's training schedule for fair comparison
    # APG: policy_updates=256, horizon_length=64, num_envs=256
    # Total steps = 256 * 64 * 256 = 4,194,304
    # Eval every 65,536 steps (4194304 / 64 intervals)

    # rl_config.num_timesteps = 4_194_304 * 4
    rl_config.num_eval_envs = 64  # Match APG's num_eval_envs
    rl_config.num_resets_per_eval = 0  # Disable extra resets between evals
    rl_config.reward_scaling = 10.0
    rl_config.episode_length = 240
    rl_config.normalize_observations = True  # Match APG
    rl_config.deterministic_eval = True  # Match APG
    rl_config.action_repeat = 1
    rl_config.unroll_length = 64  # Match APG's horizon_length
    rl_config.entropy_cost = 1e-3  # Match APG

    # rl_config.num_evals = 65  # 64 intervals + 1 initial eval for Matching APG
    # rl_config.learning_rate = 1e-4  # Match APG
    # rl_config.num_minibatches = 8
    # rl_config.num_updates_per_batch = 8
    # rl_config.num_envs = 256  # Match APG
    # rl_config.batch_size = 32  # 32 × 8 = 256 = num_envs, for correct Brax step calculation

    rl_config.num_timesteps = 20_000_000
    rl_config.num_evals = 10  
    rl_config.learning_rate = 1e-4  
    rl_config.num_minibatches = 16
    rl_config.num_updates_per_batch = 4
    rl_config.num_envs = 1024  
    rl_config.batch_size = 64  # 64 × 16 = 1024 = num_envs, for correct Brax step  

    # rl_config.num_timesteps = 20_000_000
    # rl_config.num_evals = 20
    # rl_config.deterministic_eval = True
    # rl_config.num_resets_per_eval = 0

    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(256, 128),  # Match APG
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="state",
    )


  elif env_name in (
      "BarkourJoystick",
      "H1InplaceGaitTracking",
      "H1JoystickGaitTracking",
      "Op3Joystick",
      "SpotFlatTerrainJoystick",
      "SpotGetup",
      "SpotJoystickGaitTracking",
  ):
    pass  # use default config
  else:
    raise ValueError(f"Unsupported env: {env_name}")

  return rl_config


def rsl_rl_config(
    env_name: str, unused_impl: Optional[str] = None
) -> config_dict.ConfigDict:
  """Returns tuned RSL-RL PPO config for the given environment."""

  rl_config = config_dict.create(
      seed=1,
      runner_class_name="OnPolicyRunner",
      policy=config_dict.create(
          init_noise_std=1.0,
          actor_hidden_dims=[512, 256, 128],
          critic_hidden_dims=[512, 256, 128],
          # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
          activation="elu",
          class_name="ActorCritic",
      ),
      algorithm=config_dict.create(
          class_name="PPO",
          value_loss_coef=1.0,
          use_clipped_value_loss=True,
          clip_param=0.2,
          entropy_coef=0.001,
          num_learning_epochs=5,
          # mini batch size = num_envs*nsteps / nminibatches
          num_mini_batches=4,
          learning_rate=3.0e-4,  # 5.e-4
          schedule="fixed",  # could be adaptive, fixed
          gamma=0.99,
          lam=0.95,
          desired_kl=0.01,
          max_grad_norm=1.0,
      ),
      num_steps_per_env=24,  # per iteration
      max_iterations=100000,  # number of policy updates
      empirical_normalization=True,
      # logging
      save_interval=50,  # check for potential saves every this many iterations
      experiment_name="test",
      run_name="",
      # load and resume
      resume=False,
      load_run="-1",  # -1 = last run
      checkpoint=-1,  # -1 = last saved model
      resume_path=None,  # updated from load_run and chkpt
  )

  if env_name in (
      "Go1Getup",
      "BerkeleyHumanoidJoystickFlatTerrain",
      "G1Joystick",
      "Go1JoystickFlatTerrain",
  ):
    rl_config.max_iterations = 1000
  if env_name == "Go1JoystickFlatTerrain":
    rl_config.algorithm.learning_rate = 3e-4
    rl_config.algorithm.schedule = "fixed"

  return rl_config
