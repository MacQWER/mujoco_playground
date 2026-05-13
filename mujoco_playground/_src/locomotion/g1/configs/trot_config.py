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
"""Default config for G1Trot."""

from ml_collections import config_dict

from mujoco_playground._src.locomotion.g1 import g1_constants as consts
from mujoco_playground._src.locomotion.g1.configs import config_blocks


def default_config() -> config_dict.ConfigDict:
    """Builds a simple in-place stepping config for G1."""
    cfg = config_blocks.get_sim_config()

    cfg.env = config_blocks.get_env_config()
    cfg.env.action_scale = 0.5
    cfg.env.step_k = consts.STEP_K
    cfg.env.gait_scale = 0.2
    cfg.env.step_height = 0.10
    cfg.env.step_height_min = 0.08
    cfg.env.foot_traj_vel_weight = 0.0
    cfg.env.reference_state_init = True

    # Keep policy obs exactly aligned with G1Joystick2:
    # linvel(3), gyro(3), gravity(3), command(3), qpos(29), qvel(29),
    # last_action(29), gait_phase(4) = 103-D.
    cfg.obs = config_dict.ConfigDict()
    cfg.obs.policy_terms = [
        config_blocks.make_obs_term("base_linear_velocity", "linvel", 1.0),
        config_blocks.make_obs_term("base_angular_velocity", "gyro", 1.0),
        config_blocks.make_obs_term("projected_gravity", "gravity", 1.0),
        config_blocks.make_obs_term("command", None, 1.0),
        config_blocks.make_obs_term("joint_positions", "joint_pos", 1.0),
        config_blocks.make_obs_term("joint_velocities", "joint_vel", 1.0),
        config_blocks.make_obs_term("last_action", None, 1.0),
        config_blocks.make_obs_term("gait_phase", None, 1.0),
    ]

    cfg.rewards = config_blocks.get_base_rewards_config()
    cfg.rewards.terms.tracking_lin_vel = config_blocks.make_reward_term(
        "tracking_lin_vel", 1.0
    )
    cfg.rewards.terms.tracking_ang_vel = config_blocks.make_reward_term(
        "tracking_ang_vel", 0.5
    )
    cfg.rewards.terms.base_height_tracking = config_blocks.make_reward_term(
        "base_height_tracking", 0.5
    )
    cfg.rewards.terms.joint_pose_tracking = config_blocks.make_reward_term(
        "joint_pose_tracking", 1.0
    )
    cfg.rewards.terms.orientation = config_blocks.make_reward_term(
        "orientation", -1.0
    )
    cfg.rewards.terms.feet_traj = config_blocks.make_reward_term(
        "feet_traj", -2.0
    )
    cfg.rewards.terms.gait_phase_tracking = config_blocks.make_reward_term(
        "gait_phase_tracking", 1.0
    )
    cfg.rewards.terms.action_rate = config_blocks.make_reward_term(
        "action_rate", -0.01
    )
    cfg.rewards.terms.termination = config_blocks.make_reward_term(
        "termination", -50.0
    )
    for name, term in cfg.rewards.terms.items():
        cfg.rewards.scales[name] = term.scale

    cfg.rewards.tracking_sigma = 0.25
    cfg.rewards.base_height_sigma = 0.01
    cfg.rewards.joint_pose_tracking_sigma = 0.5
    cfg.rewards.gait_phase_tracking_sigma = 0.25
    cfg.rewards.soft_collision_margin = 0.02
    cfg.rewards.soft_collision_temp = 0.01

    # Start with the easiest version: no push disturbance and no obs noise.
    cfg.disturbance = config_blocks.get_disturbance_config()
    cfg.disturbance.enable = False

    cfg.noise_config = config_blocks.get_noise_config()
    cfg.noise_config.level = 0.0

    # Commands stay zero. The observation slot is kept for policy compatibility.
    cfg.command_config = config_blocks.get_base_command_config()
    cfg.command_config.a = [0.0, 0.0, 0.0]
    cfg.command_config.b = [0.0, 0.0, 0.0]
    cfg.lin_vel_x = [0.0, 0.0]
    cfg.lin_vel_y = [0.0, 0.0]
    cfg.ang_vel_yaw = [0.0, 0.0]

    return cfg
