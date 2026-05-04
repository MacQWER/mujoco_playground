# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
# ==============================================================================
"""Default config for G1Joystick2."""

from ml_collections import config_dict
from mujoco_playground._src.locomotion.g1.configs import config_blocks
from mujoco_playground._src.locomotion.g1 import g1_constants as consts


def default_config() -> config_dict.ConfigDict:
    cfg = config_blocks.get_sim_config()

    cfg.env = config_blocks.get_env_config()
    cfg.env.action_scale = 0.5
    cfg.env.step_k = consts.STEP_K
    cfg.env.gait_scale = consts.GAIT_SCALE
    cfg.env.step_height = 0.12
    cfg.env.step_height_min = 0.0
    cfg.env.foot_traj_vel_weight = 0.2
    cfg.env.randomize_gait_phase_on_resume = False
    cfg.env.stationary_cmd_threshold = 0.01
    cfg.env.stationary_w_cmd_threshold = 0.05

    # Observation config (8 terms, 127-D).
    cfg.obs = config_dict.ConfigDict()
    cfg.obs.policy_terms = [
        config_blocks.make_obs_term("base_angular_velocity", "gyro", consts.OBS_W_LOCAL_SCALE),
        config_blocks.make_obs_term("projected_gravity", "gravity", 1.0),
        config_blocks.make_obs_term("command", None, 1.0),
        config_blocks.make_obs_term("joint_positions", "joint_pos", 1.0),
        config_blocks.make_obs_term("joint_velocities", "joint_vel", consts.OBS_JOINT_VELS_SCALE),
        config_blocks.make_obs_term("last_action", None, 1.0),
        config_blocks.make_obs_term("kinematic_reference", None, 1.0),
        config_blocks.make_obs_term("gait_phase", None, 1.0),
    ]

    # Reward config (24 terms, G1 original scales).
    cfg.rewards = config_blocks.get_base_rewards_config()
    cfg.rewards.terms.tracking_lin_vel = config_blocks.make_reward_term("tracking_lin_vel", 1.0)
    cfg.rewards.terms.tracking_ang_vel = config_blocks.make_reward_term("tracking_ang_vel", 0.75)
    cfg.rewards.terms.lin_vel_z = config_blocks.make_reward_term("lin_vel_z", 0.0)
    cfg.rewards.terms.ang_vel_xy = config_blocks.make_reward_term("ang_vel_xy", -0.15)
    cfg.rewards.terms.orientation = config_blocks.make_reward_term("orientation", -2.0)
    cfg.rewards.terms.base_height = config_blocks.make_reward_term("base_height", 0.0)
    cfg.rewards.terms.torques = config_blocks.make_reward_term("torques", 0.0)
    cfg.rewards.terms.action_rate = config_blocks.make_reward_term("action_rate", 0.0)
    cfg.rewards.terms.energy = config_blocks.make_reward_term("energy", 0.0)
    cfg.rewards.terms.dof_acc = config_blocks.make_reward_term("dof_acc", 0.0)
    cfg.rewards.terms.feet_clearance = config_blocks.make_reward_term("feet_clearance", 0.0)
    cfg.rewards.terms.feet_air_time = config_blocks.make_reward_term("feet_air_time", 2.0)
    cfg.rewards.terms.feet_slip = config_blocks.make_reward_term("feet_slip", -0.25)
    cfg.rewards.terms.feet_height = config_blocks.make_reward_term("feet_height", 0.0)
    cfg.rewards.terms.feet_phase = config_blocks.make_reward_term("feet_phase", 1.0)
    cfg.rewards.terms.feet_traj = config_blocks.make_reward_term("feet_traj", -5.0)
    cfg.rewards.terms.gait_phase_tracking = config_blocks.make_reward_term("gait_phase_tracking", 1.0)
    cfg.rewards.terms.alive = config_blocks.make_reward_term("alive", 0.0)
    cfg.rewards.terms.stand_still = config_blocks.make_reward_term("stand_still", -1.0)
    cfg.rewards.terms.termination = config_blocks.make_reward_term("termination", -100.0)
    cfg.rewards.terms.collision = config_blocks.make_reward_term("collision", -0.1)
    cfg.rewards.terms.contact_force = config_blocks.make_reward_term("contact_force", -0.01)
    cfg.rewards.terms.joint_deviation_knee = config_blocks.make_reward_term("joint_deviation_knee", -0.1)
    cfg.rewards.terms.joint_deviation_hip = config_blocks.make_reward_term("joint_deviation_hip", -0.25)
    cfg.rewards.terms.dof_pos_limits = config_blocks.make_reward_term("dof_pos_limits", -1.0)
    cfg.rewards.terms.pose = config_blocks.make_reward_term("pose", -0.1)

    for name, term in cfg.rewards.terms.items():
        cfg.rewards.scales[name] = term.scale

    cfg.rewards.tracking_sigma = 0.25
    cfg.rewards.max_foot_height = 0.15
    cfg.rewards.base_height_target = 0.5
    cfg.rewards.max_contact_force = 500.0
    cfg.rewards.gait_phase_tracking_sigma = 0.25

    # Disturbance config.
    cfg.disturbance = config_blocks.get_disturbance_config()
    cfg.disturbance.enable = True

    # Noise config.
    cfg.noise_config = config_blocks.get_noise_config()
    cfg.noise_config.level = 1.0

    # Command config.
    cfg.command_config = config_blocks.get_base_command_config()
    cfg.command_config.a = [1.0, 0.8, 1.0]
    cfg.command_config.b = [0.9, 0.25, 0.5]
    cfg.lin_vel_x = [-1.0, 1.0]
    cfg.lin_vel_y = [-0.5, 0.5]
    cfg.ang_vel_yaw = [-1.0, 1.0]

    return cfg
