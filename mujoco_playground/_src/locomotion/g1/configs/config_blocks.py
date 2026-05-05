# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
# ==============================================================================
"""Shared config building blocks for G1 environments."""

from ml_collections import config_dict


def get_sim_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    # Match the per-actuator gains from g1_mjx_feetonly.xml.
    cfg.Kp = [
        75.0, 75.0, 75.0, 75.0, 20.0, 2.0,  # left leg
        75.0, 75.0, 75.0, 75.0, 20.0, 2.0,  # right leg
        75.0, 75.0, 75.0,  # waist
        75.0, 75.0, 75.0, 75.0, 2.0, 2.0, 2.0,  # left arm
        75.0, 75.0, 75.0, 75.0, 2.0, 2.0, 2.0,  # right arm
    ]
    cfg.Kd = [
        2.0, 2.0, 2.0, 2.0, 1.0, 0.2,  # left leg
        2.0, 2.0, 2.0, 2.0, 1.0, 0.2,  # right leg
        2.0, 2.0, 2.0,  # waist
        2.0, 2.0, 2.0, 2.0, 0.2, 0.2, 0.2,  # left arm
        2.0, 2.0, 2.0, 2.0, 0.2, 0.2, 0.2,  # right arm
    ]
    cfg.sim_dt = 0.002
    cfg.ctrl_dt = 0.02
    cfg.episode_length = 1000
    cfg.soft_joint_pos_limit_factor = 0.95
    cfg.restricted_joint_range = False
    cfg.impl = "jax"
    cfg.nconmax = 8 * 8192
    cfg.njmax = 29 * 2 + 8 * 4
    return cfg


def get_env_config() -> config_dict.ConfigDict:
    env = config_dict.ConfigDict()
    # APG default solver/contact settings for G1Joystick2 training.
    env.impratio = 100.0
    env.iterations = 1
    env.solimp = [0.015, 0.99, 0.031]
    env.solref = [0.02, 1.0]
    return env


def get_disturbance_config() -> config_dict.ConfigDict:
    dist = config_dict.ConfigDict()
    dist.enable = True
    dist.velocity_kick = [0.0, 1.0]
    dist.kick_durations = [0.05, 0.2]
    dist.kick_wait_times = [1.0, 3.0]
    return dist


def get_noise_config() -> config_dict.ConfigDict:
    noise = config_dict.ConfigDict()
    noise.level = 1.0
    noise.scales = config_dict.ConfigDict()
    noise.scales.joint_pos = 0.03
    noise.scales.joint_vel = 1.5
    noise.scales.gyro = 0.2
    noise.scales.gravity = 0.05
    noise.scales.linvel = 0.1
    return noise


def get_base_command_config() -> config_dict.ConfigDict:
    cmd = config_dict.ConfigDict()
    return cmd


def get_base_rewards_config() -> config_dict.ConfigDict:
    rewards = config_dict.ConfigDict()
    rewards.scales = config_dict.ConfigDict()
    rewards.terms = config_dict.ConfigDict()
    return rewards


def make_obs_term(
    func: str,
    noise: str | None = None,
    scale: float = 1.0,
    enabled: bool = True,
) -> dict:
    return {
        "func": func,
        "noise": noise,
        "scale": scale,
        "enabled": enabled,
    }


def make_reward_term(
    func: str,
    scale: float,
    enabled: bool = True,
) -> dict:
    return {
        "func": func,
        "scale": scale,
        "enabled": enabled,
    }
