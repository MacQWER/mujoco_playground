from ml_collections import config_dict
from mujoco_playground._src.locomotion.go2 import go2_constants as consts

# ==========================================
# 1. Base Shared Blocks (通用基础组件)
# ==========================================

def get_sim_config() -> config_dict.ConfigDict:
    """Base physics and simulator parameters."""
    cfg = config_dict.ConfigDict()
    cfg.Kp = 35.0
    cfg.Kd = 0.5
    cfg.sim_dt = 0.002
    cfg.ctrl_dt = 0.02
    cfg.episode_length = 240
    cfg.soft_joint_pos_limit_factor = 0.95
    cfg.impl = "jax"
    cfg.nconmax = 4 * 8192
    cfg.njmax = 40
    return cfg

def get_env_config() -> config_dict.ConfigDict:
    """Base environment parameters."""
    env = config_dict.ConfigDict()
    env.impratio = 100
    env.iterations = 1
    return env

def get_disturbance_config() -> config_dict.ConfigDict:
    """External disturbance settings."""
    dist = config_dict.ConfigDict()
    dist.enable = True
    dist.velocity_kick = [0.0, 1.0]
    dist.kick_durations = [0.05, 0.2]
    dist.kick_wait_times = [1.0, 3.0]
    return dist

def get_noise_config() -> config_dict.ConfigDict:
    """Observation noise settings."""
    noise = config_dict.ConfigDict()
    noise.level = 1.0
    noise.scales = config_dict.ConfigDict()
    noise.scales.joint_pos = 0.03
    noise.scales.joint_vel = 1.5
    noise.scales.gyro = 0.2
    noise.scales.gravity = 0.05
    return noise

def get_base_command_config() -> config_dict.ConfigDict:
    """Base command config."""
    cmd = config_dict.ConfigDict()
    return cmd

def get_base_rewards_config() -> config_dict.ConfigDict:
    """Base reward setting."""
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
