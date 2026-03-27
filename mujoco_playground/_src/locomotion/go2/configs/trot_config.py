from ml_collections import config_dict
from mujoco_playground._src.locomotion.go2.configs import config_blocks
from mujoco_playground._src.locomotion.go2 import go2_constants as consts

def default_config() -> config_dict.ConfigDict:
    """Main config builder for Trot task."""
    # base sim config
    cfg = config_blocks.get_sim_config()
    
    # env config
    cfg.env = config_blocks.get_env_config()
    cfg.env.termination_height = 0.1
    cfg.env.step_k = consts.STEP_K
    cfg.env.gait_scale = consts.GAIT_SCALE
    cfg.env.err_threshold = 0.1
    cfg.env.action_scale = [0.3, 0.5, 0.5] * 4
    cfg.env.reset2ref = False
    cfg.env.reference_state_init = True

    # obs config
    cfg.obs = config_dict.ConfigDict()
    
    # 格式: (使用的函数名, 噪声配置名, 缩放系数)
    cfg.obs.policy_terms = [
        config_blocks.make_obs_term("base_angular_velocity", "gyro", consts.OBS_W_LOCAL_SCALE),
        config_blocks.make_obs_term("projected_gravity", "gravity", 1.0),
        config_blocks.make_obs_term("zero_command", None, 1.0),  # Trot 无指令
        config_blocks.make_obs_term("joint_positions", "joint_pos", 1.0),
        config_blocks.make_obs_term("joint_velocities", "joint_vel", consts.OBS_JOINT_VELS_SCALE),
        config_blocks.make_obs_term("last_action", None, 1.0),
        config_blocks.make_obs_term("kinematic_reference", None, 1.0),
        config_blocks.make_obs_term("zero_anchor_action", None, 1.0),  # Trot 无 Anchor
    ]
    
    # reward config
    cfg.rewards = config_blocks.get_base_rewards_config()
    cfg.rewards.terms.min_reference_tracking = config_blocks.make_reward_term(
        "min_reference_tracking", -2.5 * 3e-3
    )
    cfg.rewards.terms.reference_tracking = config_blocks.make_reward_term(
        "reference_tracking", -10.0
    )
    cfg.rewards.terms.feet_height = config_blocks.make_reward_term(
        "feet_height", -10.0
    )
    cfg.rewards.terms.base_tracking = config_blocks.make_reward_term(
        "base_tracking", -1.0
    )
    for name, term in cfg.rewards.terms.items():
        cfg.rewards.scales[name] = term.scale

    # disturbance config
    cfg.disturbance = config_blocks.get_disturbance_config()
    cfg.disturbance.enable = True

    # noise config
    cfg.noise_config = config_blocks.get_noise_config()
    cfg.noise_config.level = 1.0
    
    return cfg
