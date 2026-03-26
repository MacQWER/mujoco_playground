from ml_collections import config_dict
from mujoco_playground._src.locomotion.go2.configs import config_blocks
from mujoco_playground._src.locomotion.go2 import go2_constants as consts

def default_config() -> config_dict.ConfigDict:
    """Main config builder for Trot task."""
    # base sim config
    cfg = config_blocks.get_sim_config()
    
    # env config
    cfg.env = config_blocks.get_env_config()
    cfg.env.anchor_action_scale = [0.3, 0.5, 0.5] * 4
    cfg.env.residual_action_scale = [0.5, 0.8, 0.8] * 4
    cfg.env.step_k = consts.STEP_K
    cfg.env.gait_scale = consts.GAIT_SCALE
    cfg.env.step_height = 0.1128
    cfg.env.step_height_min = 0.0
    cfg.env.foot_traj_vel_weight = 0.2

    # obs config
    cfg.obs = config_dict.ConfigDict()
    # 1. Residual Policy
    cfg.obs.policy_terms = [
        ("base_angular_velocity", "gyro",      consts.OBS_W_LOCAL_SCALE),
        ("projected_gravity",     "gravity",   1.0),
        ("command",               None,        1.0),  # 看真实指令
        ("joint_positions",       "joint_pos", 1.0),
        ("joint_velocities",      "joint_vel", consts.OBS_JOINT_VELS_SCALE),
        ("last_action",           None,        1.0),
        ("kinematic_reference",   None,        1.0),
        ("anchor_action",         None,        1.0),  # 看前置动作
    ]
    
    # 2. Anchor Policy 
    cfg.obs.anchor_terms = [
        ("base_angular_velocity", "gyro",      consts.OBS_W_LOCAL_SCALE),
        ("projected_gravity",     "gravity",   1.0),
        ("zero_command",          None,        1.0),  # 屏蔽指令
        ("joint_positions",       "joint_pos", 1.0),
        ("joint_velocities",      "joint_vel", consts.OBS_JOINT_VELS_SCALE),
        ("last_action",           None,        1.0),
        ("kinematic_reference",   None,        1.0),
        ("zero_anchor_action",    None,        1.0),  # 屏蔽动作
    ]
    
    # reward config
    cfg.rewards = config_blocks.get_base_rewards_config()
    # Tracking
    cfg.rewards.scales.tracking_lin_vel = 3.0
    cfg.rewards.scales.tracking_ang_vel = 2.0
    cfg.rewards.scales.base_height_tracking = 0.5
    cfg.rewards.scales.joint_pose_tracking = 0.1
    cfg.rewards.scales.joint_vel_tracking = 0.01
    cfg.rewards.scales.gait_phase_tracking = 1.0
    
    # Anchor Heuristics
    cfg.rewards.scales.feet_traj = -5.0
    
    # Smoothness & Physics
    cfg.rewards.scales.lin_vel_z = -1.0
    cfg.rewards.scales.ang_vel_xy = -0.1
    cfg.rewards.scales.orientation = -10.0
    cfg.rewards.scales.torques = -0.0002
    cfg.rewards.scales.action_rate = -0.01
    cfg.rewards.scales.energy = -0.001
    
    # Feet Interaction
    cfg.rewards.scales.feet_slip = -1.0
    cfg.rewards.scales.feet_air_time = 5.0
    cfg.rewards.scales.dof_pos_limits = -1.0
    cfg.rewards.scales.stand_still = -0.5
    cfg.rewards.scales.termination = -10.0  
    
    # Hyperparameters
    cfg.rewards.tracking_sigma = 0.25
    cfg.rewards.base_height_sigma = 0.01
    cfg.rewards.joint_pose_tracking_sigma = 0.5
    cfg.rewards.joint_vel_tracking_sigma = 2.0
    cfg.rewards.gait_phase_tracking_sigma = 0.25

    # anchor config
    cfg.anchor = config_dict.ConfigDict()
    cfg.anchor.path = consts.ANCHOR_PATH

    # disturbance config
    cfg.disturbance = config_blocks.get_disturbance_config()
    cfg.disturbance.enable = True

    # noise config
    cfg.noise_config = config_blocks.get_noise_config()
    cfg.noise_config.level = 1.0

    # command config
    cfg.command_config = config_blocks.get_base_command_config()
    cfg.command_config.a = [0.5, 0.2, 0.5]
    cfg.command_config.b = [1.0, 1.0, 1.0]
    
    return cfg