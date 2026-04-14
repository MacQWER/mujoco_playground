from ml_collections import config_dict
from mujoco_playground._src.locomotion.go2.configs import config_blocks
from mujoco_playground._src.locomotion.go2 import go2_constants as consts

def default_config() -> config_dict.ConfigDict:
    """Main config builder for Trot task."""
    # base sim config
    cfg = config_blocks.get_sim_config()
    # cfg.Kp = 35.0
    # cfg.Kd = 0.5
    
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
        config_blocks.make_obs_term("base_angular_velocity", "gyro", consts.OBS_W_LOCAL_SCALE),
        config_blocks.make_obs_term("projected_gravity", "gravity", 1.0),
        config_blocks.make_obs_term("command", None, 1.0),  # 看真实指令
        config_blocks.make_obs_term("joint_positions", "joint_pos", 1.0),
        config_blocks.make_obs_term("joint_velocities", "joint_vel", consts.OBS_JOINT_VELS_SCALE),
        config_blocks.make_obs_term("last_action", None, 1.0),
        config_blocks.make_obs_term("kinematic_reference", None, 1.0),
        config_blocks.make_obs_term("anchor_action", None, 1.0),  # 看前置动作
    ]
    
    # 2. Anchor Policy 
    cfg.obs.anchor_terms = [
        config_blocks.make_obs_term("base_angular_velocity", "gyro", consts.OBS_W_LOCAL_SCALE),
        config_blocks.make_obs_term("projected_gravity", "gravity", 1.0),
        config_blocks.make_obs_term("zero_command", None, 1.0),  # 屏蔽指令
        config_blocks.make_obs_term("joint_positions", "joint_pos", 1.0),
        config_blocks.make_obs_term("joint_velocities", "joint_vel", consts.OBS_JOINT_VELS_SCALE),
        config_blocks.make_obs_term("last_action", None, 1.0),
        config_blocks.make_obs_term("kinematic_reference", None, 1.0),
        config_blocks.make_obs_term("zero_anchor_action", None, 1.0),  # 屏蔽动作
    ]
    
    # reward config
    cfg.rewards = config_blocks.get_base_rewards_config()

    # =================================================================
    # ORIGINAL REWARD CONFIG (TrotGo2 style - dense rewards, 19 terms)
    # Commented out, preserved for reference.
    # =================================================================
    cfg.rewards.terms.tracking_lin_vel = config_blocks.make_reward_term("tracking_lin_vel", 3.0)
    cfg.rewards.terms.tracking_ang_vel = config_blocks.make_reward_term("tracking_ang_vel", 2.0)
    cfg.rewards.terms.base_height_tracking = config_blocks.make_reward_term("base_height_tracking", 0.5)
    cfg.rewards.terms.joint_pose_tracking = config_blocks.make_reward_term("joint_pose_tracking", 0.1)
    cfg.rewards.terms.joint_vel_tracking = config_blocks.make_reward_term("joint_vel_tracking", 0.01)
    cfg.rewards.terms.gait_phase_tracking = config_blocks.make_reward_term("gait_phase_tracking", 1.0)
    cfg.rewards.terms.feet_traj = config_blocks.make_reward_term("feet_traj", -5.0)
    cfg.rewards.terms.lin_vel_z = config_blocks.make_reward_term("lin_vel_z", -1.0)
    cfg.rewards.terms.ang_vel_xy = config_blocks.make_reward_term("ang_vel_xy", -0.1)
    cfg.rewards.terms.orientation = config_blocks.make_reward_term("orientation", -10.0)
    cfg.rewards.terms.torques = config_blocks.make_reward_term("torques", -0.0002)
    cfg.rewards.terms.action_rate = config_blocks.make_reward_term("action_rate", -0.01)
    cfg.rewards.terms.energy = config_blocks.make_reward_term("energy", -0.001)
    cfg.rewards.terms.feet_slip = config_blocks.make_reward_term("feet_slip", -1.0)
    cfg.rewards.terms.feet_air_time = config_blocks.make_reward_term("feet_air_time", 5.0)
    cfg.rewards.terms.dof_pos_limits = config_blocks.make_reward_term("dof_pos_limits", -1.0)
    cfg.rewards.terms.stand_still = config_blocks.make_reward_term("stand_still", -0.5)
    cfg.rewards.terms.termination = config_blocks.make_reward_term("termination", -10.0)
    for name, term in cfg.rewards.terms.items():
        cfg.rewards.scales[name] = term.scale

    # =================================================================
    # SMOOTH REWARD CONFIG (sparse rewards, ~12 terms)
    # Inspired by smooth_joystick design.
    # =================================================================
    # cfg.rewards.terms = config_dict.ConfigDict()

    # # Tracking (core)
    # cfg.rewards.terms.tracking_lin_vel = config_blocks.make_reward_term("tracking_lin_vel", 1.5)
    # cfg.rewards.terms.tracking_ang_vel = config_blocks.make_reward_term("tracking_ang_vel", 1.0)

    # # Base stability
    # cfg.rewards.terms.lin_vel_z = config_blocks.make_reward_term("lin_vel_z", -0.5)
    # cfg.rewards.terms.ang_vel_xy = config_blocks.make_reward_term("ang_vel_xy", -0.05)
    # cfg.rewards.terms.orientation = config_blocks.make_reward_term("orientation", -5.0)

    # # Regularization
    # cfg.rewards.terms.torques = config_blocks.make_reward_term("torques", -0.0002)
    # cfg.rewards.terms.action_rate = config_blocks.make_reward_term("action_rate", -0.01)
    # cfg.rewards.terms.energy = config_blocks.make_reward_term("energy", -0.001)

    # # Feet
    # cfg.rewards.terms.feet_slip = config_blocks.make_reward_term("feet_slip", -0.1)
    # cfg.rewards.terms.feet_air_time = config_blocks.make_reward_term("feet_air_time", 0.1)

    # # Trot Gait
    # cfg.rewards.terms.contact_count_penalty = config_blocks.make_reward_term("contact_count_penalty", -0.05)
    # cfg.rewards.terms.diagonal_sync_penalty = config_blocks.make_reward_term("diagonal_sync_penalty", -0.1)

    # # Other
    # cfg.rewards.terms.pose = config_blocks.make_reward_term("pose", 0.1)
    # cfg.rewards.terms.termination = config_blocks.make_reward_term("termination", -1.0)
    # cfg.rewards.terms.stand_still = config_blocks.make_reward_term("stand_still", -1.0)
    # cfg.rewards.terms.dof_pos_limits = config_blocks.make_reward_term("dof_pos_limits", -1.0)

    # for name, term in cfg.rewards.terms.items():
    #     cfg.rewards.scales[name] = term.scale
    
    # Hyperparameters
    cfg.rewards.tracking_sigma = 0.25
    cfg.rewards.base_height_sigma = 0.01
    cfg.rewards.joint_pose_tracking_sigma = 0.5
    cfg.rewards.joint_vel_tracking_sigma = 2.0
    cfg.rewards.gait_phase_tracking_sigma = 0.25
    cfg.rewards.max_foot_height = 0.1

    # anchor config
    cfg.anchor = config_dict.ConfigDict()
    cfg.anchor.path = consts.ANCHOR_PATH

    # disturbance config
    cfg.disturbance = config_blocks.get_disturbance_config()
    cfg.disturbance.enable = True

    # assistive wrench config
    cfg.assistive_wrench = config_blocks.get_assistive_wrench_config()
    cfg.assistive_wrench.enable = True
    cfg.assistive_wrench.enable_feedforward = True
    cfg.assistive_wrench.ff_mass_mode = "subtree"
    cfg.assistive_wrench.force_limit = 200.0
    cfg.assistive_wrench.torque_limit = 50.0
    cfg.assistive_wrench.gains.xy_d = 100.0
    cfg.assistive_wrench.gains.z_p = 500.0
    cfg.assistive_wrench.gains.z_d = 50.0
    cfg.assistive_wrench.gains.roll_p = 50.0
    cfg.assistive_wrench.gains.roll_d = 5.0
    cfg.assistive_wrench.gains.pitch_p = 50.0
    cfg.assistive_wrench.gains.pitch_d = 5.0
    cfg.assistive_wrench.gains.yaw_d = 50.0
    cfg.assistive_wrench.beta.initial = 1.0
    cfg.assistive_wrench.beta.final = 0.0
    cfg.assistive_wrench.curriculum.mode = "staircase"
    cfg.assistive_wrench.curriculum.staircase_levels = 5
    cfg.assistive_wrench.curriculum.start_step = 0
    cfg.assistive_wrench.curriculum.end_step = int(256 * 64 * 0.5)  # 50% training progress


    # noise config
    cfg.noise_config = config_blocks.get_noise_config()
    cfg.noise_config.level = 1.0

    # command config
    cfg.command_config = config_blocks.get_base_command_config()
    cfg.command_config.a = [0.5, 0.2, 0.5]
    cfg.command_config.b = [1.0, 1.0, 1.0]
    
    return cfg
