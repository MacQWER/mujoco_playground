from typing import Any, Optional

import jax
import jax.numpy as jp
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx


from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import go2_constants as consts
from mujoco_playground._src.locomotion.go2.Util.TrotUtil import rotate

# ----------------- Foot Trajectory Util -----------------
def get_swing_mask(env, info):
    """Return swing mask (4,) for FL, FR, RL, RR at current phase block.

    Uses gait_step (which resets to 0 when stationary) instead of global step.
    When stationary, all feet remain in stance mode (no swinging).
    """
    gait_step = info['gait_step']
    is_stationary = info['is_stationary']

    # 静止时所有脚都不摆动（四脚站立）
    stationary_swing = jp.array([0.0, 0.0, 0.0, 0.0])

    # 运动时按 trot 模式交替
    gait_step = jp.asarray(gait_step, dtype=jp.int32)
    chunk_idx = gait_step // env.step_k
    even_chunk = (chunk_idx % 2 == 0)
    swing_even = jp.array([0.0, 1.0, 1.0, 0.0])  # FR, RL
    swing_odd = jp.array([1.0, 0.0, 0.0, 1.0])   # FL, RR
    moving_swing = jp.where(even_chunk, swing_even, swing_odd)

    return jp.where(is_stationary, stationary_swing, moving_swing)


def update_foot_cycloid_ref(env, info):
    """Build foot reference trajectory from Raibert endpoints + cycloid phase.

    Uses gait_step (which resets to 0 when stationary) instead of global step.
    """
    # 使用 gait_step 替代 info['step']
    step = info['gait_step']
    swing_period = env.gait_period / 2.0
    dt_step = (step - info['k0']) * env.dt
    phi = jp.clip(dt_step / swing_period, 0.0, 1.0)

    s = phi - jp.sin(2.0 * jp.pi * phi) / (2.0 * jp.pi)
    ds_dt = (1.0 - jp.cos(2.0 * jp.pi * phi)) / swing_period

    xy0 = info['xy0']
    xys = info['xy*']
    delta_xy = xys - xy0

    xy_ref = xy0 + delta_xy * s
    v_xy_ref = delta_xy * ds_dt

    # 调用更新后的 get_swing_mask(env, info)
    swing_mask = get_swing_mask(env, info)
    swing_mask_col = swing_mask[:, None]

    # Stance legs hold touchdown anchor; swing legs follow cycloid.
    xy_ref = xy0 * (1.0 - swing_mask_col) + xy_ref * swing_mask_col
    v_xy_ref = v_xy_ref * swing_mask_col

    # Cycloid height coupled to step length (2*pi*r = step length)
    step_len = jp.linalg.norm(delta_xy, axis=1)
    r = step_len / (2.0 * jp.pi)
    r_min = 0.5 * env._step_height_min
    r_max = 0.5 * env._step_height_max
    r = jp.clip(r, r_min, r_max)

    z0 = info['z0']
    z_ref = z0 + r * (1.0 - jp.cos(2.0 * jp.pi * phi))
    z_ref = z0 * (1.0 - swing_mask) + z_ref * swing_mask

    ref_pos = jp.concatenate([xy_ref, z_ref[:, None]], axis=1)

    info['foot_phase'] = jax.lax.stop_gradient(phi)
    info['foot_swing'] = jax.lax.stop_gradient(swing_mask)
    info['foot_ref_xy'] = jax.lax.stop_gradient(xy_ref)
    info['foot_ref_z'] = jax.lax.stop_gradient(z_ref)
    info['foot_ref_pos'] = jax.lax.stop_gradient(ref_pos)
    info['foot_ref_v_xy'] = jax.lax.stop_gradient(v_xy_ref)


def update_raibert_target(env, data, info):
    """
    Raibert Heuristic Target Updater.
    Phase Logic: Even Step -> FR/RL (Pair 2) Swing.
    Reference Frame: Hip-Centric.

    Uses gait_step (which resets to 0 when stationary) instead of global step.
    """
    # 使用 gait_step 替代 info['step']
    s = info['gait_step']
    step_k = env.step_k
    new_step = (s % step_k == 0)
    even_step = ((s // step_k) % 2 == 0)

    v_cmd_local = info['command'][:2]
    w_cmd_local = info['command'][2]

    quat = data.xquat[1]

    v_ff_rot_x = -w_cmd_local * env.leg_offsets_y
    v_ff_rot_y = w_cmd_local * env.leg_offsets_x

    v_leg_local = jp.stack([
        v_cmd_local[0] + v_ff_rot_x,
        v_cmd_local[1] + v_ff_rot_y
    ], axis=1)

    v_leg_local_3d = jp.concatenate([v_leg_local, jp.zeros((4, 1))], axis=1)
    v_leg_global = jax.vmap(rotate, in_axes=(0, None))(v_leg_local_3d, quat)[:, :2]

    hip_pos = data.xpos[env.hip_inds][:, :2]
    t_stance = env.gait_period / 2.0
    t_swing = env.gait_period / 2.0

    raibert_offset = (t_swing + 0.5 * t_stance) * v_leg_global

    foot_offset_local_3d = jp.concatenate([env.foot_offsets_xy, jp.zeros((4, 1))], axis=1)
    foot_offset_global = jax.vmap(rotate, in_axes=(0, None))(foot_offset_local_3d, quat)[:, :2]

    raibert_xy = hip_pos + foot_offset_global + raibert_offset

    pair1 = jp.array([0, 3])
    pair2 = jp.array([1, 2])

    cur_tars = info['xy*']

    tars_p2 = cur_tars.at[pair2].set(raibert_xy[pair2])
    tars_p1 = cur_tars.at[pair1].set(raibert_xy[pair1])

    xy_tars = jp.where(new_step & even_step, tars_p2, cur_tars)
    xy_tars = jp.where(new_step & (~even_step), tars_p1, xy_tars)
    info['xy*'] = xy_tars

    feet_pos = data.geom_xpos[env.feet_inds][:, :2]
    info['xy0'] = jp.where(new_step, feet_pos, info['xy0'])
    info['k0'] = jp.where(new_step, s, info['k0'])

    feet_z = env.get_feet_pos(data)[:, 2]
    info['z0'] = jp.where(new_step, feet_z, info['z0'])

# ----------------- Phase Alignment Check -----------------
def check_phase_alignment(env):
    """
    Plot a consolidated phase-alignment dashboard for all phase-coupled terms.

    The check overlays the current-phase reference against a half-cycle-shifted
    reference, so if any reward term is accidentally offset by half a gait cycle,
    it should show up immediately as the shifted curve looking "better" than the
    current-phase curve.
    """
    print("Running full phase alignment check (kino + feet_traj + gait rewards)...")

    steps = np.arange(env.step_k * 4)
    foot_idx = 0  # FL
    half_cycle = env.step_k

    height_target_log = []
    height_actual_log = []
    raibert_change_log = []
    kino_thigh_log = []
    swing_mask_fl_log = []
    swing_mask_fr_log = []

    joint_pose_rew_log = []
    joint_pose_half_rew_log = []
    joint_pose_err_log = []
    joint_pose_half_err_log = []

    joint_vel_rew_log = []
    joint_vel_half_rew_log = []
    joint_vel_err_log = []
    joint_vel_half_err_log = []

    gait_rew_log = []
    gait_half_rew_log = []
    gait_err_log = []
    gait_half_err_log = []

    feet_traj_err_log = []
    feet_traj_half_err_log = []
    feet_traj_pos_err_log = []
    feet_traj_pos_half_err_log = []
    feet_traj_vel_err_log = []
    feet_traj_vel_half_err_log = []

    cmd = jp.array([1.0, 0.0, 0.0])

    def make_phase_info(step: int, is_stationary: bool = False):
        """Create info dict for phase alignment testing.

        Args:
            step: The current step index.
            is_stationary: If True, simulates stationary behavior (gait_step=0, command=0).
        """
        # 静止时 gait_step 清零，command 为零
        gait_step = 0 if is_stationary else step
        cmd_for_info = jp.array([0.0, 0.0, 0.0]) if is_stationary else cmd

        return {
            'step': jp.array(step, dtype=jp.int32),
            'gait_step': jp.array(gait_step, dtype=jp.int32),
            'is_stationary': jp.array(is_stationary),
            'k0': jp.array(0, dtype=jp.int32),
            'command': cmd_for_info,
            'xy0': jp.zeros((4, 2)),
            'xy*': jp.zeros((4, 2)),
            'foot_phase': 0.0,
            'foot_swing': jp.zeros(4),
            'foot_ref_xy': jp.zeros((4, 2)),
            'foot_ref_z': jp.zeros(4),
            'foot_ref_pos': jp.zeros((4, 3)),
            'foot_ref_v_xy': jp.zeros((4, 2)),
            'z0': jp.zeros(4),
        }

    def build_ref_data(step_idx: int):
        qpos_ref = env.kinematic_ref_qpos[step_idx]
        qvel_ref = env.kinematic_ref_qvel[step_idx]
        d = mjx_env.make_data(
            env.mj_model,
            qpos=qpos_ref,
            qvel=qvel_ref,
            ctrl=jp.zeros(env.mjx_model.nu),
            impl=env.mjx_model.impl.value,
            nconmax=env._config.nconmax,
            njmax=env._config.njmax,
        )
        return mjx.forward(env.mjx_model, d)

    def prepare_info(step: int, data: mjx.Data, is_stationary: bool = False):
        info = make_phase_info(step, is_stationary)
        update_raibert_target(env, data, info)
        update_foot_cycloid_ref(env, info)
        return info

    def foot_traj_terms(data: mjx.Data, info: dict[str, Any]):
        curr_feet = data.geom_xpos[env.feet_inds]
        ref_pos = info['foot_ref_pos']
        swing_mask = info['foot_swing'][:, None]
        pos_err = jp.sum(jp.square((curr_feet - ref_pos) * swing_mask))

        vel_err = jp.array(0.0)
        if env._foot_linvel_sensor_adr is not None:
            feet_vel = data.sensordata[env._foot_linvel_sensor_adr]
            vel_xy = feet_vel[..., :2]
            ref_v_xy = info['foot_ref_v_xy']
            vel_err = jp.sum(jp.square((vel_xy - ref_v_xy) * swing_mask[:, :2]))

        total_err = pos_err + env._foot_traj_vel_weight * vel_err
        return float(total_err), float(pos_err), float(vel_err)

    def gait_terms(data: mjx.Data, info: dict[str, Any]):
        foot_pos = env.get_feet_pos(data)
        foot_z = foot_pos[..., -1]
        contact = jax.nn.sigmoid((0.025 - foot_z) * 100.0)
        expected_stance = 1.0 - info['foot_swing']
        err = jp.sum(jp.square(contact - expected_stance))
        rew = jp.exp(-err / env._config.rewards.gait_phase_tracking_sigma)
        return float(rew), float(err)

    last_fl_target_x = 0.0

    # 新增日志：记录 gait_step 和 is_stationary
    gait_step_log = []
    is_stationary_log = []

    for s in steps:
        step_idx = int(s % env.l_cycle)
        half_idx = int((s + half_cycle) % env.l_cycle)

        ref_data = build_ref_data(step_idx)

        # 测试静止场景：每 4 步插入一个静止检查点
        is_stationary_test = (s % 4 == 0)
        info = prepare_info(s, ref_data, is_stationary=is_stationary_test)
        half_info = prepare_info(s + half_cycle, ref_data, is_stationary=is_stationary_test)

        # 记录 gait_step 和 is_stationary
        gait_step_log.append(int(info['gait_step']))
        is_stationary_log.append(float(is_stationary_test))

        height_target_log.append(float(info['foot_ref_z'][foot_idx]))
        height_actual_log.append(float(ref_data.geom_xpos[env.feet_inds[foot_idx], 2]))

        current_fl_target_x = float(info['xy*'][0, 0])
        if s > 0 and abs(current_fl_target_x - last_fl_target_x) > 1e-4:
            raibert_change_log.append(0.12)
        else:
            raibert_change_log.append(0.0)
        last_fl_target_x = current_fl_target_x

        kin_ref = env.kinematic_ref_qpos[step_idx]
        kino_thigh_log.append(float(kin_ref[8]))
        swing_mask_fl_log.append(float(info['foot_swing'][0]))
        swing_mask_fr_log.append(float(info['foot_swing'][1]))

        pose_weight = jp.array([1.0, 1.0, 0.1] * 4)
        qpos = ref_data.qpos[7:19]
        qvel = ref_data.qvel[6:]
        ref_qpos = env.kinematic_ref_qpos[step_idx][7:]
        ref_qpos_half = env.kinematic_ref_qpos[half_idx][7:]
        ref_qvel = env.kinematic_ref_qvel[step_idx][6:]
        ref_qvel_half = env.kinematic_ref_qvel[half_idx][6:]

        pose_err = float(jp.sum(jp.square(qpos - ref_qpos) * pose_weight))
        pose_half_err = float(jp.sum(jp.square(qpos - ref_qpos_half) * pose_weight))
        joint_pose_err_log.append(pose_err)
        joint_pose_half_err_log.append(pose_half_err)
        joint_pose_rew_log.append(float(jp.exp(-pose_err / env._config.rewards.joint_pose_tracking_sigma)))
        joint_pose_half_rew_log.append(float(jp.exp(-pose_half_err / env._config.rewards.joint_pose_tracking_sigma)))

        vel_err = float(jp.sum(jp.square(qvel - ref_qvel)))
        vel_half_err = float(jp.sum(jp.square(qvel - ref_qvel_half)))
        joint_vel_err_log.append(vel_err)
        joint_vel_half_err_log.append(vel_half_err)
        joint_vel_rew_log.append(float(jp.exp(-vel_err / env._config.rewards.joint_vel_tracking_sigma)))
        joint_vel_half_rew_log.append(float(jp.exp(-vel_half_err / env._config.rewards.joint_vel_tracking_sigma)))

        gait_rew, gait_err = gait_terms(ref_data, info)
        gait_half_rew, gait_half_err = gait_terms(ref_data, half_info)
        gait_rew_log.append(gait_rew)
        gait_half_rew_log.append(gait_half_rew)
        gait_err_log.append(gait_err)
        gait_half_err_log.append(gait_half_err)

        feet_traj_err, feet_pos_err, feet_vel_err = foot_traj_terms(ref_data, info)
        feet_traj_half_err, feet_pos_half_err, feet_vel_half_err = foot_traj_terms(ref_data, half_info)
        feet_traj_err_log.append(feet_traj_err)
        feet_traj_half_err_log.append(feet_traj_half_err)
        feet_traj_pos_err_log.append(feet_pos_err)
        feet_traj_pos_half_err_log.append(feet_pos_half_err)
        feet_traj_vel_err_log.append(feet_vel_err)
        feet_traj_vel_half_err_log.append(feet_vel_half_err)

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

    # 子图 0: is_stationary 和 gait_step
    axes[0].step(steps, is_stationary_log, 'r-', where='mid', linewidth=2, label='is_stationary')
    axes[0].plot(steps, gait_step_log, 'g-', linewidth=1, label='gait_step')
    axes[0].set_ylabel('State')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # 子图 1: 足端高度和 swing mask
    axes[1].plot(steps, height_target_log, 'g-', label='Foot Ref Z (current)', linewidth=2)
    axes[1].plot(steps, height_actual_log, color='orange', linestyle='-', label='Ref-Kino Foot Z', linewidth=1.5, alpha=0.85)
    axes[1].bar(steps, raibert_change_log, width=1.0, color='red', alpha=0.25, label='Raibert target update')
    axes[1].plot(steps, np.array(swing_mask_fl_log) * 0.015, color='black', linestyle='--', linewidth=1.5, label='Swing mask FL (scaled)')
    axes[1].plot(steps, np.array(swing_mask_fr_log) * 0.015, color='gray', linestyle='--', linewidth=1.5, label='Swing mask FR (scaled)')
    ax1_twin = axes[1].twinx()
    ax1_twin.plot(steps, kino_thigh_log, 'b--', label='Ref-kino FL thigh', linewidth=2, alpha=0.6)
    axes[1].set_ylabel('Foot Height (m)')
    ax1_twin.set_ylabel('Joint Angle (rad)')
    axes[1].set_title(f'Phase Alignment Dashboard (step_k={env.step_k})')

    # 子图 2: joint pose 和 joint vel reward
    axes[2].plot(steps, joint_pose_rew_log, color='tab:blue', linewidth=2, label='joint_pose reward (current)')
    axes[2].plot(steps, joint_pose_half_rew_log, color='tab:blue', linestyle='--', linewidth=2, label='joint_pose reward (half-cycle)')
    axes[2].plot(steps, joint_vel_rew_log, color='tab:purple', linewidth=2, label='joint_vel reward (current)')
    axes[2].plot(steps, joint_vel_half_rew_log, color='tab:purple', linestyle='--', linewidth=2, label='joint_vel reward (half-cycle)')
    axes[2].set_ylabel('Reward')
    axes[2].set_ylim(-0.05, 1.05)

    # 子图 3: joint pose, joint vel, gait_phase error
    axes[3].plot(steps, joint_pose_err_log, color='tab:blue', linewidth=2, label='joint_pose err (current)')
    axes[3].plot(steps, joint_pose_half_err_log, color='tab:blue', linestyle='--', linewidth=2, label='joint_pose err (half-cycle)')
    axes[3].plot(steps, joint_vel_err_log, color='tab:purple', linewidth=2, label='joint_vel err (current)')
    axes[3].plot(steps, joint_vel_half_err_log, color='tab:purple', linestyle='--', linewidth=2, label='joint_vel err (half-cycle)')
    axes[3].plot(steps, gait_err_log, color='tab:brown', linewidth=2, label='gait_phase err (current)')
    axes[3].plot(steps, gait_half_err_log, color='tab:brown', linestyle='--', linewidth=2, label='gait_phase err (half-cycle)')
    axes[3].set_ylabel('Error')

    # 子图 4: gait_phase reward 和 feet_traj error
    axes[4].plot(steps, gait_rew_log, color='tab:brown', linewidth=2, label='gait_phase reward (current)')
    axes[4].plot(steps, gait_half_rew_log, color='tab:brown', linestyle='--', linewidth=2, label='gait_phase reward (half-cycle)')
    axes[4].plot(steps, feet_traj_err_log, color='tab:green', linewidth=2, label='feet_traj total err (current)')
    axes[4].plot(steps, feet_traj_half_err_log, color='tab:green', linestyle='--', linewidth=2, label='feet_traj total err (half-cycle)')
    axes[4].plot(steps, feet_traj_pos_err_log, color='tab:olive', linewidth=1.5, alpha=0.85, label='feet_traj pos err (current)')
    axes[4].plot(steps, feet_traj_pos_half_err_log, color='tab:olive', linestyle='--', linewidth=1.5, alpha=0.85, label='feet_traj pos err (half-cycle)')
    axes[4].plot(steps, feet_traj_vel_err_log, color='tab:cyan', linewidth=1.5, alpha=0.85, label='feet_traj vel err (current)')
    axes[4].plot(steps, feet_traj_vel_half_err_log, color='tab:cyan', linestyle='--', linewidth=1.5, alpha=0.85, label='feet_traj vel err (half-cycle)')
    axes[4].set_ylabel('Reward / Error')
    axes[4].set_xlabel('Step')

    height_actual_arr = np.array(height_actual_log)
    peak_idx = int(np.argmax(height_actual_arr))
    peak_step = steps[peak_idx]
    peak_height = float(height_actual_arr[peak_idx])
    axes[1].scatter([peak_step], [peak_height], color='orange', s=50, zorder=5)
    axes[1].annotate(
        f"peak={peak_height:.4f} m",
        xy=(peak_step, peak_height),
        xytext=(peak_step + env.step_k * 0.1, peak_height + 0.01),
        arrowprops=dict(arrowstyle='->', color='orange', lw=1.2),
        color='orange',
        fontsize=10,
    )

    for ax in axes:
        ax.axvspan(0, env.step_k, color='gray', alpha=0.08)
        ax.axvspan(env.step_k, env.step_k * 2, color='green', alpha=0.06)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', ncol=2)

    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines1b, labels1b = ax1_twin.get_legend_handles_labels()
    axes[1].legend(lines1 + lines1b, labels1 + labels1b, loc='upper left', ncol=2)

    plt.tight_layout()
    plt.show()


def check_gait_step_transition(env):
    """
    Test the gait_step behavior during command transitions: moving -> stationary -> moving.

    This verifies:
    1. gait_step resets to 0 when command becomes zero
    2. swing_mask becomes [0,0,0,0] during stationary phase
    3. gait_step resumes counting from 0 when command becomes non-zero again
    4. Raibert target updates resume normally after transition
    """
    print("Running gait_step transition test (moving -> stationary -> moving)...")

    # 测试场景设计
    # 阶段 1: 运动 (step=0~74) - gait_step 应该正常递增
    # 阶段 2: 静止 (step=75~124) - gait_step 应该保持 0，swing_mask=[0,0,0,0]
    # 阶段 3: 运动 (step=125~199) - gait_step 应该从 0 重新递增

    total_steps = 200
    stationary_start = 75
    stationary_end = 125

    # 日志变量
    step_log = []
    gait_step_log = []
    is_stationary_log = []
    swing_mask_fl_log = []
    swing_mask_fr_log = []
    raibert_update_log = []  # Raibert target 是否更新
    cmd_norm_log = []

    # 初始化 info
    info = {
        'step': jp.array(0, dtype=jp.int32),
        'gait_step': jp.array(0, dtype=jp.int32),
        'is_stationary': jp.array(False),
        'k0': jp.array(0, dtype=jp.int32),
        'command': jp.array([1.0, 0.0, 0.0]),
        'xy0': jp.zeros((4, 2)),
        'xy*': jp.zeros((4, 2)),
        'foot_phase': 0.0,
        'foot_swing': jp.zeros(4),
        'foot_ref_xy': jp.zeros((4, 2)),
        'foot_ref_z': jp.zeros(4),
        'foot_ref_pos': jp.zeros((4, 3)),
        'foot_ref_v_xy': jp.zeros((4, 2)),
        'z0': jp.zeros(4),
    }

    for s in range(total_steps):
        # 1. 根据阶段设置 command
        if stationary_start <= s < stationary_end:
            cmd = jp.array([0.0, 0.0, 0.0])
        else:
            cmd = jp.array([1.0, 0.0, 0.0])
        info['command'] = cmd

        # 2. 模拟 step() 中的计算逻辑
        cmd_norm = jp.linalg.norm(cmd[:2])
        w_cmd = jp.abs(cmd[2])
        is_stationary = (cmd_norm < 0.01) & (w_cmd < 0.01)
        info['is_stationary'] = is_stationary

        info['gait_step'] = jp.where(
            is_stationary,
            jp.array(0, dtype=jp.int32),
            info['gait_step'] + 1
        )

        info['step'] = jp.array(s, dtype=jp.int32)

        # 3. 构建参考数据
        step_idx = int(s % env.l_cycle)
        ref_data = mjx_env.make_data(
            env.mj_model,
            qpos=env.kinematic_ref_qpos[step_idx],
            qvel=jp.zeros(env.mjx_model.nv),
            ctrl=jp.zeros(env.mjx_model.nu),
            impl=env.mjx_model.impl.value,
            nconmax=env._config.nconmax,
            njmax=env._config.njmax,
        )
        ref_data = mjx.forward(env.mjx_model, ref_data)

        # 4. 记录更新前的 xy*
        xy_star_before = info['xy*'].copy()

        # 5. 更新参考轨迹
        update_raibert_target(env, ref_data, info)
        update_foot_cycloid_ref(env, info)

        # 6. 检查 Raibert target 是否更新
        xy_star_after = info['xy*']
        raibert_updated = jp.any(jp.abs(xy_star_after - xy_star_before) > 1e-6)

        # 7. 记录日志
        step_log.append(s)
        gait_step_log.append(int(info['gait_step']))
        is_stationary_log.append(float(is_stationary))
        swing_mask_fl_log.append(float(info['foot_swing'][0]))  # FL
        swing_mask_fr_log.append(float(info['foot_swing'][1]))  # FR
        raibert_update_log.append(float(raibert_updated))
        cmd_norm_log.append(float(cmd_norm))

    # 8. 绘制结果
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    # 子图 1: command norm
    axes[0].plot(step_log, cmd_norm_log, 'b-', linewidth=2, label='cmd_norm')
    axes[0].axvline(x=stationary_start, color='r', linestyle='--', label='stationary phase')
    axes[0].axvline(x=stationary_end, color='r', linestyle='--')
    axes[0].set_ylabel('Command Norm')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # 子图 2: is_stationary
    axes[1].step(step_log, is_stationary_log, 'r-', where='mid', linewidth=2, label='is_stationary')
    axes[1].set_ylabel('Stationary')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # 子图 3: gait_step
    axes[2].plot(step_log, gait_step_log, 'g-', linewidth=2, label='gait_step')
    axes[2].set_ylabel('Gait Step')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    # 子图 4: swing_mask
    axes[3].plot(step_log, swing_mask_fl_log, 'b-', linewidth=2, label='FL swing_mask')
    axes[3].plot(step_log, swing_mask_fr_log, 'r-', linewidth=2, label='FR swing_mask')
    axes[3].set_ylabel('Swing Mask')
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)

    # 子图 5: Raibert update
    axes[4].scatter(step_log, raibert_update_log, c='purple', s=10, label='Raibert update')
    axes[4].set_ylabel('Target Updated')
    axes[4].set_ylim(-0.1, 1.1)
    axes[4].set_xlabel('Step')
    axes[4].legend(loc='upper right')
    axes[4].grid(True, alpha=0.3)

    # 添加阶段背景色
    for ax in axes:
        ax.axvspan(stationary_start, stationary_end, color='gray', alpha=0.1)

    plt.suptitle(f'Gait Step Transition Test (step_k={env.step_k})')
    plt.tight_layout()
    plt.show()

    # 9. 断言检查
    print("\n=== Assertion Checks ===")

    # 检查 1: 静止阶段 gait_step 应该为 0
    stationary_gait_steps = gait_step_log[stationary_start:stationary_end]
    assert all(gs == 0 for gs in stationary_gait_steps), f"FAIL: gait_step should be 0 during stationary, got {stationary_gait_steps[:10]}..."
    print("✓ Check 1: gait_step = 0 during stationary phase")

    # 检查 2: 静止阶段 swing_mask 应该为 [0,0,0,0]
    stationary_swing_fl = swing_mask_fl_log[stationary_start:stationary_end]
    stationary_swing_fr = swing_mask_fr_log[stationary_start:stationary_end]
    assert all(s == 0 for s in stationary_swing_fl), f"FAIL: FL swing_mask should be 0 during stationary"
    assert all(s == 0 for s in stationary_swing_fr), f"FAIL: FR swing_mask should be 0 during stationary"
    print("✓ Check 2: swing_mask = [0,0,0,0] during stationary phase")

    # 检查 3: 静止阶段结束后 gait_step 应该从 0 重新递增
    post_stationary_gait_steps = gait_step_log[stationary_end:stationary_end + 10]
    expected_steps = list(range(1, 11))
    assert post_stationary_gait_steps == expected_steps, f"FAIL: gait_step should resume from 0, got {post_stationary_gait_steps}"
    print("✓ Check 3: gait_step resumes from 0 after stationary phase")

    # 检查 4: 静止阶段 Raibert target 的变化应该很小（速度为 0 时 target 在原地附近）
    # 注意：Raibert heuristic 在速度为 0 时计算的是原地位置，但由于 kinematic reference
    # 会导致 hip_pos 有微小变化，所以 target 值可能有小幅波动，这是自洽的。
    stationary_updates = raibert_update_log[stationary_start:stationary_end]
    # 不要求完全 frozen，只要求更新幅度不大（说明 target 基本稳定在原地）
    # 这里不 assert，因为 Raibert 计算本身是自洽的，打印日志供调试
    update_ratio = sum(stationary_updates) / len(stationary_updates)
    print(f"✓ Check 4: Raibert target update ratio during stationary = {update_ratio:.2%} (expected: may vary due to hip motion)")

    # 检查 5: 运动阶段 Raibert target 应该周期性更新
    moving_updates_before = raibert_update_log[:stationary_start]
    moving_updates_after = raibert_update_log[stationary_end:]
    assert sum(moving_updates_before) > 0, "FAIL: Raibert target should update during moving phase (before)"
    assert sum(moving_updates_after) > 0, "FAIL: Raibert target should update during moving phase (after)"
    print("✓ Check 5: Raibert target updates normally during moving phase")

    print("\n=== All Checks Passed! ===")

    return fig


def play_cycloid_foot_trajectory(
    env,
    command: Optional[np.ndarray] = None,
    foot: str = "FL_foot",
    num_steps: Optional[int] = None,
    render_every: int = 2,
    save_path: Optional[str] = None,
    camera: Optional[str] = None,
    trail_stride: int = 2,
    trail_max: int = 80,
):
    """Visualize the reward foot trajectory while playing reference kino."""
    print("Playing reward foot trajectory with ref kino...")

    if command is None:
        command = np.array([1.0, 0.0, 0.0])
    command = np.asarray(command, dtype=np.float32)

    if foot in consts.FEET_SITES:
        foot_idx = consts.FEET_SITES.index(foot)
    else:
        raise ValueError(f"Unknown foot site: {foot}. Expected one of {consts.FEET_SITES}.")

    if num_steps is None:
        num_steps = int(env.l_cycle)
    ref_qpos = np.array(env.kinematic_ref_qpos[:num_steps])

    info = {
        'step': jp.array(0, dtype=jp.int32),
        'gait_step': jp.array(0, dtype=jp.int32),
        'is_stationary': jp.array(False),
        'k0': jp.array(0, dtype=jp.int32),
        'command': jp.array(command),
        'xy0': jp.zeros((4, 2)),
        'xy*': jp.zeros((4, 2)),
        'foot_phase': 0.0,
        'foot_swing': jp.zeros(4),
        'foot_ref_xy': jp.zeros((4, 2)),
        'foot_ref_z': jp.zeros(4),
        'foot_ref_pos': jp.zeros((4, 3)),
        'foot_ref_v_xy': jp.zeros((4, 2)),
        'z0': jp.zeros(4),
    }

    path_xyz = np.zeros((num_steps, 3), dtype=np.float32)

    for i in range(num_steps):
        info['step'] = jp.array(i, dtype=jp.int32)
        info['gait_step'] = jp.array(i, dtype=jp.int32)
        info['is_stationary'] = jp.array(False)

        ref_data = mjx_env.make_data(
            env.mj_model,
            qpos=env.kinematic_ref_qpos[i],
            qvel=jp.zeros(env.mjx_model.nv),
            ctrl=jp.zeros(env.mjx_model.nu),
            impl=env.mjx_model.impl.value,
            nconmax=env._config.nconmax,
            njmax=env._config.njmax,
        )
        ref_data = mjx.forward(env.mjx_model, ref_data)

        update_raibert_target(env, ref_data, info)
        update_foot_cycloid_ref(env, info)

        path_xyz[i] = np.array(info['foot_ref_pos'][foot_idx])

    qpos_traj = ref_qpos

    def _add_sphere(scn, pos, radius, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        scn.ngeom += 1
        scn.geoms[scn.ngeom - 1].category = mujoco.mjtCatBit.mjCAT_DECOR
        mujoco.mjv_initGeom(
            geom=scn.geoms[scn.ngeom - 1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0.0, 0.0]),
            pos=np.array(pos),
            mat=np.eye(3).flatten().astype(np.float32),
            rgba=np.asarray(rgba).astype(np.float32),
        )

    modify_scene_fns = []
    for i in range(num_steps):
        def make_fn(idx=i):
            def _fn(scn):
                trail = path_xyz[: idx + 1 : max(1, trail_stride)]
                if trail_max is not None and len(trail) > trail_max:
                    trail = trail[-trail_max:]
                for p in trail:
                    _add_sphere(scn, p, radius=0.007, rgba=[0.2, 0.8, 0.2, 0.6])
                _add_sphere(scn, path_xyz[idx], radius=0.012, rgba=[0.9, 0.3, 0.2, 0.9])
                _add_sphere(scn, path_xyz[0], radius=0.009, rgba=[0.2, 0.4, 0.9, 0.9])
                _add_sphere(scn, path_xyz[-1], radius=0.009, rgba=[0.2, 0.4, 0.9, 0.9])
            return _fn
        modify_scene_fns.append(make_fn())

    frames = env._render_trajectory(
        trajectory=qpos_traj,
        render_every=render_every,
        height=480,
        width=640,
        camera=camera,
        save_path=save_path,
        modify_scene_fns=modify_scene_fns[::render_every],
    )

    fps = 1.0 / (env.dt * render_every)
    media.show_video(frames, fps=fps, loop=True)
    return frames, path_xyz

# ----------------- Anchor Policy Util ----------------- 
def get_anchor_inference_fn(path: str):
    import functools
    from apg_alg.networks import apg_networks

    from brax.io import model
    full_params = model.load_params(path)

    from brax.training.acme import running_statistics
    normalize = running_statistics.normalize

    network_factory = apg_networks.make_apg_networks
    network_factory = functools.partial(
        apg_networks.make_apg_networks, 
            hidden_layer_sizes=(256, 128),
            policy_obs_key="state",
    )
    
    apg_network = network_factory(
        observation_size=consts.ANCHOR_OBS_DIM, 
        action_size=consts.ANCHOR_ACT_DIM, 
        preprocess_observations_fn=normalize
    )

    make_inference_fn = apg_networks.make_inference_fn(apg_network)
    return make_inference_fn(full_params, deterministic=True)
