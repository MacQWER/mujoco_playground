import jax
import jax.numpy as jp
from mujoco_playground._src.locomotion.go2.Util.TrotUtil import rotate_inv

# ==========================================
# 1. 噪声处理器
# ==========================================
def apply_uniform_noise(rng: jax.Array, x: jax.Array, noise_level: float, scale: float):
    """通用的均匀分布加噪函数"""
    rng, noise_rng = jax.random.split(rng)
    noise = (2 * jax.random.uniform(noise_rng, shape=x.shape) - 1) * noise_level * scale
    return x + noise, rng

# ==========================================
# 2. 物理状态传感器
# ==========================================
def base_angular_velocity(data, info, **kwargs) -> jax.Array:
    """获取基座在局部坐标系下的角速度"""
    return rotate_inv(data.cvel[1, :3], data.xquat[1])

def projected_gravity(data, info, **kwargs) -> jax.Array:
    """获取重力向量在基座局部坐标系下的投影"""
    g_world = jp.array([0.0, 0.0, -1.0])
    return rotate_inv(g_world, data.xquat[1])

def joint_positions(data, info, **kwargs) -> jax.Array:
    """获取关节相对默认姿态的偏差"""
    angles = data.qpos[7:19]
    return angles - kwargs['default_ap_pose']

def joint_velocities(data, info, **kwargs) -> jax.Array:
    """获取关节速度"""
    return data.qvel[6:]

# ==========================================
# 3. 任务与上下文特征
# ==========================================
def command(data, info, **kwargs) -> jax.Array:
    """获取当前的真实指令 [vx, vy, wz]"""
    return info['command']

def zero_command(data, info, **kwargs) -> jax.Array:
    """占位符指令 (用于 Trot 或 Anchor 屏蔽特权)"""
    return jp.zeros(3)

def last_action(data, info, **kwargs) -> jax.Array:
    """获取上一步的动作"""
    return info['last_action']

def anchor_action(data, info, **kwargs) -> jax.Array:
    """获取 Anchor Policy 输出的前置动作"""
    return info['anchor_action']

def zero_anchor_action(data, info, **kwargs) -> jax.Array:
    """占位符动作 (用于 Trot 或 Anchor 屏蔽前置动作)"""
    return jp.zeros(12)

def kinematic_reference(data, info, **kwargs) -> jax.Array:
    """获取当前步态相位的运动学参考位姿

    Uses gait_step (which resets to 0 when stationary) instead of global step.
    When stationary, reference pose stays at the initial pose (index 0).
    """
    l_cycle = kwargs['l_cycle']
    step_idx = jp.array(info['gait_step'] % l_cycle, int)
    return kwargs['kin_ref_qpos'][step_idx][7:]


def gait_phase(data, info, **kwargs) -> jax.Array:
    """步态相位循环编码 [sin(θ), cos(θ)]

    解决观测歧义：让策略明确知道当前步态相位，
    消除 kinematic_reference 在不同相位返回相同值的歧义。

    静止时返回 [0, 0] 作为特殊标记。
    """
    l_cycle = kwargs['l_cycle']
    is_stationary = info.get('is_stationary', False)

    # 运动时使用 gait_step，静止时返回特殊标记
    step = info['gait_step']
    phase = 2.0 * jp.pi * step / l_cycle
    phase_enc = jp.array([jp.sin(phase), jp.cos(phase)])

    # 静止时返回 [0, 0]
    return jp.where(is_stationary, jp.zeros(2), phase_enc)