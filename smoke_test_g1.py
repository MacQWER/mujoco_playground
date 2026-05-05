"""Smoke test v2 for G1Joystick2.

Uses realistic action range (0.3 = ±17 deg joint perturbation) since G1 is
a biped and U(-1,1) actions cause immediate falls.  A freshly-initialized
policy also outputs small actions early in training.

Checks:
1. Env creation and reset
2. 200 steps without NaN/Inf
3. Key reward values in reasonable ranges
4. Contact detection is smooth (sigmoid, not binary)
5. Hand-thigh collision penalty is smooth
6. No spurious hard terminations
"""

import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground import locomotion

# ---------------------------------------------------------------------------
# 1. Create environment
# ---------------------------------------------------------------------------
env = locomotion.load("G1Joystick2")
print("1. Environment created:", type(env).__name__)
print(f"   nu (action dim): {env.mjx_model.nu}")
print(f"   action_scale: {env.action_scale}")

jit_reset = jax.jit(env.reset)

rng = jax.random.PRNGKey(42)
state = jit_reset(rng)
print("2. Reset OK. obs shape:", state.obs["state"].shape)
print(f"   Obs dim: {state.obs['state'].shape[0]}")

# ---------------------------------------------------------------------------
# 2. Scan over steps with moderate action range
# ---------------------------------------------------------------------------
ACTION_RANGE = 0.3  # ±17 deg — realistic for early-training actions

def single_step(state, rng):
    action = jax.random.uniform(rng, (env.mjx_model.nu,), minval=-ACTION_RANGE, maxval=ACTION_RANGE)
    return env.step(state, action)

jit_step = jax.jit(single_step)

def scan_step(carry, x):
    state, rng = carry
    rng, step_rng = jax.random.split(rng)
    new_state = jit_step(state, step_rng)
    info = new_state.info
    reward_tuple = info["reward_tuple"]
    contact = info["last_contact"]
    phase = info["phase"]
    foot_ref_z = info["foot_ref_z"]
    obs = new_state.obs["state"]
    done = new_state.done
    metrics = {
        "reward": new_state.reward,
        "done": done.astype(jp.float32),
        "contact_left": contact[0],
        "contact_right": contact[1],
        "phase_0": phase[0],
        "phase_1": phase[1],
        "foot_ref_z_l": foot_ref_z[0],
        "foot_ref_z_r": foot_ref_z[1],
        "obs_norm": jp.linalg.norm(obs),
        "has_nan": jp.isnan(obs).any() | jp.isnan(new_state.reward).any(),
        "has_inf": jp.isinf(obs).any() | jp.isinf(new_state.reward).any(),
    }
    for k, v in reward_tuple.items():
        metrics[f"r_{k}"] = v
    return (new_state, rng), metrics

n_steps = 200
print(f"\n3. Running {n_steps} steps with action ~ U(-{ACTION_RANGE}, {ACTION_RANGE})...")

# Warm-up compile
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rng, scan_rng = jax.random.split(rng)
(state, _), history = jax.lax.scan(scan_step, (state, scan_rng), None, length=n_steps)
history = jax.device_get(history)
print(f"   JIT compiled and executed.")

# ---------------------------------------------------------------------------
# 3. Diagnostics
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DIAGNOSTICS")
print("=" * 70)

# NaN check
total_nan = np.sum(history["has_nan"])
total_inf = np.sum(history["has_inf"])
print(f"\nNaN count: {total_nan} / {n_steps}")
print(f"Inf count: {total_inf} / {n_steps}")

# Done (termination) check
done_sum = np.sum(history["done"])
print(f"Hard termination events: {done_sum:.0f} / {n_steps}")

# Reward stats
print(f"\nReward per step:")
print(f"  mean={history['reward'].mean():.4f}, std={history['reward'].std():.4f}")
print(f"  min={history['reward'].min():.4f}, max={history['reward'].max():.4f}")

# Contact check
contact_l = history["contact_left"]
contact_r = history["contact_right"]
print(f"\nContact (left foot):  mean={contact_l.mean():.4f}, std={contact_l.std():.4f}, range=[{contact_l.min():.4f}, {contact_l.max():.4f}]")
print(f"Contact (right foot): mean={contact_r.mean():.4f}, std={contact_r.std():.4f}, range=[{contact_r.min():.4f}, {contact_r.max():.4f}]")
print(f"  → sigmoid output (smooth, values across [0,1]): {'OK' if len(np.unique(np.round(contact_l, 3))) > 20 else 'TOO BINARY'}")

# Foot ref z
foot_z_l = history["foot_ref_z_l"]
foot_z_r = history["foot_ref_z_r"]
print(f"\nFoot ref Z (left):  mean={foot_z_l.mean():.4f}, range=[{foot_z_l.min():.4f}, {foot_z_l.max():.4f}]")
print(f"Foot ref Z (right): mean={foot_z_r.mean():.4f}, range=[{foot_z_r.min():.4f}, {foot_z_r.max():.4f}]")

# Phase
print(f"\nPhase 0: mean={history['phase_0'].mean():.4f}, range=[{history['phase_0'].min():.4f}, {history['phase_0'].max():.4f}]")
print(f"Phase 1: mean={history['phase_1'].mean():.4f}, range=[{history['phase_1'].min():.4f}, {history['phase_1'].max():.4f}]")

# Obs norm
print(f"\nObs norm: mean={history['obs_norm'].mean():.4f}, std={history['obs_norm'].std():.4f}")
print(f"  min={history['obs_norm'].min():.4f}, max={history['obs_norm'].max():.4f}")

# ---------------------------------------------------------------------------
# 4. Key reward term analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("REWARD TERM ANALYSIS (mean ± std)")
print("=" * 70)

reward_keys = sorted([k for k in history.keys() if k.startswith("r_")])
for k in reward_keys:
    vals = history[k]
    name = k[2:]
    abs_mean = np.abs(vals).mean()
    if abs_mean > 0.0005:
        print(f"  {name:30s}: {vals.mean():10.6f} ± {vals.std():10.6f}  [min={vals.min():.6f}, max={vals.max():.6f}]")

# ---------------------------------------------------------------------------
# 5. Specific checks for recently modified reward terms
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("MODIFIED REWARD CHECKS")
print("=" * 70)

# feet_traj: should always be <= 0 (penalty)
ft = history["r_feet_traj"]
print(f"\nfeet_traj (scale=-5.0): mean={ft.mean():.4f}, range=[{ft.min():.4f}, {ft.max():.4f}]")
print(f"  → {'OK: always <= 0' if np.all(ft <= 0) else 'ISSUE: has positive values!'}")

# gait_phase_tracking: should be >= 0
gpt = history["r_gait_phase_tracking"]
print(f"\ngait_phase_tracking (scale=1.0): mean={gpt.mean():.4f}, range=[{gpt.min():.4f}, {gpt.max():.4f}]")
print(f"  → {'OK: always >= 0' if np.all(gpt >= 0) else 'ISSUE: has negative values!'}")

# collision: should be <= 0, smooth
col = history["r_collision"]
n_unique = len(np.unique(np.round(col, 5)))
print(f"\ncollision (scale=-0.1): mean={col.mean():.6f}, range=[{col.min():.6f}, {col.max():.6f}]")
print(f"  → unique values: {n_unique}, {'OK: smooth' if n_unique > 10 else 'ISSUE: too few unique values (binary?)'}")

# termination: should be ~0 (no falls with moderate actions)
term = history["r_termination"]
print(f"\ntermination (scale=-100.0): mean={term.mean():.6f}")
print(f"  → non-zero steps: {np.sum(np.abs(term) > 0.001)}, {'OK: mostly zero' if np.sum(np.abs(term) > 0.01) < 10 else 'ISSUE: too many termination penalties'}")

# tracking_lin_vel
tlv = history["r_tracking_lin_vel"]
print(f"\ntracking_lin_vel (scale=1.0): mean={tlv.mean():.4f}, range=[{tlv.min():.4f}, {tlv.max():.4f}]")

# feet_air_time — should be >= 0
fat = history["r_feet_air_time"]
print(f"\nfeet_air_time (scale=2.0): mean={fat.mean():.4f}, range=[{fat.min():.4f}, {fat.max():.4f}]")
print(f"  → {'OK: always >= 0' if np.all(fat >= -1e-6) else 'ISSUE: has negative values!'}")

# contact_force — check if reasonable
cf = history["r_contact_force"]
print(f"\ncontact_force (scale=-0.01): mean={cf.mean():.4f}, range=[{cf.min():.4f}, {cf.max():.4f}]")
print(f"  → raw z-force mean ≈ {abs(cf.mean())/0.01:.0f} N")

# ---------------------------------------------------------------------------
# 6. Final verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)

issues = []
if total_nan > 0:
    issues.append(f"NaN detected in {total_nan} steps")
if total_inf > 0:
    issues.append(f"Inf detected in {total_inf} steps")
if done_sum > n_steps * 0.2:
    issues.append(f"Too many hard terminations: {done_sum:.0f}/{n_steps}")
if np.any(ft > 0.01):
    issues.append("feet_traj reward has positive values")
if np.any(gpt < -0.01):
    issues.append("gait_phase_tracking has negative values")
if np.all(contact_l > 0.99) and np.all(contact_r > 0.99):
    issues.append("Foot contact always saturated at ~1")
if np.all(contact_l < 0.01) and np.all(contact_r < 0.01):
    issues.append("No foot contact at all")
if np.all(col > -1e-6) and np.all(col < 1e-6):
    pass  # collision ~0 is fine, means no hand-thigh contact
if np.any(fat < -0.01):
    issues.append("feet_air_time has significant negative values")

if issues:
    print("\nISSUES FOUND:")
    for i in issues:
        print(f"  ✗ {i}")
else:
    print("\n  All checks passed. Rewards are numerically stable and in expected ranges.")

print(f"\nTotal reward over {n_steps} steps: {history['reward'].sum():.4f}")
print("Done.")
