"""Visualize PPO solimp/solref sweep results for paper."""

import os, re, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "sweep")
# Use latest timestamp
timestamps = sorted(set(
    f.split("_")[1] + "_" + f.split("_")[2]
    for f in os.listdir(LOG_DIR)
    if f.startswith("ppo_2026") and f.endswith(".log")
))
latest_ts = timestamps[-1]  # e.g. "20260504_152133"

BASE = np.array([0.01, 0.5, 0.03])
HIGH = np.array([0.95, 0.99, 0.001])

results = {}
for f in sorted(glob.glob(f"{LOG_DIR}/ppo_{latest_ts}_*.log")):
    with open(f) as fh:
        content = fh.read()
    m = re.search(r"\[1/1\]\s+(\S+)", content)
    if not m:
        continue
    parts = m.group(1).split("_")
    a0 = float(parts[0][2:])
    a1 = float(parts[1][2:])
    a2 = float(parts[2][2:])
    ar0 = float(parts[3][3:])
    m_r = re.search(r"eval/episode_reward\s+(-?[\d.]+)", content)
    if m_r:
        results[(a0, a1, a2, ar0)] = float(m_r.group(1))

a_vals = sorted(set(k[i] for k in results for i in range(4)))
a0_vals = a_vals  # all same: [0, 0.33, 0.67, 1.0]
a1_vals = a_vals
a2_vals = a_vals
ar0_vals = [0.0, 1.0]

rew = np.array(list(results.values()))
print(f"Reward range: [{rew.min():.2f}, {rew.max():.2f}]")

# ─── Style ───
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ─── FIG 1: 2×4 grid ───
fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True, sharey=True)
cmap = plt.cm.RdYlGn
vmin, vmax = -45, -20

for row, ar0 in enumerate(ar0_vals):
    for col, a2 in enumerate(a2_vals):
        ax = axes[row][col]
        grid = np.full((4, 4), np.nan)
        for i, a0 in enumerate(a0_vals):
            for j, a1 in enumerate(a1_vals):
                if (a0, a1, a2, ar0) in results:
                    grid[3 - j, i] = results[(a0, a1, a2, ar0)]  # flip y so a1=0 at bottom
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", origin="lower",
                        extent=[-0.165, 1.165, -0.165, 1.165])
        # Annotate cells
        for i, a0 in enumerate(a0_vals):
            for j, a1 in enumerate(a1_vals):
                val = results.get((a0, a1, a2, ar0))
                if val is not None:
                    ax.text(a0, a1, f"{val:.1f}", ha="center", va="center", fontsize=6.5,
                            color="black" if val > -32 else "white")
        ax.set_title(f"a$_2$={a2:.2f}" + ("  (ar0=0)" if ar0 == 0 else "  (ar0=1)"), fontsize=10)
        ax.set_xticks(a0_vals)
        ax.set_yticks(a1_vals)
        ax.set_xticklabels([f"{v:.2f}" for v in a0_vals], fontsize=7)
        ax.set_yticklabels([f"{v:.2f}" for v in a1_vals], fontsize=7)

for col in range(4):
    axes[1][col].set_xlabel("a$_0$ (solimp[0])")
for row in range(2):
    axes[row][0].set_ylabel("a$_1$ (solimp[1])")

cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
cbar.set_label("Episode Reward")

fig.suptitle("PPO Sweep: solimp/solref → PushBox Reward", fontsize=13, y=1.01)
plt.tight_layout()
os.makedirs("logs/sweep/figures", exist_ok=True)
fig.savefig("logs/sweep/figures/sweep_heatmap_grid.png")
print("Saved: logs/sweep/figures/sweep_heatmap_grid.png")

# ─── FIG 2: Marginal effects ───
fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

labels = ["a0 (solimp[0])", "a1 (solimp[1])", "a2 (solimp[2])", "ar0 (solref[0])"]
for dim in range(4):
    ax = axes[dim]
    vals = sorted(set(k[dim] for k in results))
    data = []
    for v in vals:
        rewards = [results[k] for k in results if k[dim] == v]
        data.append(rewards)
    bp = ax.boxplot(data, labels=[f"{v:.2f}" for v in vals], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.set_title(labels[dim])
    ax.set_ylabel("Episode Reward")

fig.suptitle("Marginal Effect of Each Parameter on Reward", fontsize=12)
plt.tight_layout()
fig.savefig("logs/sweep/figures/sweep_marginal.png")
print("Saved: logs/sweep/figures/sweep_marginal.png")

# ─── FIG 3: Top 10 vs Bottom 10 bar chart with actual solimp/solref values ───
fig, ax = plt.subplots(figsize=(14, 6))

all_sorted = sorted(results.items(), key=lambda x: x[1], reverse=True)
top10 = all_sorted[:10]
bot10 = all_sorted[-10:]

def fmt_params(a0, a1, a2, ar0):
    si = [(1 - a0) * BASE[0] + a0 * HIGH[0],
          (1 - a1) * BASE[1] + a1 * HIGH[1],
          (1 - a2) * BASE[2] + a2 * HIGH[2]]
    sr0 = (1 - ar0) * 0.004 + ar0 * 0.02
    return f"[{si[0]:.2f},{si[1]:.2f},{si[2]:.3f}] | [{sr0:.3f}]"

combined = top10 + bot10
labels = [f"a=[{a0:.1f},{a1:.1f},{a2:.1f},{ar0:.1f}]\n{fmt_params(a0,a1,a2,ar0)}"
          for (a0, a1, a2, ar0), _ in combined]
values = [v for _, v in combined]
colors = ["#2ca02c"] * 10 + ["#d62728"] * 10

bars = ax.barh(range(20), values, color=colors)
ax.set_yticks(range(20))
ax.set_yticklabels(labels, fontsize=8)
ax.axvline(x=0, color="black", linewidth=0.5)
ax.set_xlabel("Episode Reward")
ax.set_title("Top 10 vs Bottom 10 Configurations")
ax.invert_yaxis()

# Add value labels
for bar, val in zip(bars, values):
    ax.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", ha="right", va="center", fontsize=8, color="white")

plt.tight_layout()
fig.savefig("logs/sweep/figures/sweep_top_bottom.png")
print("Saved: logs/sweep/figures/sweep_top_bottom.png")

# ─── FIG 4: Best vs worst solimp/solref trajectory comparison concept ───
# Show where in the 4D space good/bad configs cluster
fig = plt.figure(figsize=(12, 5))

# Plot a0 vs a1, color by reward, split by ar0
for idx, ar0 in enumerate([0.0, 1.0]):
    ax = fig.add_subplot(1, 2, idx + 1)
    subset = {k: v for k, v in results.items() if k[3] == ar0}
    a0s = [k[0] for k in subset]
    a1s = [k[1] for k in subset]
    a2s = [k[2] for k in subset]
    rews = list(subset.values())
    scatter = ax.scatter(a0s, a1s, c=rews, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                         s=120, edgecolors="black", linewidth=0.5)
    for (a0, a1, a2, _), r in subset.items():
        ax.annotate(f"a2={a2:.1f}\n{r:.1f}", (a0, a1), fontsize=5.5, ha="center",
                    textcoords="offset points", xytext=(0, -14))
    ax.set_xlabel("a0"); ax.set_ylabel("a1")
    ax.set_title(f"ar0 = {ar0:.1f} (solref[0] = {(1-ar0)*0.004 + ar0*0.02:.3f})")
    ax.set_xticks(a0_vals); ax.set_yticks(a1_vals)
    cbar = plt.colorbar(scatter, ax=ax); cbar.set_label("Reward")

fig.suptitle("Reward Distribution in a0×a1 Space", fontsize=12)
plt.tight_layout()
fig.savefig("logs/sweep/figures/sweep_scatter.png")
print("Saved: logs/sweep/figures/sweep_scatter.png")

plt.close("all")
print("\nDone! 4 figures saved to logs/sweep/figures/")
