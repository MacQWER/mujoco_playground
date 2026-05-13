"""Visualize APG solimp/solref sweep results."""

import csv
import glob
import os
import re
import statistics

import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "sweep")
FIGURE_DIR = os.path.join(LOG_DIR, "figures")

BASE_SOLIMP = np.array([0.01, 0.5, 0.03])
HIGH_SOLIMP = np.array([0.95, 0.99, 0.001])
BASE_SOLREF0 = 0.004
HIGH_SOLREF0 = 0.02

A_LEVELS = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
AR0_LEVELS = [0.0, 1.0]


def _build_grid():
  return [
      (a0, a1, a2, ar0)
      for a0 in A_LEVELS
      for a1 in A_LEVELS
      for a2 in A_LEVELS
      for ar0 in AR0_LEVELS
  ]


def _params(a0, a1, a2, ar0):
  a = np.array([a0, a1, a2])
  solimp = (1.0 - a) * BASE_SOLIMP + a * HIGH_SOLIMP
  solref = np.array([
      (1.0 - ar0) * BASE_SOLREF0 + ar0 * HIGH_SOLREF0,
      1.0,
  ])
  return solimp, solref


def _parse_reward(text):
  reward = None
  pattern = re.compile(r"wandb:\s+eval/episode_reward\s+(-?\d+(?:\.\d+)?)")
  for match in pattern.finditer(text):
    reward = float(match.group(1))
  return reward


def _parse_penetration(text):
  match = re.search(r"video done \(max penetration: ([0-9.]+)\)", text)
  if match is None:
    return None
  return float(match.group(1))


def _parse_run_id(text):
  match = re.search(r"runs/([A-Za-z0-9]+)", text)
  if match is None:
    return ""
  return match.group(1)


def _timestamp(path):
  match = re.search(r"apg_(\d{8}_\d{6})_", os.path.basename(path))
  if match is None:
    return ""
  return match.group(1)


def load_results():
  """Loads the latest completed APG result per launcher slot."""
  grid = _build_grid()
  by_slot = {}

  for path in glob.glob(os.path.join(LOG_DIR, "apg_*_slot*.log")):
    slot_match = re.search(r"_slot(\d+)\.log$", path)
    if slot_match is None:
      continue

    slot = int(slot_match.group(1))
    if slot >= len(grid):
      continue

    with open(path, "r", encoding="utf-8", errors="replace") as f:
      text = f.read()

    if "video done" not in text:
      continue

    reward = _parse_reward(text)
    if reward is None:
      continue

    a0, a1, a2, ar0 = grid[slot]
    solimp, solref = _params(a0, a1, a2, ar0)
    record = {
        "slot": slot,
        "timestamp": _timestamp(path),
        "path": path,
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "ar0": ar0,
        "solimp0": float(solimp[0]),
        "solimp1": float(solimp[1]),
        "solimp2": float(solimp[2]),
        "solref0": float(solref[0]),
        "reward": reward,
        "max_penetration": _parse_penetration(text),
        "run_id": _parse_run_id(text),
    }

    previous = by_slot.get(slot)
    if previous is None or (record["timestamp"], path) > (
        previous["timestamp"],
        previous["path"],
    ):
      by_slot[slot] = record

  return [by_slot[i] for i in sorted(by_slot)]


def _result_map(results):
  return {
      (r["a0"], r["a1"], r["a2"], r["ar0"]): r["reward"]
      for r in results
  }


def _tick_labels(param):
  if param == "a0":
    return [f"{a:.2f}\n{_params(a, 0, 0, 0)[0][0]:.3f}" for a in A_LEVELS]
  if param == "a1":
    return [f"{a:.2f}\n{_params(0, a, 0, 0)[0][1]:.3f}" for a in A_LEVELS]
  if param == "a2":
    return [f"{a:.2f}\n{_params(0, 0, a, 0)[0][2]:.3f}" for a in A_LEVELS]
  return [f"{a:.0f}\n{_params(0, 0, 0, a)[1][0]:.3f}" for a in AR0_LEVELS]


def save_csv(results):
  os.makedirs(FIGURE_DIR, exist_ok=True)
  path = os.path.join(FIGURE_DIR, "apg_sweep_results.csv")
  fieldnames = [
      "slot",
      "a0",
      "a1",
      "a2",
      "ar0",
      "solimp0",
      "solimp1",
      "solimp2",
      "solref0",
      "reward",
      "max_penetration",
      "run_id",
      "path",
  ]
  with open(path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
      writer.writerow({k: row[k] for k in fieldnames})
  return path


def plot_heatmap_grid(results):
  rewards = _result_map(results)
  values = np.array([r["reward"] for r in results])
  vmin, vmax = float(values.min()), float(values.max())

  fig, axes = plt.subplots(
      2,
      4,
      figsize=(15.0, 7.0),
      sharex=True,
      sharey=True,
      constrained_layout=True,
  )

  cmap = plt.cm.RdYlGn
  for row, ar0 in enumerate(AR0_LEVELS):
    solref0 = _params(0, 0, 0, ar0)[1][0]
    for col, a2 in enumerate(A_LEVELS):
      ax = axes[row][col]
      grid = np.full((4, 4), np.nan)
      for i, a0 in enumerate(A_LEVELS):
        for j, a1 in enumerate(A_LEVELS):
          grid[j, i] = rewards[(a0, a1, a2, ar0)]

      im = ax.imshow(
          grid,
          cmap=cmap,
          vmin=vmin,
          vmax=vmax,
          origin="lower",
          extent=[-0.5, 3.5, -0.5, 3.5],
      )
      for i, a0 in enumerate(A_LEVELS):
        for j, a1 in enumerate(A_LEVELS):
          reward = rewards[(a0, a1, a2, ar0)]
          text_color = "white" if reward < -34 else "black"
          ax.text(
              i,
              j,
              f"{reward:.1f}",
              ha="center",
              va="center",
              fontsize=7,
              color=text_color,
          )

      solimp2 = _params(0, 0, a2, 0)[0][2]
      ax.set_title(f"a2={a2:.2f}  solimp[2]={solimp2:.3f}")
      ax.set_xticks(range(4), [f"{a:.2f}" for a in A_LEVELS], fontsize=8)
      ax.set_yticks(range(4), [f"{a:.2f}" for a in A_LEVELS], fontsize=8)
      if row == 1:
        ax.set_xlabel("a0  solimp[0]")
      if col == 0:
        ax.set_ylabel("a1  solimp[1]")

  cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92)
  cbar.set_label("eval/episode_reward (higher is better)")
  fig.text(0.012, 0.71, "solref[0]=0.004", rotation=90, va="center")
  fig.text(0.012, 0.29, "solref[0]=0.020", rotation=90, va="center")
  fig.suptitle("APG sweep reward: a0 x a1 heatmaps, faceted by a2 and solref")

  path = os.path.join(FIGURE_DIR, "apg_heatmap_grid.png")
  fig.savefig(path)
  plt.close(fig)
  return path


def plot_solref_delta(results):
  rewards = _result_map(results)
  deltas = []
  for a0 in A_LEVELS:
    for a1 in A_LEVELS:
      for a2 in A_LEVELS:
        deltas.append(
            rewards[(a0, a1, a2, 1.0)] - rewards[(a0, a1, a2, 0.0)]
        )
  limit = max(abs(float(np.min(deltas))), abs(float(np.max(deltas))))

  fig, axes = plt.subplots(
      1,
      4,
      figsize=(15.0, 3.8),
      sharex=True,
      sharey=True,
      constrained_layout=True,
  )
  for col, a2 in enumerate(A_LEVELS):
    ax = axes[col]
    grid = np.full((4, 4), np.nan)
    for i, a0 in enumerate(A_LEVELS):
      for j, a1 in enumerate(A_LEVELS):
        grid[j, i] = (
            rewards[(a0, a1, a2, 1.0)] - rewards[(a0, a1, a2, 0.0)]
        )

    im = ax.imshow(
        grid,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        origin="lower",
        extent=[-0.5, 3.5, -0.5, 3.5],
    )
    for i in range(4):
      for j in range(4):
        value = grid[j, i]
        text_color = "white" if abs(value) > 0.55 * limit else "black"
        ax.text(
            i,
            j,
            f"{value:+.1f}",
            ha="center",
            va="center",
            fontsize=7,
            color=text_color,
        )

    solimp2 = _params(0, 0, a2, 0)[0][2]
    ax.set_title(f"a2={a2:.2f}  solimp[2]={solimp2:.3f}")
    ax.set_xticks(range(4), [f"{a:.2f}" for a in A_LEVELS], fontsize=8)
    ax.set_yticks(range(4), [f"{a:.2f}" for a in A_LEVELS], fontsize=8)
    ax.set_xlabel("a0  solimp[0]")
    if col == 0:
      ax.set_ylabel("a1  solimp[1]")

  cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
  cbar.set_label("reward delta: solref[0]=0.020 minus 0.004")
  fig.suptitle("Effect of solref[0]: positive means solref[0]=0.020 is better")

  path = os.path.join(FIGURE_DIR, "apg_solref_delta.png")
  fig.savefig(path)
  plt.close(fig)
  return path


def plot_main_effects(results):
  fig, axes = plt.subplots(
      1,
      4,
      figsize=(14.0, 3.8),
      constrained_layout=True,
  )
  params = [
      ("a0", A_LEVELS, "a0 / solimp[0]"),
      ("a1", A_LEVELS, "a1 / solimp[1]"),
      ("a2", A_LEVELS, "a2 / solimp[2]"),
      ("ar0", list(reversed(AR0_LEVELS)), "ar0 / solref[0]"),
  ]

  for ax, (key, levels, title) in zip(axes, params):
    grouped = [
        [r["reward"] for r in results if abs(r[key] - level) < 1e-9]
        for level in levels
    ]
    bp = ax.boxplot(grouped, patch_artist=True, showmeans=True)
    for patch in bp["boxes"]:
      patch.set_facecolor("#d9e8ff")
      patch.set_edgecolor("#3b5b8a")
    for median in bp["medians"]:
      median.set_color("#1f1f1f")
    means = [statistics.mean(xs) for xs in grouped]
    ax.plot(range(1, len(levels) + 1), means, color="#c03a2b", marker="o")
    ax.set_title(title)
    ax.set_xticks(range(1, len(levels) + 1), _tick_labels(key), fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel("eval/episode_reward")

  fig.suptitle("Marginal effects with variance from other parameters")
  path = os.path.join(FIGURE_DIR, "apg_main_effects.png")
  fig.savefig(path)
  plt.close(fig)
  return path


def _factor_tick_labels(key, levels):
  """Tick labels with alpha, physical value, and soft/hard annotation."""
  labels = []
  for i, level in enumerate(levels):
    if key == "a0":
      phys = _params(level, 0, 0, 0)[0][0]
    elif key == "a1":
      phys = _params(0, level, 0, 0)[0][1]
    elif key == "a2":
      phys = _params(0, 0, level, 0)[0][2]
    else:  # ar0
      phys = _params(0, 0, 0, level)[1][0]

    is_first = i == 0
    is_last = i == len(levels) - 1
    label = f"{phys:.3f}"
    if is_first:
      label += "\nsoft"
    elif is_last:
      label += "\nstiff" if key == "ar0" else "\nhard"
    labels.append(label)
  return labels


def plot_factor_boxplots(results):
  """2x2 grouped boxplots with jittered scatter overlay."""
  specs = [
      ("a0", A_LEVELS, "solimp[0]"),
      ("a1", A_LEVELS, "solimp[1]"),
      ("a2", A_LEVELS, "solimp[2]"),
      ("ar0", list(reversed(AR0_LEVELS)), "solref[0]"),
  ]
  colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
  rng = np.random.default_rng(7)

  fig, axes = plt.subplots(
      2, 2, figsize=(12.5, 8.0), sharey=True, constrained_layout=True
  )

  for ax, (key, levels, title), color in zip(axes.ravel(), specs, colors):
    grouped = [
        [r["reward"] for r in results if abs(r[key] - level) < 1e-9]
        for level in levels
    ]
    data = [vals for vals in grouped if vals]
    positions = [i + 1 for i, vals in enumerate(grouped) if vals]
    tick_labels = _factor_tick_labels(key, levels)
    labels = [tick_labels[i] for i, vals in enumerate(grouped) if vals]
    if not data:
      ax.set_title(f"{title} (no data)")
      continue

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#111111",
            "markeredgecolor": "#111111",
            "markersize": 4,
        },
    )
    for patch in bp["boxes"]:
      patch.set_facecolor(color)
      patch.set_alpha(0.35)
      patch.set_edgecolor(color)
    for median in bp["medians"]:
      median.set_color("#111111")
      median.set_linewidth(1.5)

    means = [statistics.mean(vals) for vals in data]
    ax.plot(positions, means, color=color, marker="o", linewidth=1.8)

    for x_pos, vals in zip(positions, data):
      jitter = rng.uniform(-0.08, 0.08, len(vals))
      ax.scatter(
          np.full(len(vals), x_pos) + jitter,
          vals,
          s=24,
          color=color,
          edgecolors="black",
          linewidths=0.35,
          alpha=0.8,
          zorder=3,
      )

    ax.set_title(title)
    ax.set_xticks(positions, labels)
    ax.grid(axis="y", alpha=0.22)

  axes[0, 0].set_ylabel("eval/episode_reward")
  axes[1, 0].set_ylabel("eval/episode_reward")
  fig.suptitle("APG PushBox sweep — factor main effects / grouped boxplots")
  p = os.path.join(FIGURE_DIR, "apg_factor_boxplots.png")
  fig.savefig(p)
  plt.close(fig)
  return p


def plot_3d_scatter(results):
  fig = plt.figure(figsize=(13.0, 5.5), constrained_layout=True)
  values = np.array([r["reward"] for r in results])
  vmin, vmax = float(values.min()), float(values.max())
  best = max(results, key=lambda r: r["reward"])

  for idx, ar0 in enumerate(AR0_LEVELS):
    ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
    subset = [r for r in results if abs(r["ar0"] - ar0) < 1e-9]
    scatter = ax.scatter(
        [r["solimp0"] for r in subset],
        [r["solimp1"] for r in subset],
        [r["solimp2"] for r in subset],
        c=[r["reward"] for r in subset],
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        s=58,
        depthshade=False,
        edgecolors="black",
        linewidths=0.35,
    )
    if abs(best["ar0"] - ar0) < 1e-9:
      ax.scatter(
          [best["solimp0"]],
          [best["solimp1"]],
          [best["solimp2"]],
          marker="*",
          s=220,
          color="gold",
          edgecolors="black",
          linewidths=0.8,
      )
      ax.text(
          best["solimp0"],
          best["solimp1"],
          best["solimp2"],
          f" best\n {best['reward']:.2f}",
          fontsize=8,
      )

    solref0 = _params(0, 0, 0, ar0)[1][0]
    ax.set_title(f"solref[0]={solref0:.3f}")
    ax.set_xlabel("solimp[0]")
    ax.set_ylabel("solimp[1]")
    ax.set_zlabel("solimp[2]")
    ax.view_init(elev=23, azim=130)

  cbar = fig.colorbar(scatter, ax=fig.axes, shrink=0.8, pad=0.08)
  cbar.set_label("eval/episode_reward")
  fig.suptitle("3D lattice view of solimp values, split by solref")

  path = os.path.join(FIGURE_DIR, "apg_3d_scatter.png")
  fig.savefig(path)
  plt.close(fig)
  return path


def plot_solref_fixed_best_projection(results):
  """Projects the solref[0]=0.020 subset onto solimp[0] x solimp[1]."""
  subset = [r for r in results if abs(r["solref0"] - HIGH_SOLREF0) < 1e-9]
  values = np.array([r["reward"] for r in subset])
  vmin, vmax = float(values.min()), float(values.max())

  s0_values = sorted(set(r["solimp0"] for r in subset))
  s1_values = sorted(set(r["solimp1"] for r in subset))

  best_grid = np.full((len(s1_values), len(s0_values)), np.nan)
  mean_grid = np.full_like(best_grid, np.nan)
  robust_grid = np.zeros_like(best_grid)
  best_s2_grid = np.full_like(best_grid, np.nan)

  for i, solimp0 in enumerate(s0_values):
    for j, solimp1 in enumerate(s1_values):
      cell = [
          r
          for r in subset
          if abs(r["solimp0"] - solimp0) < 1e-9
          and abs(r["solimp1"] - solimp1) < 1e-9
      ]
      best = max(cell, key=lambda r: r["reward"])
      best_grid[j, i] = best["reward"]
      best_s2_grid[j, i] = best["solimp2"]
      mean_grid[j, i] = statistics.mean(r["reward"] for r in cell)
      robust_grid[j, i] = sum(r["reward"] >= -16.0 for r in cell)

  fig, ax = plt.subplots(figsize=(7.6, 6.2), constrained_layout=True)
  im = ax.imshow(
      best_grid,
      cmap="RdYlGn",
      vmin=vmin,
      vmax=vmax,
      origin="lower",
      extent=[-0.5, len(s0_values) - 0.5, -0.5, len(s1_values) - 0.5],
  )

  for i in range(len(s0_values)):
    for j in range(len(s1_values)):
      reward = best_grid[j, i]
      text_color = "white" if reward < -28 else "black"
      ax.text(
          i,
          j,
          (
              f"best {reward:.1f}\n"
              f"s2 {best_s2_grid[j, i]:.3f}\n"
              f"good {int(robust_grid[j, i])}/4"
          ),
          ha="center",
          va="center",
          fontsize=8,
          color=text_color,
      )

  ax.set_xticks(range(len(s0_values)), [f"{v:.3f}" for v in s0_values])
  ax.set_yticks(range(len(s1_values)), [f"{v:.3f}" for v in s1_values])
  ax.set_xlabel("solimp[0]")
  ax.set_ylabel("solimp[1]")
  ax.set_title(
      "APG, solref[0]=0.020: best achievable reward over solimp[2]"
  )
  cbar = fig.colorbar(im, ax=ax, shrink=0.86)
  cbar.set_label("best eval/episode_reward over solimp[2]")

  path = os.path.join(FIGURE_DIR, "apg_solref002_best_projection.png")
  fig.savefig(path)
  plt.close(fig)
  return path


def plot_solref_fixed_good_points(results):
  """Shows where the high-reward solref[0]=0.020 points cluster."""
  subset = [r for r in results if abs(r["solref0"] - HIGH_SOLREF0) < 1e-9]
  s2_values = sorted(set(r["solimp2"] for r in subset))
  markers = ["o", "s", "^", "D"]
  offsets = {
      s2: (idx - (len(s2_values) - 1) / 2.0) * 0.022
      for idx, s2 in enumerate(s2_values)
  }
  values = np.array([r["reward"] for r in subset])

  fig, ax = plt.subplots(figsize=(7.6, 6.2), constrained_layout=True)
  for idx, solimp2 in enumerate(s2_values):
    cell = [r for r in subset if abs(r["solimp2"] - solimp2) < 1e-9]
    xs = [r["solimp0"] + offsets[solimp2] for r in cell]
    ys = [r["solimp1"] + offsets[solimp2] for r in cell]
    scatter = ax.scatter(
        xs,
        ys,
        c=[r["reward"] for r in cell],
        cmap="RdYlGn",
        vmin=float(values.min()),
        vmax=float(values.max()),
        marker=markers[idx],
        s=[150 if r["reward"] >= -14.0 else 72 for r in cell],
        edgecolors=["black" if r["reward"] >= -14.0 else "#777777" for r in cell],
        linewidths=[0.9 if r["reward"] >= -14.0 else 0.35 for r in cell],
        alpha=0.95,
        label=f"solimp[2]={solimp2:.3f}",
    )

  good = [r for r in subset if r["reward"] >= -14.0]
  for row in good:
    ax.annotate(
        f"{row['reward']:.1f}",
        (row["solimp0"], row["solimp1"]),
        textcoords="offset points",
        xytext=(0, 9),
        ha="center",
        fontsize=7,
    )

  ax.set_xlabel("solimp[0]")
  ax.set_ylabel("solimp[1]")
  ax.set_title(
      "APG, solref[0]=0.020: high-reward solimp points cluster"
  )
  ax.set_xticks(sorted(set(r["solimp0"] for r in subset)))
  ax.set_yticks(sorted(set(r["solimp1"] for r in subset)))
  ax.grid(alpha=0.25)
  ax.legend(loc="lower right", fontsize=8)
  cbar = fig.colorbar(scatter, ax=ax, shrink=0.86)
  cbar.set_label("eval/episode_reward")

  path = os.path.join(FIGURE_DIR, "apg_solref002_good_points.png")
  fig.savefig(path)
  plt.close(fig)
  return path


def plot_solref_fixed_by_solimp2(results):
  """Saves one clean heatmap per solimp[2] for solref[0]=0.020."""
  subset = [r for r in results if abs(r["solref0"] - HIGH_SOLREF0) < 1e-9]
  values = np.array([r["reward"] for r in subset])
  vmin, vmax = float(values.min()), float(values.max())
  s0_values = sorted(set(r["solimp0"] for r in subset))
  s1_values = sorted(set(r["solimp1"] for r in subset))
  s2_values = sorted(set(r["solimp2"] for r in subset), reverse=True)

  paths = []
  for solimp2 in s2_values:
    grid = np.full((len(s1_values), len(s0_values)), np.nan)
    for i, solimp0 in enumerate(s0_values):
      for j, solimp1 in enumerate(s1_values):
        match = [
            r
            for r in subset
            if abs(r["solimp0"] - solimp0) < 1e-9
            and abs(r["solimp1"] - solimp1) < 1e-9
            and abs(r["solimp2"] - solimp2) < 1e-9
        ]
        grid[j, i] = match[0]["reward"]

    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    im = ax.imshow(
        grid,
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=[-0.5, len(s0_values) - 0.5, -0.5, len(s1_values) - 0.5],
    )
    for i in range(len(s0_values)):
      for j in range(len(s1_values)):
        reward = grid[j, i]
        text_color = "white" if reward < -28 else "black"
        ax.text(
            i,
            j,
            f"{reward:.1f}",
            ha="center",
            va="center",
            fontsize=12,
            color=text_color,
        )

    ax.set_xticks(range(len(s0_values)), [f"{v:.3f}" for v in s0_values])
    ax.set_yticks(range(len(s1_values)), [f"{v:.3f}" for v in s1_values])
    ax.set_xlabel("solimp[0]")
    ax.set_ylabel("solimp[1]")
    ax.set_title(
        f"APG reward, solref=[0.02, 1.0], solimp[2]={solimp2:.3f}"
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("eval/episode_reward")

    path = os.path.join(
        FIGURE_DIR, f"apg_solref002_solimp2_{solimp2:.3f}.png"
    )
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

  fig, axes = plt.subplots(
      2,
      2,
      figsize=(11.8, 9.8),
      sharex=True,
      sharey=True,
      constrained_layout=True,
  )
  for ax, solimp2 in zip(axes.ravel(), s2_values):
    grid = np.full((len(s1_values), len(s0_values)), np.nan)
    for i, solimp0 in enumerate(s0_values):
      for j, solimp1 in enumerate(s1_values):
        match = [
            r
            for r in subset
            if abs(r["solimp0"] - solimp0) < 1e-9
            and abs(r["solimp1"] - solimp1) < 1e-9
            and abs(r["solimp2"] - solimp2) < 1e-9
        ]
        grid[j, i] = match[0]["reward"]

    im = ax.imshow(
        grid,
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=[-0.5, len(s0_values) - 0.5, -0.5, len(s1_values) - 0.5],
    )
    for i in range(len(s0_values)):
      for j in range(len(s1_values)):
        reward = grid[j, i]
        text_color = "white" if reward < -28 else "black"
        ax.text(
            i,
            j,
            f"{reward:.1f}",
            ha="center",
            va="center",
            fontsize=8,
            color=text_color,
        )

    ax.set_title(f"solimp[2]={solimp2:.3f}")
    ax.set_xticks(range(len(s0_values)), [f"{v:.3f}" for v in s0_values])
    ax.set_yticks(range(len(s1_values)), [f"{v:.3f}" for v in s1_values])
    ax.set_xlabel("solimp[0]")
    ax.set_ylabel("solimp[1]")

  cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86)
  cbar.set_label("eval/episode_reward")
  fig.suptitle("APG reward heatmaps at solref=[0.02, 1.0]")
  combined_path = os.path.join(FIGURE_DIR, "apg_solref002_solimp2_4panel.png")
  fig.savefig(combined_path)
  plt.close(fig)
  paths.append(combined_path)
  return paths


def plot_solref_fixed_3d(results):
  """3D view of solimp[0], solimp[1], solimp[2] at solref[0]=0.020."""
  subset = [r for r in results if abs(r["solref0"] - HIGH_SOLREF0) < 1e-9]
  values = np.array([r["reward"] for r in subset])
  best = max(subset, key=lambda r: r["reward"])

  fig = plt.figure(figsize=(8.6, 6.8), constrained_layout=True)
  ax = fig.add_subplot(1, 1, 1, projection="3d")
  scatter = ax.scatter(
      [r["solimp0"] for r in subset],
      [r["solimp1"] for r in subset],
      [r["solimp2"] for r in subset],
      c=[r["reward"] for r in subset],
      cmap="RdYlGn",
      vmin=float(values.min()),
      vmax=float(values.max()),
      s=[105 if r["reward"] >= -14.0 else 58 for r in subset],
      edgecolors=["black" if r["reward"] >= -14.0 else "#777777" for r in subset],
      linewidths=[0.85 if r["reward"] >= -14.0 else 0.35 for r in subset],
      depthshade=True,
  )
  ax.scatter(
      [best["solimp0"]],
      [best["solimp1"]],
      [best["solimp2"]],
      marker="*",
      s=260,
      color="gold",
      edgecolors="black",
      linewidths=1.0,
  )
  ax.text(
      best["solimp0"],
      best["solimp1"],
      best["solimp2"],
      f" best {best['reward']:.2f}",
      fontsize=8,
  )
  ax.set_xlabel("solimp[0]")
  ax.set_ylabel("solimp[1]")
  ax.set_zlabel("solimp[2]")
  ax.set_title("APG solimp sweep, solref=[0.02, 1.0]")
  ax.view_init(elev=24, azim=135)
  cbar = fig.colorbar(scatter, ax=ax, shrink=0.78, pad=0.08)
  cbar.set_label("eval/episode_reward")

  path = os.path.join(FIGURE_DIR, "apg_solref002_solimp_3d.png")
  fig.savefig(path)
  plt.close(fig)
  return path


def plot_top_bottom(results):
  sorted_results = sorted(results, key=lambda r: r["reward"], reverse=True)
  selected = sorted_results[:10] + sorted_results[-10:]
  labels = []
  rewards = []
  colors = []
  for i, row in enumerate(selected):
    labels.append(
        "slot {slot}: a=[{a0:.2f},{a1:.2f},{a2:.2f},{ar0:.0f}]\n"
        "si=[{solimp0:.3f},{solimp1:.3f},{solimp2:.3f}] "
        "sr0={solref0:.3f}".format(**row)
    )
    rewards.append(row["reward"])
    colors.append("#2e7d32" if i < 10 else "#b23b3b")

  fig, ax = plt.subplots(figsize=(12.5, 7.0), constrained_layout=True)
  bars = ax.barh(range(len(selected)), rewards, color=colors)
  ax.set_yticks(range(len(selected)), labels, fontsize=7)
  ax.invert_yaxis()
  ax.set_xlabel("eval/episode_reward")
  ax.set_title("Top 10 and bottom 10 APG sweep configurations")
  ax.grid(axis="x", alpha=0.25)
  for bar, value in zip(bars, rewards):
    ax.text(
        value - 0.45,
        bar.get_y() + bar.get_height() / 2.0,
        f"{value:.1f}",
        ha="right",
        va="center",
        fontsize=8,
        color="white",
    )

  path = os.path.join(FIGURE_DIR, "apg_top_bottom.png")
  fig.savefig(path)
  plt.close(fig)
  return path


def main():
  os.makedirs(FIGURE_DIR, exist_ok=True)
  plt.rcParams.update({
      "font.size": 10,
      "axes.titlesize": 11,
      "axes.labelsize": 10,
      "figure.dpi": 150,
      "savefig.dpi": 300,
      "savefig.bbox": "tight",
  })

  results = load_results()
  if len(results) != 128:
    raise RuntimeError(f"Expected 128 completed APG runs, found {len(results)}")

  rewards = np.array([r["reward"] for r in results])
  print(f"Loaded {len(results)} APG results")
  print(f"Reward range: {rewards.min():.2f} to {rewards.max():.2f}")
  print(f"Reward mean/median: {rewards.mean():.2f} / {np.median(rewards):.2f}")

  paths = [
      save_csv(results),
      plot_heatmap_grid(results),
      plot_solref_delta(results),
      plot_main_effects(results),
      plot_factor_boxplots(results),
      plot_3d_scatter(results),
      plot_solref_fixed_best_projection(results),
      plot_solref_fixed_good_points(results),
      *plot_solref_fixed_by_solimp2(results),
      plot_solref_fixed_3d(results),
      plot_top_bottom(results),
  ]
  print("Saved:")
  for path in paths:
    print(f"  {path}")


if __name__ == "__main__":
  main()
