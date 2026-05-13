"""Plot PPO/APG sweep results with shared color scales."""

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "logs" / "sweep" / "figures"

ALGORITHMS = ("apg", "ppo")
HIGH_SOLREF0 = 0.02


def load_results(algo):
  path = FIGURE_DIR / f"{algo}_sweep_results.csv"
  rows = []
  with path.open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      parsed = {}
      for key, value in row.items():
        if key == "slot":
          parsed[key] = int(value)
        elif key in ("path", "run_id", "timestamp"):
          parsed[key] = value
        elif value == "":
          parsed[key] = None
        else:
          parsed[key] = float(value)
      rows.append(parsed)
  if len(rows) != 128:
    raise RuntimeError(f"Expected 128 {algo.upper()} rows, found {len(rows)}")
  return rows


def unique_values(results, key):
  return sorted({row[key] for row in results})


def result_map(results):
  return {
      (row["a0"], row["a1"], row["a2"], row["ar0"]): row["reward"]
      for row in results
  }


def reward_scale(datasets):
  values = [
      row["reward"]
      for results in datasets.values()
      for row in results
  ]
  return min(values), max(values)


def solref_delta_limit(datasets):
  deltas = []
  for results in datasets.values():
    rewards = result_map(results)
    a0_values = unique_values(results, "a0")
    a1_values = unique_values(results, "a1")
    a2_values = unique_values(results, "a2")
    for a0 in a0_values:
      for a1 in a1_values:
        for a2 in a2_values:
          deltas.append(
              rewards[(a0, a1, a2, 1.0)]
              - rewards[(a0, a1, a2, 0.0)]
          )
  return max(abs(min(deltas)), abs(max(deltas)))


def text_color(value, vmin, vmax):
  if vmax <= vmin:
    return "black"
  normalized = (value - vmin) / (vmax - vmin)
  return "white" if normalized < 0.28 else "black"


def plot_heatmap_grid(algo, results, scale):
  rewards = result_map(results)
  vmin, vmax = scale
  a0_values = unique_values(results, "a0")
  a1_values = unique_values(results, "a1")
  a2_values = unique_values(results, "a2")
  ar0_values = unique_values(results, "ar0")

  fig, axes = plt.subplots(
      2,
      4,
      figsize=(15.0, 7.0),
      sharex=True,
      sharey=True,
      constrained_layout=True,
  )

  for row_idx, ar0 in enumerate(ar0_values):
    solref0 = next(r["solref0"] for r in results if abs(r["ar0"] - ar0) < 1e-9)
    for col_idx, a2 in enumerate(a2_values):
      ax = axes[row_idx][col_idx]
      grid = np.full((len(a1_values), len(a0_values)), np.nan)
      for i, a0 in enumerate(a0_values):
        for j, a1 in enumerate(a1_values):
          grid[j, i] = rewards[(a0, a1, a2, ar0)]

      image = ax.imshow(
          grid,
          cmap="RdYlGn",
          vmin=vmin,
          vmax=vmax,
          origin="lower",
          extent=[-0.5, len(a0_values) - 0.5, -0.5, len(a1_values) - 0.5],
      )
      for i in range(len(a0_values)):
        for j in range(len(a1_values)):
          reward = grid[j, i]
          ax.text(
              i,
              j,
              f"{reward:.1f}",
              ha="center",
              va="center",
              fontsize=7,
              color=text_color(reward, vmin, vmax),
          )

      solimp2 = next(
          r["solimp2"] for r in results if abs(r["a2"] - a2) < 1e-9
      )
      ax.set_title(f"solimp[2]={solimp2:.3f}")
      ax.set_xticks(
          range(len(a0_values)),
          [
              f"{next(r['solimp0'] for r in results if abs(r['a0'] - a0) < 1e-9):.3f}"
              for a0 in a0_values
          ],
          fontsize=8,
      )
      ax.set_yticks(
          range(len(a1_values)),
          [
              f"{next(r['solimp1'] for r in results if abs(r['a1'] - a1) < 1e-9):.3f}"
              for a1 in a1_values
          ],
          fontsize=8,
      )
      if row_idx == 1:
        ax.set_xlabel("solimp[0]")
      if col_idx == 0:
        ax.set_ylabel("solimp[1]")

    fig.text(
        0.012,
        0.71 if row_idx == 0 else 0.29,
        f"solref[0]={solref0:.3f}",
        rotation=90,
        va="center",
    )

  colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.92)
  colorbar.set_label("eval/episode_reward, shared PPO/APG scale")
  fig.suptitle(
      f"{algo.upper()} sweep reward, shared PPO/APG color scale "
      f"[{vmin:.1f}, {vmax:.1f}]"
  )

  path = FIGURE_DIR / f"{algo}_heatmap_grid.png"
  fig.savefig(path)
  plt.close(fig)
  return path


def plot_solref_delta(algo, results, limit):
  rewards = result_map(results)
  a0_values = unique_values(results, "a0")
  a1_values = unique_values(results, "a1")
  a2_values = unique_values(results, "a2")

  fig, axes = plt.subplots(
      1,
      4,
      figsize=(15.0, 3.8),
      sharex=True,
      sharey=True,
      constrained_layout=True,
  )

  for col_idx, a2 in enumerate(a2_values):
    ax = axes[col_idx]
    grid = np.full((len(a1_values), len(a0_values)), np.nan)
    for i, a0 in enumerate(a0_values):
      for j, a1 in enumerate(a1_values):
        grid[j, i] = (
            rewards[(a0, a1, a2, 1.0)]
            - rewards[(a0, a1, a2, 0.0)]
        )

    image = ax.imshow(
        grid,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        origin="lower",
        extent=[-0.5, len(a0_values) - 0.5, -0.5, len(a1_values) - 0.5],
    )
    for i in range(len(a0_values)):
      for j in range(len(a1_values)):
        value = grid[j, i]
        ax.text(
            i,
            j,
            f"{value:+.1f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if abs(value) > 0.55 * limit else "black",
        )

    solimp2 = next(
        r["solimp2"] for r in results if abs(r["a2"] - a2) < 1e-9
    )
    ax.set_title(f"solimp[2]={solimp2:.3f}")
    ax.set_xticks(
        range(len(a0_values)),
        [
            f"{next(r['solimp0'] for r in results if abs(r['a0'] - a0) < 1e-9):.3f}"
            for a0 in a0_values
        ],
        fontsize=8,
    )
    ax.set_yticks(
        range(len(a1_values)),
        [
            f"{next(r['solimp1'] for r in results if abs(r['a1'] - a1) < 1e-9):.3f}"
            for a1 in a1_values
        ],
        fontsize=8,
    )
    ax.set_xlabel("solimp[0]")
    if col_idx == 0:
      ax.set_ylabel("solimp[1]")

  colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9)
  colorbar.set_label("solref delta, shared PPO/APG scale")
  fig.suptitle(
      f"{algo.upper()} solref effect: solref[0]=0.020 minus 0.004 "
      f"(shared +/-{limit:.1f})"
  )

  path = FIGURE_DIR / f"{algo}_solref_delta.png"
  fig.savefig(path)
  plt.close(fig)
  return path


def plot_main_effects(algo, results):
  fig, axes = plt.subplots(
      1,
      4,
      figsize=(14.0, 3.8),
      constrained_layout=True,
  )
  params = [
      ("a0", "solimp[0]", "solimp0"),
      ("a1", "solimp[1]", "solimp1"),
      ("a2", "solimp[2]", "solimp2"),
      ("ar0", "solref[0]", "solref0"),
  ]

  for ax, (grid_key, title, value_key) in zip(axes, params):
    levels = unique_values(results, grid_key)
    if grid_key == "ar0":
      levels = list(reversed(levels))
    grouped = [
        [row["reward"] for row in results if abs(row[grid_key] - level) < 1e-9]
        for level in levels
    ]
    bp = ax.boxplot(grouped, patch_artist=True, showmeans=True)
    for patch in bp["boxes"]:
      patch.set_facecolor("#d9e8ff")
      patch.set_edgecolor("#3b5b8a")
    for median in bp["medians"]:
      median.set_color("#1f1f1f")
    ax.plot(
        range(1, len(levels) + 1),
        [statistics.mean(values) for values in grouped],
        color="#c03a2b",
        marker="o",
    )
    labels = [
        f"{next(r[value_key] for r in results if abs(r[grid_key] - level) < 1e-9):.3f}"
        for level in levels
    ]
    ax.set_title(title)
    ax.set_xticks(range(1, len(levels) + 1), labels, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel("eval/episode_reward")

  fig.suptitle(f"{algo.upper()} marginal effects")
  path = FIGURE_DIR / f"{algo}_main_effects.png"
  fig.savefig(path)
  plt.close(fig)
  return path


def plot_factor_boxplots(algo, results):
  """2x2 grouped boxplots with jittered scatter overlay."""
  value_keys = {"a0": "solimp0", "a1": "solimp1", "a2": "solimp2", "ar0": "solref0"}
  specs = [
      ("a0", "solimp[0]"),
      ("a1", "solimp[1]"),
      ("a2", "solimp[2]"),
      ("ar0", "solref[0]"),
  ]
  colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
  rng = np.random.default_rng(7)

  fig, axes = plt.subplots(
      2, 2, figsize=(12.5, 8.0), sharey=True, constrained_layout=True
  )

  for ax, (grid_key, title), color in zip(axes.ravel(), specs, colors):
    levels = unique_values(results, grid_key)
    if grid_key == "ar0":
      levels = list(reversed(levels))
    grouped = [
        [row["reward"] for row in results if abs(row[grid_key] - level) < 1e-9]
        for level in levels
    ]
    data = [vals for vals in grouped if vals]
    positions = [i + 1 for i, vals in enumerate(grouped) if vals]

    # Build tick labels: alpha value + physical value + soft/hard annotation
    tick_labels = []
    for i, level in enumerate(levels):
      match = next(r for r in results if abs(r[grid_key] - level) < 1e-9)
      phys = match[value_keys[grid_key]]
      is_first = i == 0
      is_last = i == len(levels) - 1
      label = f"{phys:.3f}"
      if is_first:
        label += "\nsoft"
      elif is_last:
        label += "\nstiff" if grid_key == "ar0" else "\nhard"
      tick_labels.append(label)
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
  fig.suptitle(
      f"{algo.upper()} PushBox sweep — factor main effects / grouped boxplots"
  )
  p = FIGURE_DIR / f"{algo}_factor_boxplots.png"
  fig.savefig(p)
  plt.close(fig)
  return p


def plot_3d_scatter(algo, results, scale):
  vmin, vmax = scale
  ar0_values = unique_values(results, "ar0")
  best = max(results, key=lambda row: row["reward"])

  fig = plt.figure(figsize=(13.0, 5.5), constrained_layout=True)
  scatter = None
  for idx, ar0 in enumerate(ar0_values):
    ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
    subset = [row for row in results if abs(row["ar0"] - ar0) < 1e-9]
    scatter = ax.scatter(
        [row["solimp0"] for row in subset],
        [row["solimp1"] for row in subset],
        [row["solimp2"] for row in subset],
        c=[row["reward"] for row in subset],
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
    solref0 = subset[0]["solref0"]
    ax.set_title(f"solref[0]={solref0:.3f}")
    ax.set_xlabel("solimp[0]")
    ax.set_ylabel("solimp[1]")
    ax.set_zlabel("solimp[2]")
    ax.view_init(elev=23, azim=130)

  colorbar = fig.colorbar(scatter, ax=fig.axes, shrink=0.8, pad=0.08)
  colorbar.set_label("eval/episode_reward, shared PPO/APG scale")
  fig.suptitle(f"{algo.upper()} 3D lattice view, shared color scale")

  path = FIGURE_DIR / f"{algo}_3d_scatter.png"
  fig.savefig(path)
  plt.close(fig)
  return path


def plot_solref_fixed_by_solimp2(algo, results, scale):
  vmin, vmax = scale
  subset = [row for row in results if abs(row["solref0"] - HIGH_SOLREF0) < 1e-9]
  s0_values = unique_values(subset, "solimp0")
  s1_values = unique_values(subset, "solimp1")
  s2_values = sorted(unique_values(subset, "solimp2"), reverse=True)

  paths = []
  for solimp2 in s2_values:
    grid = np.full((len(s1_values), len(s0_values)), np.nan)
    for i, solimp0 in enumerate(s0_values):
      for j, solimp1 in enumerate(s1_values):
        match = next(
            row
            for row in subset
            if abs(row["solimp0"] - solimp0) < 1e-9
            and abs(row["solimp1"] - solimp1) < 1e-9
            and abs(row["solimp2"] - solimp2) < 1e-9
        )
        grid[j, i] = match["reward"]

    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    image = ax.imshow(
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
        ax.text(
            i,
            j,
            f"{reward:.1f}",
            ha="center",
            va="center",
            fontsize=12,
            color=text_color(reward, vmin, vmax),
        )

    ax.set_xticks(range(len(s0_values)), [f"{value:.3f}" for value in s0_values])
    ax.set_yticks(range(len(s1_values)), [f"{value:.3f}" for value in s1_values])
    ax.set_xlabel("solimp[0]")
    ax.set_ylabel("solimp[1]")
    ax.set_title(
        f"{algo.upper()} reward, solref=[0.02, 1.0], "
        f"solimp[2]={solimp2:.3f}"
    )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.86)
    colorbar.set_label("eval/episode_reward, shared PPO/APG scale")

    path = FIGURE_DIR / f"{algo}_solref002_solimp2_{solimp2:.3f}.png"
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
        match = next(
            row
            for row in subset
            if abs(row["solimp0"] - solimp0) < 1e-9
            and abs(row["solimp1"] - solimp1) < 1e-9
            and abs(row["solimp2"] - solimp2) < 1e-9
        )
        grid[j, i] = match["reward"]

    image = ax.imshow(
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
        ax.text(
            i,
            j,
            f"{reward:.1f}",
            ha="center",
            va="center",
            fontsize=8,
            color=text_color(reward, vmin, vmax),
        )

    ax.set_title(f"solimp[2]={solimp2:.3f}")
    ax.set_xticks(range(len(s0_values)), [f"{value:.3f}" for value in s0_values])
    ax.set_yticks(range(len(s1_values)), [f"{value:.3f}" for value in s1_values])
    ax.set_xlabel("solimp[0]")
    ax.set_ylabel("solimp[1]")

  colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86)
  colorbar.set_label("eval/episode_reward, shared PPO/APG scale")
  fig.suptitle(
      f"{algo.upper()} reward at solref=[0.02, 1.0], "
      f"shared color scale [{vmin:.1f}, {vmax:.1f}]"
  )

  path = FIGURE_DIR / f"{algo}_solref002_solimp2_4panel.png"
  fig.savefig(path)
  plt.close(fig)
  paths.append(path)
  return paths


def plot_solref_fixed_3d(algo, results, scale):
  vmin, vmax = scale
  subset = [row for row in results if abs(row["solref0"] - HIGH_SOLREF0) < 1e-9]
  best = max(subset, key=lambda row: row["reward"])

  fig = plt.figure(figsize=(8.6, 6.8), constrained_layout=True)
  ax = fig.add_subplot(1, 1, 1, projection="3d")
  scatter = ax.scatter(
      [row["solimp0"] for row in subset],
      [row["solimp1"] for row in subset],
      [row["solimp2"] for row in subset],
      c=[row["reward"] for row in subset],
      cmap="RdYlGn",
      vmin=vmin,
      vmax=vmax,
      s=68,
      edgecolors="black",
      linewidths=0.45,
      depthshade=False,
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
  ax.set_title(
      f"{algo.upper()} solimp sweep, solref=[0.02, 1.0], shared color scale"
  )
  ax.view_init(elev=24, azim=135)
  colorbar = fig.colorbar(scatter, ax=ax, shrink=0.78, pad=0.08)
  colorbar.set_label("eval/episode_reward, shared PPO/APG scale")

  path = FIGURE_DIR / f"{algo}_solref002_solimp_3d.png"
  fig.savefig(path)
  plt.close(fig)
  return path


def plot_top_bottom(algo, results):
  sorted_results = sorted(results, key=lambda row: row["reward"], reverse=True)
  selected = sorted_results[:10] + sorted_results[-10:]
  labels = []
  rewards = []
  colors = []
  for idx, row in enumerate(selected):
    labels.append(
        "slot {slot}: si=[{solimp0:.3f},{solimp1:.3f},{solimp2:.3f}] "
        "sr0={solref0:.3f}".format(**row)
    )
    rewards.append(row["reward"])
    colors.append("#2e7d32" if idx < 10 else "#b23b3b")

  fig, ax = plt.subplots(figsize=(12.5, 7.0), constrained_layout=True)
  bars = ax.barh(range(len(selected)), rewards, color=colors)
  ax.set_yticks(range(len(selected)), labels, fontsize=7)
  ax.invert_yaxis()
  ax.set_xlabel("eval/episode_reward")
  ax.set_title(f"Top 10 and bottom 10 {algo.upper()} sweep configurations")
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

  path = FIGURE_DIR / f"{algo}_top_bottom.png"
  fig.savefig(path)
  plt.close(fig)
  return path


def main():
  FIGURE_DIR.mkdir(parents=True, exist_ok=True)
  plt.rcParams.update({
      "font.size": 10,
      "axes.titlesize": 11,
      "axes.labelsize": 10,
      "figure.dpi": 150,
      "savefig.dpi": 300,
      "savefig.bbox": "tight",
  })

  datasets = {algo: load_results(algo) for algo in ALGORITHMS}
  shared_reward_scale = reward_scale(datasets)
  shared_delta_limit = solref_delta_limit(datasets)

  print(
      "Shared reward scale: "
      f"{shared_reward_scale[0]:.2f} to {shared_reward_scale[1]:.2f}"
  )
  print(f"Shared solref delta scale: +/-{shared_delta_limit:.2f}")

  paths = []
  for algo, results in datasets.items():
    paths.extend([
        plot_heatmap_grid(algo, results, shared_reward_scale),
        plot_solref_delta(algo, results, shared_delta_limit),
        plot_main_effects(algo, results),
        plot_factor_boxplots(algo, results),
        plot_3d_scatter(algo, results, shared_reward_scale),
        *plot_solref_fixed_by_solimp2(algo, results, shared_reward_scale),
        plot_solref_fixed_3d(algo, results, shared_reward_scale),
        plot_top_bottom(algo, results),
    ])

  print("Saved:")
  for path in paths:
    print(f"  {path}")


if __name__ == "__main__":
  main()
