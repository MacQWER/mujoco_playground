"""Plot APG and Go2Joystick2 PPO sweep surfaces on shared scales."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, LogNorm
from matplotlib.ticker import NullFormatter, NullLocator
from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APG_CSV = (
    REPO_ROOT / "logs" / "go2_sweep" / "figures"
    / "apg_go2_sweep_results.csv"
)
DEFAULT_PPO_CSV = (
    REPO_ROOT / "logs" / "go2_sweep"
    / "go2joystick2_ppo_x64_65eval" / "figures"
    / "ppo_go2_sweep_results.csv"
)
DEFAULT_SOFTNESS_CSV = (
    REPO_ROOT / "logs" / "go2_sweep"
    / "softness_ranking_augmented_mid42.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "logs" / "go2_sweep" / "comparisons"
    / "apg_vs_go2joystick2_ppo_matched_solref_scale_20260806"
)

METRIC = "eval/episode_reward"
FIXED_SOLIMP1 = 0.95
SOLREF0_VALUES = (0.1, 0.02, 0.004)
SURFACE_RESOLUTION = 140


def _float_key(*values):
  return tuple(round(float(value), 12) for value in values)


def _load_results(path):
  with path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

  results = []
  for row in rows:
    parsed = dict(row)
    for field in ("solimp0", "solimp1", "solimp2", "solref0", METRIC):
      parsed[field] = float(row[field])
    if abs(parsed["solimp1"] - FIXED_SOLIMP1) < 1e-9:
      results.append(parsed)
  return results


def _load_penetration(path):
  penetration = {}
  with path.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      key = _float_key(
          row["solimp0"], row["solimp1"], row["solimp2"], row["solref0"]
      )
      penetration[key] = float(row["penetration_m"]) * 1000.0
  return penetration


def _surface_key(row):
  return _float_key(
      row["solimp0"], row["solimp1"], row["solimp2"], row["solref0"]
  )


def _grid_key(row):
  return _float_key(row["solimp0"], row["solimp2"], row["solref0"])


def _validate_inputs(datasets, penetration):
  expected_solrefs = {_float_key(value)[0] for value in SOLREF0_VALUES}
  reference_grid = None

  for name, rows in datasets.items():
    grid = {_grid_key(row) for row in rows}
    if len(grid) != len(rows):
      raise ValueError(f"{name} contains duplicate surface coordinates")
    if {key[2] for key in grid} != expected_solrefs:
      raise ValueError(f"{name} does not contain all three solref[0] groups")
    if reference_grid is None:
      reference_grid = grid
    elif grid != reference_grid:
      raise ValueError("APG and PPO do not use the same surface grid")

    missing = sorted({_surface_key(row) for row in rows} - penetration.keys())
    if missing:
      raise ValueError(f"{name} has {len(missing)} points without calibration")

  if len(reference_grid) != 57:
    raise ValueError(
        f"Expected 57 shared points at solimp[1]=0.95, got {len(reference_grid)}"
    )


def _nice_reward_limits(values):
  raw_min = min(values)
  raw_max = max(values)
  padding = 0.03 * (raw_max - raw_min)
  step = 5.0
  lower = step * math.floor((raw_min - padding) / step)
  upper = step * math.ceil((raw_max + padding) / step)
  return lower, upper


def _interpolate(points, values, grid_x, grid_y):
  interpolator = CloughTocher2DInterpolator(points, values)
  grid = interpolator(grid_x, grid_y)
  if np.isnan(grid).any():
    nearest = NearestNDInterpolator(points, values)
    grid = np.where(np.isnan(grid), nearest(grid_x, grid_y), grid)
  return np.clip(grid, np.min(values), np.max(values))


def _shaded_colors(cmap, norm, penetration_grid, reward_grid):
  facecolors = cmap(norm(penetration_grid))
  light = LightSource(azdeg=320, altdeg=38)
  shade = light.hillshade(reward_grid, vert_exag=0.75, fraction=1.15)
  shade = 0.72 + 0.28 * shade
  facecolors[..., :3] = np.clip(
      facecolors[..., :3] * shade[..., None], 0.0, 1.0
  )
  return facecolors


def _solref_slug(value):
  return f"{value:g}".replace(".", "p")


def _plot_surface(
    rows,
    penetration,
    dataset_slug,
    display_name,
    solref0,
    scales,
    output_dir,
):
  group = [row for row in rows if abs(row["solref0"] - solref0) < 1e-9]
  points = np.array([
      [np.log10(row["solimp0"]), np.log10(row["solimp2"])]
      for row in group
  ])
  rewards = np.array([row[METRIC] for row in group], dtype=float)
  penetrations = np.array(
      [penetration[_surface_key(row)] for row in group], dtype=float
  )

  grid_x, grid_y = scales["grid"]
  reward_grid = _interpolate(points, rewards, grid_x, grid_y)
  log_penetration_grid = _interpolate(
      points, np.log10(penetrations), grid_x, grid_y
  )
  penetration_grid = 10 ** log_penetration_grid

  cmap = plt.get_cmap("coolwarm_r")
  norm = LogNorm(vmin=scales["penetration_min"], vmax=scales["penetration_max"])
  facecolors = _shaded_colors(
      cmap, norm, penetration_grid, reward_grid
  )

  fig = plt.figure(figsize=(10.4, 7.2))
  fig.subplots_adjust(left=0.01, right=0.79, top=0.82, bottom=0.07)
  ax = fig.add_subplot(1, 1, 1, projection="3d")
  ax.plot_surface(
      grid_x,
      grid_y,
      reward_grid,
      facecolors=facecolors,
      rstride=1,
      cstride=1,
      linewidth=0.08,
      edgecolor=(0.12, 0.12, 0.12, 0.12),
      antialiased=True,
      shade=False,
      alpha=0.97,
  )
  ax.scatter(
      points[:, 0],
      points[:, 1],
      rewards,
      c=penetrations,
      cmap=cmap,
      norm=norm,
      s=22,
      edgecolors="#202020",
      linewidths=0.45,
      depthshade=False,
      zorder=5,
  )
  ax.contour(
      grid_x,
      grid_y,
      reward_grid,
      zdir="z",
      offset=scales["reward_min"],
      levels=np.linspace(scales["reward_min"], scales["reward_max"], 10),
      colors="#303030",
      linewidths=0.45,
      alpha=0.62,
  )

  ax.set_xlabel("solimp[0] (log scale)", labelpad=10, fontsize=14)
  ax.set_ylabel("solimp[2] (log scale)", labelpad=14, fontsize=14)
  ax.set_zlabel(METRIC, labelpad=10, fontsize=14)
  ax.set_xlim(scales["x_log_min"], scales["x_log_max"])
  ax.set_ylim(scales["y_log_min"], scales["y_log_max"])
  ax.set_zlim(scales["reward_min"], scales["reward_max"])
  ax.set_xticks(np.log10(scales["x_values"]))
  x_ticklabels = [f"{value:g}" for value in scales["x_values"]]
  x_ticklabels[-2] += "\n"
  x_ticklabels[-1] = "\n" + x_ticklabels[-1]
  ax.set_xticklabels(x_ticklabels)
  ax.set_yticks(np.log10(scales["y_values"]))
  y_ticklabels = [f"{value:g}" for value in scales["y_values"]]
  y_ticklabels[0] = "\n\n" + y_ticklabels[0]
  ax.set_yticklabels(y_ticklabels)
  z_ticks = list(
      np.arange(
          math.ceil(scales["reward_min"] / 10.0) * 10.0,
          math.floor(scales["reward_max"] / 10.0) * 10.0 + 1.0,
          10.0,
      )
  )
  if z_ticks[0] != scales["reward_min"]:
    z_ticks.insert(0, scales["reward_min"])
  if z_ticks[-1] != scales["reward_max"]:
    z_ticks.append(scales["reward_max"])
  ax.set_zticks(z_ticks)
  ax.xaxis.set_tick_params(labelsize=11, pad=1)
  ax.yaxis.set_tick_params(labelsize=11, pad=7)
  ax.zaxis.set_tick_params(labelsize=11, pad=5)
  ax.view_init(elev=27, azim=-52)
  ax.set_box_aspect((1.2, 1.0, 0.62))
  ax.xaxis.set_pane_color((0.96, 0.96, 0.96, 1.0))
  ax.yaxis.set_pane_color((0.96, 0.96, 0.96, 1.0))
  ax.zaxis.set_pane_color((0.985, 0.985, 0.985, 1.0))
  for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis._axinfo["grid"]["color"] = (0.68, 0.68, 0.68, 0.75)
    axis._axinfo["grid"]["linewidth"] = 0.75
  ax.set_title(
      f"{display_name} reward surface\n"
      f"solimp[1]={FIXED_SOLIMP1:g}, solref[0]={solref0:g}",
      fontsize=20,
      pad=20,
      loc="left",
  )

  mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
  mappable.set_array([])
  cbar = fig.colorbar(mappable, ax=ax, shrink=0.66, pad=0.11)
  tick_mm = np.geomspace(
      scales["penetration_min"], scales["penetration_max"], num=5
  )
  cbar.set_ticks(tick_mm)
  cbar.set_ticklabels([f"{value:.1f}" for value in tick_mm])
  cbar.ax.yaxis.set_minor_locator(NullLocator())
  cbar.ax.yaxis.set_minor_formatter(NullFormatter())
  cbar.ax.tick_params(labelsize=11)
  cbar.set_label("penetration (mm); red = harder", fontsize=14)

  path = output_dir / (
      f"{dataset_slug}_matched_solref_scale_surface_solref_"
      f"{_solref_slug(solref0)}.png"
  )
  fig.savefig(path, dpi=300, bbox_inches="tight")
  plt.close(fig)
  return path


def _write_snapshot(path, rows, source_fields):
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=source_fields)
    writer.writeheader()
    writer.writerows(rows)


def _parse_args():
  parser = argparse.ArgumentParser(
      description=(
          "Generate six Go2 APG/PPO reward surfaces with scales shared "
          "within each matching solref group."
      )
  )
  parser.add_argument("--apg_csv", type=Path, default=DEFAULT_APG_CSV)
  parser.add_argument("--ppo_csv", type=Path, default=DEFAULT_PPO_CSV)
  parser.add_argument(
      "--softness_csv", type=Path, default=DEFAULT_SOFTNESS_CSV
  )
  parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Allow replacing files inside the comparison output directory.",
  )
  return parser.parse_args()


def main():
  args = _parse_args()
  datasets = {
      "apg_go2": _load_results(args.apg_csv),
      "ppo_go2joystick2": _load_results(args.ppo_csv),
  }
  penetration = _load_penetration(args.softness_csv)
  _validate_inputs(datasets, penetration)

  all_rows = [row for rows in datasets.values() for row in rows]
  x_values = np.array(sorted({row["solimp0"] for row in all_rows}))
  y_values = np.array(sorted({row["solimp2"] for row in all_rows}))
  x_log_values = np.log10(x_values)
  y_log_values = np.log10(y_values)
  grid_x, grid_y = np.meshgrid(
      np.linspace(x_log_values.min(), x_log_values.max(), SURFACE_RESOLUTION),
      np.linspace(y_log_values.min(), y_log_values.max(), SURFACE_RESOLUTION),
  )
  geometry_scales = {
      "x_values": x_values,
      "y_values": y_values,
      "x_log_min": x_log_values.min(),
      "x_log_max": x_log_values.max(),
      "y_log_min": y_log_values.min(),
      "y_log_max": y_log_values.max(),
      "grid": (grid_x, grid_y),
  }
  scales_by_solref = {}
  for solref0 in SOLREF0_VALUES:
    pair_rows = [
        row for row in all_rows if abs(row["solref0"] - solref0) < 1e-9
    ]
    reward_min, reward_max = _nice_reward_limits(
        [row[METRIC] for row in pair_rows]
    )
    pair_penetrations = [
        penetration[_surface_key(row)] for row in pair_rows
    ]
    scales_by_solref[_float_key(solref0)[0]] = {
        **geometry_scales,
        "reward_min": reward_min,
        "reward_max": reward_max,
        "penetration_min": min(pair_penetrations),
        "penetration_max": max(pair_penetrations),
    }

  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  targets = [
      output_dir / "apg_go2_solimp1_0p95_surface_data.csv",
      output_dir / "ppo_go2joystick2_solimp1_0p95_surface_data.csv",
      output_dir / "matched_solref_scale_metadata.json",
  ]
  for dataset_slug in datasets:
    for solref0 in SOLREF0_VALUES:
      targets.append(
          output_dir / (
              f"{dataset_slug}_matched_solref_scale_surface_solref_"
              f"{_solref_slug(solref0)}.png"
          )
      )
  existing = [path for path in targets if path.exists()]
  if existing and not args.overwrite:
    names = ", ".join(path.name for path in existing)
    raise FileExistsError(
        f"Refusing to overwrite existing comparison files: {names}"
    )

  apg_fields = list(datasets["apg_go2"][0].keys())
  ppo_fields = list(datasets["ppo_go2joystick2"][0].keys())
  _write_snapshot(targets[0], datasets["apg_go2"], apg_fields)
  _write_snapshot(targets[1], datasets["ppo_go2joystick2"], ppo_fields)

  metadata = {
      "metric": METRIC,
      "fixed_solimp1": FIXED_SOLIMP1,
      "scale_scope": "matching_solref_pair",
      "solref0_values": list(SOLREF0_VALUES),
      "source_csvs": {
          "apg_go2": str(args.apg_csv.resolve()),
          "ppo_go2joystick2": str(args.ppo_csv.resolve()),
          "softness_calibration": str(args.softness_csv.resolve()),
      },
      "rows_used": {name: len(rows) for name, rows in datasets.items()},
      "raw_reward_ranges": {
          name: [
              min(row[METRIC] for row in rows),
              max(row[METRIC] for row in rows),
          ]
          for name, rows in datasets.items()
      },
      "common_geometry": {
          "solimp0_x": [float(x_values.min()), float(x_values.max())],
          "solimp2_y": [float(y_values.min()), float(y_values.max())],
          "view": {"elevation": 27, "azimuth": -52},
      },
      "scales_by_solref": {
          f"{solref0:g}": {
              "reward_z": [
                  scales_by_solref[_float_key(solref0)[0]]["reward_min"],
                  scales_by_solref[_float_key(solref0)[0]]["reward_max"],
              ],
              "penetration_color_mm": [
                  scales_by_solref[_float_key(solref0)[0]][
                      "penetration_min"
                  ],
                  scales_by_solref[_float_key(solref0)[0]][
                      "penetration_max"
                  ],
              ],
          }
          for solref0 in SOLREF0_VALUES
      },
      "notes": [
          "APG and PPO share scales only when solref[0] matches.",
          "Different solref[0] groups use independent z and color scales.",
          "All six figures still use identical x/y axes, view, and size.",
          "The APG source is restricted to its solimp[1]=0.95 subset.",
          "Surface color is calibrated penetration; lower means harder contact.",
          "Black-edged markers are measured sweep samples.",
      ],
  }
  with targets[2].open("w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
    f.write(os.linesep)

  plot_specs = [
      ("apg_go2", "APG Go2"),
      ("ppo_go2joystick2", "PPO Go2Joystick2"),
  ]
  paths = []
  for dataset_slug, display_name in plot_specs:
    for solref0 in SOLREF0_VALUES:
      paths.append(
          _plot_surface(
              datasets[dataset_slug],
              penetration,
              dataset_slug,
              display_name,
              solref0,
              scales_by_solref[_float_key(solref0)[0]],
              output_dir,
          )
      )

  print(f"Output directory: {output_dir}")
  for solref0 in SOLREF0_VALUES:
    scales = scales_by_solref[_float_key(solref0)[0]]
    print(
        f"solref[0]={solref0:g}: shared reward z "
        f"[{scales['reward_min']:g}, {scales['reward_max']:g}], "
        f"penetration color [{scales['penetration_min']:.3f}, "
        f"{scales['penetration_max']:.3f}] mm"
    )
  for path in paths:
    print(path)


if __name__ == "__main__":
  main()
