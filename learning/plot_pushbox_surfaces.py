"""Plot PushBox reward surfaces with ball-drop hardness calibration."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.colors import to_rgb

from learning.ball_drop_softness import max_penetration

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "logs" / "sweep" / "figures"
PENETRATION_CSV = FIGURE_DIR / "pushbox_ball_drop_penetration.csv"

ALGORITHMS = ("apg", "ppo")
BASE_COLORS = [
    "#b2182b",
    "#d6604d",
    "#f4a582",
    "#fdae61",
    "#abdda4",
    "#66c2a5",
    "#3288bd",
    "#5e4fa2",
]


def _key_from_values(solimp0, solimp1, solimp2, solref0):
  return (
      round(float(solimp0), 12),
      round(float(solimp1), 12),
      round(float(solimp2), 12),
      round(float(solref0), 12),
  )


def _key(row):
  return _key_from_values(
      row["solimp0"], row["solimp1"], row["solimp2"], row["solref0"]
  )


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


def load_penetration_cache():
  if not PENETRATION_CSV.exists():
    return {}
  values = {}
  with PENETRATION_CSV.open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      key = _key_from_values(
          row["solimp0"], row["solimp1"], row["solimp2"], row["solref0"]
      )
      values[key] = float(row["penetration_m"])
  return values


def write_penetration_cache(values):
  FIGURE_DIR.mkdir(parents=True, exist_ok=True)
  with PENETRATION_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "solimp0",
        "solimp1",
        "solimp2",
        "solref0",
        "penetration_m",
        "penetration_mm",
    ])
    for key in sorted(values):
      pen = values[key]
      writer.writerow([*key, f"{pen:.12g}", f"{pen * 1000.0:.12g}"])


def calibrate_penetration(datasets, force=False):
  required = sorted({_key(row) for rows in datasets.values() for row in rows})
  values = {} if force else load_penetration_cache()
  missing = [key for key in required if key not in values]

  if missing:
    print(f"Calibrating {len(missing)} PushBox parameter combinations...")
    for index, key in enumerate(missing, 1):
      values[key] = max_penetration(*key)
      if index % 16 == 0 or index == len(missing):
        print(f"  ball-drop {index}/{len(missing)}")
    write_penetration_cache(values)
    print(f"Saved calibrated penetration: {PENETRATION_CSV}")

  return {key: values[key] for key in required}


def smooth_surface(points, values, grid_x, grid_y, vmin=None, vmax=None):
  from scipy.interpolate import CloughTocher2DInterpolator
  from scipy.interpolate import NearestNDInterpolator

  interp = CloughTocher2DInterpolator(points, values)
  grid = interp(grid_x, grid_y)
  if np.isnan(grid).any():
    nearest = NearestNDInterpolator(points, values)
    grid = np.where(np.isnan(grid), nearest(grid_x, grid_y), grid)
  if vmin is not None and vmax is not None:
    grid = np.clip(grid, vmin, vmax)
  return grid


def hillshade(facecolors, height_grid):
  from matplotlib.colors import LightSource

  light = LightSource(azdeg=320, altdeg=38)
  shade = light.hillshade(height_grid, vert_exag=0.65, fraction=1.05)
  shade = 0.82 + 0.18 * shade
  shaded = facecolors.copy()
  shaded[..., :3] = np.clip(facecolors[..., :3] * shade[..., None], 0.0, 1.0)
  return shaded


def slice_facecolors(base_color, penetration_grid, pen_min, pen_max, height_grid):
  base = np.array(to_rgb(base_color), dtype=float)
  softness = (penetration_grid - pen_min) / max(pen_max - pen_min, 1e-12)
  softness = np.clip(softness, 0.0, 1.0)

  rgb = (
      base[None, None, :] * (0.42 + 0.28 * softness[..., None])
      + np.ones(3)[None, None, :] * (0.35 * softness[..., None])
  )
  rgba = np.concatenate(
      [np.clip(rgb, 0.0, 1.0), np.full((*rgb.shape[:2], 1), 0.96)],
      axis=-1,
  )
  return hillshade(rgba, height_grid)


def style_3d_axis(ax, tick_size=8.0, label_size=9.0):
  ax.set_proj_type("ortho")
  ax.view_init(elev=25, azim=-58)
  try:
    ax.set_box_aspect((1.18, 1.0, 0.58), zoom=0.88)
  except TypeError:
    ax.set_box_aspect((1.18, 1.0, 0.58))
  ax.xaxis.set_pane_color((0.965, 0.965, 0.965, 1.0))
  ax.yaxis.set_pane_color((0.965, 0.965, 0.965, 1.0))
  ax.zaxis.set_pane_color((0.985, 0.985, 0.985, 1.0))
  for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis._axinfo["grid"]["color"] = (0.68, 0.68, 0.68, 0.58)
    axis._axinfo["grid"]["linewidth"] = 0.6
  ax.tick_params(axis="both", which="major", labelsize=tick_size, pad=3)
  ax.zaxis.set_tick_params(labelsize=tick_size, pad=3)
  for label in (
      *ax.xaxis.get_ticklabels(),
      *ax.yaxis.get_ticklabels(),
      *ax.zaxis.get_ticklabels(),
  ):
    label.set_color("#111111")
    label.set_bbox({
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.78,
        "pad": 0.5,
    })
  for label in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
    label.set_color("#111111")
    label.set_fontweight("semibold")
    label.set_fontsize(label_size)


def param_label(value):
  return f"{value:.3g}"


def param_name(value):
  return f"{value:g}".replace("-", "m").replace(".", "p")


def style_parameter_axis(ax, tick_size=9.5, label_size=11.5, zoom=0.96):
  ax.set_proj_type("ortho")
  ax.view_init(elev=24, azim=-52)
  try:
    ax.set_box_aspect((1.08, 1.0, 0.76), zoom=zoom)
  except TypeError:
    ax.set_box_aspect((1.08, 1.0, 0.76))
  ax.xaxis.set_pane_color((0.955, 0.955, 0.955, 1.0))
  ax.yaxis.set_pane_color((0.955, 0.955, 0.955, 1.0))
  ax.zaxis.set_pane_color((0.985, 0.985, 0.985, 1.0))
  for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis._axinfo["grid"]["color"] = (0.58, 0.58, 0.58, 0.46)
    axis._axinfo["grid"]["linewidth"] = 0.7
  ax.tick_params(axis="both", which="major", labelsize=tick_size, pad=5)
  ax.zaxis.set_tick_params(labelsize=tick_size, pad=5)
  for label in (
      *ax.xaxis.get_ticklabels(),
      *ax.yaxis.get_ticklabels(),
      *ax.zaxis.get_ticklabels(),
  ):
    label.set_color("#101010")
    label.set_bbox({
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.82,
        "pad": 0.45,
    })
  for label in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
    label.set_color("#101010")
    label.set_fontsize(label_size)
    label.set_fontweight("semibold")


def style_voxel_axis(ax, tick_size=9.5, label_size=11.5, zoom=0.92):
  ax.set_proj_type("ortho")
  ax.view_init(elev=24, azim=-48)
  try:
    ax.set_box_aspect((1.0, 1.0, 1.0), zoom=zoom)
  except TypeError:
    ax.set_box_aspect((1.0, 1.0, 1.0))
  ax.xaxis.set_pane_color((0.96, 0.96, 0.96, 1.0))
  ax.yaxis.set_pane_color((0.96, 0.96, 0.96, 1.0))
  ax.zaxis.set_pane_color((0.985, 0.985, 0.985, 1.0))
  for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis._axinfo["grid"]["color"] = (0.55, 0.55, 0.55, 0.46)
    axis._axinfo["grid"]["linewidth"] = 0.75
  ax.tick_params(axis="both", which="major", labelsize=tick_size, pad=4)
  ax.zaxis.set_tick_params(labelsize=tick_size, pad=4)
  for label in (
      *ax.xaxis.get_ticklabels(),
      *ax.yaxis.get_ticklabels(),
      *ax.zaxis.get_ticklabels(),
  ):
    label.set_color("#101010")
    label.set_bbox({
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.82,
        "pad": 0.45,
    })
  for label in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
    label.set_color("#101010")
    label.set_fontsize(label_size)
    label.set_fontweight("semibold")


def build_slice_colors(results, penetration):
  slice_keys = sorted({
      (row["solref0"], row["solimp1"])
      for row in results
  })
  means = {}
  for slice_key in slice_keys:
    sr0, s1 = slice_key
    values = [
        penetration[_key(row)]
        for row in results
        if abs(row["solref0"] - sr0) < 1e-12
        and abs(row["solimp1"] - s1) < 1e-12
    ]
    means[slice_key] = float(np.mean(values))

  ordered = sorted(slice_keys, key=lambda item: means[item], reverse=True)
  return {
      slice_key: BASE_COLORS[index]
      for index, slice_key in enumerate(ordered)
  }, means


def draw_reward_parameter_volume(
    ax,
    results,
    sr0,
    reward_norm,
    cmap,
    tick_size=9.5,
    label_size=11.5,
    zoom=0.96,
):
  """Draw reward as stacked heatmap slices in solimp parameter space."""
  solimp0_values = sorted({row["solimp0"] for row in results})
  solimp1_values = sorted({row["solimp1"] for row in results})
  solimp2_values = sorted({row["solimp2"] for row in results})

  grid_x_values = np.linspace(min(solimp0_values), max(solimp0_values), 120)
  grid_y_values = np.linspace(min(solimp1_values), max(solimp1_values), 120)
  grid_x, grid_y = np.meshgrid(grid_x_values, grid_y_values)

  for s2 in solimp2_values:
    plane = [
        row
        for row in results
        if abs(row["solref0"] - sr0) < 1e-12
        and abs(row["solimp2"] - s2) < 1e-12
    ]
    points = np.array([
        [row["solimp0"], row["solimp1"]]
        for row in plane
    ])
    rewards = np.array([row["reward"] for row in plane], dtype=float)
    reward_grid = smooth_surface(
        points,
        rewards,
        grid_x,
        grid_y,
        reward_norm.vmin,
        reward_norm.vmax,
    )
    facecolors = cmap(reward_norm(reward_grid))
    facecolors[..., 3] = 0.82
    z_grid = np.full_like(grid_x, s2, dtype=float)

    ax.plot_surface(
        grid_x,
        grid_y,
        z_grid,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0.035,
        edgecolor=(0.05, 0.05, 0.05, 0.13),
        antialiased=True,
        shade=False,
    )
    ax.contour(
        grid_x,
        grid_y,
        reward_grid,
        zdir="z",
        offset=s2,
        levels=6,
        colors="#242424",
        linewidths=0.35,
        alpha=0.42,
    )

  ax.set_xlim(min(solimp0_values), max(solimp0_values))
  ax.set_ylim(min(solimp1_values), max(solimp1_values))
  ax.set_zlim(min(solimp2_values), max(solimp2_values))
  ax.set_xticks(solimp0_values)
  ax.set_yticks(solimp1_values)
  ax.set_zticks(solimp2_values)
  ax.set_xticklabels([param_label(value) for value in solimp0_values])
  ax.set_yticklabels([
      f"\n{param_label(value)}" if index == 0 else param_label(value)
      for index, value in enumerate(solimp1_values)
  ])
  ax.set_zticklabels([param_label(value) for value in solimp2_values])
  ax.set_xlabel("solimp[0]", labelpad=17)
  ax.set_ylabel("solimp[1]", labelpad=17)
  ax.set_zlabel("solimp[2]", labelpad=13)
  style_parameter_axis(
      ax,
      tick_size=tick_size,
      label_size=label_size,
      zoom=zoom,
  )


def reward_scale_from_datasets(datasets):
  rewards = np.array([
      row["reward"]
      for rows in datasets.values()
      for row in rows
  ], dtype=float)
  return float(rewards.min()), float(rewards.max())


def add_reward_colorbar(fig, reward_norm, cmap, position):
  mappable = plt.cm.ScalarMappable(norm=reward_norm, cmap=cmap)
  mappable.set_array([])
  cax = fig.add_axes(position)
  colorbar = fig.colorbar(mappable, cax=cax)
  colorbar.set_label("reward", fontsize=11)
  colorbar.ax.tick_params(labelsize=9)
  colorbar.ax.set_title("high", fontsize=9, pad=5)
  colorbar.ax.text(
      0.5,
      -0.035,
      "low",
      transform=colorbar.ax.transAxes,
      ha="center",
      va="top",
      fontsize=9,
  )


def draw_reward_voxels(
    ax,
    results,
    sr0,
    reward_norm,
    cmap,
    alpha=0.28,
    cube_size=0.66,
    tick_size=9.5,
    label_size=11.5,
    zoom=0.92,
):
  """Draw each sweep sample as a translucent cube in solimp parameter space."""
  solimp0_values = sorted({row["solimp0"] for row in results})
  solimp1_values = sorted({row["solimp1"] for row in results})
  solimp2_values = sorted({row["solimp2"] for row in results})
  x_index = {value: index for index, value in enumerate(solimp0_values)}
  y_index = {value: index for index, value in enumerate(solimp1_values)}
  z_index = {value: index for index, value in enumerate(solimp2_values)}

  group = [
      row
      for row in results
      if abs(row["solref0"] - sr0) < 1e-12
  ]
  half = cube_size / 2.0
  xs = np.array([x_index[row["solimp0"]] - half for row in group], dtype=float)
  ys = np.array([y_index[row["solimp1"]] - half for row in group], dtype=float)
  zs = np.array([z_index[row["solimp2"]] - half for row in group], dtype=float)
  rewards = np.array([row["reward"] for row in group], dtype=float)
  colors = cmap(reward_norm(rewards))
  colors[:, 3] = alpha

  order = np.argsort(rewards)
  ax.bar3d(
      xs[order],
      ys[order],
      zs[order],
      cube_size,
      cube_size,
      cube_size,
      color=colors[order],
      edgecolor=(0.05, 0.05, 0.05, 0.35),
      linewidth=0.42,
      shade=True,
      zsort="average",
  )

  ax.set_xlim(-0.65, len(solimp0_values) - 0.35)
  ax.set_ylim(-0.65, len(solimp1_values) - 0.35)
  ax.set_zlim(-0.65, len(solimp2_values) - 0.35)
  ax.set_xticks(range(len(solimp0_values)))
  ax.set_yticks(range(len(solimp1_values)))
  ax.set_zticks(range(len(solimp2_values)))
  ax.set_xticklabels([param_label(value) for value in solimp0_values])
  ax.set_yticklabels([
      f"\n{param_label(value)}" if index == 0 else param_label(value)
      for index, value in enumerate(solimp1_values)
  ])
  ax.set_zticklabels([param_label(value) for value in solimp2_values])
  ax.set_xlabel("solimp[0]", labelpad=16)
  ax.set_ylabel("solimp[1]", labelpad=16)
  ax.set_zlabel("solimp[2]", labelpad=12)
  style_voxel_axis(
      ax,
      tick_size=tick_size,
      label_size=label_size,
      zoom=zoom,
  )


def reward_lattice(results, sr0, cells_per_axis):
  from scipy.interpolate import RegularGridInterpolator

  solimp0_values = sorted({row["solimp0"] for row in results})
  solimp1_values = sorted({row["solimp1"] for row in results})
  solimp2_values = sorted({row["solimp2"] for row in results})
  x_index = {value: index for index, value in enumerate(solimp0_values)}
  y_index = {value: index for index, value in enumerate(solimp1_values)}
  z_index = {value: index for index, value in enumerate(solimp2_values)}

  lattice = np.full(
      (len(solimp0_values), len(solimp1_values), len(solimp2_values)),
      np.nan,
      dtype=float,
  )
  for row in results:
    if abs(row["solref0"] - sr0) > 1e-12:
      continue
    lattice[
        x_index[row["solimp0"]],
        y_index[row["solimp1"]],
        z_index[row["solimp2"]],
    ] = row["reward"]

  if np.isnan(lattice).any():
    raise RuntimeError(f"Missing reward values for solref[0]={sr0}")

  axes = (
      np.arange(len(solimp0_values), dtype=float),
      np.arange(len(solimp1_values), dtype=float),
      np.arange(len(solimp2_values), dtype=float),
  )
  interpolator = RegularGridInterpolator(axes, lattice)
  edges = np.linspace(-0.5, len(solimp0_values) - 0.5, cells_per_axis + 1)
  centers = 0.5 * (edges[:-1] + edges[1:])
  sample_points = np.clip(centers, 0.0, len(solimp0_values) - 1.0)
  sample_x, sample_y, sample_z = np.meshgrid(
      sample_points,
      sample_points,
      sample_points,
      indexing="ij",
  )
  points = np.column_stack([
      sample_x.ravel(),
      sample_y.ravel(),
      sample_z.ravel(),
  ])
  values = interpolator(points).reshape(
      cells_per_axis,
      cells_per_axis,
      cells_per_axis,
  )
  return edges, values, solimp0_values, solimp1_values, solimp2_values


def draw_cube_outline(ax, low, high):
  corners = [
      (low, low, low),
      (high, low, low),
      (high, high, low),
      (low, high, low),
      (low, low, high),
      (high, low, high),
      (high, high, high),
      (low, high, high),
  ]
  edges = [
      (0, 1),
      (1, 2),
      (2, 3),
      (3, 0),
      (4, 5),
      (5, 6),
      (6, 7),
      (7, 4),
      (0, 4),
      (1, 5),
      (2, 6),
      (3, 7),
  ]
  for start, end in edges:
    xs = [corners[start][0], corners[end][0]]
    ys = [corners[start][1], corners[end][1]]
    zs = [corners[start][2], corners[end][2]]
    ax.plot(xs, ys, zs, color=(0.05, 0.05, 0.05, 0.42), linewidth=1.1)


def draw_reward_cube(
    ax,
    results,
    sr0,
    reward_norm,
    cmap,
    cells_per_axis=12,
    alpha=0.12,
    tick_size=9.5,
    label_size=11.5,
    zoom=0.92,
):
  """Draw a continuous translucent cube interpolated from the sweep samples."""
  (
      edges,
      rewards,
      solimp0_values,
      solimp1_values,
      solimp2_values,
  ) = reward_lattice(results, sr0, cells_per_axis)

  lower_edges = edges[:-1]
  step = float(edges[1] - edges[0])
  xs, ys, zs = np.meshgrid(
      lower_edges,
      lower_edges,
      lower_edges,
      indexing="ij",
  )
  colors = cmap(reward_norm(rewards.ravel()))
  colors[:, 3] = alpha

  ax.bar3d(
      xs.ravel(),
      ys.ravel(),
      zs.ravel(),
      step,
      step,
      step,
      color=colors,
      edgecolor=(0.0, 0.0, 0.0, 0.0),
      linewidth=0.0,
      shade=False,
      zsort="average",
  )
  draw_cube_outline(ax, edges[0], edges[-1])

  ticks = range(len(solimp0_values))
  ax.set_xlim(edges[0], edges[-1])
  ax.set_ylim(edges[0], edges[-1])
  ax.set_zlim(edges[0], edges[-1])
  ax.set_xticks(ticks)
  ax.set_yticks(ticks)
  ax.set_zticks(ticks)
  ax.set_xticklabels([param_label(value) for value in solimp0_values])
  ax.set_yticklabels([
      f"\n{param_label(value)}" if index == 0 else param_label(value)
      for index, value in enumerate(solimp1_values)
  ])
  ax.set_zticklabels([param_label(value) for value in solimp2_values])
  ax.set_xlabel("solimp[0]", labelpad=16)
  ax.set_ylabel("solimp[1]", labelpad=16)
  ax.set_zlabel("solimp[2]", labelpad=12)
  style_voxel_axis(
      ax,
      tick_size=tick_size,
      label_size=label_size,
      zoom=zoom,
  )


def plot_reward_cube_by_algo(algo, results, reward_scale, cells_per_axis):
  solref_values = sorted({row["solref0"] for row in results})
  reward_norm = Normalize(vmin=reward_scale[0], vmax=reward_scale[1])
  cmap = plt.get_cmap("coolwarm")

  fig = plt.figure(figsize=(14.6, 7.8))
  fig.subplots_adjust(left=0.02, right=0.87, bottom=0.08, top=0.86,
                      wspace=0.0)
  for index, sr0 in enumerate(solref_values, 1):
    ax = fig.add_subplot(1, len(solref_values), index, projection="3d")
    draw_reward_cube(
        ax,
        results,
        sr0,
        reward_norm,
        cmap,
        cells_per_axis=cells_per_axis,
        alpha=0.12,
        tick_size=8.2,
        label_size=10.2,
        zoom=0.78,
    )
    ax.set_title(f"solref[0]={sr0:.3f}", fontsize=15, pad=12)

  fig.suptitle(
      f"{algo.upper()} PushBox reward field in solimp parameter space",
      fontsize=17,
  )
  fig.text(
      0.5,
      0.025,
      "x=solimp[0], y=solimp[1], z=solimp[2]; the transparent cube is a "
      "trilinear interpolation of reward.",
      ha="center",
      fontsize=11,
      color="#333333",
  )
  add_reward_colorbar(fig, reward_norm, cmap, [0.905, 0.21, 0.020, 0.54])

  path = FIGURE_DIR / f"{algo}_pushbox_solimp_parameter_reward_cube.png"
  fig.savefig(path, dpi=320)
  plt.close(fig)
  return path


def plot_reward_cube_single_panels(algo, results, reward_scale, cells_per_axis):
  solref_values = sorted({row["solref0"] for row in results})
  reward_norm = Normalize(vmin=reward_scale[0], vmax=reward_scale[1])
  cmap = plt.get_cmap("coolwarm")
  paths = []

  for sr0 in solref_values:
    fig = plt.figure(figsize=(8.6, 7.5))
    fig.subplots_adjust(left=0.02, right=0.84, bottom=0.08, top=0.86)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    draw_reward_cube(
        ax,
        results,
        sr0,
        reward_norm,
        cmap,
        cells_per_axis=cells_per_axis,
        alpha=0.12,
    )
    ax.set_title(
        f"{algo.upper()} PushBox, solref[0]={sr0:.3f}",
        fontsize=15,
        pad=12,
    )
    fig.text(
        0.45,
        0.028,
        "x=solimp[0], y=solimp[1], z=solimp[2]; color=reward.",
        ha="center",
        fontsize=10,
        color="#333333",
    )
    add_reward_colorbar(fig, reward_norm, cmap, [0.88, 0.22, 0.030, 0.52])

    path = FIGURE_DIR / (
        f"{algo}_pushbox_solimp_parameter_reward_cube_"
        f"solref_{param_name(sr0)}.png"
    )
    fig.savefig(path, dpi=320)
    plt.close(fig)
    paths.append(path)

  return paths


def plot_reward_cube_combined(datasets, reward_scale, cells_per_axis):
  reward_norm = Normalize(vmin=reward_scale[0], vmax=reward_scale[1])
  cmap = plt.get_cmap("coolwarm")

  fig = plt.figure(figsize=(16.2, 12.0))
  fig.subplots_adjust(left=0.02, right=0.87, bottom=0.06, top=0.90,
                      wspace=0.0, hspace=0.08)
  for row_idx, (algo, results) in enumerate(datasets.items()):
    solref_values = sorted({row["solref0"] for row in results})
    for col_idx, sr0 in enumerate(solref_values):
      index = row_idx * len(solref_values) + col_idx + 1
      ax = fig.add_subplot(len(datasets), len(solref_values), index,
                           projection="3d")
      draw_reward_cube(
          ax,
          results,
          sr0,
          reward_norm,
          cmap,
          cells_per_axis=cells_per_axis,
          alpha=0.12,
          tick_size=7.2,
          label_size=8.8,
          zoom=0.70,
      )
      ax.set_title(
          f"{algo.upper()}, solref[0]={sr0:.3f}",
          fontsize=14,
          pad=10,
      )

  fig.suptitle(
      "PushBox reward field in solimp parameter space",
      fontsize=18,
  )
  fig.text(
      0.5,
      0.025,
      "Shared color scale across all panels: red is higher reward and blue is "
      "lower reward.",
      ha="center",
      fontsize=11,
      color="#333333",
  )
  add_reward_colorbar(fig, reward_norm, cmap, [0.905, 0.22, 0.020, 0.55])

  path = FIGURE_DIR / "pushbox_solimp_parameter_reward_cube_all.png"
  fig.savefig(path, dpi=320)
  plt.close(fig)
  return path


def plot_reward_voxels_by_algo(algo, results, reward_scale):
  solref_values = sorted({row["solref0"] for row in results})
  reward_norm = Normalize(vmin=reward_scale[0], vmax=reward_scale[1])
  cmap = plt.get_cmap("coolwarm")

  fig = plt.figure(figsize=(14.6, 7.8))
  fig.subplots_adjust(left=0.02, right=0.87, bottom=0.08, top=0.86,
                      wspace=0.0)
  for index, sr0 in enumerate(solref_values, 1):
    ax = fig.add_subplot(1, len(solref_values), index, projection="3d")
    draw_reward_voxels(
        ax,
        results,
        sr0,
        reward_norm,
        cmap,
        alpha=0.28,
        tick_size=8.2,
        label_size=10.2,
        zoom=0.78,
    )
    ax.set_title(f"solref[0]={sr0:.3f}", fontsize=15, pad=12)

  fig.suptitle(
      f"{algo.upper()} PushBox reward voxels in solimp parameter space",
      fontsize=17,
  )
  fig.text(
      0.5,
      0.025,
      "x=solimp[0], y=solimp[1], z=solimp[2]; each translucent cube is one "
      "sweep result, colored by reward.",
      ha="center",
      fontsize=11,
      color="#333333",
  )
  add_reward_colorbar(fig, reward_norm, cmap, [0.905, 0.21, 0.020, 0.54])

  path = FIGURE_DIR / f"{algo}_pushbox_solimp_parameter_reward_voxels.png"
  fig.savefig(path, dpi=320)
  plt.close(fig)
  return path


def plot_reward_voxels_single_panels(algo, results, reward_scale):
  solref_values = sorted({row["solref0"] for row in results})
  reward_norm = Normalize(vmin=reward_scale[0], vmax=reward_scale[1])
  cmap = plt.get_cmap("coolwarm")
  paths = []

  for sr0 in solref_values:
    fig = plt.figure(figsize=(8.6, 7.5))
    fig.subplots_adjust(left=0.02, right=0.84, bottom=0.08, top=0.86)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    draw_reward_voxels(ax, results, sr0, reward_norm, cmap, alpha=0.28)
    ax.set_title(
        f"{algo.upper()} PushBox, solref[0]={sr0:.3f}",
        fontsize=15,
        pad=12,
    )
    fig.text(
        0.45,
        0.028,
        "x=solimp[0], y=solimp[1], z=solimp[2]; color=reward.",
        ha="center",
        fontsize=10,
        color="#333333",
    )
    add_reward_colorbar(fig, reward_norm, cmap, [0.88, 0.22, 0.030, 0.52])

    path = FIGURE_DIR / (
        f"{algo}_pushbox_solimp_parameter_reward_voxels_"
        f"solref_{param_name(sr0)}.png"
    )
    fig.savefig(path, dpi=320)
    plt.close(fig)
    paths.append(path)

  return paths


def plot_reward_voxels_combined(datasets, reward_scale):
  reward_norm = Normalize(vmin=reward_scale[0], vmax=reward_scale[1])
  cmap = plt.get_cmap("coolwarm")

  fig = plt.figure(figsize=(16.2, 12.0))
  fig.subplots_adjust(left=0.02, right=0.87, bottom=0.06, top=0.90,
                      wspace=0.0, hspace=0.08)
  for row_idx, (algo, results) in enumerate(datasets.items()):
    solref_values = sorted({row["solref0"] for row in results})
    for col_idx, sr0 in enumerate(solref_values):
      index = row_idx * len(solref_values) + col_idx + 1
      ax = fig.add_subplot(len(datasets), len(solref_values), index,
                           projection="3d")
      draw_reward_voxels(
          ax,
          results,
          sr0,
          reward_norm,
          cmap,
          alpha=0.28,
          tick_size=7.2,
          label_size=8.8,
          zoom=0.70,
      )
      ax.set_title(
          f"{algo.upper()}, solref[0]={sr0:.3f}",
          fontsize=14,
          pad=10,
      )

  fig.suptitle(
      "PushBox reward voxels in solimp parameter space",
      fontsize=18,
  )
  fig.text(
      0.5,
      0.025,
      "Shared color scale across all panels: red is higher reward and blue is "
      "lower reward.",
      ha="center",
      fontsize=11,
      color="#333333",
  )
  add_reward_colorbar(fig, reward_norm, cmap, [0.905, 0.22, 0.020, 0.55])

  path = FIGURE_DIR / "pushbox_solimp_parameter_reward_voxels_all.png"
  fig.savefig(path, dpi=320)
  plt.close(fig)
  return path


def plot_reward_parameter_volume_by_algo(algo, results, reward_scale):
  solref_values = sorted({row["solref0"] for row in results})
  reward_norm = Normalize(vmin=reward_scale[0], vmax=reward_scale[1])
  cmap = plt.get_cmap("coolwarm")

  fig = plt.figure(figsize=(14.6, 7.8))
  fig.subplots_adjust(left=0.02, right=0.87, bottom=0.08, top=0.86,
                      wspace=0.0)
  axes = []
  for index, sr0 in enumerate(solref_values, 1):
    ax = fig.add_subplot(1, len(solref_values), index, projection="3d")
    draw_reward_parameter_volume(
        ax,
        results,
        sr0,
        reward_norm,
        cmap,
        tick_size=8.0,
        label_size=10.0,
        zoom=0.72,
    )
    ax.set_title(f"solref[0]={sr0:.3f}", fontsize=15, pad=12)
    axes.append(ax)

  fig.suptitle(
      f"{algo.upper()} PushBox reward heatmap in solimp parameter space",
      fontsize=17,
  )
  fig.text(
      0.5,
      0.025,
      "x=solimp[0], y=solimp[1], z=solimp[2]; each horizontal slice fixes "
      "solimp[2] and interpolates reward over solimp[0]-solimp[1].",
      ha="center",
      fontsize=11,
      color="#333333",
  )
  add_reward_colorbar(fig, reward_norm, cmap, [0.905, 0.21, 0.020, 0.54])

  path = FIGURE_DIR / f"{algo}_pushbox_solimp_parameter_reward_heatmap.png"
  fig.savefig(path, dpi=320)
  plt.close(fig)
  return path


def plot_reward_parameter_volume_single_panels(algo, results, reward_scale):
  solref_values = sorted({row["solref0"] for row in results})
  reward_norm = Normalize(vmin=reward_scale[0], vmax=reward_scale[1])
  cmap = plt.get_cmap("coolwarm")
  paths = []

  for sr0 in solref_values:
    fig = plt.figure(figsize=(8.8, 7.5))
    fig.subplots_adjust(left=0.02, right=0.84, bottom=0.08, top=0.86)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    draw_reward_parameter_volume(ax, results, sr0, reward_norm, cmap)
    ax.set_title(
        f"{algo.upper()} PushBox, solref[0]={sr0:.3f}",
        fontsize=15,
        pad=12,
    )
    fig.text(
        0.45,
        0.028,
        "x=solimp[0], y=solimp[1], z=solimp[2]; color=reward.",
        ha="center",
        fontsize=10,
        color="#333333",
    )
    add_reward_colorbar(fig, reward_norm, cmap, [0.88, 0.22, 0.030, 0.52])

    path = FIGURE_DIR / (
        f"{algo}_pushbox_solimp_parameter_reward_heatmap_"
        f"solref_{param_name(sr0)}.png"
    )
    fig.savefig(path, dpi=320)
    plt.close(fig)
    paths.append(path)

  return paths


def plot_reward_parameter_volume_combined(datasets, reward_scale):
  reward_norm = Normalize(vmin=reward_scale[0], vmax=reward_scale[1])
  cmap = plt.get_cmap("coolwarm")

  fig = plt.figure(figsize=(16.2, 12.0))
  fig.subplots_adjust(left=0.02, right=0.87, bottom=0.06, top=0.90,
                      wspace=0.0, hspace=0.08)
  axes = []
  for row_idx, (algo, results) in enumerate(datasets.items()):
    solref_values = sorted({row["solref0"] for row in results})
    for col_idx, sr0 in enumerate(solref_values):
      index = row_idx * len(solref_values) + col_idx + 1
      ax = fig.add_subplot(len(datasets), len(solref_values), index,
                           projection="3d")
      draw_reward_parameter_volume(
          ax,
          results,
          sr0,
          reward_norm,
          cmap,
          tick_size=7.2,
          label_size=8.8,
          zoom=0.64,
      )
      ax.set_title(
          f"{algo.upper()}, solref[0]={sr0:.3f}",
          fontsize=14,
          pad=10,
      )
      axes.append(ax)

  fig.suptitle(
      "PushBox reward heatmap in solimp parameter space",
      fontsize=18,
  )
  fig.text(
      0.5,
      0.025,
      "Shared color scale across all panels: red is higher reward and blue is "
      "lower reward.",
      ha="center",
      fontsize=11,
      color="#333333",
  )
  add_reward_colorbar(fig, reward_norm, cmap, [0.905, 0.22, 0.020, 0.55])

  path = FIGURE_DIR / "pushbox_solimp_parameter_reward_heatmap_all.png"
  fig.savefig(path, dpi=320)
  plt.close(fig)
  return path


def plot_surface_grid(algo, results, penetration, shared_pen_scale=None):
  solref_values = sorted({row["solref0"] for row in results})
  solimp1_values = sorted({row["solimp1"] for row in results})
  solimp0_values = sorted({row["solimp0"] for row in results})
  solimp2_values = sorted({row["solimp2"] for row in results})

  x_log_values = np.log10(np.array(solimp0_values, dtype=float))
  y_log_values = np.log10(np.array(solimp2_values, dtype=float))
  grid_x_values = np.linspace(min(x_log_values), max(x_log_values), 110)
  grid_y_values = np.linspace(min(y_log_values), max(y_log_values), 110)
  grid_x, grid_y = np.meshgrid(grid_x_values, grid_y_values)

  reward_values = np.array([row["reward"] for row in results], dtype=float)
  reward_min = float(reward_values.min())
  reward_max = float(reward_values.max())
  reward_pad = 0.04 * max(reward_max - reward_min, 1e-9)

  if shared_pen_scale is None:
    pen_values = np.array([penetration[_key(row)] for row in results], dtype=float)
    pen_min = float(pen_values.min())
    pen_max = float(pen_values.max())
  else:
    pen_min, pen_max = shared_pen_scale

  slice_colors, slice_means = build_slice_colors(results, penetration)

  fig = plt.figure(figsize=(18.0, 8.8))
  fig.subplots_adjust(left=0.035, right=0.99, bottom=0.075, top=0.88,
                      wspace=0.02, hspace=0.05)

  for row_idx, sr0 in enumerate(solref_values):
    for col_idx, s1 in enumerate(solimp1_values):
      index = row_idx * len(solimp1_values) + col_idx + 1
      ax = fig.add_subplot(
          len(solref_values), len(solimp1_values), index, projection="3d"
      )
      group = [
          row
          for row in results
          if abs(row["solref0"] - sr0) < 1e-12
          and abs(row["solimp1"] - s1) < 1e-12
      ]
      points = np.array([
          [np.log10(row["solimp0"]), np.log10(row["solimp2"])]
          for row in group
      ])
      rewards = np.array([row["reward"] for row in group], dtype=float)
      pens = np.array([penetration[_key(row)] for row in group], dtype=float)

      reward_grid = smooth_surface(
          points, rewards, grid_x, grid_y, reward_min, reward_max
      )
      pen_grid = smooth_surface(points, pens, grid_x, grid_y, pen_min, pen_max)
      facecolors = slice_facecolors(
          slice_colors[(sr0, s1)], pen_grid, pen_min, pen_max, reward_grid
      )

      ax.plot_surface(
          grid_x,
          grid_y,
          reward_grid,
          facecolors=facecolors,
          rstride=1,
          cstride=1,
          linewidth=0.04,
          edgecolor=(0.16, 0.16, 0.16, 0.13),
          antialiased=True,
          shade=False,
      )
      ax.contour(
          grid_x,
          grid_y,
          reward_grid,
          zdir="z",
          offset=reward_min - reward_pad,
          levels=7,
          colors="#2f2f2f",
          linewidths=0.35,
          alpha=0.48,
      )

      ax.set_xlim(min(x_log_values), max(x_log_values))
      ax.set_ylim(min(y_log_values), max(y_log_values))
      ax.set_zlim(reward_min - reward_pad, reward_max + reward_pad)
      ax.set_xticks(x_log_values)
      ax.set_yticks(y_log_values)
      ax.set_xticklabels([param_label(value) for value in solimp0_values])
      ax.set_yticklabels([param_label(value) for value in solimp2_values])
      if row_idx == len(solref_values) - 1:
        ax.set_xlabel("solimp[0]", labelpad=4)
      else:
        ax.set_xticklabels([])
      if col_idx == 0:
        ax.set_ylabel("solimp[2]", labelpad=4)
      else:
        ax.set_yticklabels([])
      ax.set_zlabel("reward", labelpad=3)
      mean_pen = slice_means[(sr0, s1)] * 1000.0
      ax.set_title(
          f"solimp[1]={s1:.3f}, solref[0]={sr0:.3f}\n"
          f"mean pen={mean_pen:.1f} mm",
          fontsize=8.5,
          pad=2,
      )
      style_3d_axis(ax, tick_size=7.0, label_size=8.0)

  fig.suptitle(
      f"{algo.upper()} PushBox reward surfaces "
      "(warm=larger mean penetration, cool=smaller; dark=hard, light=soft)",
      fontsize=13,
  )
  fig.text(
      0.5,
      0.02,
      "x/y axes use log-spaced coordinates with original parameter tick labels; "
      "penetration is calibrated by ball-drop for each train parameter set.",
      ha="center",
      fontsize=9,
      color="#333333",
  )

  path = FIGURE_DIR / f"{algo}_pushbox_solimp0_solimp2_surface_2x4.png"
  fig.savefig(path, dpi=300)
  plt.close(fig)
  return path


def plot_surface_by_solref(algo, results, penetration, shared_pen_scale=None):
  """One larger 2x2 figure per solref[0] value for readable axes."""
  solref_values = sorted({row["solref0"] for row in results})
  solimp1_values = sorted({row["solimp1"] for row in results})
  solimp0_values = sorted({row["solimp0"] for row in results})
  solimp2_values = sorted({row["solimp2"] for row in results})

  x_log_values = np.log10(np.array(solimp0_values, dtype=float))
  y_log_values = np.log10(np.array(solimp2_values, dtype=float))
  grid_x_values = np.linspace(min(x_log_values), max(x_log_values), 120)
  grid_y_values = np.linspace(min(y_log_values), max(y_log_values), 120)
  grid_x, grid_y = np.meshgrid(grid_x_values, grid_y_values)

  reward_values = np.array([row["reward"] for row in results], dtype=float)
  reward_min = float(reward_values.min())
  reward_max = float(reward_values.max())
  reward_pad = 0.04 * max(reward_max - reward_min, 1e-9)

  if shared_pen_scale is None:
    pen_values = np.array([penetration[_key(row)] for row in results], dtype=float)
    pen_min = float(pen_values.min())
    pen_max = float(pen_values.max())
  else:
    pen_min, pen_max = shared_pen_scale

  slice_colors, slice_means = build_slice_colors(results, penetration)
  paths = []

  for sr0 in solref_values:
    fig = plt.figure(figsize=(19.0, 13.4))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.065, top=0.88,
                        wspace=0.0, hspace=0.015)

    for col_idx, s1 in enumerate(solimp1_values):
      ax = fig.add_subplot(2, 2, col_idx + 1, projection="3d")
      group = [
          row
          for row in results
          if abs(row["solref0"] - sr0) < 1e-12
          and abs(row["solimp1"] - s1) < 1e-12
      ]
      points = np.array([
          [np.log10(row["solimp0"]), np.log10(row["solimp2"])]
          for row in group
      ])
      rewards = np.array([row["reward"] for row in group], dtype=float)
      pens = np.array([penetration[_key(row)] for row in group], dtype=float)

      reward_grid = smooth_surface(
          points, rewards, grid_x, grid_y, reward_min, reward_max
      )
      pen_grid = smooth_surface(points, pens, grid_x, grid_y, pen_min, pen_max)
      facecolors = slice_facecolors(
          slice_colors[(sr0, s1)], pen_grid, pen_min, pen_max, reward_grid
      )

      ax.plot_surface(
          grid_x,
          grid_y,
          reward_grid,
          facecolors=facecolors,
          rstride=1,
          cstride=1,
          linewidth=0.04,
          edgecolor=(0.16, 0.16, 0.16, 0.11),
          antialiased=True,
          shade=False,
      )
      ax.contour(
          grid_x,
          grid_y,
          reward_grid,
          zdir="z",
          offset=reward_min - reward_pad,
          levels=7,
          colors="#303030",
          linewidths=0.36,
          alpha=0.42,
      )

      ax.set_xlim(min(x_log_values), max(x_log_values))
      ax.set_ylim(min(y_log_values), max(y_log_values))
      ax.set_zlim(reward_min - reward_pad, reward_max + reward_pad)
      ax.set_xticks(x_log_values)
      ax.set_yticks(y_log_values)
      ax.set_xticklabels([param_label(value) for value in solimp0_values])
      ax.set_yticklabels([param_label(value) for value in solimp2_values])
      ax.set_xlabel("solimp[0]", labelpad=18)
      ax.set_ylabel("solimp[2]", labelpad=18)
      ax.set_zlabel("reward", labelpad=10)
      mean_pen = slice_means[(sr0, s1)] * 1000.0
      ax.set_title(
          f"solimp[1]={s1:.3f}\nmean pen={mean_pen:.1f} mm",
          fontsize=12.5,
          pad=8,
      )
      style_3d_axis(ax, tick_size=11.5, label_size=13.0)

    fig.suptitle(
        f"{algo.upper()} PushBox reward surfaces, solref[0]={sr0:.3f}",
        fontsize=17,
    )
    fig.text(
        0.5,
        0.018,
        "Each panel fixes solimp[1]; x=solimp[0], y=solimp[2], z=reward. "
        "Darker surface color means smaller ball-drop penetration, i.e. harder.",
        ha="center",
        fontsize=12,
        color="#333333",
    )

    sr_name = f"{sr0:g}".replace(".", "p")
    path = FIGURE_DIR / (
        f"{algo}_pushbox_solimp0_solimp2_surface_solref_{sr_name}_2x2.png"
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

  return paths


def parse_args():
  parser = argparse.ArgumentParser(
      description="Plot PushBox reward over solimp sweeps."
  )
  parser.add_argument(
      "--algo",
      choices=("apg", "ppo", "both"),
      default="both",
  )
  parser.add_argument(
      "--mode",
      choices=("cube", "voxels", "volume", "surface", "all"),
      default="cube",
      help=(
          "cube plots one continuous translucent interpolated cube; voxels "
          "plots each sweep result as a translucent cube; volume plots stacked "
          "heatmap slices; surface keeps the older z=reward plots."
      ),
  )
  parser.add_argument(
      "--cube-cells",
      type=int,
      default=16,
      help="Number of interpolated cells per axis for --mode cube.",
  )
  parser.add_argument(
      "--force-calibration",
      action="store_true",
      help="Recompute ball-drop penetration even if a cache exists.",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  FIGURE_DIR.mkdir(parents=True, exist_ok=True)
  plt.rcParams.update({
      "font.size": 9,
      "axes.titlesize": 9,
      "axes.labelsize": 8,
      "figure.dpi": 150,
      "savefig.dpi": 300,
      "savefig.bbox": "tight",
  })

  algos = ALGORITHMS if args.algo == "both" else (args.algo,)
  datasets = {algo: load_results(algo) for algo in algos}
  reward_scale = reward_scale_from_datasets(datasets)
  print(
      "Reward color scale: "
      f"{reward_scale[0]:.5g}-{reward_scale[1]:.5g} "
      "(blue=low, red=high)"
  )

  paths = []
  if args.mode in ("cube", "all"):
    if len(datasets) > 1:
      paths.append(
          plot_reward_cube_combined(
              datasets,
              reward_scale,
              args.cube_cells,
          )
      )
    for algo, results in datasets.items():
      paths.append(
          plot_reward_cube_by_algo(
              algo,
              results,
              reward_scale,
              args.cube_cells,
          )
      )
      paths.extend(
          plot_reward_cube_single_panels(
              algo,
              results,
              reward_scale,
              args.cube_cells,
          )
      )

  if args.mode in ("voxels", "all"):
    if len(datasets) > 1:
      paths.append(plot_reward_voxels_combined(datasets, reward_scale))
    for algo, results in datasets.items():
      paths.append(plot_reward_voxels_by_algo(algo, results, reward_scale))
      paths.extend(
          plot_reward_voxels_single_panels(algo, results, reward_scale)
      )

  if args.mode in ("volume", "all"):
    if len(datasets) > 1:
      paths.append(plot_reward_parameter_volume_combined(datasets, reward_scale))
    for algo, results in datasets.items():
      paths.append(
          plot_reward_parameter_volume_by_algo(algo, results, reward_scale)
      )
      paths.extend(
          plot_reward_parameter_volume_single_panels(
              algo, results, reward_scale
          )
      )

  if args.mode in ("surface", "all"):
    penetration = calibrate_penetration(datasets, force=args.force_calibration)
    pen_values = np.array(list(penetration.values()), dtype=float)
    shared_pen_scale = (float(pen_values.min()), float(pen_values.max()))
    print(
        "Calibrated penetration range: "
        f"{shared_pen_scale[0] * 1000.0:.3f}-"
        f"{shared_pen_scale[1] * 1000.0:.3f} mm"
    )
    for algo, results in datasets.items():
      paths.append(
          plot_surface_grid(algo, results, penetration, shared_pen_scale)
      )
      paths.extend(
          plot_surface_by_solref(algo, results, penetration, shared_pen_scale)
      )

  print("Saved:")
  for path in paths:
    print(f"  {path}")


if __name__ == "__main__":
  main()
